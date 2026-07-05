# 第 8 章 异构算力：CPU、GPU 与 NPU

> 本章讲三件事：一部手机上 CPU、GPU、NPU 三类计算单元各自的特性与约束；LiteRT-LM 如何用一个工厂把它们封装在同一套接口之后；以及一个常被误解的现象——换一个后端，连模型的输出文字都会变。

第 1 章把端侧的硬约束分成内存容量、内存带宽、功耗与异构三类。前两类已经在第 6、7 章处理，这一章处理功耗与异构这一类里的异构部分。手机上能做计算的不止一处：CPU、GPU、NPU 各有各的强项和限制。一个端侧运行时既要用得上它们，又要对上层封装掉它们的差异。先从一个反直觉的现象说起。

## 同一个模型，三种后端

第 2 章"二十个问题"的第 17 问，来自一个真实的上游困惑（`LiteRT-LM#2281`）：同一个 `.litertlm` 文件，`--backend=cpu` 和 `--backend=gpu` 跑出来，不只是速度不同，连**输出的文字都可能不一样**。这与直觉相悖：不是同一个模型、同一个提示词吗？

先接受这个事实，本章最后一节解释它的成因。现在只需记住一点：后端不是"同一个计算的不同速度"，它们是**不同的计算路径**。算子实现不同，数值精度处理不同，反量化方式也可能有细微差别。理解了这一点，"可插拔后端"这个设计的分量才显出来。它并非可选的增强特性，而是端侧必须面对的现实：一部手机上多种计算单元并存，运行时必须在它们之间做选择，又要让上层代码对这个选择无感。

## 一个工厂，按 Backend 分派

LiteRT-LM 用一个枚举把后端列全（`runtime/executor/executor_settings_base.h:34`）：

```cpp
enum class Backend {
  UNSPECIFIED,
  CPU_ARTISAN,             // (1)
  GPU_ARTISAN,
  CPU,                     // (2)
  GPU,
  GOOGLE_TENSOR_ARTISAN,
  NPU,                     // (3)
};
```

`(1)` 这一组带 `ARTISAN` 后缀的是手写算子路径；`(2)` 不带后缀的 `CPU`、`GPU` 走 LiteRT 编译路径。同一类硬件被枚举成两个值，因为它们背后是两套算子实现——本章聚焦 `CPU`/`GPU`/`NPU` 这条编译路径。`(3)` `NPU` 单独一档，下文会看到它在工厂里也走一条独立分支。枚举里还有 `GOOGLE_TENSOR_ARTISAN`，对应 Pixel 的 Tensor 芯片，此处不展开。

选哪个实现，交给一个工厂函数（`runtime/executor/llm_litert_compiled_model_executor_factory.cc:165`）：

```cpp
Backend backend = executor_settings.GetBackend();   // (1)
switch (backend) {
  case Backend::CPU:
  case Backend::GPU:                                 // (2)
    return CreateCpuOrGpuLlmLiteRtCompiledModelExecutor(executor_settings,
                                                        lrt_env, resources);
  case Backend::NPU:                                 // (3)
    return CreateNpuLlmLiteRtCompiledModelExecutor(executor_settings, lrt_env,
                                                   resources);
  default:
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported backend: ", backend));
}
```

`(1)` 后端来自 `LlmExecutorSettings`，而它最终由 CLI 的 `--backend` 或上层配置写入。分派的输入是一个纯数据字段，不是运行时探测。`(2)` `CPU` 和 `GPU` 共用 `case`、落到同一个创建函数。这并非实现上的妥协：这条路径读出的是同一个 `.tflite` 子图（`ModelType::kTfLitePrefillDecode`，`:135`），CPU 与 GPU 的差异被推迟到 LiteRT 编译期由 delegate 决定，工厂层不必区分。`(3)` `NPU` 走独立分支，因为它加载的是另一组模型文件（下一节 NPU 的 embedder 子模型就是证据）。`default` 分支把 `CPU_ARTISAN` 这类未接入编译路径的后端挡在门外，返回 `InvalidArgumentError` 而非崩溃。这是工厂作为唯一入口的价值：非法后端在这里一次性拦下。

<div class="aside-compare">

后端管理的另两种形态可作参照。llama.cpp 维护一个后端注册表，按编译开关依次注册 CUDA、Metal、SYCL、Vulkan、WebGPU 等，CPU 永远殿后兜底（`llama.cpp/ggml/src/ggml-backend-reg.cpp:117`–`:165 @ b9873`），一个二进制可以带多个后端、运行时探测设备分层调度。MLC-LLM 则走提前编译：用 TVM 把模型按具体目标（某款 GPU、某个架构）编译成专用产物（官方文档，mlc.ai，访问于 2026-07）。三家分布在一条谱系上：MLC 在编译期锁定目标换最深的按机特化，llama.cpp 在运行期探测换一包通吃，LiteRT-LM 居中——执行器按 Backend 工厂分派、模型按后端各自编译，部署时选定组合。谱系上没有对错，只有「产物数量 × 运行时自由度」的兑换率。

</div>

### 静态形状与动态形状：第二次分派

CPU/GPU 那条路径内部还有一次分派，按模型导出时是静态形状还是动态形状再分（`llm_litert_compiled_model_executor_factory.cc:137`）：

```cpp
ASSIGN_OR_RETURN(bool is_dynamic_model, IsDynamicModel(*litert_model));
if (is_dynamic_model) {
  ASSIGN_OR_RETURN(executor, LlmLiteRtCompiledModelExecutorDynamic::Create(
                                 executor_settings, lrt_env, resources));  // (1)
} else {
  ASSIGN_OR_RETURN(executor, LlmLiteRtCompiledModelExecutorStatic::Create(
                                 executor_settings, lrt_env, resources));  // (2)
}
```

这个 `is_dynamic_model` 不是配置项，而是从模型文件里读出来的事实。`IsDynamicModel` 检查 prefill 子图里两组张量的形状是不是运行时可变（`llm_litert_compiled_model_executor_factory.cc:98`）：

```cpp
ASSIGN_OR_RETURN(bool is_k_dynamic, IsDynamicTensor(k_tensor));
// ...
ASSIGN_OR_RETURN(bool is_v_dynamic, IsDynamicTensor(v_tensor));
RET_CHECK(is_k_dynamic == is_v_dynamic)
    << "KV cache k and v need to be dynamic or static at the same time.";  // (1)
is_kv_cache_dynamic = is_k_dynamic && is_v_dynamic;
// ...
ASSIGN_OR_RETURN(is_seq_len_dynamic, IsDynamicTensor(position_tensor));
RET_CHECK(is_kv_cache_dynamic == is_seq_len_dynamic)
    << "KV cache and seq len need to be dynamic or static at the same time.";  // (2)
return is_kv_cache_dynamic;
```

`(1)` 判据取自 KV cache 的 K、V 两个张量的形状是否动态，并用 `RET_CHECK` 强制两者一致：一个模型不允许 K 动态而 V 静态。`(2)` 更强的一条约束在最后：KV cache 的动静态必须与序列长度（position 张量）的动静态一致，否则直接报错退出。换句话说，一个导出的模型要么整体是动态形状，要么整体是静态形状，不存在混合态。这条 `RET_CHECK` 把"半动态"的非法组合挡在加载阶段，而不是留到运行时崩溃。

两条路径的性能含义不同。静态形状走 `Static` 执行器：prefill 与 decode 的张量形状在编译期就定死，KV cache 按最大上下文长度一次性分配，运行时形状不再变，delegate 只需编译一次。动态形状走 `Dynamic` 执行器：KV cache 随 decode 步逐步增长，每次增长 `kv_increment_size` 个位置（默认 16，`llm_executor_settings.h:110`）。动态形状的好处是短对话不必按最大长度预分配缓冲区，峰值内存更省；代价是形状变化可能触发底层重新准备。正文后面会看到，`prefill_chunk_size` 这个参数的注释明确写着"only applicable to dynamically exported models"（`llm_executor_settings.h:113`）——分块 prefill 只对动态模型有意义，因为静态模型的 prefill 形状已经固定。这层静态/动态的选择对上层同样不可见。

无论走哪条分支，工厂返回的都是 `absl::StatusOr<std::unique_ptr<LlmExecutor>>`。上层拿到的是同一个 `LlmExecutor` 抽象（第 5 章那个接口），不知道底下是 CPU 还是 NPU、是 Static 还是 Dynamic。这就是"可插拔后端"落到代码里的样子：一个枚举、一个工厂、一个共同接口。加一个新后端，就是加一个 `case`，上层一行不改。

<figure>
{{#include figs/fig-8-1.svg}}
<figcaption>图 8-1　后端工厂分派：Backend 枚举经工厂函数分成 CPU/GPU 一路、NPU 一路，都产出同一个 LlmExecutor 抽象。上层对后端差异无感知。</figcaption>
</figure>

## CPU：随时可用，但需管理线程与亲和性

CPU 的好处是随时都在、什么算子都能跑。它的挑战落在功耗与调度上，需要两件配置到位：线程绑核与线程数量。

### CPU 亲和性：只认 Pixel Tensor 的一张硬编码表

手机 CPU 是大小核混合的：几个高性能核，几个高能效核。推理这种计算密集的任务跑在性能核上才快；但若不加干预，系统调度器可能把线程迁到能效核上，吞吐随之下降。LiteRT-LM 为此提供了 CPU 亲和性工具（`runtime/engine/cpu_affinity_utils.h`），对外三个函数：判定是不是 Pixel Tensor 设备（`:25`）、查出性能核编号（`:29`）、把当前线程绑上去（`:35`）。性能核编号不是运行时算出来的，是按芯片型号硬编码的一张表（`cpu_affinity_utils.cc:57`）：

```cpp
const TensorCoreAffinity kTensorAffinities[] = {
    {PixelSoc::kTensorG3, {4, 5, 6, 7, 8}},   // (1)
    {PixelSoc::kTensorG4, {4, 5, 6, 7}},
    {PixelSoc::kTensorG5, {2, 3, 4, 5, 6, 7}},
    {PixelSoc::kTensorG6, {2, 3, 4, 5, 6, 7}},
};
```

`(1)` 每一行是一款 Pixel SoC 的中大核编号。核编号随代际变化：G3 用 4 到 8 号核，G4 收窄到 4 到 7 号，G5、G6 则从 2 号起。这套编号对应各代 Tensor SoC 的物理核布局，硬编码而非探测，因为 Android 没有一个可移植的接口能报告"哪些核是大核"。

识别芯片型号的逻辑值得展开，正文早先只提了半句。判定走 `GetCurrentPixelSoc`（`cpu_affinity_utils.cc:66`），它读两个系统属性并做双重校验：

```cpp
static const PixelSoc soc = []() {
  char manufacturer[PROP_VALUE_MAX] = {0};
  char soc_model[PROP_VALUE_MAX] = {0};
  __system_property_get("ro.soc.manufacturer", manufacturer);  // (1)
  __system_property_get("ro.soc.model", soc_model);
  if (absl::string_view(manufacturer) != "Google") {
    return PixelSoc::kUnknown;                                 // (2)
  }
  absl::string_view soc_str(soc_model);
  if (soc_str == "Tensor G3") return PixelSoc::kTensorG3;      // (3)
  // ...
  return PixelSoc::kUnknown;
}();
return soc;
```

`(1)` 读的是两个属性：制造商 `ro.soc.manufacturer` 与型号 `ro.soc.model`。`(2)` 先卡制造商——不是 Google 直接判为未知，这道校验能挡掉那些恰好把型号字符串取名 "Tensor" 的第三方设备。`(3)` 再逐一比对型号字符串。整段包在一个 `static` 局部变量的 lambda 初始化里，C++ 保证它只求值一次，此后每次调用直接返回缓存结果，避免反复读系统属性。认不出就返回 `kUnknown`，`IsPixelTensorDevice()` 随之返回 false。

这张表和这道校验一起划定了工具的适用边界：只认 Google 自家的 Tensor 芯片。高通、联发科的设备走不到这里，`GetPixelPerformanceCores()` 对未知 SoC 返回空表。空表进 `SetCpuAffinity` 会被第一行直接短路（`cpu_affinity_utils.cc:104`）：`cpu_affinity_cores.empty()` 为真时记一条 warning 就返回 `OkStatus`，绑核这一步在非 Pixel 设备上等于跳过。更彻底的是非 Android 编译分支——整个 `cpu_affinity_utils.cc` 的实现被 `#if defined(__ANDROID__)` 包住，在非 Android 平台上 `IsPixelTensorDevice` 恒为 false、`SetCpuAffinity` 是空实现直接返回成功（`cpu_affinity_utils.cc:130`）。作者手上那台 Mac 走的正是这条空实现分支。

绑核的动作落在 `SetCpuAffinity`（`cpu_affinity_utils.cc:103`），本质是一次 Linux 系统调用：

```cpp
cpu_set_t mask;
CPU_ZERO(&mask);
for (int cpu : cpu_affinity_cores) {
  CPU_SET(cpu, &mask);                              // (1)
}
if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {  // (2)
  return absl::InternalError(
      absl::StrCat("Failed to set CPU affinity: ", strerror(errno)));
}
```

`(1)` 把每个性能核编号写进一个位掩码。`(2)` `sched_setaffinity` 的第一个参数是 0，代表当前线程。这是一个建议而非命令，头文件注释说得明白："The scheduler will then attempt to run the thread on these cores most of the time"（`cpu_affinity_utils.h:31`）。调度器多数时候会照办，但没有硬保证。绑核失败不致命：只记一条 warning，推理照跑，只是可能落到能效核上慢一点。

这段代码在推理流水线里的位置很靠前。它不在 decode 循环里，而在引擎创建时执行一次（`engine_factory.h:136`）：`if (IsPixelTensorDevice())` 为真才查核、绑核，且绑的是引擎创建线程本身。头文件注释说明这次绑核会波及该线程之后创建的子线程（"the current thread and any child threads it creates"，`cpu_affinity_utils.h:31`），推理线程池由此继承同一份亲和性。绑一次，此后整个会话的推理线程都倾向留在性能核上。

### 线程数：为什么默认是 4

线程本身由一个线程池管（`runtime/framework/threadpool.h:51`），构造时给一个上限 `max_num_threads`（`:57`）。这里说的是算子内并行度——一个矩阵乘法拆给几个线程一起算。用几个线程是个权衡：线程多了未必更快，因为 decode 阶段的瓶颈是内存带宽而非算力（第 1 章的带宽约束），线程再多也快不过内存往核里喂权重的速度；而线程一多，功耗和发热却是实打实地涨。

这个数默认是 4（`CpuConfig::number_of_threads`，`runtime/executor/llm_executor_settings.h:120`），注释直接写"The default value is 4"。CLI 用 `--num_cpu_threads` 覆盖（对应上游需求 `LiteRT-LM#2505`）。覆盖路径落在 CPU 后端专属的配置分支里（`litert_lm_lib.cc:523`）：

```cpp
if (backend == Backend::CPU) {
  auto& executor_settings = engine_settings.GetMutableMainExecutorSettings();
  ASSIGN_OR_RETURN(
      auto cpu_settings,
      executor_settings.MutableBackendConfig<litert::lm::CpuConfig>());
  if (settings.num_cpu_threads > 0) {                    // (1)
    cpu_settings.number_of_threads = settings.num_cpu_threads;
  }
  cpu_settings.prefill_chunk_size = settings.prefill_chunk_size;  // (2)
  executor_settings.SetBackendConfig(cpu_settings);
}
```

`(1)` 只有传了正值才覆盖默认的 4，传 0 或不传就保留默认。`(2)` 同一分支里顺带把 `prefill_chunk_size` 也写进去——前面说过它只对动态导出的模型生效，静态模型忽略这个值。整个分支用 `backend == Backend::CPU` 守卫，GPU、NPU 各有自己的配置分支，`CpuConfig` 只在 CPU 路径上被读写。

这两件事合起来才是 CPU 后端在功耗与带宽约束下的推荐配置：绑核让线程留在性能核上，线程数压在内存带宽上限附近而不是把核开满。要把"多线程未必更快"这句话从断言变成实测，可以扫描线程数：固定设备、模型、上下文，让 `--num_cpu_threads` 取 1、2、4、8，各跑三次取中位数，对比 prefill 与 decode 吞吐。可以预期 prefill 对线程数更敏感（它更接近算力受限，第 2 章），decode 增益会更早触顶（带宽先封顶）。本书基准数据集固定后端与线程默认值，未做这一维扫描（附录 D），此处把方法记录在案，结论待实测回填。

## GPU：并行强，还能在设备上采样

GPU 的强项是大规模并行，适合矩阵运算。它在端侧的一个优化第 5 章已经埋过引子：片上采样（on-device sampling），把采样这一步直接放在 GPU 上做，省掉 decode 每步一次 logits 回传 CPU。

回忆第 5 章的两条采样路径。设备上采样之所以快，是因为采样这一步不必把 logits 搬回 CPU。这次省掉的拷贝值多少，可以当面算一笔账。decode 每一步都会产出一整组 logits，长度等于词表大小。以 Gemma 系列约 26 万的词表、logits 按 FP16（2 字节）计，一步的 logits 就是 26 万 × 2 字节 ≈ 512 KiB。若采样在 CPU 做，这 512 KiB 每步都要从 GPU 显存拷回 CPU 内存（一次 device→host 传输）；在 GPU 上就地采样，这次拷贝省下。单看一步不大，但 decode 是整个生成里最频繁的操作，生成 512 个 token 就是 512 次这样的往返。第 1 章说过 decode 是内存带宽受限的：每一步真正的工作量是把几个 GiB 的权重读一遍，相比之下 512 KiB 的 logits 回传占比不高，所以这次省拷贝对整机吞吐的贡献是二阶的，很难从整机数字里单独剥离出来（本书未单测）。它更实在的价值在延迟链路上：省掉 device→host 同步，decode 每步少一次跨设备等待。

采样器怎么建、跑在哪，都在 `InitializeSampler` 里定（`runtime/executor/llm_litert_compiled_model_executor.h:160`，实现见 `.cc:1335`）：

```cpp
ASSIGN_OR_RETURN(auto sampler_backend, GetSamplerBackend(executor_settings_));  // (1)
// ...
gpu_sampler_max_top_k_ = sampler_params.k();          // (2)
ASSIGN_OR_RETURN(
    sampler_,
    CreateSampler(sampler_backend, output_heads, std::move(sampler_params),
                  env_.Get(), /*sequence_size=*/1, vocab_size, data_type));
// ...
const bool runs_embedding_on_gpu = (embedding_lookup_ == nullptr);  // (3)
sampler_handles_input_ =
    (!executor_settings_.GetAdvancedSettings().has_value() ||
     executor_settings_.GetAdvancedSettings()->sampler_handles_input) &&
    sampler_->CanHandleInput() && !signatures_.input_tokens.empty() &&
    !runs_embedding_on_gpu;
```

`(1)` 采样器有自己的 `sampler_backend`，与主执行器的后端相互独立配置：GPU 主后端可以配 GPU 采样器，让 logits 不离开显存。`(2)` `gpu_sampler_max_top_k_` 存下 top-k 的 `k`。GPU 上做 top-k 要预先知道候选个数才能开好缓冲区，这个字段就是给 GPU 采样路径准备的（`.h:354`）；预分配的是"最多取 k 个候选"这块显存，避免每步动态分配。`(3)` `sampler_handles_input_` 是关键开关。注意它的最后一个条件 `!runs_embedding_on_gpu`：如果 embedding 本来就在 GPU 上查（`embedding_lookup_ == nullptr`），这条优化反而关掉。两条 GPU 优化路径不叠加，代码在这里划了清楚的边界。

### 让 token 不出显存：输入张量接管的完整通路

开关为真只是前提，"省搬运"到底省在哪几步，要看采样器如何接管 decode 的输入张量。这条路径此前正文一句带过，这里摊开。

`sampler_handles_input_` 为真时，`InitializeSampler` 先给采样器备好两块输入缓冲——decode 的 position 张量和 attention mask 张量的"上一步"副本（`decode_prev_input_pos_`、`decode_prev_mask_`），然后做一次"设一次再复位"的预热（`llm_litert_compiled_model_executor.cc:1383`）：

```cpp
RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/false));
RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/true));  // (1)
```

`(1)` 先 `false` 再 `true` 这一对调用是有意的：注释写着"Set, then reset the input handling to get the underlying model ready, but not to bind the input tensors"。先设一遍让底层模型把输入形状准备好，再复位解绑，避免在初始化阶段就把张量绑死。真正的绑定发生在 decode 循环里。

绑定的核心是 `SetSamplerInputHandling`（`llm_litert_compiled_model_executor.cc:1404`），它把一组张量指针和一个回调函数一起交给采样器：

```cpp
return sampler_->SetInputTensorsAndInferenceFunc(
    &decode_input_buffers_[signatures_.input_tokens], &decode_prev_input_pos_,
    &decode_input_buffers_[signatures_.input_positions],
    has_input_attn_mask ? &decode_prev_mask_ : nullptr,
    has_input_attn_mask ? &decode_input_buffers_[*signatures_.input_attn_mask]
                        : nullptr,
    BindTensorsAndRunDecodeStatic, this);  // (1)
```

采样器拿到的第一个指针是 decode 的输入 token 张量（`input_tokens`）。含义是：采样器选出下一个 token 之后，直接把它写进这块输入张量——这块张量正是下一步 decode 的输入。token 从产生到被消费，全程留在设备缓冲里，不经过一次 device→host→device 的往返。`(1)` 最后传进去的 `BindTensorsAndRunDecodeStatic` 是一个静态回调，采样器在设备上定完 token 后回调它推进下一步 decode（`llm_litert_compiled_model_executor.cc:952`）：

```cpp
int LlmLiteRtCompiledModelExecutorBase::BindTensorsAndRunDecodeStatic(
    void* arg) {
  auto self = static_cast<LlmLiteRtCompiledModelExecutorBase*>(arg);
  auto status = self->BindTensorsAndRunDecode(/*output_logits=*/nullptr);  // (1)
  // ...
  return status.raw_code();
}
```

`(1)` 回调里绑好张量就直接跑下一步 decode。控制权在设备侧循环，CPU 不必在每步之间介入把 token 搬来搬去。这就形成"token 不出显存"的完整回路：采样在 GPU 上完成，选出的 token 写回设备输入张量，回调在设备上触发下一次 decode。

还有一处双缓冲的细节。position 和 mask 张量需要区分"这一步"和"上一步"，因为 decode 推进时当前步要读上一步的位置。`SwapSamplerInputTensors` 用 `std::swap` 把"当前"和"上一步"两块缓冲对调指针，再重新绑定（`llm_litert_compiled_model_executor.cc:1391`）：读旧写新、交换指针，不额外分配也不拷贝内容。这与第 6 章 KV cache 的双缓冲是同一套思路，都用指针交换回避对同一块缓冲的读写冲突。

这正是第 18 问的答案。两个后端的整体吞吐差距见附录 D：本书基准（Apple M5 Pro，Gemma 4 E4B，context 1024，decode 128 token）上 gpu 的 decode ≈ 50.6 tok/s、cpu ≈ 24.7 tok/s，gpu 约为 cpu 的 2 倍；prefill 在同条件下 gpu ≈ 999、cpu ≈ 259，gpu 约为 cpu 的 3.9 倍〔基准 D〕。片上采样只是这个整体差距里的一项，无法单独计价。

## NPU：能效最高，接口最封闭

NPU 是三者里能效比最高的——同样的活，它最省电、最不发热，这在端侧是硬通货。代价是它最封闭：接口是厂商的（比如高通的 QNN），能跑的算子有限，灵活性最低。

本节的内容基于对代码的阅读，没有真机验证（作者手上是一台 Mac，跑不了手机 NPU），所以只讲代码能佐证的部分。NPU 走的是工厂里那条独立路径（`CreateNpuLlmLiteRtCompiledModelExecutor`），产出一个专门的执行器（`runtime/executor/llm_litert_npu_compiled_model_executor.h:51`）。这个执行器的类注释写着 "Component intended to be used with an NPU variant of Gemma3"（`:50`）——它不是通用执行器，是为 NPU 版 Gemma3 专门做的。

最能说明结构差异的是它内部的一组子模型 struct。CPU/GPU 执行器持有一个 `CompiledModel`（主图），NPU 执行器却持有好几个，各管一段计算。其中之一是 embedder（`:266`）：

```cpp
struct EmbedderContext {
  ::litert::Model embedder_model;                    // (1)
  ::litert::CompiledModel embedder_compiled_model;
  InferenceContext inference_context;
  // ...
};

struct EmbedderPerLayerContext { /* ... */ };        // (2)

struct NpuAuxiliaryContext {                         // (3)
  ::litert::CompiledModel npu_auxiliary_compiled_model;
  // ...
};
```

`(1)` embedder 是**一个独立的编译模型**——它有自己的 `Model` 和 `CompiledModel`，而不是主图里的一层。"把 token 变成 embedding"在 NPU 上被切成单独一个模型跑，CPU/GPU 那边这步是融在主图里的。`(2)` 还有一个 per-layer 的 embedder，`(3)` 一个 auxiliary context（`:316`）。注释说它"contains several signatures for Mask, RoPE and KV cache update computation"（`:314`）。也就是说，CPU/GPU 上一个前向就算完的东西，NPU 上被拆成了 embedder、per-layer embedder、mask/RoPE/KV-cache 更新、主 LLM 图等好几个独立编译模型串起来。

这不是猜测，延迟统计字段（`:60`–`:82`）把这条流水线逐段列了出来：`prefill_embedder_inference_latency_us`、`prefill_mask_inference_latency_us`、`prefill_rope_inference_latency_us`、`prefill_llm_inference_latency_us`、`prefill_cache_update_inference_latency_us`——每一段都有独立计时。一个前向被切成这么多段独立计时，正说明它们是各自独立的推理调用。为什么这么切，涉及 NPU 算子约束下的工程取舍（NPU 能跑的算子有限，把不友好的部分单独拆出来是常见做法），没有真机不宜妄下结论，此处只把代码能确证的结构差异记录在案。

## 为什么换后端连输出都变

回到开头的困惑（第 17 问，`LiteRT-LM#2281`）。现在有了前面的铺垫，可以给一个合理的解释。

三个后端是三条不同的计算路径：算子的实现不同，数值精度的处理不同（第 7 章那条激活精度谱系在不同后端上落点也不同），量化权重的反量化方式也可能有细微差别。这些差别单看每一步都极小，小到肉眼看不见。但它们会累积进 logits——那组决定"下一个字是什么"的分数。

关键在最后一步：采样。当两个候选 token 的分数本就接近时，后端间那点数值微差，足以让排序翻个个儿，选出不同的 token。一旦某一步选了不同的字，后面整段就顺着岔开了。所以换后端输出会变，不是谁算错了，而是浮点计算在不同硬件路径上本就不可能逐比特一致。自回归又把这点微差沿着序列放大了。（这是基于代码与浮点常识的解释；`LiteRT-LM#2281` 的具体现象按上游报告，属【文档】级。）

这个现象有个实际含义：端侧的"可复现"要带上后端这个条件。本书附录 D 的基准数据集因此严格固定后端——换了后端，就是另一组数据，不能混着比（这也是第 2 章数字纪律的由来）。

## 小结

三类后端，各有强项也各有麻烦：CPU 随时可用但要做线程与亲和性的功课，GPU 并行强还能片上采样省搬运，NPU 最省电但最封闭。LiteRT-LM 用一个枚举加一个工厂把它们藏在同一个 `LlmExecutor` 接口后面，上层无感。而"换后端连输出都变"这个反直觉现象，根子是浮点在不同硬件路径上算不出逐比特一致的结果，又被自回归放大。它也提醒我们：端侧的性能与正确性讨论，都要带上"哪个后端"这个前提。

下一章是第三篇的压轴：推测解码与 MTP——一次模型前向，怎么吐出不止一个字。

---

## 练习与自查

1. **省了什么。** 片上采样每步省一次约 1 MiB 的 logits 回搬，按 50 tok/s 折合每秒约 50 MB。对照权重流每秒上百 GB，这不到 0.1%。那片上采样省下的主要是什么？
2. **绑核逻辑。** CPU 亲和性为什么绑中大核而不是全部核？两个理由。
3. **工厂价值。** `default` 分支返回 `InvalidArgumentError` 而非崩溃，这个设计防住了什么？
4. **输出漂移。** 换一个后端，同一 prompt 的输出为什么可能不同？这算 bug 吗？
5. **设计题。** 要给运行时接入一个新后端 XPU，从本章的分发路径出发，列出至少三处必须改动的位置。


<!-- NPU 无真机，全程标注"基于代码分析"。片上采样省多少、cpu 线程数扫描、cpu vs gpu 对比 待基准 D 回填〔基准 D〕。#2281 现象按【文档】级引用，成因为基于浮点常识的解释、未臆测 issue 内部。图 8-2(数据路径) 表 8-1(权衡) 规格见 notes.md，本轮出签名图 8-1。 -->
