# 第 8 章 异构算力：CPU、GPU 与 NPU

> 使命：讲清一部手机上三类计算单元各自的脾气，LiteRT-LM 如何用一个工厂把它们藏在同一套接口后面，以及一个让很多人困惑的现象——为什么换个后端，连模型的输出都会变。

第 1 章说过第三堵墙有两半：功耗和异构。这一章处理异构这半。手机上不止一个能算的地方，CPU、GPU、NPU 各有各的强项和麻烦。一个端侧运行时既要用得上它们，又要对上层藏起它们的差异。先从一个让人挠头的现象说起。

## 同一个模型，三种脾气

第 2 章"二十个问题"的第 17 问，来自一个真实的上游困惑（`LiteRT-LM#2281`）：同一个 `.litertlm` 文件，`--backend=cpu` 和 `--backend=gpu` 跑出来，不只是速度不同，连**输出的文字都可能不一样**。这反直觉——不是同一个模型、同一个提示词吗？

先接受这个事实，本章最后一节解释它的成因。现在只需记住：后端不是"同一个计算的不同速度"，它们是**不同的计算路径**。理解了这一点，"可插拔后端"这个设计的分量才显出来——它不是锦上添花，是端侧必须面对的现实。

## 一个工厂，按 Backend 分派

LiteRT-LM 用一个枚举把后端列全（`runtime/executor/executor_settings_base.h:34 @ v0.13.1`）：

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

选哪个实现，交给一个工厂函数（`runtime/executor/llm_litert_compiled_model_executor_factory.cc:165 @ v0.13.1`）：

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

`(1)` 后端来自 `LlmExecutorSettings`，而它最终由 CLI 的 `--backend` 或上层配置写入。分派的输入是一个纯数据字段，不是运行时探测。`(2)` `CPU` 和 `GPU` 共用 `case`、落到同一个创建函数，这不是偷懒：这条路径读出的是同一个 `.tflite` 子图（`ModelType::kTfLitePrefillDecode`，`:135`），CPU 与 GPU 的差异被推迟到 LiteRT 编译期由 delegate 决定，工厂层不必区分。`(3)` `NPU` 走独立分支，因为它加载的根本是另一组模型文件（下一节 NPU 的 embedder 子模型就是证据）。`default` 分支把 `CPU_ARTISAN` 这类未接入编译路径的后端挡在门外，返回错误而非崩溃——这是工厂作为唯一入口的价值：非法后端在这里一次性拦下。

CPU/GPU 那条路径内部还有一次分派，按模型是否动态形状再分（`:139`）：

```cpp
ASSIGN_OR_RETURN(bool is_dynamic_model, IsDynamicModel(*litert_model));
if (is_dynamic_model) {
  ASSIGN_OR_RETURN(executor, LlmLiteRtCompiledModelExecutorDynamic::Create(...));
} else {
  ASSIGN_OR_RETURN(executor, LlmLiteRtCompiledModelExecutorStatic::Create(...));  // (1)
}
```

`(1)` 静态形状走 `Static` 执行器：形状在编译期定死，运行时不再变，省掉重新编译的开销。这层选择对上层同样不可见。

无论走哪条分支，工厂返回的都是 `absl::StatusOr<std::unique_ptr<LlmExecutor>>`——上层拿到的是同一个 `LlmExecutor` 抽象（第 5 章那个接口），完全不知道底下是 CPU 还是 NPU、是 Static 还是 Dynamic。这就是"可插拔后端"落到代码里的样子：一个枚举、一个工厂、一个共同接口。加一个新后端，就是加一个 `case`，上层一行不改。

<figure>
{{#include figs/fig-8-1.svg}}
<figcaption>图 8-1　后端工厂分派：Backend 枚举经工厂函数分成 CPU/GPU 一路、NPU 一路，都产出同一个 LlmExecutor 抽象。上层对后端差异无感知。</figcaption>
</figure>

## CPU：随时可用，但要做线程功课

CPU 的好处是随时都在、什么算子都能跑。它的挑战是第三堵墙的另一半——功耗与调度。

手机 CPU 是大小核混合的：几个高性能核，几个高能效核。推理这种重活，跑在性能核上才快；但如果不管不顾，系统调度器可能把线程挪到能效核上，速度就掉了。LiteRT-LM 为此提供了 CPU 亲和性工具（`runtime/engine/cpu_affinity_utils.h @ v0.13.1`），三个函数：判定是不是 Pixel Tensor 设备（`:25`）、查出性能核编号（`:29`）、把当前线程绑上去（`:35`）。"性能核编号"不是运行时算出来的，是按芯片型号硬编码的一张表（`cpu_affinity_utils.cc:57 @ v0.13.1`）：

```cpp
const TensorCoreAffinity kTensorAffinities[] = {
    {PixelSoc::kTensorG3, {4, 5, 6, 7, 8}},   // (1)
    {PixelSoc::kTensorG4, {4, 5, 6, 7}},
    {PixelSoc::kTensorG5, {2, 3, 4, 5, 6, 7}},
    {PixelSoc::kTensorG6, {2, 3, 4, 5, 6, 7}},
};
```

`(1)` 每一行是一款 Pixel SoC 的中大核编号：G3 用 4~8 号核、G4 只用 4~7 号，核编号还随代际变（G5/G6 从 2 号起）。芯片型号靠读 Android 系统属性 `ro.soc.model` 识别（`:80`），认不出就返回空表。这张表也划定了这套工具的适用边界：它只认 Google 自家的 Tensor 芯片，高通、联发科的设备走不到这里——`IsPixelTensorDevice()` 直接返回 false。

绑核的动作落在 `SetCpuAffinity`（`cpu_affinity_utils.cc:103 @ v0.13.1`），本质是一次 Linux 系统调用：

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

`(1)` 把每个性能核编号写进一个位掩码；`(2)` `sched_setaffinity` 的第一个参数是 0，代表"当前线程"——这是一个建议而非命令，头文件的注释也说得明白："The scheduler will then attempt to run the thread on these cores most of the time"（`cpu_affinity_utils.h:31`）。调度器多数时候会照办，但没有硬保证。绑核失败不致命：只记一条 warning，推理照跑，只是可能落到能效核上慢一点。

这段代码在"一个 token 的一生"里的位置很靠前——它不在 decode 循环里，而在引擎创建时执行一次（`engine_factory.h:136 @ v0.13.1`）：`if (IsPixelTensorDevice())` 才查核、绑核。绑一次，此后整个会话的推理线程都倾向留在性能核上。

线程本身由一个线程池管（`runtime/framework/threadpool.h:51 @ v0.13.1`），构造时给一个上限 `max_num_threads`（`:57`）。用几个线程是个权衡：多了未必快（内存带宽是瓶颈，第 1 章），还更耗电、更烫。这个数默认是 4（`CpuConfig::number_of_threads`，`runtime/executor/llm_executor_settings.h:120 @ v0.13.1`），CLI 用 `--num_cpu_threads` 覆盖（对应上游需求 `LiteRT-LM#2505`）。它的接线只有两行（`litert_lm_lib.cc:528 @ v0.13.1`）：

```cpp
if (settings.num_cpu_threads > 0) {                  // (1)
  cpu_settings.number_of_threads = settings.num_cpu_threads;
}
```

`(1)` 只有传了正值才覆盖默认的 4——传 0 或不传就保留默认。这些都是"可持续速度"的功课：不是把资源开满，而是让线程数匹配内存带宽这个真正的上限。绑核选性能核、线程数压在带宽上限附近，两件事合起来才是 CPU 后端在第三堵墙下的正确姿势。

## GPU：并行强，还能就地采样

GPU 的强项是大规模并行，适合矩阵运算这种活。它在端侧的一个精巧优化，第 5 章已经埋过引子——**片上采样**。

回忆第 5 章的两条采样路径。内部采样之所以快，是因为采样这一步可以直接在 GPU 上做，不用把 logits 搬回 CPU。这个"省一次搬运"值多少？decode 每一步都会产出一整组 logits——词表有多大，这组数就有多大，动辄几十万个。如果采样在 CPU 做，这几十万个数每步都要从 GPU 拷回 CPU；在 GPU 上就地采样，这次拷贝就省了。

采样器怎么建、跑在哪，都在 `InitializeSampler` 里定（`runtime/executor/llm_litert_compiled_model_executor.h:160 @ v0.13.1`，实现见 `.cc:1335`）：

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

`(1)` 采样器有自己的 `sampler_backend`，跟主执行器的后端各算各的：GPU 主后端可以配 GPU 采样器，让 logits 不离开显存。`(2)` `gpu_sampler_max_top_k_` 存下 top-k 的 `k`：GPU 上做 top-k 要预先知道候选个数才能开好缓冲区，这个字段就是给 GPU 采样路径准备的（`.h:354`）。`(3)` `sampler_handles_input_` 是关键开关——它为真时，采样器接管 decode 的输入张量，选出的 token 直接在设备上喂回下一步，连"把新 token 拷回来再拷过去"都省了。注意它的最后一个条件 `!runs_embedding_on_gpu`：如果 embedding 本来就在 GPU 上查（`embedding_lookup_ == nullptr`），这条优化反而关掉：两条 GPU 优化路径不叠加，代码在这里划了清楚的边界。

这正是第 18 问的答案。decode 是整个生成里最频繁的操作，每一步省掉这一次数据搬运，累积起来不小。（这一项很难从整机数字中单独剥离，本书未单测；两个后端的整体差距见附录 D——本书基准上 gpu 的 decode 约为 cpu 的 2 倍、prefill 约 3.9 倍〔基准 D〕。）

## NPU：能效最高，接口最封闭

NPU 是三者里能效比最高的——同样的活，它最省电、最不发热，这在端侧是硬通货。代价是它最封闭：接口是厂商的（比如高通的 QNN），能跑的算子有限，灵活性最低。

本节的内容基于对代码的阅读，没有真机验证（作者手上是一台 Mac，跑不了手机 NPU），所以只讲代码能佐证的部分。NPU 走的是工厂里那条独立路径（`CreateNpuLlmLiteRtCompiledModelExecutor`），产出一个专门的执行器（`runtime/executor/llm_litert_npu_compiled_model_executor.h:51 @ v0.13.1`）。这个执行器的类注释写着 "Component intended to be used with an NPU variant of Gemma3"（`:50`）——它不是通用执行器，是为 NPU 版 Gemma3 专门做的。

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

下一章是第三部的压轴：推测解码与 MTP——一次模型前向，怎么吐出不止一个字。

---

## 参考

- 后端枚举与工厂：`runtime/executor/executor_settings_base.h:34 @ v0.13.1`（`Backend`，`CPU_ARTISAN`:39、CPU/GPU 于 `:45`/`:48`、`NPU`:54）；`runtime/executor/llm_litert_compiled_model_executor_factory.cc @ v0.13.1`（`CreateLlmLiteRtCompiledModelExecutor`:165；`GetBackend`:168；分派:170/174；default 拦截:177；CPU/GPU 内部动/静态分派:139；读取 prefill/decode 子图:135）。
- 片上采样：`runtime/executor/llm_litert_compiled_model_executor.h @ v0.13.1`（`InitializeSampler`:160；`gpu_sampler_max_top_k_`:354）；实现于 `.cc:1335`（`GetSamplerBackend`、`sampler_handles_input_`、`runs_embedding_on_gpu`）。
- 线程与亲和性：`runtime/framework/threadpool.h:51 @ v0.13.1`（`max_num_threads` 构造:57）；`runtime/engine/cpu_affinity_utils.h @ v0.13.1`（`IsPixelTensorDevice`:25、性能核:29、设亲和性:35）；实现于 `.cc`（`kTensorAffinities` 表:57、SoC 识别:80、`SetCpuAffinity`/`sched_setaffinity`:103）；调用点 `runtime/engine/engine_factory.h:136 @ v0.13.1`；线程数默认 `runtime/executor/llm_executor_settings.h:120 @ v0.13.1`、CLI 接线 `runtime/engine/litert_lm_lib.cc:528 @ v0.13.1`（`--num_cpu_threads`）。
- NPU：`runtime/executor/llm_litert_npu_compiled_model_executor.h @ v0.13.1`（类:51、类注释:50；`LatencyStats` 逐段计时:60-82；`EmbedderContext`:266、`EmbedderPerLayerContext`:287、`NpuAuxiliaryContext`:316/注释:314）。

<!-- NPU 无真机，全程标注"基于代码分析"。片上采样省多少、cpu 线程数扫描、cpu vs gpu 对比 待基准 D 回填〔基准 D〕。#2281 现象按【文档】级引用，成因为基于浮点常识的解释、未臆测 issue 内部。图 8-2(数据路径) 表 8-1(权衡) 规格见 notes.md，本轮出签名图 8-1。 -->
