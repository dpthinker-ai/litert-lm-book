# 第 8 章 异构算力：CPU、GPU 与 NPU

> 本章说明 CPU、GPU、NPU 的适用条件与运行时路径。重点分析 buffer 交接、同步成本，以及后端变化为何可能改变输出。

第 6、7 章分析了 KV cache 与模型文件两笔内存账；本章转向三类约束中的异构后端。手机 SoC 同时包含 CPU、GPU 与 NPU，各自支持的算子、数值路径和部署条件不同。运行时需要选择相应的执行路径，同时保持上层接口稳定。

## 8.1　后端选择改变执行路径

上游 issue `LiteRT-LM#2281` 报告：同一个 `.litertlm` 文件分别使用 `--backend=cpu` 和 `--backend=gpu` 时，速度不同，输出文本也可能不同。[^ch08-issue-2281] 这也是第 2 章第 17 问的来源。

后端选择会改变实际计算路径。算子内核、数值精度处理和反量化实现都可能随之变化。LiteRT-LM 因此需要在运行时选择后端，并将这些差异封装在共同接口内。

## 8.2　Backend 工厂分派

LiteRT-LM 用一个枚举表示后端：

```cpp
// runtime/executor/executor_settings_base.h:34
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

`(1)` 带 `ARTISAN` 后缀的枚举值对应手写算子路径；`(2)` 不带后缀的 `CPU`、`GPU` 走 LiteRT 编译路径。同一类硬件对应两个枚举值，是因为底层算子实现不同。本章聚焦 `CPU`、`GPU` 与 `NPU` 这条编译路径。`(3)` `NPU` 在工厂中进入独立分支。`GOOGLE_TENSOR_ARTISAN` 对应 Pixel Tensor 的手写路径，此处不展开。

工厂函数按 `Backend` 枚举值选择实现：

```cpp
// runtime/executor/llm_litert_compiled_model_executor_factory.cc:165
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

`(1)` 后端由 CLI 的 `--backend` 或上层配置写入 `LlmExecutorSettings`；工厂按配置分派，不探测设备。`(2)` `CPU` 和 `GPU` 共用一个创建函数。这条路径读取同一个 prefill/decode 子图（`ModelType::kTfLitePrefillDecode`），硬件差异由 LiteRT 编译选项和 delegate 处理。`(3)` `NPU` 使用专门的执行器和模型资源。`default` 分支对未接入这条编译路径的枚举值返回 `InvalidArgumentError`，使错误在工厂边界暴露。

<div class="aside-compare">

llama.cpp 维护后端注册表，按编译选项注册 CUDA、Metal、SYCL、Vulkan、WebGPU 等实现，并将 CPU 作为后备路径。[^ch08-llamacpp-registry]一个二进制可以包含多个后端，并在运行时探测设备。MLC-LLM 使用 TVM 针对具体目标编译模型库。[^ch08-mlc-compile] LiteRT-LM 则由 `Backend` 工厂选择执行器，并要求部署产物提供相应的模型资源。三种设计的主要差异是目标特化程度、产物数量与运行时选择空间。

</div>

### 8.2.1　静态与动态形状分派

CPU/GPU 创建函数还会根据导出模型的形状类型分派静态或动态执行器：

```cpp
// runtime/executor/llm_litert_compiled_model_executor_factory.cc:138
ASSIGN_OR_RETURN(bool is_dynamic_model, IsDynamicModel(*litert_model));
if (is_dynamic_model) {
  ASSIGN_OR_RETURN(executor, LlmLiteRtCompiledModelExecutorDynamic::Create(
                                 executor_settings, lrt_env, resources));  // (1)
} else {
  ASSIGN_OR_RETURN(executor, LlmLiteRtCompiledModelExecutorStatic::Create(
                                 executor_settings, lrt_env, resources));  // (2)
}
```

`is_dynamic_model` 由模型张量形状计算，不是用户配置项。`IsDynamicModel` 检查 prefill 子图中两组张量的形状是否可在运行时变化：

```cpp
// runtime/executor/llm_litert_compiled_model_executor_factory.cc:98
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

`(1)` 检查 KV cache 的 K、V 张量是否同为动态或同为静态。`(2)` 再要求 KV cache 与序列长度张量采用一致的动静态设置。混合配置会返回失败状态，因此执行器不会按不一致的形状继续初始化。

两条路径的缓冲管理不同。静态执行器使用导出时确定的 prefill、decode 与 KV cache 形状；动态执行器按 `kv_increment_size` 扩展 KV cache，默认每次增加 16 个位置。动态形状允许缓冲区随当前序列长度增长，但底层 delegate 可能需要重新准备张量；实际峰值内存仍取决于 delegate 的分配策略。`prefill_chunk_size` 的注释注明它只适用于动态导出的模型。两种执行器都实现 `LlmExecutor`，上层不需要按形状类型分支。

### 8.2.2　后端枚举如何下沉为编译选项

工厂分支只决定创建哪一种执行器。CPU 与 GPU 的硬件请求继续下沉到 `CreateCompilationOptions`。GPU 分支设置激活精度、缓冲模式与 GPU 专属选项，最后请求 `kGpu`；CPU 分支设置线程数与 XNNPACK 缓存，最后请求 `kCpu`：

```cpp
// runtime/executor/llm_executor_settings_utils.cc:76-86
// runtime/executor/llm_executor_settings_utils.cc:214-255
switch (executor_settings.GetBackend()) {
  case Backend::GPU: {
    // ...
    if (activation_data_type == ActivationDataType::FLOAT32) {
      gpu_compilation_options.SetPrecision(GpuOptions::Precision::kFp32);
    } else {
      gpu_compilation_options.SetPrecision(GpuOptions::Precision::kFp16);
    }
    // ...
    compilation_options.SetHardwareAccelerators(HwAccelerators::kGpu);
    break;
  }
  case Backend::CPU: {
    // ...
    cpu_compilation_options.SetNumThreads(num_threads);
    // ...
    compilation_options.SetHardwareAccelerators(HwAccelerators::kCpu);
    break;
  }
```

GPU 的 `external_tensor_mode` 来自 `GpuConfig`，默认值为 false。false 不表示所有张量都成为 delegate 内部张量。代码仍把名称匹配 `kv_cache_` 的张量标为 external；采用单缓冲 KV cache 时，还把 `param_tensor` 标为 external 和 buffer storage。GPU sampler 启用后，`logits` 也成为 external tensor。这些规则使跨 signature 复用和设备侧采样有可绑定的缓冲，但不能单凭名称证明底层没有格式转换。

CPU 路径没有对应的 external tensor 开关。它把 `CpuConfig::number_of_threads` 传给 CPU 编译选项，并启用 XNNPACK 的动态 fully connected 标志。因此，`--backend=cpu` 不只是工厂选择，还会改变线程池、delegate 与权重缓存路径。比较 CPU 和 GPU 时，初始化时间也必须分开记录；GPU program cache 与 CPU XNNPACK weight cache 并非同一项缓存。

NPU 不经过上述 CPU/GPU switch。专用执行器创建的主模型选项同时允许 `kNpu | kCpu`，Android 上还为 Qualcomm HTP 与 Google Tensor 设置 burst 模式。同一执行器里的 text embedder 则用仅含 `kCpu` 的选项单独编译。因此，配置成 `Backend::NPU` 只说明进入 NPU 专用执行器，不能据此声称每个子图和每个算子都驻留在 NPU。

CPU、GPU 与 NPU 分支都返回 `absl::StatusOr<std::unique_ptr<LlmExecutor>>`。上层只依赖第 5 章介绍的 `LlmExecutor` 接口，不需要区分硬件后端或形状类型。接入新后端仍需实现执行器、准备相应模型资源并在工厂中增加分支；只要接口契约不变，调用方无需随之修改。

<figure>
{{#include figs/fig-8-1.svg}}
<figcaption>图 8-1　工厂按 Backend 选择 CPU/GPU 通用执行器或 NPU 专用执行器，两条路径都返回 LlmExecutor 接口。</figcaption>
</figure>

## 8.3　CPU：线程配置与 Pixel 亲和性

CPU 通常具有较宽的算子覆盖范围，也是没有可用加速器时的后备执行单元。LiteRT-LM 为 CPU 路径提供线程数配置，并在部分 Pixel Tensor 设备上设置线程亲和性。这两项配置影响调度方式，但不能脱离具体设备推导出统一的性能最优值。

### 8.3.1　CPU 亲和性：Pixel Tensor 的硬编码核表

移动 SoC 常把不同性能特征的 CPU 核组合在一起。LiteRT-LM 提供亲和性工具，用于限制推理线程可运行的 CPU 集合。相关接口分别识别 Pixel Tensor 设备、查询预设的性能核编号，以及设置当前线程的 affinity mask。核编号不是运行时推导的，而是按芯片型号硬编码：

```cpp
// runtime/engine/cpu_affinity_utils.h:25
// runtime/engine/cpu_affinity_utils.h:29
// runtime/engine/cpu_affinity_utils.h:35
// runtime/engine/cpu_affinity_utils.cc:57
const TensorCoreAffinity kTensorAffinities[] = {
    {PixelSoc::kTensorG3, {4, 5, 6, 7, 8}},   // (1)
    {PixelSoc::kTensorG4, {4, 5, 6, 7}},
    {PixelSoc::kTensorG5, {2, 3, 4, 5, 6, 7}},
    {PixelSoc::kTensorG6, {2, 3, 4, 5, 6, 7}},
};
```

`(1)` 每一行给出一款 Pixel SoC 的预设核编号。G3 使用 4 至 8 号核，G4 使用 4 至 7 号核，G5、G6 使用 2 至 7 号核。当前实现不分析运行时拓扑，而是依赖这张版本内置表；表外设备不会自动得到同类配置。

`GetCurrentPixelSoc` 读取两个系统属性来识别芯片：

```cpp
// runtime/engine/cpu_affinity_utils.cc:66
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

`(1)` 读取制造商 `ro.soc.manufacturer` 与型号 `ro.soc.model`。`(2)` 制造商不是 Google 时返回 `kUnknown`。`(3)` 随后逐一匹配型号字符串。结果保存在函数内的 `static` 局部变量中，初始化后不再重复读取系统属性。未匹配的型号同样返回 `kUnknown`，`IsPixelTensorDevice()` 因而返回 false。

这张表与属性检查把适用范围限定为已列出的 Google Tensor 芯片。`GetPixelPerformanceCores()` 对未知 SoC 返回空向量；`SetCpuAffinity` 遇到空向量时记录 warning 并返回 `OkStatus`。非 Android 编译分支中，`IsPixelTensorDevice` 恒为 false，`SetCpuAffinity` 也直接返回成功。因此，这段代码不能为 Qualcomm、MediaTek 或未列入表中的 Pixel SoC 自动选择 CPU 核。

`SetCpuAffinity` 最终调用 Linux 的 `sched_setaffinity`：

```cpp
// runtime/engine/cpu_affinity_utils.cc:115
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

`(1)` 把预设核编号写入位掩码。`(2)` 第一个参数为 0，表示调用线程。系统调用成功后，这个 mask 限制该线程允许运行的 CPU 集合；调度器仍可在集合内部选择 CPU，但这不是软性建议。系统调用失败时，`SetCpuAffinity` 返回 `InternalError`。调用方 `EngineFactory::Create` 只记录 warning 并继续创建引擎，因此亲和性设置失败不会直接终止引擎创建。

亲和性设置在引擎创建时执行一次，不在 decode 循环内。`EngineFactory::Create` 先判断 `IsPixelTensorDevice()`，再对调用线程设置 mask；该调用不按 CPU、GPU 或 NPU 后端分支。Linux 新建线程会继承创建线程的 affinity mask，因此随后由该线程创建的工作线程通常继承相同集合。这个结论不覆盖此前已存在的线程，也不排除子线程之后重新设置自己的 mask。

### 8.3.2　线程数：默认值与设备扫描

CPU 算子通过线程池并行执行，构造参数 `max_num_threads` 给出线程数上限。增加线程数可能提高矩阵运算吞吐，也会增加调度开销，并逐步受到内存带宽限制。线程数对功耗、温度和持续性能的影响需要在目标设备上同时测量，本章的扫描只记录吞吐。

`CpuConfig::number_of_threads` 的默认值为 4，注释直接写 "The default value is 4"。C++ 命令行入口用 `--num_cpu_threads` 覆盖（对应上游需求 `LiteRT-LM#2505`[^ch08-issue-2505]；本书采集用的 Python CLI 未暴露该 flag）。覆盖路径位于 CPU 后端专属的配置分支：

```cpp
// runtime/executor/llm_executor_settings.h:120
// runtime/engine/litert_lm_lib.cc:523
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

`(1)` 只有正值才覆盖默认的 4；传 0 或不传时保留默认值。`(2)` 同一分支还设置 `prefill_chunk_size`，而该参数只对动态导出的模型生效。整个分支由 `backend == Backend::CPU` 守卫，`CpuConfig` 不会用于 GPU 或 NPU 路径。

本书在一台 Qualcomm 设备上扫描了 1、2、4、8 个线程，context 为 1024，每个条件运行 3 次并取中位数（见附录 D 第十三节）。decode 吞吐依次为 4.4、7.4、10.1、13.4 tokens/s，prefill 吞吐依次为 22.8、45.3、77.9、131.6 tokens/s。把线程数从 4 加倍到 8，decode 提高约 33%，prefill 提高约 69%，都低于理想的 100% 增幅。相对 1 线程，8 线程的加速比分别约为 3.0 倍和 5.8 倍，对应约 38% 和 72% 的并行效率。数据只表明 8 线程在这四个测试点中吞吐最高，默认 4 线程不是该设备上的最高吞吐测试点。它不能证明 8 线程尚未饱和，也不能确定全局最优值。`--num_cpu_threads` 允许针对目标设备继续扫描，并把功耗、温度与持续性能纳入选择。

## 8.4　GPU：并行执行与设备侧采样

GPU 适合并行执行矩阵运算。第 5 章介绍过设备侧采样（device-side sampling）：采样器直接消费 GPU 产生的 logits，避免在每个 decode step 把完整 logits 张量传递给 host 采样器。

decode 每一步都会产生长度等于词表大小的 logits。以本书基准模型的 262144 词表和 FP32 logits 计算，张量大小为 262144 × 4 字节 = 1 MiB。host 采样需要让这 1 MiB 数据对 CPU 可见；设备侧采样则在 GPU 侧消费它。生成 512 个 token 时，host 采样要处理 512 份完整 logits。设备侧采样只减少完整 logits 的跨设备传输量。选中的少量 token id 仍会复制到 host，用于返回结果和更新运行时状态。本书没有隔离测量该优化的吞吐或时延收益，因此不能从 CPU/GPU 整机结果中单独归因。

<figure>
{{#include figs/fig-8-2.svg}}
<figcaption>图 8-2　设备侧采样保留完整 logits 在 GPU 侧；选中 token 可直接写入下一步设备输入，同时少量 token id 仍返回 host。</figcaption>
</figure>

采样器的后端和输入接管条件 中确定（声明，实现）：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.h:160
// runtime/executor/llm_litert_compiled_model_executor.cc:1335
ASSIGN_OR_RETURN(auto sampler_backend, GetSamplerBackend(executor_settings_));  // (1)
// ...
ASSIGN_OR_RETURN(
    sampler_,
    CreateSampler(sampler_backend, output_heads, std::move(sampler_params),
                  env_.Get(), /*sequence_size=*/1, vocab_size, data_type));
// ...
const bool runs_embedding_on_gpu = (embedding_lookup_ == nullptr);  // (2)
sampler_handles_input_ =
    (!executor_settings_.GetAdvancedSettings().has_value() ||
     executor_settings_.GetAdvancedSettings()->sampler_handles_input) &&
    sampler_->CanHandleInput() && !signatures_.input_tokens.empty() &&
    !runs_embedding_on_gpu;
```

`(1)` 采样器通过独立的 `sampler_backend` 创建，主执行器使用 GPU 时可以同时选择 GPU sampler，使完整 logits 无需返回 host。`(2)` 只有采样器声明 `CanHandleInput()`、模型具有 `input_tokens`，并且 embedding 不在主 GPU 图中执行时，`sampler_handles_input_` 才会为 true。源码注释把 `embedding_lookup_ == nullptr` 定义为 embedding 在 GPU 图内执行。因此，"GPU 采样"和"采样器接管下一步输入"是两个不同条件；GPU 采样成立，不保证输入接管也成立。

### 8.4.1　避免完整 logits 回传：采样器接管下一步输入

`sampler_handles_input_` 为 true 时，采样器除输出 token id 外，还可以填写下一步 decode 的输入张量。这个机制减少输入准备中的往返，但不会取消 token id 返回 host 的步骤。

`InitializeSampler` 先为 decode 的 position 和 attention mask 保存上一轮缓冲（`decode_prev_input_pos_`、`decode_prev_mask_`），再设置并立即复位输入接管：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:1383
RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/false));
RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/true));  // (1)
```

`(1)` 源码注释说明，这两个调用先准备底层模型，再解除输入张量绑定。实际绑定在 decode 阶段通过 `SwapSamplerInputTensors` 完成。

`SetSamplerInputHandling` 把张量指针和回调传递给采样器：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:1402
return sampler_->SetInputTensorsAndInferenceFunc(
    &decode_input_buffers_[signatures_.input_tokens], &decode_prev_input_pos_,
    &decode_input_buffers_[signatures_.input_positions],
    has_input_attn_mask ? &decode_prev_mask_ : nullptr,
    has_input_attn_mask ? &decode_input_buffers_[*signatures_.input_attn_mask]
                        : nullptr,
    BindTensorsAndRunDecodeStatic, this);  // (1)
```

第一个参数指向 decode 的 `input_tokens` 缓冲。`Sampler::CanHandleInput()` 的接口契约说明，支持该能力的采样器可以用输出 token 填写输入 token，并更新 position、mask 等张量。`(1)` 传入的 `BindTensorsAndRunDecodeStatic` 回调会绑定这些张量并执行下一步 decode：

```cpp
// runtime/components/sampler.h:54
// runtime/executor/llm_litert_compiled_model_executor.cc:952
int LlmLiteRtCompiledModelExecutorBase::BindTensorsAndRunDecodeStatic(
    void* arg) {
  auto self = static_cast<LlmLiteRtCompiledModelExecutorBase*>(arg);
  auto status = self->BindTensorsAndRunDecode(/*output_logits=*/nullptr);  // (1)
  // ...
  return status.raw_code();
}
```

`(1)` 回调将 `output_logits` 设为 null，使用已经绑定的缓冲执行 decode。host 无需先读取 token id，再填写下一步设备输入。不过，`Decode()` 在 `SampleLogits` 返回后仍调用 `CopyFromTensorBuffer2D<int>`，把采样结果复制为 host 侧 `std::vector`。完整 logits 留在设备侧，选中 token 可直接写入设备输入，同时少量 token id 仍返回 host；decode 控制流程仍有 CPU 参与。

position 和 mask 张量需要保留当前轮与上一轮两组缓冲。`SwapSamplerInputTensors` 用 `std::swap` 交换两组 `TensorBuffer` 句柄，然后重新绑定。这两个交换语句本身不复制张量内容；它们通过轮换缓冲避免在同一轮同时覆盖仍需读取的数据。

附录 D 的整机基准使用 Apple M5 Pro、Gemma 4 E4B、context 1024，并生成 128 个 token。GPU 的 decode 吞吐约为 50.6 tokens/s，CPU 约为 24.7 tokens/s；prefill 分别约为 999 与 259 tokens/s。该结果同时包含算子、内存、delegate 与采样路径差异，不能据此计算设备侧采样的单独收益。

## 8.5　缓冲驻留：共享条件与退化路径

后端名称回答“由哪条执行路径处理”，buffer 类型则回答“数据目前能被谁直接访问”。两者不能互相替代。图 8-3 把配置、编译与缓冲绑定放在同一条链上。CPU/GPU 共用执行器类型，不表示它们共用缓冲策略；NPU 使用专用执行器，也不表示整条链只经过 NPU。

<figure>
{{#include figs/fig-8-3.svg}}
<figcaption>图 8-3　Backend 先选择执行器，再转换为编译选项；最终的数据驻留还取决于各 compiled model 创建的 buffer。</figcaption>
</figure>

零拷贝（zero-copy）指生产者与消费者复用同一底层存储，不为交接数据再复制一份字节。对异构执行而言，“代码里没有 `memcpy`”只是必要条件，不是充分条件。至少还要满足三项要求：两个计算阶段接受同一种 buffer 类型；张量形状、元素类型、布局与对齐相容；生产者完成事件能传递给消费者。delegate 或驱动仍可能在绑定、格式转换或同步时创建内部副本。

常规 CPU/GPU 执行器把 compiled model 创建的 `TensorBuffer` 保存下来。每次绑定前，它调用 `Duplicate()` 生成句柄，清除输出句柄上的旧事件，再调用 `RunAsync`。decode 完成提交后，执行器交换 KV cache 的输入、输出 map：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:913-949
for (const auto& [input_name, input_buffer] : decode_input_buffers_) {
  LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
  decode_input_buffers[input_name] = std::move(input_buffer_dup);
}
// ...
auto output_buffer_dup =
    output_logits && output_name == signatures_.output_logits
        ? output_logits->Duplicate()
        : output_buffer.Duplicate();
// ...
output_buffer_dup->ClearEvent();
// ...
compiled_model_->RunAsync(kDecodeSignatureRunner, decode_input_buffers,
                          decode_output_buffers, async));
```

这些语句没有显式复制张量数据。`ClearEvent()` 与 `RunAsync` 又表明，句柄除指向存储外还参与完成事件的管理。后续消费者读取结果时，等待可能发生在 buffer 锁定、采样或下一次绑定处。仅统计 `RunAsync` 调用本身会漏掉这部分等待。

静态执行器对 CPU 和 GPU 的 KV cache 采取不同策略。CPU 为输入 buffer 调用 `Duplicate()`，把句柄同时放入输出 map；GPU 则为输出单独调用 `CreateOutputBuffer`。源码注释把 CPU 方案称为 single buffer，以降低内存和运行开销。GPU 方案采用输入、输出两套缓冲并在 step 后交换。另有一种 GPU 单缓冲模型通过 `param_tensor` 描述 cache 更新位置；是否采用该路径由 signature 中是否存在该输入决定。

下面几条路径说明共享何时退化为显式复制或同步。

| 交接位置 | LiteRT-LM 中的动作 | 直接共享所需条件 | 条件不满足时的路径 |
|---|---|---|---|
| CPU KV cache 的输入→输出 | `Duplicate()` 同一输入 buffer | CPU kernel 允许就地更新；布局一致 | 模型或后端不允许时，不能沿用该单缓冲方案 |
| GPU KV cache 的本轮→下一轮 | 两套 buffer 交换 map | 输出 buffer 可直接作为下一轮输入；事件可传递 | delegate 内部格式转换或等待不在 `std::swap` 中体现 |
| NPU embedder/辅助图→主模型 | 对主模型输入 buffer 调用 `Duplicate()` | 子图接受同一 storage、类型和布局 | buffer 类型不相容时需重新分配；驱动内部行为尚未实测 |
| GPU logits→约束解码 | 先读到 host，mask 后写回 | 当前实现只在 logits 已是 host memory 时直接修改 | 非 host buffer 执行完整 logits 回读与回写 |
| Session Clone / 多候选 KV 变换 | 新分配后 `memcpy` | 语义要求独立副本或 batch 重排 | 必然产生按 `PackedSize()` 计的复制 |

> 表 8-1　“共享句柄”“无显式复制”和“端到端零拷贝”是三种不同结论；只有最后一项需要把 delegate 与驱动行为也纳入证据。

约束解码给出一条可以直接确认的退化路径。若 logits 已在 host memory，执行器原地调用 `MaskLogits`。若 buffer 不在 host，代码先用 `CopyFromTensorBuffer` 读出全部 FP32 或 FP16 logits，修改后再调用 `Write` 写回。本书基准模型的 262144 词表对应 1 MiB FP32 logits；这条路径每个 decode step 都处理一整份 logits，而不是只处理最终 token id。

KV cache 也有明确的深复制路径。多候选 decode 在首次 prefill/decode 转换时，`CopyKvCacheBuffers` 锁定源、目标，并按 batch 广播或抽取一个分支。两种分支都调用 `memcpy`。Session Clone 则通过 `CopyTensorBuffer` 分配同等大小的目标，再复制 `PackedSize()` 字节。这些路径的开销不能归入 kernel 执行时间。

<figure>
{{#include figs/fig-8-4.svg}}
<figcaption>图 8-4　buffer 只有在存储、布局与完成事件均相容时才可直接交接，否则会出现重分配、回读回写或显式同步。</figcaption>
</figure>

### 8.5.1　源码中的 buffer 类型不相容案例

NPU 创建路径包含一个具体修正。Gemma3n 的第 19 组 prefill KV cache 没有连接到模型算子，LiteRT 因而为它分配 host memory。源码注释明确指出，该 buffer 与 NPU transformer 不相容。执行器改从 decode signature 创建 `cache_k19` 与 `cache_v19`，清零后替换原 buffer。

这个案例说明，张量名称和形状相同仍不足以复用。buffer 的来源 signature 会影响实际 storage。若诊断只比较地址或 `Duplicate()` 调用次数，就会遗漏这类后端可访问性问题。适当的检查顺序是先记录 `BufferType()`、元素类型、shape 与 stride，再核对 producer/consumer 的 signature，最后观察 delegate 是否接受绑定。

### 8.5.2　如何观察同步与复制成本

LiteRT-LM 的 `BenchmarkInfo` 记录 executor 初始化、整次 prefill、整次 decode 与 TTFT。prefill/decode 计时从 turn start 到 turn end，因而适合作为端到端基线。它不自动拆分 kernel、等待、采样和复制。要定位边界成本，需要在同一设备、同一模型和同一输入上做成对实验，并且每次只改变一个路径条件。

设备侧采样可以用两组配置做近似隔离。主执行器都使用 GPU，并固定 FP32 激活、greedy 参数、prompt 与生成长度；一组用 CPU sampler，另一组用 GPU sampler。固定 FP32 是为了避免同时引入 FP16 logits：静态执行器只在主后端为 GPU、激活为 FP16、sampler 也为 GPU 时强制创建 FP16 logits buffer。即便如此，两组差值仍包含 sampler kernel 与同步，不能直接当成总线传输时间。

设两组完成 \\(N\\) 个 decode step 的时间分别为 \\(T_{cpu}\\) 和 \\(T_{gpu}\\)，每步差值为

$$
\Delta t = \frac{T_{cpu}-T_{gpu}}{N}.
$$

给一个只用于说明算法的可复算示例：若 256 步的总差为 128 ms，则 \\(\Delta t=0.5\\) ms/step。1 MiB 除以 0.5 ms，按 1 GB = 10^9 字节换算，等效速率约为 2.10 GB/s。这个数只能称为“按 logits 大小折算的等效速率”，因为分母还包含等待和两种 sampler 的计算差。它不是内存总线或互连带宽的测量值。

另一个实验是保持 GPU sampler 不变，仅切换 `sampler_handles_input`。false 路径由 host 读取 token id 并准备下一步输入；true 路径允许 sampler 填写 token、position 和 mask。该开关真正生效还要求 sampler 支持输入接管、signature 含 `input_tokens`，且 embedding 不在 GPU 主图内。若前置条件不成立，两组配置会落到同一路径，差值没有解释力。

异步测量还要确认等待策略。静态 prefill 发现任一输入为 Metal memory 时，会改用同步 `Run`；其他路径才可能调用 `RunAsync`。GPU benchmark 模式又把同步等待类型设为 active，并可等待权重转换完成。因此，“打开 benchmark”并非只增加计时器。跨版本或跨工具比较时，必须记录等待模式和 warm-up 处理。

调试日志也会扰动被测路径。`num_logits_to_print_after_decode` 看似只打印少量值，但 `LogTensor` 在不能直接取得 CPU span 时，会先把整个 tensor 复制到 `std::vector`。它适合检查数值，不适合与关闭日志的吞吐结果混合统计。系统级 timeline 则应把 CPU 填充、delegate invoke、buffer lock、采样和 token 回调分别标记；只看到设备 kernel 之间有空隙，不能直接断定空隙全是复制。

NPU 执行器提供更细的内置分项。`LatencyStats` 分别累计输入准备、embedder、mask、RoPE、主 LLM、cache update、sampling 和 token queue 时间。prefill 对各子图调用前后使用 `absl::Now()` 计时。这些字段能帮助定位阶段，但本书两台设备都没有进入推理，因而没有可报告的 NPU 分项数据。

## 8.6　NPU：专用多子图路径与部署约束

LiteRT-LM v0.13.1 的 NPU 专用执行器包含 Qualcomm QNN/HTP 与 Google Tensor 配置，并使用专门的多子图路径。本书的两台探测设备均为 Qualcomm 平台。它们都未完成一次 NPU 推理，也没有采集功耗数据，因此本章不比较 NPU 与 CPU、GPU 的能效。

### 8.6.1　NPU executor 仍是一条异构流水线

主 transformer 的编译选项允许 NPU 与 CPU，embedder 则显式采用 CPU 选项。`NpuConfig` 还分别控制 mask、KV cache update、per-layer embedding 与 greedy sampling 的实现选择。构造函数根据这些配置切换 `MaskUpdateMethod`、`KVCacheUpdateMethod` 和 per-layer lookup 路径。所以，NPU executor 的性能取决于多段计算和交接，不等于一个 transformer NPU kernel 的耗时。

logits 路径也不同。主 NPU 模型的 decode 输出可以带 per-tensor 量化参数，创建执行器时读取 scale 与 zero point。需要返回 logits 时，执行器创建 host FP32 buffer，并调用 `DequantizeLogits` 写入该 buffer。常规内部 decode 则直接在 NPU 输出 buffer 上调用 `ApplyGreedySampling`，并可选择 NEON 实现。这两条路径不能用同一份“logits 回传成本”估算。

以上结论都来自冻结版源码。它们确认了组件边界、配置和调用顺序，却不能确认目标设备上的 delegate 分区、实际 storage 类型或同步时长。后文涉及 NPU buffer 共享和 latency 字段时，也沿用这一证据边界。

工厂通过 `CreateNpuLlmLiteRtCompiledModelExecutor` 创建 `LlmLiteRtNpuCompiledModelExecutor`。冻结版源码的类注释把它限定为 Gemma3 的 NPU 变体，不能据此推断任意模型都可进入这条路径。

### 8.6.2　CPU/GPU embedder 与 NPU 多子图

CPU/GPU 执行器并非总把 embedding 融入主图。初始化时，`InitializeEmbeddingLookups` 查询 `ModelType::kTfLiteEmbedder`；模型包包含该段时，它创建独立的 `EmbeddingLookupManager`。`EmbeddingLookupText::Initialize` 又为 embedder 创建自己的 `CompiledModel`。包内没有独立 embedder 时，`embedding_lookup_` 保持为 null；此时主图必须提供 `input_tokens`，否则输入缓冲初始化返回失败。`InitializeSampler` 把 GPU 后端上的这个分支称为 embedding 在 GPU 图内执行。因此，CPU/GPU 路径同时支持独立 embedder 与主图内 embedding，具体形式由模型包和主图签名决定。

NPU 执行器在主 LLM 图之外还定义了多组上下文。以下三个结构展示了其中的 embedder、per-layer embedder 与辅助子图：

```cpp
// runtime/executor/llm_litert_npu_compiled_model_executor.h:266
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

`(1)` `EmbedderContext` 持有独立的 embedder `CompiledModel`。`(2)` per-layer embedder 也有自己的 compiled model。CPU/GPU 路径同样支持这两类可选 embedder；NPU 路径的额外拆分体现在 `(3)`：`NpuAuxiliaryContext` 的注释明确列出 Mask、RoPE 与 KV cache update 三组 signature。它们与主 LLM compiled model 由 NPU 执行器分别调用。

延迟统计字段按阶段记录 `prefill_embedder_inference_latency_us`、`prefill_mask_inference_latency_us`、`prefill_rope_inference_latency_us`、`prefill_llm_inference_latency_us` 与 `prefill_cache_update_inference_latency_us`。prefill 实现也依次调用 embedder、RoPE、mask、主 LLM 与 cache update 路径，确认了分段执行结构。源码没有说明每个子图拆分的完整设计理由。本书据此只陈述调用边界，不把它归因于未经验证的算子限制。

### 8.6.3　两台真机到达的初始化阶段

第一台设备是 nubia P0210（canoe，HTP V81）。测试使用 `gemma-4-E2B-it_qualcomm_sm8750.litertlm`，过程记录在 `experiments/data/npu_enablement.md`。在这台设备上，dispatch 桥加载、QNN manager 初始化、模型分段读取与 DispatchDelegate 解析均已通过，1.18 GB 的 QNN context 也创建成功。随后没有进入模型推理。独立运行 `qnn-platform-validator` 时，DSP 测试返回 "Please use testsig if using unsigned images"；同时，设备 `/vendor` 中的厂商库对 shell 不可读。两项现象与当前 ROM 和 DSP 镜像签名或访问限制一致。这次实验没有改变 ROM，也没有使用厂商签名产物，因而不能进一步分离各项安全策略的影响。

该模型包名标注目标为 sm8750，记录中识别为 HTP V79 代；第一台设备为 HTP V81。这个静态差异说明预编译 context 还存在架构兼容条件。执行在验证该条件前已经停止。本次测试没有直接观察到 V79/V81 不匹配引起的运行时错误，因此不能把它列为真机确认的另一个失败原因。

第二台设备是 HONOR MEP-AN00，系统库显示 HTP V79，且 `/odm` 中的 QNN 库可由 shell 读取。V79 代与模型包标注的目标代际更接近，但仅凭库名不能证明具体 SoC 与预编译 context 完全匹配。LiteRT-LM 先因宿主库版本门槛失败；换用 QAIRT 2.46 宿主库后，流程进入 QNN manager 创建 backend/device 的阶段并在此停止，仍未创建可执行推理会话。`qnn-platform-validator` 使用原厂路径和 QAIRT 2.42 V79 组件时都报告相同的 unsigned-images 诊断。这个对照降低了“单一宿主 SDK 版本错配”作为唯一原因的可能性，并支持签名或 DSP 访问策略相关的解释。它不足以证明所有版本交互均已排除，也不足以把失败唯一归因于 OEM ROM。

子模型通过重复的 `TensorBuffer` 句柄衔接。prefill 路径对主模型 embeddings 输入缓冲调用 `Duplicate()`，并把所得句柄作为 embedder 输出；decode、verify、per-layer embedder 与 mask 路径采用同类连接。代码在这些连接点没有显式执行字节复制。`Duplicate()` 共享底层缓冲的句柄语义可以减少子图之间的显式搬运。本书没有测量实际驱动同步与设备内存行为，因此不能把衔接成本记为零。

第一台设备完成了 QNN context 创建，第二台设备在 backend/device 创建阶段停止；两者均未完成 prefill 或 decode。因此，本书没有 NPU 吞吐、时延或功耗数据。NPU 执行行为仍以冻结版源码分析为主，真机记录只用于界定部署链路和已观察到的失败阶段。

## 8.7　后端差异为何可能改变输出

`LiteRT-LM#2281` 报告了 CPU 与 GPU 后端输出不同的现象。[^ch08-issue-2281] 本节从冻结版代码路径和浮点计算常识解释可能机制。issue 中的具体样例没有在本书设备上复现，因此本节不判断该报告的具体根因。

不同后端可能使用不同的算子内核、累加顺序、激活精度与反量化实现。浮点运算不满足结合律，执行顺序和中间精度变化都可能使 logits 出现细小差异。冻结版代码只能确认执行路径不同；具体差异来自哪一个算子，需要逐层对齐中间张量才能确定。

当两个候选 token 的 logits 接近时，微小数值差异可能改变 argmax 或采样结果。某一步选择不同 token 后，两次运行的后续上下文和计算输入也随之不同。不同硬件路径不保证逐比特一致，但输出差异也不能自动视为正常。排查时应固定模型、prompt、后端配置、采样参数、随机种子和线程设置，再比较 logits 与中间张量。按第 5 章核验的 `TopPSampler` 实现，greedy 由 `k == 1` 决定；单独把温度设为 0 不等于 greedy。若排除随机采样后差异仍在，还需检查数值容差、算子实现和确定性设置。

### 8.7.1　从首个分叉 step 开始诊断

只比较两段最终文本会放大自回归效应，却不能定位起点。诊断应记录每个 step 的输入 token、输出 token 和候选 logits，先找到首个 token id 不同的位置。该位置以前的上下文完全一致，才适合比较两条后端路径。

后端 A/B 还要显式固定 sampler backend。`SessionConfig` 未指定 sampler 时，主后端为 GPU 就选择 GPU sampler；其他主后端选择 CPU sampler。因此，单独切换 `--backend=cpu/gpu` 会同时改变主模型与 sampler，不能把差异只归因于矩阵 kernel。诊断配置应显式指定 sampler，并关闭 MTP、约束解码与重复惩罚等会修改 logits 或执行步数的功能。

可以按以下顺序缩小范围：

1. 校验模型文件、tokenizer 与 prompt token id 完全一致。记录 activation type、线程数、delegate 缓存状态和全部 sampler 参数。
2. 使用 `k=1`，在两个独立 session 中运行到首个分叉 step。不要让一条路径选出的 token 继续作为另一条路径的输入。
3. 调用不采样的 `DecodeLogits`，把两份 logits 转换为同一种 host 类型后比较。接口在 `LlmExecutorBase` 中定义，输出形状为 `[batch, sequence, vocab]`；具体 compiled executor 返回当前输出 buffer 的句柄。
4. 同时计算最大绝对误差、相对误差与 top-2 margin。若误差远小于 margin，top-1 理应稳定；若二者同量级，token 选择可能翻转。再按算子或子图边界逐层比较，定位首个超出容差的位置。

给一个不代表实测结果的数值例子。CPU 路径中，候选 A、B 的 logits 分别为 12.002 和 12.000，top-2 margin 为 0.002。GPU 路径若得到 11.998 和 12.001，B 会成为 top-1。四个值的最大跨后端差为 0.004，大于 CPU 路径的 margin。最终文本可能从这一 token 起完全不同。诊断结论只能落在“首个分叉处的数值扰动足以改变 argmax”，不能据此判定整条 GPU 路径错误。

冻结版提供了三个有针对性的复核开关。GPU 激活可在 FP16 与 FP32 间切换；`allow_src_quantized_fc_conv_ops` 控制部分 GPU 是否采用量化 FC/Conv，源码明确提示它可能以质量风险换取性能；`hint_waiting_for_completion` 要求 OpenCL invoke 后等待队列完成，注释说明它用于规避 AMD 和 Mali 上的已知质量问题。三项应分别切换。若同时改变，无法判断是精度、量化 kernel 还是同步改变了结果。

`num_logits_to_print_after_decode` 只能打印 logits 开头、中间和末尾的若干值。top-2 token 不一定落在这些区间内。完整诊断更适合由测试 harness 消费 `DecodeLogits`，把全量 logits 保存为二进制或计算摘要；开启日志时还要排除上一节说明的回读开销。

端侧实验记录必须包含后端及其配置。附录 D 因此把不同后端作为不同实验条件，不合并比较。

## 小结

| 后端 | 底层栈 | 实现特点 | 约束 | 本书证据 |
|---|---|---|---|---|
| CPU | LiteRT CPU / XNNPACK | 算子覆盖较宽，线程数可配置 | Pixel 亲和性依赖硬编码 SoC 表；4 线程只是默认值，需按设备扫描 | decode ≈ 24.7、prefill ≈ 259 tokens/s |
| GPU | LiteRT GPU delegate | 并行执行；设备侧采样避免完整 logits 回传 | token id 仍返回 host；缓冲与 delegate 行为依平台而异 | decode ≈ 50.6、prefill ≈ 999 tokens/s |
| NPU | LiteRT NPU；本书真机为 QNN（HTP） | 专用多子图执行路径 | 模型包、加速器代际、运行时组件与签名/访问条件需匹配 | 两台 Qualcomm 设备到达初始化失败阶段；未完成推理，无吞吐与功耗数据 |

> 表 8-2　三类后端的实现边界与证据范围。CPU/GPU 数据来自 Apple M5 Pro、Gemma 4 E4B、context 1024、生成 128 个 token 的基准〔基准 D〕；NPU 一列只记录两台 Qualcomm 设备的加载与失败阶段。

LiteRT-LM 通过 `Backend` 枚举和工厂函数选择 CPU/GPU 通用执行器或 NPU 专用执行器，并统一返回 `LlmExecutor`。CPU 线程数需要按设备扫描，Pixel 亲和性只覆盖硬编码 SoC。GPU 设备侧采样避免完整 logits 回传，但仍返回 token id。`Duplicate()` 只能证明 LiteRT-LM 这一层没有显式复制，端到端零拷贝还取决于 storage、布局与同步事件。NPU 代码采用 CPU embedder、NPU/CPU 主模型选项和额外辅助子图，本书尚无成功推理和能效数据。后端还会改变激活精度、sampler 和数值计算路径，因此性能和输出复现记录都必须注明完整配置。

---

## 练习与自查

1. 采样数据路径。设备侧采样避免回传完整 logits，但 token id 仍需返回 host。分别说明这两条数据路径承担的职责。
2. 核心绑定语义。`sched_setaffinity` 成功后限制的是什么？为什么不能把它描述成调度器的软性建议？
3. 线程扫描解读。线程数从 4 增至 8 时吞吐仍提高，为什么这组数据仍不能证明 8 线程未饱和或已经达到全局最优？
4. 输出分叉排查。更换后端后，同一 prompt 的输出为什么可能不同？哪些控制变量应先固定，才能判断差异是否来自实现缺陷？
5. 新后端接入清单。为运行时接入新后端 XPU，需要修改哪些执行器、模型资源、配置与工厂分支？列出至少三处，并说明哪些上层接口可以保持不变。
6. 证据边界判断。一处代码用 `Duplicate()` 连接两个子图。还需要收集哪些证据，才能把这条路径称为端到端零拷贝？
7. 采样开销折算。GPU 主模型搭配 CPU/GPU sampler 的两组运行相差 128 ms，共生成 256 个 token。计算每步差值，并解释为什么它不能直接当成互连带宽。

[^ch08-issue-2281]: 4ntoine，[*Different inference result depending on backend*](https://github.com/google-ai-edge/LiteRT-LM/issues/2281)，LiteRT-LM issue #2281，2026-05-15；访问日期：2026-07-18。

[^ch08-mlc-compile]: MLC-LLM，[*Compile Model Libraries*](https://llm.mlc.ai/docs/compilation/compile_models.html)，MLC-LLM 0.1.0 文档；访问日期：2026-07-18。

[^ch08-issue-2505]: Yegorsh，[*Add an option to use custom number of CPU threads in the CLI app*](https://github.com/google-ai-edge/LiteRT-LM/issues/2505)，LiteRT-LM issue #2505，2026-06-08；访问日期：2026-07-18。
[^ch08-llamacpp-registry]: ggml-org，[*llama.cpp 源码 ggml/src/ggml-backend-reg.cpp:117-165*](https://github.com/ggml-org/llama.cpp/blob/b9873/ggml/src/ggml-backend-reg.cpp#L117-L165)，版本 b9873；访问日期：2026-08-31。
