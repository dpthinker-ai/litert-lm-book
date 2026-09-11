# 第 5 章 Decode：单步解码循环

> 本章说明一个 decode step 如何从 logits（模型每步为词表中每个 token 打出的原始分数）得到 token id，再形成可发送的文本，并核对取消、停止与回调的边界。

prefill 已经把提示词写进 KV cache，第一次前向传播也完成了。接下来是 decode 阶段：模型逐 token 生成，直到满足停止条件。第 1 章在明确题设（4B 理想 INT4 权重、50 GB/s 有效带宽）下估算的 25 tokens/s 是 decode 阶段的带宽侧上限。对于 batch=1 的稠密模型，decode 通常受权重访存约束；实际瓶颈仍取决于后端实现和运行条件。本章分步说明一个 decode step：logits 怎么变成 token（5.2、5.3 节）、什么时候停（5.4 节）、文本什么时候才能发送（5.5 节）。

decode 阶段的单步操作在代码里叫 `DecodeOneStep`。`Decode` 在循环中反复调用它，直到满足停止条件。循环的主体如下：

```cpp
// runtime/core/tasks.cc:486
while (true) {
  if (cancelled != nullptr && cancelled->load()) {          // (1)
    // ...
    return absl::CancelledError("Process cancelled.");
  }
  // ...
  absl::StatusOr<bool> all_done =
      run_one_step.Run(std::move(decoded_ids_to_use));      // (2)
  // ...
  ASSIGN_OR_RETURN(int current_step, executor.GetCurrentStep());
  int num_decode_steps = current_step - executor_step_before_decode;
  if (ShouldStop(*all_done, benchmark_decode_token_count, num_decode_steps,
                 current_step, max_num_tokens, max_output_tokens)) {  // (3)
    break;
  }
}
```

代码行 `(1)` 在新一轮开始时读取取消标志；`(2)` 完成前向、采样和文本解码；`(3)` 在本轮末尾检查停止序列与长度上限。`all_done` 表示所有输出候选都已命中停止序列。

取消检查不会中断正在执行的 `Run`：标志若在本轮计算期间置位，本轮结果仍会继续处理，循环最早在下一次调用 `Run` 前才观察到它。取消也不属于 `ShouldStop` 的判断条件。

## 5.1　单步解码的完整流程

一次 decode step 依次执行以下操作：

1. 把上一个 token 送入模型并执行一次前向，得到 logits。它是词表中每个 token 的未归一化分数，维度等于词表规模。
2. 可选地处理 logits：降低近期已出现 token 的分数（重复惩罚），或屏蔽不符合语法约束的 token（约束解码，第 10 章）。
3. 从 logits 中采样一个 token id。
4. 把该 id 解码成文本片段，累积到结果或传递给流式回调。
5. 判断是否满足停止条件；若不满足，则把新 token 作为下一步的输入。

第 1 步通常占单步开销的主要部分，但其余四步决定输出的内容、停止时机和发送时机：采样策略限定候选分布（5.3 节），停止判定要处理多 token 的停止序列（5.4 节），文本解码还得暂存尚不能显示的 token（5.5 节）。

<figure>
{{#include figs/fig-5-1.svg}}
<figcaption>图 5-1　取消在迭代开始时检查；内部与外部采样汇合后，代码先处理文本与流式回调，再执行 ShouldStop。</figcaption>
</figure>

## 5.2　两条路径：内部采样与外部采样

采样有两条路径，区别在于由谁执行采样；`DecodeOneStep` 的注释明确列出了这两种情形。

内部采样路径调用执行器的 `Decode()`，向 `Tasks` 层直接返回 token id，不暴露 logits。采样具体在 CPU、GPU 还是其他后端完成，由具体执行器决定。若采样与 logits 都留在加速设备上，可称为设备侧采样（device-side sampling）；第 8 章分析其实现条件。仅从内部路径这一接口形态，不能直接推断它一定更快。

外部采样路径调用 `DecodeLogits`，再把 logits 传递给显式传入的 `Sampler`。这个采样器既可以是 CPU 实现，也可以是 GPU 实现。“外部”描述的是采样器由执行器外部调用，不等于 logits 必然从 GPU 回传 CPU。只有选择 CPU 采样器且 logits 不在宿主可直接访问的内存中时，才需要下载数据。

选择路径的依据只有一个：构造时有没有传入采样器。约束解码在两条路径上都可用，只是接入点不同——外部路径在采样前调用 `MaskLogits`，内部路径把约束解码器打包进参数交给执行器；两条路径的具体能力仍受采样器和执行器实现约束。代码如下：

```cpp
// runtime/core/tasks.cc:330-378
absl::StatusOr<std::vector<std::vector<int>>> DecodeAndSample(
    std::optional<litert::TensorBuffer> decoded_ids) {
  if (sampler_) {  // External sampling path            // (1)
    // ...
    ASSIGN_OR_RETURN(auto output_logits, executor_.DecodeLogits(inputs));  // (2)
    if (constrained_decoder_) {
      RETURN_IF_ERROR(constrained_decoder_->MaskLogits(output_logits));    // (3)
    }
    RETURN_IF_ERROR(sampler_.value()->SampleToIdAndScoreBuffer(
        output_logits, decoded_ids.value(), &scores_tensor_));             // (4)
    ASSIGN_OR_RETURN(auto token_ids,
                     tokenizer_.TensorBufferToTokenIds(decoded_ids.value()));
    return token_ids;
  } else {  // Internal sampling path                    // (5)
    // ...
    ASSIGN_OR_RETURN(output_tokens, executor_.Decode());                   // (6)
    return output_tokens;
  }
}
```

构造 `DecodeOneStep` 时是否传入采样器，决定了进入 `(1)` 标记的哪条分支。外部路径由 `(2)` 取得 logits，`(3)` 可按约束修改 logits，`(4)` 再调用采样器并填写可选的得分张量。内部路径由 `(6)` 调用执行器的组合接口。两条路径都返回 token id；代码只保证内部路径不把 logits 暴露给 `Tasks` 层，并未保证 logits 一定驻留在 GPU。

两条路径的返回类型都是 `std::vector<std::vector<int>>`：外层对应输出候选，内层是本次调用为该候选生成的 token id。`Run` 随后用同一套停止检测和文本解码逻辑处理它们。

### 5.2.1　CPU 外部采样的数据传输量

外部路径的成本可以分段看，执行器阶段与采样器阶段各自有计时标记：

```cpp
// runtime/core/tasks.cc:338-360
if (benchmark_info_.has_value()) {
  RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));  // (1)
}
ASSIGN_OR_RETURN(auto output_logits, executor_.DecodeLogits(inputs));  // (2)
if (benchmark_info_.has_value()) {
  RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));  // (3)
}
// ...
if (benchmark_info_.has_value()) {
  RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("sampling"));         // (4)
}
RETURN_IF_ERROR(sampler_.value()->SampleToIdAndScoreBuffer(
    output_logits, decoded_ids.value(), &scores_tensor_));            // (5)
```

`TimeMarkDelta` 成对出现，用同一个标签标记一段耗时。`(1)``(3)` 标记 `executor_decode`，`(4)` 与后续同名调用标记 `sampling`。基准工具据此分别报告执行器与采样器阶段的耗时（见附录 D）。

`DecodeLogits` 返回的 `output_logits` 形状是 `[batch, seq, vocab]`。decode 阶段的常见单步形状中，`seq` 为 1；若 `batch` 也为 1，元素数就等于词表规模。本书基准模型的 decode signature 为 `[1, 1, 262144]`，输出类型为 float32（见附录 D），因此该模型一次完整 logits 传输的数据量为

$$ 262144 \times 4\ \text{B} = 1048576\ \text{B} = 1\ \text{MiB} $$

这 1 MiB 只在 CPU 采样器需要把设备数据复制到宿主时构成额外传输。`TopPSampler` 会先尝试直接取得宿主可访问的内存视图，失败后才调用 `CopyFromTensorBuffer` 复制；float16 输入则经 `Read` 后转换为 float32。

第 1 章的同一示例中，4B 模型的理想 INT4 权重约为 1.86 GiB；1 MiB 约为它的 0.5‰。这个比例只能比较字节量，不能换算成时延比例。权重读取、设备到宿主复制、同步和 CPU 采样经过不同的数据通路，也可能存在重叠。设备侧采样是否降低单步时延，必须在同一模型、后端和设备上对照测量。

## 5.3　从 logits 选择 token

采样策略规定如何从 logits 中选出 token。LiteRT-LM 的采样器实现同一个 `Sampler` 接口，其中的核心方法如下：

```cpp
// runtime/components/sampler.h:34
virtual absl::Status SampleToIdAndScoreBuffer(
    const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,   // (1)
    TensorBuffer* scores_tensor) = 0;                              // (2)
```

接口规定 `logits_tensor` 的形状为 `[batch_size, sequence_size, vocab_size]`，`ids_tensor` 为 `[batch_size, sequence_size]`。`scores_tensor` 可以为空；非空时，采样器写入所选 token 的对数概率。这些张量均以 `TensorBuffer` 传递，因此 CPU 与设备侧采样实现可以共享同一接口。

### 5.3.1　采样器工厂与后端降级

具体实现由 `CreateSampler` 按请求的后端分派。GPU 分支先调用 `CreateGpuSampler`；只有它返回 `kUnavailable` 时，工厂才转入 CPU 分支：

```cpp
// runtime/components/sampler_factory.cc:707-740
    case Backend::GPU: {
      // ...
      auto sampler_or =
          CreateGpuSampler(batch_size, sampler_params, env, sequence_size_value,
                           vocab_size.value(), activation_data_type);
      if (sampler_or.ok() ||
          sampler_or.status().code() != absl::StatusCode::kUnavailable) {
        // For a normal failure or success, return the result.
        return sampler_or;                                       // (1)
      }
      // For a failure due to GPU sampler unavailable, fall back to CPU.
      ABSL_LOG(WARNING)
          << "GPU sampler unavailable. Falling back to CPU sampling. To use "
             "GPU sampling, please make sure libLiteRtTopKWebGpuSampler.so or "
             "libLiteRtTopKOpenClSampler.so is available at LD_LIBRARY_PATH "
             "on device. You can find the shared library under prebuilt/";
      ABSL_FALLTHROUGH_INTENDED;                                 // (2)
    }
    case Backend::CPU:
      return CreateCpuSampler(batch_size, sequence_size_value, sampler_params);
```

代码行 `(1)` 直接返回成功结果或 `kUnavailable` 以外的错误；只有 `kUnavailable` 才经 `(2)` 转入 `CreateCpuSampler`。转入 CPU 后是否发生设备到宿主复制，仍取决于 logits 的内存可访问性，而不是由这个 `switch` 单独决定。

GPU 采样器能否使用，取决于平台、编译选项与 `LiteRtEnvironment`：`CreateGpuSampler` 按条件依次尝试 WebGPU、OpenCL 或 Metal，动态库按符号名加载，OpenCL 路径在动态加载失败后还会尝试静态链接入口。所以设备侧采样并不一定依赖独立的 `.so`，警告里提到的库名只是其中一种部署形态。

<div class="aside-compare">

llama.cpp 在 `common_sampler_init` 中按参数把 top-k、top-p、min-p、重复惩罚等处理器加入采样链。[^ch05-llamacpp-sampler] LiteRT-LM 的工厂选择一个 `Sampler` 实现，执行器还可以使用内部采样路径。两者的采样流程组合方式不同；这段初始化代码不能说明端到端时延差异。

</div>

CPU 采样实现是 `TopPSampler`，它把四种常见策略组合在一起：

- 贪心（greedy）：令 `k = 1`，直接选择分数最高的 token。
- 温度（temperature）：softmax 前用温度缩放 logits。较低温度通常使非并列分布更集中，较高温度使分布更平缓。
- top-k：只在分数最高的 k 个 token 里采样，截去低概率长尾。
- top-p（核采样）：按概率降序保留累计值达到 p 的最小前缀；本实现只在 top-k 候选内计算。

创建函数接收的参数正对应这几项：

```cpp
// runtime/components/top_p_cpu_sampler.h:30
static absl::StatusOr<std::unique_ptr<TopPSampler>> Create(int k, float p,  // (1)
                                                           float temperature,
                                                           int batch_size,
                                                           int sequence_size,
                                                           int seed);        // (2)
```

代码行 `(1)` 的 `Create` 接收 `k`、`p` 和 `temperature`。在该实现中，贪心对应 `k == 1`，直接选择全词表最大 logit；不是把 k 设为很大。`k > 1` 时，代码先取 top-k，再在其中应用温度和 top-p。

代码行 `(2)` 的 `seed` 用于构造 `generator_`。在实现、输入和其他参数相同的条件下，固定 seed 可复现同一随机数序列。

附录 D 第八节记录了本书基准环境中的输出：同一提示词在温度 0、相同 seed 和其余默认参数下，两次输出一致。这只证明该配置在同一环境内可复现（换了运行时版本就可能得到另一句，见 2.1 节），也不能单凭温度 0 把它定义为 greedy。

温度 1.0 时，更换 seed 可以得到不同输出；默认参数下也出现过不同 seed 产生相同序列的情况〔基准 D〕。随机采样允许相同结果，单次重复不能证明采样未生效。

### 5.3.2　CPU 采样实现：候选选择与数值边界

`TopPSampler` 把工作交给采样工具库里的三个函数：`TopKTokenIds` 取候选集，`Softmax` 归一化，`TopKTopPSampling` 完成 top-p 截断与随机选择。

`TopKTokenIds` 有两条路径：

```cpp
// runtime/components/sampling_cpu_util.cc:35-85
if (k == 1) {  // Greedy sampling. Use std::max_element to be more efficient.
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < sequence_size; ++s) {
      auto max_iterator = std::max_element(
          logits.begin() + (b * sequence_size + s) * vocab_size,
          logits.begin() + (b * sequence_size + s + 1) * vocab_size);   // (1)
      // ...
    }
  }
} else {
  // ...
  std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
                   desc_prob_comp);                                     // (2)
  std::copy(indices.begin(), indices.begin() + k,
            output_indices[b].begin() + s * k);
}
```

`k == 1` 时，`(1)` 用 `std::max_element` 线性扫描取 argmax，不做排序。`k > 1` 时，`(2)` 用 `std::nth_element` 把最高的 k 个索引移到前 k 个位置。其平均复杂度为 O(vocab)，且不保证候选内部有序。排序留到 top-p 阶段。

选出候选后，`Softmax` 把候选 logits 归一化成概率：

```cpp
// runtime/components/sampling_cpu_util.cc:88-168
float sum_of_exps = 0.0;
float current_temp =
    std::max(temperature, std::numeric_limits<float>::epsilon());        // (1)
for (size_t i = 0; i < k; ++i) {
  probabilities[b][s * k + i] =
      std::exp((logits[offset + topk_token_ids[topk_offset + i]] -
                max_logit_values[b][s]) /                                // (2)
               current_temp);
  sum_of_exps += probabilities[b][s * k + i];
}

if (sum_of_exps <= std::numeric_limits<float>::epsilon()) {              // (3)
  float uniform_prob = 1.0 / static_cast<float>(k);
  std::fill(probabilities[b].begin() + s * k,
            probabilities[b].begin() + (s + 1) * k, uniform_prob);
} else if (std::isinf(sum_of_exps)) {                                    // (4)
  std::fill(probabilities[b].begin() + s * k,
            probabilities[b].begin() + (s + 1) * k, 0.0f);
  probabilities[b][s * k + max_logit_idx] = 1.0f;
} else {
  // ...
}
```

计算指数前，`(2)` 先减去候选中的最大 logit，这是防止指数溢出的标准数值手法：对有限输入，指数自变量不大于 0，最大项为 \\(\exp(0) = 1\\)。`(1)` 把非负温度限制为不小于 `epsilon`，避免除以 0。温度为 0 时会使用这个最小正数，但接口并未因此把它定义为 greedy。greedy 分支由 `k == 1` 决定。

代码行 `(3)``(4)` 是防御性分支，不是有限 logits 下的正常执行路径。只要 k 为正且候选 logits 都是有限值，减最大值后至少有一项等于 1，其余项位于 `[0, 1]`。因此，`sum_of_exps` 位于 `[1, k]`：它不会小于 `epsilon`，也不会成为 `inf`。代码没有在这里显式拒绝 NaN 或无穷 logits，不能据此断言任意输入都会在通过这两个分支后得到合法概率分布。

归一化之后是 top-p 截断与采样：

```cpp
// runtime/components/sampling_cpu_util.cc:210-284
if (k == 1) {  // Greedy sampling. Return the topk_token_ids directly.
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < sequence_size; ++s) {
      sampled_ids[b][s] = (*topk_token_ids)[b][s];
      sampled_scores[b][s] = 1.0f;                                       // (1)
    }
  }
  return sampled_ids;
}
// ...
std::sort(index_of_topk.begin(), index_of_topk.end(), desc_prob_comp);   // (2)
// ...
double cumulative_prob = 0.0;
int final_sample_size = 0;
for (int i = 0; i < k; ++i) {
  // ...
  cumulative_prob += (*probabilities)[b][s * k + index_of_topk[i]];       // (3)
  final_sample_size = i + 1;
  if (cumulative_prob >= p) {
    break;                                                               // (4)
  }
}
```

`k == 1` 时，`(1)` 直接返回唯一候选，并在工具函数内把概率记为 `1.0f`。`TopPSampler` 随后写入其对数，即 0。该路径仍在前面调用了 `Softmax`，但跳过后续排序、累计和随机数生成。

`k > 1` 时，`(2)` 只对 k 个候选按概率降序排序，复杂度为 O(k log k)。`(3)``(4)` 按顺序累加，累计值达到 p 后停止，得到 top-k 范围内满足阈值的最小前缀。

代码随后在 `[0, cumulative_prob)` 上取均匀随机数，并按累计区间选择 token。整条路径包含 O(vocab) 的部分选择、O(k log k) 的候选排序和 O(k) 的累计过程。

## 5.4　停止条件

每个 decode step 结束后都要检查停止条件。判断集中在纯函数 `ShouldStop` 中：

```cpp
// runtime/core/tasks.cc:85-107
bool ShouldStop(bool hit_stop_tokens, int benchmark_decode_token_count,
                int num_decoded_steps, int current_step, int max_num_tokens,
                int max_output_tokens) {
  if (hit_stop_tokens && benchmark_decode_token_count == 0) {  // (1)
    // ...
    return true;
  } else if (benchmark_decode_token_count > 0 &&
             num_decoded_steps >= benchmark_decode_token_count) {  // (2)
    // ...
    return true;
  } else if (current_step >= max_num_tokens) {          // (3)
    // ...
    return true;
  } else if (num_decoded_steps >= max_output_tokens) {  // (4)
    // ...
    return true;
  }
  return false;
}
```

四个分支依次检查停止序列、基准步数、执行器的最大 token 数和输出长度。`(1)` 只在 `benchmark_decode_token_count == 0` 时接受停止序列；指定基准步数后，`(2)` 以固定步数结束。`(3)` 比较执行器当前位置与 `max_num_tokens`，其中当前位置包含 prefill 已占用的 token。`(4)` 只比较本次 decode 已执行的步数与 `max_output_tokens`。`num_decoded_steps` 由当前位置减去 decode 开始前的位置得到。

代码行 `(3)` 和 `(4)` 使用不同计数器，因此长提示词可能先触及执行器上限，短输出配置则可能先触发 `max_output_tokens`。取消不在 `ShouldStop` 中；循环只在下一次迭代开始时检查 `cancelled`。它不会抢占已经开始的前向或采样。

流式回调发生在 `ShouldStop` 之前：代码先收集本轮可发送的文本，至少一个候选产生非空文本时，才以 `TaskState::kProcessing` 调用回调。因此，未完整 BPE（byte-pair encoding）序列或停止序列的部分匹配可能让某个 decode step 不产生回调。

循环结束后，`DecodeStreaming` 再调用一次最终回调。其状态为 `kDone`、`kMaxNumTokensReached` 或错误状态。

## 5.5　未完整文本序列与停止序列暂存

流式输出要处理两类暂存：token id 序列尚不能解码成有效文本，以及若干 token id 可能构成停止序列的前缀。两者都发生在 `DecodeOneStep::Run` 中，但判断顺序不同。

运行时按 token id 序列执行停止匹配，不比较字符子串。若停止序列为 `[A, B, C]`，收到 A 或 `[A, B]` 时还不能发送对应文本。后续 token 仍可能使序列完整命中。

`StopTokenDetector` 为每个候选和每条停止序列记录匹配进度，`ProcessTokens` 每次接收各候选的一个新 token id。查询部分匹配长度的接口很简单：

```cpp
// runtime/components/stop_token_detector.cc:161
int StopTokenDetector::MaxPartialStopTokenLength(int index) const {
  return max_batch_item_match_progress_[index];
}
```

当前实现直接返回各停止序列匹配进度的最大值。`Run` 只在候选尚未完整命中时查询它：0 表示没有部分匹配，正数表示仍有相应数量的 token id 可能属于停止前缀。

头文件注释提到完整命中后返回 -1，但当前实现不会返回 -1。完整命中由前面的 `GetStopTokensFound()` 分支处理。

```cpp
// runtime/core/tasks.cc:183
} else if (!stop_token_detector_.GetStopTokensFound()[i]) {
  bpe_partial_token_ids_[i].clear();
  int max_length = stop_token_detector_.MaxPartialStopTokenLength(i);
  if (max_length > 0) {
    pending_stop_tokens_[i].push(decoded_result.value()[i].value()); // (1)
    pending_stop_token_ids_[i].push(step_tokens[i]);
    num_buffered_tokens_[i] += step_tokens[i].size();
  }
  while (num_buffered_tokens_[i] > max_length) {                      // (2)
    result_text_[i] += pending_stop_tokens_[i].front();              // (3)
    pending_stop_tokens_[i].pop();
    // ...
  }
  if (max_length == 0) {
    result_text_[i] += decoded_result.value()[i].value();            // (4)
    // ...
  }
}
```

代码行 `(1)` 把已成功解码、但仍可能属于停止前缀的文本块和 token id 向量分别入队。`num_buffered_tokens_` 统计 token id 数，不是队列项数。`(2)``(3)` 在计数超过当前最大部分匹配长度时，从队首释放完整文本块。

一个文本块可能由多个此前暂存的 BPE token id 合并而成。因此，队列不能简单等同于“恰好保存最后 `max_length` 个独立 token”。`max_length == 0` 时，代码先释放已有队列，再由 `(4)` 加入当前文本。

完整命中时，`GetStopTokensFound()[i]` 为真，上述 `else if` 被跳过。触发命中的当前 token 不会进入 `pending_stop_tokens_`，此前暂存的前缀也不会加入 `result_text_`。

`AllDone()` 只有在所有输出候选都命中后才返回真。图 5-2 采用单候选且每个 token 都能独立解码的情形。

<figure>
{{#include figs/fig-5-2.svg}}
<figcaption>图 5-2　单候选下停止序列的暂存时序：失配时释放前缀并发送当前文本，完整命中时不发送已暂存前缀和触发命中的 token。</figcaption>
</figure>

另一类暂存针对尚不能解码的 token id 序列。HuggingFace 和 SentencePiece 实现会检查解码结果是否以 Unicode 替换字符 U+FFFD 结尾。若是，则返回 `DataLossError`，表示还需要后续 token。`Run` 用 `bpe_partial_token_ids_` 保存这些 id：

```cpp
// runtime/core/tasks.cc:175-184
ASSIGN_OR_RETURN(step_tokens, tokenizer_.MergeTokenIds(              // (1)
                                  bpe_partial_token_ids_, step_tokens));
auto decoded_result =
    tokenizer_.TokenIdsToTexts(num_output_candidates_, step_tokens);
for (int i = 0; i < num_output_candidates_; ++i) {
  if (Tokenizer::IsIncompleteBpeSequence(decoded_result.value()[i])) {  // (2)
    bpe_partial_token_ids_[i] = step_tokens[i];                         // (3)
  } else if (!stop_token_detector_.GetStopTokensFound()[i]) {
    bpe_partial_token_ids_[i].clear();                                  // (4)
    // ...
```

代码行 `(1)` 先把此前暂存的 id 与本轮 id 合并，再整体解码。`IsIncompleteBpeSequence` 只检查返回状态是否为 `kDataLoss`。若仍不完整，`(3)` 保存合并后的 id，本轮不产生文本；解码成功后，`(4)` 清空 BPE 暂存，再进入停止序列的暂存逻辑。

停止检测先于 BPE 合并执行，按本轮原始 token id 推进：`ProcessTokens` 先读取原始 id，随后 tokenizer 才尝试合并和解码。只有成功解码且尚未完整命中停止序列的文本，才可能进入 `pending_stop_tokens_`。

模型元数据中的停止字符串会先尝试映射成单个 token id。失败时再调用 `TextToTokenIds` 得到一段 id。会话配置以 `std::vector<std::vector<int>>` 保存停止序列。某个停止字符串最终对应一个还是多个 id，取决于模型 tokenizer，不能从聊天模板文本直接推断。

BPE 暂存等待 tokenizer 成功解码；停止序列暂存避免提前发送仍可能属于停止前缀的文本。回调只发送经过这两层处理后留下的非空结果。

### 5.5.1　文本片段间隔不等于逐 token 时延

附录 D 第十四节的手机验证组使用 HONOR MEP-AN00、Gemma 4 E4B 和 GPU OpenCL，固定输入为 77 token，上下文上限为 4096，关闭 MTP。3 次请求中，每次 decode 计数都是 201 token，非空文本回调却只有 200 次，这已经足以说明统计回调次数不能替代运行时的 token 计数。

客户端记录的是相邻两次文本回调之间的间隔。一次回调可能对应一个 token，也可能是暂存后合成的一段文本，停止序列还可能不作为正文发送；仅凭这个英文样例的数量差，不能断定发生了 BPE 暂存。要测逐 token 时延（ITL），必须另外取得 token 生成事件及其与回调的对应关系。

面向阅读体验时，文本片段间隔仍有用途，它能显示客户端是否长时间没有收到新文本；报告应同时保存 token 数、回调片段数、每片段原文与时间戳，两种口径才能分开核对。附录中的数字只描述客户端相邻两次收到文本的间隔，不代表每个 token 的执行耗时，也未计入应用显示这些文本所需的时间。

## 小结

一个 decode step 依次完成前向、可选的 logits 处理、采样、文本解码和停止判断。内部与外部采样描述控制边界，设备侧与 CPU 采样描述执行位置，两组概念不能混用。`ShouldStop` 处理停止序列和长度上限，取消由下一轮循环开头单独检查。流式文本还要经过 BPE 与停止序列两层暂存；没有可发送文本的步骤不会触发 `kProcessing` 回调。

第 6 章将分析 KV cache 的容量与带宽开销。

---

## 练习与自查

1. 传输量变体。若 logits 以 FP16 传输（词表仍为 262144），单步数据量是多少？它占 1.86 GiB 权重数据的比例是多少？为什么这个比例不能直接换算为时延比例？
2. 停止条件推理。benchmark 模式指定 decode 128 步，第 20 步命中停止序列，循环会停吗？依据 `ShouldStop` 的哪个分支？
3. 暂存时序。停止序列最长 3 个 token。模型依次产出 A、B、C，其中 A、B 是某停止序列的前缀而 C 使匹配失败。写出每一步用户实际收到的文本。
4. BPE 暂存。为什么流式解码要把此前暂存的 token id 与新 id 合并后整体转换？
5. 动手实验。用 5 个不同 seed 跑温度 1.0（命令见附录 D 第八节），统计得到几种不同输出，并解释为什么启用随机采样仍可能得到相同输出。

[^ch05-llamacpp-sampler]: ggml-org，[*llama.cpp 源码 common/sampling.cpp:334-358*](https://github.com/ggml-org/llama.cpp/blob/b9873/common/sampling.cpp#L334-L358)，版本 b9873；访问日期：2026-08-31。
