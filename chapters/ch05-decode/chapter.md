# 第 5 章 Decode：单步解码循环

> 本章走通一次 decode step 的全部环节：从模型前向算出 logits，到一个可显示的文本单元流式发给用户。这是理解后续性能优化章节的前提。

prefill 已经把提示词写进 KV cache，第一段前向也完成了。接下来是 decode 阶段：模型逐 token 生成，直到满足停止条件。第 1 章讲过的那条 25 tokens/s 上限，约束的正是这个阶段的每一步——decode 是受内存带宽约束（memory-bound）的逐 token 串行生成，每生成一个 token 都要把全部权重从内存读一遍。

decode 阶段的单步操作在代码里叫 `DecodeOneStep`（`runtime/core/tasks.cc:111`）。整个 decode 就是一个循环，反复调用它，直到满足停止条件（`Decode`，`tasks.cc:446`）。循环的主干只有几行，先看骨架：

```cpp
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

主循环的结构就在这三个编号上。(2) 是这一步的实质计算：一次前向、一次采样、一次文本解码，都压在 `run_one_step.Run` 里，返回值 `all_done` 表示停止 token 是否已全部命中。(3) 是每轮末尾的停止判定：`ShouldStop` 汇总 `all_done` 和几个计数器，决定 `break` 还是继续下一轮。(1) 是每轮开头对取消标志的探测。第 4 章那个 `std::atomic<bool>* cancelled` 被放在循环最前面检查，一旦置位就立即退出，不等这一步算完。三个动作构成一次 decode step：探测取消、执行一步、判定停止。下面逐层拆开这一步。

## 单步解码的完整流程

一次 decode step，从上一个 token 到下一个 token，要走这么几步：

1. 把上一个 token 送入模型，前向一次，输出一组 **logits**——词表里每个 token 的未归一化分数，维度等于词表规模，通常几万到数十万维；
2. （可选）对 logits 做处理：压低近期已出现 token 的分数（重复惩罚），或把不合语法的 token 屏蔽掉（约束解码，第 10 章）；
3. 从 logits 里**采样**出一个 token id；
4. 把这个 id 解码回文本片段，累积或流式发给用户；
5. 判定是否满足停止条件；不满足则把新 token 当作上一个 token，回到第 1 步。

第 1 步是受内存带宽约束的前向：每一步都要把全部权重读一遍（第 1 章）。其余四步是本章的重点。它们看似简单，但每一步都有一个需要处理的边界情形：采样要抑制低概率长尾 token，停止判定要避免误判，文本解码要避免输出半个字符。LiteRT-LM 对每一处都有明确处理。

<figure>
{{#include figs/fig-5-1.svg}}
<figcaption>图 5-1　单步解码循环的完整流程。核心分岔是内部采样与外部采样两条路径：前者由执行器一步返回 token，后者把 logits 交回上层做可定制的处理与采样。</figcaption>
</figure>

## 两条路径：内部采样与外部采样

采样在这里分成两条路径。代码注释写明这是为内部采样和外部采样两种情形准备的（`tasks.cc:110`）。

内部采样由执行器一步完成。上层调用 `Decode`，执行器内部把前向和采样都做完，直接返回 token id。这条路径更快，因为采样可以在 GPU 上就地完成，省掉把整组 logits 从 GPU 显存搬回 CPU 内存的开销。片上采样的具体机制第 8 章展开，本章后面会算清这笔搬运开销到底占单步耗时多少。

外部采样时执行器只做到前向，把整组 logits 返回（`DecodeLogits`，`tasks.cc:342`），由上层的 logits 处理器和采样器完成处理与采样。这条路径要多付一次 logits 回搬的代价，换来的是灵活性：重复惩罚、约束解码这些需要修改 logits 的功能只能在这条路径上实现。

两条路径服务不同需求。纯文本生成、不需要修改 logits 时走内部采样以求速度；需要工具调用、结构化输出、约束解码时走外部采样以求灵活。`DecodeOneStep` 里的 `DecodeAndSample`（`tasks.cc:319`）是这个分岔的落点，靠一个成员指针分道：

```cpp
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

分岔就是 (1) 那个 `if (sampler_)`：构造 `DecodeOneStep` 时是否传入采样器，决定走哪条路径。外部路径 (2) 调 `executor_.DecodeLogits`，只做前向，把整组 logits 返回；(3) 是约束解码的挂钩点，`MaskLogits` 在采样前把不合语法的 token 分数置为负无穷（第 10 章）；(4) 把处理过的 logits 交给外部 `Sampler` 采样，并写回 `scores_tensor_`（采样得分，供候选排序使用）。内部路径 (5) 只有一行实质代码：(6) 的 `executor_.Decode()` 把前向和采样一次做完，logits 不离开 GPU，直接返回 token id。

两条路径返回的都是同一种 `std::vector<std::vector<int>>`（外层是候选批，内层是这一步各候选的 token）。从 `DecodeAndSample` 返回后，上层代码不再关心它来自哪条路径。分岔收敛在这一个函数内，出口统一，其余代码不被两条路径的差异污染。

### 外部采样的 logits 回搬开销

前面说外部路径要多付一次 logits 回搬的代价。这笔账可以算清楚。`DecodeAndSample` 的外部路径把前向与采样分成两段计时（`tasks.cc:339-360`）：

```cpp
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

`TimeMarkDelta` 成对出现，用同一个标签划一段耗时：(1)(3) 把 `DecodeLogits` 前向框成 `executor_decode` 段，(4) 与其后配对的 `TimeMarkDelta("sampling")` 把 (5) 的采样框成 `sampling` 段。基准工具据此分别报告前向与采样各自的耗时（基准指标的采集见附录 D）。

`DecodeLogits` 返回的 `output_logits` 形状是 `[batch, seq, vocab]`。decode 阶段 `seq` 为 1，若 `batch` 也为 1，单步返回的就是一个长度等于词表规模的向量。本书基准模型实剖出的词表规模是 262144（decode signature 的 logits 形状 `[1, 1, 262144]`，附录 D），logits 以 float32 存储，单步回搬的字节量是：

262144 × 4 B = 1048576 B = 1 MiB

这是外部采样相对内部采样多付的数据搬运量，每一步都要付一次。它是否显著，取决于它和前向本身的耗时之比。第 1 章给出的 memory-bound 前向要读一遍全部权重：以一个 4 bit 量化、约 30 亿参数的模型为例，单步前向至少要读约 30 亿 × 0.5 B ≈ 1.4 GiB 权重。把 1 MiB 的 logits 回搬放在 1.4 GiB 的权重读取旁边，前者约为后者的 0.7‰。据此推断，在此类模型上外部采样的 logits 回搬相对前向本身可以忽略；真正让外部路径变慢的，是采样在 CPU 上执行（下一节的排序与累积），而非数据搬运本身。词表规模越大、模型越小、后端显存带宽越低，这笔搬运占比越高，具体数值须以附录 D 在目标设备上的实测为准。

## 采样：从 logits 向量里选出一个 token

不管走哪条路径，都要从 logits 里选出一个 token。怎么选，就是采样策略。LiteRT-LM 的采样器都实现同一个 `Sampler` 抽象（`runtime/components/sampler.h:34`）。这个抽象只强制一个核心方法：

```cpp
virtual absl::Status SampleToIdAndScoreBuffer(
    const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,   // (1)
    TensorBuffer* scores_tensor) = 0;                              // (2)
```

签名里有两个设计取向。(1) 的 `logits_tensor` 形状是 `[batch_size, sequence_size, vocab_size]`，`ids_tensor` 是 `[batch_size, sequence_size]`：采样器一次处理一整批候选，而非逐个处理，因为 `num_output_candidates > 1` 时（束搜索、多候选生成）批处理能摊薄固定开销。(2) 的 `scores_tensor` 可空：传入则把采样到的 token 的概率也写回，供上层做候选排序；不需要则传 `nullptr`。方法直接读写 `TensorBuffer` 而非 `std::vector`，是为了让实现能落在 GPU 上。这个抽象既能包 CPU 采样器，也能包片上采样器（第 8 章）。

### 采样器从哪来：工厂分派与一条降级链

具体用哪个采样器实现，由工厂函数按后端分派（`CreateSampler`，`runtime/components/sampler_factory.cc:707`）。GPU 分支里藏着一条设计得很完整的降级链：

```cpp
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

日志文本交代了背景：GPU 片上采样器不是编进主库的，而是独立的动态库（WebGPU 或 OpenCL 两种实现），运行时按符号名动态加载（`GetSamplerCApi`，定义 `:283`、OpenCL 调用点 `:360`，加载 `libLiteRtTopKOpenClSampler.so` 并解析 `Create`/`Destroy`/`SampleToIdAndScoreBuffer` 等 C 符号）。设备上没有这个 `.so` 时，加载失败返回 `kUnavailable`。(1) 处的判断把错误分成两类：真正的失败（参数错、初始化错）原样上抛；仅仅是「不可用」则 (2) 用显式标注的 `ABSL_FALLTHROUGH_INTENDED` 落进 CPU 分支，换 CPU 采样器继续跑。功能不受影响，代价是每步 decode 多一次 logits 回搬（上一节刚算过这笔账）。把可选的加速件做成独立动态库加运行时探测，主库不背 GPU 采样的依赖，没有它照样正确，这与第 8 章 CPU 亲和性只在特定设备上生效是同一种工程姿态：加速是机会性的，正确性是无条件的。

<div class="aside-compare">

llama.cpp 的采样只有一条路径：全部在宿主侧执行，做法是把采样器组装成一条链，按参数依次挂上 top-k、top-p、min-p、重复惩罚等环节（`llama.cpp/common/sampling.cpp:334`–`:358 @ b9873`），logits 每步都回到 CPU。这相当于只保留本章的「外部采样」路径：换来任意组合、任意顺序的灵活性，代价正是本章算过的那笔每步回搬。LiteRT-LM 的两条路径与片上采样，多出的是「纯聊天场景省掉回搬」的选项，少掉的是采样环节自由堆叠的空间。两家的取舍映射的是目标场景：一个偏研究与服务器部署的多样采样需求，一个偏端侧固定负载的每一毫秒。

</div>

常用的一种实现是 `TopPSampler`（`runtime/components/top_p_cpu_sampler.h:30`）。它用一个类覆盖多种策略，靠参数区分：

```cpp
static absl::StatusOr<std::unique_ptr<TopPSampler>> Create(int k, float p,  // (1)
                                                           float temperature,
                                                           int batch_size,
                                                           int sequence_size,
                                                           int seed);        // (2)
```

(1) 一个 `Create` 收 `k`、`p`、`temperature` 三个采样参数。四种策略是它们的组合，而非四个类：k 取极大、temperature 取 0，得到贪心；给定 k 和 p，得到 top-k 叠 top-p；temperature 单独调节随机度。(2) 的 `seed` 落到成员 `generator_`（一个 `std::default_random_engine`），采样的随机性全部来自这一个种子。固定种子即可复现整段输出，这是调试和基准测量能对齐的前提。四种策略如下：

- **贪心（greedy）**：直接选分数最高的 token。确定、可复现，但输出容易单调、重复。
- **温度（temperature）**：采样前用温度缩放 logits，使 softmax 分布更均匀或更尖锐。温度低趋于贪心，温度高更随机、输出更发散。
- **top-k**：只在分数最高的 k 个 token 里采样，截去低概率长尾。
- **top-p（核采样）**：只在累计概率达到 p 的最小候选集里采样，是 top-k 的自适应版本——概率集中时候选少，分散时候选多。

这些是可复现、可观测的行为，本书在基准机上实测过（附录 D 第八节）。同一个提示词，温度 0、同种子跑两次，输出逐字一致：贪心路径确定可复现。温度 1.0 时输出随种子可变：换一个种子得到了另一个句子。但实验里还出现了第三种情况：默认采样参数下，两个不同的种子产出了逐字相同的序列。这不是 bug。分布尖锐时（top-k/top-p 截断后高概率 token 一家独大），采样在多数步上都会命中同一个 token，**开采样不等于每次必不同**。判断「采样是否生效」要换种子多跑几次，而不是跑两次看见相同就下结论。

| 策略 | 选取方式 | 特点 |
|---|---|---|
| greedy | 取最高分 | 确定、可复现、易重复 |
| temperature | 先缩放再采样 | 调节随机性 |
| top-k | 前 k 个内采样 | 截去低概率长尾 |
| top-p | 累计概率 p 内采样 | 自适应候选集 |

> 表 5-1　四种采样策略。它们不互斥，实际常组合使用（如 top-p 叠加温度）。

### CPU 采样的真实实现：数值稳定与退化分支

表 5-1 概括了四种策略，但真正实现它们的是 `sampling_cpu_util.cc`。这里把 `TopPSampler` 背后的三个函数走读一遍，因为它们直接关系到正确性与性能：`TopKTokenIds` 取候选集，`Softmax` 归一化，`TopKTopPSampling` 做 top-p 截断与采样。

先看候选集选取。`TopKTokenIds` 有两条路径（`sampling_cpu_util.cc:47`）：

```cpp
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

`k == 1` 是贪心的专门快路径。(1) 用 `std::max_element` 一趟线性扫描直接取 argmax，O(vocab) 时间、不排序。贪心不必走完整条 top-p 流水线，这里被特判掉了。`k > 1` 时 (2) 用 `std::nth_element` 做部分选择：它把最高的 k 个索引挪到前 k 个位置，平均 O(vocab) 时间，但不保证这 k 个内部有序。相比对整个词表排序（O(vocab log vocab)），部分选择省下大头，代价是候选内部的排序留给后面按需处理。

选出候选后，`Softmax` 把候选的 logits 归一化成概率。这一步的数值稳定处理值得细看（`sampling_cpu_util.cc:135`）：

```cpp
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
}
```

三处细节。(2) 指数计算前先减去最大 logit `max_logit_values[b][s]`，这是数值稳定 softmax 的标准做法：直接对原始 logit 取指数会溢出（logit 可能到几十上百），减去最大值后指数的自变量非正，`exp` 落在 (0, 1]，不溢出，且归一化后结果与减法前完全等价。(1) 把温度用 `std::max(temperature, epsilon)` 钳到至少一个最小正数，避免除零：温度设 0 时，除以 `epsilon` 相当于把分布拉到极尖，退化为贪心。

(3)(4) 是两个退化分支。(3) 当所有指数之和小到接近 0（浮点下溢），改为均匀分布兜底，避免后续除以 0。(4) 当和为 `inf`（极小温度让某个指数溢出到无穷），把概率质量全压到最大 logit 对应的 token，也就是退化为贪心。这两个分支处理的是温度趋近 0 或数值极端时的边界，保证 softmax 永远返回一个合法的概率分布，而非 NaN。

归一化之后是 top-p 截断与采样（`sampling_cpu_util.cc:213`）：

```cpp
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
double cumulative_prob = 0.0;
int final_sample_size = 0;
for (int i = 0; i < k; ++i) {
  cumulative_prob += (*probabilities)[b][s * k + index_of_topk[i]];       // (3)
  final_sample_size = i + 1;
  if (cumulative_prob >= p) {
    break;                                                               // (4)
  }
}
```

这里又有一处 `k == 1` 快路径 (1)：贪心直接返回候选，并把得分记为 `1.0f`（确定性选择，概率视为 1）。这是本函数里第二处对贪心的特判——`TopKTokenIds` 的 (1) 特判了候选选取，这里特判了采样。贪心在实现上是两处独立的快路径，不走完整的排序与累积流程。

`k > 1` 时 (2) 只对这 k 个候选按概率降序排序，O(k log k)，而非对整个词表排。(3)(4) 是 top-p 的核心：按降序累加概率，一旦累计值达到 p 就 (4) 停下，`final_sample_size` 记住到此为止选入了几个 token。这就是核采样的自适应性来源：某个 token 概率就占了 0.9、p 设 0.9 时候选集只有它一个；概率平摊时要累加很多个才够 0.9，候选集自然变大。截断之后在 `[0, cumulative_prob)` 上取均匀随机数，落到哪个累积区间就选哪个 token（`sampling_cpu_util.cc:273`）。整条 top-p 路径的复杂度由 `nth_element` 的 O(vocab) 和这里的 O(k log k) 主导，k 通常在几十到几百，远小于词表规模，排序开销不大。

## 什么时候停

字不能一直吐下去。每一步之后都要问：该停了吗？这个判断集中在一个纯函数里（`ShouldStop`，`tasks.cc:86`），把所有停止条件收在一处：

```cpp
bool ShouldStop(bool hit_stop_tokens, int benchmark_decode_token_count,
                int num_decoded_steps, int current_step, int max_num_tokens,
                int max_output_tokens) {
  if (hit_stop_tokens && benchmark_decode_token_count == 0) {  // (1)
    return true;
  } else if (benchmark_decode_token_count > 0 &&
             num_decoded_steps >= benchmark_decode_token_count) {  // (2)
    return true;
  } else if (current_step >= max_num_tokens) {          // (3)
    return true;
  } else if (num_decoded_steps >= max_output_tokens) {  // (4)
    return true;
  }
  return false;
}
```

四个 `else if` 就是全部停止理由，按优先级排开。(1) 命中停止 token（模型自己说"我说完了"），但有个前置条件 `benchmark_decode_token_count == 0`：基准测量模式下要求固定跑满 N 步，这时候即便模型想停也不停，否则测出来的 tokens/s 会因提前结束而失真。(2) 是基准模式的对偶：指定了跑 N 步，跑够 N 步就停，不管模型说没说完。(3) `current_step >= max_num_tokens` 撞的是 KV cache 的物理上限：`current_step` 是 KV cache 里已经占了多少格，`max_num_tokens` 是这块 cache 一共几格（第 6 章会算这个数），放不下就必须停。(4) `num_decoded_steps >= max_output_tokens` 是用户设的输出长度上限，跟 cache 上限是两码事：cache 可能还很空，但用户只要 100 个 token。

留意 (3) 和 (4) 用的是两个不同计数器。`current_step` 从 prefill 就开始累加（提示词也占 cache），`num_decoded_steps` 只数 decode 阶段吐了几个。所以一个长提示词会让 (3) 提前触发而 (4) 还远没到——这正是"上下文放不下"和"输出够长了"两种停法的分野。至于取消，它不在 `ShouldStop` 里，而在循环开头单独探（前面那个 `cancelled->load()`）：取消要立即生效、不等这一步算完，语义上跟"这一步之后判该不该继续"不同，所以没混进来。把停止判定抽成纯函数（无副作用、只读参数），主循环只管"转"，停不停单独放、单独测：五个整型加一个布尔进去，一个布尔出来，测试可以穷举边界。

## 那个"吐半个字"的坑

第 2 章清单里的第 11、12 问，每个用过流式生成的人都撞见过：字偶尔会"吐半个"，停止词只出现一半时行为诡异。这一节交代它们。

先看停止词的部分匹配。停止词可能是"###"这样的多字符序列，而模型是一个 token 一个 token 出的。当它刚吐出"#"，你不知道接下来是"##"（真的要停）还是"#号说明"（不该停）。贸然把"#"吐给用户，万一后面真是停止词，你就多吐了不该吐的。LiteRT-LM 的停止符检测器为此逐 token 追踪每条停止序列的匹配进度（`StopTokenDetector`，`runtime/components/stop_token_detector.h:45`；`ProcessTokens` 在 `:67`）。关键是它暴露的这个查询：

```cpp
// Returns the maximum length of the partial stop token sequence found for the
// given batch index. zero if no partial stop token sequence is found or -1 if
// the stop token is already found.
int MaxPartialStopTokenLength(int index) const;  // (1)
```

(1) 这个函数回答的正是"现在匹配了多长"：返回 0 表示当前 token 跟任何停止序列都不沾边、可以直接输出；返回正数 `L` 表示末尾 `L` 个 token 构成了某条停止序列的前缀，悬而未决；返回 -1 表示停止序列已完整命中。`Run` 就靠这个返回值决定暂存多少 token（`runtime/core/tasks.cc:187`）：

```cpp
int max_length = stop_token_detector_.MaxPartialStopTokenLength(i);
if (max_length > 0) {
  pending_stop_tokens_[i].push(decoded_result.value()[i].value());   // (1)
  pending_stop_token_ids_[i].push(step_tokens[i]);
  num_buffered_tokens_[i] += step_tokens[i].size();
}
while (num_buffered_tokens_[i] > max_length) {                        // (2)
  result_text_[i] += pending_stop_tokens_[i].front();                // (3)
  pending_stop_tokens_[i].pop();
  // ...
}
if (max_length == 0) {                                                // (4)
  result_text_[i] += decoded_result.value()[i].value();              // (5)
  // ...
}
```

这段是"暂存—释放"的全部机制。(1) 一旦 `max_length > 0`，就把这个 token 的文本推进 `pending_stop_tokens_[i]` 这个队列，先不输出：它可能是停止词的一部分。(2) 的 `while` 是释放逻辑：队列里暂存的 token 数一旦超过 `max_length`，说明多暂存的那些**不可能**再是停止序列的前缀了（停止序列最多 `max_length` 长），于是 (3) 把队首多出来的部分放进 `result_text_` 正常输出。暂存的永远只是"最后 `max_length` 个 token"这个滑动窗口。(4)(5) 是最常见的情况：`max_length == 0`，当前 token 跟停止词无关，直接进 `result_text_`，零延迟。

那停止序列真命中了怎么办？此时 `AllDone()` 会返回真、`GetStopTokensFound()[i]` 置位，队列里暂存的整段就再也不会走到 (3) 被输出，它们连同停止词一起被丢弃。这是"暂存—释放"的另一半：暂存的内容，确认是停止词就**整段作废**，不是就顺次放出。代价是命中前缀期间输出会滞后至多 `max_length` 个 token，换来的是绝不会把半截停止词漏给用户。

再看"半个字"。子词分词（第 3 章）意味着一个 token 未必是一个完整的字，尤其是中文和 emoji，一个字可能由好几个 token 拼成。decode 一步只出一个 token，如果它是某个字的前半截，直接转文本就是乱码。`Run` 里处理这个的是同一套暂存逻辑，但用另一个成员队列（`runtime/core/tasks.cc:175`）：

```cpp
ASSIGN_OR_RETURN(step_tokens, tokenizer_.MergeTokenIds(              // (1)
                                  bpe_partial_token_ids_, step_tokens));
auto decoded_result =
    tokenizer_.TokenIdsToTexts(num_output_candidates_, step_tokens);
for (int i = 0; i < num_output_candidates_; ++i) {
  if (Tokenizer::IsIncompleteBpeSequence(decoded_result.value()[i])) {  // (2)
    bpe_partial_token_ids_[i] = step_tokens[i];                         // (3)
  } else if (!stop_token_detector_.GetStopTokensFound()[i]) {
    bpe_partial_token_ids_[i].clear();                                  // (4)
    // ... 前面那段停止词暂存/释放逻辑 ...
```

顺序是关键。(1) 每一步先把上一步攒下的半截 token（`bpe_partial_token_ids_`）跟这一步的新 token 拼起来，再一起转文本：转文本用的从来不是孤立的一个 token，而是"可能补全了的一串"。(2) `IsIncompleteBpeSequence` 判断拼出来的这串是不是仍然凑不成一个完整字符（比如一个 UTF-8 多字节序列缺了尾巴）；(3) 如果还不完整，就把这串整个存回 `bpe_partial_token_ids_[i]`，这一步不输出任何文本，等下一步再拼。(4) 一旦凑成完整字符，先清空暂存，再进入前面那段停止词判定——两层缓冲是串联的：先过 BPE 补全这关，够成完整字符了，才轮到停止词那关判定要不要暂存。

两个队列各管一件事：`bpe_partial_token_ids_` 保证输出的是完整字符，`pending_stop_tokens_` 保证输出的不含半截停止词。它们共用 `Run` 里那个 `for (int i ...)` 逐候选走一遍，互不干扰。

一个如实的补充：部分匹配路径在本书基准环境下无法从命令行触发。CLI 没有自定义停止词的参数，而基准模型的默认停止符来自聊天模板的收尾标记（实剖模板用 `<turn|>`，附录 D 第六节的同一份解剖）；据此推断它映射为单个专用 token，单 token 停止词一步即完全命中，暂存窗口长度为零。这套机制真正服务的是多 token 停止序列：引擎配置里 `stop_token_ids_` 的类型是 `std::vector<std::vector<int>>`（`runtime/engine/engine_settings.h:290`），每个停止词本身就是一段 id 序列，嵌入式集成方经 API 自定义字符串停止词时，部分匹配与暂存就成为必需。

这两处边界（停止词暂存、BPE 半字暂存）共同的模式是：**decode 是逐 token 的，但用户要的是完整、正确的文本单元，中间需要一层缓冲把"逐 token"翻译成"逐可显示单元"。** 看懂这层缓冲，你就看懂了流式生成为什么不是"算一个吐一个"这么简单。

## 小结

一个 decode step：前向出 logits →（可选）处理 → 采样出 token → 转文本流式输出 → 判定停止。其中有三处设计：`DecodeAndSample` 里那个 `if (sampler_)` 分出内部/外部两条采样路径（快与灵活的取舍，出口统一成同一种 token 向量）、`ShouldStop` 用四个 `else if` 把停止逻辑单独收拢成纯函数、以及 `Run` 里 `bpe_partial_token_ids_` 和 `pending_stop_tokens_` 两个队列串联起来的缓冲——前者保证吐出的是完整字符，后者保证不含半截停止词，共同把"逐 token"翻译成"逐可显示单元"。

第三篇（第 6-9 章）转入性能优化：先算清楚 KV cache 到底占了多少，以及那条 25 tokens/s 的上限，实测为什么还够不着。

---

## 练习与自查

1. **回搬账变体。** 若 logits 以 fp16 回传（词表仍为 262144），单步回搬多少字节？占 1.4 GiB 权重读取的比例变为多少？
2. **停止条件推理。** benchmark 模式指定 decode 128 步，第 20 步命中停止词，循环会停吗？依据 `ShouldStop` 的哪个分支？
3. **暂存时序。** 停止序列最长 3 个 token。模型依次产出 A、B、C，其中 A、B 是某停止序列的前缀而 C 使匹配失败。写出每一步用户实际收到的文本。
4. **半字机制。** 为什么流式解码转文本时用「暂存串 + 新 token」整体转换，而不是逐个 token 单独转换？
5. **动手实验。** 用 5 个不同 seed 跑温度 1.0（命令见附录 D 第八节），统计得到几种不同输出，并解释「开采样不等于每次必不同」。

> 提示与参考答案见附录 E。

<!-- 温度对比已实测回填（附录 D 第八节）。停止词暂存：CLI 不暴露自定义停止词、默认停止符为单 token，不可从命令行触发，正文已据实说明（结案）。 -->
