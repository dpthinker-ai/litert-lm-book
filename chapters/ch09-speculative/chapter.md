# 第 9 章 一次前向，多个 token：推测解码与 MTP

> 使命：说明 MTP 如何让 drafter 草拟多个 token，再由基础模型一次验证。分析接受比例与每轮运行成本如何共同决定加速比。

第 1 章讨论了缓解 decode 内存带宽约束的两类方法。硬件可以提高带宽，量化可以减少每 token 读取的字节数。第三种方法是推测解码（speculative decoding）。基础模型每读取一次权重，尽量确认多个 token。本章据此定量分析第 2 章第 19 问：推测解码为什么可能加速，又会在什么条件下减速。

## 推测解码的目标：一次验证多个候选 token

dense 模型在 decode 阶段通常每生成一个 token 就读取一次模型权重。普通 decode 前向只确定一个 token。推测解码试图增加每次基础模型权重读取所确认的 token 数。

推测解码先由低成本的 drafter 草拟若干 token。基础模型（base 模型）再用一次前向验证这些候选。匹配的最长前缀被接受；第一个不匹配位置改用基础模型给出的 token。验证多个候选只执行一次基础模型前向，而不是对每个候选分别执行一次。若一轮接受多个 token，这次前向的成本便由这些 token 分担。

推测解码有多种实现。drafter 可以是独立模型，也可以来自与基础模型联合训练的预测头。这类预测头称为多 token 预测（Multi-Token Prediction，MTP）头。Google 发布的 Gemma 4 MTP drafter 使用目标模型的 activation，并与目标模型共享 KV cache。[^ch09-google-mtp] LiteRT-LM 在运行时把 drafter 装载为独立模型，对应成员是 `mtp_drafter_model_`。验证使用基础模型的 `"verify"` signature。常量定义见 `runtime/executor/llm_litert_mtp_drafter.cc:63`。`base_model.FindSignature(kVerifySignatureRunner)` 的调用见 `runtime/executor/llm_litert_mtp_drafter.cc:227-228`。

## 机制：串行草拟，批量验证

一次 `Draft()`（`runtime/executor/llm_litert_mtp_drafter.cc:453-469`）包含三个阶段：

```cpp
ASSIGN_OR_RETURN(std::vector<int> drafted_tokens,
                 RunDraftingLoop(token_id, activations));          // (1)

RETURN_IF_ERROR(PrepareVerifierInputBuffers(
    position, token_id, drafted_tokens, input_kv_cache_buffers));  // (2)
// ...
ASSIGN_OR_RETURN(std::vector<int> verifier_id_vector, RunVerification());  // (3)
```

`(1)` 草拟 G 个 token。`(2)` 把上一个已确认 token 和 G 个草稿拼成 verify 输入。`(3)` 由基础模型一次完成验证。接受循环再比较两组 token。

第一步是草拟。drafter 逐个生成接下来的 G 个候选 token。对应函数是 `RunDraftingLoop`，见 `runtime/executor/llm_litert_mtp_drafter.cc:328-371`。G 是草拟步数（代码里 `num_draft_steps_`）。循环体一步一个 token：

```cpp
for (int i = 0; i < num_draft_steps_; ++i) {                      // (1)
  RETURN_IF_ERROR(embedding_manager_.LookupDecode(last_drafted_token_id,
                                                  embedding_vector));
  // ...
  // Concatenated embedding + activation has shape [B = 1, T = 1, D = 3072]
  RETURN_IF_ERROR(ConcatenateEmbeddingsAndActivations(              // (2)
      embedding_vector, *activations_ptr, *drafter_activations_buffer));
  // ...
  LITERT_RETURN_IF_ERROR(mtp_drafter_model_.RunAsync(              // (3)
      drafter_signature_.Key(), active_drafter_input_buffers_,
      active_drafter_output_buffers_, async));
  // ...
  RET_CHECK_EQ(id_vector.size(), 1);                               // (4)
  drafted_tokens.push_back(id_vector[0]);
  last_drafted_token_id = id_vector[0];                            // (5)
  activations_ptr = &active_drafter_output_buffers_["projected_activations"];
}
```

`(1)` 循环 G 次，每次串行生成一个 token。`(3)` 每步运行独立装载的 `mtp_drafter_model_`，而非基础模型。`(2)` 的输入由词嵌入和上一步的隐藏态 activation 拼接而成。源码注释以 `[B=1, T=1, D=3072]` 为例，即 1536 维词嵌入加 1536 维 activation。具体维度随模型而定。本书基准模型的 drafter 输入为 `[1, 1, 5120]`，由 2560 维词嵌入和 2560 维 activation 组成。两部分都与主干的 `model_dimension = 2560` 一致，见附录 D。`(4)` 断言每步只产出一个 token。`(5)` 把本步输出作为下一步输入，drafter 仍按自回归方式生成。总成本包括 G 次 drafter 前向和一次 verify 前向。MTP 是否加速取决于这个总成本，不能只看 drafter 的模型尺寸。

第二步执行批量验证。运行时把上一个已确认 token 和 G 个草稿交给基础模型的 verify signature。对应函数为 `RunVerification`，见 `runtime/executor/llm_litert_mtp_drafter.cc:437-450`：

```cpp
LITERT_RETURN_IF_ERROR(base_model_.RunAsync(                       // (1)
    verify_signature_.Key(), active_verifier_input_buffers_,
    active_verifier_output_buffers_, async));
// ...
LITERT_ASSIGN_OR_RETURN(auto id_vector,
                        CopyFromTensorBuffer<int32_t>(verifier_id_tensor_));
RET_CHECK_EQ(id_vector.size(), num_draft_steps_ + 1);             // (2)
return id_vector;
```

`(1)` 在一次 `Draft()` 调用内，`RunAsync` 是唯一一次基础模型前向。G 个草稿位置在这次前向中一起验证。`(2)` 返回长度为 G+1 的 id 序列，每个位置对应一个基础模型输出；最后一个位置在 G 个草稿全部匹配时使用。“一次基础模型前向”仅指一次 `Draft()`。prefill 后首次调用 `Decode()` 还会先执行一次普通 decode。

第三步是接受。逐位比对草拟结果与验证结果，接受最长的匹配前缀。

<figure>
{{#include figs/fig-9-1.svg}}
<figcaption>图 9-1　一次 `Draft()` 调用执行 G 次 drafter 前向和一次基础模型 verify，返回匹配前缀及一个 bonus token，即 1 到 G+1 个 token；首次 `Decode()` 还包含一次普通 decode。</figcaption>
</figure>

## 接受循环：匹配前缀与 bonus token

第三步处理接受与回退（`Draft()` 接受循环，`runtime/executor/llm_litert_mtp_drafter.cc:471-484`）：

```cpp
int num_correct_tokens = 0;
int bonus_token = -1;
for (int i = 0; i < num_draft_steps_; ++i) {
  last_verified_token_id_idx_ = i;
  if (verifier_id_vector[i] != drafted_tokens[i]) {   // (1)
    bonus_token = verifier_id_vector[i];               // (2)
    break;
  }
  num_correct_tokens++;                                // (3)
}
if (bonus_token == -1) {                               // (4)
  last_verified_token_id_idx_ = num_draft_steps_;
  bonus_token = verifier_id_vector[num_draft_steps_];  // (5)
}
```

`(1)` 从头逐位比较草拟结果与验证结果。结果一致时，`(3)` 增加 `num_correct_tokens`。遇到第一个不一致位置后，`(2)` 停止比较。基础模型在该位置的输出被记为 bonus token。若循环结束后 `bonus_token` 仍为 −1，说明 G 个草稿全部匹配；`(5)` 此时取 verify 输出的第 G+1 个 token。无论是否全部匹配，基础模型本轮都提供一个不来自已接受草稿的输出 token。

`last_verified_token_id_idx_` 不参与接受比例统计。它保存 verify 输出中最后一个有效位置的下标。下一轮 `RunDraftingLoop` 会进入 `ConcatenateEmbeddingsAndActivationsFromVerifierBuffer` 分支。该分支用此下标读取相应 activation，作为草拟起点。隐藏态来源见本章后文。

`Draft()` 最后返回接受前缀和一个 bonus token（`runtime/executor/llm_litert_mtp_drafter.cc:492-493`）：

```cpp
// The first token comes from the decode output and is always correct.
drafted_tokens.resize(num_correct_tokens);   // (1)
drafted_tokens.push_back(bonus_token);       // (2)
```

`(1)` 把草拟序列截断到接受长度，`(2)` 再追加 bonus。源码中的“always correct”表示该 token 来自基础模型输出。采用它不需要以 drafter 匹配为前提。即使 `num_correct_tokens` 为 0，`resize(0)` 后的 `push_back` 仍会返回 1 个 token。

一次 `Draft()` 至少返回 1 个 token。即使第一个草稿不匹配，函数也会返回基础模型在该位置的输出。这个结论只是每轮输出数量的下界，不是运行成本的上界。该轮还包含 G 次 drafter 前向和缓冲操作。prefill 后的首次 `Decode()` 另有一次普通基础模型前向，也不属于这个稳态下界。

## 采样约束：当前 MTP 路径采用贪心接受

接受循环使用严格相等比较 `verifier_id_vector[i] != drafted_tokens[i]`。这与带概率接受步骤的经典推测采样不同。Leviathan 等（2023）采用 rejection sampling。[^ch09-leviathan] Chen 等（2023）也采用保留目标模型分布的修正 rejection sampling。[^ch09-chen] LiteRT-LM 当前 MTP 路径没有该步骤；drafter 与 verifier 均采用贪心采样，接受条件是两个 token id 相等。

强制贪心的地方在构造采样器的辅助函数里（`CreateGreedySampler`，`runtime/executor/llm_litert_mtp_drafter.cc:65-77`）：

```cpp
proto::SamplerParameters sampler_params;
sampler_params.set_type(proto::SamplerParameters::TOP_P);
sampler_params.set_k(1);                                          // (1)
sampler_params.set_p(0.0f);                                       // (2)
sampler_params.set_temperature(1.0f);                            // (3)
sampler_params.set_seed(0);
return CreateSampler(backend, output_heads, std::move(sampler_params),
                     env.Get(), sequence_size, vocab_size,
                     activation_data_type);
```

`(1)` 把 top-k 设为 1，`(2)` 把 top-p 设为 0，候选集中只保留 argmax。`(3)` 的 temperature 为 1.0，但在 k=1 时不会引入随机选择。drafter 与 verifier 共用该工厂函数。drafter 的 `sequence_size` 为 1，见 `runtime/executor/llm_litert_mtp_drafter.cc:263-267`；verifier 的值为 G+1，见 `runtime/executor/llm_litert_mtp_drafter.cc:268-272`。接受阶段比较两侧 logits 的 argmax token。

保持采样分布不变的推测采样需要 verifier 提供各位置的概率分布。接受阶段按概率比值决定是否采用草稿；拒绝后还要从残差分布重新采样。LiteRT-LM 的 verifier 只回传形状为 `[1, G+1]` 的 token id。接受循环执行整数比较，不处理完整 logits 分布。这省去了完整概率分布的回传和逐位处理，但适用范围限于贪心 token 比较。当前代码没有实现随机采样所需的概率接受与残差重采样。即使设置 temperature 或 top-p，MTP 路径也无法保持非推测路径的采样分布。

## 验证输入的构造：位置、mask 与 KV cache 的复制

verify 使用包含 G+1 个位置的批量输入，而不是普通 decode 的单位置输入。`PrepareVerifierInputBuffers`（`runtime/executor/llm_litert_mtp_drafter.cc:374-421`）负责构造这组缓冲：

```cpp
auto* prefill_input_pos_ptr =
    static_cast<int32_t*>(verifier_input_pos_lock_and_addr.second);
for (int i = 0; i < num_draft_steps_ + 1; ++i) {                  // (1)
  *prefill_input_pos_ptr++ = position + i;
}
// ...
RETURN_IF_ERROR(FillAttentionMask(verifier_input_buffers_["mask"],
                                  /*start_step=*/position,
                                  /*steps=*/num_draft_steps_ + 1)); // (2)
// ...
RETURN_IF_ERROR(embedding_manager_.LookupPrefill(                  // (3)
    drafted_tokens_with_input_token, &verifier_input_buffers_["embeddings"],
    /*offset=*/0));
// ...
for (const auto& [input_name, input_buffer] : input_kv_cache_buffers) {
  LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate()); // (4)
  active_verifier_input_buffers_[input_name] = std::move(input_buffer_dup);
}
```

`(1)` 把 `input_pos` 依次写为 `position` 到 `position+G`。这 G+1 个位置对应一个已确认 token 和随后 G 个草稿。`(2)` 调用 `FillAttentionMask`，参数为 `start_step=position`、`steps=G+1`。因果 mask 使每个待验证位置只能读取此前的前缀，包括已确认 token 和更早的草稿位置。第 `i` 个位置的 logits 因而与逐 token decode 到该位置时具有相同的可见前缀。`(3)` 使用 `LookupPrefill` 批量查询 `[已确认 token, 草稿_1, …, 草稿_G]` 的 embeddings，接口与第 3 章的 prefill 路径相同。`(4)` 为每个 KV cache 输入复制 `TensorBuffer` 句柄，供 verify 使用。

`Duplicate()` 复制缓冲句柄而非底层数据，采用第 7 章所述的 `TensorBuffer` 浅复制语义。verify 前向会在现有 KV cache 之后追加 G+1 个位置；第 6 章介绍的双缓冲路径要求分别传入读写句柄。紧随其后的分支处理单缓冲 KV cache：

```cpp
if (active_verifier_input_buffers_.contains("param_tensor")) {   // (1)
  RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
      active_verifier_input_buffers_["param_tensor"], position,
      num_draft_steps_ + 1));
}
```

`(1)` 单缓冲 KV cache 把读写位置参数放在 `param_tensor` 中。这里从 `position` 开始填入 G+1 步对应的参数，使 verify 写入相应缓存区间。双缓冲路径传入复制后的句柄，单缓冲路径则额外设置 `param_tensor`。输出侧由 `PrepareVerifierOutputBuffers` 处理，见 `runtime/executor/llm_litert_mtp_drafter.cc:424-434`。它复制各个 KV cache 输出句柄，并调用 `ClearEvent()` 清除旧的完成事件。

## drafter 的隐藏态拼接：两条 activation 来源

MTP drafter 每步都把词嵌入与隐藏态拼接后输入模型。前文给出的示例形状为 `[1536+1536=3072]`。`RunDraftingLoop` 的以下分支选择隐藏态来源（`runtime/executor/llm_litert_mtp_drafter.cc:346-353`）：

```cpp
if (activations_ptr) {
  RETURN_IF_ERROR(ConcatenateEmbeddingsAndActivations(              // (1)
      embedding_vector, *activations_ptr, *drafter_activations_buffer));
} else {
  RETURN_IF_ERROR(ConcatenateEmbeddingsAndActivationsFromVerifierBuffer( // (2)
      embedding_vector, verifier_output_buffers_["activations"],
      last_verified_token_id_idx_, *drafter_activations_buffer));
}
```

`(1)` 当 `activations_ptr` 非空时，拼接参数传入的 `activations`。一轮草拟的第一步可采用这条路径；该 activation 来自基础模型此前的 decode 前向。第一步结束后，`activations_ptr` 指向 drafter 上一步输出的 `projected_activations`。赋值位置见 `runtime/executor/llm_litert_mtp_drafter.cc:368-369`。后续步骤据此自回归生成。`(2)` 当 `activations_ptr` 为空时，隐藏态来自上一轮的 verifier 输出缓冲。`last_verified_token_id_idx_` 选择被接受位置对应的 activation。接受循环维护该下标，供下一轮草拟使用。

两条来源对应两种调用时机。prefill 后首次 decode 使用路径 `(1)`。drafter 此时读取普通 decode 输出的 activation。进入稳态后，它使用上一轮 verify 中被接受位置的 activation，即路径 `(2)`。两条路径都让 drafter 获得上一个 token 和基础模型产生的上下文表示。本章没有比较其他 drafter 结构，不能据此推断相对命中率。

`ConcatenateEmbeddingsAndActivations`（`runtime/executor/llm_litert_mtp_drafter.cc:79-99`）先把词嵌入复制到输出缓冲前半段，再把 activation 复制到后半段。两段各含 `model_dimension` 个 float，组成 drafter signature 的双倍宽度输入。本书基准模型的形状为 2560 + 2560 = 5120，记录见附录 D。这段函数本身只执行两次内存复制，不含矩阵计算；它在端到端时延中的占比仍需测量，不能仅凭代码结构判定为可忽略。

## 集成：`Draft()` 的调用与 token 回写

执行器的 `Decode()` 负责把 drafter 接入推理流程，入口见 `runtime/executor/llm_litert_compiled_model_executor.cc:1003`。`mtp_drafter_ == nullptr` 时执行普通 decode，判断位置见 `runtime/executor/llm_litert_compiled_model_executor.cc:1007`；否则执行 MTP 路径。MTP 路径还会根据当前调用是否为 prefill 后的第一次 decode 选择不同分支：

```cpp
bool last_run_is_decode = llm_context_->runtime_state().ran_decode; // (1)
if (last_run_is_decode) {
  // ...
  LITERT_ASSIGN_OR_RETURN(output_tokens_vector,
                          mtp_drafter_->Draft(step_and_token.step,
                                              step_and_token.token[0]->id(),
                                              /*activations=*/std::nullopt, // (2)
                                              *input_kv_cache_buffers_,
                                              *output_kv_cache_buffers_));
  RET_CHECK_EQ(output_tokens_vector.size(), 1);
  llm_context_->runtime_state().current_step +=                    // (3)
      output_tokens_vector[0].size();
```

`(1)` `ran_decode` 表示上一次运行是否为 decode。进入稳态后执行该分支。`(2)` 将 `activations` 设为 `std::nullopt`，drafter 从 verifier 缓冲读取上一轮被接受位置的隐藏态。`(3)` 按 `Draft()` 本轮返回的 token 数增加 `current_step`。该数等于接受前缀长度加 1 个 bonus token。稳态下，一次 `Decode()` 直接返回这组 token，范围为 1 到 G+1。

另一条分支处理 prefill 后的第一次 decode。此时运行时先执行普通 decode，取得首个 token 及其 activation，再调用 `Draft()`：

```cpp
RETURN_IF_ERROR(SampleLogits(decoded_logits, output_tokens));      // (1)
// ...
token_id = output_tokens_vector[0][0];
// ...
LITERT_ASSIGN_OR_RETURN(
    auto activations, decode_output_buffers_["activations"].Duplicate()); // (2)
// ...
LITERT_ASSIGN_OR_RETURN(
    output_tokens_vector,
    mtp_drafter_->Draft(llm_context_->runtime_state().current_step - 1, // (3)
                        token_id, std::move(activations),
                        *input_kv_cache_buffers_, *output_kv_cache_buffers_));
llm_context_->runtime_state().current_step += output_tokens_vector[0].size();
output_tokens_vector[0].insert(output_tokens_vector[0].begin(), token_id); // (4)
```

`(1)` 普通 decode 先采样出 `token_id`。`(2)` 复制该次 decode 输出的 `activations`，并传给 `Draft()`；这对应上一节的第一条 activation 来源。`(3)` 传入 `current_step - 1`。此时 `DecodeLogits` 已将 `current_step` 增加一次，而草拟位置从刚得到的 token 开始计算。`(4)` 最后把这个普通 decode token 插到 `Draft()` 返回序列之前。第一次 `Decode()` 返回 2 到 G+2 个 token。其中 1 个来自普通 decode，另有 1 到 G+1 个来自 `Draft()`。

首轮与稳态的基础模型调用次数也不同。第一次 `Decode()` 执行一次普通 decode 和一次 verify，共两次基础模型前向；稳态的每次 `Decode()` 只执行一次 verify。两种分支都执行 G 次 drafter 前向。分析长期吞吐时通常采用稳态口径，但测量短输出时，首轮的额外基础模型前向不能省略。

## 接受比例与加速比

MTP 是否加速，取决于每轮产出的 token 数和该轮的运行成本。`Draft()` 在每轮结束时累加两个计数（`runtime/executor/llm_litert_mtp_drafter.cc:494-495`）：

```cpp
num_drafted_tokens_ += num_draft_steps_;    // (1)
num_verified_tokens_ += num_correct_tokens; // (2)
```

`(1)` 每轮把草稿总数增加 G。`(2)` 只累加匹配前缀的长度 `num_correct_tokens`，不含 bonus token。drafter 析构时输出这两个计数及其比值（`runtime/executor/llm_litert_mtp_drafter.cc:165-172`）：

```cpp
LlmLiteRtMtpDrafter::~LlmLiteRtMtpDrafter() {
  ABSL_LOG(INFO) << "Num drafted tokens: " << num_drafted_tokens_;
  ABSL_LOG(INFO) << "Num verified tokens: " << num_verified_tokens_;
  if (num_drafted_tokens_ > 0) {
    ABSL_LOG(INFO) << "Success rate: "                          // (1)
                   << static_cast<double>(num_verified_tokens_) /
                          num_drafted_tokens_;
  }
}
```

设共执行 R 轮 `Draft()`，第 j 轮接受的草稿数为 K_j，且 0 ≤ K_j ≤ G。日志中的 `Success rate` 是聚合接受比例：

$$ \widehat r = \frac{\sum_{j=1}^{R} K_j}{R G} $$

这个量直接给出样本中的平均接受数：`平均 K = G × r̂`。每轮还会返回 1 个 bonus token，所以稳态下 `Draft()` 的样本平均产出为：

$$ \overline N = 1 + G\widehat r $$

日志中的 r̂ 不是“每个位置独立匹配的概率”。理论曲线另设参数 p。给定前 k−1 个草稿均已匹配，第 k 个草稿继续匹配的条件概率假定恒为 p。此时 `P(K ≥ k) = p^k`，于是：

$$ E[N] = 1 + E[K] = 1 + \sum_{k=1}^{G} p^k $$

这两个口径满足 `r = E[K]/G = (Σp^k)/G`，不能把日志中的 r 直接代入 p 的位置。真实数据还可能随位置变化，未必符合恒定 p 的假设。

归一化轮次成本 q 是两个墙钟时间之比。分子是一次稳态 MTP 轮次，分母是一次普通 decode step。端到端加速比可写成：

$$ \text{speedup} \approx \frac{1 + G r}{q} $$

图 9-2 采用简化成本模型 `q = 1 + Gc`。其中 c 是归一化的有效开销参数，不是直接测得的 drafter 单步前向时间。该参数合并了 verify 与普通 decode 的成本差异、G 次 drafter 前向和缓冲操作。结合恒定条件概率 p 的假设，得到：

$$ \text{speedup}(p,c) \approx
\frac{1 + \sum_{k=1}^{G} p^k}{1 + Gc} $$

取 G=3、c=0.15，分母为 1.45。p=0.8 时，期望产出为 2.952 个 token，加速比约 2.04；p=0.4 时约为 1.12；p=0.2 时约为 0.86。令加速比等于 1，解得盈亏平衡点 p≈0.317，而不是 0.25。这些数值只说明理论曲线的形状，不代表本书设备上的实测成本。

<figure>
{{#include figs/fig-9-2.svg}}
<figcaption>图 9-2　理论曲线使用逐位条件匹配概率 p；在 G=3、c=0.15 的示意条件下，盈亏平衡点约为 0.32。日志聚合比例 r 不能直接作为横轴 p。</figcaption>
</figure>

聚合接受比例 r 增大时，平均每轮产出 `1+Gr` 随之增加。轮次成本 q 同样影响加速比。即使 r 很高，较大的 q 仍会限制收益。r 较低时，MTP 可能只返回 1 个或少量 token。该轮仍要执行 drafter 与 verify，吞吐可能低于普通 decode。Google 报告 Gemma 4 MTP drafter 在其跨模型、硬件与运行时测试中最高达到约 3 倍。[^ch09-google-mtp] 该数字不是本书设备上的预期值。

附录 D 保存了 Gemma 4 E4B 的两组 Mac 记录。测试条件为 context 1024、decode 128 token，参数分别设为 `false` 与 `auto`。本章“开启条件”一节核对的设置链路表明，v0.13.1 的 `auto` 最终沿用 C++ 默认值 `false`。两组记录都来自关闭 MTP 后的独立采样。CPU 中位数为 22.8 和 24.9 tokens/s；`false` 组的三次运行分布在 20.1 到 24.9 tokens/s。GPU 中位数为 50.0 和 50.2 tokens/s。两组差异只能反映这几次运行的波动，不能估计 MTP 的开关收益。当前没有可核查的 Mac 强制开启记录。

| 模式 | cpu decode tokens/s | gpu decode tokens/s | 说明 |
|---|---|---|---|
| 关（`false`） | 22.8 | 50.0 | 基线 |
| `auto` | 24.9 | 50.2 | v0.13.1 实为关，与基线同行为的再采样 |

> 表 9-1　Mac 归档记录中的 `false` 与 `auto` 均为关闭行为，各列为 3 次运行的中位数；该表不能用于估计 MTP 收益。

附录 D 还记录了 drafter 析构时输出的计数器。实验通过 Python SDK 把日志级别设为 VERBOSE；计数器代码见 `runtime/executor/llm_litert_mtp_drafter.cc:165-172`。故事提示词产生 213 个草稿，其中 56 个匹配，故 `r̂=56/213≈0.263`。平均每次 `Draft()` 返回 `1+3r̂≈1.79` 个 token。代码提示词产生 3069 个草稿，其中 3054 个匹配。对应的 `r̂≈0.995`，平均返回约 3.99 个 token。

在图 9-2 的示意假设 `q=1.45` 下，两个样本对应的加速比分别约为 1.23 和 2.75。把 r̂ 误作 p 会得到 0.93；该结果混用了两种口径。这两次计数实验没有同时测量 q，所以上述示意值不是实测速率。两次实验的提示词也与固定长度 benchmark 不同。故事样本的 r̂ 因而不能用于推断该 benchmark 开启 MTP 后的吞吐。

固定长度 benchmark 的输入由 `runtime/core/session_utils.cc:68-73` 构造。代码先对提示词分词，再调用 `ids.resize(N)`。N 大于原 token 数时，新增的 `int` 元素为 0；N 更小时，序列会被截断。该负载不是定长的自然文本，不能沿用前述两个提示词的 r̂。Mac 归档数据既没有强制开启组，也没有这组 benchmark 的 drafter 计数器。现有可核查的强制开启端到端对照只有 Android 记录。没有证据把其中的减速归因于低接受比例。

这组计数不需要修改或重新编译运行时。drafter 已维护 `num_drafted_tokens_` 和 `num_verified_tokens_`，并在析构时通过 `ABSL_LOG(INFO)` 输出。Python SDK 的 `set_min_log_severity`（`python/litert_lm/_ffi.py:450`）可把日志级别调到 VERBOSE。若要得到分阶段或逐轮数据，则需新增遥测接口；当前日志只能提供整个 drafter 生命周期内的聚合比例。

`LiteRT-LM#2227` 报告了 PowerVR GPU 上开启 MTP 后 decode 吞吐下降的案例。[^ch09-issue-2227] 该 issue 使用 LiteRT-LM 0.11.0、Gemma 4 E2B 和俄文分类负载；其中关于 GPU 路径的原因分析明确标为假设。本章的公式只说明两类可能条件：r 偏低，或归一化轮次成本 q 偏高。它不能从吞吐数据中区分二者，也不能据此确认 issue 的根因。

附录 D 第十三节记录了一台 Qualcomm 机型上的 MTP 减速。合成负载使用 `benchmark_prefill_tokens=1024`。强制开启 MTP 后，CPU decode 从 10.0 降到 2.8 tokens/s；GPU 从 18.0 降到 12.6 tokens/s。在该设备和负载的这次测试中，MTP 组吞吐更低。实验没有同步记录 r 或分解 q，不能把原因归到其中一个参数。

同一台手机另有自然代码提示词的记录，参数为 `benchmark_prefill_tokens=0`。GPU 的单次观测从 16.0 变为 32.0 tokens/s，decode 长度为 128。CPU 的单次观测从 11.6 变为 12.6 tokens/s，decode 长度为 64。每个条件只运行 1 次，两种后端的 decode 长度也不同。记录只能说明这些测试中存在观测差异，不能估计稳定加速比或比较后端绝对值。自然代码吞吐测试与 r̂ 计数也不是同一次运行。现有数据不能把差异分解为接受比例和轮次成本两部分。

本书基准模型的 verify signature 中，`input_pos` 形状为 `[4]`，故 G=3。一次 `Draft()` 最多返回 4 个 token。附录 D 另有两组自然代码长生成的单次记录。Mac GPU 从 58.7 变为 133.9 tokens/s，比值为 2.28。手机 GPU 从 18.62 变为 37.38 tokens/s，比值为 2.01。两组记录使用不同采集入口，没有重复运行，也没有在同一次运行中记录 r。吞吐比因而不能唯一确定 q 或 c，也不能估计跨设备的稳定差异。

在 `r=1`、`q=1+3c` 的假设下，2.28 倍和 2.01 倍分别对应 c≈0.25 和 c≈0.33。这里的 c 是由端到端结果反推的有效参数。它包含 verify、drafter 和运行时操作，不是 drafter 模型的实测单步成本。若实际 r 小于 1，反推出的 c 也会更小。这些数值不能证明两台设备已经达到吞吐上限。在相同简化模型下，即使 r=1，3 倍加速也要求 c≤1/9≈0.11。现有数据不足以把官方结果[^ch09-google-mtp] 归因于某种硬件路径。drafter 文件约 45 MB 也不能推出 c≈0.02，因为模型存储大小不是端到端时延的比例尺。

推测解码是否加速由 r 和 q 共同决定。聚合接受比例 r 决定每轮平均产出，归一化成本 q 表示取得这些输出所需的时间。两者都会随模型、输入内容、生成位置、设备和后端变化。评估目标负载时，需要在同一次测试中记录接受比例与吞吐。

## 多 token 返回后的停止检测与回退

MTP 还改变了 executor 的返回形态。普通 decode 通常返回一个 token，一次 MTP `Decode()` 则可能返回一段序列。任务层必须按顺序处理这段序列，因为停止序列可能在批次中间命中，BPE 片段也可能跨越两次 MTP 调用。

`DecodeOneStep::Run` 先取得 executor 返回的二维 token 序列，并检查各候选的序列长度是否相同。检查代码见 `runtime/core/tasks.cc:147-156`。随后按位置逐个处理，见 `runtime/core/tasks.cc:163-179`：

```cpp
for (size_t step = 0; step < sequence_length; ++step) {             // (1)
  std::vector<std::vector<int>> step_tokens;
  // ... 每个候选在当前位置取一个 token
  RETURN_IF_ERROR(stop_token_detector_.ProcessTokens(step_tokens)); // (2)
  ASSIGN_OR_RETURN(step_tokens, tokenizer_.MergeTokenIds(           // (3)
                                    bpe_partial_token_ids_, step_tokens));
  auto decoded_result =
      tokenizer_.TokenIdsToTexts(num_output_candidates_, step_tokens);
```

`(1)` 将本轮返回序列展开为逐位置处理。`(2)` 先推进停止序列检测器，`(3)` 再合并未完成的 BPE token id。即使 executor 一次返回多个 token，停止检测与文本解码仍按 token 顺序推进。

停止序列在批次中间命中时，v0.13.1 会调整 executor 的逻辑位置。代码以 `sequence_length - step` 计算回退量，再调用 `SetCurrentStep`（`runtime/core/tasks.cc:223-233`）：

```cpp
if (all_done) {
  if (step != sequence_length - 1) {
    int diff = sequence_length - step;                               // (1)
    ASSIGN_OR_RETURN(int current_step, executor_.GetCurrentStep());
    RETURN_IF_ERROR(executor_.SetCurrentStep(current_step - diff));   // (2)
  }
  return true;
}
```

假设本轮返回 4 个 token，停止 token 位于下标 1。可见输出只保留下标 0；`diff=4-1=3`，`(2)` 回退停止位置及其后的两个位置。若停止 token 位于本批最后一个位置，`if` 条件不成立，代码不会调用 `SetCurrentStep`。这是两个不同分支，不能把批中回退规则推广到批次末位。

compiled executor 的 `SetCurrentStep` 检查新位置不超过已处理 token 数且不小于 0。检查通过后，它只更新逻辑 `current_step`，不清除 KV cache 缓冲。对应代码见 `runtime/executor/llm_litert_compiled_model_executor.cc:1452-1479`。后续调用读取调整后的逻辑位置。

<figure>
{{#include figs/fig-9-3.svg}}
<figcaption>图 9-3　示例中的停止 token 位于 4-token 批次的下标 1，任务层保留下标 0，并将逻辑位置回退 3；命中批次末位时不执行该回退分支。</figcaption>
</figure>

`DecodeOneStep` 在每次 `Run` 开始时清空结果，再累积本批可见文本和 token id（`runtime/core/tasks.cc:158-214`）。外层循环在 `Run` 返回后至多发送一次 `TaskState::kProcessing` 回调，而且只有 `any_updates` 为真时才发送（`runtime/core/tasks.cc:523-566`）。一次 executor `Decode()` 可能不产生可见更新，也可能产生一次携带多个 token id 的更新。回调次数既不等于 executor 调用次数，也不等于生成 token 数。统计可见输出时应读取回调中的 token id；统计 executor 的位置推进量时应读取 `current_step` 差值。

| 终止条件 | 检查位置 | 对本轮多 token 序列的处理 |
|---|---|---|
| 停止 token 或停止序列 | `DecodeOneStep::Run` 内逐 token 检查 | 批中命中时回退停止位置及后续位置；末位命中不回退 |
| benchmark decode token 数量 | `Run` 返回后调用 `ShouldStop` | 任务层没有按剩余额度截短本批 |
| KV cache 最大长度 | `Run` 返回后调用 `ShouldStop` | 使用本轮后的绝对 `current_step` 判断 |
| `max_output_tokens` | `Run` 返回后调用 `ShouldStop` | 使用本轮后的 `current_step` 增量判断 |
| 取消标志 | 下一轮 `Run` 之前检查 | 已完成并回调的上一批不会被撤销 |

> 表 9-2　停止序列在批内逐 token 检查；三个数值上限在整批返回后检查，取消标志则在下一批开始前检查。

外层循环在回调之后读取 `current_step`，并以它相对 decode 起点的增量作为 `num_decode_steps`（`runtime/core/tasks.cc:569-572`）。`ShouldStop` 再将这个增量与 benchmark 数量、`max_output_tokens` 比较，并将绝对 `current_step` 与 KV cache 上限比较（`runtime/core/tasks.cc:86-105`）。这里的 step 数是 token 位置增量，不是 `Run` 调用次数。

据此可以给出任务层的上界推断。若某项数值限制只剩 B 个 token，而本轮在没有提前命中停止序列时推进 L 个位置，且 L>B，任务层将在回调后停止，越过阈值 L-B 个位置。稳态 MTP 一轮最多返回 G+1 个 token，任务层的理论最大越界量为 G；prefill 后第一次 `Decode()` 最多返回 G+2 个 token，对应 G+1。这个推断来自后置检查和 `current_step` 增量，前提是 executor 能完成该次调用。KV cache 容量还可能由 executor 或模型形状提前约束，不能仅凭任务层代码断言真实调用一定越界。

现有 `max_output_tokens` 测试使用一次返回一个 token 的 `FakeLlmExecutor`（`runtime/core/pipeline_test.cc:117-121`、`runtime/core/pipeline_test.cc:249-272`）。它没有覆盖一次返回多 token 的边界。需要新增可为单个候选返回 token 序列的任务层测试 executor，再用真实 MTP 模型核对执行器状态：

| 输入序列与条件 | 应检查的结果 | 用途 |
|---|---|---|
| `[a, stop, b, c]`，停止 token 位于下标 1 | 只输出 `a`，最终位置为批前位置加 1 | 验证批中命中及 `diff=3` 的回退 |
| `[a, b, c, stop]`，停止 token 位于末位 | 可见输出、最终位置和 `SetCurrentStep` 调用次数 | 覆盖末位不回退分支 |
| 上一批末尾是停止序列前缀，本批首 token 完成匹配 | pending 队列和跨批停止行为 | 验证批边界不改变停止序列语义 |
| 剩余 `max_output_tokens=1`，executor 返回 4 个 token | 输出长度、回调 token id 和最终位置 | 验证后置检查与越界量 |
| 4 个 token 均可见且启用 streaming | 回调次数及单次回调的 token id 数 | 验证一个批次至多一次可见更新 |

> 表 9-3　多 token 返回测试需同时检查可见文本、回调 token id、逻辑位置和回退调用；只检查最终字符串不足以覆盖状态语义。

一次 executor `Decode()` 表示一轮运行调用，`current_step` 增量表示本轮推进的 token 位置数。MTP 性能分析用前者计算轮次成本，用后者计算产出和数值终止条件。流式接口还要区分可见 token 与因 BPE 或停止前缀而暂存的 token。

## 开启条件：模型能力与固定草拟步数

模型文件需要同时包含 MTP drafter 和相应的 verify signature；调用方还必须显式启用推测解码。草拟步数 G 由 verify signature 的形状固定。

`HasSpeculativeDecodingSupport` 检查模型是否支持推测解码，声明见 `schema/capabilities/speculative_decoding.h:33-45`。头文件提供两个重载：一个接受 `std::istream&`，另一个接受文件路径并转调前者。转调代码见 `schema/capabilities/speculative_decoding.cc:81-88`。具体判断位于 `schema/capabilities/speculative_decoding.cc:40-78`：

```cpp
const std::vector<std::string> speculative_decoding_model_types = {
    "tf_lite_mtp_drafter"};                                      // (1)
// ...
if (section_object->data_type() == AnySectionDataType_TFLiteModel) {  // (2)
  // ...
  if (key->string_view() == "model_type") {                     // (3)
    // ...
    if (std::find(speculative_decoding_model_types.begin(),
                  speculative_decoding_model_types.end(),
                  value_string->string_view()) !=
        speculative_decoding_model_types.end()) {
      return true;                                               // (4)
    }
  }
}
```

`(2)` 遍历 `.litertlm` 的 section，`(3)` 读取 TFLite 模型 section 的 `model_type` 元数据。出现一项 `"tf_lite_mtp_drafter"`，`(4)` 即返回 true。能力判断取决于模型文件中是否包含相应 drafter section。第 7 章介绍的 `.litertlm` 容器可同时保存基础模型和这个子模型。

模型能力查询与自动启用在 v0.13.1 尚未连通。CLI 参数 `--enable-speculative-decoding` 接受 `auto`、`true`、`false`（`python/litert_lm_cli/common.py:108-119`）。`parse_speculative_decoding` 把 `auto` 和缺省值映射为 `None`，见 `python/litert_lm_cli/common.py:21-41`。另外两项映射为相应的布尔值。Python 绑定只在值非 `None` 时调用 setter（`python/litert_lm/engine.py:113-116`）。所以 `auto` 保留 C++ 默认值 `false`（`runtime/executor/llm_executor_settings.h:257-258`）。CLI help 写明会根据模型元数据自动判断，位置在 `python/litert_lm_cli/common.py:114-117`。v0.13.1 的引擎创建路径没有调用 `HasSpeculativeDecodingSupport`。该标志经 `runtime/engine/litert_lm_lib.cc:592` 写入执行器设置。执行器构造时据此决定是否创建 drafter（`runtime/executor/llm_litert_compiled_model_executor.cc:1807-1822`）：

```cpp
if (advanced_settings.has_value() &&
    advanced_settings->enable_speculative_decoding) {           // (1)
  // ...
  ASSIGN_OR_RETURN(mtp_drafter, LlmLiteRtMtpDrafter::Create(    // (2)
                                    lrt_env, resources, executor_settings,
                                    *compiled_model, *embedding_lookup,
                                    ple_manager_opt));
}
```

`(1)` 为真时才创建 drafter。`(2)` 的 `Create` 调用 `resources.GetTFLiteModel(ModelType::kTfLiteMtpDrafter)`，取出 drafter section。该 section 随后被编译为独立模型，见 `runtime/executor/llm_litert_mtp_drafter.cc:193-197`。模型文件不含该 section 时，创建过程返回错误。v0.13.1 也把 `HasSpeculativeDecodingSupport` 作为查询接口提供给调用方。C API 见 `schema/capabilities/capabilities_c.cc:45-55`。Kotlin JNI 见 `kotlin/java/com/google/ai/edge/litertlm/jni/litertlm.cc:1233-1236`。需要自动行为的应用必须先查询能力，再显式设置启用标志。

草拟步数 G 也不能在运行时调整。它由模型 verify signature 的形状决定（`runtime/executor/llm_litert_mtp_drafter.cc:252-256`）：

```cpp
LITERT_ASSIGN_OR_RETURN(auto input_pos_tensor_type,
                        verify_signature.InputTensorType("input_pos"));
// Expecred shape: [T = G + 1] where G is the number of draft steps
const auto& input_pos_dims = input_pos_tensor_type.Layout().Dimensions();
num_draft_steps = input_pos_dims[0] - 1;                         // (1)
```

`(1)` 用 verify signature 的 `input_pos` 第一维减一得到 G。verify 一次接收 G+1 个位置，其中一个是起始的已确认 token，其余 G 个对应草拟步数。signature 的张量形状在模型导出时固定，运行时只能读取。`CreateGreedySampler` 把 verifier 采样器的 `sequence_size` 设为 `num_draft_steps + 1`。对应代码见 `runtime/executor/llm_litert_mtp_drafter.cc:268-272`。采样器输出与 verify signature 对齐。

<div class="aside-compare">

llama.cpp 把 draft 模型保存为独立文件。调用方通过 `--model-draft` 在运行时指定文件，见 `llama.cpp/common/arg.cpp:3763 @ b9873`。草拟与验证逻辑位于 `common/speculative.cpp`。LiteRT-LM 的 MTP 把 drafter section 与基础模型放在同一个 `.litertlm` 文件中。基础模型还提供 verify signature。前者允许调用方在运行时选择兼容的 draft 模型；后者由模型发布者固定 drafter 与 verify 的组合。v0.13.1 可以查询模型能力，但不会据此自动启用 MTP，调用方仍需显式设置。

</div>

## 小结

LiteRT-LM 的 MTP 路径让基础模型一次 verify 前向确认多个草稿位置。稳态下，一轮 `Draft()` 接受 K 个草稿并返回 K+1 个 token。prefill 后首次 `Decode()` 还会执行一次普通 decode，因此返回 K+2 个 token。该次调用包含两次基础模型前向。当前实现采用贪心 token 比较，没有保持随机采样分布不变的概率接受步骤。

日志中的 `Success rate` 是聚合接受比例 `r̂=ΣK/(RG)`，样本平均产出为 `1+Gr̂`。理论曲线中的 p 是逐位条件匹配概率，不能用 r̂ 替代。端到端加速比还取决于归一化轮次成本 q。故事和代码样本分别得到 `r̂≈0.263` 与 `r̂≈0.995`，但计数与主 benchmark 吞吐来自不同负载。现有记录不能把吞吐差异分解为 r 和 q；完整测试需要在同一次运行中记录接受比例与吞吐。

---

## 练习与自查

1. 盈亏平衡：设 G=3、有效开销参数 c=0.1，逐位条件匹配概率为恒定 p。一轮期望产出为 `1+p+p²+p³`，归一化成本为 `1+3c`。求加速比达到 1 时的 p。
2. 返回下界：第一次草稿不匹配时，`Draft()` 为什么仍返回 1 个 token？这个结论为什么不等于“端到端成本不会增加”？
3. 形状约束：草拟步数 G 为什么在模型导出时固定？从 verify signature 的哪一个维度读出？
4. 口径换算：G=3、日志聚合比例 `r̂=0.4` 时，平均每轮接受多少草稿、返回多少 token？为什么不能把 0.4 直接代入理论曲线中的 p？
5. 结构对照：drafter 输入形状为 `[1, 1, 5120]`。5120 由哪两部分组成？它们分别来自哪里？

[^ch09-google-mtp]: Olivier Lacombe、Maarten Grootendorst，*Accelerating Gemma 4: faster inference with multi-token prediction drafters*，Google，2026-05-05，<https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/>（访问 2026-07-18）。

[^ch09-leviathan]: Yaniv Leviathan、Matan Kalman、Yossi Matias，*Fast Inference from Transformers via Speculative Decoding*，ICML 2023，arXiv:2211.17192，<https://arxiv.org/abs/2211.17192>（访问 2026-07-18）。

[^ch09-chen]: Charlie Chen、Sebastian Borgeaud、Geoffrey Irving、Jean-Baptiste Lespiau、Laurent Sifre、John Jumper，*Accelerating Large Language Model Decoding with Speculative Sampling*，2023，arXiv:2302.01318，<https://arxiv.org/abs/2302.01318>（访问 2026-07-18）。

[^ch09-issue-2227]: Shoolife，*MTP / speculative decoding regresses decode tok/s on PowerVR GPU (Tensor G6) — even with GPU sampler fully loaded*，LiteRT-LM issue #2227，2026-05-11，<https://github.com/google-ai-edge/LiteRT-LM/issues/2227>（访问 2026-07-18）。
