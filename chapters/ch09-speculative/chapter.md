# 第 9 章 一次前向，多个 token：投机解码与 MTP

> 本章说明 MTP 如何让 drafter 草拟多个 token，再由基础模型一次验证。分析接受比例与每轮运行成本如何共同决定加速比。

第 1 章给出缓解 decode 内存带宽约束的两条思路：提高有效带宽，或减少每 token 读取的字节数；量化属于第二条。投机解码（speculative decoding）是减少每 token 字节数的另一种办法：基础模型每读取一次权重，尽可能确认多个 token。本章据此定量分析第 2 章表 2-2 的问题 19：投机解码为什么可能加速，又会在什么条件下减速。前半章（9.2 至 9.7 节）按代码依次说明草拟、验证、接受与集成，后半章（9.8 至 9.10 节）计算加速比、处理多 token 返回带来的停止检测问题，并说明开启条件。

## 9.1　投机解码的目标：一次验证多个候选 token

稠密模型在 decode 阶段通常每生成一个 token 就读取一次模型权重，一次前向只确定一个 token。投机解码要改变的正是这个比例：让每次权重读取确认尽可能多的 token。

做法分两步：先由低成本的 drafter 草拟若干 token，再由基础模型（base 模型）用一次前向验证全部候选。比对之后，最长的匹配前缀全部接受，第一个不匹配的位置改用基础模型自己给出的 token。这次验证前向和普通 decode 前向一样只读一遍权重，一轮若接受多个 token，读权重的成本便由它们分摊。

投机解码有多种实现，差别主要在 drafter 的来源：它可以是一个独立的小模型，也可以是与基础模型联合训练的预测头，即多 token 预测（Multi-Token Prediction，MTP）头。Google 发布的 Gemma 4 MTP drafter 使用基础模型的 activation，并与基础模型共享 KV cache。[^ch09-google-mtp] LiteRT-LM 在运行时把这个 drafter 装载为独立模型；验证则使用基础模型的 verify signature，即基础模型导出时附加的一个固定形状调用入口（signature 的定义见 1.1 节）。

## 9.2　机制：串行草拟，批量验证

一次 `Draft()` 调用包含三个阶段：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:461-468
ASSIGN_OR_RETURN(std::vector<int> drafted_tokens,
                 RunDraftingLoop(token_id, activations));          // (1)

RETURN_IF_ERROR(PrepareVerifierInputBuffers(
    position, token_id, drafted_tokens, input_kv_cache_buffers));  // (2)
// ...
ASSIGN_OR_RETURN(std::vector<int> verifier_id_vector, RunVerification());  // (3)
```

代码行 `(1)` 草拟 \\(G\\) 个 token；`(2)` 把上一个已确认 token 和 \\(G\\) 个草稿拼成 verify 输入；`(3)` 由基础模型一次完成验证。三步之后，接受循环再比较两组 token。

第一步是草拟：drafter 逐个生成接下来的 \\(G\\) 个候选 token，\\(G\\) 是草拟步数（代码里的 `num_draft_steps_`）。循环每次生成一个 token：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:336-370
for (int i = 0; i < num_draft_steps_; ++i) {                      // (1)
  // ...
  // Concatenated embedding + activation has shape [B = 1, T = 1, D = 3072]
  // ...
  RETURN_IF_ERROR(embedding_manager_.LookupDecode(last_drafted_token_id,
                                                  embedding_vector));
  // ...
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

代码行 `(1)` 循环 \\(G\\) 次，每次串行生成一个 token。`(2)` 拼接本步输入，由词嵌入与上一步的隐藏态 activation 拼接组成：源码注释以 `[B=1, T=1, D=3072]` 为例，即 1536 维词嵌入加 1536 维 activation；具体维度随模型而定，本书基准模型的 drafter 输入为 `[1, 1, 5120]`，由 2560 维词嵌入和 2560 维 activation 组成，两部分都与主干的 `model_dimension = 2560` 一致（见附录 D）。`(3)` 每步运行的是独立装载的 drafter 模型，而非基础模型。`(4)` 断言每步只产出一个 token，`(5)` 再把本步输出作为下一步输入，drafter 仍按自回归方式生成。可见草拟本身有开销：一轮的总成本包括 \\(G\\) 次 drafter 前向和一次 verify 前向，MTP 是否加速取决于这个总成本，不能只看 drafter 的模型尺寸。

第二步是批量验证：运行时把上一个已确认 token 和 \\(G\\) 个草稿传给基础模型的 verify signature：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:439-450
LITERT_RETURN_IF_ERROR(base_model_.RunAsync(                       // (1)
    verify_signature_.Key(), active_verifier_input_buffers_,
    active_verifier_output_buffers_, async));
// ...
LITERT_ASSIGN_OR_RETURN(auto id_vector,
                        CopyFromTensorBuffer<int32_t>(verifier_id_tensor_));
RET_CHECK_EQ(id_vector.size(), num_draft_steps_ + 1);             // (2)
return id_vector;
```

代码行 `(1)` 这次 `RunAsync` 是整个 `Draft()` 调用内唯一的基础模型前向，\\(G\\) 个草稿位置在其中一起验证。`(2)` 返回长度为 \\(G+1\\) 的 id 序列，每个位置对应一个基础模型输出，最后一个位置留给 \\(G\\) 个草稿全部匹配的情形。“唯一”的范围仅限一次 `Draft()` 之内：prefill 后首次调用 `Decode()` 还会先执行一次普通 decode（见 9.7 节）。

第三步是接受：逐位比对草拟结果与验证结果，接受最长的匹配前缀。

<figure>
{{#include figs/fig-9-1.svg}}
<figcaption>图 9-1　一次 Draft() 调用执行 G 次 drafter 前向和一次基础模型 verify，返回匹配前缀及一个 bonus token，即 1 到 G+1 个 token；首次 Decode() 还包含一次普通 decode。</figcaption>
</figure>

## 9.3　接受循环：匹配前缀与 bonus token

第三步的接受与回退由 `Draft()` 末尾的接受循环完成：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:471-484
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

代码行 `(1)` 从头逐位比较草拟结果与验证结果：位置一致时 `(3)` 把 `num_correct_tokens` 加一；遇到第一个不一致的位置，`(2)` 停止比较，并把基础模型在该位置的输出记为 bonus token。若循环结束后 `(4)` 发现 `bonus_token` 仍为 −1，说明 \\(G\\) 个草稿全部匹配，`(5)` 此时取 verify 输出的第 \\(G+1\\) 个 token 作为 bonus。两条路径的结果一致：无论草稿是否全部匹配，基础模型本轮都会提供一个不来自已接受草稿的输出 token。

循环里同步维护的 `last_verified_token_id_idx_` 不参与接受比例统计，它保存的是 verify 输出中最后一个有效位置的下标：下一轮草拟从 verifier 缓冲取隐藏态时，用这个下标读取相应 activation 作为起点（隐藏态的两条来源见 9.6 节）。

`Draft()` 最后返回接受前缀和一个 bonus token：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:491-493
// The first token comes from the decode output and is always correct.
drafted_tokens.resize(num_correct_tokens);   // (1)
drafted_tokens.push_back(bonus_token);       // (2)
```

代码行 `(1)` 把草拟序列截断到接受长度，`(2)` 再追加 bonus。源码注释里的“always correct”说的是：bonus token 来自基础模型输出，采用它不以 drafter 匹配为前提。即使 `num_correct_tokens` 为 0，`resize(0)` 后的 `push_back` 也保证返回 1 个 token。

所以一次 `Draft()` 至少返回 1 个 token。不过这只是每轮输出数量的下界，不是运行成本的上界：即使一个草稿都没被接受，该轮也已经执行了 \\(G\\) 次 drafter 前向和缓冲操作；prefill 后的首次 `Decode()` 另有一次普通基础模型前向，同样不在这个稳态口径之内。

## 9.4　采样约束：当前 MTP 路径采用贪心接受

接受循环用的是严格相等比较，这与经典推测采样不同：Leviathan 等（2023）[^ch09-leviathan] 与 Chen 等（2023）[^ch09-chen] 都在接受阶段引入概率步骤（rejection sampling 及其修正形式），使输出分布与目标模型（即本章的基础模型）保持一致。LiteRT-LM 当前的 MTP 路径没有这一步，drafter 与 verifier 均采用贪心采样，接受条件就是两个 token id 相等。

贪心设置位于构造采样器的辅助函数里：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:68-77
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

代码行 `(1)` 把 top-k 设为 1，`(2)` 把 top-p 设为 0，候选集中只保留 argmax。`(3)` 的 temperature 为 1.0，但在 k=1 时不会引入随机选择。drafter 与 verifier 共用这个函数，只是 `sequence_size` 不同：drafter 为 1，verifier 为 \\(G+1\\)。接受阶段比较的因而是两侧 logits 的 argmax token。

要保持采样分布不变，verifier 需要提供各位置的完整概率分布：接受阶段按概率比值决定是否采用草稿，拒绝后还要从残差分布重新采样。LiteRT-LM 的 verifier 只回传形状为 `[1, G+1]` 的 token id，接受循环随之只做整数比较。这省去了完整分布的回传和逐位处理，代价是适用范围收窄到贪心比较：即使设置 temperature 或 top-p，MTP 路径也无法保持非推测路径的采样分布。

## 9.5　验证输入的构造：位置、mask 与 KV cache 的复制

verify 使用包含 \\(G+1\\) 个位置的批量输入，而不是普通 decode 的单位置输入。构造这组缓冲的函数执行四个步骤：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:383-415
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

代码行 `(1)` 把 `input_pos` 依次写为 `position` 到 `position+G`，这 \\(G+1\\) 个位置对应一个已确认 token 和随后 \\(G\\) 个草稿。`(2)` 填充 attention mask，起点为 `position`、步数为 \\(G+1\\)：因果 mask 使每个待验证位置只能读取此前的前缀，包括已确认 token 和更早的草稿位置，第 `i` 个位置的 logits 因而与逐 token decode 到该位置时具有相同的可见前缀。`(3)` 用 prefill 路径的批量查表取得 `[已确认 token, 草稿_1, …, 草稿_G]` 的 embeddings。`(4)` 为每个 KV cache 输入复制 `TensorBuffer` 句柄，供 verify 使用。

`Duplicate()` 复制的是缓冲句柄而不是底层数据（第 8 章讨论过这种浅复制语义）。复制的原因在于：verify 前向要在现有 KV cache 之后追加 \\(G+1\\) 个位置，而第 6 章介绍的双缓冲路径要求读写两侧分别传入句柄。单缓冲 KV cache 则由紧随其后的分支处理：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:416-420
if (active_verifier_input_buffers_.contains("param_tensor")) {   // (1)
  RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
      active_verifier_input_buffers_["param_tensor"], position,
      num_draft_steps_ + 1));
}
```

代码行 `(1)` 处理 6.3.1 节的单缓冲路径：读写位置放在参数张量里，这里从 `position` 开始填入 \\(G+1\\) 步对应的参数，使 verify 写入相应的缓存区间。两条路径至此对齐：双缓冲传入复制后的读写句柄，单缓冲额外设置参数张量。输出侧由另一个函数处理，复制各个 KV cache 输出句柄，并清除旧的完成事件。

## 9.6　drafter 的隐藏态拼接：两条 activation 来源

MTP drafter 每步都把词嵌入与隐藏态拼接后输入模型。草拟循环的以下分支决定隐藏态从哪里来：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:346-353
if (activations_ptr) {
  RETURN_IF_ERROR(ConcatenateEmbeddingsAndActivations(              // (1)
      embedding_vector, *activations_ptr, *drafter_activations_buffer));
} else {
  RETURN_IF_ERROR(ConcatenateEmbeddingsAndActivationsFromVerifierBuffer( // (2)
      embedding_vector, verifier_output_buffers_["activations"],
      last_verified_token_id_idx_, *drafter_activations_buffer));
}
```

代码行 `(1)` 在 `activations_ptr` 非空时拼接参数传入的 activations。调用方传入 activations 时，第一步执行的就是这条路径，该 activation 来自基础模型此前的 decode 前向；第一步结束后，`activations_ptr` 改指 drafter 上一步输出的 `projected_activations`，因此后续各步总是执行分支 `(1)`，按自回归方式继续。`(2)` 只在第一步且未传入 activations 时执行：隐藏态取自上一轮的 verifier 输出缓冲，由接受循环维护的 `last_verified_token_id_idx_` 选出被接受位置对应的 activation。

两条来源对应两种调用时机：prefill 后的首次 decode 传入普通 decode 输出的 activation，执行路径 `(1)`；进入稳态后不再传入，drafter 改用上一轮 verify 中被接受位置的 activation，即路径 `(2)`。无论哪条路径，drafter 的输入都是上一个 token 加基础模型产生的上下文表示。本章没有比较其他 drafter 结构，不能据此推断相对命中率。

拼接本身很简单：先把词嵌入复制到输出缓冲前半段，再把 activation 复制到后半段，两段各含 `model_dimension` 个 float，组成 drafter signature 的双倍宽度输入（本书基准模型为 2560 + 2560 = 5120，记录见附录 D）。这段代码只执行两次内存复制、不含矩阵计算，但它在端到端时延中的占比仍需测量，不能仅凭代码结构判定为可忽略。

## 9.7　集成：`Draft()` 的调用与 token 回写

执行器的 `Decode()` 负责把 drafter 接入推理流程：没有装载 drafter 时执行普通 decode，否则进入 MTP 路径，并按当前调用是否为 prefill 后的第一次 decode 再分两支。稳态分支如下：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:1030-1043
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

代码行 `(1)` 的 `ran_decode` 表示上一次运行是否为 decode，进入稳态后执行该分支。`(2)` 将 activations 设为空，即 9.6 节的路径 `(2)`：drafter 从 verifier 缓冲读取上一轮被接受位置的隐藏态。`(3)` 按 `Draft()` 本轮返回的 token 数增加 `current_step`，该数等于接受前缀长度加 1 个 bonus token。稳态下，一次 `Decode()` 直接返回这组 token，数量在 1 到 \\(G+1\\) 之间。

另一条分支处理 prefill 后的第一次 decode。此时运行时先执行普通 decode，取得首个 token 及其 activation，再调用 `Draft()`：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:1057-1079
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

代码行 `(1)` 普通 decode 先采样出 `token_id`。`(2)` 复制该次 decode 输出的 activations 并传给 `Draft()`，对应上一节的第一条 activation 来源。`(3)` 传入 `current_step - 1`：此时普通 decode 已把 `current_step` 加过一次，而草拟位置要从刚得到的 token 起算。`(4)` 最后把这个普通 decode token 插到 `Draft()` 返回序列之前，于是第一次 `Decode()` 返回 2 到 \\(G+2\\) 个 token：1 个来自普通 decode，1 到 \\(G+1\\) 个来自 `Draft()`。

首轮与稳态的差别还体现在基础模型调用次数上：第一次 `Decode()` 执行一次普通 decode 加一次 verify，共两次基础模型前向；稳态的每次 `Decode()` 只执行一次 verify；两者的 drafter 前向都是 \\(G\\) 次。分析长期吞吐通常采用稳态口径，但测量短输出时，首轮多出的那次基础模型前向不能省略。

## 9.8　接受比例与加速比

MTP 是否加速，取决于每轮产出的 token 数和该轮的运行成本。先看产出怎么计数。`Draft()` 在每轮结束时累加两个计数：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:494-495
num_drafted_tokens_ += num_draft_steps_;    // (1)
num_verified_tokens_ += num_correct_tokens; // (2)
```

代码行 `(1)` 每轮把草稿总数增加 \\(G\\)，`(2)` 只累加匹配前缀的长度，不含 bonus token。drafter 析构时输出这两个计数及其比值：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:165-173
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

设共执行 \\(R\\) 轮 `Draft()`，第 \\(j\\) 轮接受的草稿数为 \\(K_j\\)，且 \\(0 \le K_j \le G\\)。代码行 `(1)` 打印的 `Success rate` 是聚合接受比例：

$$ \widehat r = \frac{\sum_{j=1}^{R} K_j}{R G} $$

这个量直接给出样本中的平均接受数 \\(G\widehat r\\)。每轮还会返回 1 个 bonus token，所以稳态下 `Draft()` 的样本平均产出为：

$$ \overline N = 1 + G\widehat r $$

日志中的 \\(\widehat r\\) 不是“每个位置独立匹配的概率”。理论曲线另设参数 \\(p\\)：给定前 \\(k-1\\) 个草稿均已匹配，第 \\(k\\) 个草稿继续匹配的条件概率假定恒为 \\(p\\)。此时 \\(P(K \ge k) = p^k\\)，于是：

$$ E[N] = 1 + E[K] = 1 + \sum_{k=1}^{G} p^k $$

这两个口径满足 \\(r = E[K]/G = \bigl(\sum_{k=1}^{G} p^k\bigr)/G\\)，不能把日志中的 \\(\widehat r\\) 直接代入 \\(p\\) 的位置。真实数据还可能随位置变化，未必符合恒定 \\(p\\) 的假设。

再看成本。归一化轮次成本 \\(q\\) 是两个墙钟时间之比：分子是一次稳态 MTP 轮次，分母是一次普通 decode step。端到端加速比可写成：

$$ \text{speedup} \approx \frac{1 + G r}{q} $$

图 9-2 采用简化成本模型 \\(q = 1 + Gc\\)。其中 \\(c\\) 是归一化的有效开销参数，不是直接测得的 drafter 单步前向时间：它合并了 verify 与普通 decode 的成本差异、\\(G\\) 次 drafter 前向和缓冲操作。结合恒定条件概率 \\(p\\) 的假设，得到：

$$ \text{speedup}(p,c) \approx
\frac{1 + \sum_{k=1}^{G} p^k}{1 + Gc} $$

取 \\(G=3\\)、\\(c=0.15\\)，分母为 1.45。\\(p=0.8\\) 时，期望产出为 2.952 个 token，加速比约 2.04；\\(p=0.4\\) 时约为 1.12；\\(p=0.2\\) 时约为 0.86。令加速比等于 1，解得盈亏平衡点 \\(p\approx 0.317\\)，而不是 0.25。这些数值只说明理论曲线的形状，不代表本书设备上的实测成本。

<figure>
{{#include figs/fig-9-2.svg}}
<figcaption>图 9-2　理论曲线使用逐位条件匹配概率 p；在 G=3、c=0.15 的示意条件下，盈亏平衡点约为 0.32。日志聚合比例 r̂ 不能直接作为横轴 p。</figcaption>
</figure>

曲线的两端各说明一种情况。聚合接受比例 \\(r\\) 增大时，平均每轮产出 \\(1+Gr\\) 随之增加，但分母上的轮次成本 \\(q\\) 同样起作用，\\(r\\) 再高，较大的 \\(q\\) 仍会降低加速比；反过来，\\(r\\) 较低时一轮可能只返回 1 个或少量 token，drafter 与 verify 的成本却不因此减少，吞吐甚至可能低于普通 decode。Google 报告 Gemma 4 MTP drafter 在其跨模型、硬件与运行时测试中最高达到约 3 倍[^ch09-google-mtp]；该数字不是本书设备上的预期值。

本书的实测记录分三类，各自能回答的问题不同。第一类是 Mac 上的两组主基准记录（Gemma 4 E4B、context 1024、decode 128 token），参数分别设为 `false` 与 `auto`。这两组并不构成开关对照：9.10 节核对的设置链路表明，`auto` 最终沿用 C++ 默认值 `false`，两组都是关闭 MTP 后的独立采样。CPU 中位数分别为 22.8 和 24.9 tokens/s（`false` 组的三次运行分布在 20.1 到 24.9 tokens/s），GPU 中位数为 50.0 和 50.2 tokens/s。两组之间的差异只反映运行波动，估计不了 MTP 的开关收益；可核查的 Mac 强制开启记录目前没有。

| 模式 | cpu decode tokens/s | gpu decode tokens/s | 说明 |
|---|---|---|---|
| 关（`false`） | 22.8 | 50.0 | 基线 |
| `auto` | 24.9 | 50.2 | 实为关，与基线同行为的再采样 |

> 表 9-1　Mac 归档记录中的 `false` 与 `auto` 均为关闭行为，各列为 3 次运行的中位数〔基准 D〕；该表不能用于估计 MTP 收益。

第二类是接受比例的实测样本。附录 D 记录了 drafter 析构时输出的计数器，实验通过 Python SDK 把日志级别设为 VERBOSE。故事提示词产生 213 个草稿、56 个匹配，\\(\widehat r=56/213\approx 0.263\\)，平均每次 `Draft()` 返回 \\(1+3\widehat r\approx 1.79\\) 个 token；代码提示词产生 3069 个草稿、3054 个匹配，\\(\widehat r\approx 0.995\\)，平均返回约 3.99 个 token。在图 9-2 的示意假设 \\(q=1.45\\) 下，两个样本对应的加速比分别约为 1.23 和 2.75；若把 \\(\widehat r\\) 误作 \\(p\\) 代入，会得到 0.93，那是两种口径的混用。这些只是示意值：计数实验没有同时测量 \\(q\\)，得不出实测速率；两次实验的提示词也与固定长度 benchmark 不同，故事样本的 \\(\widehat r\\) 推断不了该 benchmark 开启 MTP 后的吞吐。

差别源自负载的构造方式。固定长度 benchmark 的输入构造很直接：代码先对提示词分词，再把 id 序列调整到指定长度，N 大于原 token 数时新增的元素全为 0，更小时序列被截断。这样的负载与自然文本差别很大，前述两个 \\(\widehat r\\) 都不适用。Mac 归档数据既没有强制开启组，也没有这组 benchmark 的 drafter 计数器；可核查的强制开启端到端对照只有 Android 记录，而其中的减速同样没有证据归因于低接受比例。

复现这组计数不需要修改或重新编译运行时：drafter 本来就维护这两个计数并在析构时经 `ABSL_LOG(INFO)` 输出，用 Python SDK 把日志级别设为 VERBOSE 即可看到。局限也在这里：日志只提供整个 drafter 生命周期内的聚合比例，要得到分阶段或逐轮数据，就需要新增遥测接口。

第三类是强制开启后的端到端对照，方向并不一致。上游有减速报告：`LiteRT-LM#2227` 记录了 PowerVR GPU 上开启 MTP 后 decode 吞吐下降的案例，环境为 LiteRT-LM 0.11.0、Gemma 4 E2B 和俄文分类负载，其中关于 GPU 路径的原因分析明确标为假设。[^ch09-issue-2227] 对照本章公式，减速只有两类可能条件：\\(r\\) 偏低，或归一化轮次成本 \\(q\\) 偏高；公式区分不了二者，也确认不了该 issue 的根因。本书基准数据里也有一例：附录 D 第十三节记录了一台 Qualcomm 机型在合成负载（prefill 1024 个 token）下的 MTP 减速，强制开启后 CPU decode 从 10.0 降到 2.8 tokens/s，GPU 从 18.0 降到 12.6 tokens/s。这次实验同样没有同步记录 \\(r\\) 或分解 \\(q\\)，原因归不到其中任何一个参数上。

同一台手机在自然代码提示词下的方向则相反：GPU 单次观测从 16.0 变为 32.0 tokens/s（decode 长度 128），CPU 从 11.6 变为 12.6 tokens/s（decode 长度 64）〔基准 D〕。不过每个条件只运行 1 次，两种后端的 decode 长度也不同，这些记录只能说明测试中存在观测差异，估计不了稳定加速比，也比较不了后端绝对值；它们与 \\(\widehat r\\) 计数不在同一次运行里，同样分解不出接受比例和轮次成本各自的贡献。

增速一侧还有另两条记录。本书基准模型的 verify signature 中 `input_pos` 形状为 `[4]`，故 \\(G=3\\)，一次 `Draft()` 最多返回 4 个 token。附录 D 的两组自然代码长生成单次记录里，Mac GPU 从 58.7 变为 133.9 tokens/s（比值 2.28），手机 GPU 从 18.62 变为 37.38 tokens/s（比值 2.01）。这两组使用不同采集入口，没有重复运行，也没有在同一次运行中记录 \\(r\\)，吞吐比因而确定不了 \\(q\\) 或 \\(c\\)，也估计不了跨设备的稳定差异。

这两个比值可以用来反推口径。在 \\(r=1\\)、\\(q=1+3c\\) 的假设下，2.28 倍和 2.01 倍分别对应 \\(c\approx 0.25\\) 和 \\(c\approx 0.33\\)。注意这个 \\(c\\) 是由端到端结果反推的有效参数，混杂了 verify、drafter 和运行时操作，不是 drafter 模型的实测单步成本；若实际 \\(r\\) 小于 1，反推出的 \\(c\\) 还会更小。作为参照，同一简化模型下即使 \\(r=1\\)，3 倍加速也要求 \\(c\le 1/9\approx 0.11\\)。所以这些数值既不能证明两台设备已达吞吐上限，也不足以把官方结果[^ch09-google-mtp] 归因于某种硬件路径；drafter 文件约 45 MB 同样推不出 \\(c\approx 0.02\\)，模型存储大小与端到端时延不成比例。

投机解码是否加速由 \\(r\\) 和 \\(q\\) 共同决定：聚合接受比例 \\(r\\) 决定每轮平均产出，归一化成本 \\(q\\) 决定取得这些产出要花的时间，而两者都随模型、输入内容、生成位置、设备和后端变化。评估目标负载时，需要在同一次测试中记录接受比例与吞吐。

## 9.9　多 token 返回后的停止检测与回退

MTP 还改变了 executor 的返回形态。普通 decode 通常返回一个 token，一次 MTP `Decode()` 则可能返回一段序列。任务层必须按顺序处理这段序列，因为停止序列可能在批次中间命中，BPE 片段也可能跨越两次 MTP 调用。

任务层的单步运行函数先取得 executor 返回的二维 token 序列，检查各候选的序列长度是否相同，随后按位置逐个处理：

```cpp
// runtime/core/tasks.cc:163-178
for (size_t step = 0; step < sequence_length; ++step) {             // (1)
  std::vector<std::vector<int>> step_tokens;
  // ...
  RETURN_IF_ERROR(stop_token_detector_.ProcessTokens(step_tokens)); // (2)
  ASSIGN_OR_RETURN(step_tokens, tokenizer_.MergeTokenIds(           // (3)
                                    bpe_partial_token_ids_, step_tokens));
  auto decoded_result =
      tokenizer_.TokenIdsToTexts(num_output_candidates_, step_tokens);
```

代码行 `(1)` 将本轮返回序列展开为逐位置处理，省略的几行从每个候选取出当前位置的 token，填入 `step_tokens`。`(2)` 先推进停止序列检测器，`(3)` 再合并未完成的 BPE token id。即使 executor 一次返回多个 token，停止检测与文本解码仍按 token 顺序推进。

停止序列在批次中间命中时，代码会调整 executor 的逻辑位置。回退量按 `sequence_length - step` 计算：

```cpp
// runtime/core/tasks.cc:225-234
if (all_done) {
  if (step != sequence_length - 1) {
    // ...
    int diff = sequence_length - step;                               // (1)
    ASSIGN_OR_RETURN(int current_step, executor_.GetCurrentStep());
    RETURN_IF_ERROR(executor_.SetCurrentStep(current_step - diff));   // (2)
  }
  return true;
}
```

假设本轮返回 4 个 token、停止 token 位于下标 1：可见输出只保留下标 0，代码行 `(1)` 算出 `diff=4-1=3`，`(2)` 回退停止位置及其后的两个位置。停止 token 恰好位于本批末位时则不同，`if` 条件不成立，代码不调用 `SetCurrentStep`；批中回退规则因而不能推广到批次末位。

compiled executor 的 `SetCurrentStep` 检查新位置不超过已处理 token 数且不小于 0；检查通过后只更新逻辑 `current_step`，不清除 KV cache 缓冲，后续调用读取的就是调整后的逻辑位置。

<figure>
{{#include figs/fig-9-3.svg}}
<figcaption>图 9-3　示例中的停止 token 位于 4-token 批次的下标 1，任务层保留下标 0，并将逻辑位置回退 3；命中批次末位时不执行该回退分支。</figcaption>
</figure>

单步运行函数在每次开始时清空结果，再累积本批可见文本和 token id。外层循环在它返回后至多发送一次 `TaskState::kProcessing` 回调，而且只有确有更新时才发送。一次 executor `Decode()` 可能不产生可见更新，也可能产生一次携带多个 token id 的更新，于是回调次数既不等于 executor 调用次数，也不等于生成 token 数。统计可见输出时应读取回调中的 token id；统计 executor 的位置推进量时应读取 `current_step` 差值。

| 终止条件 | 检查位置 | 对本轮多 token 序列的处理 |
|---|---|---|
| 停止 token 或停止序列 | 单步运行函数内逐 token 检查 | 批中命中时回退停止位置及后续位置；末位命中不回退 |
| benchmark decode token 数量 | 单步返回后调用 `ShouldStop` | 任务层没有按剩余额度截短本批 |
| KV cache 最大长度 | 单步返回后调用 `ShouldStop` | 使用本轮后的绝对 `current_step` 判断 |
| `max_output_tokens` | 单步返回后调用 `ShouldStop` | 使用本轮后的 `current_step` 增量判断 |
| 取消标志 | 下一轮开始之前检查 | 已完成并回调的上一批不会被撤销 |

> 表 9-2　停止序列在批内逐 token 检查；三个数值上限在整批返回后检查，取消标志则在下一批开始前检查。

外层循环在回调之后读取 `current_step`，并以它相对 decode 起点的增量作为已生成步数。`ShouldStop` 再将这个增量与 benchmark 数量、`max_output_tokens` 比较，并将绝对 `current_step` 与 KV cache 上限比较。这里的 step 数是 token 位置增量，不是单步运行函数的调用次数。

据此可以给出任务层的上界推断。若某项数值限制只剩 \\(B\\) 个 token，而本轮在没有提前命中停止序列时推进 \\(L\\) 个位置，且 \\(L>B\\)，任务层将在回调后停止，越过阈值 \\(L-B\\) 个位置。稳态 MTP 一轮最多返回 \\(G+1\\) 个 token，任务层的理论最大越界量为 \\(G\\)；prefill 后第一次 `Decode()` 最多返回 \\(G+2\\) 个 token，对应 \\(G+1\\)。这个推断来自后置检查和 `current_step` 增量，前提是 executor 能完成该次调用。KV cache 容量还可能由 executor 或模型形状提前约束，不能仅凭任务层代码断言真实调用一定越界。

现有 `max_output_tokens` 测试使用一次返回一个 token 的假执行器，没有覆盖一次返回多 token 的边界；补齐它需要新增可为单个候选返回 token 序列的任务层测试 executor，再用真实 MTP 模型核对执行器状态：

| 输入序列与条件 | 应检查的结果 | 用途 |
|---|---|---|
| `[a, stop, b, c]`，停止 token 位于下标 1 | 只输出 `a`，最终位置为批前位置加 1 | 验证批中命中及 `diff=3` 的回退 |
| `[a, b, c, stop]`，停止 token 位于末位 | 可见输出、最终位置和 `SetCurrentStep` 调用次数 | 覆盖末位不回退分支 |
| 上一批末尾是停止序列前缀，本批首 token 完成匹配 | pending 队列和跨批停止行为 | 验证批边界不改变停止序列语义 |
| 剩余 `max_output_tokens=1`，executor 返回 4 个 token | 输出长度、回调 token id 和最终位置 | 验证后置检查与越界量 |
| 4 个 token 均可见且启用 streaming | 回调次数及单次回调的 token id 数 | 验证一个批次至多一次可见更新 |

> 表 9-3　多 token 返回测试需同时检查可见文本、回调 token id、逻辑位置和回退调用；只检查最终字符串不足以覆盖状态语义。

一次 executor `Decode()` 是一轮运行调用，`current_step` 增量是本轮推进的 token 位置数；MTP 性能分析用调用次数计算轮次成本，用位置增量计算产出和数值终止条件。流式接口还要区分可见 token 与因 BPE 或停止前缀而暂存的 token。

## 9.10　开启条件：模型能力与固定草拟步数

开启 MTP 要满足两个条件：模型文件同时包含 MTP drafter 和相应的 verify signature，且调用方显式启用投机解码。草拟步数 \\(G\\) 则由 verify signature 的形状固定。

第一个条件由 `HasSpeculativeDecodingSupport` 检查。它有两个重载，一个接受输入流，另一个接受文件路径并转调流式重载；判断本身只看模型文件的目录：

```cpp
// schema/capabilities/speculative_decoding.cc:40-75
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

代码行 `(1)` 列出的能力类型只有 `tf_lite_mtp_drafter` 一种。`(2)` 遍历 `.litertlm` 的 section，`(3)` 读取 TFLite 模型 section 的 `model_type` 属性，出现一项 `tf_lite_mtp_drafter` 时 `(4)` 即返回 true。能力判断因此只看模型文件里有没有相应的 drafter section；第 7 章介绍的 `.litertlm` 容器正好可以把基础模型和这个子模型放在同一个文件里。

第二个条件目前有一处与帮助文档不符的行为：能力查询与自动启用尚未关联。CLI 参数 `--enable-speculative-decoding` 接受 `auto`、`true`、`false`，help 写明 `auto` 会根据模型元数据自动判断；但逐级核对设置链路可以发现并非如此。CLI 把 `auto` 与缺省值都映射为空值、另外两项映射为相应布尔值，Python 绑定只在值非空时调用 setter，于是 `auto` 保留 C++ 默认值 `false`；引擎创建路径也没有调用 `HasSpeculativeDecodingSupport`。标志最终写入执行器设置，执行器构造时据此决定是否创建 drafter：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:1810-1822
if (advanced_settings.has_value() &&
    advanced_settings->enable_speculative_decoding) {           // (1)
  // ...
  ASSIGN_OR_RETURN(mtp_drafter, LlmLiteRtMtpDrafter::Create(    // (2)
                                    lrt_env, resources, executor_settings,
                                    *compiled_model, *embedding_lookup,
                                    ple_manager_opt));
}
```

代码行 `(1)` 为真时才创建 drafter。`(2)` 的 `Create` 从模型资源里取出 drafter section，把它编译为独立模型；模型文件不含该 section 时，创建过程返回错误。查询接口本身是可用的：C API 和 Kotlin JNI 都提供 `HasSpeculativeDecodingSupport`，需要自动行为的应用应当先查询能力，再显式设置启用标志。

草拟步数 \\(G\\) 也不能在运行时调整，它由模型 verify signature 的形状决定：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:252-256
LITERT_ASSIGN_OR_RETURN(auto input_pos_tensor_type,
                        verify_signature.InputTensorType("input_pos"));
// Expecred shape: [T = G + 1] where G is the number of draft steps
const auto& input_pos_dims = input_pos_tensor_type.Layout().Dimensions();
num_draft_steps = input_pos_dims[0] - 1;                         // (1)
```

代码行 `(1)` 用 verify signature 的 `input_pos` 第一维减一得到 \\(G\\)：verify 一次接收 \\(G+1\\) 个位置，其中一个是起始的已确认 token，其余 \\(G\\) 个对应草拟步数。signature 的张量形状在模型导出时固定，运行时只能读取；构造采样器时也据此把 verifier 采样器的 `sequence_size` 设为 \\(G+1\\)，使采样器输出与 verify signature 对齐。

<div class="aside-compare">

llama.cpp 把 draft 模型保存为独立文件，调用方通过 `--model-draft` 在运行时指定，草拟与验证逻辑在独立的 speculative 模块中。[^ch09-llamacpp-draft] LiteRT-LM 的 MTP 则把 drafter section 与基础模型放进同一个 `.litertlm` 文件，verify signature 也由基础模型提供。独立文件让调用方可以在运行时选择兼容的 draft 模型，单文件则由模型发布者固定 drafter 与 verify 的组合。LiteRT-LM 可以查询模型能力，但不会据此自动启用 MTP，调用方仍需显式设置。

</div>

## 小结

LiteRT-LM 的 MTP 路径让基础模型用一次 verify 前向确认多个草稿位置：稳态下一轮 `Draft()` 接受 \\(K\\) 个草稿、返回 \\(K+1\\) 个 token；prefill 后首次 `Decode()` 还会先执行一次普通 decode，返回 \\(K+2\\) 个 token，代价是两次基础模型前向。当前实现采用贪心 token 比较，没有保持随机采样分布不变的概率接受步骤。

日志中的 `Success rate` 是聚合接受比例 \\(\widehat r=\sum K_j/(RG)\\)，对应样本平均产出 \\(1+G\widehat r\\)；理论曲线中的 \\(p\\) 是逐位条件匹配概率，两者不能互替。端到端加速比还取决于归一化轮次成本 \\(q\\)。故事和代码样本分别得到 \\(\widehat r\approx 0.263\\) 与 \\(\widehat r\approx 0.995\\)，但计数与主 benchmark 吞吐来自不同负载，现有记录分解不出 \\(r\\) 和 \\(q\\) 各自的贡献；完整测试需要在同一次运行中记录接受比例与吞吐。

---

## 练习与自查

1. 盈亏平衡。设 \\(G=3\\)、有效开销参数 \\(c=0.1\\)，逐位条件匹配概率为恒定 \\(p\\)。一轮期望产出为 \\(1+p+p^2+p^3\\)，归一化成本为 \\(1+3c\\)。求加速比达到 1 时的 \\(p\\)。
2. 返回下界。第一次草稿不匹配时，`Draft()` 为什么仍返回 1 个 token？这个结论为什么不等于“端到端成本不会增加”？
3. 形状约束。草拟步数 \\(G\\) 为什么在模型导出时固定？从 verify signature 的哪一个维度读出？
4. 口径换算。\\(G=3\\)、日志聚合比例 \\(\widehat r=0.4\\) 时，平均每轮接受多少草稿、返回多少 token？为什么不能把 0.4 直接代入理论曲线中的 \\(p\\)？
5. 结构对照。drafter 输入形状为 `[1, 1, 5120]`。5120 由哪两部分组成？它们分别来自哪里？

[^ch09-google-mtp]: Olivier Lacombe、Maarten Grootendorst，[*Accelerating Gemma 4: faster inference with multi-token prediction drafters*](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)，Google，2026-05-05；访问日期：2026-07-18。

[^ch09-leviathan]: Yaniv Leviathan、Matan Kalman、Yossi Matias，[*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192)，ICML 2023，arXiv:2211.17192；访问日期：2026-07-18。

[^ch09-chen]: Charlie Chen、Sebastian Borgeaud、Geoffrey Irving、Jean-Baptiste Lespiau、Laurent Sifre、John Jumper，[*Accelerating Large Language Model Decoding with Speculative Sampling*](https://arxiv.org/abs/2302.01318)，2023，arXiv:2302.01318；访问日期：2026-07-18。

[^ch09-issue-2227]: Shoolife，[*MTP / speculative decoding regresses decode tok/s on PowerVR GPU (Tensor G6) — even with GPU sampler fully loaded*](https://github.com/google-ai-edge/LiteRT-LM/issues/2227)，LiteRT-LM issue #2227，2026-05-11；访问日期：2026-07-18。
[^ch09-llamacpp-draft]: ggml-org，[*llama.cpp 源码 common/arg.cpp:3763 与 common/speculative.cpp*](https://github.com/ggml-org/llama.cpp/blob/b9873/common/arg.cpp#L3763)，版本 b9873；访问日期：2026-08-31。
