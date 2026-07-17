# 第 9 章 一次前向，多个 token：推测解码与 MTP

> 使命：讲透一件反直觉的事——先用一个便宜的草稿模型（drafter）推测接下来的若干 token，再让基础模型一次前向把这一串并行验完，加速比为何完全取决于接受率。这是第三篇的收尾，讲端侧如何在内存带宽约束下再取得成倍的吞吐。

第 1 章分析 decode 的内存带宽约束时给过两条出路：提高带宽（端侧作者无从选择硬件），或减少每 token 要读的字节（量化，第 7 章做了）。这一章是第三条路：不改带宽、不改模型，用一次前向产出多个 token。这正是第 2 章「二十个问题」的第 19 问——推测解码（speculative decoding）靠预测，为什么反而更快，什么时候又更慢。

## 用一次前向产出多个 token

先把成本摆清。decode 的瓶颈是内存带宽：每生成一个 token，都要把全部权重从内存读一遍（第 1 章）。这一次前向的带宽开销很高，而它只换来一个 token。问题随之而来：这一次开销最高的前向，能不能一次产出多个 token。

推测解码的回答是能，前提是先做一次预测。用一个又小又快的模型先草拟接下来的若干 token，再让基础模型（base 模型）一次前向把这几个 token 一并验证。预测正确的部分直接采用，预测错误的位置用基础模型算出的正确 token 回退兜底。关键在于验证那几个草稿 token 只花基础模型一次前向，而不是逐个前向。开销最高的那次前向，被摊到了多个 token 上。

推测解码有几种形态。草稿模型可以是完全独立的另一个模型，也可以是和主模型共享主干的多头结构，即 MTP（Multi-Token Prediction，多 token 预测），Gemma 4 走的就是后者。在 LiteRT-LM 的运行时里，drafter 装载为一个独立的小模型（成员 `mtp_drafter_model_`），验证则复用基础模型上一个专门的 `"verify"` signature（常量定义 `runtime/executor/llm_litert_mtp_drafter.cc:63`，取用见 `base_model.FindSignature(kVerifySignatureRunner)` 于 `:227`）。所以无论训练时共不共享主干，运行时看到的都是同一套结构：小模型草拟、基础模型验证。

## 机制：草拟，然后一次验一串

一次 `Draft()`（`runtime/executor/llm_litert_mtp_drafter.cc:453`）分三步，函数体本身就是这三步的骨架：

```cpp
ASSIGN_OR_RETURN(std::vector<int> drafted_tokens,
                 RunDraftingLoop(token_id, activations));          // (1)

RETURN_IF_ERROR(PrepareVerifierInputBuffers(
    position, token_id, drafted_tokens, input_kv_cache_buffers));  // (2)
// ...
ASSIGN_OR_RETURN(std::vector<int> verifier_id_vector, RunVerification());  // (3)
```

`(1)` 草拟，`(2)` 把「上一个真 token 加草拟出的那串」拼成 verify 的输入，`(3)` 基础模型一次前向验完。三步之后才是接受循环。

第一步是草拟。drafter 逐个生成接下来的 G 个候选 token（`RunDraftingLoop`，`:328`）。G 是草拟步数（代码里 `num_draft_steps_`）。循环体一步一个 token：

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

`(1)` 循环 G 次，每次生成一个 token，串行。`(3)` 每步跑的是 `mtp_drafter_model_`，一个独立装载的小模型，不是基础模型，这是运行时能低成本草拟的前提。`(2)` 是 MTP 这一类方案的特征：drafter 的输入不只是词嵌入，还拼上了上一步的隐藏态 activation（源码注释以 `[B=1, T=1, D=3072]` 为例，即 1536 维词嵌入接 1536 维 activation；维度随模型而定，本书基准模型实剖出的 drafter 输入是 `[1, 1, 5120]`，即 2560 + 2560，与主干 `model_dimension = 2560` 一致，见附录 D），使草稿头以基础模型的语义状态为条件继续预测。`(4)` 每步只产出一个 token（`RET_CHECK_EQ(..., 1)` 作断言兜底）。`(5)` 把这一步的输出喂回下一步的输入：drafter 是自回归的，只是它的自回归开销远低于基础模型一次前向。这几步再串行，总开销也远不及基础模型一次前向。

第二步，一次验一串。把「上一个真 token 加草拟的 G 个」拼起来，交给基础模型的 verify signature（`RunVerification`，`:437`）：

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

`(1)` 这一次 `RunAsync` 是整套机制里带宽开销最高、也是唯一一次基础模型前向；G 个位置的验证全压在这一次里。`(2)` 它返回长度 G+1 的 id 序列，对 G+1 个位置各给出一个正确答案，多出来的那一位后面会用到。验证 G 个草稿 token 只用基础模型一次前向，这正是加速的来源。

第三步是接受。逐位比对草拟结果与验证结果，接受最长的匹配前缀。

<figure>
{{#include figs/fig-9-1.svg}}
<figcaption>图 9-1　推测解码一轮：drafter 逐个草拟 G 个 token，基础模型一次前向验证 G+1 个位置，接受最长匹配前缀再加一个 bonus token。一次前向，产出 1 到 G+1 个 token。</figcaption>
</figure>

## 接受循环：匹配前缀，加一个 bonus 保证产出下界

第三步值得逐行看，因为它藏着一个回退设计（`Draft()` 接受循环，`:471`–`:484`）：

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

`(1)` 从头逐位比对草拟结果与验证结果。`(3)` 一致就接受、`num_correct_tokens` 加一。`(2)` 一旦碰到第一个不一致就停下，把基础模型在这个位置给出的正确 token 记为 bonus：这一位草拟错了，但基础模型已经算出了对的，不浪费。`(4)` 循环走完 `bonus_token` 仍是 −1，说明 G 个全部命中；`(5)` 此时直接取第 G+1 个位置的验证结果——verify 本就多算了这一位（上一步 `RunVerification` 返回 G+1 个 id），全部命中时它就是额外获得的第 G+1 个 token。

被反复更新的 `last_verified_token_id_idx_` 不用于统计：它记住接受前缀在 verify 输出缓冲里的下标，下一轮 `RunDraftingLoop` 走 `ConcatenateEmbeddingsAndActivationsFromVerifierBuffer` 分支时，正是靠这个下标取出接受位置的 activation 作为草拟的新起点。接受循环和下一轮草拟就这样接上了（这条数据流后文详解）。

最后输出「接受的前缀加一个 bonus」（`:492`–`:493`）：

```cpp
// The first token comes from the decode output and is always correct.
drafted_tokens.resize(num_correct_tokens);   // (1)
drafted_tokens.push_back(bonus_token);       // (2)
```

`(1)` 把草拟序列截断到接受的长度，`(2)` 接上 bonus。源码注释点明第一个 token 来自 decode 输出、永远正确，所以哪怕 `num_correct_tokens` 为 0（第一个 token 就预测错），`resize(0)` 后 `push_back` 仍留下 1 个 bonus token。

这就是产出下界的保证：哪怕第一个 token 就预测错，也能拿到基础模型给的 1 个正确 token。论产出的 token 数，推测解码不会比普通 decode 差——最坏一次前向出一个 token，与普通 decode 打平。接受率高时单 token 成本大幅下降；接受率低时产出 token 数不减，只多付了 drafter 与 verify 的计算开销。

## 采样约束：这条 MTP 路径只做贪心接受

上面的接受循环有一个容易被略过的前提：判定用的是严格相等 `verifier_id_vector[i] != drafted_tokens[i]`。这不是文献里经典的推测采样。Leviathan 等（2023）与 Chen 等（2023）提出的推测解码带一个概率接受步骤（rejection sampling），能证明采样分布与直接用基础模型逐 token 采样完全一致，即分布无损。LiteRT-LM 当前这条 MTP 路径没有走那一步，drafter 与 verifier 都被强制为贪心采样，接受判定退化成逐位 token 相等。

强制贪心的地方在构造采样器的辅助函数里（`CreateGreedySampler`，`:65`–`:79`）：

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

`(1)` top-k 写死为 1，`(2)` top-p 写死为 0，两者合起来把候选集裁到唯一的 argmax；`(3)` temperature 固定 1.0，但在 k=1 的前提下已无采样自由度。drafter 与 verifier 共用这个工厂函数，只是 `sequence_size` 不同：drafter 每步一个位置（`:264`–`:267`），verifier 一次 G+1 个位置（`:268`–`:272`）。因此两侧都取各自 logits 的 argmax，验证阶段的判定就成了「drafter 的 argmax 是否等于 verifier 的 argmax」。

这个选择有明确的工程含义。分布无损的推测采样需要 verifier 回传每个位置的完整概率分布，接受时按 q(x)/p(x) 的比例做概率取舍，拒绝时还要从残差分布 max(0, q−p) 重采样。这要求把 verifier 的 logits 分布搬回宿主内存并逐位算比值，端侧要多付一份数据搬运与逐位计算。贪心接受把这一切省成一次整数相等比较：verifier 只需回传采样后的 token id（`verifier_id_tensor_` 形状 `[1, G+1]`），不必回传分布。代价是这条路径只在贪心解码（temperature=0 意义上的确定性输出）下等价于普通 decode；一旦用户想要带温度或 top-p 的随机采样，当前 MTP 路径并不保证与非推测路径同分布。据此推断，端侧在这里用简单换取了搬运与算力的节省，把无损采样留给了后续演进。

## 验证输入的构造：位置、mask 与 KV cache 的复制

「一次前向验一串」在缓冲层面怎么落地，值得摊开。verify 走的不是 decode 那种一次一个位置的形状，而是 prefill 形状：G+1 个位置一次喂入。构造这份输入的是 `PrepareVerifierInputBuffers`（`:374`–`:423`）：

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

`(1)` `input_pos` 连续填 `position` 到 `position+G`：这 G+1 个位置紧接在已生成序列之后，验证的是「若第 `position` 位是真 token、随后是草稿的 G 个，各位置基础模型给出什么」。`(2)` `FillAttentionMask` 以 `start_step=position`、`steps=G+1` 铺注意力 mask，形成因果结构：每个待验位置只能看到它之前的前缀（包括真 token 和更早的草稿位置），看不到自己之后的草稿。这一点保证验证的语义正确——第 `i` 个位置得到的 logits 只依赖前 `i` 个 token，与逐 token decode 到该位置时的条件一致，因此逐位比对才有意义。`(3)` 把 `[真 token, 草稿_1, …, 草稿_G]` 经 `LookupPrefill` 一次查成 embeddings，正是 prefill 那套批量查嵌入的接口（第 3 章）。`(4)` 对每个 KV cache 输入调 `Duplicate()`，把基础模型现有的 KV cache 句柄复制一份给 verify 用。

`Duplicate()` 复制的是缓冲句柄而非底层数据（TensorBuffer 的浅复制语义，第 7 章）。之所以要这一层，是因为 verify 前向要在已有 KV cache 之上追加 G+1 个位置的写入，而运行时对同一块缓冲有读写分离的约束（第 6 章的双缓冲）。紧接着的分支处理单缓冲 KV cache 的情形：

```cpp
if (active_verifier_input_buffers_.contains("param_tensor")) {   // (1)
  RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
      active_verifier_input_buffers_["param_tensor"], position,
      num_draft_steps_ + 1));
}
```

`(1)` 当模型走单缓冲 KV cache（把缓存的读写位置参数打包进一个 `param_tensor`）时，这里按 `position` 起、`G+1` 步填好那份参数张量，让 verify 前向知道该把这 G+1 个位置写到缓存的哪一段。两种 KV cache 布局在这里各有一条路径，`Duplicate()` 对应双缓冲、`param_tensor` 对应单缓冲。输出侧同理，`PrepareVerifierOutputBuffers`（`:424`）也对每个输出 KV cache 缓冲做 `Duplicate()` 再 `ClearEvent()`，为异步执行的完成事件让路。

## drafter 的隐藏态拼接：两条 activation 来源

回到草拟循环。MTP 头之所以能「续着基础模型的思路」预测，靠的是每步都把上一步的隐藏态拼进输入。正文前面给过 `[1536+1536=3072]` 的拼接，这里把两条 activation 来源讲清。`RunDraftingLoop` 的循环体里有一个分支（`:346`–`:353`）：

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

`(1)` `activations_ptr` 非空时走这条：拼接的 activation 来自参数传入的 `activations`。这只在一轮草拟的第一步成立，且这份 activation 由基础模型上一次 decode 前向产出（下一节讲集成时会看到它从哪来）。第一步之后，`activations_ptr` 被改指向 drafter 自己上一步的 `projected_activations`（循环末尾 `:369`），后续步骤就用 drafter 自产的隐藏态自回归下去。`(2)` `activations_ptr` 为空时走这条：从 verifier 的输出缓冲 `verifier_output_buffers_["activations"]` 里，按 `last_verified_token_id_idx_` 取出上一轮被接受位置的 activation。这正是接受循环里那个下标的用途——上一轮验证时基础模型顺带算出了每个位置的隐藏态，接受前缀的末位隐藏态就是这一轮草拟的起点。

两条来源对应两种进入草拟的时机。首个 decode 之后，drafter 拿到的是基础模型 decode 输出的 activation（走 `(1)`）；进入稳态后，drafter 拿到的是上一轮 verify 缓冲里接受位置的 activation（走 `(2)`）。无论哪条，语义都一样：草稿头从基础模型算出的隐藏态接着往下预测，而不是从零起步。这解释了 MTP 草稿为何比一个完全独立、只看 token id 的小模型更容易命中——它拿到的不只是上一个 token，还有基础模型对上下文的内部表示。

拼接本身是一次内存拷贝（`ConcatenateEmbeddingsAndActivations`，`:80` 起）：先把词嵌入 memcpy 进输出缓冲前半段，再把 activation memcpy 进后半段。两段各 `model_dimension` 维 float，拼成双倍宽度，正是 drafter signature 的 `activations` 输入形状（本书基准模型为 2560 + 2560 = 5120，实剖见附录 D）。这里没有额外计算，只有缓冲布局的拼装，开销可忽略；drafter 每步的主要成本仍是那一次 `RunAsync` 小模型前向。

## 集成：Draft 由谁调用，接受的 token 如何回写

机制章讲到这里都在 drafter 内部。把它接回运行时的缝在执行器的 decode 里（`llm_litert_compiled_model_executor.cc:1003` 起的 `Decode`）。有没有装载 drafter，决定走哪条路：`mtp_drafter_ == nullptr` 时走普通 decode（`:1007`），否则走推测路径。推测路径本身又分两个分支，区别在于这是不是 prefill 后的首个 decode：

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

`(1)` `ran_decode` 记录上一次运行是不是 decode。进入稳态（上一次已是 decode）时走这个分支：`activations` 传 `std::nullopt`（`(2)`），对应上一节的第二条 activation 来源——drafter 会从 verifier 缓冲里按接受下标取隐藏态，不需要外部再喂。`(3)` 是「一次前向前进多个位置」在记账层面的落地：`current_step` 不再加一，而是加上 `Draft` 这一轮实际产出的 token 数（接受前缀长度加 1 个 bonus）。这一行把机制的成倍产出兑现成了序列位置的成倍前进。

另一条分支处理 prefill 后的首个 decode（源码注释说明 MTP 要先做一次普通 decode 拿到 activation 再喂给草稿）：

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

`(1)` 先老老实实跑一次普通 decode，采样出一个 `token_id`。`(2)` 把这次 decode 输出的 `activations` 复制一份，作为参数喂进 `Draft`，这正是上一节第一条 activation 来源。`(3)` 传入的 `position` 是 `current_step - 1`：源码注释解释，普通 decode 已经在 `DecodeLogits` 里把 `current_step` 加过一次，而草稿要从这次真 token 所在的位置起算，因此要减一。`(4)` 收尾时把这个真 `token_id` 插回结果序列开头——`Draft` 返回的是「接受前缀加 bonus」，不含触发这一轮的那个真 token，这里补上，输出序列才完整。两个分支殊途同归：都以一个已确定正确的真 token 加它的隐藏态为起点，`current_step` 都按实际产出前进。

这条缝是理解 MTP 端到端行为的关键。对上层来说，一次 `Decode` 调用可能返回 1 到 G+1 个 token，序列位置一次前进相应的步数，而这背后只发生了一次基础模型前向（verify）加上若干次廉价的 drafter 前向。首个 decode 那次「浪费」的普通 decode 只发生一次，稳态后每轮都省在一次前向里。

## 接受率经济学：加速比取决于预测准不准

现在回答第 19 问的后半段：什么时候反而更慢。

关键指标是接受率——草拟的 token 里有多少被接受。每一轮结束，`Draft()` 累加两个计数（`:494`–`:495`）：

```cpp
num_drafted_tokens_ += num_draft_steps_;    // (1)
num_verified_tokens_ += num_correct_tokens; // (2)
```

`(1)` 分母，每轮固定加 G，与接受多少无关。`(2)` 分子，只加真正被接受的 `num_correct_tokens`，不含 bonus——bonus 是 verify 前向顺带算出的，不计入 drafter 的命中。drafter 析构时把这个比值打印出来（`:164`–`:172`）：

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

`(1)` 这个 "Success rate" 就是接受率，只在进程退出时打印一次，落在日志里、CLI 不透出——这也是本书实测无法实证归因接受率的直接原因。分子分母的定义决定了一个上界：G 步全部命中时接受率为 1.0；bonus 不进分子，所以这个数永远达不到「平均每轮产出 token 数除以 G」那个更乐观的口径。

把这笔账做成定量模型，能看清盈亏平衡在哪。设逐位接受概率为 α（贪心接受下，α 是「drafter 的 argmax 等于 verifier 的 argmax」的概率，随文体与模型而变），各位置近似独立。一轮草拟 G 步，第 k 位被接受当且仅当前 k 位全接受，概率 α^k。加上那个必得的 bonus，一轮的期望产出为：

$$ E[\text{产出}] = 1 + \sum_{k=1}^{G} \alpha^k $$

再看成本。设基础模型一次前向的开销为 c_base、drafter 一步前向的开销为 c_draft，一轮总开销约 c_base + G·c_draft（verify 一次，drafter G 步）。普通 decode 每 token 开销 c_base。于是加速比约为：

$$ \text{speedup} \approx \frac{E[\text{产出}]}{1 + G \cdot c_{\text{draft}} / c_{\text{base}}} $$

分子随 α 单调增，分母是固定的额外开销比。令 speedup = 1 解出的 α 就是盈亏平衡接受率：α 高于它，推测解码更快；低于它，反而更慢。分母里 c_draft/c_base 越大（drafter 相对基础模型不够便宜，或在某后端上开销比失衡），盈亏平衡点越高，越难划算。这把定性的权衡变成了可代入数字的判据。

算一个例子。取 G=3、drafter 开销约为基础模型的 15%（c_draft/c_base = 0.15），分母 = 1 + 3×0.15 = 1.45。若 α = 0.8，分子 = 1 + 0.8 + 0.64 + 0.512 = 2.95，speedup ≈ 2.95 / 1.45 ≈ 2.0；若 α 掉到 0.4，分子 = 1 + 0.4 + 0.16 + 0.064 = 1.62，speedup ≈ 1.62 / 1.45 ≈ 1.12；若 α = 0.2，分子 ≈ 1.25，speedup ≈ 0.86，已经低于 1，也就是更慢。（这里的 c_draft/c_base 与 α 均为示意取值，用于说明公式形状，非实测；实测见下。）接受率从 0.8 掉到 0.2，同一套代码从提速一倍变成拖慢，这就是「加速比取决于预测准不准」的定量含义。

<figure>
{{#include figs/fig-9-2.svg}}
<figcaption>图 9-2　接受率—收益曲线（按闭式公式推算，非实测）：盈亏平衡接受率约 0.25，α 越高收益越陡。</figcaption>
</figure>

- 接受率高时：一次基础模型前向产出多个 token，而基础模型前向是开销最高的一项，于是每 token 摊到的成本大幅下降，明显更快。官方报告 Gemma 4 上可达约 3 倍（官方博客口径，参见附录 F）。
- 接受率低时：草稿大多被丢弃，drafter 那几步的计算被丢弃、成为净开销，还多搭了 verify 相对普通 decode 的额外开销。当这些额外开销超过省下的前向，净收益为负，结果更慢。

这里如实报告本书自己的实测〔基准 D〕。主基准 Gemma 4 E4B 在基准机（context 1024、decode 128 token）上采了三档：关、`auto`、强制 `true`。按上一节的代码链路，`auto` 在 v0.13.1 实为关，所以「关」与「auto」两行本质是同一行为的两次采样：CPU 后端 22.8 对 24.9 tokens/s、GPU 后端 50.0 对 50.2 tokens/s，差异落在批内抖动幅度里（CPU 那批三次运行本身就散布在 20.1 到 24.9 之间）。真正开启的是强制 `true` 的一组，GPU 得 49.0，与同条件「关」的 50.2 同样在抖动内——两条口径下都没有复现 3 倍。

| 模式 | cpu decode tok/s | gpu decode tok/s | 说明 |
|---|---|---|---|
| 关（`false`） | 22.8 | 50.0 | 基线 |
| `auto` | 24.9 | 50.2 | v0.13.1 实为关，与基线同行为的再采样 |
| 强制 `true` | — | 49.0 | 唯一真正开启的一组，与关在抖动内 |

> 表 9-1　MTP 实测对照（Gemma 4 E4B，context 1024，decode 128 token，各 3 次中位数〔基准 D〕）。官方博客的「约 3 倍」口径在本基准未复现——收益取决于接受率。

这不推翻机制，反而与上面的公式相容。接受率现已实测：用 Python SDK 把日志级别调到 VERBOSE，drafter 析构时会打印 `Num drafted/verified tokens` 与 `Success rate`（`llm_litert_mtp_drafter.cc:166-169`，实录见附录 D）。两类文体的对照极有说服力：写一段机器人学画画的创造性故事，α ≈ 26%——恰好压在盈亏平衡（α* ≈ 0.25）上方边缘，代入公式 speedup ≈ 0.93，这就是本书 benchmark「开关无差异」的直接原因；写一段 fibonacci 函数加解释，α ≈ 99.5%，代入公式 speedup ≈ 2.7，落入官方「约 3 倍」的口径区间。接受率不是模型常数，是文体的函数：benchmark 的合成负载与创造性写作落在低 α 区，官方演示的高可预测文本落在高 α 区，两个看似矛盾的数字因此同时为真。

这条实测路径值得记下来，因为它绕过了「CLI 不输出接受率」的限制，且不用改码重编：计数器本来就在 drafter 里累加（`num_drafted_tokens_` / `num_verified_tokens_`），析构时经 `ABSL_LOG(INFO)` 打印；Python SDK 暴露了 `set_min_log_severity`（`python/litert_lm/_ffi.py:450`），调到 VERBOSE 即可看到。若想做成按周期输出或落进 benchmark 统计，仍需把计数器经执行器暴露出去（`Draft()` 每轮结束处，`:494` 之后），本书未改上游代码。

由此也能理解那个反直觉现象（第 19 问后半，`LiteRT-LM#2227`）：在某些 GPU（如 PowerVR）上，MTP 反而拖慢 decode。成因可以从公式推出——当 α 不够高、或 drafter 与 verify 在那块硬件上的 c_draft/c_base 偏大时，加速比跌破 1。该现象无真机可复现，按上游报告所述。

所以推测解码不是无条件的加速，是一个有条件的权衡：收益取决于接受率，接受率不足时净收益为负。文体也影响接受率——套路性强的文本（比如代码）容易预测、接受率高，天马行空的散文难预测、接受率低。

## 开启条件：模型声明支持，草拟步数导出时定死

要开推测解码，得先过两道门槛：模型自己声明支持，以及草拟步数 G 早在导出时就定死。

其一，一个模型支不支持推测解码，写在它的能力声明里。运行时用 `HasSpeculativeDecodingSupport`（`schema/capabilities/speculative_decoding.h:33`、`:44`）判断——头文件给了两个重载，一个接受 `std::istream&`、一个接受文件路径，后者只是打开文件转调前者。真正的判断在 `.cc` 里，机制很朴素（`speculative_decoding.cc:40`–`:73`）：

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

`(2)` 遍历 `.litertlm` 的所有 section，`(3)` 找每个 TFLite 模型 section 的 `model_type` 元数据，`(4)` 只要有一个等于 `(1)` 里那个字符串 `"tf_lite_mtp_drafter"` 就返回 true。也就是说，「支持推测解码」不是一个布尔开关，而是文件里有没有打包一个 `model_type` 标为 mtp drafter 的子模型。这正是第 7 章讲的 `.litertlm` 容器里那些 section 的用途之一：drafter 就是和基础模型打包在同一个文件里的另一个 section。

从能力声明到 drafter 装载，链路各环都能落到代码，但中间有一环在 v0.13.1 还没接上。CLI 侧，`--enable-speculative-decoding` 取 `auto`、`true`、`false` 三选一（`python/litert_lm_cli/common.py:108`），经 `parse_speculative_decoding`（`:21`）映射：`auto` 与缺省映射为 `None`、`true` 映射为 `True`（强制开启，模型不支持则报错）、`false` 映射为 `False`。关键在 `None` 的走向：Python 绑定层只在值非 `None` 时才调 setter（`python/litert_lm/engine.py:113`），`auto` 因此不触碰 C++ 侧的默认值——而默认是关（`bool enable_speculative_decoding = false`，`runtime/executor/llm_executor_settings.h:258`）。所以 v0.13.1 里 `auto` 的实际行为是关：CLI help 宣称的「按模型元数据自动判断」（`common.py:115`）尚未接线，引擎创建路径并不调用 `HasSpeculativeDecodingSupport` 做探测。这个标志经引擎注入执行器设置（`.enable_speculative_decoding = settings.enable_speculative_decoding`，`runtime/engine/litert_lm_lib.cc:592`）。执行器构造时看这个标志：

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

`(1)` 标志为真才装载 drafter，`(2)` `Create` 内部用 `resources.GetTFLiteModel(ModelType::kTfLiteMtpDrafter)`（`:194`）取出那个 drafter section 编译成独立小模型。若文件里根本没打包 drafter section，这一步取不到模型、`Create` 失败——这就把「强制开启但模型不支持则报错」的 CLI 语义落到了实处。至于能力探测，`HasSpeculativeDecodingSupport` 在 v0.13.1 只作为查询接口暴露给 SDK 调用方（C API 见 `schema/capabilities/capabilities_c.cc:50`，Kotlin JNI 见 `kotlin/java/com/google/ai/edge/litertlm/jni/litertlm.cc:1233`），引擎自己不调它：想按元数据自动开启的应用，得自己查、自己把标志置真。

其二，草拟步数 G 不是运行时可调的旋钮。它由模型 verify signature 的形状固定（`:250`–`:256`）：

```cpp
LITERT_ASSIGN_OR_RETURN(auto input_pos_tensor_type,
                        verify_signature.InputTensorType("input_pos"));
// Expecred shape: [T = G + 1] where G is the number of draft steps
const auto& input_pos_dims = input_pos_tensor_type.Layout().Dimensions();
num_draft_steps = input_pos_dims[0] - 1;                         // (1)
```

`(1)` G 直接读自 verify signature 的 `input_pos` 张量维度减一——signature 能同时喂进几个位置，G 就是几。verify 一次吃 G+1 个位置（源码注释 `[T = G + 1]`），减去起头那个真 token，就是草拟步数。一个模型草拟几步，在它被导出、signature 形状定死的那一刻就固定了，运行时只能读、不能改。这是第 4 章「固定形状」约束在推测解码上的又一次体现：连草拟几步这个看似是超参的量，也被烘焙进了 signature。这也解释了为何 `CreateGreedySampler` 构造 verifier 采样器时把 `sequence_size` 设为 `num_draft_steps + 1`（`:270`）：采样器的形状必须与 verify signature 一次输出 G+1 个位置对齐。

<div class="aside-compare">

推测解码的组织方式，llama.cpp 给出对照：draft 模型是一个独立的模型文件，用 `--model-draft` 在运行时指定（`llama.cpp/common/arg.cpp:3763 @ b9873`），配一套通用的草拟-验证循环（`common/speculative.cpp`），任何词表兼容的小模型都能当 drafter。LiteRT-LM 的 MTP 则把 drafter 作为一个段打包进同一个 `.litertlm`、verify signature 编进主模型（本章实剖）。前者自由：可以给 70B 配 1B，随时换搭配；后者省心：模型发布者选好、验证好、一个文件带走，运行时按能力声明自动开启。自由度与开箱即用，仍是那道熟悉的选择题。

</div>

## 小结

推测解码是降低 decode 阶段带宽压力的第三类手段：不改带宽、不改模型，用一个便宜的 drafter 先预测、基础模型一次前向验一串，把开销最高的那次前向摊到多个 token 上。bonus 设计保证它论 token 数不亏，接受率决定它到底快多少。这一章把机制拆到了缓冲与集成两层：verify 走 prefill 形状、以因果 mask 保证逐位比对有意义、对 KV cache 做 `Duplicate()` 或填 `param_tensor`；drafter 每步拼接基础模型的隐藏态自回归下去，起点来自 decode 输出或上一轮接受位置；执行器 decode 用 `current_step += 产出数` 把成倍产出兑现为序列位置的成倍前进。当前这条 MTP 路径做的是贪心接受而非分布无损的推测采样，用一次整数相等比较换掉了分布回传与逐位概率计算。加速比可写成 E[产出] 除以固定额外开销比，存在一个盈亏平衡接受率；〔基准 D〕E4B 在合成负载下未复现 3 倍，现已实证归因：创造性文本 α ≈ 26%，恰压在盈亏平衡边缘，而代码类文本 α ≈ 99.5%，落入官方 3 倍口径区——接受率是文体的函数，两组数字不矛盾。

第三篇到此收尾。内存容量与带宽约束、硬件差异约束，各有对策：KV cache 与量化针对内存与带宽，异构后端针对硬件差异，推测解码在接受率够高时能再取得成倍的吞吐。下一部走出纯文本，去看这套运行时怎么长出多模态、工具调用这些能力，以及它如何变成六种语言的 SDK。

---

## 练习与自查

1. **盈亏平衡。** 设 drafter 单步成本是 base 前向的 0.1 倍，G = 3。若每个位置独立以概率 p 被接受（前缀截断），一轮期望产出约 1 + p + p² + p³ 个 token。推测解码不亏的最低 p 约是多少？
2. **保底机制。** 为什么最坏情况下（首个草拟就错）推测解码的产出 token 数也不少于普通 decode？
3. **形状约束。** 草拟步数 G 为什么在模型导出时就定死？从 verify signature 的哪一个维度读出？
4. **现象解释。** 本书基准里 MTP 开关无显著差异。用实测接受率解释：两类文体的 α 各是多少？代入加速比公式后，各自的 speedup 落在什么位置？
5. **实剖对照。** drafter 输入形状是 `[1, 1, 5120]`。这 5120 由哪两半拼成？各自从哪里来？


<!-- MTP 已实测（附录 D）：auto 档在 v0.13.1 实为关；强制 true 与关在抖动内、未复现 3x。2026-07-17 接受率实证归因：Python SDK VERBOSE 日志读出析构打印的计数器（故事 α≈0.26 / 代码 α≈0.995），文体接受率对比由此补上，无需改码。#2227 无真机，按【文档】级引 issue。 -->
