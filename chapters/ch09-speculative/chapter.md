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
// runtime/executor/llm_litert_mtp_drafter.cc:794
  ABSL_ASSIGN_OR_RETURN(DraftingResult drafting_result, // (1)
                        RunDraftingLoop(token_id, activations, constraint_,
                                        constraint_state_.get()));

  ABSL_RETURN_IF_ERROR(PrepareVerifierInputBuffers( // (2)
      position, token_id, drafting_result.drafted_tokens,
      state_buffers.input_buffers));
// ...
  ABSL_ASSIGN_OR_RETURN(
      std::vector<int> verifier_id_vector,
      RunVerification(drafting_result.draft_constraint_states)); // (3)
```

代码行 `(1)` 草拟 \\(G\\) 个 token；`(2)` 把上一个已确认 token 和 \\(G\\) 个草稿拼成 verify 输入；`(3)` 由基础模型一次完成验证。三步之后，接受循环再比较两组 token。

第一步是草拟：drafter 逐个生成接下来的 \\(G\\) 个候选 token，\\(G\\) 是草拟步数（代码里的 `num_draft_steps_`）。循环每次生成一个 token：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:574
  for (int i = 0; i < num_draft_steps_; ++i) { // (1)
// ...
    // Concatenated embedding + activation has shape [B = 1, T = 1, D = 3072]
// ...
    ABSL_RETURN_IF_ERROR(embedding_manager_.LookupDecode(last_drafted_token_id,
                                                         embedding_vector));
    if (activations_ptr) {
      ABSL_RETURN_IF_ERROR(ConcatenateEmbeddingsAndActivations( // (2)
          embedding_vector, *activations_ptr, *drafter_activations_buffer));
// ...
    LITERT_RETURN_IF_ERROR(mtp_drafter_model_.RunAsync( // (3)
        drafter_signature_.Key(), active_drafter_input_buffers_,
        active_drafter_output_buffers_, async));
// ...
    RET_CHECK_EQ(id_vector.size(), 1); // (4)
    int sampled_draft_id = id_vector[0];
    result.drafted_tokens.push_back(sampled_draft_id);
// ...
    last_drafted_token_id = sampled_draft_id; // (5)
    activations_ptr = &active_drafter_output_buffers_["projected_activations"];
  }
```

代码行 `(1)` 循环 \\(G\\) 次，每次串行生成一个 token。`(2)` 拼接本步输入，由词嵌入与上一步的隐藏态 activation 拼接组成：源码注释以 `[B=1, T=1, D=3072]` 为例，即 1536 维词嵌入加 1536 维 activation；具体维度随模型而定，本书基准模型的 drafter 输入为 `[1, 1, 5120]`，由 2560 维词嵌入和 2560 维 activation 组成，两部分都与主干的 `model_dimension = 2560` 一致（见附录 D）。`(3)` 每步运行的是独立装载的 drafter 模型，而非基础模型。`(4)` 断言每步只产出一个 token，`(5)` 再把本步输出作为下一步输入，drafter 仍按自回归方式生成。可见草拟本身有开销：一轮的总成本包括 \\(G\\) 次 drafter 前向和一次 verify 前向，MTP 是否加速取决于这个总成本，不能只看 drafter 的模型尺寸。

第二步是批量验证：运行时把上一个已确认 token 和 \\(G\\) 个草稿传给基础模型的 verify signature：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:723
  LITERT_RETURN_IF_ERROR(base_model_.RunAsync( // (1)
      verify_signature_.Key(), active_verifier_input_buffers_,
      active_verifier_output_buffers_, async));
// ...
  LITERT_ASSIGN_OR_RETURN(auto id_vector,
                          CopyFromTensorBuffer<int32_t>(verifier_id_tensor_));
  RET_CHECK_EQ(id_vector.size(), num_draft_steps_ + 1); // (2)
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
// runtime/executor/llm_litert_mtp_drafter.cc:808
  int num_correct_tokens = 0;
  while (num_correct_tokens < num_draft_steps_ && // (1)
         verifier_id_vector[num_correct_tokens] ==
             drafting_result.drafted_tokens[num_correct_tokens]) {
    ++num_correct_tokens; // (2)
  }
  int bonus_token = verifier_id_vector[num_correct_tokens]; // (3)
  last_verified_token_id_idx_ = num_correct_tokens; // (4)
```

代码行 `(1)` 从头逐位比较草拟结果与验证结果；每匹配一个位置，`(2)` 将接受长度加一。遇到第一个不匹配位置或接受完全部 \\(G\\) 个草稿时，循环结束。`(3)` 取接受前缀之后的基础模型输出作为 bonus token：若全部匹配，取的就是 verify 输出的第 \\(G+1\\) 个 token。`(4)` 将这个位置下标保存下来，供下一轮读取 activation。

`last_verified_token_id_idx_` 不参与接受比例统计。它保存 bonus token 对应的 verify 输出位置下标。下一轮草拟用这个下标从 verifier 缓冲读取 activation 作为起点（隐藏态的两条来源见 9.6 节）。

`Draft()` 最后返回接受前缀和一个 bonus token：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:841
  std::vector<int> output_tokens = std::move(drafting_result.drafted_tokens);
  output_tokens.resize(num_correct_tokens); // (1)
  output_tokens.push_back(bonus_token); // (2)
```

代码行 `(1)` 把草拟序列截断到接受长度，`(2)` 再追加 bonus。bonus token 来自基础模型输出，采用它不以 drafter 匹配为前提。即使 `num_correct_tokens` 为 0，`resize(0)` 后的 `push_back` 也保证返回 1 个 token。

所以一次 `Draft()` 至少返回 1 个 token。不过这只是每轮输出数量的下界，不是运行成本的上界：即使一个草稿都没被接受，该轮也已经执行了 \\(G\\) 次 drafter 前向和缓冲操作；prefill 后的首次 `Decode()` 另有一次普通基础模型前向，同样不在这个稳态口径之内。

## 9.4　采样约束：当前 MTP 路径采用贪心接受

接受循环用的是严格相等比较，这与经典推测采样不同：Leviathan 等（2023）[^ch09-leviathan] 与 Chen 等（2023）[^ch09-chen] 都在接受阶段引入概率步骤（rejection sampling 及其修正形式），使输出分布与目标模型（即本章的基础模型）保持一致。LiteRT-LM 当前的 MTP 路径没有这一步，drafter 与 verifier 均采用贪心采样，接受条件就是两个 token id 相等。

贪心设置位于构造采样器的辅助函数里：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:79
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(1); // (1)
  sampler_params.set_p(0.0f); // (2)
  sampler_params.set_temperature(1.0f); // (3)
  sampler_params.set_seed(0);
  return CreateSampler(backend, output_heads, std::move(sampler_params), env,
                       sequence_size, vocab_size, activation_data_type);
```

代码行 `(1)` 把 top-k 设为 1，`(2)` 把 top-p 设为 0，候选集中只保留 argmax。`(3)` 的 temperature 为 1.0，但在 k=1 时不会引入随机选择。drafter 与 verifier 共用这个函数，只是 `sequence_size` 不同：drafter 为 1，verifier 为 \\(G+1\\)。未施加约束时，接受阶段比较的是两侧 logits 的 argmax token。

运行时也可以为草拟与验证提供约束。约束先根据当前生成状态给出 logits mask，修改候选分数，再执行贪心采样。草拟端逐步应用 mask；验证端根据各草稿前缀的约束状态，一次处理整段 logits：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:599
    if (constraint != nullptr && current_draft_state != nullptr) {
      ABSL_ASSIGN_OR_RETURN(auto mask,
                            constraint->ComputeMask(*current_draft_state));
      ABSL_RETURN_IF_ERROR(ApplyMaskToLogits(
          active_drafter_output_buffers_["logits"], mask.get(), vocab_size_));
    }

    ABSL_RETURN_IF_ERROR(drafter_sampler_->SampleToIdAndScoreBuffer(
        active_drafter_output_buffers_["logits"], drafter_id_tensor_,
        /*scores_tensor=*/nullptr));
// ...
    ABSL_RETURN_IF_ERROR(ApplyMasksToLogitsSequence(
        active_verifier_output_buffers_.at("logits"), masks, vocab_size_));
  }

  ABSL_RETURN_IF_ERROR(verifier_sampler_->SampleToIdAndScoreBuffer(
      active_verifier_output_buffers_.at("logits"), verifier_id_tensor_,
      /*scores_tensor=*/nullptr));
```

这条路径仍然比较贪心 token，只是比较前的 logits 已经过约束处理。草拟过程中会为各候选位置保存约束状态；接受前缀确定后，运行时保留相应状态，再按 bonus token 推进。被拒绝的草稿状态不会作为下一轮起点。每个 decode 回合的首轮或约束对象改变时，运行时会重新初始化约束状态。这里描述的是源码处理流程，不能代替具体约束与模型组合的正确性测试。

要保持采样分布不变，verifier 需要提供各位置的完整概率分布：接受阶段按概率比值决定是否采用草稿，拒绝后还要从残差分布重新采样。基础模型 verify 前向产生 logits，verifier 采样器再将它们转成形状为 `[1, G+1]` 的 token id；接受循环只比较这些整数。接受阶段不使用完整概率分布，适用范围因而收窄到贪心比较。即使设置 temperature 或 top-p，MTP 路径也不能据此保持非推测路径的随机采样分布。

## 9.5　验证输入的构造：位置、mask 与 KV cache 的复制

verify 的 KV cache 读写缓冲由执行器的状态对象提供。`Draft()` 要求传入对象是 `LitertState`，再按 verify signature 取得对应缓冲：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:765
  auto* litert_state = dynamic_cast<LitertState*>(&state);
  RET_CHECK(litert_state != nullptr);
  LITERT_ASSIGN_OR_RETURN(
      auto state_buffers,
      litert_state->GetStateBuffers(base_model_, verify_signature_.Key()));
```

状态对象负责缓冲的组织与复用；drafter 从输出侧缓冲读取此前的上下文，verify 则使用同一状态提供的输入、输出映射。句柄复制不会另建一份完整 KV cache。

verify 使用包含 \\(G+1\\) 个位置的批量输入，而不是普通 decode 的单位置输入。构造这组缓冲的函数执行四个步骤：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:641
    auto* prefill_input_pos_ptr =
        static_cast<int32_t*>(verifier_input_pos_lock_and_addr.second);
    for (int i = 0; i < num_draft_steps_ + 1; ++i) { // (1)
      *prefill_input_pos_ptr++ = position + i;
    }
// ...
    ABSL_RETURN_IF_ERROR(FillAttentionMask(mask_buf, // (2)
                                           /*start_timestep=*/position,
                                           /*steps=*/num_draft_steps_ + 1,
                                           attn_params_.global_type));
// ...
  if (verifier_signatures_.input_embeddings.has_value()) {
    ABSL_RETURN_IF_ERROR(embedding_manager_.LookupPrefill( // (3)
        drafted_tokens_with_input_token,
        &active_verifier_input_buffers_[*verifier_signatures_.input_embeddings],
        /*offset=*/0));
  }
// ...
  for (const auto& [input_name, input_buffer] : input_kv_cache_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate()); // (4)
    active_verifier_input_buffers_[input_name] = std::move(input_buffer_dup);
  }
```

代码行 `(1)` 把 `input_pos` 依次写为 `position` 到 `position+G`，这 \\(G+1\\) 个位置对应一个已确认 token 和随后 \\(G\\) 个草稿。`(2)` 按模型的 attention 参数填充全局 mask，起点为 `position`、步数为 \\(G+1\\)。因果 mask 限制各位置读取当前及此前的输入，阻止看到后续草稿。若 signature 还含局部 mask，另一个分支按滑动窗口大小和 verify 模式填充它。可见前缀因此还取决于模型使用全局注意力还是局部注意力，不能一律理解成读取全部历史。`(3)` 用 prefill 路径的批量查表取得 `[已确认 token, 草稿_1, …, 草稿_G]` 的 embeddings。`(4)` 为每个 KV cache 输入复制 `TensorBuffer` 句柄，供 verify 使用。

`Duplicate()` 复制的是缓冲句柄而不是底层数据（第 8 章讨论过这种浅复制语义）。复制的原因在于：verify 前向要在现有 KV cache 之后追加 \\(G+1\\) 个位置，而第 6 章介绍的双缓冲路径要求读写两侧分别传入句柄。单缓冲 KV cache 则由紧随其后的分支处理：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:698
  if (verifier_signatures_.input_int32_param.has_value()) { // (1)
    ABSL_RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
        active_verifier_input_buffers_[*verifier_signatures_.input_int32_param],
        position, num_draft_steps_ + 1));
  }
```

代码行 `(1)` 处理 6.3.1 节的单缓冲路径，读写位置放在参数张量里。这里从 `position` 开始填入 \\(G+1\\) 步对应的参数，使 verify 写入相应的缓存区间。单缓冲与双缓冲所需的读写句柄由状态对象提供；单缓冲额外设置参数张量。输出侧由另一个函数处理，复制各个 KV cache 输出句柄，并清除旧的完成事件。

## 9.6　drafter 的隐藏态拼接：两条 activation 来源

MTP drafter 每步都把词嵌入与隐藏态拼接后输入模型。草拟循环的以下分支决定隐藏态从哪里来：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:584
    if (activations_ptr) {
      ABSL_RETURN_IF_ERROR(ConcatenateEmbeddingsAndActivations( // (1)
          embedding_vector, *activations_ptr, *drafter_activations_buffer));
    } else {
      ABSL_RETURN_IF_ERROR(
          ConcatenateEmbeddingsAndActivationsFromVerifierBuffer( // (2)
              embedding_vector, verifier_output_buffers_["activations"],
              last_verified_token_id_idx_, *drafter_activations_buffer));
    }
```

代码行 `(1)` 在 `activations_ptr` 非空时拼接参数传入的 activations。调用方传入 activations 时，第一步执行的就是这条路径，该 activation 来自基础模型此前的 decode 前向。第一步结束后，`activations_ptr` 改指 drafter 上一步输出的 `projected_activations`，后续各步总是执行分支 `(1)`，按自回归方式继续。`(2)` 只在第一步且未传入 activations 时执行：隐藏态取自上一轮的 verifier 输出缓冲，由接受循环维护的 `last_verified_token_id_idx_` 选出被接受位置对应的 activation。

两条来源对应两种调用时机：prefill 后的首次 decode 传入普通 decode 输出的 activation，执行路径 `(1)`；进入稳态后不再传入，草拟第一步从上一轮 verify 缓冲取 activation，即路径 `(2)`。每轮草拟第一步使用基础模型产生的上下文表示；后续步骤使用 drafter 自己输出的 projected activation。本章没有比较其他 drafter 结构，不能据此推断相对命中率。

拼接本身很简单：先把词嵌入复制到输出缓冲前半段，再把 activation 复制到后半段，两段各含 `model_dimension` 个 float，组成 drafter signature 的双倍宽度输入（本书基准模型为 2560 + 2560 = 5120，记录见附录 D）。这段代码只执行两次内存复制、不含矩阵计算，但它在端到端时延中的占比仍需测量，不能仅凭代码结构判定为可忽略。

## 9.7　集成：`Draft()` 的调用与 token 回写

执行器的 `Decode()` 负责把 drafter 接入推理流程：没有装载 drafter 时执行普通 decode，否则进入 MTP 路径，并按当前调用是否为 prefill 后的第一次 decode 再分两支。稳态分支如下：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:1081
    bool last_run_is_decode = llm_context_->runtime_state().ran_decode; // (1)
    if (last_run_is_decode) {
// ...
      LITERT_ASSIGN_OR_RETURN(output_tokens_vector,
                              mtp_drafter_->Draft(step_and_token.step,
                                                  step_and_token.token[0]->id(),
                                                  /*activations=*/std::nullopt, // (2)
                                                  *active_state, constraint));
      RET_CHECK_EQ(output_tokens_vector.size(), 1);
      llm_context_->runtime_state().current_step += // (3)
          output_tokens_vector[0].size();
```

代码行 `(1)` 的 `ran_decode` 表示上一次运行是否为 decode，进入稳态后执行该分支。`(2)` 将 activations 设为空，即 9.6 节的路径 `(2)`：drafter 从 verifier 缓冲读取上一轮被接受位置的隐藏态。`(3)` 按 `Draft()` 本轮返回的 token 数增加 `current_step`，该数等于接受前缀长度加 1 个 bonus token。稳态下，一次 `Decode()` 直接返回这组 token，数量在 1 到 \\(G+1\\) 之间。

另一条分支处理 prefill 后的第一次 decode。此时运行时先执行普通 decode，取得首个 token 及其 activation，再调用 `Draft()`：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:1109
        ABSL_RETURN_IF_ERROR(SampleLogits(decoded_logits, output_tokens)); // (1)
// ...
        token_id = output_tokens_vector[0][0];
// ...
      LITERT_ASSIGN_OR_RETURN(
          auto activations, decode_output_buffers_["activations"].Duplicate()); // (2)
// ...
      LITERT_ASSIGN_OR_RETURN(
          output_tokens_vector,
          mtp_drafter_->Draft(llm_context_->runtime_state().current_step - 1, // (3)
                              token_id, std::move(activations), *active_state,
                              constraint));
      llm_context_->runtime_state().current_step +=
          output_tokens_vector[0].size();
      output_tokens_vector[0].insert(output_tokens_vector[0].begin(), token_id); // (4)
```

代码行 `(1)` 普通 decode 先采样出 `token_id`。`(2)` 复制该次 decode 输出的 activations 并传给 `Draft()`，对应上一节的第一条 activation 来源。`(3)` 传入 `current_step - 1`：此时普通 decode 已把 `current_step` 加过一次，而草拟位置要从刚得到的 token 起算。`(4)` 最后把这个普通 decode token 插到 `Draft()` 返回序列之前，于是第一次 `Decode()` 返回 2 到 \\(G+2\\) 个 token：1 个来自普通 decode，1 到 \\(G+1\\) 个来自 `Draft()`。

首轮与稳态的差别还体现在基础模型调用次数上。第一次 `Decode()` 执行一次普通 decode 加一次 verify，共两次基础模型前向。稳态的每次 `Decode()` 只执行一次 verify；两者的 drafter 前向都是 \\(G\\) 次。分析长期吞吐通常采用稳态口径，但测量短输出时，首轮多出的那次基础模型前向不能省略。

## 9.8　接受比例与加速比

MTP 是否加速，取决于每轮产出的 token 数和该轮的运行成本。先看产出怎么计数。`Draft()` 在每轮结束时累加两个计数：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:844
  num_drafted_tokens_ += num_draft_steps_; // (1)
  num_verified_tokens_ += num_correct_tokens; // (2)
```

代码行 `(1)` 每轮把草稿总数增加 \\(G\\)，`(2)` 只累加匹配前缀的长度，不含 bonus token。drafter 析构时输出这两个计数及其比值：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:294
LlmLiteRtMtpDrafter::~LlmLiteRtMtpDrafter() {
  ABSL_VLOG(1) << "Num drafted tokens: " << num_drafted_tokens_;
  ABSL_VLOG(1) << "Num verified tokens: " << num_verified_tokens_;
  if (num_drafted_tokens_ > 0) {
    ABSL_LOG(INFO) << "MTP Drafter - Success rate: " // (1)
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

以下实测记录沿用附录 D 的历史软件环境，未在 v0.17.0 上重跑，不能当作新版性能结果。本书的实测记录分三类，各自能回答的问题不同。第一类是 Mac 上的两组主基准记录（Gemma 4 E4B、context 1024、decode 128 token），参数分别设为 `false` 与 `auto`。这两组并不构成开关对照：旧版 v0.13.1 的设置链路中，`auto` 最终沿用 C++ 默认值 `false`，两组都是关闭 MTP 后的独立采样。CPU 中位数分别为 22.8 和 24.9 tokens/s（`false` 组的三次运行分布在 20.1 到 24.9 tokens/s），GPU 中位数为 50.0 和 50.2 tokens/s。两组之间的差异只反映运行波动，估计不了 MTP 的开关收益。这批 Mac 主基准没有强制开启组；自然代码负载另有单次补测摘要，见下文第三类记录。

| 模式 | cpu decode tokens/s | gpu decode tokens/s | 说明 |
|---|---|---|---|
| 关（`false`） | 22.8 | 50.0 | 基线 |
| `auto` | 24.9 | 50.2 | 实为关，与基线同行为的再采样 |

> 表 9-1　Mac 归档记录中的 `false` 与 `auto` 均为关闭行为，各列为 3 次运行的中位数〔基准 D〕；该表不能用于估计 MTP 收益。

第二类是接受比例的历史实测样本。附录 D 记录了旧版 drafter 析构时输出的计数器，实验通过 Python SDK 把日志级别设为 VERBOSE。故事提示词产生 213 个草稿、56 个匹配，\\(\widehat r=56/213\approx 0.263\\)，平均每次 `Draft()` 返回 \\(1+3\widehat r\approx 1.79\\) 个 token；代码提示词产生 3069 个草稿、3054 个匹配，\\(\widehat r\approx 0.995\\)，平均返回约 3.99 个 token。在图 9-2 的示意假设 \\(q=1.45\\) 下，两个样本对应的加速比分别约为 1.23 和 2.75；若把 \\(\widehat r\\) 误作 \\(p\\) 代入，会得到 0.93，那是两种口径的混用。这些只是示意值：计数实验没有同时测量 \\(q\\)，得不出实测速率；两次实验的提示词也与固定长度 benchmark 不同，故事样本的 \\(\widehat r\\) 推断不了该 benchmark 开启 MTP 后的吞吐。

差别源自负载的构造方式。固定长度 benchmark 的输入构造很直接：代码先对提示词分词，再把 id 序列调整到指定长度，N 大于原 token 数时新增的元素全为 0，更小时序列被截断。这样的负载与自然文本差别很大，前述两个 \\(\widehat r\\) 都不适用。Mac 的这组固定长度 benchmark 没有强制开启组，也没有相应的 drafter 计数器。合成负载下的强制开启对照来自 Android，其减速同样没有证据归因于低接受比例。自然代码负载的单次记录不能替代这组对照。

这组历史实验通过 Python SDK 的 VERBOSE 日志读取计数。v0.17.0 中，两个原始计数由 `ABSL_VLOG(1)` 输出，聚合比例仍由 `ABSL_LOG(INFO)` 输出。复现时需要启用 VLOG 级别 1，并确认日志中确实出现两个计数，不能只凭 INFO 级别的比例恢复原始样本量。局限也在这里：日志只提供整个 drafter 生命周期内的聚合比例，要得到分阶段或逐轮数据，就需要新增遥测接口。

第三类是强制开启后的端到端对照，方向并不一致。上游有减速报告：`LiteRT-LM#2227` 记录了 PowerVR GPU 上开启 MTP 后 decode 吞吐下降的案例，环境为 LiteRT-LM 0.11.0、Gemma 4 E2B 和俄文分类负载，其中关于 GPU 路径的原因分析明确标为假设。[^ch09-issue-2227] 对照本章公式，减速只有两类可能条件：\\(r\\) 偏低，或归一化轮次成本 \\(q\\) 偏高；公式区分不了二者，也确认不了该 issue 的根因。本书基准数据里也有一例：附录 D 第十三节记录了一台 Qualcomm 机型在合成负载（prefill 1024 个 token）下的 MTP 减速，强制开启后 CPU decode 从 10.0 降到 2.8 tokens/s，GPU 从 18.0 降到 12.6 tokens/s。这次实验同样没有同步记录 \\(r\\) 或分解 \\(q\\)，原因归不到其中任何一个参数上。

同一台手机在自然代码提示词下的方向则相反：GPU 单次观测从 16.0 变为 32.0 tokens/s（decode 长度 128），CPU 从 11.6 变为 12.6 tokens/s（decode 长度 64）〔基准 D〕。不过每个条件只运行 1 次，两种后端的 decode 长度也不同，这些记录只能说明测试中存在观测差异，估计不了稳定加速比，也比较不了后端绝对值；它们与 \\(\widehat r\\) 计数不在同一次运行里，同样分解不出接受比例和轮次成本各自的贡献。

增速一侧还有另两条记录。本书基准模型的 verify signature 中 `input_pos` 形状为 `[4]`，故 \\(G=3\\)，一次 `Draft()` 最多返回 4 个 token。附录 D 的自然代码长生成补测摘要记载，Mac GPU 从 58.7 变为 133.9 tokens/s（比值 2.28），手机 GPU 从 18.62 变为 37.38 tokens/s（比值 2.01）。摘要没有保存完整 prompt、调用参数和原始计时日志，Mac 组也缺少输出 token 数。这两组使用不同采集入口，各条件只有单次记录，也没有在同一次运行中记录 \\(r\\)，吞吐比因而确定不了 \\(q\\) 或 \\(c\\)，也估计不了跨设备的稳定差异。

这两个比值可以用来反推口径。在 \\(r=1\\)、\\(q=1+3c\\) 的假设下，2.28 倍和 2.01 倍分别对应 \\(c\approx 0.25\\) 和 \\(c\approx 0.33\\)。注意这个 \\(c\\) 是由端到端结果反推的有效参数，混杂了 verify、drafter 和运行时操作，不是 drafter 模型的实测单步成本；若实际 \\(r\\) 小于 1，反推出的 \\(c\\) 还会更小。作为参照，同一简化模型下即使 \\(r=1\\)，3 倍加速也要求 \\(c\le 1/9\approx 0.11\\)。所以这些数值既不能证明两台设备已达吞吐上限，也不足以把官方结果[^ch09-google-mtp] 归因于某种硬件路径；drafter 文件约 45 MB 同样推不出 \\(c\approx 0.02\\)，模型存储大小与端到端时延不成比例。

投机解码是否加速由 \\(r\\) 和 \\(q\\) 共同决定：聚合接受比例 \\(r\\) 决定每轮平均产出，归一化成本 \\(q\\) 决定取得这些产出要花的时间。两者都随模型、输入内容、生成位置、设备和后端变化。评估目标负载时，需要在同一次测试中记录接受比例与吞吐。

## 9.9　多 token 返回后的停止检测与回退

MTP 还改变了 executor 的返回形态。普通 decode 通常返回一个 token，一次 MTP `Decode()` 则可能返回一段序列。任务层必须按顺序处理这段序列，因为停止序列可能在批次中间命中，BPE 片段也可能跨越两次 MTP 调用。

任务层的单步运行函数先取得 executor 返回的二维 token 序列，检查各候选的序列长度是否相同。随后按位置逐个过滤停止 token，再交给流式 detokenizer：

```cpp
// runtime/core/tasks.cc:302
    for (size_t step = 0; step < sequence_length; ++step) { // (1)
      std::vector<std::vector<int>> step_tokens;
// ...
      ABSL_ASSIGN_OR_RETURN(auto tokens_to_feed,
                            stop_token_filter_.ProcessStep(step_tokens)); // (2)

      ABSL_ASSIGN_OR_RETURN(auto released_outputs,
                            detokenizer_.ProcessStep(tokens_to_feed)); // (3)
```

代码行 `(1)` 将本轮返回序列展开为逐位置处理，省略的几行从每个候选取出当前位置的 token，填入 `step_tokens`。`(2)` 先由停止过滤器暂存可能组成停止序列的 token，只放行已确认不属于停止序列的部分。`(3)` 再让流式 detokenizer 处理放行的 token，提取相邻解码结果的稳定前缀，并暂存尚未稳定的文本。即使 executor 一次返回多个 token，停止检测与文本解码仍按 token 顺序推进。

停止序列在批次中间命中时，代码会调整 executor 的逻辑位置。回退量按 `sequence_length - step` 计算：

```cpp
// runtime/core/tasks.cc:331
      if (all_done) {
        if (step != sequence_length - 1) {
// ...
          int diff = sequence_length - step; // (1)
          ABSL_ASSIGN_OR_RETURN(int current_step, executor_.GetCurrentStep());
          ABSL_RETURN_IF_ERROR(executor_.SetCurrentStep(current_step - diff)); // (2)
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

单步运行函数在每次开始时清空结果，再累积本批可见文本和 token id。外层循环在它返回后至多发送一次 `TaskState::kProcessing` 回调，而且只有确有更新时才发送。循环结束后还会调用 detokenizer 的 `Flush()`；若释放了剩余文本，streaming 模式会再发送一次更新。因此每轮循环至多一次更新，并不意味着整个生成过程的更新都与 executor 调用一一对应。回调次数既不等于 executor 调用次数，也不等于生成 token 数。统计可见输出时应读取回调中的 token id；统计 executor 的位置推进量时应读取 `current_step` 差值。

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

核对一次返回多 token 的边界时，测试 executor 必须能为单个候选返回一段序列。除了逐 token 的常规测试，还应覆盖下面这些批内和跨批条件，再用真实 MTP 模型核对执行器状态：

| 输入序列与条件 | 应检查的结果 | 用途 |
|---|---|---|
| `[a, stop, b, c]`，停止 token 位于下标 1 | 只输出 `a`，最终位置为批前位置加 1 | 验证批中命中及 `diff=3` 的回退 |
| `[a, b, c, stop]`，停止 token 位于末位 | 可见输出、最终位置和 `SetCurrentStep` 调用次数 | 覆盖末位不回退分支 |
| 上一批末尾是停止序列前缀，本批首 token 完成匹配 | 停止过滤器的暂存队列和跨批停止行为 | 验证批边界不改变停止序列语义 |
| 剩余 `max_output_tokens=1`，executor 返回 4 个 token | 输出长度、回调 token id 和最终位置 | 验证后置检查与越界量 |
| 4 个 token 均可见且启用 streaming | 回调次数及单次回调的 token id 数 | 验证一个批次至多一次可见更新 |

> 表 9-3　多 token 返回测试需同时检查可见文本、回调 token id、逻辑位置和回退调用；只检查最终字符串不足以覆盖状态语义。

一次 executor `Decode()` 是一轮运行调用，`current_step` 增量是本轮推进的 token 位置数。MTP 性能分析用调用次数计算轮次成本，用位置增量计算产出和数值终止条件。流式接口还要区分可见 token 与因 BPE 或停止前缀而暂存的 token。

## 9.10　开启条件：模型能力与固定草拟步数

开启 MTP 要满足模型和设置两类条件：模型文件包含 MTP drafter 和相应的 verify signature，执行器设置启用投机解码，并且只使用一个输出候选。草拟步数 \\(G\\) 则由 verify signature 的形状固定。

能力查询扫描模型文件目录中的 TFLite section，将发现的 drafter 信息写入 `supports_speculative_decoding`：

```cpp
// schema/capabilities/capabilities.cc:98
          if (item->key()->string_view() == "model_type") { // (1)
            const auto* value = item->value_as_StringValue();
            if (value == nullptr || value->value() == nullptr) continue;
            absl::string_view model_type = value->value()->string_view();
// ...
            } else if (model_type == "tf_lite_mtp_drafter") { // (2)
              has_speculative_decoding = true;
            }
// ...
  llm_cap.supports_speculative_decoding = has_speculative_decoding; // (3)
```

代码行 `(1)` 读取 TFLite section 的 `model_type` 属性，`(2)` 发现 `tf_lite_mtp_drafter` 后将标志置为 true，`(3)` 将它写入能力结果。这个检查只证明容器包含 drafter section；verify signature、张量类型和后端能否正确执行，还要在创建执行器时验证。第 7 章介绍的 `.litertlm` 容器可以把基础模型和 drafter 放在同一个文件里。

Python CLI 使用 `--speculative-decoding true` 或 `--speculative-decoding false` 指定开关。也可以省略参数值，直接写 `--speculative-decoding` 来启用。不传该选项时，CLI 读取模型配置中的开关；它读取的是调用配置，并非根据能力查询自动启用。Python SDK 接收的值若仍为空，就不调用 C API setter，C++ 默认值为 `false`。执行器构造时根据最终设置决定是否创建 drafter：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:1878
    const auto& advanced_settings = executor_settings.GetAdvancedSettings();
    if (advanced_settings.has_value() &&
        advanced_settings->enable_speculative_decoding) { // (1)
      RET_CHECK_EQ(batch_size, 1)
          << "Speculative decoding (MTP) only supports a single output head.";
// ...
      ABSL_ASSIGN_OR_RETURN(
          mtp_drafter,
          LlmLiteRtMtpDrafter::Create(lrt_env, resources, executor_settings, // (2)
                                      *compiled_model, *embedding_lookup,
                                      ple_manager_opt, executor_metadata));
    }
```

代码行 `(1)` 为真时才创建 drafter，并要求输出候选数为 1。`(2)` 的 `Create` 从模型资源里取出 drafter section，把它编译为独立模型；缺少该 section 或基础模型的 verify signature 时，创建过程返回错误。应用可以先通过 C API 或 Kotlin 的模型能力接口查询支持标志，再决定是否启用。查询返回 true 不能替代执行器创建和真实模型验证。

> **版本注记**：v0.13.1 的 `--enable-speculative-decoding auto` 在未显式设置开关时沿用 C++ 默认值 `false`，表 9-1 保留这一历史口径。v0.17.0 的 CLI 不再接受 `auto`；旧选项名仍接受 `true`、`false`，但会给出弃用提示。

草拟步数 \\(G\\) 也不能在运行时调整，它由模型 verify signature 的形状决定：

```cpp
// runtime/executor/llm_litert_mtp_drafter.cc:412
    LITERT_ASSIGN_OR_RETURN(auto input_pos_tensor_type,
                            verify_signature.InputTensorType("input_pos"));
    // Expecred shape: [T = G + 1] where G is the number of draft steps
    const auto& input_pos_dims = input_pos_tensor_type.Layout().Dimensions();
    num_draft_steps = input_pos_dims[0] - 1; // (1)
```

代码行 `(1)` 用 verify signature 的 `input_pos` 第一维减一得到 \\(G\\)：verify 一次接收 \\(G+1\\) 个位置，其中一个是起始的已确认 token，其余 \\(G\\) 个对应草拟步数。该 verify signature 的输入长度在模型导出时固定，运行时从该形状读取草拟步数；构造采样器时也据此把 verifier 采样器的 `sequence_size` 设为 \\(G+1\\)，使采样器输出与 verify signature 对齐。

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
