# 第 5 章 Decode：逐 token 的心跳

> 使命：走通一次 decode 循环的全部步骤——从模型算出一堆概率，到一个字落到你屏幕上。这是全书最核心的一章，也是理解后面所有性能优化的前提。

提示词吞进去了，KV cache 填好了第一段。现在轮到 decode：模型逐个往外蹦字，直到说完。第 1 章那台"挤牙膏"的手机，挤的就是这一步的每一次心跳。

一次心跳做的事，代码里叫 `DecodeOneStep`（`runtime/core/tasks.cc:111 @ v0.13.1`）。整个 decode 就是一个循环，反复调它，直到该停（`Decode`，`tasks.cc:446 @ v0.13.1`）。这个循环的骨架只有几行，先把它摆出来：

```cpp
// runtime/core/tasks.cc:486 @ v0.13.1
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

整章的脊柱就在这三行编号里。(2) 是那次心跳：一次前向、一次采样、一次转文本，全压在 `run_one_step.Run` 里，返回值 `all_done` 说的是"停止 token 全部命中了吗"。(3) 是每轮末尾的分诊台：`ShouldStop` 综合 `all_done` 和几个计数器，决定 `break` 还是再转一圈。(1) 是每轮开头对取消标志的探测——第 4 章那个 `std::atomic<bool>* cancelled`，decode 循环把它放在最前面看，一旦置位就立刻退出，不必等这一步算完。三个动作凑成一次心跳：探取消、跑一步、判停止。下面把这一步拆开，一层层看。

## 一次心跳的全貌

一次 decode，从"上一个 token"到"下一个 token"，中间要走这么几步：

1. 把上一个 token 喂进模型，模型前向一次，输出一组 **logits**——词表里每个 token 的"分数"，几万个数；
2. （可选）对 logits 做加工：比如压低刚出现过的词的分数（重复惩罚），或屏蔽掉不合语法的词（约束解码，第 10 章）；
3. 从 logits 里**采样**出一个 token id；
4. 把这个 id 转回文本片段，累积或流式吐给用户；
5. 判断该不该停；不停，就把这个新 token 当作"上一个 token"，回到第 1 步。

这五步里，第 1 步是带宽墙的现场（每次前向读一遍全部权重，第 1 章）。其余四步是这一章的主角。它们看着简单，但每一步都藏着一个坑：采样要防长尾胡话，停止要防误判，转文本要防吐出半个字。LiteRT-LM 对每个坑都有交代。

<figure>
{{#include figs/fig-5-1.svg}}
<figcaption>图 5-1　一次 decode 心跳的完整循环。核心分岔是内部采样与外部采样两条路径：前者让执行器一步出 token，后者把 logits 交回上层做可定制的加工与采样。</figcaption>
</figure>

## 两条路径：内部采样与外部采样

采样在这里分成**两条路径**。代码里明明白白写着这是为"内部采样和外部采样"两种情形准备的（`tasks.cc:110 @ v0.13.1` 的注释）。

- **内部采样**：执行器一步到位。你调它 `Decode`，它内部把前向、采样都做完，直接返回一个 token id。快，因为采样可以在 GPU 上就地完成，省掉把几万个 logits 从 GPU 搬回 CPU 的开销（这个"片上采样"的省法，第 8 章细讲）。
- **外部采样**：执行器只做到前向，把整组 logits 交出来（`DecodeLogits`，`tasks.cc:342 @ v0.13.1`），由上层的 logits 处理器和采样器在外面完成加工与采样。慢一点（logits 要出来），但换来了灵活：重复惩罚、约束解码这些"要动 logits"的功能，只能在这条路径上做。

为什么不统一成一条？因为它们服务不同的需求。纯聊天、不需要动 logits 时，走内部采样图快；需要工具调用、结构化输出、约束解码时，走外部采样图灵活。选择权留给场景，而不是强行二选一。`DecodeOneStep` 内部那个 `DecodeAndSample`（`tasks.cc:319 @ v0.13.1`）就是这个分岔的落点。它靠一个成员指针分道：

```cpp
// runtime/core/tasks.cc:319 @ v0.13.1
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

分岔就是 (1) 那个 `if (sampler_)`：构造 `DecodeOneStep` 时传没传采样器，决定了走哪条。外部路径 (2) 调 `executor_.DecodeLogits`，只做前向、把整组 logits 交出来；(3) 是约束解码的挂钩点——`MaskLogits` 在采样前把不合语法的 token 分数压到负无穷（第 10 章）；(4) 把加工过的 logits 交给外部 `Sampler` 采样，顺便写回一个 `scores_tensor_`（采样得分，用于候选排序）。内部路径 (5) 只有一行实质：(6) 的 `executor_.Decode()` 把前向和采样一次做完，logits 不出 GPU，直接返回 token id。

两条路径返回的都是同一种 `std::vector<std::vector<int>>`（外层是候选批、内层是这一步各候选的 token），从 `DecodeAndSample` 出来后，上层代码不再关心它来自哪条路。分岔只在这一个函数里，出口统一，其余代码因此不被两条路径污染。

## 采样：从一堆分数里挑一个字

不管走哪条路径，总要从 logits 里挑出一个 token。怎么挑，就是采样策略。LiteRT-LM 的采样器都实现同一个 `Sampler` 抽象（`runtime/components/sampler.h:34 @ v0.13.1`）。这个抽象只强制一个核心方法：

```cpp
// runtime/components/sampler.h:45 @ v0.13.1
virtual absl::Status SampleToIdAndScoreBuffer(
    const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,   // (1)
    TensorBuffer* scores_tensor) = 0;                              // (2)
```

签名里藏着两个设计取向。(1) 的 `logits_tensor` 形状是 `[batch_size, sequence_size, vocab_size]`，`ids_tensor` 是 `[batch_size, sequence_size]`：采样器一次吃一整批候选，不是一个一个来，因为 `num_output_candidates > 1` 时（束搜索、多候选生成）批处理能摊薄开销。(2) 的 `scores_tensor` 是可空的：传了就把采样到的 token 的对数概率也写回去，上层拿它做候选排序；不需要就传 `nullptr`。方法直接读写 `TensorBuffer` 而不是 `std::vector`，是为了让实现能落在 GPU 上——这个抽象既能包 CPU 采样器，也能包片上采样器（第 8 章）。

常用的一种实现是 `TopPSampler`（`runtime/components/top_p_cpu_sampler.h:30 @ v0.13.1`）。它一个类就覆盖了几种策略，靠参数区分：

```cpp
// runtime/components/top_p_cpu_sampler.h:38 @ v0.13.1
static absl::StatusOr<std::unique_ptr<TopPSampler>> Create(int k, float p,  // (1)
                                                           float temperature,
                                                           int batch_size,
                                                           int sequence_size,
                                                           int seed);        // (2)
```

(1) 一个 `Create` 收 `k`、`p`、`temperature` 三个旋钮，四种策略是它们的组合而非四个类：k 关到极大、temperature 关到 0，就是贪心；给 k、给 p，就是 top-k 叠 top-p；temperature 单独调随机度。(2) 的 `seed` 落到成员 `generator_`（一个 `std::default_random_engine`），采样的随机性全从这一个种子来——固定种子就能复现整段输出，这是调试和基准测量能对齐的前提。四种策略摊开看：

- **贪心（greedy）**：直接挑分数最高的。确定、可复现，但容易呆板、重复。
- **温度（temperature）**：采样前先把分数整体"拉平"或"拉尖"。温度低趋于贪心，温度高更随机、输出更发散。
- **top-k**：只在分数最高的 k 个里挑，掐掉长尾的胡话。
- **top-p（核采样）**：只在累计概率达到 p 的那批里挑，是 top-k 的自适应版——概率集中时候选少，分散时候选多。

这些不是玄学，是可观察的。本章的第一个实验很简单：同一个提示词，温度设 0 和设 1.0 各跑一次，看输出从"每次一样、平实"变成"每次不同、发散"。（实验命令见附录 C，一跑便知；本书不收录这组输出样本。）

| 策略 | 怎么挑 | 特点 |
|---|---|---|
| greedy | 最高分 | 确定、可复现、易重复 |
| temperature | 先缩放再采样 | 调节随机性 |
| top-k | 前 k 个里采样 | 掐掉长尾 |
| top-p | 累计概率 p 内采样 | 自适应候选集 |

> 表 5-1　四种采样策略。它们不是互斥的——实际常常组合使用（如 top-p + 温度）。

## 什么时候停

字不能一直吐下去。每一步之后都要问：该停了吗？这个判断集中在一个纯函数里（`ShouldStop`，`tasks.cc:86 @ v0.13.1`），把所有停止条件收在一处：

```cpp
// runtime/core/tasks.cc:86 @ v0.13.1
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

先看停止词的部分匹配。停止词可能是"###"这样的多字符序列，而模型是一个 token 一个 token 出的。当它刚吐出"#"，你不知道接下来是"##"（真的要停）还是"#号说明"（不该停）。贸然把"#"吐给用户，万一后面真是停止词，你就多吐了不该吐的。LiteRT-LM 的停止符检测器为此逐 token 追踪每条停止序列的匹配进度（`StopTokenDetector`，`runtime/components/stop_token_detector.h:45 @ v0.13.1`；`ProcessTokens` 在 `:67`）。关键是它暴露的这个查询：

```cpp
// runtime/components/stop_token_detector.h:93 @ v0.13.1
// Returns the maximum length of the partial stop token sequence found for the
// given batch index. zero if no partial stop token sequence is found or -1 if
// the stop token is already found.
int MaxPartialStopTokenLength(int index) const;  // (1)
```

(1) 这个函数回答的正是"现在匹配了多长"：返回 0 表示当前 token 跟任何停止序列都不沾边、可以放心吐；返回正数 `L` 表示末尾 `L` 个 token 构成了某条停止序列的前缀，悬而未决；返回 -1 表示停止序列已完整命中。`Run` 就靠这个返回值决定攥住多少 token：

```cpp
// runtime/core/tasks.cc:187 @ v0.13.1
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

这段是"攥住—回吐"的全部机制。(1) 一旦 `max_length > 0`，就把这个 token 的文本推进 `pending_stop_tokens_[i]` 这个队列，先不吐——它可能是停止词的一部分。(2) 的 `while` 是回吐逻辑：队列里攥着的 token 数一旦超过 `max_length`，说明多攥的那些**不可能**再是停止序列的前缀了（停止序列最多 `max_length` 长），于是 (3) 把队首多出来的部分放进 `result_text_` 正常吐出。攥在手里的永远只是"最后 `max_length` 个 token"这个滑动窗口。(4)(5) 是最常见的情况：`max_length == 0`，当前 token 跟停止词无关，直接进 `result_text_`，零延迟。

那停止序列真命中了怎么办？此时 `AllDone()` 会返回真、`GetStopTokensFound()[i]` 置位，队列里攥着的整段就再也不会走到 (3) 被吐出——它们连同停止词一起被丢弃。这就是"回吐"的另一半：攥住的东西，看清是停止词就**整段作废**，不是就顺次放出。代价是命中前缀期间输出会滞后至多 `max_length` 个 token，换来的是绝不会把半截停止词漏给用户。

再看"半个字"。子词分词（第 3 章）意味着一个 token 未必是一个完整的字，尤其是中文和 emoji，一个字可能由好几个 token 拼成。decode 一步只出一个 token，如果它是某个字的前半截，直接转文本就是乱码。`Run` 里处理这个的是同一套暂存逻辑，但用另一个成员队列：

```cpp
// runtime/core/tasks.cc:175 @ v0.13.1
ASSIGN_OR_RETURN(step_tokens, tokenizer_.MergeTokenIds(              // (1)
                                  bpe_partial_token_ids_, step_tokens));
auto decoded_result =
    tokenizer_.TokenIdsToTexts(num_output_candidates_, step_tokens);
for (int i = 0; i < num_output_candidates_; ++i) {
  if (Tokenizer::IsIncompleteBpeSequence(decoded_result.value()[i])) {  // (2)
    bpe_partial_token_ids_[i] = step_tokens[i];                         // (3)
  } else if (!stop_token_detector_.GetStopTokensFound()[i]) {
    bpe_partial_token_ids_[i].clear();                                  // (4)
    // ... 前面那段停止词攥住/回吐逻辑 ...
```

顺序是关键。(1) 每一步先把上一步攒下的半截 token（`bpe_partial_token_ids_`）跟这一步的新 token 拼起来，再一起转文本：转文本用的从来不是孤立的一个 token，而是"可能补全了的一串"。(2) `IsIncompleteBpeSequence` 判断拼出来的这串是不是仍然凑不成一个完整字符（比如一个 UTF-8 多字节序列缺了尾巴）；(3) 如果还不完整，就把这串整个存回 `bpe_partial_token_ids_[i]`，这一步一个字都不吐，等下一步再拼。(4) 一旦凑成完整字符，先清空暂存，再进入前面那段停止词判定——两层缓冲是串联的：先过 BPE 补全这关，够成完整字符了，才轮到停止词那关判要不要攥。

两个队列各管一件事：`bpe_partial_token_ids_` 保证吐出去的是完整字符，`pending_stop_tokens_` 保证吐出去的不含半截停止词。它们共用 `Run` 里那个 `for (int i ...)` 逐候选走一遍，互不干扰。

这两个坑（停止词回吐、半字暂存）共同的模式是：**decode 是逐 token 的，但用户要的是完整、正确的文本单元，中间需要一层缓冲把"逐 token"翻译成"逐可显示单元"。** 看懂这层缓冲，你就看懂了流式生成为什么不是"算一个吐一个"这么简单。

## 小结

一次 decode 心跳：前向出 logits → （可选）加工 → 采样出 token → 转文本流式吐出 → 判断停止。其中有三处设计：`DecodeAndSample` 里那个 `if (sampler_)` 分出内部/外部两条采样路径（快与灵活的取舍，出口统一成同一种 token 向量）、`ShouldStop` 用四个 `else if` 把停止逻辑单独收拢成纯函数、以及 `Run` 里 `bpe_partial_token_ids_` 和 `pending_stop_tokens_` 两个队列串联起来的缓冲——前者保证吐出的是完整字符，后者保证不含半截停止词，共同把"逐 token"翻译成"逐可显示单元"。

下一部（第 6-9 章），我们回头凿墙——先算清楚 KV cache 到底占了多少，以及那条 25 tokens/s 的上限，实测为什么还够不着。

---

## 参考

- decode 循环与单步：`runtime/core/tasks.cc @ v0.13.1`。本章贴出：主循环骨架（`while (true)`:486，含取消探测与 `ShouldStop` 调用）；`ShouldStop` 纯函数全文（:86）；`DecodeAndSample` 内/外采样分岔（:319，外部路径 `DecodeLogits`:342、`MaskLogits`、`SampleToIdAndScoreBuffer`，内部路径 `executor_.Decode()`）；`Run` 里的 BPE 补全（`MergeTokenIds`:175、`IsIncompleteBpeSequence`:181）与停止词攥住/回吐（:187）。相关：`DecodeOneStep` 类:111；`Decode` 入口:446；内/外采样注释:109。
- 采样器：`runtime/components/sampler.h @ v0.13.1`（`Sampler` 抽象:34，贴出核心方法 `SampleToIdAndScoreBuffer`:45）；`runtime/components/top_p_cpu_sampler.h @ v0.13.1`（`TopPSampler`:30，贴出 `Create` 签名:38，含 k/p/temperature/seed）。
- 停止符检测：`runtime/components/stop_token_detector.h @ v0.13.1`（`StopTokenDetector`:45；`ProcessTokens`:67；贴出 `MaxPartialStopTokenLength`:93 含返回值语义注释；`GetStopTokensFound`:89；`AllDone`:85）。

<!-- 实验数字（温度对比、停止词回吐用例）待基准数据集采集后回填。 -->
