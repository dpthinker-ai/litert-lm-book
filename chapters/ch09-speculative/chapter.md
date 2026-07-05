# 第 9 章 一次前向，多个 token：推测解码与 MTP

> 使命：讲透"用猜的反而更快"这件反直觉的事——drafter 怎么草拟、base 模型怎么一次验证一串、以及快多少为什么全看接受率。这是第三部的压轴，讲端侧怎么从带宽墙上再抠出成倍的速度。

第 1 章立带宽墙时，给过两条出路：提高带宽（端侧作者说了不算），或减少每 token 要读的字节（量化，第 7 章做了）。这一章是第三条——**别改带宽，也别改模型，改成一次前向多吐几个字**。这就是第 2 章"二十个问题"的第 19 问：推测解码靠"猜"，为什么反而更快？什么时候又更慢？

## 一次昂贵前向，能不能多榨几个字

先把账摆清。decode 的瓶颈是带宽：每生成一个 token，都要把全部权重读一遍（第 1 章）。这一次前向很贵，而它只换来一个字。问题自然浮现：**这一次昂贵的前向，能不能一次多榨出几个字？**

推测解码的回答是：能，只要你愿意先"猜"。找一个又小又快的模型先草拟接下来的几个字，再让大模型**一次前向**把这几个字一起验一遍。猜对的部分直接采用，猜错的地方用大模型的正确答案兜底。关键在于——验证那几个草拟字，只花大模型**一次**前向，而不是几次。一次昂贵前向，摊到了多个字上。

推测解码有几种形态：草稿模型可以是完全独立的另一个模型，也可以是和主模型共享主干的多头结构（MTP，Multi-Token Prediction，Gemma 4 走的就是这一路）。LiteRT-LM 的实现里，drafter 是一个独立的小模型，验证则用 base 模型上一个专门的 `"verify"` signature（`runtime/executor/llm_litert_mtp_drafter.cc:63`、`:228 @ v0.13.1`）。名字不同，内核一致。

## 机制：草拟，然后一次验一串

把一次 `Draft()`（`llm_litert_mtp_drafter.cc:453 @ v0.13.1`）拆开看，三步。

**第一步，草拟。** drafter 小模型逐个吐出接下来的 $G$ 个候选 token（`RunDraftingLoop`，`:328`，循环在 `:336`）。$G$ 是草拟步数（代码里 `num_draft_steps`）。drafter 小，所以这几步便宜——就算它串行地一个个猜，也远不及大模型一次前向贵。

**第二步，一次验一串。** 把"上一个真 token + 草拟的 $G$ 个"拼起来，交给 base 模型的 verify signature，**一次前向**跑完（`RunVerification`，`:437`，`base_model_.RunAsync(verify_signature)`）。它一口气对 $G+1$ 个位置各给出一个"正确答案"，返回一个长度 $G+1$ 的 id 序列（`:449` 有 `RET_CHECK_EQ(id_vector.size(), num_draft_steps_ + 1)` 兜底）。注意这里的要害：验证 $G$ 个草拟字，只用了 base 模型**一次**前向——这正是提速的来源。

**第三步，接受。** 逐位比对草拟和验证结果，接受最长的匹配前缀。

<figure>
{{#include figs/fig-9-1.svg}}
<figcaption>图 9-1　推测解码一轮：drafter 逐个草拟 G 个 token，base 模型一次前向验证 G+1 个位置，接受最长匹配前缀再加一个 bonus token。一次昂贵前向，产出 1 到 G+1 个 token。</figcaption>
</figure>

## 接受循环：匹配前缀，加一个 bonus 兜底

第三步值得逐行看，因为它藏着一个漂亮的兜底设计（`Draft()` 接受循环，`:473`–`:483`）：

```cpp
int num_correct_tokens = 0;
int bonus_token = -1;
for (int i = 0; i < num_draft_steps_; ++i) {
  last_verified_token_id_idx_ = i;
  if (verifier_id_vector[i] != drafted_tokens[i]) {
    bonus_token = verifier_id_vector[i];   // 首个猜错处，用 base 的正确答案
    break;
  }
  num_correct_tokens++;                     // 猜对，接受
}
if (bonus_token == -1) {                     // 全猜对
  last_verified_token_id_idx_ = num_draft_steps_;
  bonus_token = verifier_id_vector[num_draft_steps_];  // 白赚第 G+1 个
}
```

逻辑是这样：从头逐位比，草拟和验证一致就接受、计数加一；一旦碰到第一个不一致，就停下，取 base 模型在这个位置的正确 token 当作 **bonus**。如果一路全对，那就白赚——verify 本就多算了一个位置（第 $G+1$ 个），直接把它当 bonus 收下。

最后输出"接受的前缀 + 一个 bonus"（`:490`–`:492`，源码注释点明"第一个 token 来自 decode 输出，永远正确"）。

这个 bonus 设计是保底：**哪怕第一个字就猜错，你也能拿到 base 给的 1 个正确 token。** 所以论产出的 token 数，推测解码永远不会比普通 decode 差——最坏也是一次前向出一个字，和普通 decode 打平。它赚的时候赚很多（一次前向出 $G+1$ 个字），亏的时候不亏 token（只是白做了 drafter 的功）。

## 接受率经济学：快多少，看你猜得准不准

现在回答第 19 问的后半段：什么时候反而更慢？

关键指标是**接受率**——草拟的字里有多少被接受。代码在析构时就打印它（`num_verified_tokens_ / num_drafted_tokens_`，`:165`–`:171`），足见它是这套机制的命门。每一轮，代码累加两个数：草拟了多少（`num_drafted_tokens_ += num_draft_steps_`）、接受了多少（`num_verified_tokens_ += num_correct_tokens`，`:493`–`:494`）。

算一笔经济账。一轮推测解码的成本 ≈ drafter 草拟 $G$ 步 + base 一次前向；产出是 1 到 $G+1$ 个 token。

- 接受率高时：一次 base 前向出好几个字，而 base 前向是最贵的那一项——于是每个字摊到的成本大降，明显更快。官方报告 Gemma 4 上可达约 3 倍（官方博客口径，【文档】级）。
- 接受率低时：草拟大多被丢弃，drafter 那几步白做，还多搭了 verify 的开销。如果这些额外开销超过了省下的前向，净结果就是**更慢**。

这里要如实报告本书自己的实测〔基准 D〕：主基准 Gemma 4 E4B 在基准机上开/关 MTP，decode 差异落在运行间抖动范围内（gpu 50.0 → 50.2 tok/s；强制开启也正常运行、49.0），**没有复现 3 倍**。这不推翻机制，反而印证了它：benchmark 模式喂的是合成负载，近似"最难猜的文体"，接受率上不去，收益自然出不来；官方口径来自其特定的模型、硬件与负载。收益全看接受率——这句话对两头都成立。

这正是那个反直觉现象的出处（第 19 问后半，`LiteRT-LM#2227`）：在某些 GPU（如 PowerVR）上，MTP 反而拖慢了 decode。成因可以从上面的账推出来——当接受率不够高、或 drafter/verify 在那块硬件上的相对开销偏大时，这笔赌注就亏了。（该现象无真机可复现，按上游报告的【文档】级引用。）

所以推测解码不是无条件的加速，是一个赌注：**赌 drafter 猜得够准**。文体也影响赔率——套路性强的文本（比如代码）好猜、接受率高，天马行空的散文难猜、接受率低。

## 它不是随便就能开的

最后两个务实的点。

其一，一个模型支不支持推测解码，是写在它的能力声明里的。运行时用 `HasSpeculativeDecodingSupport`（`schema/capabilities/speculative_decoding.h:33`、`:44 @ v0.13.1`）读模型元数据来判断——这正是第 7 章说的 `.litertlm` 里那段"能力声明"的用途之一。CLI 的 `--enable-speculative-decoding=auto` 就是让运行时照这个声明自动决定开不开。

其二，草拟步数 $G$ 不是运行时随便调的旋钮。它由模型 verify signature 的形状固定（`num_draft_steps` = verify 的 `input_pos` 维度减一，`:256`）——也就是说，一个模型草拟几步，在它被做出来时就定死了。这是第 4 章"固定形状"约束在推测解码上的又一次体现。

## 小结

推测解码是凿带宽墙的第三招：不改带宽、不改模型，靠一个便宜的 drafter 先猜、base 模型一次前向验一串，把最贵的那次前向摊到多个 token 上。bonus 设计保证它论 token 数不亏，但接受率决定它到底快多少——赌 drafter 猜得准，赌赢了成倍加速，赌输了（接受率低、开销大）反而更慢。它是一个有条件的加速，条件就是接受率够高。

第三部到此收尾。三堵墙，我们各凿了一遍：KV cache 与量化对付内存与带宽，异构后端对付硬件差异，推测解码再从带宽墙上抠出成倍的速度。下一部，我们走出纯文本，去看这套运行时怎么长出多模态、工具调用这些能力，以及它如何变成六种语言的 SDK。

---

## 参考

- MTP drafter 实现：`runtime/executor/llm_litert_mtp_drafter.cc @ v0.13.1`（`Draft`:453；`RunDraftingLoop`:328；`RunVerification`:437；接受循环:473-483；输出:490-492；接受率统计:165-171、493-494；`num_draft_steps`:256；verify signature:63、228）。
- 能力声明：`schema/capabilities/speculative_decoding.h @ v0.13.1`（`HasSpeculativeDecodingSupport`:33、44）。
- "约 3 倍"：Google 官方博客 *Accelerating Gemma 4: faster inference with multi-token prediction drafters*，https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/（经 LiteRT-LM 仓库 README 索引，访问 2026-07-05）。本书实测未复现，见〔基准 D〕。

<!-- MTP 开/关已实测（附录 D）：差异在抖动内、未复现 3x，正文已如实报告并以接受率经济学解释。文体接受率对比未做（CLI 不输出接受率）。#2227 无真机，成因为经济账推断、按【文档】级引 issue。图 9-2(接受率-收益曲线) 表 9-1(开/关实测) 需实测数据，待基准 D；本轮出签名图 9-1(时序)。 -->
