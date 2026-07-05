# 第 9 章 一次前向，多个 token：推测解码与 MTP

> 使命：讲透"用猜的反而更快"这件反直觉的事——drafter 怎么草拟、base 模型怎么一次验证一串、以及快多少为什么全看接受率。这是第三部的压轴，讲端侧怎么从带宽墙上再抠出成倍的速度。

第 1 章立带宽墙时，给过两条出路：提高带宽（端侧作者说了不算），或减少每 token 要读的字节（量化，第 7 章做了）。这一章是第三条——**别改带宽，也别改模型，改成一次前向多吐几个字**。这就是第 2 章"二十个问题"的第 19 问：推测解码靠"猜"，为什么反而更快？什么时候又更慢？

## 一次昂贵前向，能不能多榨几个字

先把账摆清。decode 的瓶颈是带宽：每生成一个 token，都要把全部权重读一遍（第 1 章）。这一次前向很贵，而它只换来一个字。问题自然浮现：**这一次昂贵的前向，能不能一次多榨出几个字？**

推测解码的回答是：能，只要你愿意先"猜"。找一个又小又快的模型先草拟接下来的几个字，再让大模型**一次前向**把这几个字一起验一遍。猜对的部分直接采用，猜错的地方用大模型的正确答案兜底。关键在于——验证那几个草拟字，只花大模型**一次**前向，而不是几次。一次昂贵前向，摊到了多个字上。

推测解码有几种形态：草稿模型可以是完全独立的另一个模型，也可以是和主模型共享主干的多头结构（MTP，Multi-Token Prediction，Gemma 4 走的就是这一路）。在 LiteRT-LM 的运行时里，drafter 装载为一个独立的小模型（成员 `mtp_drafter_model_`），验证则用 base 模型上一个专门的 `"verify"` signature（常量定义 `runtime/executor/llm_litert_mtp_drafter.cc:62`，取用 `base_model.FindSignature(kVerifySignatureRunner)` 在 `:227 @ v0.13.1`）——所以不管训练时共不共享主干，运行时看到的都是同一套"小模型草拟 + 大模型验证"。

## 机制：草拟，然后一次验一串

一次 `Draft()`（`runtime/executor/llm_litert_mtp_drafter.cc:453 @ v0.13.1`）分三步，函数体本身就是这三步的骨架：

```cpp
ASSIGN_OR_RETURN(std::vector<int> drafted_tokens,
                 RunDraftingLoop(token_id, activations));          // (1)

RETURN_IF_ERROR(PrepareVerifierInputBuffers(
    position, token_id, drafted_tokens, input_kv_cache_buffers));  // (2)
// ...
ASSIGN_OR_RETURN(std::vector<int> verifier_id_vector, RunVerification());  // (3)
```

`(1)` 草拟，`(2)` 把"上一个真 token + 草拟出的那串"拼成 verify 的输入，`(3)` base 模型一次前向验完。三步之后才是接受循环。

**第一步，草拟。** drafter 逐个吐出接下来的 G 个候选 token（`RunDraftingLoop`，`:328`）。G 是草拟步数（代码里 `num_draft_steps_`）。循环体一步一个 token：

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

`(1)` 循环 G 次，每次一个 token，串行。`(3)` 每步跑的是 `mtp_drafter_model_`——一个**独立装载的小模型**，不是 base 模型，这是运行时能便宜草拟的前提。`(2)` 是 MTP 这一路的特征：drafter 的输入不只是词嵌入，还拼上了上一步的隐藏态 activation（源码注释写明拼完是 `[B=1, T=1, D=3072]`，即 1536 的词嵌入接 1536 的 activation），让草稿头能"续着 base 模型的思路"往下猜。`(4)` 每步只出一个 token（`RET_CHECK_EQ(..., 1)` 兜底）。`(5)` 把这一步的输出喂回下一步的输入：drafter 是自回归的，只是它自回归很便宜。这几步再串行，也远不及大模型一次前向贵。

**第二步，一次验一串。** 把"上一个真 token + 草拟的 G 个"拼起来，交给 base 模型的 verify signature（`RunVerification`，`:437`）：

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

`(1)` 一次 `RunAsync` 就是整套机制里最贵、也是唯一一次 base 模型前向；G 个位置的验证全压在这一次里。`(2)` 它吐回长度 G+1 的 id 序列——对 G+1 个位置各给出一个"正确答案"，多出来的那一位后面会派上用场。验证 G 个草拟字只用 base **一次**前向，这正是提速的来源。

**第三步，接受。** 逐位比对草拟和验证结果，接受最长的匹配前缀。

<figure>
{{#include figs/fig-9-1.svg}}
<figcaption>图 9-1　推测解码一轮：drafter 逐个草拟 G 个 token，base 模型一次前向验证 G+1 个位置，接受最长匹配前缀再加一个 bonus token。一次昂贵前向，产出 1 到 G+1 个 token。</figcaption>
</figure>

## 接受循环：匹配前缀，加一个 bonus 兜底

第三步值得逐行看，因为它藏着一个兜底设计（`Draft()` 接受循环，`:471`–`:484`）：

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

`(1)` 从头逐位比对草拟和验证。`(3)` 一致就接受、`num_correct_tokens` 加一。`(2)` 一旦碰到第一个不一致就停，把 base 模型在这个位置给出的正确 token 记为 **bonus**——这一位草拟错了，但 base 已经算出了对的，不浪费。`(4)` 循环走完 `bonus_token` 还是 −1，说明 G 个全对；`(5)` 这时直接收下第 G+1 个位置的验证结果：verify 本来就多算了这一位（上一步 `RunVerification` 返回 G+1 个 id），全对时它就是白送的第 G+1 个字。

被反复更新的 `last_verified_token_id_idx_` 不是记账用的：它记住接受前缀在 verify 输出缓冲里的下标，下一轮 `RunDraftingLoop` 走 `ConcatenateEmbeddingsAndActivationsFromVerifierBuffer` 分支时，正是靠这个下标取出接受位置的 activation 作为草拟的新起点——接受循环和下一轮草拟就这样接上了。

最后输出"接受的前缀 + 一个 bonus"（`:492`–`:493`）：

```cpp
// The first token comes from the decode output and is always correct.
drafted_tokens.resize(num_correct_tokens);   // (1)
drafted_tokens.push_back(bonus_token);       // (2)
```

`(1)` 把草拟序列截断到接受的长度，`(2)` 接上 bonus。源码注释点明"第一个 token 来自 decode 输出，永远正确"——所以哪怕 `num_correct_tokens` 为 0（第一个字就猜错），`resize(0)` 后 `push_back` 仍留下 1 个 bonus token。

这就是保底：**哪怕第一个字就猜错，你也能拿到 base 给的 1 个正确 token。** 论产出的 token 数，推测解码永远不会比普通 decode 差——最坏一次前向出一个字，和普通 decode 打平。赚的时候赚很多（一次前向出 G+1 个字），亏的时候不亏 token（只是白做了 drafter 的功）。

## 接受率经济学：快多少，看你猜得准不准

现在回答第 19 问的后半段：什么时候反而更慢？

关键指标是**接受率**——草拟的字里有多少被接受。每一轮结束，`Draft()` 累加两个计数（`:494`–`:495`）：

```cpp
num_drafted_tokens_ += num_draft_steps_;    // (1)
num_verified_tokens_ += num_correct_tokens; // (2)
```

`(1)` 分母，每轮固定加 G——这一轮草拟了多少，与接受多少无关。`(2)` 分子，只加真正被接受的 `num_correct_tokens`（不含 bonus，bonus 是 base 免费给的，不算 drafter 的功劳）。drafter 析构时把这个比值打出来（`:164`–`:172`）：

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

`(1)` 这个 "Success rate" 就是接受率，只在进程退出时打一次。它落在日志里、CLI 不透出——这也是后面本书实测无法实证归因接受率的直接原因。分子分母的定义决定了一个上界：G 步全接受时接受率是 1.0，bonus 不进分子，所以这个数永远到不了"平均每轮产出 token 数 / G"那个更乐观的口径。

算一笔经济账。一轮推测解码的成本 ≈ drafter 草拟 G 步 + base 一次前向；产出是 1 到 G+1 个 token。

- 接受率高时：一次 base 前向出好几个字，而 base 前向是最贵的那一项——于是每个字摊到的成本大降，明显更快。官方报告 Gemma 4 上可达约 3 倍（官方博客口径，【文档】级）。
- 接受率低时：草拟大多被丢弃，drafter 那几步白做，还多搭了 verify 的开销。如果这些额外开销超过了省下的前向，净结果就是**更慢**。

这里要如实报告本书自己的实测〔基准 D〕：主基准 Gemma 4 E4B 在基准机上开/关 MTP，cpu 22.8 → 24.9、gpu 50.0 → 50.2 tok/s（cpu 那一批三次运行本身就散布在 20.1-24.9 之间，这些差异都落在批内抖动幅度里；强制开启也正常运行、49.0），**没有复现 3 倍**。这不推翻机制，而与它相容：benchmark 喂的是合成负载，我们推测其接受率很低（近似"最难猜的文体"），收益自然出不来——但 CLI 不输出接受率，无法实证归因；官方口径来自其特定的模型、硬件与负载。收益全看接受率，这句话对两头都成立。

这正是那个反直觉现象的出处（第 19 问后半，`LiteRT-LM#2227`）：在某些 GPU（如 PowerVR）上，MTP 反而拖慢了 decode。成因可以从上面的账推出来——当接受率不够高、或 drafter/verify 在那块硬件上的相对开销偏大时，这笔赌注就亏了。（该现象无真机可复现，按上游报告的【文档】级引用。）

所以推测解码不是无条件的加速，是一个赌注：**赌 drafter 猜得够准**。文体也影响赔率——套路性强的文本（比如代码）好猜、接受率高，天马行空的散文难猜、接受率低。

## 它不是随便就能开的

要开推测解码，得先过两道门槛：模型自己声明支持，以及草拟步数 G 早在导出时就定死。

其一，一个模型支不支持推测解码，写在它的能力声明里。运行时用 `HasSpeculativeDecodingSupport`（`schema/capabilities/speculative_decoding.h:33`、`:44 @ v0.13.1`）判断——头文件给了两个重载，一个吃 `std::istream&`、一个吃文件路径，后者只是打开文件转调前者。真正的判断在 `.cc` 里，机制很朴素（`speculative_decoding.cc:40`–`:73`）：

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

`(2)` 遍历 `.litertlm` 的所有 section，`(3)` 找每个 TFLite 模型 section 的 `model_type` 元数据，`(4)` 只要有一个等于 `(1)` 里那个字符串 `"tf_lite_mtp_drafter"` 就返回 true。换句话说，"支持推测解码"不是一个布尔开关，而是**文件里有没有打包一个 model_type 标为 mtp drafter 的子模型**——这正是第 7 章说的 `.litertlm` 容器里那些 section 的用途之一：drafter 就是和 base 模型打包在同一个文件里的另一个 section。CLI 的 `--enable-speculative-decoding=auto` 就是让运行时照这个探测结果自动决定开不开。

其二，草拟步数 G 不是运行时随便调的旋钮。它由模型 verify signature 的形状固定（`:250`–`:256`）：

```cpp
LITERT_ASSIGN_OR_RETURN(auto input_pos_tensor_type,
                        verify_signature.InputTensorType("input_pos"));
// Expecred shape: [T = G + 1] where G is the number of draft steps
const auto& input_pos_dims = input_pos_tensor_type.Layout().Dimensions();
num_draft_steps = input_pos_dims[0] - 1;                         // (1)
```

`(1)` G 直接读自 verify signature 的 `input_pos` 张量维度减一——signature 能同时喂进几个位置，G 就是几。verify 一次吃 G+1 个位置（源码注释 `[T = G + 1]`），减去起头那个真 token，就是草拟步数。一个模型草拟几步，在它被导出、signature 形状定死的那一刻就固定了，运行时只能读、不能改。这是第 4 章"固定形状"约束在推测解码上的又一次体现：连"猜几步"这个看似是超参的东西，也被烘焙进了 signature。

## 小结

推测解码是凿带宽墙的第三招：不改带宽、不改模型，靠一个便宜的 drafter 先猜、base 模型一次前向验一串，把最贵的那次前向摊到多个 token 上。bonus 设计保证它论 token 数不亏，但接受率决定它到底快多少——赌 drafter 猜得准，赌赢了成倍加速，赌输了（接受率低、开销大）反而更慢。它是一个有条件的加速，条件就是接受率够高。

第三部到此收尾。三堵墙，我们各凿了一遍：KV cache 与量化对付内存与带宽，异构后端对付硬件差异，推测解码则在接受率够高时，能从带宽墙上再抠出成倍的速度。下一部，我们走出纯文本，去看这套运行时怎么长出多模态、工具调用这些能力，以及它如何变成六种语言的 SDK。

---

## 参考

- MTP drafter 实现：`runtime/executor/llm_litert_mtp_drafter.cc @ v0.13.1`（`Draft` 三步骨架:453、463-469；`RunDraftingLoop` 循环体:328、336-370；`RunVerification`:437、440-450；接受循环:471-484；输出:492-493；接受率统计:494-495、析构打印:164-172；`num_draft_steps` 由 verify signature 形状定:250-256；`"verify"` signature 常量:62、取用:227；drafter 独立小模型成员 `mtp_drafter_model_`）。
- 能力声明：`schema/capabilities/speculative_decoding.h @ v0.13.1`（`HasSpeculativeDecodingSupport`:33、44）；探测逻辑 `schema/capabilities/speculative_decoding.cc:40-73`（扫 section 的 `model_type` 是否为 `"tf_lite_mtp_drafter"`）。
- "约 3 倍"：Google 官方博客 *Accelerating Gemma 4: faster inference with multi-token prediction drafters*，https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/（经 LiteRT-LM 仓库 README 索引，访问 2026-07-05）。本书实测未复现，见〔基准 D〕。

<!-- MTP 开/关已实测（附录 D）：差异在抖动内、未复现 3x，正文已如实报告并以接受率经济学解释。文体接受率对比未做（CLI 不输出接受率）。#2227 无真机，成因为经济账推断、按【文档】级引 issue。图 9-2(接受率-收益曲线) 表 9-1(开/关实测) 需实测数据，待基准 D；本轮出签名图 9-1(时序)。 -->
