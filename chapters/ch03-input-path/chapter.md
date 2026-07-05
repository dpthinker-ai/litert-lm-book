# 第 3 章 输入之路：从 Engine API 到 token 序列

> 使命：走通输入侧的全程——你敲进去的一句话，如何变成模型能吃的一串数字。沿途会遇到本书第一个真正精巧的设计：多轮对话怎么做到不重算历史。

第二部是全书的脊柱：跟着一个 token 走完它的一生。这一章是它的上半程——从你调用 API，到一串 token id 准备好被喂进模型。生成还没开始，但成败已经埋在这里。

## 门面：为什么是 Engine 和 Session 两层

打开 LiteRT-LM 的对外接口，第一眼看到的是两个类：`Engine` 和 `Session`。使用者的典型动作是：用 `Engine` 开一个 `Session`，再让 `Session` 生成内容（`runtime/engine/engine.h:55 @ v0.13.1` 的注释里就是这个例子）。

为什么要分两层？因为它们的生命周期和成本完全不同。

- **Engine 重**。它持有模型权重——那 1.86 GiB（第 1 章）。加载一次要几秒、占几 GiB 内存。它应该被创建一次、长期复用。
- **Session 轻**。它代表一次对话，持有的是这次对话的状态：KV cache、采样配置、步数。它可以随开随关，一个 Engine 能开出多个 Session。

这个划分不是随手为之，它对应第 2 章那条"状态即对象"原则：把"不变的、昂贵的"（权重）和"多变的、廉价的"（对话状态）分到两级，各自有各自的生命周期。`SessionInterface` 这个抽象（`runtime/engine/engine.h:70 @ v0.13.1`）定义了一次会话能做的事——它同时暴露了高层和低层两套接口，这个"两套接口"的设计后面两章都要用到：

- 高层：`GenerateContent`（`engine.h:112 @ v0.13.1`）一次给完整结果，`GenerateContentStream`（`engine.h:128 @ v0.13.1`）流式回调逐段吐字；
- 低层：`RunPrefill`（`engine.h:174 @ v0.13.1`）和 `RunDecode`（`engine.h:188 @ v0.13.1`）把 prefill 和 decode 拆开，让调用方能精细控制。高层接口不过是这两步的组合。

第 4 章讲 `RunPrefill` 里发生了什么，第 5 章讲 `RunDecode`。这一章先把输入准备好。

## 从消息到文本：对话与模板

多数使用者不直接摆弄 token，而是发"消息"：一句 user 说的话，期待一句 model 的回答。把消息组织成模型认得的格式，是 `Conversation` 这一层的活（第 2 章五层架构里的"对话与编排层"）。

模型并不理解"谁是 user、谁是 model"。它只认一长串文本，其中用特殊的标记把角色圈出来。把结构化的消息渲染成这样一长串带标记的文本，靠的是**聊天模板**（prompt template）。不同模型的模板不同——Gemma、Qwen 各有各的圈法，LiteRT-LM 为此准备了按模型类型分派的处理器（`model_data_processor`，第 10 章的工具调用还会回到它）。`ConversationConfig`（`runtime/conversation/conversation.h:56 @ v0.13.1`）就是配置这一层行为的地方：开场白、模板、是否开启约束解码，都在这里设。

到这一步，你的一句"帮我改写这段"已经变成了一长串带角色标记的纯文本。下一步该把它切成 token 了——但在那之前，有一个多轮对话绕不开的问题。

## 不重算历史：模板 diff 增量渲染

问题是这样的。多轮对话里，第二轮的输入在逻辑上是"历史全文 + 新消息"。如果每一轮都把整段历史重新渲染、重新 prefill，那么对话越长，每轮的开销越大——第十轮要把前九轮重算一遍。这正是第 1 章带宽墙在对话场景下的放大。

LiteRT-LM 的对策朴素而有效：**只 prefill 新增的那一小段。**

它的做法是做一次"文本差"（diff）。把"只含旧消息"渲染成一个字符串，再把"旧消息 + 新消息"渲染成另一个字符串，两者相减，多出来的那一截，就是本轮真正需要 prefill 的增量文本。旧的部分早已在 KV cache 里（第 6 章），无需重来。

它把"逻辑上每轮都是全量历史"翻译成了"物理上每轮只处理增量"。表面看只是个字符串相减，背后接住的是整个 KV cache 复用的收益。

> 版本注记
> diff 的具体实现（在 `runtime/conversation/conversation.cc`）涉及模板渲染前如何剥离图像等大块数据、如何处理边界，细节较多，属本章补读清单里未清零的一项。本书正文只承诺讲清"为什么做 diff、diff 出的是什么"这一层设计意图；精确到行的实现走查，留待补读完成后回填，不在此处臆测行号。

## 从文本到数字：两种 tokenizer

增量文本有了，最后一步是把它切成 token id——模型只吃数字。做这件事的叫 tokenizer，LiteRT-LM 支持两种（`runtime/components/` 下）：

- **SentencePiece**（`sentencepiece_tokenizer.h @ v0.13.1`）：Gemma 等模型用的分词器，基于子词单元；
- **HuggingFace**（`huggingface_tokenizer.h @ v0.13.1`）：兼容 HuggingFace 生态的分词器。

两者都实现同一个 `tokenizer.h` 抽象接口——又一次"接口隔离"原则：上层只管"把这段文本变成 id 序列"，不关心底下是哪种分词器。这也解释了第 5 章末尾那个"吐半个字"的现象的一半来由：子词分词意味着一个 token 未必是一个完整的字，跨 token 的边界要小心处理。

至此，输入之路走完。你敲进去的一句话，历经"消息 → 套模板 → diff 增量 → 分词"，变成了一串准备好的 token id。

<figure>
{{#include figs/fig-3-1.svg}}
<figcaption>图 3-1　输入侧数据流：一句话经对话模板渲染、与历史做 diff 取增量、再分词，最终成为一串 token id。只有增量部分需要 prefill——这是多轮对话不重算历史的关键。</figcaption>
</figure>

## 小结

输入侧有两个关键设计：Engine/Session 的两级抽象（把昂贵的权重和廉价的对话状态分开），以及模板 diff 增量渲染（把逻辑全量翻译成物理增量）。一个管空间，一个管时间。

那串 token id 现在躺在门口。下一章，`RunPrefill` 会把它一口吞进模型。

---

## 参考

- Engine / Session 接口：`runtime/engine/engine.h @ v0.13.1`（SessionInterface:70；GenerateContent:112；GenerateContentStream:128；RunPrefill:174；RunDecode:188）。
- 对话配置：`runtime/conversation/conversation.h:56 @ v0.13.1`（ConversationConfig）。
- 分词：`runtime/components/{sentencepiece_tokenizer,huggingface_tokenizer,tokenizer}.h @ v0.13.1`。

<!-- 缺口（见 notes.md）：conversation.cc 的 diff 实现细节需补读，补读后回填精确行号引用。 -->
