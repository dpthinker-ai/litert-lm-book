# 第 3 章 输入之路：从 Engine API 到 token 序列

> 使命：走通输入侧的全程——你敲进去的一句话，如何变成模型能吃的一串数字。沿途会遇到本书第一个真正精巧的设计：多轮对话怎么做到不重算历史。

第二部是全书的脊柱：跟着一个 token 走完它的一生。这一章是它的上半程——从你调用 API，到一串 token id 准备好被喂进模型。生成还没开始，但成败已经埋在这里。

## 门面：为什么是 Engine 和 Session 两层

打开 LiteRT-LM 的对外接口，第一眼看到的是两个类：`Engine` 和 `Session`。头文件顶部的 Example usage 注释（`runtime/engine/engine.h:44 @ v0.13.1`）把典型用法压成了五步：

```cpp
// Create the engine.
auto engine = Engine::CreateEngine(EngineSettings::CreateDefault(
    model_assets, litert::lm::Backend::CPU));            // (1)
CHECK_OK(engine);

// Create the session.
auto session = engine->CreateSession(SessionConfig::CreateDefault());  // (2)
CHECK_OK(session);

// Run generate content.
auto responses = (*session)->GenerateContent({InputText("What's the tallest
building in the world?")});                              // (3)
```

(1) 建 Engine 时要交两样东西：`ModelAssets`（模型权重从哪来）和 `Backend`（跑在 CPU 还是 GPU）。这一步会把以 GiB 计的权重加载进内存——第 1 章那笔账的量级。(2) 建 Session 只吃一个 `SessionConfig`，不碰权重。(3) 生成挂在 Session 上，不挂在 Engine 上。三行代码，两级对象，界限清清楚楚。

为什么要分两层？因为它们的生命周期和成本完全不同。

- **Engine 重**。它持有模型权重——以 GiB 计。加载一次要几秒、占几 GiB 内存。它应该被创建一次、长期复用。
- **Session 轻**。它代表一次对话，持有的是这次对话的状态：KV cache、采样配置、步数。它可以随开随关，一个 Engine 能开出多个 Session。

这个划分对应第 2 章那条"状态即对象"原则：把"不变的、昂贵的"（权重）和"多变的、廉价的"（对话状态）分到两级，各自有各自的生命周期。`SessionInterface` 这个抽象（`runtime/engine/engine.h:70 @ v0.13.1`）定义了一次会话能做的事——它同时暴露了高层和低层两套接口。高层的两个入口签名如下（`engine.h:112,128 @ v0.13.1`）：

```cpp
virtual absl::StatusOr<Responses> GenerateContent(
    const std::vector<InputData>& contents) = 0;         // (1)

virtual absl::Status GenerateContentStream(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) = 0;  // (2)
```

(1) `GenerateContent` 阻塞到底、一次给完整 `Responses`。(2) `GenerateContentStream` 立即返回，结果通过 `callback` 逐段流式吐出——注释约定：生成成功时回调收到一个空 `Responses` 表示结束，出错时收到错误状态并不再有后续，被取消时收到 Cancellation 错误。两者的输入都是 `std::vector<InputData>`，不是裸字符串：`InputData` 是文本、图像、音频的统一载体，多模态在接口层就已经预留了位置（第 10 章的主角）。

低层接口把 prefill 和 decode 拆成两个独立方法（`engine.h:174,188 @ v0.13.1`）：

```cpp
// Adds the input prompt/query to the model for starting the prefilling
// process. Note that the user can break down their prompt/query into
// multiple chunks and call this function multiple times.
virtual absl::Status RunPrefill(const std::vector<InputData>& contents) = 0;  // (1)
// ...
virtual absl::StatusOr<Responses> RunDecode() = 0;       // (2)
```

(1) `RunPrefill` 只把输入写进 KV cache、不产出 token，注释明说可以分多次调用把长 prompt 切块喂进去。(2) `RunDecode` 才开始逐 token 预测。高层的 `GenerateContent` 不过是这两步的组合：先 prefill 再 decode。把它们拆开，调用方就能在两步之间插手——最典型的用法就是下面要讲的 `Clone`。第 4 章讲 `RunPrefill` 里发生了什么，第 5 章讲 `RunDecode`。

### 共享前缀：Clone 把一次 prefill 分叉成多条对话

拆分接口最见功力的一处，是 `SessionInterface::Clone`（`runtime/engine/engine.h:231 @ v0.13.1`）。它的注释直接给了一个"共享前缀"的例子：

```cpp
// Example usage:
//   Session session1 = engine->CreateSession(...);
//   session1->Prefill("What is the tallest building ");   // (1)
//   Session session2 = session1->Clone();                 // (2)
//   session1->Prefill("in the world?");                   // (3)
//   session1->Decode();
//   session2->Prefill("in France?");                      // (3)
//   session2->Decode();
```

(1) `session1` 先 prefill 了公共前缀 "What is the tallest building "。(2) `Clone` 在这一刻分叉：`session2` 继承了 `session1` 到此为止的全部状态——包括那段前缀已经算好的 KV cache。(3) 之后两个 session 各走各的："in the world?" 和 "in France?" 只需各自 prefill 自己那半句。公共前缀那段算力，两条对话分摊，只付了一次。

这正是低层接口存在的理由：因为 prefill 能被单独调用、能在中途 `Clone`，"共享前缀"才成为可能。如果只有 `GenerateContent` 这种一锤子接口，前缀就无从复用。同一个"不重算已算过的东西"的动机，接下来会以另一种形态出现在对话层——那里没有显式的 `Clone`，靠的是文本 diff。

## 从消息到文本：对话与模板

多数使用者不直接摆弄 token，而是发"消息"：一句 user 说的话，期待一句 model 的回答。把消息组织成模型认得的格式，是 `Conversation` 这一层的活（第 2 章五层架构里的"对话与编排层"）。它内部持有一个 `Engine::Session`，替你维护历史、套模板、调度 prefill/decode。

模型并不理解"谁是 user、谁是 model"。它只认一长串文本，其中用特殊的标记把角色圈出来。把结构化的消息渲染成这样一长串带标记的文本，靠的是**聊天模板**（prompt template）。这里的消息本身是一个有序 JSON——`Message` 就是 `nlohmann::ordered_json` 的别名（`runtime/conversation/io_types.h:26 @ v0.13.1`），形如 `{"role":"user","content":"..."}`。用 JSON 而非固定 struct，是为了让多模型、多模态、工具调用、思考通道这些可变字段都能塞进同一个类型，对话层主体不必为每种模型改结构。

不同模型的模板不同——Gemma、Qwen 各有各的圈法，LiteRT-LM 为此准备了按模型类型分派的处理器（`model_data_processor`，第 10 章的工具调用还会回到它）。`ConversationConfig`（`runtime/conversation/conversation.h:56 @ v0.13.1`）就是配置这一层行为的地方。它暴露的只读入口能看出这一层管哪些事：

```cpp
const Preface& GetPreface() const { return preface_; }              // (1)
const PromptTemplate& GetPromptTemplate() const { return prompt_template_; }  // (2)
// ...
bool constrained_decoding_enabled() const {                        // (3)
  return constrained_decoding_enabled_;
}
bool prefill_preface_on_init() const { return prefill_preface_on_init_; }  // (4)
```

(1) `Preface` 是开场白：系统指令、few-shot 示例、可用工具描述的统一载体，它定义整段对话的背景。(2) `PromptTemplate` 默认从模型元数据里的 jinja 模板读，也可在这里覆盖。(3) 约束解码开关，开启后模型被强制输出结构合法的函数调用（第 10 章）。(4) `prefill_preface_on_init` 决定要不要在创建对话时就把 Preface 预先 prefill 进 KV cache——代价是初始化更久，回报是首条用户消息的响应更快。这四个开关，接下来 diff 那一节里会用到最后一个。

到这一步，你的一句"帮我改写这段"已经变成了一长串带角色标记的纯文本。下一步该把它切成 token 了——但在那之前，有一个多轮对话绕不开的问题。

## 不重算历史：模板 diff 增量渲染

问题是这样的。多轮对话里，第二轮的输入在逻辑上是"历史全文 + 新消息"。如果每一轮都把整段历史重新渲染、重新 prefill，那么对话越长，每轮的开销越大——第十轮要把前九轮重算一遍。这是纯粹的重复计算：历史的 KV cache 本来就在手里，却每轮重付一遍 prefill 的算力，TTFT 随对话变长越来越糟。

LiteRT-LM 的对策朴素而有效：**只 prefill 新增的那一小段。** 实现落在 `Conversation::GetPrefillTextForMessages`（`runtime/conversation/conversation.cc:751 @ v0.13.1`）。它的做法是把同一套模板跑两遍——一遍只喂旧消息，一遍喂旧消息加新消息，再相减：

```cpp
// Render the `old` string.
std::string old_string;
if (!old_messages.empty() || !include_preface) {
  ASSIGN_OR_RETURN(old_string, prompt_template_.Apply(old_context));  // (1)
}
// Copy the `old` template context to the `new` template context.
PromptTemplateInput new_context = old_context;
// Add new messages to the `new` template context.
// ...
ASSIGN_OR_RETURN(std::string new_string,
                 prompt_template_.Apply(new_context));                // (2)
```

(1) `old_string` 是把 Preface 加旧消息完整渲染出来的字符串。(2) `new_context` 直接拷贝自 `old_context`、再追加新消息，`new_string` 是渲染加了新消息之后的结果。因为 `new_context` 是从 `old_context` 拷来的，两次渲染的前半段用的是同一份上下文——这保证了 `new_string` 应当以 `old_string` 为前缀。相减的逻辑就压在最后几行（`conversation.cc:806,813,820 @ v0.13.1`）：

```cpp
if (old_string.length() > new_string.length()) {                     // (1)
  return absl::InternalError(/* ... shorter than previous ... */);
}
if (new_string.substr(0, old_string.size()) != old_string) {         // (2)
  return absl::InternalError(/* ... does not start with previous ... */);
}
return new_string.substr(old_string.length());                       // (3)
```

(3) 就是那个"文本差"：`new_string` 砍掉 `old_string` 那段前缀，剩下的尾巴，就是本轮真正需要 prefill 的增量文本。旧的部分早已在 KV cache 里（第 6 章），无需重来。

值得停下来看的是 (1) 和 (2) 这两道防线，它们暴露了这个 diff 的一个前提假设：**新渲染必须是旧渲染的字符串前缀。** 一旦模板不满足"追加消息只会在尾部加内容"——比如某些模型的模板会在结尾放一个固定的收尾标记、加新消息时要先把它挪走——`new_string` 就可能比 `old_string` 短，或者不以它开头。代码没有去猜、去做通用的最长公共前缀，而是直接返回 `InternalError` 把渲染结果整个打印出来。这是一个刻意的设计取舍：diff 只在"纯前缀增长"这一类模板上成立，不成立就当场报错，而不是悄悄算错、把错位的文本 prefill 进 KV cache 污染整段对话。第 4 章会看到，prefill 进去的东西是没法轻易撤回的，所以这里宁可炸也不将就。

diff 出的增量文本，随后交给 `GetInputDataVectorForMessages`（`conversation.cc:824 @ v0.13.1`）转成 `InputData` 向量、送进 `Session::RunPrefill`。它把"逻辑上每轮都是全量历史"翻译成了"物理上每轮只处理增量"。表面看只是个字符串相减，背后接住的是整个 KV cache 复用的收益——和上一节 `Clone` 共享前缀是同一个动机的两种长相：一个靠拷贝会话状态，一个靠比对渲染文本。

## 从文本到数字：两种 tokenizer

增量文本有了，最后一步是把它切成 token id——模型只吃数字。做这件事的叫 tokenizer，它们都实现同一个抽象接口 `Tokenizer`（`runtime/components/tokenizer.h:41 @ v0.13.1`）：

```cpp
class Tokenizer {
 public:
  virtual TokenizerType GetTokenizerType() const = 0;
  // Encodes the given input text to token ids. Includes tokenizer pre/post
  // processing.
  virtual absl::StatusOr<TokenIds> TextToTokenIds(absl::string_view text) = 0;  // (1)
  // ...
  // Decodes the given sequence of token ids into a string.
  // Returns absl::DataLossError if any of the tokens are part of an incomplete
  // BPE sequence.
  virtual absl::StatusOr<std::string> TokenIdsToText(                 // (2)
      const TokenIds& token_ids) = 0;
```

(1) 输入侧只用得到 `TextToTokenIds`：文本进、id 序列出。(2) 反方向的 `TokenIdsToText` 是输出侧（第 5 章）用的，注释里那句"incomplete BPE sequence 会返回 `DataLossError`"，正是第 5 章末尾"吐半个字"现象的接口层伏笔——解码到半个 BPE 序列时，tokenizer 会明确拒绝，而不是吐出乱码。

LiteRT-LM 提供两种实现，都继承这个接口。SentencePiece 版（Gemma 等模型用）的编码实现薄得几乎透明（`runtime/components/sentencepiece_tokenizer.cc:65 @ v0.13.1`）：

```cpp
absl::StatusOr<std::vector<int>> SentencePieceTokenizer::TextToTokenIds(
    absl::string_view text) {
  std::vector<int> ids;
  auto status = processor_->Encode(text, &ids);            // (1)
  if (!status.ok()) {
    return status;
  }
  return ids;
}
```

(1) 真正干活的是 `processor_`——一个 `sentencepiece::SentencePieceProcessor`。LiteRT-LM 这一层只做了薄薄一层包装：把第三方库的 `Encode` 转接到统一接口上，错误原样透传。HuggingFace 版（`runtime/components/huggingface_tokenizer.cc:55 @ v0.13.1`）同样是转接，但多了一个耐人寻味的细节：

```cpp
absl::StatusOr<std::vector<int>> HuggingFaceTokenizer::TextToTokenIds(
    absl::string_view text) {
  {
    // Disable leak check as Google's default leak checker does not properly
    // support Rust's lazy_static initialization.
    // TODO(b/379364190) - Remove this once the leak checker is fixed.
    absl::LeakCheckDisabler disabler;                      // (1)
    return tokenizer_->Encode(std::string{text});          // (2)
  }
}
```

(2) 底层 `tokenizer_` 是 HuggingFace 的 Rust 分词器，通过 FFI 调用。(1) 那行 `LeakCheckDisabler` 泄漏了实现真相：这是个跨语言边界的封装，Rust 的 `lazy_static` 初始化会被 Google 的泄漏检查器误报，只能临时关掉检查。两种 tokenizer 都实现同一个 `Tokenizer` 抽象——又一次"接口隔离"原则：上层只管"把这段文本变成 id 序列"，不关心底下是 C++ 的 SentencePiece 还是 Rust 的 HuggingFace，更不关心后者还要跟泄漏检查器打架。

这个抽象还解释了第 5 章末尾那个"吐半个字"现象的一半来由：SentencePiece 的解码路径（`sentencepiece_tokenizer.cc:84 @ v0.13.1`）会把 byte token 攒进一个 `chunk_byte_token_ids` 缓冲、等凑齐一个完整字符再吐——子词分词意味着一个 token 未必是一个完整的字，跨 token 的边界必须小心处理。编码是这条边界的正向，解码是反向，同一条规则的两面。

## 还差半步：token id 变成 embedding

token id 只是编号，进模型前会先经查表变成一个高维向量——**embedding**（嵌入）。这一步由 `EmbeddingLookup` 接口负责（`runtime/components/embedding_lookup/embedding_lookup.h:63 @ v0.13.1`）：

```cpp
// For a given list of tokens, looks up the embeddings, concatenates them and
// returns the result through the output tensor.
//
// bytes_offset is used to indicate what byte to start writing to in the
// output_tensor. This is used in cases where the output_tensor has already
// had some embeddings written to it.
virtual absl::Status LookupPrefill(absl::Span<const int> tokens,   // (1)
                                   litert::TensorBuffer* output_tensor,
                                   size_t byte_offset) = 0;         // (2)
```

(1) prefill 阶段一次查一批 token 的 embedding，拼接后写进 `output_tensor`。(2) 那个 `byte_offset` 参数是给增量续写用的：当 `output_tensor` 里已经有一部分 embedding，新的一批从指定字节偏移接着写——和上一节 diff 增量、上上节 Clone 共享前缀是同一种"接着已有的往下补，别从头来"的思路，只不过这次落在了张量的字节层面。id 是名字，embedding 才是模型真正计算的对象。这半步平时不用你操心，但记住它：第 10 章讲图片怎么进模型时，它会成为主角——图像编码器吐出的正是这种 embedding，绕过了 tokenizer 直接从这一步接入。

至此，输入之路走完。你敲进去的一句话，历经"消息 → 套模板 → diff 增量 → 分词 → 查表"，变成了一串准备好的向量。

<figure>
{{#include figs/fig-3-1.svg}}
<figcaption>图 3-1　输入侧数据流：一句话经对话模板渲染、与历史做 diff 取增量、再分词，最终成为一串 token id。只有增量部分需要 prefill——这是多轮对话不重算历史的关键。</figcaption>
</figure>

## 小结

输入侧有两个关键设计：Engine/Session 的两级抽象（把昂贵的权重和廉价的对话状态分开），以及模板 diff 增量渲染（把逻辑全量翻译成物理增量）。一个管空间，一个管时间。而 Clone 共享前缀、diff 增量、embedding 的 `byte_offset` 续写，三处不同层面的代码指向同一条准则：已经算过的，别再算第二遍。

那串 token id 现在躺在门口。下一章，`RunPrefill` 会把它一口吞进模型。

---

## 参考

- Engine / Session 接口：`runtime/engine/engine.h @ v0.13.1`（Example usage:44；SessionInterface:70；GenerateContent:112；GenerateContentStream:128；RunPrefill:174；RunDecode:188；Clone 共享前缀示例:231）。
- 对话配置与 diff：`runtime/conversation/conversation.h:56 @ v0.13.1`（ConversationConfig）；`runtime/conversation/conversation.cc @ v0.13.1`（GetPrefillTextForMessages:751；前缀相减:806/813/820；GetInputDataVectorForMessages:824）。
- 消息类型：`runtime/conversation/io_types.h:26 @ v0.13.1`（Message = nlohmann::ordered_json）。
- 分词：`runtime/components/tokenizer.h:41 @ v0.13.1`（Tokenizer 抽象）；`runtime/components/sentencepiece_tokenizer.cc:65 @ v0.13.1`（TextToTokenIds）；`runtime/components/huggingface_tokenizer.cc:55 @ v0.13.1`（TextToTokenIds）；SentencePiece 解码的 byte-token 处理:84。
- token id 到 embedding：`runtime/components/embedding_lookup/embedding_lookup.h:63 @ v0.13.1`（LookupPrefill 批量重载）。
