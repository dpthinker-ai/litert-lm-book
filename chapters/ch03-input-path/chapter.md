# 第 3 章 输入侧：从 Engine API 到 token 序列

> 本章划清 Engine、Session 与 Conversation 的所有权分层，说明 Clone 如何共享上下文、何时才复制 KV cache；再说明模板如何从消息中取得增量文本，并由 tokenizer 编码。执行器按模型 signature 直接提交所得 token id，或先生成 prefill embedding。

调用方提交文本或多模态数据，模型执行器最终接收 token id 或 embedding。中间还需处理对象生命周期、消息历史、模板渲染与分词，图 3-1 列出这条路径。跨轮上下文复用（3.1.1 与 3.3 节）与单次 prefill 缓冲区内的数据放置（3.5 节）是两个不同问题，后文分开处理。

<figure>
{{#include figs/fig-3-1.svg}}
<figcaption>图 3-1　处理器与模板先生成本轮文本，tokenizer 再编码为 token id；全历史回退需先通过前缀校验。</figcaption>
</figure>

## 3.1　公共 API 分层：Engine 与 Session

LiteRT-LM 的底层生成接口以 `Engine` 与 `Session` 为核心。`Engine` 初始化模型、tokenizer 与 embedding 组件等共享资源。它还负责创建 Session。`SessionInterface` 保存一次交互的内部状态，并提供生成、prefill 与 decode 操作。

`EngineT::CreateSession` 接收 `SessionConfig`，返回一个由调用方持有的 Session。同一个 Engine 可以创建多个 Session。生成入口位于 Session，而不是 Engine。共享模型资源与每次交互的状态分属不同的所有权边界。

接口不约定两类对象的内存占用或创建时延。这些数值应随具体模型、后端与设备一起测量。

`SessionInterface` 同时提供高层生成接口与拆分后的 prefill、decode 接口。高层的两个入口如下：

```cpp
// runtime/engine/engine.h:112
// runtime/engine/engine.h:128
virtual absl::StatusOr<Responses> GenerateContent(
    const std::vector<InputData>& contents) = 0;         // (1)

virtual absl::Status GenerateContentStream(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) = 0;  // (2)
```

(1) `GenerateContent` 同步返回生成结果。(2) `GenerateContentStream` 调度异步工作，结果经 `callback` 返回。正常结束时，回调收到空 `Responses`。出错或取消时，回调收到相应状态。两者都接收 `std::vector<InputData>`，其中 `InputData` 可以包含文本、图像或音频。

低层接口把 prefill 与 decode 拆成两个方法：

```cpp
// runtime/engine/engine.h:174
// runtime/engine/engine.h:188
// Adds the input prompt/query to the model for starting the prefilling
// process. Note that the user can break down their prompt/query into
// multiple chunks and call this function multiple times.
virtual absl::Status RunPrefill(const std::vector<InputData>& contents) = 0;  // (1)
// ...
virtual absl::StatusOr<Responses> RunDecode() = 0;       // (2)
```

(1) `RunPrefill` 接收 prompt，可以分块调用。(2) `RunDecode` 根据 Session 中已有的上下文开始生成。`SessionAdvanced::GenerateContent` 先调用 `RunPrefill`，成功后再调用 `RunDecode`。拆分接口允许调用方在两步之间建立检查点或克隆 Session。prefill 的任务组织见第 4 章；decode 见第 5 章。

Engine 的具体实现由 `EngineFactory` 选择：`Instance()` 返回单例工厂；`registry_` 保存引擎类型到创建函数的映射，`preferred_engines_` 保存各 Backend 的候选顺序；`CreateDefault` 按 Backend 遍历候选项，选择第一个已经注册的类型。

注册宏 `LITERT_LM_REGISTER_ENGINE` 会生成文件级静态 `EngineRegisterer`。`engine_advanced_impl.cc` 用这个宏注册 `kAdvancedLiteRTCompiledModel`。可用类型取决于相应实现是否链接进最终程序；头文件也要求调用方确保目标引擎已经注册。

### 3.1.1　共享前缀：基于同一上下文创建分支

`SessionInterface::Clone` 要求新 Session 取得调用点之前的设置与上下文。接口示例先对公共问题前缀执行 prefill，再克隆 Session。两条分支随后补入不同后缀，从相同的已处理上下文开始。

接口语义只能说明克隆后的行为，不能据此判定 KV cache 是立即深拷贝、引用共享还是写时复制。v0.13.1 的实现先共享上下文，分支修改时再分离。复制发生的位置需要沿实现路径确认。

#### Clone 的实现：任务顺序与写时复制

`SessionInterface::Clone` 的默认实现只返回 `UnimplementedError`。`SessionAdvanced` 覆写了该方法。它的同步入口先调用 `CloneAsyncLocked`，再等待已经排入队列的任务完成：

```cpp
// runtime/engine/engine.h:245
// runtime/core/session_advanced.cc:389
absl::StatusOr<std::unique_ptr<SessionInterface>> SessionAdvanced::Clone() {
  absl::Status status = absl::OkStatus();
  std::unique_ptr<SessionInterface> session;
  {
    absl::MutexLock lock(mutex_);
    ASSIGN_OR_RETURN(
        session,
        CloneAsyncLocked([&status](absl::StatusOr<Responses> responses) {
          status = responses.status();
        }));
  }
  RETURN_IF_ERROR(WaitUntilDone());                                  // (1)
  RETURN_IF_ERROR(status);
  return session;
}
```

(1) `WaitUntilDone` 使同步入口在克隆任务完成后才返回。`CloneAsyncLocked` 注册新的 Session，并把克隆任务接到源 Session 的 `last_task_ids_` 之后。源 Session 与新 Session 的后续任务都依赖该任务。这段代码只确定任务顺序，没有复制 KV cache。

线程执行管理器运行克隆任务时，调用 `ResourceManager::CloneContextHandler`，其关键部分如下：

```cpp
// runtime/framework/resource_management/resource_manager.cc:610
auto processed_context = llm_context_handler->shared_processed_context();  // (1)

// ...
return ContextHandler::Bundle(
    processed_context, std::make_unique<RuntimeConfig>(runtime_config),
    std::make_unique<RuntimeState>(runtime_state), std::move(audio_context)); // (2)
```

(1) 新旧 `ContextHandler` 持有同一个 `SharedProcessedContext`。该对象保存实际的 `ProcessedContext`，并用于写时复制。(2) `RuntimeConfig` 与 `RuntimeState` 按值复制。这里没有复制 KV cache buffer。`ResourceManager` 的接口注释同样说明两个 handler 共享 processed context。

后续分支需要分离时才发生深拷贝。`LockedLlmExecutor::Prefill` 会比较输入与已处理 token。若两者已经分叉，且当前 handler 不是共享链中最长的分支，代码便调用 `SaveProcessedContextAndSeparateLoadedHandler`。该函数再调用 `llm_executor->CloneContext()`，把原上下文保存到旧的共享对象，并为当前 handler 建立新的共享对象。

在 LiteRT compiled model 执行器中，`CloneContext()` 调用 `CloneKVCacheBuffers()`。后者遍历 KV cache 输入缓冲，逐个调用 `CopyTensorBuffer`。KV cache 深拷贝发生在这里。

NPU 执行器覆写了 `CloneContext()`。它在 prefill 输入缓冲中匹配 K、V、C cache 名称。匹配到的 buffer 传递给 `CopyTensorBuffer`。两条后端路径都在写时分离阶段复制；`CloneContextHandler` 初次共享上下文时不复制这些 buffer。

`Clone` 的初始成本不能按一次完整 KV cache 拷贝估算。后续分支的 prefill 行为决定 buffer 是否以及何时复制。

## 3.2　对话层：消息与模板文本

`Conversation` 是面向多轮消息的高层接口。它负责模板渲染、角色消息、多模态输入、历史管理与模型特定处理。`Conversation::Create` 先请求 Engine 创建一个 Session。构造完成后，Conversation 保存对 Engine 的引用，并独占它创建的 Session。

<figure>
{{#include figs/fig-3-2.svg}}
<figcaption>图 3-2　Engine 创建 Session；Conversation 保留 Engine 引用、独占一个 Session，并维护消息历史、模板与处理器。</figcaption>
</figure>

角色消息必须先转换为模型模板约定的文本或多模态输入。`Message` 是 `nlohmann::ordered_json` 的别名。它既可以表示 `{"role":"user","content":"..."}`，也可以携带工具调用或多模态字段。`PromptTemplate` 把消息序列、工具与额外上下文传递给 Jinja 模板。模型对应的 `ModelDataProcessor` 再负责输入转换与输出解析。

`ConversationConfig` 定义这层处理所需的配置。下列只读入口对应 Preface、模板、约束解码开关与 Preface 预填充选项：

```cpp
// runtime/conversation/conversation.h:56
const Preface& GetPreface() const { return preface_; }              // (1)
const PromptTemplate& GetPromptTemplate() const { return prompt_template_; }  // (2)
// ...
bool constrained_decoding_enabled() const {                        // (3)
  return constrained_decoding_enabled_;
}
bool prefill_preface_on_init() const { return prefill_preface_on_init_; }  // (4)
```

(1) `Preface` 包含对话开始时的消息、工具与额外上下文。(2) 调用方可以覆盖 `PromptTemplate`；未覆盖时，创建逻辑从模型元数据读取 Jinja 模板。(3) 该布尔值控制约束解码配置，具体应用见第 10 章。

(4) `prefill_preface_on_init` 为 true 且 Preface 非空时，`Conversation::Create` 生成输入并调用 `RunPrefill`。头文件说明，这会增加初始化时间并缩短首次响应时间。具体差值仍需在目标设备上测量。

`DataProcessorConfig` 是六种配置的 `std::variant`。表 3-1 只列 v0.13.1 头文件中的默认字段。部分字段会被模型元数据覆盖，因而不能把这些值视为所有模型文件的固定配置。

| 配置类型 | 图像相关默认值 | 工具调用标记 | 其他默认字段 |
|---|---|---|---|
| `Gemma3DataProcessorConfig` | boi/eoi 为 `<start_of_image>` / `<end_of_image>`；输入尺寸 768×768 | Markdown `tool_code` 代码块 | 工具语法类型为 `python` |
| `Gemma4DataProcessorConfig` | boi/eoi 为 <code>&lt;&#124;image&gt;</code> / <code>&lt;image&#124;&gt;</code>；patch 16×16，最多 2520 个 patch，池化核为 3 | <code>&lt;&#124;tool_call&gt;</code> / <code>&lt;tool_call&#124;&gt;</code> | 约束模式默认为文本或函数调用均可 |
| `Qwen3DataProcessorConfig` | — | `<tool_call>` / `</tool_call>` | — |
| `FunctionGemmaDataProcessorConfig` | — | `<start_function_call>` / `<end_function_call>` | 约束模式默认为文本或函数调用均可 |
| `FastVlmDataProcessorConfig` | 输入尺寸 1024×1024 | — | — |
| `GenericDataProcessorConfig` | — | — | model 角色默认为 `assistant` |

> 表 3-1　模型特定 data processor 配置的默认差异。各配置类的字段定义见 `runtime/conversation/model_data_processor/` 目录下对应的头文件。

### 3.2.1　MiniJinja 与模板兼容性改写

`PromptTemplate` 通过生成的 FFI 头调用 Rust MiniJinja。MiniJinja 不支持任意 Python 方法调用，而模型模板可能包含 `s.startswith("foo")` 一类写法。构造 `PromptTemplate` 时，LiteRT-LM 默认先调用 `EditTemplateForMinijinja`，再创建 MiniJinja 模板对象。改写函数使用 RE2 做文本替换：

```cpp
// runtime/components/prompt_template.cc:26
// runtime/components/prompt_template.cc:73-78
// runtime/components/prompt_template.cc:40
  RE2::GlobalReplace(&modified_template, R"regex(\.startswith\((.*?)\))regex",
                     R"( is startingwith \1)");                        // (1)
  RE2::GlobalReplace(&modified_template, R"regex(\.endswith\((.*?)\))regex",
                     R"( is endingwith \1)");
  // ...
  RE2::GlobalReplace(&modified_template, R"regex(\.split\((.*?)\)\[0\])regex",
                     R"( | split(\1) | first)");                       // (2)
  // ...
  RE2::GlobalReplace(&modified_template, R"regex({% generation %})regex", ""); // (3)
```

(1) 与 (2) 把若干 Python 风格的方法改写为 MiniJinja 测试或过滤器。(3) 删除 MiniJinja 不识别的 generation 标记。该函数只处理字符串，不分析 Jinja 语法树，因此只转换列出的模式。未覆盖的模板语法会原样进入 `Apply`，并可能在 MiniJinja 渲染时返回错误。

`PromptTemplateInput` 还包含 `now`，默认值取对象构造时的当前时间；引用它的模板两次渲染就可能得到不同结果。LiteRT-LM 因此在同一对新旧渲染中复用模板输入对象的非消息字段，并对渲染结果做前缀校验。

## 3.3　增量文本：单轮模板与全历史回退

Session 已经保留先前 prefill 和 decode 形成的上下文。下一轮只应提交新增输入，不应把旧历史再次提交给同一个 Session。`Conversation::GetSingleTurnText` 先检查模板是否支持单轮渲染（single-turn rendering）：

```cpp
// runtime/conversation/conversation.cc:251
if (prompt_template_.GetCapabilities().supports_single_turn) {
  auto single_turn_text =
      GetSingleTurnTextFromSingleTurnTemplate(message, optional_args);
  if (!absl::IsUnimplemented(single_turn_text.status())) {
    return single_turn_text;                                      // (1)
  }
}
return GetSingleTurnTextFromFullHistory(message, optional_args);  // (2)
```

(1) 模板与处理器支持该能力时，代码直接渲染当前轮。(2) 能力未声明，或处理器返回 `Unimplemented` 时，代码退回全历史路径。“每轮都渲染两次全历史”只描述回退路径，不适用于所有模型模板。

全历史回退在 `GetSingleTurnTextFromFullHistory` 中实现。首轮且 Preface 尚未预填充时，代码把 Preface 与新消息一次性渲染。其他情况先渲染旧消息，再把新消息加入模板输入，得到新的完整字符串：

```cpp
// runtime/conversation/conversation.cc:192
// runtime/conversation/conversation.cc:215
std::string old_string;
if (!IsEmptyPreface(preface_) || !history_.empty()) {
  old_tmpl_input.add_generation_prompt = false;
  ASSIGN_OR_RETURN(old_string, prompt_template_.Apply(old_tmpl_input));
}

PromptTemplateInput new_tmpl_input = std::move(old_tmpl_input);
// ...
new_tmpl_input.add_generation_prompt = true;
ASSIGN_OR_RETURN(const std::string& new_string,
                 prompt_template_.Apply(new_tmpl_input));
if (new_string.substr(0, old_string.size()) != old_string) {       // (1)
  return absl::InternalError(absl::StrCat(
      "The new rendered template string does not start with the previous "
      "rendered template string. \nold_string: ",
      old_string, "\nnew_string: ", new_string));
}
return {new_string.substr(old_string.size(),                       // (2)
                          new_string.size() - old_string.size())};
```

(1) 代码逐字节比较两次渲染结果，不从模板输入的继承关系推定字符串前缀。(2) 只有检查通过才返回新增后缀。模板若重排旧消息、修改结尾标记，或根据消息数量改写前部内容，函数会返回 `InternalError`。该路径不会改用最长公共前缀，也不会把未经验证的后缀提交给 Session。

`GetPrefillTextForMessages` 在回退并重新填充被过滤的 channel 内容时，也渲染前后状态并检查字符串前缀。这里的两个模板输入变量名为 `old_context` 与 `new_context`：

```cpp
// runtime/conversation/conversation.cc:751
// runtime/conversation/conversation.cc:856
PromptTemplateInput new_context = old_context;                       // (1)
// ...
ASSIGN_OR_RETURN(std::string new_string, prompt_template_.Apply(new_context));

if (old_string.length() > new_string.length()) {                     // (2)
  return absl::InternalError(
      absl::StrCat("The new rendered string is shorter than the previous "
                   "rendered string. \nold_string: ",
                   old_string, "\nnew_string: ", new_string));
}
if (new_string.substr(0, old_string.size()) != old_string) {         // (3)
  return absl::InternalError(
      absl::StrCat("The new rendered string does not start with the previous "
                   "rendered string. \nold_string: ",
                   old_string, "\nnew_string: ", new_string));
}
return new_string.substr(old_string.length());                       // (4)
```

(1) 复制模板输入会让 `now`、工具和额外上下文在这一对渲染中保持相同。追加消息仍可能改变模板前部或尾部，这次复制不能保证新渲染保留旧前缀。(2) 至 (4) 才给出检查条件与返回逻辑。

`include_preface` 只控制该辅助函数在旧消息为空时是否渲染 Preface。值为 true 时，`old_string` 留空，返回值包含 Preface。值为 false 时，代码先渲染 Preface，再从新字符串中减去它。调用方根据 `prefill_preface_on_init` 传入相反条件。

全历史回退会向模板引擎重复提交旧消息。若每轮增加近似固定长度的消息，前 n 轮累计提交的历史文本量随轮数呈二次增长。以每轮渲染出约 500 字符估算：第 10 轮的回退先渲染约 4500 字符的旧串，再渲染约 5000 字符的新串；前 10 轮累计提交约 5 万字符，而十轮的新增输入合计只有 5000 字符。这里只能推导待渲染输入量，不能由此给出具体时延。单轮路径不重复渲染全历史。是否为回退路径增加缓存，要在目标模板与设备上测量；缓存还需定义 Preface、模板或 `extra_context` 变化时的失效规则。

`SendMessageAsync` 把本轮文本传递给 `ModelDataProcessor::ToInputDataVector`，再调用 `Session::RunPrefillAsync`。旧上下文能否复用，取决于单轮语义或前缀校验，而不是 `PromptTemplateInput` 的复制操作。

## 3.4　文本编码：两种 tokenizer

文本处理完成后，tokenizer 将字符串编码为 token id 序列。LiteRT-LM 的实现都遵循 `Tokenizer` 接口：

```cpp
// runtime/components/tokenizer.h:41
class Tokenizer {
 public:
  // ...
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
  // ...
};
```

(1) `TextToTokenIds` 完成输入侧编码。(2) `TokenIdsToText` 用于输出侧；接口要求不完整的 BPE 序列返回 `DataLossError`。

SentencePiece 实现把编码调用转发给 `SentencePieceProcessor`：

```cpp
// runtime/components/sentencepiece_tokenizer.cc:65
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

(1) 这一层把第三方库的 `Encode` 结果与错误状态转接到统一接口。HuggingFace 实现把编码转发给 `tokenizers::Tokenizer`：

```cpp
// runtime/components/huggingface_tokenizer.h:61
// runtime/components/huggingface_tokenizer.cc:55
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

(1) 源码注释说明，Google 的默认泄漏检查器不能正确处理 Rust `lazy_static` 初始化。代码只在该调用作用域内禁用检查。(2) `Encode` 接收一份 `std::string`。上层调用方只依赖 `Tokenizer`，不需要区分 SentencePiece 与 HuggingFace 实现。

使用哪种实现由模型文件与构建配置共同决定。`ModelResourcesLitertLm::GetTokenizer` 先查 `.litertlm` 中的 SentencePiece section，命中即创建对应实现；否则再查 HuggingFace section，从其 JSON 数据创建。这两段数据由 loader 按 section 类型取出，对应 2.7 节 section 类型清单中的 `SP_Tokenizer` 与 `HF_Tokenizer_Zlib`。两个分支各由编译宏门控；section 存在而对应支持未编译进程序时，该函数返回 `UnimplementedError`。

输出侧还需处理 token 边界。SentencePiece 解码会暂存被 `HasBpeSuffix` 判定为不完整的 byte token。后续 token 到来后，代码再解码该缓冲。第 5 章说明这条输出路径。

## 3.5　prefill 输入：token id 与 embedding

主模型可以在内部根据 token id 查找向量。运行时也可以先把 id 转换为 embedding（嵌入）。`LlmLiteRtCompiledModelExecutorBase::Prefill` 根据 `use_token_as_lookup` 选择路径。该值为 true 时，代码把 id 写入 token 输入缓冲；否则调用 `EmbeddingLookupManager` 填充 embedding 输入缓冲。主机侧 embedding lookup 不是所有模型都必经的步骤。

`EmbeddingLookup` 的批量 prefill 接口如下：

```cpp
// runtime/components/embedding_lookup/embedding_lookup.h:63
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

(1) 接口查找一组 token 的 embedding，并按顺序写入 `output_tensor`。(2) `byte_offset` 只是当前输出张量内的起始字节位置。当同一张量的前部已经写入其他 embedding 时，本次调用从该位置继续写。

管理器接收 `token_offset`，据此换算字节偏移。公式是“每个 token 的 float 数量 × `sizeof(float)` × `token_offset`”。compiled model 执行器传入当前 prefill 缓冲的 `input_idx`。

`input_idx` 描述本次缓冲区中的布局位置。若已有 pending token，代码先将它从 0 增至 1。`input_idx` 不是跨轮 embedding 缓存标志，也不表示旧对话的 embedding 仍在这张输出张量中。

### 3.5.1　编译 embedding 模型与批量写入

`EmbeddingLookupText::LookupInternal` 对非负 token 运行一个已经编译的 embedding 模型：

```cpp
// runtime/components/embedding_lookup/embedding_lookup_text.cc:50
  if (token < 0) {
    memcpy(buffer.data(), default_embedding_vector_.data(), buffer.size());  // (1)
    return absl::OkStatus();
  }

  // The input tensor size was verified when the model was loaded.
  input_buffers_[0].Write(absl::MakeSpan(const_cast<const int*>(&token), 1));

  compiled_model_->Run(signature_key_.value(), input_buffers_, output_buffers_); // (2)
```

(1) 负数 token 在文本路径中得到 `default_embedding_vector_`。(2) 非负 token 先写入输入缓冲，再调用 `compiled_model_->Run`。调用结束后，代码把输出缓冲读到指定位置。若模型配置了完整的多模态 lookup，管理器会在相同偏移调用相应实现。多模态 embedding 的接入见第 10 章。

批量重载先检查输出张量的 rank、维度与写入范围。写入循环从 `byte_offset` 指定的位置开始：

```cpp
// runtime/components/embedding_lookup/embedding_lookup_text.cc:225
  prefill_output_ptr += byte_offset;                       // (1)
  for (int token : tokens) {
    absl::Span<uint8_t> output_buffer(
        reinterpret_cast<uint8_t*>(prefill_output_ptr), bytes_per_token);
    RETURN_IF_ERROR(LookupInternal(token, output_buffer)); // (2)
    prefill_output_ptr += bytes_per_token;
  }

  // If there are fewer tokens than the output tensor can hold, we need to treat
  // the remaining tokens as if they were 0.
  size_t starting_token = byte_offset / bytes_per_token + tokens.size();
  size_t num_tokens_to_fill = prefill_output_layout.Dimensions()[1];
  for (int i = starting_token; i < num_tokens_to_fill; ++i) {
    memcpy(prefill_output_ptr, default_embedding_vector_.data(),  // (3)
           bytes_per_token);
    prefill_output_ptr += bytes_per_token;
  }
```

(1) 指针移动到当前张量的写入起点。(2) 代码逐 token 调用 `LookupInternal`。(3) 从本次写入末尾到张量第二维末尾的槽位都复制默认 embedding。`starting_token` 同时计入偏移处已有的 token 与本次 token。padding 仍属于同一个输出张量的布局。

模板和 tokenizer 已将消息转换为 token id。执行器按模型 signature 选择直接提交 id，或在当前 prefill 缓冲中生成 embedding。跨轮复用由 Session 的上下文与模板增量语义负责；`byte_offset` 只处理单次缓冲区内的写入位置。

## 小结

Engine 管理共享模型资源并创建 Session。Conversation 独占一个 Session，并在其上维护历史、模板和模型特定处理器。模板支持单轮渲染时，Conversation 直接生成本轮文本。全历史回退只有在新渲染保留旧前缀时才返回后缀。tokenizer 再把文本编码为 token id。

Session Clone 与模板后缀提取都涉及跨轮上下文复用，但实现不同。Clone 先共享 `SharedProcessedContext`，需要分离时才复制 KV cache。模板路径通过单轮渲染或前缀校验避免重复提交旧文本。embedding 的 `byte_offset` 不属于这组跨轮机制，只定位当前输出张量中的写入位置。`RunPrefill` 的任务组织见第 4 章。

---

## 练习与自查

1. 模板前缀反例。设计一个聊天模板，使新渲染不再以旧渲染为前缀，并说明显式前缀检查会返回什么结果。
2. 克隆路径梳理。沿代码列出 Session Clone 的初始共享路径与分支分离路径。哪一步才调用 `CopyTensorBuffer`？
3. 渲染路径对比。比较单轮渲染与全历史回退：各自需要模板或处理器提供什么能力，失败时如何处理？
4. 写入偏移计算。假设每个 embedding 含 2048 个 `float`，当前 `token_offset` 为 3。按正文的偏移公式计算 `byte_offset`，并说明它为何不能代表跨轮复用。
5. 模板改写盲区。构造一个未被 `EditTemplateForMinijinja` 规则覆盖的 Python 风格模板片段，说明错误会在哪次调用中暴露。
