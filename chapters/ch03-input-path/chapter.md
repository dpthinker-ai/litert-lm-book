# 第 3 章 输入侧：从 Engine API 到 token 序列

> 本章目标：走通推理流水线的输入侧全程——一段用户输入如何被组装、渲染、分词，最终成为模型可以处理的 token id 序列。其中多轮对话的增量渲染，是本书遇到的第一处与性能直接相关的设计。

第二篇是全书的主线，覆盖推理流水线从输入到输出的完整链路。本章处理其前半段：从调用方发起 API 调用，到一串 token id 就绪、等待进入 prefill。生成尚未开始，但这一段的组织方式已经决定了后续几项开销的量级。

## 公共 API 分层：Engine 与 Session

LiteRT-LM 的对外接口以两个类为核心：`Engine` 与 `Session`。头文件顶部的 Example usage 注释（`runtime/engine/engine.h:44`）给出了典型用法的最小示例：

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

(1) 创建 Engine 需要两个输入：`ModelAssets`（模型权重的来源）与 `Backend`（在 CPU 还是 GPU 上执行）。这一步把以 GiB 计的权重加载进内存，量级与第 1 章内存容量约束一节的估算一致。(2) 创建 Session 只接收一个 `SessionConfig`，不触及权重。(3) 生成入口挂在 Session 上而非 Engine 上。两级对象的职责边界清晰。

分层的依据是两者的生命周期与成本不同。Engine 持有以 GiB 计的模型权重，加载一次耗时数秒、常驻数 GiB 内存，应当创建一次并长期复用。Session 代表一次对话，只持有该对话的状态：KV cache、采样配置、当前步数。它的创建与销毁开销小，一个 Engine 可以派生多个 Session。

这一划分对应第 2 章的原则：把不变且加载昂贵的权重，与多变且创建廉价的对话状态，分到两级各自管理生命周期。`SessionInterface` 抽象（`runtime/engine/engine.h:70`）定义了一次会话能执行的操作，同时暴露高层与低层两套接口。高层的两个入口签名如下（`engine.h:112,128`）：

```cpp
virtual absl::StatusOr<Responses> GenerateContent(
    const std::vector<InputData>& contents) = 0;         // (1)

virtual absl::Status GenerateContentStream(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) = 0;  // (2)
```

(1) `GenerateContent` 阻塞至生成结束，一次返回完整的 `Responses`。(2) `GenerateContentStream` 立即返回，结果通过 `callback` 逐段流式返回。注释约定了三种终止语义：生成正常结束时，回调收到一个空 `Responses`；出错时收到错误状态且不再有后续回调；被取消时收到 Cancellation 错误。两者的输入都是 `std::vector<InputData>` 而非裸字符串，`InputData` 是文本、图像、音频的统一载体，多模态在接口层已预留位置（见第 10 章）。

低层接口把 prefill 与 decode 拆成两个独立方法（`engine.h:174,188`）：

```cpp
// Adds the input prompt/query to the model for starting the prefilling
// process. Note that the user can break down their prompt/query into
// multiple chunks and call this function multiple times.
virtual absl::Status RunPrefill(const std::vector<InputData>& contents) = 0;  // (1)
// ...
virtual absl::StatusOr<Responses> RunDecode() = 0;       // (2)
```

(1) `RunPrefill` 只把输入写入 KV cache，不产出 token；注释指出可以分多次调用，把长 prompt 切块送入。(2) `RunDecode` 才开始逐 token 预测。高层的 `GenerateContent` 是这两步的顺序组合：先 prefill 后 decode。拆开之后，调用方可以在两步之间介入，最典型的用法是下文的 `Clone`。`RunPrefill` 内部的执行见第 4 章，`RunDecode` 见第 5 章。

### 共享前缀：Clone 将一次 prefill 分叉为多条对话

拆分接口的一处直接收益，是 `SessionInterface::Clone`（`runtime/engine/engine.h:245`）。它的注释给出了一个共享前缀的例子：

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

(1) `session1` 先 prefill 公共前缀 "What is the tallest building "。(2) `Clone` 在此处分叉：`session2` 继承 `session1` 到调用点为止的全部状态，包括该前缀已经计算好的 KV cache。(3) 此后两个 session 各自延续："in the world?" 与 "in France?" 各只需 prefill 自己的后半句。公共前缀的 prefill 计算只执行一次，被两条对话复用。

这是低层接口存在的理由之一：因为 prefill 可以单独调用、并可在中途 `Clone`，共享前缀才成为可能。若只有 `GenerateContent` 这种单次调用、不可中途介入的接口（one-shot 接口），前缀无从复用。避免重复计算这一动机，稍后会在对话层以另一种形态出现，那里没有显式的 `Clone`，靠的是渲染文本相减。

#### Clone 的实现：一次排入任务队列的异步克隆

接口注释把 `Clone` 说成"继承到调用点为止的全部状态"，容易让人以为它是一次就地的状态拷贝。实现层并非如此。`engine.h` 里的 `Clone` 只是一个返回 `UnimplementedError` 的默认桩（`runtime/engine/engine.h:245`）：

```cpp
virtual absl::StatusOr<std::unique_ptr<SessionInterface>> Clone() {
  return absl::UnimplementedError("Not implemented.");         // (1)
};
```

(1) 基类不提供任何克隆逻辑，未覆写该方法的 Session 实现调用 `Clone` 会直接得到 `UnimplementedError`。真正实现克隆的是 `SessionAdvanced::Clone`（`runtime/core/session_advanced.cc:389`）。它本身只是异步版本的同步封装：

```cpp
absl::StatusOr<std::unique_ptr<SessionInterface>> SessionAdvanced::Clone() {
  absl::Status status = absl::OkStatus();
  std::unique_ptr<SessionInterface> session;
  {
    absl::MutexLock lock(mutex_);
    ASSIGN_OR_RETURN(
        session,
        CloneAsyncLocked([&status](absl::StatusOr<Responses> responses) {  // (1)
          status = responses.status();
        }));
  }
  RETURN_IF_ERROR(WaitUntilDone());                            // (2)
  RETURN_IF_ERROR(status);
  return session;
}
```

(1) 同步 `Clone` 先在持锁状态下调用 `CloneAsyncLocked` 排入克隆任务，并立即拿到指向新 session 的句柄。(2) 随后 `WaitUntilDone` 阻塞，直到该任务在后台执行完成。换言之，返回的 `SessionInterface` 在函数返回时其克隆动作已经落地，但克隆本身是走异步任务队列完成的，不是在调用线程里同步 memcpy 一份状态。

克隆任务的登记发生在 `CloneAsyncLocked`（`runtime/core/session_advanced.cc:412`）：

```cpp
ASSIGN_OR_RETURN(auto task_id, execution_manager_lock->GetNewTaskId());

ASSIGN_OR_RETURN(auto session_id, execution_manager_lock->RegisterNewSession(  // (1)
                                      session_info_->session_config,
                                      session_info_->benchmark_info));

RETURN_IF_ERROR(execution_manager_lock->AddCloneSessionTask(  // (2)
    session_id_, task_id, last_task_ids_, session_id,
    std::make_shared<std::atomic<bool>>(false), std::move(callback)));

last_task_ids_ = {task_id};                                   // (3)

ASSIGN_OR_RETURN(auto session_info,
                 execution_manager_lock->GetSessionInfo(session_id));

return absl::WrapUnique(new SessionAdvanced(session_id, execution_manager_,  // (4)
                                            tokenizer_, session_info,
                                            session_state_, last_task_ids_));
```

(1) 先向执行管理器注册一个新 session、拿到 `session_id`。(2) `AddCloneSessionTask` 把克隆任务挂进任务图，第三个参数 `last_task_ids_` 是本 session 当前所有未完成任务的 id，克隆任务被串在它们之后，确保克隆发生在此前 prefill 都执行完之后的那个状态点。(3) 随即把 `last_task_ids_` 重置为这个克隆任务，让源 session 后续的操作排在克隆之后。(4) 新 `SessionAdvanced` 与源 session 共享同一个 `execution_manager_` 与 `tokenizer_`，并带上指向克隆任务的 `last_task_ids_`，因此对新 session 的第一次 prefill 会自动排在克隆完成之后。

这一设计把 `Clone` 纳入了与 prefill、decode 同一套任务图调度（第 4 章展开执行管理器）。它带来两点直接后果。其一，`Clone` 是与前序任务串行、与其他 session 可并发的，源 session 上尚未完成的 prefill 会先执行，克隆看到的是一个确定的状态点，而非某个竞态中的中间态。其二，KV cache 的实际复制发生在后台任务里，其代价取决于该处采用的复制策略（引用共享、写时复制或深拷贝）。v0.13.1 的执行管理器把这一步封装在 `AddCloneSessionTask` 内部，正文不展开其后端相关的具体拷贝路径；据接口语义可推断，克隆后两个 session 对各自 KV cache 的写入互不影响，否则 `session1` 续写 "in the world?" 会污染 `session2` 的前缀，与注释给出的用法矛盾。

## 从消息到文本：对话层与聊天模板

多数调用方不直接操作 token，而是发送消息：一条 user 角色的输入，期待一条 model 角色的回复。把消息组织成模型能处理的格式，是 `Conversation` 层的职责（对应第 2 章五层架构中的对话与编排层）。它内部持有一个 `Engine::Session`，负责维护对话历史、应用聊天模板并调度 prefill/decode。

模型不理解 user 与 model 这样的角色概念，它的输入是一段连续文本，角色边界由特殊标记（special token）界定。把结构化的消息渲染为这样一段带标记的文本，靠的是聊天模板（chat template / prompt template）。消息本身是一个有序 JSON，`Message` 是 `nlohmann::ordered_json` 的别名（`runtime/conversation/io_types.h:26`），形如 `{"role":"user","content":"..."}`。用 JSON 而非固定 struct，是为了让多模型、多模态、工具调用、思考通道这些可变字段共用同一类型，对话层主体不必为每种模型改动数据结构。

不同模型的模板语法不同，Gemma 与 Qwen 各有各的角色标记与拼接方式。LiteRT-LM 为此准备了按模型类型分派的处理器（`model_data_processor`，工具调用见第 10 章）。`ConversationConfig`（`runtime/conversation/conversation.h:56`）配置这一层的行为。它暴露的只读入口反映了这一层管辖的范围：

```cpp
const Preface& GetPreface() const { return preface_; }              // (1)
const PromptTemplate& GetPromptTemplate() const { return prompt_template_; }  // (2)
// ...
bool constrained_decoding_enabled() const {                        // (3)
  return constrained_decoding_enabled_;
}
bool prefill_preface_on_init() const { return prefill_preface_on_init_; }  // (4)
```

(1) `Preface` 是开场部分：把系统指令、few-shot 示例、可用工具描述组织在一起，定义整段对话的背景。(2) `PromptTemplate` 默认从模型元数据中的 Jinja 模板读取，也可在此覆盖。(3) 约束解码（constrained decoding）开关，开启后模型被强制输出结构合法的函数调用（见第 10 章）。(4) `prefill_preface_on_init` 决定是否在创建对话时就把 Preface 预先 prefill 进 KV cache：代价是初始化耗时增加，收益是首条用户消息的响应更快。这四个开关中，最后一个会在下一节的增量渲染里用到。

### 模板引擎的真身：MiniJinja 与一层正则改写

「渲染」由谁执行值得专门交代，因为它跨了一次语言边界。`PromptTemplate::Apply` 底层调用的不是 C++ 实现的 Jinja，而是 Rust 库 MiniJinja，经生成的 FFI 头接入（`runtime/components/prompt_template.cc:26` 的 `#include "runtime/components/rust/minijinja_template.rs.h"`）。MiniJinja 是 Jinja2 的 Rust 重实现，但与 Python 版并非完全兼容：它不支持在模板里调用任意 Python 方法，而 HuggingFace 模型的 `tokenizer_config.json` 里的聊天模板恰恰常写 `s.startswith("foo")` 这类 Python 习语。LiteRT-LM 的办法是渲染前先用一组 RE2 正则把模板改写成 MiniJinja 认识的语法（`EditTemplateForMinijinja`，`prompt_template.cc:40`）：

```cpp
  RE2::GlobalReplace(&modified_template, R"regex(\.startswith\\((.*?)\\))regex",
                     R"( is startingwith \1)");                        // (1)
  RE2::GlobalReplace(&modified_template, R"regex(\.endswith\\((.*?)\\))regex",
                     R"( is endingwith \1)");
  // ...
  RE2::GlobalReplace(&modified_template, R"regex(\.split\\((.*?)\\)\[0\])regex",
                     R"( | split(\1) | first)");                       // (2)
  // ...
  RE2::GlobalReplace(&modified_template, R"regex({% generation %})regex", ""); // (3)
```

(1) 把 Python 的方法调用改写成 MiniJinja 的测试语法，(2) 把下标访问改写成过滤器管道，共十余条规则；(3) 直接删掉 MiniJinja 不认识的 `{% generation %}` 标记。这层改写是文本级的正则替换，不是语法解析，脆弱性是明摆着的：模板里若出现规则未覆盖的 Python 习语、或恰好长得像规则左边的普通文本，渲染就会出错或变形。工程上它换来的是兼容大量现成 HuggingFace 模板而不必逐个手改。

这层实现与下一节的关系在于一个前提：diff 增量的整个算法建立在「同一模板对同一输入的渲染是确定的」之上。渲染两次、相减取尾，只有每次渲染逐字节一致才成立。MiniJinja 的确定性渲染提供了这个保证；而正则改写发生在渲染之前、只做一次，不影响确定性。

<div class="aside-compare">

「模板随模型文件走」不是 LiteRT-LM 独有的选择。llama.cpp 的 GGUF 格式同样把聊天模板作为元数据打包进模型文件，键名 `tokenizer.chat_template`（`llama.cpp/src/llama-arch.cpp:343 @ b9873`），渲染引擎则是自带的 C++ 实现（`common/chat.cpp`）。两家都面对同一个现实：Jinja 模板生态源自 Python，端侧运行时没有 Python，只能各自重新实现一个 Jinja 子集——LiteRT-LM 选了 Rust 的 MiniJinja 加一层正则改写，llama.cpp 选了自研 C++ 引擎。权衡的两端是维护成本（借力现成库 vs 自己维护）与兼容面（改写规则 vs 引擎逐步补齐），没有免费的一边。

</div>

至此用户输入已被渲染为一段带角色标记的纯文本。下一步是把它切成 token，但在此之前，多轮对话有一个绕不开的问题。

## 增量渲染：多轮对话不重算历史

问题如下。多轮对话中，第 n 轮的输入在逻辑上是"历史全文 + 新消息"。若每一轮都把整段历史重新渲染、重新 prefill，则对话越长每轮开销越大，第 10 轮要把前 9 轮重算一遍。这是重复计算：历史对应的 KV cache 已经驻留在内存中，却每轮重新支付一遍 prefill 的算力，TTFT（首 token 时延）随对话轮数线性增长。

LiteRT-LM 采用的方法是只 prefill 新增的那一段。实现落在 `Conversation::GetPrefillTextForMessages`（`runtime/conversation/conversation.cc:751`）。它把同一套模板渲染两遍：一遍只含旧消息，一遍含旧消息加新消息，再相减：

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

(1) `old_string` 是 Preface 加旧消息完整渲染后的字符串。(2) `new_context` 从 `old_context` 拷贝而来再追加新消息，`new_string` 是加入新消息后重新渲染的结果。由于 `new_context` 拷贝自 `old_context`，两次渲染的前半段基于同一份上下文，这保证了 `new_string` 应当以 `old_string` 为前缀。

这里的 `include_preface` 参数控制第一轮的相减语义，值得单独说明。当 `old_messages` 为空（首轮对话）时，是否渲染 `old_string` 取决于 `include_preface`（`conversation.cc:786` 的条件 `if (!old_messages.empty() || !include_preface)`）。若 `include_preface` 为 true，`old_string` 保持为空，于是 Preface 会被算进返回的增量文本、随首条消息一并 prefill；若为 false，`old_string` 含 Preface，Preface 会被从增量里减去。这对应上一节 `prefill_preface_on_init` 的两种取值：Preface 已在初始化时预先 prefill，则首轮增量不应再包含它。

相减的逻辑在函数末尾（`conversation.cc:806,813,820`）：

```cpp
if (old_string.length() > new_string.length()) {                     // (1)
  return absl::InternalError(/* ... shorter than previous ... */);
}
if (new_string.substr(0, old_string.size()) != old_string) {         // (2)
  return absl::InternalError(/* ... does not start with previous ... */);
}
return new_string.substr(old_string.length());                       // (3)
```

(3) 就是这段增量文本：`new_string` 去掉 `old_string` 前缀后剩下的尾部，即本轮需要 prefill 的部分。旧的部分对应的 KV cache 已经驻留（第 6 章展开 KV cache 结构），无需重新计算。

(1) 和 (2) 这两处前置校验（precondition check）暴露了增量渲染的前提假设：新渲染必须是旧渲染的字符串前缀。一旦模板不满足"追加消息只在尾部增加内容"这一性质，例如某些模型的模板在结尾固定放一个收尾标记、加新消息时要先移除它，则 `new_string` 可能比 `old_string` 短，或不以它开头。代码没有退而求最长公共前缀，而是直接返回 `InternalError` 并把两次渲染结果整个打印出来。这是一处明确的取舍：增量渲染只在"纯前缀增长"这一类模板上成立，不成立时选择直接返回错误而非降级处理，以免把错位的文本 prefill 进 KV cache、污染整段对话。此处宁可报错也不将就一个可能错误的结果：被污染的 KV cache 不是免费能撤回的，它的写入与回退代价第 6 章展开。

#### 增量渲染的成本账：省下的是 prefill，不是渲染

这里有一处容易被忽略的成本。`GetPrefillTextForMessages` 每一轮都完整渲染两遍：一遍"旧全量"、一遍"旧全量 + 新"。第 n 轮渲染的字符数与到该轮为止的历史长度成正比，整段 n 轮对话累计处理的字符数是 O(n²) 量级。也就是说，增量渲染并没有省下渲染本身的开销，反而每轮都把历史重新渲染了一遍（甚至两遍）。

它省下的是 prefill 的算力。要判断这笔账是否划算，需要对比两侧的量级。模板渲染是纯字符串处理，走 CPU，单位是每字符若干纳秒；prefill 是对每个 token 跑一遍模型前向，在端侧 4B 级量化模型上，单 token 的 prefill 涉及数十亿次浮点运算，二者相差多个数量级。设历史有 H 个 token，全量重算方案每轮多付 H 个 token 的 prefill；增量渲染方案每轮多付的是"再渲染一遍 H 个 token 对应文本"的字符串开销。前者是模型前向，后者是字符串扫描，即便渲染累计到 O(n²)，其绝对开销仍远小于被省下的 O(n²) 量级 prefill 前向。净收益为正，且随对话变长收益越大。（该量级对比为基于代码路径的分析，具体数字取决于设备、模型与量化，此处不给绝对时延。）

如果要进一步压掉渲染侧的 O(n²)，理论上可以缓存上一轮的 `new_string` 作为下一轮的 `old_string`，省去重复渲染旧消息。v0.13.1 未做这一步，每轮都从 Preface 起重新渲染。据此推断，这一取舍的依据是渲染开销相对 prefill 可忽略，缓存渲染文本引入的一致性维护（模板、Preface、extra_context 任一变化都需失效缓存）不值得。

diff 出的增量文本随后交给 `GetInputDataVectorForMessages`（`conversation.cc:824`）转为 `InputData` 向量，送入 `Session::RunPrefill`。它把"逻辑上每轮都是全量历史"翻译成"物理上每轮只处理增量"。表面是字符串相减，背后是整个 KV cache 复用的收益，与上一节 `Clone` 共享前缀是同一优化动机的两种实现形态：一个复制会话状态，一个比对渲染文本。

## 从文本到数字：两种 tokenizer

增量文本有了，最后一步是把它切成 token id——模型只接受数字作为输入。做这件事的叫 tokenizer，它们都实现同一个抽象接口 `Tokenizer`（`runtime/components/tokenizer.h:41`）：

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

(1) 输入侧只用得到 `TextToTokenIds`：文本进、id 序列出。(2) 反方向的 `TokenIdsToText` 是输出侧（第 5 章）用的，注释里那句"incomplete BPE sequence 会返回 `DataLossError`"，正是第 5 章末尾「输出不完整 token」现象的接口层伏笔：解码到不完整的 BPE 序列时，tokenizer 会明确拒绝，而不是产生乱码。

LiteRT-LM 提供两种实现，都继承这个接口。SentencePiece 版（Gemma 等模型用）的编码实现薄得几乎透明（`runtime/components/sentencepiece_tokenizer.cc:65`）：

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

(1) 真正干活的是 `processor_`——一个 `sentencepiece::SentencePieceProcessor`。LiteRT-LM 这一层只做了薄薄一层包装：把第三方库的 `Encode` 转接到统一接口上，错误原样透传。HuggingFace 版（`runtime/components/huggingface_tokenizer.cc:55`）同样是转接，但多了一个值得留意的细节：

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

(2) 底层 `tokenizer_` 是 HuggingFace 的 Rust 分词器，通过 FFI 调用。(1) 那行 `LeakCheckDisabler` 泄漏了实现真相：这是个跨语言边界的封装，Rust 的 `lazy_static` 初始化会被 Google 的泄漏检查器误报，只能临时关掉检查。两种 tokenizer 都实现同一个 `Tokenizer` 抽象——又一次"接口隔离"原则：上层只管"把这段文本变成 id 序列"，不关心底下是 C++ 的 SentencePiece 还是 Rust 的 HuggingFace，更不关心后者还需禁用泄漏检查器。

这个抽象还解释了第 5 章末尾那个「输出不完整 token」现象的一半来由：SentencePiece 的解码路径（`sentencepiece_tokenizer.cc:84`）会把 byte token 累积进 `chunk_byte_token_ids` 缓冲，等凑齐一个完整字符再输出。子词分词意味着一个 token 未必是一个完整的字，跨 token 的边界必须小心处理。编码是这条边界的正向，解码是反向，同一条规则的两面。

## 还差半步：token id 变成 embedding

token id 只是编号，进模型前会先经查表变成一个高维向量——**embedding**（嵌入）。这一步由 `EmbeddingLookup` 接口负责（`runtime/components/embedding_lookup/embedding_lookup.h:63`）：

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

(1) prefill 阶段一次查一批 token 的 embedding，拼接后写进 `output_tensor`。(2) 那个 `byte_offset` 参数是给增量续写用的：当 `output_tensor` 里已经有一部分 embedding，新的一批从指定字节偏移接着写。这和上一节 diff 增量、上上节 Clone 共享前缀是同一种"在前面已有内容基础上续写而非从头计算"的思路，只不过这次落在了张量的字节层面。

### 查表其实是跑一个编译子图

「查表」这个说法需要修正一处直觉。文本实现 `EmbeddingLookupText` 的单 token 路径长这样（`LookupInternal`，`runtime/components/embedding_lookup/embedding_lookup_text.cc:50`）：

```cpp
  if (token < 0) {
    memcpy(buffer.data(), default_embedding_vector_.data(), buffer.size());  // (1)
    return absl::OkStatus();
  }

  // The input tensor size was verified when the model was loaded.
  input_buffers_[0].Write(absl::MakeSpan(const_cast<const int*>(&token), 1));

  compiled_model_->Run(signature_key_.value(), input_buffers_, output_buffers_); // (2)
```

(2) 是要点：embedding 不是对一张权重表做内存索引，而是把 token id 写进输入 buffer、**跑一次编译好的 TFLite 子模型**，输出才是那个高维向量。这个子模型在模型文件里是独立的一段：实剖本书基准模型，`embedder` 段占 171 MB、签名输入恰是 `token_ids[1, 1]`——一次一个 token，与下面批量路径逐 token 调用的行为互相印证（附录 D）。做成独立子图的收益与第 7 章的量化直接相关：嵌入表是量化过的（第 7 章那个混合方案里嵌入层压到 int4），「查表」实际包含解量化，编译器把这一步连同取数一起编译成算子；同时它与主模型解耦，多模态路径可以单独复用或替换。(1) 是一个此刻看似特殊、在第 10 章会变得清晰的分支：**负数 token 不查表，直接返回一个预置的默认向量**。第 10 章图像占位符 `kSpecialToken` 的值恰是 -1，文本查表路径对这些占位先填默认向量，真正的图像 embedding 随后由视觉执行器覆写。

批量的 `LookupPrefill`（`:148`）先做一串防御性校验（rank 一致、维度逐一相等、写入范围不越界），然后是主循环和一段收尾（`:225`）：

```cpp
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

(1) 指针先跳过 `byte_offset`，落到续写起点。(2) 批量路径没有批量算子：循环体逐 token 调 `LookupInternal`，也就是每个 token 跑一次子模型的 `Run`。(3) 是 padding 语义：输出张量是定长的（第 4 章的固定形状 prefill 窗口），token 数不足时，剩余槽位全部填上默认向量。源码注释的原文是：把剩下的位置当作 token 0 对待。第 4 章算 prefill 工单的填充浪费时，浪费的那些槽位在物理上就是这一段 `memcpy` 填出来的。

id 是名字，embedding 才是模型真正计算的对象。这半步平时是自动完成的，但值得记住：第 10 章讲图片怎么进模型时，它会成为主角——图像编码器输出的正是这种 embedding，绕过 tokenizer 直接从这一步接入。

至此，输入之路走完。输入的一句话，历经"消息 → 应用模板 → diff 增量 → 分词 → 查表"，变成了一串准备好的向量。

<figure>
{{#include figs/fig-3-1.svg}}
<figcaption>图 3-1　输入侧数据流：一句话经对话模板渲染、与历史做 diff 取增量、再分词，最终成为一串 token id。只有增量部分需要 prefill——这是多轮对话不重算历史的关键。</figcaption>
</figure>

## 小结

输入侧有两个关键设计：Engine/Session 的两级抽象（把昂贵的权重和廉价的对话状态分开），以及模板 diff 增量渲染（把逻辑全量翻译成物理增量）。一个管空间，一个管时间。而 Clone 共享前缀、diff 增量、embedding 的 `byte_offset` 续写，三处不同层面的代码指向同一条准则：已经算过的，别再算第二遍。

那串 token id 现在已经就绪。下一章，`RunPrefill` 会将它送入模型处理。

---

## 练习与自查

1. **构造反例。** 设计一个聊天模板，使「新渲染是旧渲染的前缀」这一 diff 前提被破坏，并指出代码中哪一道防线会拦住它。
2. **对比题。** Clone 共享前缀与模板 diff 增量都避免了重复计算。两者各复用了什么、各付出什么代价？
3. **代码验证。** 打开 `conversation.cc` 的相减三行，说明长度检查与前缀检查各拦截哪一类模板行为。
4. **机制理解。** embedding 查表为什么实现为跑一个编译子模型，而不是对权重数组做内存索引？给出两个理由。
5. **脆弱性演示。** 构造一个模板片段，使 `EditTemplateForMinijinja` 的某条正则产生误改写（提示：让 `.startswith(` 出现在不该被改写的位置）。

