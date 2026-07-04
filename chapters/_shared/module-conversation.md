# 模块素材：对话层 (Conversation)  `conversation`

> 来源：litert-lm-guide/data.json（多智能体源码分析）。**这是素材，不是正文**。
> 引用进正文前，须按 CLAUDE.md 第二节逐条 Read 源码核验、补 `@ v0.13.1`。

**一句话**：面向使用者的高层多轮对话 API。它在 Engine/Session 之上替你维护对话历史、套聊天模板、按角色组织消息、处理多模态输入，并解析模型输出里的工具调用(function call)与思考(thinking)通道。一句 SendMessage 就能完成‘拼 prompt → prefill → decode → 解析’的全过程。

**在架构中的位置**：位于 ② 编排层、紧贴对外接口。上游：用户直接用 Conversation::Create + SendMessage/SendMessageAsync。下游：内部持有一个 Engine::Session，把渲染好的文本经 Session::RunPrefill/RunDecode 送进核心流水线；并用 ModelDataProcessor 解析输出。它是‘有状态多轮’的封装，而 engine 层的 Session 是‘无对话语义的底座’。

## 关键文件
- `runtime/conversation/conversation.h` — 对话层主接口：Conversation、ConversationConfig(+Builder)、OptionalArgs。
- `runtime/conversation/conversation.cc` — 实现：模板渲染(增量 diff)、历史管理、prefill/decode 调度、输出解析、克隆、任务取消。
- `runtime/conversation/io_types.h` — 核心类型：Message=nlohmann::ordered_json、Preface/JsonPreface、Channel。
- `runtime/conversation/prompt_utils.{h,cc}` — 把 Preface/history/message 渲染进 PromptTemplateInput；RenderSingleTurnTemplateCommon；StripBlobsFromTemplateInput(渲染前剥离大 blob)。
- `runtime/conversation/channel_util.{h,cc}` — channel 解析：把 <|channel>...<channel|> 标记的内容(如 thinking)从正文中分流。
- `runtime/conversation/internal_callback_util.{h,cc}` — 把底层 Session 的 token 流回调适配成对话级 Message 回调(含 channel/tool 解析)。
- `runtime/conversation/model_data_processor/` — 按模型类型解析输出/渲染输入的处理器集合：gemma3/gemma4/qwen3/function_gemma/fastvlm/generic + 工厂。

## 核心抽象
- **Conversation** (class) 〔`runtime/conversation/conversation.h:492`〕：有状态多轮对话核心。SendMessage(阻塞)/SendMessageAsync(回调流式) 发消息；RunTextScoring 给文本打分(perplexity)；GetHistory/AccessHistory 读历史(mutex 保护)；Clone 克隆整段对话(独立历史与 KV cache)；CancelProcess/CancelGroup 取消；GetTokenCount/GetBenchmarkInfo 观测。内部持 Engine::Session + ModelDataProcessor + PromptTemplate + Preface。
- **ConversationConfig + Builder** (class) 〔`runtime/conversation/conversation.h:57`〕：对话配置，用链式 Builder 构造。可设 Preface、覆盖 PromptTemplate/DataProcessorConfig、开启约束解码(函数调用)、prefill_preface_on_init(首响应更快)、channels、是否把 channel 内容从 KV cache 过滤、是否流式 tool call、enable_thinking、重复惩罚等。CreateDefault(engine) 给默认值。
- **Message (= nlohmann::ordered_json)** (type) 〔`runtime/conversation/io_types.h:26`〕：消息就是一个有序 JSON，形如 {"role":"user","content":"..."}。role 区分 user/model/system；输出消息里还可能带 message["channels"][name](思考)与工具调用结构。用 JSON 而非固定 struct 让多模型/多模态字段都能塞下。
- **Preface / JsonPreface** (struct) 〔`runtime/conversation/io_types.h:30`〕：对话开场白：messages(系统指令/few-shot/历史) + tools(可用工具描述) + extra_context(模板渲染的额外上下文)。它提供整段对话的初始背景，工具定义也在这里(驱动约束解码)。
- **Channel** (struct) 〔`runtime/conversation/io_types.h:44`〕：响应通道定义：{channel_name, start, end}。模型输出中由 start/end 标记包裹的内容(如 <|channel>thought<channel|>)被分流到 message["channels"][name]，可选择不写进 KV cache——这是‘思考过程’与‘最终答案’分离的机制。
- **OptionalArgs** (struct) 〔`runtime/conversation/conversation.h:380`〕：单次发送的可选参数：has_pending_message(只 prefill 不 decode，用于连续追加多条消息后只生成一次)、decoding_constraint(本次约束)、max_output_tokens、task_group_id(便于成组取消)、extra_context、enable_thinking。
- **ModelDataProcessor (+工厂 + 各模型实现)** (class) 〔`runtime/conversation/model_data_processor/model_data_processor.h`〕：模型相关的输入渲染与输出解析抽象。工厂按模型类型选具体实现：gemma3/gemma4/qwen3 各自的聊天格式、function_gemma(函数调用解析)、fastvlm(视觉)、generic(通用)。把‘各模型聊天格式差异’收敛在此，对话层主体保持通用。
- **RenderSingleTurnTemplateCommon / 模板增量** (function) 〔`runtime/conversation/prompt_utils.h:59`〕：渲染聊天模板的公共逻辑。对话层 GetPrefillTextForMessages 用‘旧消息渲染串’与‘新+旧消息渲染串’做 diff，只把增量文本送去 prefill——避免每轮重渲染整段历史。

## 数据流
1. 用户 Conversation::Create(engine, config) 创建对话；可选 prefill_preface_on_init 在创建时就把 Preface(系统指令/工具) prefill 进 KV cache。
2. SendMessage(message, optional_args)：message 是 {role, content} 的 JSON；先把它加入(待)历史。
3. 渲染模板：用 PromptTemplate + ModelDataProcessor 把 Preface+history+新消息渲染成文本，并对‘旧/新’渲染结果做 diff 得到本轮需要 prefill 的增量文本(StripBlobsFromTemplateInput 先剥离大 blob)。
4. GetInputDataVectorForMessages 把增量文本(及多模态 blob)转成 InputData，调用 Session::RunPrefill 写入 KV cache。
5. 若 has_pending_message=true 则到此为止(只追加上下文，不生成)；否则调用 Session::RunDecode 触发生成。
6. internal_callback_util 把底层 token 流接住，channel_util 分流 thinking/channel 内容，ModelDataProcessor 解析工具调用(配合约束解码保证结构合法)。
7. 得到完整/流式 Message：正文进 content，思考进 message["channels"]，工具调用以结构化字段返回；同时把这条 model 消息写回 history。
8. 可选的 channel 内容按配置从 KV cache 过滤(rewind 到 checkpoint)；多轮继续复用已有 KV cache。

## 概念
- **多轮历史管理**：Conversation 自动累积 history_(每条 Message)，下一轮把历史作为上下文，使用者无需手动拼接。GetHistory/AccessHistory 提供读访问，且用 mutex 保证线程安全。
- **聊天模板 (Prompt Template / Jinja)**：不同模型有不同的对话格式(谁是 user/model 前后缀、控制 token)。模板默认从模型元数据 jinja_prompt_template 读，也可覆盖。渲染时用 diff 只取增量，省去重复 prefill。
- **角色与 Message(JSON)**：用 {role: user/model/system, content: ...} 表达对话。用 JSON 而非固定结构，方便容纳多模态内容、工具调用、channel 等可变字段。
- **Preface(开场白)**：system 指令 + 可用 tools + few-shot 示例 + 额外上下文的统一载体。它定义对话的‘人设/能力/背景’，工具定义还会驱动约束解码。
- **Channel(思考/分流通道)**：模型把‘思考过程’用特殊标记包裹输出，Conversation 据 Channel 定义把它从正文分流到 message["channels"]，并可选择不污染 KV cache——实现 reasoning 与答案分离。
- **Tool Use / Function Calling**：在 Preface.tools 声明工具后开启约束解码，模型被强制输出结构合法的函数调用；function_gemma 等 data processor 负责解析成结构化结果回填。
- **模型专属数据处理器**：ModelDataProcessor 工厂按模型(gemma3/gemma4/qwen3/...)选实现，把‘各家聊天格式与输出解析差异’封装起来，对话主体保持模型无关。
- **消息合并 prefill (has_pending_message)**：连续追加多条消息时只想生成一次：前几条 has_pending_message=true 只 prefill，最后一条设 false 才触发 decode。减少不必要的解码。

## 优化
- **模板增量渲染(diff)**：只把新增消息渲染出的增量文本送去 prefill，而不是每轮重渲染整段历史，省 token 与时间。
- **Preface 预热(prefill_preface_on_init)**：创建时就把系统指令/工具 prefill 进 KV cache，首条用户消息的响应更快(代价是初始化更久)。
- **channel 内容从 KV cache 过滤**：把 thinking 等 channel 内容 rewind 出 KV cache，避免思考过程占用并污染后续上下文。
- **Clone 复用 prefill 状态**：克隆对话(含 KV cache)做分叉探索，省去重复 prefill 同一段前缀。
- **渲染前剥离大 blob**：StripBlobsFromTemplateInput 在模板渲染/字符串化前剥离图像/音频等大数据，避免无谓拷贝与格式化开销。

## 关键代码片段（待核验 @ v0.13.1）
**最简多轮对话(C++)** — 待核验：`runtime/conversation/conversation.h:462`
```cpp
ASSIGN_OR_RETURN(auto engine, Engine::Create(model_assets));
ASSIGN_OR_RETURN(auto cfg, ConversationConfig::CreateDefault(*engine));
ASSIGN_OR_RETURN(auto conv, Conversation::Create(*engine, cfg));
ASSIGN_OR_RETURN(Message reply, conv->SendMessage(
    Message{{"role","user"},{"content","Hello world!"}}));
```
**用 Builder 开启函数调用 + 思考** — 待核验：`runtime/conversation/conversation.h:124`
```cpp
ASSIGN_OR_RETURN(auto cfg, ConversationConfig::Builder()
    .SetPreface(preface_with_tools)
    .SetEnableConstrainedDecoding(true)   // 强制工具调用结构合法
    .SetEnableThinking(true)
    .SetPrefillPrefaceOnInit(true)        // 预热，首响应更快
    .Build(*engine));
```
**流式回调** — 待核验：`runtime/conversation/conversation.h:534`
```cpp
conv->SendMessageAsync(
  Message{{"role","user"},{"content","讲个笑话"}},
  [](absl::StatusOr<Message> chunk){
    if (chunk.ok()) { /* 每个 chunk 一段文本；结束时收到空 Message */ }
  });
```
**Channel 把思考与答案分离** — 待核验：`runtime/conversation/io_types.h:44`
```cpp
Channel thinking{ .channel_name="thinking",
                  .start="<|channel>thought", .end="<channel|>" };
// 输出里被标记包裹的内容 -> message["channels"]["thinking"]，
// 可配置不写入 KV cache。
```

## 入手顺序
- 先读 conversation.h 顶部的 Example usage 与 Conversation 类(SendMessage / SendMessageAsync)。
- 再读 io_types.h，理解 Message=JSON、Preface、Channel 三个基本类型。
- 看 ConversationConfig::Builder 有哪些开关(约束解码/thinking/channel/preface 预热)。
- 顺着 conversation.cc 的 GetPrefillTextForMessages 看‘模板增量 diff → RunPrefill → RunDecode’主链路。
- 想懂多模型差异与工具调用解析，读 model_data_processor/(尤其 function_gemma 与 gemma4)。
