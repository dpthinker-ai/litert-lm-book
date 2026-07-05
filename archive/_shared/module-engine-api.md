# 模块素材：Engine 公共 API 层  `engine-api`

> 来源：litert-lm-guide/data.json（多智能体源码分析）。**这是素材，不是正文**。
> 引用进正文前，须按 CLAUDE.md 第二节逐条 Read 源码核验、补 `@ v0.13.1`。

**一句话**：LiteRT-LM 的对外门面层。把底层复杂的推理引擎封装成简洁稳定的 Engine / Session 两层接口：Engine 负责加载模型、产出会话；Session 持有一次对话的全部状态（KV cache、采样配置），提供阻塞式 GenerateContent 与回调式流式生成。这里也是 CLI(litert_lm_main)、C API(c/engine.h) 与各语言绑定的统一入口。

**在架构中的位置**：运行时门面层；上游 C++/C/CLI，下游经 EngineFactory 注册的 Engine 实现。

## 关键文件
- `/Users/dpthinker/workspace/LiteRT-LM/runtime/engine/engine.h` — 接口主文件。

## 核心抽象
- **SessionInterface** (class) 〔`/Users/dpthinker/workspace/LiteRT-LM/runtime/engine/engine.h`〕：会话抽象，持有对话状态 (KV cache)。高层 GenerateContent (阻塞) 与 GenerateContentStream (回调流式)；低层 RunPrefill 加 RunDecode。还有 RunTextScoring、Clone、SaveCheckpoint/Rewind、CancelProcess、内嵌 TaskController。多数是带 UnimplementedError 默认的虚函数。
- **EngineT 与 Engine** (class) 〔`/Users/dpthinker/workspace/LiteRT-LM/runtime/engine/engine.h`〕：引擎抽象，产出 Session。模板化于 SessionT。方法：CreateSession、GetEngineSettings、GetTokenizer、WaitUntilDone。Engine 是 EngineT<SessionInterface> 别名。
- **EngineFactory** (class) 〔`/Users/dpthinker/workspace/LiteRT-LM/runtime/engine/engine_factory.h`〕：单例工厂。registry_ 映 EngineType 到 Creator；preferred_engines_ 映 Backend 到优先列表。CreateDefault 按 backend 选首个已注册引擎。LITERT_LM_REGISTER_ENGINE 静态自注册。
- **EngineSettings** (class) 〔`/Users/dpthinker/workspace/LiteRT-LM/runtime/engine/engine_settings.h`〕：引擎级配置。CreateDefault 构造；MaybeUpdateAndValidate 回填与校验。封装主 LlmExecutorSettings 与可选 Vision/Audio settings。
- **SessionConfig** (class) 〔`/Users/dpthinker/workspace/LiteRT-LM/runtime/engine/engine_settings.h`〕：会话级配置。控制模态、SamplerParameters、stop_token_ids、prompt 模板、max_output_tokens。
- **InputData** (struct) 〔`/Users/dpthinker/workspace/LiteRT-LM/runtime/engine/io_types.h`〕：多模态输入 variant。每 Input 可装原始字节或 TensorBuffer。
- **Responses** (class) 〔`/Users/dpthinker/workspace/LiteRT-LM/runtime/engine/io_types.h`〕：输出容器。持有 TaskState、texts、scores、token_ids。
- **DecodeConfig** (class) 〔`/Users/dpthinker/workspace/LiteRT-LM/runtime/engine/io_types.h`〕：单次解码参数：RepetitionPenaltyConfig、Constraint、max_output_tokens。
- **LiteRtLmSettings** (struct) 〔`/Users/dpthinker/workspace/LiteRT-LM/runtime/engine/litert_lm_lib.h`〕：CLI 总配置，RunLiteRtLm 端到端编排。

## 数据流
1. ModelAssets::Create -> EngineSettings::CreateDefault -> EngineFactory::CreateDefault -> CreateSession() -> GenerateContent 或 RunPrefill+RunDecode 或 GenerateContentStream+callback -> 读 Responses。

## 概念
- **Engine vs Session**：Engine 是重量级、可被多会话共享的资源持有者（加载一次模型）；Session 是轻量、有状态的一次对话上下文。一个 Engine 可 CreateSession 出多个 Session，各自维护独立的 KV cache。
- **阻塞式 vs 流式生成**：GenerateContent 一次性返回完整 Responses；GenerateContentStream 通过 AnyInvocable 回调逐段吐字，回调最后收到 kDone/kCancelled 状态——这是聊天 UI 打字机效果的来源。
- **EngineFactory 自注册**：用 LITERT_LM_REGISTER_ENGINE 宏把不同后端的 Engine 实现静态注册到工厂；CreateDefault 按用户选的 Backend 选出首个匹配实现，实现后端可插拔。
- **ModelAssets / EngineSettings / SessionConfig**：三层配置：ModelAssets 指向 .litertlm 模型文件；EngineSettings 是引擎级配置(后端、executor settings)；SessionConfig 是会话级配置(采样参数、停止符、最大输出长度、prompt 模板)。

## 关键代码片段（待核验 @ v0.13.1）
**典型使用流程（C++）** — 待核验：`runtime/engine/engine.h`
```cpp
auto assets = ModelAssets::Create(model_path);
auto settings = EngineSettings::CreateDefault(*assets, Backend::kCpu);
auto engine = Engine::CreateEngine(*settings);
auto session = (*engine)->CreateSession(session_config);
// 阻塞式：
auto responses = (*session)->GenerateContent({InputText("你好")});
// 流式：
(*session)->GenerateContentStream({InputText("你好")}, observer);
```

## 入手顺序
- engine.h 示例 -> litert_lm_main.cc -> engine_settings.cc/engine_factory.h/c/engine.cc。
