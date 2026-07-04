# 模块素材：多语言绑定 (Language Bindings)  `bindings`

> 来源：litert-lm-guide/data.json（多智能体源码分析）。**这是素材，不是正文**。
> 引用进正文前，须按 CLAUDE.md 第二节逐条 Read 源码核验、补 `@ v0.13.1`。

**一句话**：把 C++ 核心引擎暴露给各语言。关键在于一层稳定的 C ABI(c/engine.h)——用不透明指针(opaque handle)+ 纯 C 函数把 Engine/Session/Conversation 封成跨语言可调用的接口；Python 走 ctypes、Kotlin/Android 走 JNI、Swift 走 C 互操作、Web 走 WASM/JS 包。一套核心，多端复用。

**在架构中的位置**：最外圈的接入层。C++ 核心 → C ABI(c/engine.h, 编成共享库) → 各语言薄封装。所有语言最终都经这层 C 函数进入同一套 runtime/，因此功能对齐、行为一致。CLI(litert_lm_main / Python litert-lm) 也属于这一接入面。

## 关键文件
- `c/engine.h` — 跨语言的核心：稳定 C ABI。不透明句柄 LiteRtLmEngine/Session/Responses/Conversation + litert_lm_* 函数(create/create_session/run_prefill/run_decode/generate_content[_stream]/tokenize...)。
- `c/engine.cc` — C ABI 实现：把 C 句柄转回 C++ 对象并转调；管理生命周期与 UTF-8/JSON 编解码。
- `python/litert_lm/(engine.py, _ffi.py, session.py, conversation.py, tools.py, interfaces.py)` — Python 绑定：_ffi.py 用 ctypes 声明 C 结构/函数签名，engine.py 封装成 Engine/Session/Conversation 类。examples/ 有 simple/multimodal/tool 三个 main。
- `python/litert_lm_cli/(main.py, commands/)` — Python CLI(uv tool install litert-lm)：run/convert/import/list/rename/delete/benchmark/serve；serve 支持 OpenAI 兼容服务(openai_handler/gemini_handler)。
- `kotlin/java/com/google/ai/edge/litertlm/(Engine.kt, LiteRtLmJni.kt, Config.kt, Tool.kt, ...)` — Kotlin/Android 绑定：Engine.kt 高层 API，LiteRtLmJni.kt 声明 external fun 经 JNI 调原生库(NativeLibraryLoader 加载 .so)。
- `swift/(Engine.swift, Conversation.swift, Tool.swift, Config.swift, ...)` — Swift(iOS/macOS)绑定：Engine 是 actor(线程安全)，经 C 互操作调 c/engine.h；含 Conversation/Tool/Message/Capabilities。
- `js/packages/core, js/apps/chat` — Web 绑定：@litertlm/core npm 包(TypeScript)经 WASM 调核心；apps/chat 是 Vite 演示应用。
- `rust/(global_allocator.rs, alloc_defs.cc)` — Rust 侧主要提供全局分配器等胶水(项目内 Rust 依赖如 llguidance 经 cxx bridge 接入)。
- `Package.swift` — Swift Package：同时支持 macOS 与 iOS，便于 SwiftPM 集成。

## 核心抽象
- **C ABI 不透明句柄** (type) 〔`c/engine.h:41`〕：LiteRtLmEngine / LiteRtLmSession / LiteRtLmResponses / LiteRtLmConversation / *Config / *BenchmarkInfo 等都是 opaque struct 指针。各语言只持指针、调函数、最后调对应 *_delete 释放——这是跨 FFI 安全传递 C++ 对象的标准做法。
- **litert_lm_engine_* / session_* 函数族** (function) 〔`c/engine.h:480`〕：C 入口：engine_settings_create → engine_create → engine_create_session → session_run_prefill/run_decode(或 generate_content) → responses_get_response_text_at → *_delete。还有 tokenize/detokenize、get_start/stop_token、benchmark_info_* 一整套。
- **流式回调 LiteRtLmStreamCallback** (type) 〔`c/engine.h:752`〕：void(*)(void* data, const char* chunk, bool is_final, const char* error_msg)。run_decode_async / generate_content_stream / conversation_send_message_stream 用它把 token 流跨 FFI 边界回吐——各语言再包成自己的回调/协程/Flow。
- **Python Engine (ctypes)** (class) 〔`python/litert_lm/engine.py:48`〕：实现 interfaces.AbstractEngine。_ffi.py 用 ctypes.Structure(LiteRtLmSamplerParams/LiteRtLmInputData)与函数签名声明 C ABI；engine.py 把句柄包成 Pythonic 的 Engine/Session，c_string_p 自动 UTF-8 编码。
- **Kotlin Engine + LiteRtLmJni** (class) 〔`kotlin/java/com/google/ai/edge/litertlm/Engine.kt:36`〕：Engine(AutoCloseable) 高层 API(initialize/createSession/createConversation)；LiteRtLmJni 声明 external fun nativeCreateEngine/nativeRunDecode/nativeGenerateContentStream 等，经 JNI 调原生库，句柄以 Long 指针传递。
- **Swift Engine (actor)** (class) 〔`swift/Engine.swift:28`〕：public actor Engine —— 用 actor 保证并发安全；initialize()/createConversation() 经 C 互操作调 c/engine.h。配套 Conversation/Tool/Message/Capabilities Swift 类型。
- **litert-lm CLI** (tool) 〔`python/litert_lm_cli/main.py`〕：面向终端用户的命令：litert-lm run(本地推理)、convert/import(模型转换/导入)、serve(起 OpenAI 兼容服务)、benchmark。支持 --from-huggingface-repo 直接拉模型。

## 数据流
1. C++ 核心(runtime/)编译出含 C ABI 的共享库(.so/.dylib/.dll/.wasm)。
2. C ABI(c/engine.h)用 extern "C" + 不透明句柄把 Engine/Session/Conversation 暴露为纯 C 函数。
3. Python：ctypes(_ffi.py)按签名加载共享库 → engine.py 包成类；CLI 再包一层命令行。
4. Kotlin/Android：JNI(LiteRtLmJni external fun)→ 原生库；NativeLibraryLoader 负责 loadLibrary。
5. Swift/iOS&macOS：C 互操作直接调 c/engine.h；Engine 以 actor 暴露异步安全 API；SwiftPM(Package.swift)分发。
6. Web：核心编为 WASM，@litertlm/core(TS)封装，浏览器内运行(apps/chat 演示)。
7. 所有语言最终都进入同一套 runtime/，所以能力/行为一致，只是表层 API 风格随语言习惯不同。

## 概念
- **C ABI 作为通用桥(lingua franca)**：C++ 的名字修饰/异常/对象布局不跨语言稳定，所以先收敛成纯 C 接口(extern "C" + opaque 指针 + 错误码/状态)，几乎所有语言都能调 C，于是一层 C ABI 接通全部绑定。
- **不透明句柄与所有权**：跨 FFI 不暴露 C++ 类型，只给指针;谁创建谁(或按文档)负责调 *_delete 释放。Python 用 __del__/上下文管理、Kotlin 用 AutoCloseable、Swift 用 deinit 来对接。
- **各语言的 FFI 机制**：Python=ctypes(运行时按签名调用)、Kotlin/Java=JNI(external fun + native .so)、Swift=C 互操作(直接 import C 头)、Web=WASM。机制不同但目标一致：调到那层 C 函数。
- **跨边界的流式回调**：C 函数指针 + void* 用户数据把 token 流回吐;各语言再包成惯用形态(Python 回调、Kotlin Flow/回调、Swift AsyncStream)。错误也经回调的 error_msg 传出。
- **OpenAI 兼容服务**：Python CLI 的 serve 命令把本地引擎包成 OpenAI/Gemini 兼容的 HTTP 接口，已有客户端可零改动接入本地模型。
- **App 开发者免编译**：Kotlin/Swift/Python 提供预built SDK 与 .litertlm 单文件模型,绝大多数集成无需自己编译 C++。

## 优化
- **单一核心、多端复用**：所有绑定共享同一套 runtime/ 与优化(KV cache/MTP/量化),无需各端重写推理逻辑,行为天然一致。
- **Swift actor 并发安全**：用 actor 而非手写锁来串行化对原生引擎的访问,既安全又符合 Swift 并发模型。
- **litertlm 分段并行加载**：C ABI 暴露 parallel_file_section_loading 开关,加载 .litertlm 各段并行化,缩短启动。

## 关键代码片段（待核验 @ v0.13.1）
**C ABI 主线(c/engine.h)** — 待核验：`c/engine.h:480`
```c
LiteRtLmEngineSettings* s = litert_lm_engine_settings_create(
    model_path, "cpu", NULL, NULL);
LiteRtLmEngine* e = litert_lm_engine_create(s);
LiteRtLmSession* sess = litert_lm_engine_create_session(e, NULL);
LiteRtLmInputData in = { kLiteRtLmInputDataTypeText, "你好", 6 };
litert_lm_session_run_prefill(sess, &in, 1);
LiteRtLmResponses* r = litert_lm_session_run_decode(sess);
const char* text = litert_lm_responses_get_response_text_at(r, 0);
```
**Python(ctypes 之上的封装)** — 待核验：`python/litert_lm/examples/simple_main.py`
```python
from litert_lm import Engine
engine = Engine(model_path="model.litertlm", backend="cpu")
session = engine.create_session()
for chunk in session.generate_content_stream(["What is the capital of France?"]):
    print(chunk, end="")
```
**免编译 CLI(含 OpenAI 兼容服务)** — 待核验：`python/litert_lm_cli/commands/serve.py`
```bash
uv tool install litert-lm
litert-lm run --from-huggingface-repo=google/gemma-3n-E2B-it-litert-lm \
  gemma-3n-E2B-it-int4 --prompt="Hi"
# 起一个 OpenAI 兼容的本地服务:
litert-lm serve --model_path=model.litertlm
```
**Kotlin(JNI external fun)** — 待核验：`kotlin/java/com/google/ai/edge/litertlm/Engine.kt:175`
```kotlin
val engine = Engine(EngineConfig(modelPath = "model.litertlm", backend = Backend.CPU))
engine.initialize()
val session = engine.createSession(SessionConfig())
val reply = session.generateContent(listOf(InputData.text("你好")))
```

## 入手顺序
- 先把 c/engine.h 通读一遍——它是所有绑定的共同‘契约’,看 engine_create→create_session→run_prefill/run_decode→responses 这条主线。
- 选你的语言:Python 看 python/litert_lm/engine.py + examples/simple_main.py;Kotlin 看 Engine.kt + LiteRtLmJni.kt;Swift 看 Engine.swift + ConversationTests.swift。
- 想要命令行/服务:看 python/litert_lm_cli/commands/(run.py, serve.py)。
- 对照 engine-api 模块理解 C++ 原型,你会发现各语言 API 基本是它的镜像。
