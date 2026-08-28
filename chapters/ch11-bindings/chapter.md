# 第 11 章 多语言绑定：C ABI、JNI 与 Embind

> 本章分析 Python、Kotlin、Swift 与 Web 如何经 C ABI、JNI 或 Embind 进入同一套 C++ runtime，并说明流式回调、数据复制、字符串所有权和显式释放的边界契约。本章还比较绑定层的并发隔离方式，并介绍测试替身与跨平台构建。

LiteRT-LM v0.13.1 的 `README.md:99-106` 列出 Python、Kotlin、Swift、JavaScript、Flutter 和 C++ 六种 API。这些绑定并不共用同一条原生调用路径。Python 与 Swift 使用 `c/engine.h` 提供的 C ABI；Kotlin 的 JNI 和 Web 的 Embind 直接调用 C++。本章选择前四种绑定分析三类边界模式，Flutter 不在本章展开。

## 11.1　C ABI：C 兼容的原生边界

C++ 的名字修饰（name mangling）、异常、模板和对象布局受编译器及标准库 ABI 影响。Python `ctypes` 与 Swift C 互操作需要 C 兼容的符号和数据类型，不能直接导入任意 C++ 类的方法。

LiteRT-LM 因此在 `c/engine.h` 中定义 C 接口。这层接口是 Python 与 Swift 共用的 C 兼容边界，但不是 Kotlin 和 Web 的必经路径。

`c/engine.h:41-44` 先用不透明句柄（opaque handle）声明 engine 和 session。两者只有结构体类型名，不公开成员：

```cpp
// Opaque pointer for the LiteRT LM Engine.
typedef struct LiteRtLmEngine LiteRtLmEngine;    // (1)

// Opaque pointer for the LiteRT LM Session.
typedef struct LiteRtLmSession LiteRtLmSession;  // (2)
```

(1)、(2) 是 C 的前向声明。编译器知道 `LiteRtLmEngine` 是一个结构体类型，但头文件不公开其字段。绑定层只能把 `LiteRtLmEngine*` 作为句柄传回 C API。它不能依据该声明解引用、计算 `sizeof` 或访问成员。C++ 对象的布局因而不进入公开 ABI。

`c/engine.cc:176-182` 给出该结构体在实现文件中的完整定义：

```cpp
struct LiteRtLmEngine {
  std::unique_ptr<Engine> engine;    // (1)
};

struct LiteRtLmSession {
  std::unique_ptr<Engine::Session> session;
};
```

(1) 表明 `LiteRtLmEngine` 持有 `std::unique_ptr<Engine>`。头文件只公开句柄类型，`.cc` 文件独占句柄字段与 `Engine` 的具体布局。只要函数签名和句柄契约保持兼容，内部 C++ 类型就可以独立演进。

操作接口采用普通 C 函数，签名中只出现 C 类型和不透明指针。`c/engine.h:380-386` 定义 create 和 delete：

```cpp
// Creates a LiteRT LM Engine from the given settings. The caller is responsible
// for destroying the engine using `litert_lm_engine_delete`.
// ...
LiteRtLmEngine* litert_lm_engine_create(const LiteRtLmEngineSettings* settings);  // (1)

// Destroys a LiteRT LM Engine.
// ...
void litert_lm_engine_delete(LiteRtLmEngine* engine);                             // (2)
```

(1) 返回不透明指针，(2) 释放该指针。`litert_lm_engine_create_session`（`c/engine.h:396`）和 `litert_lm_session_run_prefill`（`c/engine.h:421`）采用同一形式。C++ 成员函数映射为 C 自由函数，对象句柄成为第一个参数。注释中的 "The caller is responsible for destroying the engine" 定义了调用方必须履行的释放契约。

<figure>
{{#include figs/fig-11-1.svg}}
<figcaption>图 11-1　Python 与 Swift 经 C ABI 调用 runtime，Kotlin 与 Web 分别经 JNI 和 Embind 直接调用 C++；三条边界路径复用同一套核心实现。</figcaption>
</figure>

## 11.2　创建与释放必须配对

不透明句柄隐藏了对象布局，也要求 API 明确对象的所有权。

C 接口不提供 C++ 析构语义，调用方必须让创建和释放成对。`litert_lm_engine_create` 返回的引擎由 `litert_lm_engine_delete` 释放（`c/engine.h:386`）。session 由 `litert_lm_session_delete` 释放（`c/engine.h:403`）。`c/engine.cc:537-556` 给出引擎的创建与释放实现：

```cpp
LiteRtLmEngine* litert_lm_engine_create(
    const LiteRtLmEngineSettings* settings) {
  if (!settings || !settings->settings) {
    return nullptr;                                          // (1)
  }

  absl::StatusOr<std::unique_ptr<Engine>> engine =
      EngineFactory::CreateDefault(*settings->settings);

  if (!engine.ok()) {
    ABSL_LOG(ERROR) << "Failed to create engine: " << engine.status();
    return nullptr;                                          // (2)
  }

  auto* c_engine = new LiteRtLmEngine;                       // (3)
  c_engine->engine = *std::move(engine);
  return c_engine;
}

void litert_lm_engine_delete(LiteRtLmEngine* engine) { delete engine; }  // (4)
```

(3) 用 `new` 分配 `LiteRtLmEngine`，再把工厂返回的 `unique_ptr<Engine>` 移入其中。(4) 删除句柄，句柄析构又触发 `unique_ptr<Engine>` 析构。RAII 仍负责句柄内部的清理，但跨语言调用方必须显式调用 delete，才能启动这条析构链。(1)、(2) 以 `nullptr` 表示创建失败；这条 C 接口没有把内部的 `absl::Status` 细节传给调用方。

绑定层需要把 create/delete 契约映射为本语言的资源管理接口。Python 提供上下文管理器和 `__del__`，Kotlin 使用 `AutoCloseable`，Swift v0.13.1 则主要依赖 `deinit`。

## 11.3　三类原生边界

四种绑定采用三类原生边界：Python 与 Swift 使用 C ABI，Kotlin 使用 JNI，Web 使用 Embind。

Python 使用 `ctypes`，在运行时按签名声明 C 结构和函数并调用共享库。`python/litert_lm/_ffi.py:171-174` 声明了引擎句柄的类型：

```python
# Engine
lib.litert_lm_engine_create.restype = ctypes.c_void_p    # (1)
lib.litert_lm_engine_create.argtypes = [ctypes.c_void_p]
lib.litert_lm_engine_delete.argtypes = [ctypes.c_void_p] # (2)
```

(1) 把 `litert_lm_engine_create` 的返回类型声明为 `c_void_p`，对应 C 侧的不透明指针；(2) 声明 delete 接收同一类型。`ctypes` 在运行时根据 `restype` 与 `argtypes` 调用共享库。这组声明无需另行编译扩展模块。`python/litert_lm/_ffi.py:24-33` 定义的 `c_string_p` 还在 `from_param` 中执行 `obj.encode("utf-8")`，统一处理传入字符串的编码。

Kotlin/Android 使用 JNI。`kotlin/java/com/google/ai/edge/litertlm/LiteRtLmJni.kt:19` 定义 `internal object LiteRtLmJni`，其中的 `external fun` 由 JNI 原生实现提供。`kotlin/java/com/google/ai/edge/litertlm/LiteRtLmJni.kt:52` 起声明 `nativeCreateEngine`：

```kotlin
external fun nativeCreateEngine(
  modelPath: String,
  backend: String,
  // ...
  audioBackendNumThreads: Int,
): Long                                        // (1)

external fun nativeDeleteEngine(enginePointer: Long)  // (2)
```

(1) 以 `Long` 承载原生指针的位模式，(2) 把同一数值传递给删除函数。Kotlin 不解释该数值指向的对象。JNI 需要一层编译后的原生实现，将 `jlong` 转回 C++ 指针；这一点与运行时声明签名的 `ctypes` 不同。

Swift（iOS 与 macOS）使用 C 互操作，直接导入 C 头文件。`swift/Engine.swift:17` 执行 `import CLiteRTLM` 后，可直接调用 `litert_lm_engine_create`。其中的 `Engine` 是一个 `actor`（`swift/Engine.swift:28-38`）：

```swift
public actor Engine {                         // (1)
  // ...
  private var handle: OpaquePointer? = nil     // (2)
```

(1) 把 `Engine` 定义为 actor。actor 隔离使其可变状态只能在该 actor 的隔离域内访问。这项语义不等于固定线程或后台线程调度。(2) 用 `OpaquePointer?` 保存 C 句柄。与 Python 的 `c_void_p` 和 Kotlin 的 `Long` 一样，绑定代码只保存并传递句柄，不访问原生对象布局。

Web 将核心编译为 WebAssembly，并在 `js/packages/core` 中提供 TypeScript 封装。该路径使用 Emscripten Embind，而不是 `c/engine.h`。Embind 对象提供 `.delete()`；`js/packages/core/src/engine.ts:137-139` 在释放 WebAssembly Engine 对象时调用该方法。

三类边界最终调用同一套 C++ runtime 实现，但这不足以证明各语言 API 的行为逐项一致。默认参数、错误翻译、调度方式和绑定层预处理都可能造成差异。本书也没有进行跨语言逐输出对照实验。图 11-1 表达的是实现复用关系，不是行为等价结论。

## 11.4　JNI 与 Embind 直接调用 C++

Kotlin 和 Web 不经过 `c/engine.h`。Kotlin JNI 的原生实现包含 runtime 的 C++ 头文件（`kotlin/java/com/google/ai/edge/litertlm/jni/litertlm.cc:37-38`）：

```cpp
#include "runtime/engine/engine.h"          // (1)
#include "runtime/engine/engine_factory.h"  // (2)
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
```

(1)、(2) 使 JNI 实现能够访问 `Engine`、`Engine::Session` 和 `EngineFactory`，不需要经 C ABI 的不透明句柄中转。`nativeCreateEngine` 在 `kotlin/java/com/google/ai/edge/litertlm/jni/litertlm.cc:542` 返回指针：

```cpp
auto engine = EngineFactory::CreateDefault(*settings);
if (!engine.ok()) {
  ThrowLiteRtLmJniException(
      env, "Failed to create engine: " + engine.status().ToString());
  return 0;
}

return reinterpret_cast<jlong>(engine->release());  // (1)
```

`EngineFactory::CreateDefault` 返回 `absl::StatusOr<std::unique_ptr<Engine>>`。(1) 调用 `release()` 后，`unique_ptr` 不再拥有该 `Engine`。裸指针 `Engine*` 随后被重解释为 `jlong`；Kotlin 保存的是这个数值句柄。原生对象仍需由释放函数删除（`kotlin/java/com/google/ai/edge/litertlm/jni/litertlm.cc:615-616`）：

```cpp
JNI_METHOD(nativeDeleteEngine)(JNIEnv* env, jclass thiz, jlong engine_pointer) {
  delete reinterpret_cast<Engine*>(engine_pointer);  // (1)
}
```

(1) 把 `jlong` 重解释为 `Engine*` 后直接删除。C ABI 的 `litert_lm_engine_delete`（`c/engine.cc:556`）先删除 `LiteRtLmEngine` 句柄，句柄中的 `unique_ptr` 随后删除 `Engine`。JNI 直接删除 `Engine` 本体，因此 Kotlin 路径不调用 `c/engine.h`。

Web 的 Embind 也在编译期从 C++ 导出 JavaScript 可见对象。源码可以确认三种边界路径及各自的句柄形态，但没有说明项目选择这些路径的全部设计理由。因此，表 11-1 只记录实现事实，不把 FFI 类型概括为普遍的选型规则。

| 语言 | FFI 机制 | 经 C ABI？ | 句柄形态 | 资源释放落点 |
|---|---|---|---|---|
| Python | ctypes（运行时声明签名） | 是 | `c_void_p` | 上下文管理器 / `__del__` |
| Kotlin | JNI（`external fun` 声明） | 否，直连 C++ | `jlong`（`Engine*` 位模式） | `AutoCloseable` |
| Swift | C 互操作（import C 头） | 是 | `OpaquePointer` | v0.13.1 由 `deinit` 释放 |
| Web | Emscripten Embind | 否，直连 C++ | 带 `.delete()` 的 JS 对象 | 手动 `.delete()` 配对 |

> 表 11-1　四种语言绑定采用三类原生边界，并以不同类型保存原生句柄。

## 11.5　流式生成如何跨越 FFI 边界

阻塞式接口在函数返回时交付结果，流式生成则需要多次传递增量文本和终止状态。C ABI 使用回调表达这组异步事件，并为回调参数规定明确的生命周期。

C ABI 表达回调的方式是函数指针加一个透传的用户数据指针。`c/engine.h:646-653` 给出回调参数契约与类型定义：

```cpp
typedef void (*LiteRtLmStreamCallback)(void* callback_data, const char* chunk,
                                       bool is_final, const char* error_msg);
```

`callback_data` 是调用方登记的 `void*`，C ABI 只负责原样传回。绑定层可用它定位本语言的对象或闭包上下文。`chunk` 携带增量文本，`is_final` 标记流结束，非空的 `error_msg` 表示错误。头文件注明 `chunk` 只在本次回调期间有效。若绑定层需要在回调返回后保留文本，就必须在回调内复制内容。

C++ 侧用 `absl::AnyInvocable` 接收 `absl::StatusOr<Responses>`。`c/engine.cc:50-75` 的 `CreateCallback` 将其适配为四参数 C 回调：

```cpp
absl::AnyInvocable<void(absl::StatusOr<litert::lm::Responses>)> CreateCallback(
    LiteRtLmStreamCallback callback, void* callback_data) {
  return [callback,
          callback_data](absl::StatusOr<litert::lm::Responses> responses) {
    if (!responses.ok()) {
      callback(callback_data, /*text=*/nullptr, /*is_final=*/true,
               responses.status().ToString().c_str());        // (1)
      return;
    }
    if (responses->GetTaskState() == litert::lm::TaskState::kDone) {
      callback(callback_data, /*text=*/nullptr, /*is_final=*/true,
               /*error_message=*/nullptr);                     // (2)
    } else if (responses->GetTaskState() ==
               litert::lm::TaskState::kMaxNumTokensReached) {
      callback(callback_data, /*text=*/nullptr, /*is_final=*/true,
               "Max number of tokens reached.");               // (3)
    } else if (responses->GetTaskState() == litert::lm::TaskState::kCancelled) {
      callback(callback_data, /*text=*/nullptr, /*is_final=*/true,
               "CANCELLED.");                                  // (4)
    } else {
      for (const auto& text : responses->GetTexts()) {
        callback(callback_data, text.data(), /*is_final=*/false,
                 /*error_message=*/nullptr);                   // (5)
      }
    }
  };
}
```

该 lambda 将 C++ 状态映射为 C 参数。(1) 在 `StatusOr` 失败时把状态字符串放入 `error_msg`，同时把 `is_final` 置为 true。(2) 表示正常结束；(3) 和 (4) 分别传递达到 token 上限与取消状态。(5) 对 `GetTexts()` 中的每条增量文本调用一次回调，并保持 `is_final` 为 false。

(5) 传入 `text.data()`，没有转移底层缓冲的所有权。绑定层只能在回调期间读取该指针；把指针本身保存到回调之外会违反接口契约。Python 或 Swift 若需要保留增量文本，应先复制为本语言拥有的字符串。

发起流式的入口位于 `c/engine.cc:653-665`：

```cpp
int litert_lm_session_run_decode_async(LiteRtLmSession* session,
                                       LiteRtLmStreamCallback callback,
                                       void* callback_data) {
  if (!session || !session->session) {
    return -1;                                                     // (1)
  }
  auto status =
      session->session->RunDecodeAsync(CreateCallback(callback, callback_data));
  if (!status.ok()) {
    return static_cast<int>(status.status().code());               // (2)
  }
  return 0;                                                        // (3)
}
```

该函数先把 C 回调和 `callback_data` 传递给 `CreateCallback`，然后调用 `RunDecodeAsync`。(1) 在参数无效时返回 `-1`，(2) 在启动失败时返回 `absl::StatusCode` 的整数值，(3) 在成功启动时返回 0。返回码只描述启动结果。后续文本和终止状态均由回调传递。绑定层还需保证回调上下文至少存活到最终回调完成。

## 11.6　多模态输入在 C 边界的扁平化

第 10 章说明了图像和音频如何进入 KV cache。本节只考察文本、图像和音频如何表示为 C ABI 可接收的数据。

C ABI 使用带类型标签的结构体表示输入（`c/engine.h:243-260`）：

```cpp
typedef enum {
  kLiteRtLmInputDataTypeText,
  kLiteRtLmInputDataTypeImage,
  kLiteRtLmInputDataTypeImageEnd,
  kLiteRtLmInputDataTypeAudio,
  kLiteRtLmInputDataTypeAudioEnd,
} LiteRtLmInputDataType;

typedef struct {
  LiteRtLmInputDataType type;
  const void* data;   // (1)
  size_t size;        // (2)
} LiteRtLmInputData;
```

每个元素由 `type` 和 `(data, size)` 组成。(1) 的 `const void*` 指向缓冲，(2) 给出字节数。文本数据采用 UTF-8，图像与音频使用相应的编码字节；`type` 决定 C++ 侧如何解释缓冲。

`c/engine.cc:127-154` 的 `ToEngineInputData` 将数组转换为 C++ `InputData`：

```cpp
std::vector<litert::lm::InputData> ToEngineInputData(
    const LiteRtLmInputData* inputs, size_t num_inputs) {
  std::vector<litert::lm::InputData> engine_inputs;
  engine_inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    switch (inputs[i].type) {
      case kLiteRtLmInputDataTypeText:
        engine_inputs.emplace_back(litert::lm::InputText(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));  // (1)
        break;
      case kLiteRtLmInputDataTypeImage:
        engine_inputs.emplace_back(litert::lm::InputImage(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));  // (2)
        break;
      case kLiteRtLmInputDataTypeImageEnd:
        engine_inputs.emplace_back(litert::lm::InputImageEnd());          // (3)
        break;
      case kLiteRtLmInputDataTypeAudio:
        engine_inputs.emplace_back(litert::lm::InputAudio(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));
        break;
      case kLiteRtLmInputDataTypeAudioEnd:
        engine_inputs.emplace_back(litert::lm::InputAudioEnd());
        break;
    }
  }
  return engine_inputs;
}
```

`switch` 根据标签构造对应类型：(1) 生成 `InputText`，(2) 生成 `InputImage`，(3) 生成不含缓冲的 `InputImageEnd`；音频分支采用相同结构。(1)、(2) 中的 `std::string(ptr, size)` 会把 C 缓冲复制到新字符串，复制量等于输入的字节数。

接口不接管调用方缓冲的所有权，也不要求缓冲在函数返回后继续存活。实现会复制数据，使 `InputData` 独立拥有内容。若要避免复制，C ABI 需要增加可验证的生命周期契约，例如所有权转移或释放回调。v0.13.1 没有实现这些方案。本书也没有测量这次复制在端到端时延中的占比。

## 11.7　跨语言字符串的所有权与编码

跨 FFI 返回 `const char*` 时，接口必须规定指针的有效期和字符编码。LiteRT-LM 分别在 C ABI 与 JNI 层处理这两个问题。

`c/engine.cc:725-733` 的 `get_response_text_at` 返回内部字符串的 `data()`：

```cpp
const char* litert_lm_responses_get_response_text_at(
    const LiteRtLmResponses* responses, int index) {
  if (!responses || index < 0 ||
      index >= responses->responses.GetTexts().size()) {
    return nullptr;
  }
  // The string_view's data is valid as long as the responses object is alive.
  return responses->responses.GetTexts()[index].data();  // (1)
}
```

`LiteRtLmResponses` 拥有 (1) 返回指针所指向的内存。该指针只在 responses 句柄存活期间有效；句柄删除后，调用方不得继续访问。调用方也不得单独释放这个 `const char*`。若需要延长字符串的生命周期，必须在删除句柄前复制内容。

消息渲染会产生一段新的字符串。`c/engine.cc:192-199` 在 conversation 句柄中设置 `last_rendered_message` 保存结果：

```cpp
struct LiteRtLmConversation {
  std::unique_ptr<Conversation> conversation;
  // This field stores the result of the last call to
  // `litert_lm_conversation_render_message_to_string`. This ties the lifetime
  // of the returned `const char*` to the `LiteRtLmConversation` object,
  // ...
  std::string last_rendered_message;  // (1)
};
```

渲染函数把结果移入该成员，再返回它的 `c_str()`，见 `c/engine.cc:1115-1116`：

```cpp
  conversation->last_rendered_message = std::move(*rendered);  // (1)
  return conversation->last_rendered_message.c_str();          // (2)
```

(1) 使 conversation 句柄拥有渲染结果，(2) 返回其 C 字符串指针。下一次渲染会覆盖 `last_rendered_message`，删除 conversation 也会销毁该成员。绑定层若要保留旧结果，必须在下一次渲染或删除句柄之前复制。调用方不需要单独释放这个指针。

JNI 的 `NewStringUTF` 接收 modified UTF-8。它对空字符和基本多文种平面（BMP）之外字符的编码与标准 UTF-8 不同，不能直接替代标准 UTF-8 解码。`kotlin/java/com/google/ai/edge/litertlm/jni/litertlm.cc:104-146` 因此定义 `NewStringStandardUTF`：

```cpp
jstring NewStringStandardUTF(JNIEnv* env, std::string standard_utf8_str) {
  jbyteArray bytes = env->NewByteArray(standard_utf8_str.length());   // (1)
  if (bytes == nullptr) return nullptr;
  env->SetByteArrayRegion(
      bytes, 0, standard_utf8_str.length(),
      reinterpret_cast<const jbyte*>(standard_utf8_str.c_str()));     // (2)
  jclass string_class = env->FindClass("java/lang/String");
  // ...
  jmethodID string_ctor =
      env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");  // (3)
  // ...
  jstring charset_name = env->NewStringUTF("UTF-8");                  // (4)
  jstring result =
      (jstring)env->NewObject(string_class, string_ctor, bytes, charset_name);  // (5)
  // ...
  return result;
}
```

(1) 创建 `byte[]`，(2) 写入标准 UTF-8 字节。(3) 获取 `String(byte[], String charsetName)` 构造器。(4) 创建 ASCII 字符串 `"UTF-8"`，(5) 显式按该字符集构造 Java `String`。省略的代码调用 `DeleteLocalRef` 清理 JNI 局部引用。该辅助函数用于响应文本、错误消息和渲染结果等多处 C++ 到 Kotlin 的字符串转换。

## 11.8　Swift 中两个不同的显式释放问题

`LiteRT-LM#2589`[^ch11-issue-2589] 与 `LiteRT-LM#2613`[^ch11-issue-2613] 都要求 Swift 提供显式释放，但涉及的对象和故障条件不同。前者讨论 `Conversation`，后者讨论 `Engine`，不能合并为同一个生命周期缺陷。

`LiteRT-LM#2589` 报告的是 v0.12.0 Swift API。[^ch11-issue-2589] 报告者在单 session 实现路径上遇到 `A session already exists`。旧 `Conversation` 的原生资源尚未释放时，创建下一个 conversation 失败。该 issue 请求为 `Conversation` 增加公开的 `close()`，以便调用方不依赖 ARC 的释放时机。

冻结的 v0.13.1 仍只在 `deinit` 中删除 Swift `Conversation` 句柄，见 `swift/Conversation.swift:65-69`：

```swift
public class Conversation {                    // (1)
  private var handle: CConversationHandle?
  // ...
  deinit {                                      // (2)
    if let handle = handle {
      litert_lm_conversation_delete(handle)     // (3)
    }
  }
```

(1) 是 ARC 管理的引用类型，(2) 在最后一个强引用释放后执行，(3) 删除 C conversation 句柄。调用方把一个变量设为 `nil`，并不能保证其他闭包、任务或对象没有继续持有引用。`deinit` 适合作为最终清理路径，却不能代替由调用方确定时点的 `close()`。

Kotlin 对同一种资源提供了公开的确定释放接口。`Conversation` 实现 `AutoCloseable`（`kotlin/java/com/google/ai/edge/litertlm/Conversation.kt:69-74`），并在 `close()` 中删除原生句柄（`kotlin/java/com/google/ai/edge/litertlm/Conversation.kt:466-470`）：

```kotlin
class Conversation(
  private val handle: Long,
  val toolManager: ToolManager = ToolManager(),
  val automaticToolCalling: Boolean = true,
) : AutoCloseable {                                    // (1)
  private val _isAlive = AtomicBoolean(true)
  // ...
  override fun close() {                              // (2)
    if (_isAlive.compareAndSet(true, false)) {
      LiteRtLmJni.nativeDeleteConversation(handle)    // (3)
    } else {
      throw IllegalStateException("Conversation is closed already.")
    }
  }
}
```

(1) 声明 `AutoCloseable`，(2) 允许调用方主动释放，(3) 同步删除原生 `Conversation*`。原子状态转换防止重复删除。第二次 `close()` 会抛出异常。调用方也可以使用 Kotlin 的 `use { }` 在作用域结束时执行 `close()`。`LiteRT-LM#2589` 用这个 `Conversation.close()` 对照 Swift API，讨论对象不是 `Engine.close()`。[^ch11-issue-2589]

`LiteRT-LM#2589` 不能证明 v0.13.1 的核心引擎普遍只允许一个 session。该 issue 明确记录的环境是 v0.12.0，错误栈落在当时的 session 实现。[^ch11-issue-2589] v0.13.1 的 `runtime/framework/resource_management/resource_manager.h:44-45` 则明确说明，共享资源可供多个 session 使用。该案例说明 Swift `Conversation` 缺少公开的确定释放接口。报告中的单 session 约束只属于其记录的版本与实现路径。

`LiteRT-LM#2613` 是独立的 Engine 级案例。[^ch11-issue-2613] v0.13.1 的 `swift/Engine.swift:203-206` 也只在 `deinit` 中调用 `litert_lm_engine_delete`。该 issue 报告，actor 的 `deinit` 不受 actor 隔离保护。删除操作可能在释放最后一个强引用的线程上执行。issue 请求增加 actor 隔离的公开 `Engine.close()`，使删除与其他 Engine 操作串行。这个案例涉及 Engine 销毁的执行上下文，与 Conversation 的单 session 报错不同。

两个 issue[^ch11-issue-2589][^ch11-issue-2613] 都反映了自动释放接口的同一项限制：调用方不能直接指定原生删除的时点或隔离域。Conversation 与 Engine 的故障条件不同，修复接口仍需分别设计。

## 11.9　并发隔离：actor 与 synchronized

Swift 和 Kotlin 都限制对 Engine 句柄的并发访问，但采用不同的语言机制。Swift 在 `swift/Engine.swift:28-38` 把 `Engine` 声明为 actor，并把 `handle` 定义为其可变状态。actor 外部调用隔离方法时需要 `await`。`await` 是潜在挂起点，但不保证发生 OS 线程切换，也不表示 actor 拥有专用后台线程。

`initialize()` 的注释明确提醒调用方避免阻塞主线程，见 `swift/Engine.swift:52-57`：

```swift
  /// **Note:** This operation can take a significant amount of time (e.g., 10 seconds) depending on
  /// the model size and device hardware. It is strongly recommended to call this method on a
  /// background thread to avoid blocking the main thread.
  public func initialize() throws {
```

`initialize()` 是同步的 actor 隔离方法。它取得 actor 隔离后会同步调用原生初始化，函数体内没有可见的挂起点。因此，`actor` 声明本身不能视为后台执行保证。应用仍需按平台集成方式安排耗时初始化，并验证主线程是否会被占用。

Kotlin 的 `Engine` 定义一把锁和 `@Volatile` 句柄（`kotlin/java/com/google/ai/edge/litertlm/Engine.kt:36-45`）。`initialize()` 与 `close()` 使用 `synchronized(lock)`（`kotlin/java/com/google/ai/edge/litertlm/Engine.kt:61-104`），`createConversation()` 也使用同一把锁（`kotlin/java/com/google/ai/edge/litertlm/Engine.kt:115-118`）。锁提供互斥，`@Volatile` 提供跨线程可见性。二者都不负责把同步原生调用调度到后台。Kotlin 的 `initialize()` 同样要求调用方选择合适的线程或协程调度器。

源码可以确认两种隔离方式的语义差异：Swift 使用编译器检查的 actor 隔离，Kotlin 使用运行时锁。没有基准数据时，不能据此比较跨 actor 调用与锁的性能，也不能推断哪一种机制在本场景中开销更低。

## 11.10　测试替身：FakeLlmExecutor

真实模型推理依赖模型文件和后端环境，不适合承担全部单元测试。LiteRT-LM 提供测试替身 `FakeLlmExecutor`，见 `runtime/executor/fake_llm_executor.h:37-59`：

```cpp
class FakeLlmExecutor : public LlmExecutor {    // (1)
 public:
  // ...
  FakeLlmExecutor(
      int vocab_size, const std::vector<std::vector<int>>& prefill_tokens_set,
      const std::vector<std::vector<int>>& decode_tokens_set,  // (2)
      int batch_size = 1,
      std::optional<std::vector<float>> audio_embedding = std::nullopt);
```

(1) 表明它实现第 4、5 章讨论的 `LlmExecutor` 接口。(2) 接收两组预设数据：`prefill_tokens_set` 描述各次 prefill 的预期输入，`decode_tokens_set` 描述各次 decode 的预定输出。该实现不加载模型，也不调用真实推理后端。

`runtime/executor/fake_llm_executor.cc:128-158` 的 `Prefill` 同时检查调用次数和输入 token：

```cpp
absl::Status FakeLlmExecutor::Prefill(const ExecutorInputs& inputs) {
  RETURN_IF_ERROR(prefill_status_);
  if (prefill_times_ >= prefill_tokens_set_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(  // (1)
        "Prefill function has been called more times than the number of "
        "expected prefill tokens.",
        prefill_times_));
  }
  // ...
  auto text_token_ids_span =
      ReferTensorBufferAsSpan<int>(text_data->GetTokenIds());
  RETURN_IF_ERROR(
      CheckEquivalent(absl::MakeSpan(prefill_tokens_set_[prefill_times_]),  // (2)
                      *text_token_ids_span));
  // ...
  processed_tokens_.AddProcessedTokens(prefill_tokens_set_[prefill_times_]);
  prefill_times_++;                                                         // (3)
  current_step_ += text_token_ids_span->size();
  // ...
  return absl::OkStatus();
}
```

(1) 在调用次数超过预设数据时返回错误。(2) 用 `CheckEquivalent` 比较实际输入与第 `prefill_times_` 组预期 token；不一致时返回 `InvalidArgumentError`。(3) 在成功后推进游标。无约束 decode 则按 `decode_times_` 返回预定 token。上层只依赖 `LlmExecutor` 接口。测试因此可以在不加载模型的情况下覆盖 prefill/decode 编排、停止条件和错误传播。该测试替身不能验证真实模型数值、后端 kernel 或设备性能。

`FakeLlmExecutor` 还实现了约束解码测试路径。它从预定 token 构造 logits，调用真实约束器修改 logits，再把结果转换回 token。对应分支位于 `runtime/executor/fake_llm_executor.cc:177-232`：

```cpp
  std::vector<std::vector<int>> output_tokens;
  if (decode_params.HasConstraintDecoder()) {                 // (1)
    auto constraint_decoder = decode_params.GetConstraintDecoder();
    // ...
    if (last_op_ == LastOp::kDecode) {                        // (2)
      // ...
      const auto& last_decode_tokens = decode_tokens_set_[decode_times_ - 1];
      // ...
      RETURN_IF_ERROR(
          constraint_decoder->UpdateConstraintState(last_token_ids));  // (3)
    }
    // ...
    DecodeIdsToLogits(decode_tokens_set_[decode_times_], vocab_size_,
                      output_logits);                          // (4)
    RETURN_IF_ERROR(constraint_decoder->MaskLogits(output_logits));    // (5)
    output_tokens = DecodeLogitsToIds(batch_size_, vocab_size_, output_logits,
                                      decode_tokens_set_);      // (6)
  } else {
    for (int i = 0; i < decode_tokens_set_[decode_times_].size(); ++i) {
      output_tokens.push_back({decode_tokens_set_[decode_times_][i]});  // (7)
    }
  }
  last_op_ = LastOp::kDecode;                                  // (8)
```

(7) 是无约束分支，直接返回预定 token。约束分支在 (4) 用 `DecodeIdsToLogits` 构造 logits，在 (5) 调用 `MaskLogits`，再由 (6) 转回 token。(2)、(3) 在连续 decode 时先用上一轮 token 更新约束状态。(8) 记录上一项操作的类型。`FakeLlmExecutor` 因而能确定性测试状态更新与 logits mask 的接口交互，但不能替代真实模型 logits 上的集成测试。

## 11.11　跨平台构建

LiteRT-LM 为 Android、iOS、Linux、macOS、Windows 和 Web 提供构建目标，并依赖 sentencepiece、llguidance、Skia 等第三方组件。仓库以 Bazel 构建文件为主，同时提供 CMake 构建入口。不同绑定还要分别生成 Android 原生库、Swift package 或 WebAssembly 产物。相关复现命令见附录 C。

## 小结

LiteRT-LM 的多语言 API 采用三类原生边界。Python 与 Swift 经 C ABI 调用 runtime。Kotlin 通过 JNI 直接持有 C++ 指针，Web 通过 Embind 导出 C++ 对象。流式回调要求绑定层在回调期间复制短生命周期字符串；多模态 C 接口则主动复制输入缓冲。返回字符串的有效期由所属句柄限定。JNI 另行处理标准 UTF-8 与 modified UTF-8 的差异。

资源管理和并发隔离属于绑定 API 契约的一部分。`LiteRT-LM#2589` 讨论 `Conversation` 的确定释放。[^ch11-issue-2589] `LiteRT-LM#2613` 讨论 `Engine` 销毁的执行上下文。[^ch11-issue-2613] 两者需要分别处理。Swift actor 与 Kotlin `synchronized` 只定义访问隔离，不自动提供后台线程调度。`FakeLlmExecutor` 支持脱离真实模型的确定性测试。跨平台构建配置则生成各平台所需产物。

---

## 练习与自查

1. 封装原理。C 头文件里的 `LiteRtLmEngine` 只有前向声明、没有成员。这样设计的收益和代价分别是什么？
2. 生命周期案例。分别说明 `LiteRT-LM#2589`[^ch11-issue-2589] 与 `LiteRT-LM#2613`[^ch11-issue-2613] 涉及的对象、报告条件和建议接口。为什么不能用其中一个 issue 证明另一个问题？
3. 所有权辨析。跨 FFI 边界返回的字符串由谁拥有，谁负责释放？由错误的一方释放会产生什么后果？
4. 测试设计。`FakeLlmExecutor` 不含神经网络，为什么仍能单测停止序列和采样编排？哪些内容不在其测试范围内？
5. 设计题。为 Go 语言编写一个最小绑定，至少需要包装哪些 C 函数？按“创建、使用、销毁”三个阶段列出。

[^ch11-issue-2589]: google-ai-edge/LiteRT-LM，[*[Swift] Add a public `close()` method to `Conversation` for deterministic session release*](https://github.com/google-ai-edge/LiteRT-LM/issues/2589)，LiteRT-LM issue #2589，2026-06-16；访问日期：2026-07-18。

[^ch11-issue-2613]: google-ai-edge/LiteRT-LM，[*[Swift] Engine teardown crashes with `litert_lm_engine_delete` running on an arbitrary thread in `deinit` - adding a public `close()` to solve*](https://github.com/google-ai-edge/LiteRT-LM/issues/2613)，LiteRT-LM issue #2613，2026-06-19；访问日期：2026-07-18。
