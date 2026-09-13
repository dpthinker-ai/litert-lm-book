# 第 12 章 多语言绑定：C ABI、JNI 与 Embind

> 本章分析 Python、Kotlin、Swift 与 Web 如何经 C ABI、JNI 或 Embind 进入同一套 C++ runtime，并说明流式回调、数据复制、字符串所有权和显式释放的边界契约。本章还比较绑定层的并发隔离方式，并介绍测试替身与跨平台构建。

LiteRT-LM 的 README 列出 Python、Kotlin、Swift、JavaScript、Flutter 和 C++ 六种 API。[^ch11-readme] 这些绑定并不共用同一条原生调用路径：Python 与 Swift 经过 C 兼容的接口头文件，Kotlin 的 JNI 和 Web 的 Embind 直接调用 C++。本章分析其中 Python、Kotlin、Swift 与 Web 四种绑定，Flutter 不展开。前四节说明三类原生边界各自怎样持有原生对象、怎样配对创建与释放（12.1 至 12.4 节）；中间四节说明数据经过边界时的契约，包括流式回调、多模态输入的复制、字符串的所有权与编码，以及 Swift 上两个显式释放问题（12.5 至 12.8 节）；最后三节说明并发隔离、测试替身与跨平台构建。

## 12.1　C ABI：C 兼容的原生边界

C++ 的名字修饰（name mangling）、异常、模板和对象布局都随编译器与标准库的 ABI 变化。Python 的 `ctypes` 与 Swift 的 C 互操作只接受 C 兼容的符号和数据类型，不能直接导入任意 C++ 类的方法。LiteRT-LM 因此提供一组 C 头文件作为接口：它是 Python 与 Swift 共用的 C 兼容边界，但不是 Kotlin 和 Web 的必经路径。

引擎头文件先用不透明句柄（opaque handle）声明 engine 和 session，两者只有结构体类型名，不公开成员：

```cpp
// c/engine.h:40-48
// Opaque pointer for the LiteRT LM Engine.
//
// Added in version 0.1.0.
typedef struct LiteRtLmEngine LiteRtLmEngine;  // (1)

// Opaque pointer for the LiteRT LM Session.
//
// Added in version 0.1.0.
typedef struct LiteRtLmSession LiteRtLmSession;  // (2)
```

代码行 `(1)`、`(2)` 是 C 的前向声明。编译器知道 `LiteRtLmEngine` 是一个结构体类型，但头文件不公开其字段，绑定层只能把 `LiteRtLmEngine*` 当作句柄传回 C API，不能解引用、计算 `sizeof` 或访问成员。C++ 对象的布局因而不进入公开 ABI。

该结构体在内部头文件中的完整定义如下：

```cpp
// c/engine_internal.h:68-74
struct LiteRtLmEngine {
  std::unique_ptr<litert::lm::Engine> engine;  // (1)
};

struct LiteRtLmSession {
  std::unique_ptr<litert::lm::Engine::Session> session;
};
```

代码行 `(1)` 表明 `LiteRtLmEngine` 持有 `std::unique_ptr<Engine>`。头文件只公开句柄类型，内部头文件保留句柄字段与 `Engine` 的具体布局；只要函数签名和句柄契约保持兼容，内部 C++ 类型就可以独立演进。

操作接口是普通的 C 函数，签名中只出现 C 类型和不透明指针：`litert_lm_engine_create` 返回句柄，`litert_lm_engine_delete` 释放它，创建 session 与执行 prefill 的函数采用同一形式。C++ 成员函数就这样映射为 C 自由函数，对象句柄成为第一个参数。头文件注释写明调用方负责用 delete 函数销毁 engine，这是调用方必须履行的释放契约。

<figure>
{{#include figs/fig-12-1.svg}}
<figcaption>图 12-1　Python 与 Swift 经 C ABI 调用 runtime，Kotlin 与 Web 分别经 JNI 和 Embind 直接调用 C++；三条边界路径复用同一套核心实现。</figcaption>
</figure>

## 12.2　创建与释放必须配对

不透明句柄隐藏了对象布局，也就要求 API 明确对象的所有权。C 接口没有 C++ 的析构语义，调用方必须让创建和释放成对：`litert_lm_engine_create` 返回的引擎由 `litert_lm_engine_delete` 释放，session 由对应的 delete 函数释放。引擎创建与释放的实现如下：

```cpp
// c/engine.cc:790-809
LiteRtLmEngine* litert_lm_engine_create(
    const LiteRtLmEngineSettings* settings) {
  if (!settings || !settings->settings) {
    return nullptr;  // (1)
  }

  absl::StatusOr<std::unique_ptr<Engine>> engine =
      EngineFactory::CreateDefault(*settings->settings);

  if (!engine.ok()) {
    ABSL_LOG(ERROR) << "Failed to create engine: " << engine.status();
    return nullptr;  // (2)
  }

  auto* c_engine = new LiteRtLmEngine;  // (3)
  c_engine->engine = *std::move(engine);
  return c_engine;
}

void litert_lm_engine_delete(LiteRtLmEngine* engine) { delete engine; }  // (4)
```

代码行 `(3)` 用 `new` 分配句柄，再把工厂返回的 `unique_ptr<Engine>` 移入其中；`(4)` 删除句柄，句柄析构又触发 `unique_ptr<Engine>` 析构。RAII 仍负责句柄内部的清理，但跨语言调用方必须显式调用 delete，才能启动这条析构链。`(1)`、`(2)` 以 `nullptr` 表示创建失败，这条 C 接口没有把内部的 `absl::Status` 细节传给调用方。

绑定层需要把 create/delete 契约映射为本语言的资源管理接口：Python 提供 `close()`、上下文管理器和 `__del__`，Kotlin 使用 `AutoCloseable`，Swift 则主要依赖 `deinit`。

## 12.3　三类原生边界

四种绑定采用三类原生边界：Python 与 Swift 使用 C ABI，Kotlin 使用 JNI，Web 使用 Embind。

Python 使用 `ctypes`，在运行时声明 C 函数签名并调用共享库。引擎句柄的类型声明如下：

```python
# python/litert_lm/_ffi.py:253-256
  # Engine
  lib.litert_lm_engine_create.restype = ctypes.c_void_p  # (1)
  lib.litert_lm_engine_create.argtypes = [ctypes.c_void_p]
  lib.litert_lm_engine_delete.argtypes = [ctypes.c_void_p]  # (2)
```

代码行 `(1)` 把创建函数的返回类型声明为 `c_void_p`，对应 C 侧的不透明指针；`(2)` 声明删除函数接收同一类型。`ctypes` 在运行时根据这些声明调用共享库，无需另行编译扩展模块。绑定里的字符串辅助类型还会在传参时统一做 UTF-8 编码。

Kotlin/Android 使用 JNI：一个内部对象集中声明 `external fun`，由编译后的原生实现提供，其中创建与删除 engine 的声明如下：

```kotlin
// kotlin/java/com/google/ai/edge/litertlm/LiteRtLmJni.kt:53-98
  external fun nativeCreateEngine(
    modelPath: String,
    backend: String,
  // ...
    audioBackendNumThreads: Int,
    maxVisionTokensPerImage: Int,
  ): Long  // (1)
  // ...
  external fun nativeDeleteEngine(enginePointer: Long)  // (2)
```

代码行 `(1)` 以 `Long` 保存原生指针的位模式，`(2)` 把同一数值传给删除函数。Kotlin 不解释这个数值指向的对象。JNI 需要一层编译后的原生实现把 `jlong` 转回 C++ 指针，这一点与运行时声明签名的 `ctypes` 不同。

Swift（iOS 与 macOS）使用 C 互操作，导入 C 模块后可直接调用 `litert_lm_engine_create`。它的 `Engine` 是一个 actor：

```swift
// swift/Engine.swift:28-38
public actor Engine {  // (1)
  // ...
  private var handle: OpaquePointer? = nil  // (2)
```

代码行 `(1)` 把 `Engine` 定义为 actor：actor 隔离使其可变状态只能在该 actor 的隔离域内访问，这项语义不等于固定线程或后台线程调度。`(2)` 用 `OpaquePointer?` 保存 C 句柄。与 Python 的 `c_void_p` 和 Kotlin 的 `Long` 一样，绑定代码只保存并传递句柄，不访问原生对象布局。

Web 把核心编译为 WebAssembly，再用 TypeScript 封装。该路径使用 Emscripten 的 Embind，而不是 C 头文件。Embind 对象提供 `.delete()`，TypeScript 封装在释放 WebAssembly 侧的 Engine 对象时调用它。

三类边界最终调用同一套 C++ runtime，但这不足以证明各语言 API 的行为逐项一致：默认参数、错误翻译、调度方式和绑定层预处理都可能造成差异，本书也没有做跨语言逐输出的对照实验。图 12-1 表达的是实现复用关系，不是行为等价结论。

## 12.4　JNI 与 Embind 直接调用 C++

Kotlin 和 Web 不经过 C 头文件。Kotlin 的 JNI 原生实现直接包含 runtime 的 C++ 头文件，能访问 `Engine`、`Engine::Session` 和 `EngineFactory`，不需要经不透明句柄中转。创建 engine 的原生函数返回指针：

```cpp
// kotlin/java/com/google/ai/edge/litertlm/jni/litertlm.cc:688-695
  auto engine = EngineFactory::CreateDefault(*settings);
  if (!engine.ok()) {
    ThrowLiteRtLmJniException(
        env, "Failed to create engine: " + engine.status().ToString());
    return 0;
  }

  return reinterpret_cast<jlong>(engine->release());  // (1)
```

`EngineFactory::CreateDefault` 返回 `absl::StatusOr<std::unique_ptr<Engine>>`。代码行 `(1)` 调用 `release()` 后，`unique_ptr` 不再拥有该 `Engine`，裸指针随后被重解释为 `jlong`，Kotlin 保存的就是这个数值句柄。原生对象仍需由释放函数删除：

```cpp
// kotlin/java/com/google/ai/edge/litertlm/jni/litertlm.cc:781-783
JNI_METHOD(nativeDeleteEngine)(JNIEnv* env, jclass thiz, jlong engine_pointer) {
  delete reinterpret_cast<Engine*>(engine_pointer);  // (1)
}
```

代码行 `(1)` 把 `jlong` 重解释为 `Engine*` 后直接删除。对比 C ABI：`litert_lm_engine_delete` 先删除句柄结构体，句柄里的 `unique_ptr` 再删除 `Engine`；JNI 则直接删除 `Engine` 本体，Kotlin 路径完全不经过 C 头文件。

Web 的 Embind 也在编译期从 C++ 导出 JavaScript 可见对象。源码可以确认三种边界路径及各自的句柄形态，但没有说明项目选择这些路径的全部设计理由。因此，表 12-1 只记录实现事实，不把 FFI 类型概括为普遍的选型规则。

| 语言 | FFI 机制 | 经 C ABI？ | 句柄形态 | 资源释放位置 |
|---|---|---|---|---|
| Python | ctypes（运行时声明签名） | 是 | `c_void_p` | `close()` / 上下文管理器 / `__del__` |
| Kotlin | JNI（`external fun` 声明） | 否，直连 C++ | `jlong`（`Engine*` 位模式） | `AutoCloseable` |
| Swift | C 互操作（import C 头） | 是 | `OpaquePointer` | 由 `deinit` 释放 |
| Web | Emscripten Embind | 否，直连 C++ | 带 `.delete()` 的 JS 对象 | 手动 `.delete()` 配对 |

> 表 12-1　四种语言绑定采用三类原生边界，并以不同类型保存原生句柄。

## 12.5　流式生成如何跨越 FFI 边界

阻塞式接口在函数返回时交付结果，流式生成则要多次传递增量文本和终止状态。C ABI 用回调表达这组异步事件，并为回调参数规定明确的生命周期。

C ABI 用函数指针接收回调，参数包括透传的用户数据指针和不透明的流式块句柄：

```cpp
// c/engine.h:1192-1193
typedef void (*LiteRtLmStreamCallback)(void* callback_data,
                                       const LiteRtLmStreamChunk* chunk);
```

`callback_data` 是调用方登记的 `void*`，C ABI 将它原样传回。绑定层可用它定位本语言的对象或闭包上下文。`chunk` 指向 `LiteRtLmStreamChunk`，访问函数分别读取文本、终止标记和错误消息。头文件规定该句柄只在本次回调期间有效；绑定层若要在回调返回后保留文本或错误消息，必须在回调内复制内容。

C++ 侧用 `absl::AnyInvocable` 接收 `absl::StatusOr<Responses>`，再把结果包装成流式块。下面保留错误分支和增量文本分支：

```cpp
// c/engine.cc:56-98
absl::AnyInvocable<void(absl::StatusOr<litert::lm::Responses>)> CreateCallback(
    LiteRtLmStreamCallback callback, void* callback_data) {
  return [callback,
          callback_data](absl::StatusOr<litert::lm::Responses> responses) {
    if (!responses.ok()) {
      LiteRtLmStreamChunk chunk;
      chunk.text = nullptr;
      chunk.is_final = true;
      std::string error_str = responses.status().ToString();
      chunk.error_msg = error_str.c_str();  // (1)
      callback(callback_data, &chunk);  // (2)
      return;
    }
  // ...
    } else {
      for (const auto& text : responses->GetTexts()) {
        LiteRtLmStreamChunk chunk;
        chunk.text = text.data();  // (3)
        chunk.is_final = false;
        chunk.error_msg = nullptr;
        callback(callback_data, &chunk);  // (4)
      }
    }
  };
}
```

错误分支在代码行 `(1)` 把局部字符串的指针写入 `chunk`，`(2)` 将局部流式块的地址交给回调。增量文本分支在 `(3)` 引用响应中的字符串，再由 `(4)` 调用回调。两条路径都没有转移字符串或流式块的所有权，绑定层不能把这些指针保存到回调之外。

省略的终止分支也使用同一种流式块：正常结束时终止标记为 true、错误消息为空；达到 token 上限和取消时，终止标记同样为 true，但带有对应错误消息。回调签名因此保持两个参数，各项数据通过句柄的访问函数读取。

发起流式的入口：

```cpp
// c/engine.cc:949-962
int litert_lm_session_run_decode_async(LiteRtLmSession* session,
                                       LiteRtLmStreamCallback callback,
                                       void* callback_data) {
  if (!session || !session->session) {
    return -1;  // (1)
  }
  auto status =
      session->session->RunDecodeAsync(CreateCallback(callback, callback_data));
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to start decode stream: " << status.status();
    return static_cast<int>(status.status().code());  // (2)
  }
  return 0;  // (3)
}
```

该函数先把 C 回调和 `callback_data` 交给适配函数，再调用 `RunDecodeAsync`。代码行 `(1)` 在参数无效时返回 -1，`(2)` 在启动失败时返回 `absl::StatusCode` 的整数值，`(3)` 在成功启动时返回 0。返回码只描述启动结果，后续文本和终止状态均由回调传递；绑定层还要保证回调上下文至少存活到最终回调完成。

## 12.6　多模态输入在 C 边界的表示与复制

第 11 章说明了图像和音频如何变成 embedding 进入 prefill。本节只说明文本、图像和音频怎样表示为 C ABI 可接收的数据。调用方提供类型标签、缓冲指针与字节数，由创建函数返回不透明的输入句柄：

```cpp
// c/engine.h:453-483
typedef enum {
  kLiteRtLmInputDataTypeText,
  kLiteRtLmInputDataTypeImage,
  kLiteRtLmInputDataTypeImageEnd,
  kLiteRtLmInputDataTypeAudio,
  kLiteRtLmInputDataTypeAudioEnd,
} LiteRtLmInputDataType;
  // ...
LiteRtLmInputData* litert_lm_input_data_create(LiteRtLmInputDataType type,
                                               const void* data, size_t size);
  // ...
void litert_lm_input_data_delete(LiteRtLmInputData* input_data);
```

`type` 决定如何解释 `(data, size)`：文本采用 UTF-8，图像与音频使用相应的编码字节。调用方不能直接构造句柄内部的 C++ 对象，也必须用 `litert_lm_input_data_delete` 释放创建的输入句柄。创建函数的文本、图像与结束标记分支如下：

```cpp
// c/engine.cc:126-153
LiteRtLmInputData* litert_lm_input_data_create(LiteRtLmInputDataType type,
                                               const void* data, size_t size) {
  switch (type) {
    case kLiteRtLmInputDataTypeText:
      return std::make_unique<LiteRtLmInputData>(
                 litert::lm::InputText(
                     std::string(static_cast<const char*>(data), size)))  // (1)
          .release();
    case kLiteRtLmInputDataTypeImage:
      return std::make_unique<LiteRtLmInputData>(
                 litert::lm::InputImage(
                     std::string(static_cast<const char*>(data), size)))  // (2)
          .release();
    case kLiteRtLmInputDataTypeImageEnd:
      return std::make_unique<LiteRtLmInputData>(litert::lm::InputImageEnd())  // (3)
          .release();
  // ...
    default:
      return nullptr;
  }
}
```

代码行 `(1)`、`(2)` 用 `std::string(ptr, size)` 复制原始缓冲，分别构造 `InputText` 和 `InputImage`。`(3)` 创建不含缓冲的 `InputImageEnd`，音频分支采用相同结构。创建函数返回后，调用方可以释放原始字节缓冲；输入句柄仍须保留到提交调用完成。

提交时，C API 接收输入句柄指针数组，再为 runtime 构造独立的输入副本：

```cpp
// c/engine.cc:100-114
absl::StatusOr<std::vector<litert::lm::InputData>> ToEngineInputData(
    const LiteRtLmInputData* const* inputs, size_t num_inputs) {
  std::vector<litert::lm::InputData> engine_inputs;
  engine_inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    if (inputs[i] != nullptr) {
      auto copy_status = litert::lm::CreateInputDataCopy(inputs[i]->data);  // (1)
      if (!copy_status.ok()) {
        return copy_status.status();
      }
      engine_inputs.push_back(std::move(*copy_status));  // (2)
    }
  }
  return engine_inputs;
}
```

代码行 `(1)` 调用 `CreateInputDataCopy`，复制失败时返回状态；`(2)` 把副本移入输入数组。对这里的文本、图像和音频字节输入，创建句柄与提交输入是两个独立的复制步骤。C API 不接管调用方输入句柄的所有权，调用方仍需在提交调用返回后释放句柄。

这种接口让原始缓冲、C 输入句柄和 runtime 输入各自拥有内容，代价是复制及其临时内存。本书没有测量这些复制在端到端时延中的占比，也不能将当前接口当作零复制输入通道。

## 12.7　跨语言字符串的所有权与编码

跨 FFI 返回 `const char*` 时，接口必须规定指针的有效期和字符编码。LiteRT-LM 分别在 C ABI 与 JNI 层处理这两个问题。

取响应文本的函数返回内部字符串的 `data()`：

```cpp
// c/engine.cc:1027-1036
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

代码行 `(1)` 返回的指针所指内存归 `LiteRtLmResponses` 所有：它只在 responses 句柄存活期间有效，句柄删除后调用方不得继续访问，也不得单独释放这个 `const char*`。若需要延长字符串的生命周期，必须在删除句柄前复制内容。

消息渲染会产生一段新的字符串，conversation 句柄用一个成员保存它：

```cpp
// c/conversation_internal.h:45-52
struct LiteRtLmConversation {
  std::unique_ptr<litert::lm::Conversation> conversation;
  // This field stores the result of the last call to
  // `litert_lm_conversation_render_message_to_string`. This ties the lifetime
  // of the returned `const char*` to the `LiteRtLmConversation` object,
  // ...
  std::string last_rendered_message;  // (1)
```

渲染函数把结果移入该成员，再返回它的 `c_str()`：

```cpp
// c/conversation.cc:623-624
  conversation->last_rendered_message = std::move(*rendered);  // (1)
  return conversation->last_rendered_message.c_str();  // (2)
```

代码行 `(1)` 让 conversation 句柄拥有渲染结果，`(2)` 返回其 C 字符串指针。下一次渲染会覆盖这个成员，删除 conversation 也会销毁它；绑定层若要保留旧结果，必须在下一次渲染或删除句柄之前复制。调用方不需要单独释放这个指针。

JNI 的 `NewStringUTF` 接收的是 modified UTF-8。它对空字符和基本多文种平面（BMP）之外字符的编码与标准 UTF-8 不同，不能直接用来转换标准 UTF-8 文本。JNI 层因此自定义了一个转换函数：

```cpp
// kotlin/java/com/google/ai/edge/litertlm/jni/litertlm.cc:122-164
jstring NewStringStandardUTF(JNIEnv* env, std::string standard_utf8_str) {
  // ...
  jbyteArray bytes = env->NewByteArray(standard_utf8_str.length());  // (1)
  if (bytes == nullptr) return nullptr;
  env->SetByteArrayRegion(
      bytes, 0, standard_utf8_str.length(),
      reinterpret_cast<const jbyte*>(standard_utf8_str.c_str()));  // (2)
  // ...
  jclass string_class = env->FindClass("java/lang/String");
  // ...
  jmethodID string_ctor =
      env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");  // (3)
  // ...
  jstring charset_name = env->NewStringUTF("UTF-8");  // (4)
  // ...
  jstring result =
      (jstring)env->NewObject(string_class, string_ctor, bytes, charset_name);  // (5)
  // ...
  return result;
}
```

代码行 `(1)` 创建 `byte[]`，`(2)` 写入标准 UTF-8 字节，`(3)` 取得 `String(byte[], String charsetName)` 构造器，`(4)` 创建 ASCII 字符串 `"UTF-8"`，`(5)` 显式按该字符集构造 Java `String`。省略的代码处理失败分支并清理 JNI 局部引用。响应文本、错误消息和渲染结果等多处 C++ 到 Kotlin 的字符串转换都经过这个函数。

## 12.8　Swift 中两个不同的显式释放问题

`LiteRT-LM#2589`[^ch11-issue-2589] 与 `LiteRT-LM#2613`[^ch11-issue-2613] 都要求 Swift 提供显式释放，但涉及的对象和故障条件不同：前者讨论 `Conversation`，后者讨论 `Engine`，不能合并为同一个生命周期缺陷。

`LiteRT-LM#2589` 报告的是 v0.12.0 的 Swift API。[^ch11-issue-2589] 报告者在单 session 实现路径上遇到 `A session already exists`：旧 `Conversation` 的原生资源尚未释放时，创建下一个 conversation 失败。该 issue 请求为 `Conversation` 增加公开的 `close()`，以便调用方不依赖 ARC 的释放时机。

当前的 Swift 封装仍只在 `deinit` 中删除 `Conversation` 句柄：

```swift
// swift/Conversation.swift:57-91
public final class Conversation: Sendable {  // (1)
  // ...
  private let handle: CConversationHandle?
  // ...
  private let engine: Engine
  // ...
  deinit {  // (2)
    if let handle = handle {
      litert_lm_conversation_delete(handle)  // (3)
    }
  }
```

代码行 `(1)` 是 ARC 管理的 `final` 引用类型，并声明遵守 `Sendable`。它保存不可变的句柄和一个 `Engine` 强引用，使引擎至少存活到该 conversation 释放。`(2)` 在最后一个强引用释放后执行，`(3)` 删除 C conversation 句柄。调用方把一个变量设为 `nil`，并不能保证其他闭包、任务或对象没有继续持有引用。`deinit` 适合作为最终清理路径，却不能代替由调用方确定时点的 `close()`。

Kotlin 对同一种资源提供了公开的确定释放接口。`Conversation` 实现 `AutoCloseable`，并在 `close()` 中删除原生句柄：

```kotlin
// kotlin/java/com/google/ai/edge/litertlm/Conversation.kt:69-749
class Conversation(
  private val handle: Long,
  val toolManager: ToolManager = ToolManager(),
  val automaticToolCalling: Boolean = true,
  val enableResponseFormat: Boolean = false,
) : AutoCloseable {  // (1)
  private val _isAlive = AtomicBoolean(true)
  // ...
  override fun close() {  // (2)
    if (_isAlive.compareAndSet(true, false)) {
      LiteRtLmJni.nativeDeleteConversation(handle)  // (3)
    } else {
      throw IllegalStateException("Conversation is closed already.")
    }
  }
```

代码行 `(1)` 声明 `AutoCloseable`，`(2)` 允许调用方主动释放，`(3)` 同步删除原生 `Conversation*`。原子状态转换防止重复删除，第二次 `close()` 会抛出异常；调用方也可以用 Kotlin 的 `use { }` 在作用域结束时执行 `close()`。`LiteRT-LM#2589` 正是用这个 `Conversation.close()` 对照 Swift API，讨论对象不是 `Engine.close()`。[^ch11-issue-2589]

`LiteRT-LM#2589` 不能证明当前核心引擎普遍只允许一个 session。该 issue 明确记录的环境是 v0.12.0，错误栈位于当时的 session 实现。[^ch11-issue-2589] 本书冻结版本的资源管理器头文件注释则明确说明，共享资源可供多个 session 使用。该案例说明的是 Swift `Conversation` 缺少公开的确定释放接口；报告中的单 session 约束只属于其记录的版本与实现路径。

`LiteRT-LM#2613` 是独立的 Engine 级案例。[^ch11-issue-2613] 当前的 Swift 封装也只在 `deinit` 中调用 `litert_lm_engine_delete`。该 issue 报告，actor 的 `deinit` 不受 actor 隔离保护，删除操作可能在释放最后一个强引用的线程上执行；issue 请求增加 actor 隔离的公开 `Engine.close()`，使删除与其他 Engine 操作串行。这个案例涉及 Engine 销毁的执行上下文，与 Conversation 的单 session 报错不同。

两个 issue[^ch11-issue-2589][^ch11-issue-2613] 都反映了自动释放接口的同一项限制：调用方不能直接指定原生删除的时点或隔离域。Conversation 与 Engine 的故障条件不同，修复接口仍需分别设计。

## 12.9　并发隔离：actor 与 synchronized

Swift 和 Kotlin 都限制对 Engine 句柄的并发访问，但采用不同的语言机制。Swift 把 `Engine` 声明为 actor，并把 `handle` 定义为其可变状态，actor 外部调用隔离方法时需要 `await`。`await` 是潜在挂起点，但不保证发生 OS 线程切换，也不表示 actor 拥有专用后台线程。

`initialize()` 的注释明确提醒调用方避免阻塞主线程：

```swift
// swift/Engine.swift:52-57
  /// **Note:** This operation can take a significant amount of time (e.g., 10 seconds) depending on
  /// the model size and device hardware. It is strongly recommended to call this method on a
  /// background thread to avoid blocking the main thread.
  ///
  /// - Throws: A `LiteRTLMError` if the engine fails to initialize.
  public func initialize() throws {
```

`initialize()` 是同步的 actor 隔离方法，经内部辅助函数同步调用原生初始化。这条调用路径没有可见的挂起点。因此，`actor` 声明本身不能视为后台执行保证，应用仍需按平台集成方式安排耗时初始化，并验证主线程是否会被占用。

Kotlin 的 `Engine` 定义一把锁和 `@Volatile` 句柄。`initialize()` 与 `close()` 使用 `synchronized(lock)`，`createConversation()` 也使用同一把锁。锁提供互斥，`@Volatile` 提供跨线程可见性，二者都不负责把同步的原生调用调度到后台；Kotlin 的 `initialize()` 同样要求调用方选择合适的线程或协程调度器。

源码可以确认两种隔离方式的语义差异：Swift 使用编译器检查的 actor 隔离，Kotlin 使用运行时锁。没有基准数据时，不能据此比较跨 actor 调用与锁的性能，也不能推断哪一种机制在本场景中开销更低。

## 12.10　测试替身：FakeLlmExecutor

真实模型推理依赖模型文件和后端环境，不适合承担全部单元测试。LiteRT-LM 提供测试替身 `FakeLlmExecutor`：

```cpp
// runtime/executor/fake_llm_executor.h:38-68
class FakeLlmExecutor : public LlmExecutor {  // (1)
 public:
  // ...
  FakeLlmExecutor(int vocab_size,
                  const std::vector<std::vector<int>>& prefill_tokens_set,
                  const std::vector<std::vector<int>>& decode_tokens_set,  // (2)
                  int batch_size = 1,
                  std::optional<std::vector<float>> projected_audio_embedding =
                      std::nullopt);
```

代码行 `(1)` 表明它实现第 4、5 章讨论的 `LlmExecutor` 接口；`(2)` 接收两组预设数据：`prefill_tokens_set` 描述各次 prefill 的预期输入，`decode_tokens_set` 描述各次 decode 的预定输出。该实现不加载模型，也不调用真实推理后端。

它的 `Prefill` 同时检查调用次数和输入 token：

```cpp
// runtime/executor/fake_llm_executor.cc:134-169
absl::Status FakeLlmExecutor::Prefill(const ExecutorInputs& inputs) {
  ABSL_RETURN_IF_ERROR(prefill_status_);
  if (prefill_times_ >= prefill_tokens_set_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(  // (1)
        "Prefill function has been called more times than the number of "
        "expected prefill tokens.",
        prefill_times_));
  }
  // ...
  ABSL_ASSIGN_OR_RETURN(auto text_data, inputs.GetTextDataPtr());
  LITERT_ASSIGN_OR_RETURN(
      auto text_token_ids_span,
      ReferTensorBufferAsSpan<int>(text_data->GetTokenIds()));
  ABSL_RETURN_IF_ERROR(CheckEquivalent<const int>(  // (2)
      absl::MakeSpan(prefill_tokens_set_[prefill_times_]),
      text_token_ids_span));
  last_op_ = LastOp::kPrefill;
  processed_tokens_.AddProcessedTokens(prefill_tokens_set_[prefill_times_]);
  prefill_times_++;  // (3)
  current_step_ += text_token_ids_span.size();
  // ...
  return absl::OkStatus();
}
```

代码行 `(1)` 在调用次数超过预设数据时返回错误；`(2)` 比较实际输入与第 `prefill_times_` 组预期 token，不一致时返回 `InvalidArgumentError`；`(3)` 在成功后推进游标。无约束的 decode 则按调用次数返回预定 token。上层只依赖 `LlmExecutor` 接口，测试因此可以在不加载模型的情况下覆盖 prefill/decode 编排、停止条件和错误传播；该测试替身不能验证真实模型数值、后端 kernel 或设备性能。

`FakeLlmExecutor` 还实现了约束解码测试路径：从预定 token 构造 logits，调用真实的约束解码处理器修改 logits，再把结果转换回 token：

```cpp
// runtime/executor/fake_llm_executor.cc:201-238
  std::vector<std::vector<int>> output_tokens;
  if (ConstrainedDecoder* constrained_decoder =
          decode_params.GetConstrainedDecoder();
      constrained_decoder != nullptr) {  // (1)
  // ...
    if (last_op_ == LastOp::kDecode) {  // (2)
  // ...
      const auto& last_decode_tokens = decode_tokens_set_[decode_times_ - 1];
  // ...
      ABSL_RETURN_IF_ERROR(constrained_decoder->UpdateState(last_token_ids));  // (3)
    }
  // ...
    DecodeIdsToLogits(decode_tokens_set_[decode_times_], vocab_size_,
                      output_logits, decode_logits_options_);  // (4)
  // ...
    ABSL_RETURN_IF_ERROR(constrained_decoder->ProcessLogits(output_logits));  // (5)
    output_tokens = DecodeLogitsToIds(batch_size_, vocab_size_, output_logits,
                                      decode_tokens_set_);  // (6)
  } else {
    for (int i = 0; i < decode_tokens_set_[decode_times_].size(); ++i) {
      output_tokens.push_back({decode_tokens_set_[decode_times_][i]});  // (7)
    }
  }
  last_op_ = LastOp::kDecode;  // (8)
```

代码行 `(7)` 是无约束分支，直接返回预定 token。约束分支在 `(4)` 构造 logits，在 `(5)` 调用 `ProcessLogits`，再由 `(6)` 转回 token；`(2)`、`(3)` 在连续 decode 时先用上一轮 token 更新约束状态，`(8)` 记录上一项操作的类型。`FakeLlmExecutor` 因而能确定性地测试状态更新与 logits mask 的接口交互，但不能替代真实模型 logits 上的集成测试。

## 12.11　跨平台构建

LiteRT-LM 为 Android、iOS、Linux、macOS、Windows 和 Web 提供构建目标，并依赖 sentencepiece、llguidance、Skia 等第三方组件。仓库以 Bazel 构建文件为主，同时提供 CMake 构建入口。不同绑定还要分别生成 Android 原生库、Swift package 或 WebAssembly 产物。相关复现命令见附录 C。

## 小结

LiteRT-LM 的多语言 API 采用三类原生边界：Python 与 Swift 经 C ABI 调用 runtime，Kotlin 通过 JNI 直接持有 C++ 指针，Web 通过 Embind 导出 C++ 对象。流式回调要求绑定层在回调期间复制短生命周期字符串；多模态 C 接口在创建输入句柄与提交输入时分别复制内容。返回字符串的有效期由所属句柄限定，JNI 另行处理标准 UTF-8 与 modified UTF-8 的差异。

资源管理和并发隔离属于绑定 API 契约的一部分。`LiteRT-LM#2589` 讨论 `Conversation` 的确定释放，[^ch11-issue-2589] `LiteRT-LM#2613` 讨论 `Engine` 销毁的执行上下文，[^ch11-issue-2613] 两者需要分别处理。Swift actor 与 Kotlin `synchronized` 只定义访问隔离，不自动提供后台线程调度。`FakeLlmExecutor` 支持脱离真实模型的确定性测试，跨平台构建配置则生成各平台所需产物。

---

## 练习与自查

1. 封装原理。C 头文件里的 `LiteRtLmEngine` 只有前向声明、没有成员。这样设计的收益和代价分别是什么？
2. 生命周期案例。分别说明 `LiteRT-LM#2589`[^ch11-issue-2589] 与 `LiteRT-LM#2613`[^ch11-issue-2613] 涉及的对象、报告条件和建议接口。为什么不能用其中一个 issue 证明另一个问题？
3. 所有权辨析。跨 FFI 边界返回的字符串由谁拥有，谁负责释放？由错误的一方释放会产生什么后果？
4. 测试设计。`FakeLlmExecutor` 不含神经网络，为什么仍能单测停止序列和采样编排？哪些内容不在其测试范围内？
5. 设计题。为 Go 语言编写一个最小绑定，至少需要包装哪些 C 函数？按“创建、使用、销毁”三个阶段列出。

[^ch11-issue-2589]: google-ai-edge/LiteRT-LM，[*[Swift] Add a public `close()` method to `Conversation` for deterministic session release*](https://github.com/google-ai-edge/LiteRT-LM/issues/2589)，LiteRT-LM issue #2589，2026-06-16；访问日期：2026-07-18。

[^ch11-issue-2613]: google-ai-edge/LiteRT-LM，[*[Swift] Engine teardown crashes with `litert_lm_engine_delete` running on an arbitrary thread in `deinit` - adding a public `close()` to solve*](https://github.com/google-ai-edge/LiteRT-LM/issues/2613)，LiteRT-LM issue #2613，2026-06-19；访问日期：2026-07-18。
[^ch11-readme]: Google AI Edge，[LiteRT-LM README](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.17.0/README.md#L108-L113)，版本 v0.17.0；访问日期：2026-09-13。
