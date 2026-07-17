# 第 11 章 一套核心，六种语言：C ABI、绑定与工程纪律

> 使命：一套 C++ 核心怎么变成 Python、Kotlin、Swift、Web 都能调用的 SDK。看清那层作为通用桥的 C ABI，走读流式回调、多模态输入、字符串所有权如何跨越 FFI 边界，再剖析一个真实的跨语言生命周期缺陷；最后看这套代码怎么保证自己可测、可构建。

前面十章讲的都是 C++ 核心里发生了什么。但真正用它的人多半不写 C++：App 开发者用 Kotlin、Swift，脚本作者用 Python，前端用 JavaScript，跨平台还有 Flutter。官方的语言 API 数下来正好六种：Python、Kotlin、Swift、JavaScript、Flutter，再加上直接调用的 C++ 核心本身。本章标题里的「六种语言」由此而来。下面挑其中最能代表不同接法的几种细看（Flutter 走的是与它们类似的桥接路子）。一套核心怎么服务这么多语言，这是最后一块工程拼图。

## 为什么中间要隔一层 C

直觉上，既然核心是 C++，各语言直接调 C++ 不就行了？不行。C++ 的名字修饰（name mangling）各家编译器不同，异常、模板、对象内存布局都不跨语言稳定，你没法可靠地从 Python 直接调一个 C++ 类的方法。

通行的解法是退一步：先把核心收敛成一层纯 **C 接口**。几乎每种语言都能调 C（这是几十年攒下的事实标准），所以只要有一层 C ABI，就为所有语言提供了统一的接入点。LiteRT-LM 的这层接口在 `c/engine.h`。

它的写法有两个要点。其一，**不透明句柄**（opaque handle）。`c/engine.h:41`、`:44` 把 engine 和 session 声明成两个只有名字、没有成员的结构体：

```cpp
// Opaque pointer for the LiteRT LM Engine.
typedef struct LiteRtLmEngine LiteRtLmEngine;    // (1)

// Opaque pointer for the LiteRT LM Session.
typedef struct LiteRtLmSession LiteRtLmSession;  // (2)
```

(1)、(2) 这两行是 C 里前向声明的标准手法：只告诉编译器「有个叫 `LiteRtLmEngine` 的结构体类型」，不给出它的字段。绑定层因此永远只能持有一个 `LiteRtLmEngine*` 指针，无法解引用、无法 `sizeof`、无法访问任何成员，C++ 侧真正的类型被这层声明挡在门外。这正是「不透明」的含义：句柄能传递、能当参数，但里面装了什么，跨语言的一侧看不见也不需要看见。

那 C++ 侧到底装了什么？答案在实现文件里。`c/engine.cc:176` 给出这个结构体的完整定义：

```cpp
struct LiteRtLmEngine {
  std::unique_ptr<Engine> engine;    // (1)
};

struct LiteRtLmSession {
  std::unique_ptr<Engine::Session> session;
};
```

(1) 这一行揭示了整层 C ABI 的结构：每个不透明句柄，内部只是一个持有真实 C++ 对象的 `std::unique_ptr`。头文件里那个没有字段的结构体，到了 `.cc` 里补全成裹着一个 `unique_ptr<Engine>` 的结构体。绑定层拿到的 `LiteRtLmEngine*`，本质是指向一个包着 C++ Engine 的盒子的指针，盒子的形状、`Engine` 的内存布局，全被 `.cc` 独占，头文件一无所知。头文件与实现文件对同一个类型给出两种视图，是把 C++ 复杂性隔离在 C ABI 之内的关键一招。

其二，**纯 C 函数**。所有操作都是普通 C 函数，签名里只出现 C 类型和不透明指针。`c/engine.h:380` 的 create、`:386` 的 delete：

```cpp
// Creates a LiteRT LM Engine from the given settings. The caller is responsible
// for destroying the engine using `litert_lm_engine_delete`.
// ...
LiteRtLmEngine* litert_lm_engine_create(const LiteRtLmEngineSettings* settings);  // (1)

// Destroys a LiteRT LM Engine.
// ...
void litert_lm_engine_delete(LiteRtLmEngine* engine);                             // (2)
```

(1) 返回一个不透明指针，(2) 收回它。除此之外还有 `litert_lm_engine_create_session`（`:396`）开会话、`litert_lm_session_run_prefill`（`:421`）跑 prefill。这些函数名，正是第 3 章那套 Engine/Session 接口的 C 语言镜像：C++ 的 `engine->CreateSession()` 在这里映射为 `litert_lm_engine_create_session(engine, ...)`，成员函数变自由函数，`this` 变第一个参数。注释里那句 "The caller is responsible for destroying the engine" 不是套话，它是下一节整节要处理的契约。

<figure>
{{#include figs/fig-11-1.svg}}
<figcaption>图 11-1　一套核心，六种语言：C++ 核心先收敛成一层 C ABI（不透明句柄 + 纯 C 函数），各语言再用各自的 FFI 机制（ctypes / JNI / C 互操作 / WASM）接上去。所有语言最终都进入同一套 runtime。（严格说 Kotlin 的 JNI 与 Web 的 Embind 直连 C++，见「JNI 的例外」一节；图中为版式简化画在同一层。）</figcaption>
</figure>

## 谁创建，谁释放

不透明句柄带来一个必须讲清的问题：内存谁来管。

C 没有析构函数，也没有垃圾回收。所以 C ABI 的规矩是**创建和释放成对**：`litert_lm_engine_create` 造出来的引擎，得由 `litert_lm_engine_delete`（`c/engine.h:386`）释放；session 有 `litert_lm_session_delete`（`:403`）。这对函数在实现侧短到几乎没有内容，`c/engine.cc:537`：

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

看 (3) 和 (4) 的对称。create 用 `new` 在堆上造一个 `LiteRtLmEngine` 盒子、把工厂产出的 `unique_ptr<Engine>` 塞进去，返回裸指针；delete 一个 `delete engine` 收场，`delete` 触发盒子的析构，盒子里的 `unique_ptr` 随之析构，真正的 `Engine` 才被销毁。这就是「谁创建谁释放」的全部实现：一个 `new` 配一个 `delete`，中间隔着 FFI 边界。一旦跨过去，C++ 的 RAII（对象析构自动清理）失效了，绑定语言那侧拿到的只是个裸指针，编译器不会替它调析构。(1)、(2) 两处返回 `nullptr` 也值得记：C ABI 用返回空指针表达失败，因为 C 没有异常也没有 `StatusOr`；C++ 内部的 `absl::Status` 到了边界一律被翻译成「指针或 NULL」这种 C 能懂的信号。

各语言怎么接这份契约？各有各的惯用形态：Python 交给上下文管理器或 `__del__`，Kotlin 落在 `AutoCloseable` 上，到了 Swift 则是 `deinit`。绑定层的主要工作，很大一部分就是把手动配对的 create/delete 包装成本语言里自然的资源管理。本章后面会看到，这件事一旦没接好，会出真问题。

## 各语言的 FFI，各有各的接法

有了 C ABI，剩下的是每种语言用自己的方式接上去。机制不同，目标一致：都是调到那层 C 函数。

**Python 用 `ctypes`**：运行时按签名声明 C 结构和函数，直接调共享库。声明写在 `python/litert_lm/_ffi.py` 里，`:171` 这几行把不透明句柄的处理暴露得很清楚：

```python
# Engine
lib.litert_lm_engine_create.restype = ctypes.c_void_p    # (1)
lib.litert_lm_engine_create.argtypes = [ctypes.c_void_p]
lib.litert_lm_engine_delete.argtypes = [ctypes.c_void_p] # (2)
```

(1) 把 `litert_lm_engine_create` 的返回类型声明成 `c_void_p`，一个不带类型的裸指针。这就是不透明句柄在 Python 里的样子：那个 `LiteRtLmEngine*` 到了 Python，退化成一个整数化的 void 指针，Python 侧同样碰不到内部。(2) 声明 delete 收一个 void 指针。ctypes 的价值就在这里：不用编译任何胶水代码，只要在运行时把 C 函数的 restype/argtypes 声明对，Python 就能直接 `lib.litert_lm_engine_create(...)`。字符串跨界也有专门处理，`_ffi.py:24` 定义了 `c_string_p`，在 `from_param` 里自动 `obj.encode("utf-8")`，替调用者省掉每次手动编码。

**Kotlin/Android 用 JNI**：声明一串 `external fun`，由 JNI 桥到原生库。`kotlin/.../LiteRtLmJni.kt:19` 是个 `internal object LiteRtLmJni`，里面全是外部函数声明，`:52` 起是 `nativeCreateEngine`：

```kotlin
external fun nativeCreateEngine(
  modelPath: String,
  backend: String,
  // ...
  audioBackendNumThreads: Int,
): Long                                        // (1)

external fun nativeDeleteEngine(enginePointer: Long)  // (2)
```

(1) 的返回类型是 `Long`：这是 JNI 里表达不透明句柄的惯用法，把 C++ 那个指针当成一个 64 位整数，原样在 Kotlin 和原生之间传。(2) 的 delete 收回同一个 `Long`。Kotlin 侧从头到尾不知道这个 `Long` 指向什么，它只是个不透明的标识值，用完要原样交还。JNI 与 ctypes 的路子相反：JNI 需要一层用 C++ 写、名字按 JNI 规则拼出来的原生实现，把 `Long` 转回真指针再往下调，这层胶水是编译期就要产出的，不像 ctypes 全在运行时。这层原生实现具体调到哪一层，下一节会看到一个出乎意料的答案。

Swift（iOS 与 macOS）走的是 C 互操作，直接 `import` C 头文件调用。`swift/Engine.swift:17` 的 `import CLiteRTLM` 之后，就能像调 Swift 函数一样调 `litert_lm_engine_create`。它的 `Engine` 是一个 `actor`（`:28`）：

```swift
public actor Engine {                         // (1)
  // ...
  private var handle: OpaquePointer? = nil     // (2)
```

(1) 用 `actor` 而非 `class`：Swift 的 actor 保证同一时刻只有一个任务能碰它的可变状态，等于用并发原语替原生引擎串行化访问，不必手写锁（对照 Kotlin 侧是显式 `synchronized(lock)`）。(2) 把句柄存成 `OpaquePointer?`，Swift 自带的不透明指针类型，语义与 Python 的 `c_void_p`、Kotlin 的 `Long` 一致：一个不能解引用、只能转交的指针。三种语言，三种类型名，同一个概念。

**Web 把核心编成 WebAssembly**，用 TypeScript 包一层在浏览器里跑（`js/packages/core`）。它走的不是原始 C ABI，而是 Emscripten 的 Embind（C++ 与 JS 之间的绑定机制）对象，句柄以带 `.delete()` 方法的 JS 对象出现（如 `js/packages/core/src/engine.ts` 里对 wasm 对象反复调 `.delete()`），但手动配对释放这条约束没变，只是换了张脸。

同一个 prompt，走 Python 和走 C++ 会得到一致的行为，因为它们最终进的是同一套 `runtime`（这也是本章开头那句「一套核心」的具体印证）。各语言的 SDK 看起来风格迥异，底下是同一个引擎；四种 FFI 机制的差别，全在如何抵达那层 C 函数、如何表示那个不透明句柄这一层。这层 C ABI 是各语言共同的调用基线，也是它们能共享同一套语义的原因。

## JNI 的例外：Android 侧直连 C++ 核心

上一节把四种 FFI 机制并排讲，隐含一个整齐的叙述：各语言都经由 `c/engine.h` 那层 C ABI 抵达核心。图 11-1 也是这么画的。但 Kotlin 侧其实是个例外，它没走 C ABI（上一节交代过，Web 走的 Embind 也不是那层原始 C ABI——这一点在本节末尾一并修正）。

翻开 JNI 的原生实现，第一处线索是它 include 的头文件。`kotlin/java/com/google/ai/edge/litertlm/jni/litertlm.cc:37`（下文简写 `litertlm.cc`）：

```cpp
#include "runtime/engine/engine.h"          // (1)
#include "runtime/engine/engine_factory.h"  // (2)
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
```

(1)、(2) 直接包含的是 `runtime/engine/` 下的 C++ 头，而不是 `c/engine.h`。也就是说，JNI 这层胶水看得见完整的 C++ 类型 `Engine`、`Engine::Session`、`EngineFactory`，它不需要经过不透明句柄的中转。再看句柄怎么造出来，`nativeCreateEngine` 的收尾一行，`litertlm.cc:542`：

```cpp
auto engine = EngineFactory::CreateDefault(*settings);
if (!engine.ok()) {
  ThrowLiteRtLmJniException(
      env, "Failed to create engine: " + engine.status().ToString());
  return 0;
}

return reinterpret_cast<jlong>(engine->release());  // (1)
```

(1) 是关键。`EngineFactory::CreateDefault` 返回 `absl::StatusOr<std::unique_ptr<Engine>>`，这里对成功值调 `engine->release()`：从 `unique_ptr` 手里夺走裸指针 `Engine*`、把所有权移出智能指针，再 `reinterpret_cast<jlong>` 把这个裸指针原样重解释成一个 64 位整数返回给 Kotlin。上一节说的「Kotlin 侧那个 `Long` 是一个 64 位整数」，到这里落到实处：它就是一个 `Engine*` 的位模式。释放侧对称，`litertlm.cc:615`：

```cpp
JNI_METHOD(nativeDeleteEngine)(JNIEnv* env, jclass thiz, jlong engine_pointer) {
  delete reinterpret_cast<Engine*>(engine_pointer);  // (1)
}
```

(1) 把 `jlong` 重解释回 `Engine*`，直接 `delete`。这与 C ABI 那层 `litert_lm_engine_delete` 里的 `delete engine`（`c/engine.cc:556`）殊途同归，但对象是不同的：C ABI delete 的是那个裹着 `unique_ptr` 的 `LiteRtLmEngine` 盒子，JNI 这里 delete 的是 `Engine` 本体。整条 Kotlin 通路，从 `nativeCreateEngine` 到 `nativeDeleteEngine`，压根没碰 `c/engine.h`。

于是要修正上一节和图 11-1 的一处隐含说法。严格讲，不是四种语言都经 C ABI，而是 Python、Swift 经 C ABI，Kotlin 与 Web 都直连 C++（图 11-1 把四者画在同一层上，是为版式简化）。这个选择有它的道理。JNI 桥无论如何都要用 C++ 写、要编译，既然胶水本来就是 C++、和核心一起编进同一个 `.so`，那么让它直接调 `EngineFactory` 与 `Engine`，比先绕一道 C ABI 再由 C ABI 转调 C++ 少一层间接，也省掉不透明句柄的封装与拆封；Web 的 Embind 同理，wasm 胶水本来也是 C++ 编的，直接导出带方法的对象即可。Python 和 Swift 的情况相反：ctypes 在运行时按 C 签名找符号，C 互操作直接 import C 头，两者都以稳定的 C ABI 为前提，绕不开那层 C。据此推断，绑定层选 C ABI 还是直连 C++，取决于这门语言的 FFI 是否本就要编一层 C++ 胶水：要编，就没有再套一层 C 的必要；不编，C ABI 就是唯一可依赖的稳定边界。

| 语言 | FFI 机制 | 经 C ABI？ | 句柄形态 | 资源释放落点 |
|---|---|---|---|---|
| Python | ctypes（运行时声明签名） | 是 | `c_void_p` | 上下文管理器 / `__del__` |
| Kotlin | JNI（`external fun` 声明） | 否，直连 C++ | `jlong`（`Engine*` 位模式） | `AutoCloseable` |
| Swift | C 互操作（import C 头） | 是 | `OpaquePointer` | `deinit`（ARC） |
| Web | Emscripten Embind | 否，直连 C++ | 带 `.delete()` 的 JS 对象 | 手动 `.delete()` 配对 |

> 表 11-1　四种语言的 FFI 机制对照。分野的判据：这门语言的 FFI 是否本就要编一层 C++ 胶水。

## 流式生成如何跨越 FFI 边界

到目前为止讲的都是阻塞式调用：create 一次、prefill 一次、拿一个结果。但生成式模型的自然形态是流式，一个 token 一个 token 地产出，UI 要边生成边显示。流式意味着回调，而回调要跨过 FFI 边界，是绑定设计里更硬的一块。

C ABI 表达回调的方式是函数指针加一个透传的用户数据指针。`c/engine.h:652` 的类型定义：

```cpp
typedef void (*LiteRtLmStreamCallback)(void* callback_data, const char* chunk,
                                       bool is_final, const char* error_msg);
```

四个参数各司其职：`callback_data` 是调用方登记时给的 `void*`，C ABI 原样透传、自己不解释，绑定层用它把 C 回调接回本语言的对象（Python 的一个 `self`、Swift 的一个闭包上下文）；`chunk` 是这一小段增量文本；`is_final` 标记流是否结束；`error_msg` 非空即表示出错。头文件注释点破了一处易踩的边界：`chunk` "It's only valid for the duration of the call"，即回调返回后这个指针就失效，绑定层要在回调内部立刻把内容拷走。

C++ 侧的回调是 `absl::AnyInvocable`，语义比 C 函数指针丰富得多（能捕获状态、能持有 `absl::Status`）。把它翻译成 C 那套四参数函数指针，是 `CreateCallback` 的活，`c/engine.cc:50`：

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

这个函数返回一个 lambda，lambda 把 C++ 侧丰富的 `StatusOr<Responses>` 塌缩成 C 那四个标量参数。走读它的分派：(1) `StatusOr` 本身失败，把状态串成字符串塞进 `error_msg`、`is_final` 置真、文本给 `nullptr`；此后按 `TaskState` 分四路。(2) `kDone` 是正常收尾，`is_final` 真、`error_msg` 空。(3) `kMaxNumTokensReached` 也是终止，但携带 "Max number of tokens reached." 作提示。(4) `kCancelled` 同理带 "CANCELLED."。(5) 是流式主路，`GetTexts()` 里每条增量各调一次回调，`is_final` 为假、文本走 `text.data()`。

注意 (5) 里的 `text.data()`。`text` 是循环里的局部引用，`data()` 指向它内部缓冲；这个指针在回调返回后就无意义，正好呼应头文件注释那句「只在调用期内有效」。C++ 侧不会为绑定语言延长这段内存的寿命，责任被明确甩给回调实现方：要么在回调内拷走，要么用完即弃。这是流式 FFI 的一条通则，也是最容易漏的地方，一旦绑定层把 `chunk` 指针存下来留到回调之外用，就是读已释放内存。

发起流式的入口，`c/engine.cc:653`：

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

它把 C 的函数指针加 `callback_data` 交给 `CreateCallback` 包成 C++ lambda，再喂给 `RunDecodeAsync`。返回值也遵循 C ABI 的信号约定：(1) 参数非法返回 `-1`，(2) 启动失败返回 `absl::Status` 的错误码（一个 int），(3) 成功返回 0。这里的返回值只表示「流是否成功启动」，真正的生成结果全走回调异步送达，这是异步接口与前面阻塞式 create/delete 的根本差别。各语言绑定要做的，是在自己这侧准备好一个 C 可调用的函数、把本语言的观察者对象打包进 `callback_data`，然后就等着那层 C++ lambda 一段段回调上来。

## 多模态输入在 C 边界的扁平化

第 10 章讲过多模态的核心机制，图像和音频怎么进 KV cache。这里补一个绑定视角：一段图像字节、一段音频字节、一段文本，形态各异，怎么统一穿过那层只认 C 类型的 ABI。

C 这侧的表示是一个打了标签的结构体，`c/engine.h:243`：

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

一个 `type` 标签加一段 `(data, size)` 的裸内存。这是 C 里表达「一段任意字节」的标准形态：(1) `const void*` 不带类型，(2) `size` 给出长度，文本时 `data` 是 UTF-8 串，图像音频时是原始字节。绑定层把一次多模态输入拆成这样一个数组，每个元素带一个标签，穿过边界。

C++ 侧再把这个扁平数组重新立体化，`c/engine.cc:127` 的 `ToEngineInputData`：

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

一个 `switch` 按标签分派，把每个 C 结构体还原成对应的 C++ 类型：(1) 文本包成 `InputText`，(2) 图像包成 `InputImage`，(3) `ImageEnd` 这类不带数据的标记直接构造一个空对象作分界。音频两路对称。这里有个绕不开的性能事实：(1)、(2) 里的 `std::string(ptr, size)` 是一次实实在在的拷贝，从 C 那侧的裸缓冲复制进一个新分配的 `std::string`。一张图的编码字节可能几十上百 KB，每次 `GenerateContent` 都要照量拷一遍。

这份拷贝能不能省？在这层边界上，答案基本是不能。零拷贝要求 C++ 侧只持有指向 C 缓冲的指针、不复制，但 C ABI 无法约束调用方那段 `data` 内存活多久：调用方完全可能在函数返回后就释放它，而 `InputData` 会被移进 pipeline、其寿命远超这次调用。要做到零拷贝，就得在 C ABI 层引入所有权转移或生命周期契约（谁在什么时候释放这段内存），那会把这层刻意收敛的简单接口复杂化。当前的选择是拿一次确定的拷贝换接口的简单与内存安全。对文本和短音频，这份拷贝可忽略；对大图，它是绑定层一处已知的固定开销。这个权衡与前一节流式回调把 `chunk` 生命周期限定在调用期内是同一套思路的两面：一边靠拷贝保证安全，一边靠「只在调用期内有效」把不拷贝的责任交给调用方。

## 跨语言字符串的所有权与编码

字符串是 FFI 里最不起眼、又最容易出错的一类数据。返回一个 `const char*` 看着简单，难点全在两个词：谁负责释放，用什么编码。LiteRT-LM 在这两处各有一个值得走读的处理。

先看所有权。C++ 函数要把一段文本还给绑定层，最直接的写法是返回内部字符串的 `data()`。`c/engine.cc:725` 的 `get_response_text_at` 就这么干：

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

(1) 返回的指针依附于 `LiteRtLmResponses` 对象：只要那个 responses 句柄没被 delete，这段内存就有效。注释把契约写得很直白，字符串的寿命绑在句柄上。这样绑定层就不必为每次取文本配一次单独的释放调用，句柄活着，串就活着；句柄一 delete，串跟着走。代价是调用方必须记住，取出的 `const char*` 不能比它来源的句柄活得久。

但有一类返回值没有天然的宿主句柄可依附。渲染一条消息成字符串，输入是 JSON、输出是一段新生成的文本，它不属于任何已有对象。这时候的解法是给句柄挂一个成员来兜住这段内存。`c/engine.cc:192` 的结构体：

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

(1) 这个 `last_rendered_message` 成员就是宿主。渲染函数把结果存进它、再返回它的 `c_str()`，`c/engine.cc:1115-1116` 收尾两行：

```cpp
  conversation->last_rendered_message = std::move(*rendered);  // (1)
  return conversation->last_rendered_message.c_str();          // (2)
```

(1) 把渲染结果 move 进句柄的成员，(2) 返回该成员的 C 串指针。于是返回的 `const char*` 生命周期又一次绑在句柄上，绑定层无需 per-call free。注意 (1) 每次渲染都覆盖上一次的结果，成员名 `last_rendered_message` 也点明了这点：只保留最近一次。绑定层若要留住旧结果，得自己在下次渲染前拷走。这是一种「把无主返回值寄养到句柄上」的通用手法，避免了在 C ABI 里再引入一个专门的字符串释放函数。

再看编码。前面 Python 侧的 `c_string_p` 自动 `encode("utf-8")` 只是入方向的一半，出方向、尤其在 JNI 里，藏着一个更隐蔽的坑。JNI 有个 `NewStringUTF` 函数能直接把 C 串变 Java `String`，但它只接受所谓 modified UTF-8：这套编码对 emoji、BMP 之外的字符、内嵌 null 的处理都与标准 UTF-8 不同，直接喂标准 UTF-8 串给它，遇到这些字符就会得到乱码或崩溃。LiteRT-LM 的模型输出显然可能带 emoji，所以 JNI 侧绕开了 `NewStringUTF`，`litertlm.cc:104`：

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

这段绕的路是：不让 JNI 替你解码，而是把字节原样交给 Java 的 `String` 构造器、由它按明确指定的字符集解码。(1) 建一个 `byte[]`，(2) 把标准 UTF-8 字节原样填进去（`SetByteArrayRegion` 只搬字节、不做任何编码转换），(3) 取 `String(byte[], String charsetName)` 这个构造器，(4) 造一个内容为 "UTF-8" 的字符集名（这个短串是纯 ASCII，用 `NewStringUTF` 没问题），(5) 用字节数组加字符集名调构造器，得到一个正确解码的 Java `String`。整段代码只为回避 `NewStringUTF` 的 modified-UTF-8 语义，把编码决定权从 JNI 收回到明确的 `String(byte[], "UTF-8")`。省略掉的几行全是 JNI 局部引用的清理（`DeleteLocalRef`），本身也是 JNI 编程里不能漏的一环，局部引用不及时释放会在长循环里耗尽引用表。这个函数在 JNI 层被反复调用，凡是要把 C++ 字符串还给 Kotlin 的地方都走它，是 Android 侧多语言文本正确性的一处基础设施。

## ARC 释放时机与单会话约束的冲突

绑定层最容易出问题的地方是生命周期管理。这里有一个真实的上游案例，正好把前面几节串起来（对应上游 issue，`LiteRT-LM#2589` 与 `#2613`）。

背景是两个约束的相遇。其一，LiteRT-LM 一个引擎同时只允许一个会话（`LiteRT-LM#2589`）。其二，Swift 用 ARC（自动引用计数）管内存，对象什么时候被销毁，取决于引用何时归零，时机是**不确定**的。看 Swift 侧的对象。在 v0.13.1，会话的 Swift 对应物是 `Conversation`，它把释放只放在 `deinit` 里，`swift/Conversation.swift:65`：

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

(1) 是 `class`（引用类型，归 ARC 管），(2) 唯一的释放路径是 `deinit`，(3) 在 `deinit` 里才调到 C ABI 的 delete。问题就出在这条唯一路径上：`deinit` 何时触发，由 ARC 决定，而 ARC 只在最后一个引用归零时才回收对象，这个时点调用方控制不了。于是场景是：你逻辑上用完了一个会话想再开一个，可旧 `Conversation` 对象还被某个引用（一个闭包捕获、一个还没出作用域的局部变量）持有。`deinit` 没跑、(3) 没执行、原生会话还占着那唯一的名额，新会话一创建就遇到「已有一个会话」的约束。

对照 Kotlin 侧，同样的资源却有一条不依赖 GC 的确定释放路径。`kotlin/.../Engine.kt:36` 的 `Engine` 实现 `AutoCloseable`：

```kotlin
class Engine(val engineConfig: EngineConfig) : AutoCloseable {  // (1)
  @Volatile private var handle: Long? = null
  // ...
  override fun close() {                        // (2)
    synchronized(lock) {
      checkInitialized()
      LiteRtLmJni.nativeDeleteEngine(handle!!)   // (3)
      handle = null                              // (4)
    }
  }
```

(1) 实现 `AutoCloseable`，(2) 的 `close()` 是一个使用者能主动、当场调的方法（配合 Kotlin 的 `use { }` 还能在作用域结束时自动触发），(3) 立刻调到原生 delete，(4) 把句柄置空，杜绝二次释放。它不等任何 GC。Python 侧走的是第三条路，`python/litert_lm/engine.py:126` 把 `close()`、`__del__`、`__exit__` 三者同时挂上：`with` 语句退出即释放，`del` 或回收时由 `__del__` 作为保障路径兜住，析构在 `close()` 里判空防重入。

问题的根源，正是第二节那条「谁创建谁释放」没有落到确定的时机上：C ABI 那边 delete 必须被调，但早期 Swift 把释放时机完全交由 ARC 的 `deinit` 决定。issue 里提的解法也顺理成章（`LiteRT-LM#2613`）：给 Swift 加一个公开的 `close()`，让使用者能主动释放会话，无需等待 ARC 回收。这恰好对齐了 Kotlin 的 `AutoCloseable.close()` 与 Python 的 `__exit__`：三种语言最终都得给出一条由调用方显式控制的释放路径，光靠语言自带的自动回收，处理不了「单会话」这种带独占语义的资源。

这个案例落到一句论断上：跨语言桥最难的往往不是调得通，而是两边的资源模型如何对齐。C 的手动配对、Swift 的 ARC、Kotlin 的 AutoCloseable，把三套资源模型衔接到一起而不产生泄漏，是绑定层最容易出 bug 的地方。

## 并发模型：actor 串行化 vs synchronized 临界区

上面那个缺陷是资源释放的时机问题，与它一体两面的是并发访问的串行化。原生引擎的句柄是共享可变状态，多线程同时碰它会出竞态，Swift 和 Kotlin 用了两套不同的并发原语来防这件事，代价也不同。

Swift 侧用 `actor`，`swift/Engine.swift:28`。actor 的语义是编译器保证同一时刻至多一个任务能访问它的可变状态，`handle: OpaquePointer?` 这个字段的并发安全由此免费获得，不用手写锁。代价藏在跨 actor 调用里：从 actor 外调 actor 的方法要 `await`，这是一个潜在的挂起点，调用可能被排队、被切到别的线程恢复。actor 之间的一次调用叫一次 hop，带上下文切换的开销。对单纯读写句柄这种极短临界区，hop 的开销可能比它保护的操作还大。

但 actor 模型在这里有一处恰到好处的收益，和一个特殊操作有关。`initialize()` 加载模型，注释明确标了它可能耗时约 10 秒，`swift/Engine.swift:52` 的说明：

```swift
  /// **Note:** This operation can take a significant amount of time (e.g., 10 seconds) depending on
  /// the model size and device hardware. It is strongly recommended to call this method on a
  /// background thread to avoid blocking the main thread.
  public func initialize() throws {
```

一个耗时约 10 秒的同步操作，在主线程上跑会冻结 UI。actor 的模型天然把这件事推到 actor 自己的执行上下文，调用方 `await engine.initialize()`，当前线程（可能是主线程）在挂起点让出，不被这 10 秒阻塞；初始化在别处跑完再恢复。注释里那句「建议在后台线程调用」，在 actor 加 `await` 的模型下几乎是自动满足的。

Kotlin 侧用的是显式临界区，`kotlin/.../Engine.kt:36`。`@Volatile private var handle: Long?` 加 `synchronized(lock)`，每个碰句柄的方法都进同一把锁的临界区。`@Volatile` 保证 handle 的写对其他线程立即可见，`synchronized` 保证临界区互斥。这套是命令式并发的经典配置，开销集中在锁的获取与释放，临界区内是纯粹的阻塞式执行、没有挂起与线程切换的语义。代价的另一面是，Kotlin 的 `initialize()` 就是个普通同步方法，会一直占着调用线程直到 10 秒跑完，所以它的文档同样写着「强烈建议在后台线程调用」，但这件事得由调用方自己用协程或线程去安排，语言不像 actor 那样替它兜住。

两套模型对着同一个问题给了不同答案。actor 把并发安全和「长耗时操作别阻塞主线程」这两件事统一在挂起模型里，代价是每次跨 actor 访问都有 hop 开销，哪怕临界区极短；synchronized 把并发安全做成显式锁、开销可预测且极低，但「别阻塞主线程」这件事甩回给调用方。据此推断，选 actor 更贴合 Swift Concurrency 生态、让 async/await 的调用方写起来自然，选 synchronized 则更贴合 Android 既有的线程模型与协程调度习惯。两者都对，取决于各自平台的并发生态，不存在一个通用更优解。

## 让核心可测：FakeLlmExecutor

最后两笔工程纪律，一笔关于测试，一笔关于构建。

推理要跑真模型、要硬件，测起来又慢又不稳定。LiteRT-LM 的对策是一个假执行器 `FakeLlmExecutor`，`runtime/executor/fake_llm_executor.h:37`：

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

(1) 它继承 `LlmExecutor`，就是第 4、5 章那层执行器抽象，上层 pipeline 只认这个接口。(2) 构造函数收两组脚本：`prefill_tokens_set` 是每次 prefill *应该*收到的 token，`decode_tokens_set` 是每次 decode *将要产出*的 token。它不加载模型、不碰硬件，全凭这两张表按脚本执行。

脚本怎么执行，先看 `Prefill`，`runtime/executor/fake_llm_executor.cc:128`：

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

这段把 fake 的双重身份讲清了。它不只是返回假数据的桩，还是个**断言器**：(2) 用 `CheckEquivalent` 把上层这次真正喂进来的 token，和脚本里第 `prefill_times_` 条预期逐一比对，对不上就返回 `InvalidArgumentError`，于是测试不但能验证上层拿到了什么，还能验证上层喂进来的是不是对的。(1) 调用次数超出脚本长度直接报错，(3) 每调一次把游标 `prefill_times_` 往前推，让第几次调用对上脚本第几条。decode 侧的无约束分支对称：`decode_tokens_set_[decode_times_]` 按游标取出下一批 token 返回。整套逻辑没有一行涉及神经网络。prefill/decode 的编排、采样、停止条件，全都能在这张脚本上脱离真实模型和硬件跑单测，且每次结果完全确定。这是接口隔离带来的直接收益：上层只依赖 `LlmExecutor`，就能把真执行器整个换成这台脚本机。

fake 还有一条容易被忽略的能力：它能模拟约束解码（第 10 章 llguidance 那套受限生成）。约束解码不是直接吐 token，而是先算 logits、再让约束器 mask 掉不合法的 token、然后从 mask 后的 logits 选 token，fake 要在无模型的前提下把这条链路也演出来。`Decode` 里的分支，`runtime/executor/fake_llm_executor.cc:177`：

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

无约束分支 (7) 就是前面说的直接取脚本 token 返回。约束分支 (1) 起要多绕一圈，目的是让约束器真正参与、从而可被测试。(4) `DecodeIdsToLogits` 把脚本里的目标 token 反算成一组 logits（构造一组恰好在目标 token 上取最大值的假 logits），(5) 把这组 logits 交给真正的 `constraint_decoder->MaskLogits` 施加约束掩码，(6) `DecodeLogitsToIds` 再从掩码后的 logits 还原出 token。这样约束器的 `MaskLogits` 逻辑就在完全确定、无模型的环境里被真实执行和验证了。(2)、(3) 处理状态延续：如果上一步也是 decode，得先用上一步产出的 token 调 `UpdateConstraintState`，把约束器的内部状态推进到当前位置，否则约束器不知道已经生成到哪。(8) 的 `last_op_` 是一个记录「上一次操作是 prefill 还是 decode」的状态机变量，正是它让 (2) 能判断要不要补这次状态更新。整条约束路径没有一次真实前向，却把约束解码的接口交互（更新状态、mask、还原）全跑了一遍，第 10 章那套受限生成因此也能进确定性单测。

## 让核心可构建

这套代码要在 Android、iOS、Linux、macOS、Windows、Web 上都编得出来，还牵着一堆第三方依赖（sentencepiece、llguidance、skia 等）。它用两套构建系统来保障：主用 Bazel，另备一套 CMake 供嵌入式或不便用 Bazel 的场景。这部分属于纯工程实现，本书不展开（细节见附录 C），但值得记住一点：一个能投产到六个平台的运行时，构建系统的分量不亚于运行时本身。

## 小结

一套 C++ 核心服务六种语言，靠的是一层收敛后的 C ABI：不透明句柄藏住 C++ 类型，纯 C 函数当各语言共同的调用基线，各绑定再用 ctypes、JNI、C 互操作、WASM 各自接上。这层桥的复杂度不在阻塞式的 create/delete，而在那些跨边界的动态数据：流式回调把 C++ 的 `AnyInvocable` 塌缩成四参数函数指针、`chunk` 只在回调期内有效；多模态输入拿一次拷贝换接口简单与内存安全；返回字符串把生命周期寄养到句柄、JNI 侧绕开 modified-UTF-8 保住 emoji。Kotlin 与 Web 甚至没走 C ABI：一个以 `Long` 承载 `Engine*` 直连 C++，一个走 Emscripten 的 Embind，因为它们本就要编一层 C++ 胶水。最难的不是接通，而是让两边的资源模型与并发模型对齐，那个 Swift ARC 撞单会话约束的缺陷、actor 与 synchronized 的两套并发取舍，都是明证。底下再有 FakeLlmExecutor 把无约束与约束解码都撑成确定性单测、双构建系统撑起跨平台，这套运行时才算真的能投产。

第四篇到此结束。从三类物理约束与完整的推理流水线，到 KV cache、量化、异构后端与推测解码的各项优化，再到多模态、工具调用与六种语言的绑定，LiteRT-LM 这台端侧推理机器，四个部分讲完了。剩下的尾声，聊聊你可以拿它做什么，以及这条路往前还通向哪里。

---

## 练习与自查

1. **封装原理。** C 头文件里 `LiteRtLmEngine` 只有前向声明、没有成员。这带来什么好处，付出什么代价？
2. **泄漏重演。** 描述 `#2589` 的故障链：从 Swift 侧一个未释放的引用，到「已有一个会话」报错，中间每一环是什么？
3. **所有权辨析。** 跨 FFI 边界返回的字符串，内存归谁、谁负责释放？错误的释放方会导致什么？
4. **测试设计。** `FakeLlmExecutor` 不含任何神经网络，为什么足以单测停止词、采样编排这类逻辑？它测不了什么？
5. **设计题。** 为 Go 语言写一个最小绑定，至少要包装哪几个 C 函数？按「创建-使用-销毁」三段列出。


<!-- #2589/#2613 为 open issue，作缺陷案例研究、按【文档】级引，不宣称已修复。Python/C++ 行为一致性实验与 FakeLlmExecutor 新用例未做（复现命令见附录 C）；表 11-1（各语言 FFI 机制）未出，素材在正文齐备、可补。2026-07-16 评审修订：Kotlin/Web 直连 C++ 的口径已在正文与图注统一。 -->
