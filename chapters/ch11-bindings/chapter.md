# 第 11 章 一套核心，六种语言：C ABI、绑定与工程纪律

> 使命：一套 C++ 核心，怎么变成 Python、Kotlin、Swift、Web 都能调的 SDK。看清那层作为"通用桥"的 C ABI，以及一个真实的跨语言生命周期坑；再顺带看这套代码是怎么保证自己可测、可构建的。

前面十章讲的都是 C++ 核心里发生了什么。但真正用它的人，多半不写 C++——App 开发者用 Kotlin、Swift，脚本作者用 Python，前端用 JavaScript，跨平台还有 Flutter。官方的语言 API 数下来正好六种：Python、Kotlin、Swift、JavaScript、Flutter，再加上直接调用的 C++ 核心本身，这就是本章标题"六种语言"的来历。本章挑其中最能代表不同接法的几种细看（Flutter 走的是与它们类似的桥接路子）。一套核心怎么服务这么多语言？这是最后一块工程拼图。

## 为什么中间要隔一层 C

直觉上，既然核心是 C++，各语言直接调 C++ 不就行了？不行。C++ 的名字修饰（name mangling）各家编译器不同，异常、模板、对象内存布局都不跨语言稳定——你没法可靠地从 Python 直接调一个 C++ 类的方法。

通行的解法是退一步：先把核心收敛成一层纯 **C 接口**。几乎每种语言都能调 C（这是几十年攒下的事实标准），所以只要有一层 C ABI，就等于给所有语言开了门。LiteRT-LM 的这层门在 `c/engine.h`。

它的写法有两个要点。其一，**不透明句柄**（opaque handle）。`c/engine.h:41`、`:44 @ v0.13.1` 把 engine 和 session 声明成两个只有名字、没有成员的结构体：

```cpp
// Opaque pointer for the LiteRT LM Engine.
typedef struct LiteRtLmEngine LiteRtLmEngine;    // (1)

// Opaque pointer for the LiteRT LM Session.
typedef struct LiteRtLmSession LiteRtLmSession;  // (2)
```

(1)、(2) 这两行是 C 里"前向声明"的标准手法：只告诉编译器"有个叫 `LiteRtLmEngine` 的结构体类型"，不给出它的字段。绑定层因此永远只能持有一个 `LiteRtLmEngine*` 指针，无法解引用、无法 `sizeof`、无法访问任何成员——C++ 侧真正的类型被这层声明挡在门外。这正是"不透明"的含义：句柄能传递、能当参数，但里面装了什么，跨语言的一侧看不见也不需要看见。

那 C++ 侧到底装了什么？答案在实现文件里。`c/engine.cc:176 @ v0.13.1` 给出这个结构体的真身：

```cpp
struct LiteRtLmEngine {
  std::unique_ptr<Engine> engine;    // (1)
};

struct LiteRtLmSession {
  std::unique_ptr<Engine::Session> session;
};
```

(1) 一句道破了整层 C ABI 的骨架：每个不透明句柄，内部只是一个持有真实 C++ 对象的 `std::unique_ptr`。头文件里那个"没有字段的结构体"，到了 `.cc` 里补全成"裹着一个 `unique_ptr<Engine>` 的结构体"。绑定层拿到的 `LiteRtLmEngine*`，本质是"指向一个包着 C++ Engine 的盒子的指针"——盒子的形状、`Engine` 的内存布局，全被 `.cc` 独占，头文件一无所知。头文件与实现文件对同一个类型给出两种视图，是把 C++ 复杂性关进 C ABI 之内的关键一招。

其二，**纯 C 函数**。所有操作都是普通 C 函数，签名里只出现 C 类型和不透明指针。`c/engine.h:380 @ v0.13.1` 的 create、`:386` 的 delete：

```cpp
// Creates a LiteRT LM Engine from the given settings. The caller is responsible
// for destroying the engine using `litert_lm_engine_delete`.
// ...
LiteRtLmEngine* litert_lm_engine_create(const LiteRtLmEngineSettings* settings);  // (1)

// Destroys a LiteRT LM Engine.
// ...
void litert_lm_engine_delete(LiteRtLmEngine* engine);                             // (2)
```

(1) 返回一个不透明指针，(2) 收回它。除此之外还有 `litert_lm_engine_create_session`（`:396`）开会话、`litert_lm_session_run_prefill`（`:421`）跑 prefill。这些函数名，正是第 3 章那套 Engine/Session 接口的 C 语言镜像：C++ 的 `engine->CreateSession()` 在这里摊平成 `litert_lm_engine_create_session(engine, ...)`——成员函数变自由函数，`this` 变第一个参数。注释里那句 "The caller is responsible for destroying the engine" 不是客套，它是下一节整节要处理的合同。

<figure>
{{#include figs/fig-11-1.svg}}
<figcaption>图 11-1　一套核心，六种语言：C++ 核心先收敛成一层 C ABI（不透明句柄 + 纯 C 函数），各语言再用各自的 FFI 机制（ctypes / JNI / C 互操作 / WASM）接上去。所有语言最终都进入同一套 runtime。</figcaption>
</figure>

## 谁创建，谁释放

不透明句柄带来一个必须讲清的问题：内存谁来管。

C 没有析构函数，也没有垃圾回收。所以 C ABI 的规矩是**创建和释放成对**：`litert_lm_engine_create` 造出来的引擎，得由 `litert_lm_engine_delete`（`c/engine.h:386 @ v0.13.1`）释放；session 有 `litert_lm_session_delete`（`:403`）。这对函数在实现侧短到几乎没有内容，`c/engine.cc:537 @ v0.13.1`：

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

看 (3) 和 (4) 的对称。create 用 `new` 在堆上造一个 `LiteRtLmEngine` 盒子、把工厂产出的 `unique_ptr<Engine>` 塞进去，返回裸指针；delete 一个 `delete engine` 收场，`delete` 触发盒子的析构，盒子里的 `unique_ptr` 随之析构，真正的 `Engine` 才被销毁。这就是"谁创建谁释放"的全部实现：一个 `new` 配一个 `delete`，中间隔着 FFI 边界。跨过这条边界，C++ 的 RAII（对象析构自动清理）失效了，绑定语言那侧拿到的只是个裸指针，编译器不会替它调析构。(1)、(2) 两处返回 `nullptr` 也值得记：C ABI 用"返回空指针"表达失败，因为 C 没有异常也没有 `StatusOr`；C++ 内部的 `absl::Status` 到了边界一律被翻译成"指针或 NULL"这种 C 能懂的信号。

各语言怎么接这份合同？各有各的惯用形态：Python 用上下文管理器或 `__del__`，Kotlin 用 `AutoCloseable`，Swift 用 `deinit`。绑定层的活，很大一部分就是把"手动配对的 create/delete"包装成本语言里自然的资源管理。下一节会看到，这件事一旦没接好，会出真问题。

## 各语言的 FFI，各有各的接法

有了 C ABI，剩下的是每种语言用自己的方式接上去。机制不同，目标一致——都是调到那层 C 函数。

**Python 用 `ctypes`**：运行时按签名声明 C 结构和函数，直接调共享库。声明写在 `python/litert_lm/_ffi.py` 里，`:170 @ v0.13.1` 这几行把不透明句柄的处理暴露得很清楚：

```python
# Engine
lib.litert_lm_engine_create.restype = ctypes.c_void_p    # (1)
lib.litert_lm_engine_create.argtypes = [ctypes.c_void_p]
lib.litert_lm_engine_delete.argtypes = [ctypes.c_void_p] # (2)
```

(1) 把 `litert_lm_engine_create` 的返回类型声明成 `c_void_p`，一个不带类型的裸指针。这就是不透明句柄在 Python 里的样子：那个 `LiteRtLmEngine*` 到了 Python，退化成一个整数化的 void 指针，Python 侧同样碰不到内部。(2) 声明 delete 收一个 void 指针。ctypes 的价值就在这里：不用编译任何胶水代码，只要在运行时把 C 函数的 restype/argtypes 声明对，Python 就能直接 `lib.litert_lm_engine_create(...)`。字符串跨界也有专门处理——`_ffi.py:24` 定义了 `c_string_p`，在 `from_param` 里自动 `obj.encode("utf-8")`，替调用者省掉每次手动编码。

**Kotlin/Android 用 JNI**：声明一串 `external fun`，由 JNI 桥到原生库。`kotlin/.../LiteRtLmJni.kt:19 @ v0.13.1` 是个 `internal object LiteRtLmJni`，里面全是外部函数声明，`:52` 起是 `nativeCreateEngine`：

```kotlin
external fun nativeCreateEngine(
  modelPath: String,
  backend: String,
  // ...
  audioBackendNumThreads: Int,
): Long                                        // (1)

external fun nativeDeleteEngine(enginePointer: Long)  // (2)
```

(1) 的返回类型是 `Long`：这是 JNI 里表达不透明句柄的惯用法，把 C++ 那个指针当成一个 64 位整数，原样在 Kotlin 和原生之间传。(2) 的 delete 收回同一个 `Long`。Kotlin 侧从头到尾不知道这个 `Long` 指向什么，它只是个"要原样还回去"的令牌。JNI 与 ctypes 的路子相反：JNI 需要一层用 C++ 写、名字按 JNI 规则拼出来的原生实现，把 `Long` 转回真指针再调 C ABI——这层胶水是编译期就要产出的，不像 ctypes 全在运行时。

**Swift（iOS 与 macOS）用 C 互操作**：直接 `import` C 头文件调用，`swift/Engine.swift:17` 的 `import CLiteRTLM`，之后就能像调 Swift 函数一样调 `litert_lm_engine_create`。它的 `Engine` 是一个 `actor`（`:28`）：

```swift
public actor Engine {                         // (1)
  // ...
  private var handle: OpaquePointer? = nil     // (2)
```

(1) 用 `actor` 而非 `class`：Swift 的 actor 保证同一时刻只有一个任务能碰它的可变状态，等于用并发原语替原生引擎串行化访问，不必手写锁（对照 Kotlin 侧是显式 `synchronized(lock)`）。(2) 把句柄存成 `OpaquePointer?`——Swift 自带的不透明指针类型，语义与 Python 的 `c_void_p`、Kotlin 的 `Long` 一致：一个不能解引用、只能转交的指针。三种语言，三种类型名，同一个概念。

**Web 把核心编成 WebAssembly**，用 TypeScript 包一层在浏览器里跑（`js/packages/core @ v0.13.1`）。它走的不是原始 C ABI，而是 Emscripten 生成的 Embind 对象——句柄以带 `.delete()` 方法的 JS 对象出现（如 `js/packages/core/src/engine.ts` 里对 wasm 对象反复调 `.delete()`），但"手动配对释放"这条约束没变，只是换了张脸。

同一个 prompt，走 Python 和走 C++ 会得到一致的行为——因为它们最终进的是同一套 `runtime`（这也是本章开头那句"一套核心"的实感）。各语言的 SDK 看起来风格迥异，底下是同一个引擎；四种 FFI 机制的差别，全在"如何抵达那层 C 函数、如何表示那个不透明句柄"这一层。

## 一个真实的坑：ARC 与"单会话"撞车

绑定层最容易栽的地方，是生命周期。这里有一个真实的上游案例，正好把前面几节串起来（对应上游 issue，`LiteRT-LM#2589` 与 `#2613`，【文档】级）。

背景是两个约束的相遇。其一，LiteRT-LM 一个引擎同时只允许一个会话（`LiteRT-LM#2589`，【文档】级）。其二，Swift 用 ARC（自动引用计数）管内存，对象什么时候被销毁，取决于引用何时归零，时机是**不确定**的。看 Swift 侧的对象——在 v0.13.1，会话的 Swift 对应物是 `Conversation`，它把释放只放在 `deinit` 里，`swift/Conversation.swift:65 @ v0.13.1`：

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

(1) 是 `class`（引用类型，归 ARC 管），(2) 唯一的释放路径是 `deinit`，(3) 在 `deinit` 里才调到 C ABI 的 delete。问题就出在这条唯一路径上：`deinit` 何时触发，由 ARC 决定，而 ARC 只在最后一个引用归零时才回收对象，这个时点你控制不了。于是场景是：你逻辑上"用完"了一个会话想再开一个，可旧 `Conversation` 对象还被某个引用（一个闭包捕获、一个还没出作用域的局部变量）拽着，`deinit` 没跑、(3) 没执行、原生会话还占着那唯一的名额——新会话一创建就撞上"已有一个会话"。

对照 Kotlin 侧，同样的资源却有一条不依赖 GC 的确定释放路径。`kotlin/.../Engine.kt:36 @ v0.13.1` 的 `Engine` 实现 `AutoCloseable`：

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

(1) 实现 `AutoCloseable`，(2) 的 `close()` 是一个使用者能**主动、当场**调的方法（配合 Kotlin 的 `use { }` 还能在作用域结束时自动触发），(3) 立刻调到原生 delete，(4) 把句柄置空，杜绝二次释放。它不等任何 GC。Python 侧走的是第三条路，`python/litert_lm/engine.py:126 @ v0.13.1` 把 `close()`、`__del__`、`__exit__` 三者同时挂上——`with` 语句退出即释放，`del` 或回收时兜底，析构在 `close()` 里判空防重入。

坑的根源，正是第二节那个"谁创建谁释放"没有落到确定的时机上：C ABI 那边 delete 必须被调，但早期 Swift 把它独家拖给了 ARC 说了算的 `deinit`。issue 里提的解法也顺理成章（`LiteRT-LM#2613`，【文档】级）——给 Swift 加一个公开的 `close()`，让使用者能主动释放会话，不必干等 ARC。这恰好对齐了 Kotlin 的 `AutoCloseable.close()` 与 Python 的 `__exit__`：三种语言最终都得给出一条"人说了算"的释放路径，光靠语言自带的自动回收兜不住"单会话"这种带独占语义的资源。

这个案例的价值不在某个 API，而在它揭示的一般道理：**跨语言桥最难的往往不是"调得通"，而是两边的资源模型如何对齐。** C 的手动配对、Swift 的 ARC、Kotlin 的 AutoCloseable——把它们缝到一起而不漏，是绑定层最容易出 bug 的地方。

## 让核心可测、可构建

最后两笔工程纪律，一笔关于测试，一笔关于构建。

**可测试性**。推理要跑真模型、要硬件，测起来又慢又不稳定。LiteRT-LM 的对策是一个假执行器 `FakeLlmExecutor`，`runtime/executor/fake_llm_executor.h:37 @ v0.13.1`：

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

(1) 它继承 `LlmExecutor`——就是第 4、5 章那层执行器抽象，上层 pipeline 只认这个接口。(2) 构造函数收两组"脚本"：`prefill_tokens_set` 是每次 prefill *应该*收到的 token，`decode_tokens_set` 是每次 decode *将要吐出*的 token。它不加载模型、不碰硬件，全凭这两张表演戏。

戏怎么演，看 `Prefill` 的实现，`runtime/executor/fake_llm_executor.cc:128 @ v0.13.1`：

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

这段把 fake 的双重身份讲清了。它不只是"返回假数据"的桩，还是个**断言器**：(2) 用 `CheckEquivalent` 把上层这次真正喂进来的 token，和脚本里第 `prefill_times_` 条预期逐一比对，对不上就返回 `InvalidArgumentError`，于是测试不但能验证"上层拿到了什么"，还能验证"上层喂进来的是不是对的"。(1) 调用次数超出脚本长度直接报错，(3) 每调一次把游标 `prefill_times_` 往前推，让"第几次调用"对上"脚本第几条"。decode 侧对称：`decode_tokens_set_[decode_times_]` 按游标取出下一批 token 返回。整套逻辑没有一行涉及神经网络——prefill/decode 的编排、采样、停止条件，全都能在这张脚本上脱离真实模型和硬件跑单测，且每次结果完全确定。这是"接口隔离"的直接红利：上层只依赖 `LlmExecutor`，就能把真执行器整个换成这台脚本机。

**可构建**。这套代码要在 Android、iOS、Linux、macOS、Windows、Web 上都编得出来，还牵着一堆第三方依赖（sentencepiece、llguidance、skia……）。它用两套构建系统兜住：主用 Bazel，另备一套 CMake 供嵌入式或不便用 Bazel 的场景。这部分是纯工程的活，本书不展开（细节见附录 C），但值得记住一点：一个能投产到六个平台的运行时，构建系统的分量不亚于运行时本身。

## 小结

一套 C++ 核心服务六种语言，靠的是一层收敛后的 C ABI：不透明句柄藏住 C++ 类型，纯 C 函数当所有绑定的公约数，各绑定再用 ctypes、JNI、C 互操作、WASM 各自接上。最难的不是接通，而是让两边的资源模型对齐——那个 Swift ARC 撞"单会话"的坑就是明证。底下再有 FakeLlmExecutor 撑起可测试性、双构建系统撑起跨平台，这套运行时才算真的"能投产"。

第四部到此结束。从三堵墙到 token 的一生，再到多模态、工具调用与六种语言的绑定——LiteRT-LM 这台端侧推理机器，四个部分讲完了。剩下的尾声，聊聊你可以拿它做什么，以及这条路往前还通向哪里。

---

## 参考

- C ABI 声明：`c/engine.h @ v0.13.1`（不透明句柄 `LiteRtLmEngine`:41、`LiteRtLmSession`:44；`litert_lm_engine_create`:380、`_delete`:386；`create_session`:396、`session_delete`:403；`run_prefill`:421）。
- C ABI 实现：`c/engine.cc @ v0.13.1`（不透明句柄真身 `struct LiteRtLmEngine`:176；`litert_lm_engine_create`/`_delete` 实现:537、:556）。
- 各语言绑定：`python/litert_lm/_ffi.py @ v0.13.1`（`c_string_p`:24；`LiteRtLmSamplerParams`:36；`engine_create`/`_delete` 的 `restype`/`argtypes`:170）；`python/litert_lm/engine.py:126 @ v0.13.1`（`close`/`__del__`/`__exit__`）；`kotlin/.../LiteRtLmJni.kt @ v0.13.1`（`object LiteRtLmJni`:19；`external fun nativeCreateEngine`:52，返回 `Long`）；`kotlin/.../Engine.kt:36 @ v0.13.1`（`AutoCloseable`；`close`:98）；`swift/Engine.swift @ v0.13.1`（`import CLiteRTLM`:17；`public actor Engine`:28；`handle: OpaquePointer?`:38）；`swift/Conversation.swift:65 @ v0.13.1`（`deinit` 释放）；`js/packages/core/src/engine.ts @ v0.13.1`（Embind `.delete()`）。
- 生命周期案例：上游 `LiteRT-LM#2589`（单会话约束）、`#2613`（Swift `close()`），【文档】级。
- 可测试性：`runtime/executor/fake_llm_executor.h:37 @ v0.13.1`；`fake_llm_executor.cc:128 @ v0.13.1`（`Prefill` 脚本比对）。构建：见附录 C。

<!-- #2589/#2613 为 open issue，作缺陷案例研究、按【文档】级引，不宣称已修复。实测（Python 与 C++ 行为一致、给 FakeLlmExecutor 写新用例）待基准 D/环境。表 11-1(各语言 FFI 机制) 规格见 notes.md，本轮出签名图 11-1。 -->
