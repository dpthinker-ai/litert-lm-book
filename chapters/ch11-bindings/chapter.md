# 第 11 章 一套核心，六种语言：C ABI、绑定与工程纪律

> 使命：一套 C++ 核心，怎么变成 Python、Kotlin、Swift、Web 都能调的 SDK。看清那层作为"通用桥"的 C ABI，以及一个真实的跨语言生命周期坑；再顺带看这套代码是怎么保证自己可测、可构建的。

前面十章讲的都是 C++ 核心里发生了什么。但真正用它的人，多半不写 C++——App 开发者用 Kotlin、Swift，脚本作者用 Python，前端用 JavaScript。一套核心怎么服务这么多语言？这是最后一块工程拼图。

## 为什么中间要隔一层 C

直觉上，既然核心是 C++，各语言直接调 C++ 不就行了？不行。C++ 的名字修饰（name mangling）各家编译器不同，异常、模板、对象内存布局都不跨语言稳定——你没法可靠地从 Python 直接调一个 C++ 类的方法。

通行的解法是退一步：先把核心收敛成一层纯 **C 接口**。几乎每种语言都能调 C（这是几十年攒下的事实标准），所以只要有一层 C ABI，就等于给所有语言开了门。LiteRT-LM 的这层门在 `c/engine.h`。

它的写法有两个要点。其一，**不透明句柄**（opaque handle）。C 接口不暴露任何 C++ 类型，只给你一个不透明的指针——`LiteRtLmEngine`、`LiteRtLmSession`（`c/engine.h:41`、`:44 @ v0.13.1`）都是"只知道有这么个东西、不知道里面长啥样"的结构体指针。你拿着它调函数，但碰不到它的内部。其二，**纯 C 函数**。所有操作都是普通 C 函数：`litert_lm_engine_create`（`:380`）造一个引擎、`litert_lm_engine_create_session`（`:396`）开一个会话、`litert_lm_session_run_prefill`（`:421`）跑 prefill——你会发现这些函数名，正是第 3 章那套 Engine/Session 接口的 C 语言镜像。

<figure>

{{#include figs/fig-11-1.svg}}

<figcaption>图 11-1　一套核心，六种语言：C++ 核心先收敛成一层 C ABI（不透明句柄 + 纯 C 函数），各语言再用各自的 FFI 机制（ctypes / JNI / C 互操作 / WASM）接上去。所有语言最终都进入同一套 runtime。</figcaption>
</figure>

## 谁创建，谁释放

不透明句柄带来一个必须讲清的问题：内存谁来管。

C 没有析构函数，也没有垃圾回收。所以 C ABI 的规矩是**创建和释放成对**：`litert_lm_engine_create` 造出来的引擎，得由 `litert_lm_engine_delete`（`c/engine.h:386 @ v0.13.1`）释放；session 有 `litert_lm_session_delete`（`:403`）。谁调了 create，谁就负责调对应的 delete——这是跨越 FFI 边界传递 C++ 对象的标准做法，也是各语言绑定必须小心接住的一件事。

各语言怎么接？各有各的惯用形态：Python 用上下文管理器或 `__del__`，Kotlin 用 `AutoCloseable`，Swift 用 `deinit`。绑定层的活，很大一部分就是把"手动配对的 create/delete"包装成本语言里自然的资源管理。下一节会看到，这件事一旦没接好，会出真问题。

## 各语言的 FFI，各有各的接法

有了 C ABI，剩下的是每种语言用自己的方式接上去。机制不同，目标一致——都是调到那层 C 函数。

- **Python** 用 `ctypes`：运行时按签名声明 C 结构和函数，直接调共享库（`python/litert_lm/_ffi.py:36 @ v0.13.1` 声明了 `LiteRtLmSamplerParams` 这样的 `ctypes.Structure`；还有个小巧的 `c_string_p`，自动把 Python 字符串编码成 UTF-8 字节，`:24`）。
- **Kotlin/Android** 用 JNI：声明一串 `external fun`，由 JNI 桥到原生库（`kotlin/.../LiteRtLmJni.kt:19 @ v0.13.1` 那个 `LiteRtLmJni` object，里面是 `nativeCreateEngine` 等外部函数，`:52`）。句柄以 `Long`（一个指针大小的整数）在 Kotlin 和原生之间传递。
- **Swift/iOS & macOS** 用 C 互操作：直接 `import` C 头文件调用（`swift/Engine.swift:17` 的 `import CLiteRTLM`）。它的 `Engine` 是一个 `actor`（`:28`）——用 Swift 的并发原语保证对原生引擎的访问是串行、安全的，而不用手写锁。
- **Web** 把核心编成 WebAssembly，用 TypeScript 包一层在浏览器里跑。

同一个 prompt，走 Python 和走 C++ 会得到一致的行为——因为它们最终进的是同一套 `runtime`（这也是本章开头那句"一套核心"的实感）。各语言的 SDK 看起来风格迥异，底下是同一个引擎。

## 一个真实的坑：ARC 与"单会话"撞车

绑定层最容易栽的地方，是生命周期。这里有一个真实的上游案例，正好把前面几节串起来（对应上游 issue，`LiteRT-LM#2589` 与 `#2613`，【文档】级）。

背景是两个约束的相遇。其一，LiteRT-LM 一个引擎同时只允许一个会话。其二，Swift 用 ARC（自动引用计数）管内存，对象什么时候被销毁，取决于引用何时归零，时机是**不确定**的。早期 Swift 绑定里，原生会话只在 `deinit` 里释放。于是问题来了：你逻辑上"用完"了一个会话，想再开一个新的，但旧会话的 Swift 对象还没被 ARC 回收、`deinit` 还没跑、原生会话还占着——新会话创建就撞上"已有一个会话"的错误。

坑的根源，正是第二节那个"谁创建谁释放"没有落到确定的时机上：C ABI 那边 `delete` 必须被调，但 Swift 把它拖到了 ARC 说了算的 `deinit`。issue 里提的解法也顺理成章——给 Swift 加一个公开的 `close()`，让使用者能**主动、确定地**释放会话，不必干等 ARC。这恰好对齐了 Kotlin 早已有的 `AutoCloseable.close()`。

这个案例的价值不在某个 API，而在它揭示的一般道理：**跨语言桥最难的往往不是"调得通"，而是两边的资源模型如何对齐。** C 的手动配对、Swift 的 ARC、Kotlin 的 AutoCloseable——把它们缝到一起而不漏，是绑定层真正的工程含量所在。

## 让核心可测、可构建

最后两笔工程纪律，一笔关于测试，一笔关于构建。

**可测试性**。推理要跑真模型、要硬件，测起来又慢又不稳定。LiteRT-LM 的对策是一个假执行器 `FakeLlmExecutor`（`runtime/executor/fake_llm_executor.h:37 @ v0.13.1`）——它实现同一个 `LlmExecutor` 接口（第 5 章），但不跑模型，而是按预先写好的脚本返回 token。有了它，上层的 prefill/decode 编排、采样、停止逻辑，全都能脱离真实模型和硬件来单测。这又是"接口隔离"的红利：因为上层只依赖 `LlmExecutor` 抽象，就能把真执行器换成假的来测。

**可构建**。这套代码要在 Android、iOS、Linux、macOS、Windows、Web 上都编得出来，还牵着一堆第三方依赖（sentencepiece、llguidance、skia……）。它用两套构建系统兜住：主用 Bazel，另备一套 CMake 供嵌入式或不便用 Bazel 的场景。这部分是纯工程的活，本书不展开（细节见附录 C），但值得记住一点：一个能投产到六个平台的运行时，构建系统的分量不亚于运行时本身。

## 小结

一套 C++ 核心服务六种语言，靠的是一层收敛后的 C ABI：不透明句柄藏住 C++ 类型，纯 C 函数当所有语言的公约数，各语言再用 ctypes、JNI、C 互操作、WASM 各自接上。最难的不是接通，而是让两边的资源模型对齐——那个 Swift ARC 撞"单会话"的坑就是明证。底下再有 FakeLlmExecutor 撑起可测试性、双构建系统撑起跨平台，这套运行时才算真的"能投产"。

第四部到此结束。至此，从三堵墙到一个 token 的一生，从凿墙的各种手艺到多模态、工具调用，再到六种语言的绑定——LiteRT-LM 这台端侧推理机器，我们从里到外拆了一遍。剩下的尾声，聊聊你可以拿它做什么，以及这条路往前还通向哪里。

---

## 参考

- C ABI：`c/engine.h @ v0.13.1`（不透明句柄 `LiteRtLmEngine`:41、`LiteRtLmSession`:44；`litert_lm_engine_create`:380、`_delete`:386；`create_session`:396、`session_delete`:403；`run_prefill`:421）。
- 各语言绑定：`python/litert_lm/_ffi.py @ v0.13.1`（`ctypes`；`c_string_p`:24；`LiteRtLmSamplerParams`:36）；`kotlin/.../LiteRtLmJni.kt:19 @ v0.13.1`（`external fun`:52）；`swift/Engine.swift @ v0.13.1`（`import CLiteRTLM`:17；`public actor Engine`:28）。
- 生命周期案例：上游 `LiteRT-LM#2589`、`#2613`（Swift `close()`），【文档】级。
- 可测试性：`runtime/executor/fake_llm_executor.h:37 @ v0.13.1`。构建：见附录 C。

<!-- #2589/#2613 为 open issue，作缺陷案例研究、按【文档】级引，不宣称已修复。实测（Python 与 C++ 行为一致、给 FakeLlmExecutor 写新用例）待基准 D/环境。表 11-1(各语言 FFI 机制) 规格见 notes.md，本轮出签名图 11-1。 -->
