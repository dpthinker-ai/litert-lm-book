# 第 4 章 Prefill：吞下提示词

> 使命：理解 prefill 为什么快、它的两条实现路径各自在权衡什么，以及支撑"生成中途能立刻取消"的那层异步底座。

上一章，你的输入变成了一串 token id。现在 `RunPrefill` 接过它，把它一口吞进模型。这一步决定了"第一个字多久出来"。

## prefill 快在哪

回到第 2 章的 Roofline 眼镜。prefill 一次处理一整段提示词——几百上千个 token 同时进模型。这些 token 共用同一批权重：权重读进来一次，服务了这一整批。算术强度高，落在算力受限区。所以 prefill 拼的是算力，能并行，相对快。

这跟下一章 decode 的境况恰好相反（decode 每步只服务一个 token，贴着带宽墙走）。同一个模型，prefill 每秒几千 token、decode 每秒几十，两个数量级的差距，根子就在这里。

在代码里，这一步的编排入口是 `Prefill`（`runtime/core/tasks.cc:413 @ v0.13.1`）。它做的事很直白：校验输入不超长，把 token 交给执行器（executor），executor 调 LiteRT 的模型把这段 token 一次性算完，顺带把它们的注意力中间结果写进 KV cache——为 decode 铺好底。

## 一道选择题：固定长度还是可变长度

executor 怎么"吞"这段 token，是一道端侧特有的选择题。

模型编译成能在硬件上跑的形式后，它接受的输入形状（多长的序列）往往是**固定**的。可提示词长度千变万化，有时 20 个 token，有时 2000 个。怎么办？LiteRT-LM 给了两条路径，对应两种模型。

**静态形状**：预先编译好若干个固定长度的 prefill 入口（signature），比如 128、512、1024 各一个。来了一段输入，就挑一个最合适的入口。代码里这组入口按长度排好序存着（`SortedPrefillSignatureMap`，`runtime/executor/llm_litert_compiled_model_executor.h:415 @ v0.13.1`）。这是移动端和 GPU 上的常见选择——固定形状让编译器能把 kernel 尽力优化，代价是要预编译多个入口、以及输入长度对不齐时的一点浪费（用 512 的入口跑 300 个 token，剩下的要填充）。

**动态形状**：输入长度可变，KV cache 也按需增长。长提示词被切成固定大小的块，一块一块 prefill（`prefill_chunk_size`）。这更灵活，桌面和服务端常见，代价是失去了一部分固定形状带来的编译期优化。

两条路径都通过同一个内部函数落地（`PrefillInternal`，`llm_litert_compiled_model_executor.h:247 @ v0.13.1`）。这又是一次"接口隔离"：上层的 `Prefill` 不关心底下是静态还是动态，选路径的逻辑藏在 executor 里。

> 对照视野
> "预编译多个固定长度入口"这个取舍在端侧很典型：宁可多占一点编译产物和填充浪费，换取运行时的确定性与峰值性能。云端更倾向动态形状（灵活、省显存），因为它不缺重新编译的算力，也不在乎多留几个 kernel。同一个问题，两端因约束不同给出相反的默认答案——这类"因地制宜"贯穿全书。

<figure>

{{#include figs/fig-4-1.svg}}

<figcaption>图 4-1　prefill 的两条路径与异步底座。静态路径按长度挑固定 signature，动态路径分块吞入；两者都经 PrefillInternal 落到 LiteRT，并把结果写进 KV cache。任务经队列异步执行，取消标志随时可打断。</figcaption>
</figure>

## 生成中途，为什么能立刻停

第 2 章那份"二十个问题"里有一问：生成中途取消，为什么能立刻停下（第 9 问）？答案的一半在 prefill 这一层。

长提示词的 prefill 可能要跑好一阵。如果这期间用户按了取消，系统不能傻等它跑完。LiteRT-LM 的办法是两手准备，都藏在 prefill 的参数里（`ExecutorPrefillParams`）：

- 一个**取消标志**：调用方传进来一个原子布尔量的指针，executor 在推进过程中会瞄它。一旦被置真，尽快收手。
- 一个**限长开关**（`max_prefill_sequence_length`）：限制单次调用能用多长的 prefill signature。为什么？因为一次跑太长的 signature 就是一段不可打断的时间。把单次长度限住，取消标志才有足够密的机会被检查到——否则你按了取消，还得等那一大段算完。

这两个设计放在一起，才凑出"立刻停"的手感。它背后还站着一层异步底座：prefill 任务并不总在主线程同步跑，而是可以提交到一个执行队列、由工作线程处理（第 8 章会展开这层线程模型）。任务之间还连着依赖链，保证 decode 一定在它依赖的 prefill 完成之后才开始。异步、可取消、限长三者合起来，端侧才有"响应跟手"的体感。

## 小结

prefill 是算力受限的一步，快，且有静态/动态两条实现路径应对端侧固定形状的约束。它也不是一个"跑起来就没法管"的黑盒：取消标志和限长开关使它随时可被打断，异步底座把它挪出主线程。

提示词已经吞进去了，KV cache 也填好了第一段。下一章，最核心的一步：decode 循环，逐字的心跳。

---

## 参考

- prefill 编排：`runtime/core/tasks.cc:413 @ v0.13.1`（`Prefill`）。
- 静态/动态路径与内部实现：`runtime/executor/llm_litert_compiled_model_executor.h @ v0.13.1`（`PrefillInternal`:247；`SortedPrefillSignatureMap`:415）。
- 取消与限长：`ExecutorPrefillParams`（`runtime/executor/llm_executor_io_types.h @ v0.13.1`，精确行号待补读核验）。

<!-- 缺口：ExecutorPrefillParams 与异步任务队列的精确行号待第 4 章补读（tasks.cc / execution_queue.cc）核验后回填。 -->
