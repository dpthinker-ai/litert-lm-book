# 第 6 章 KV cache 与会话状态

> 使命：把 KV cache 算清楚——它有多大、为什么 decode 离不开它、又为什么它拖慢了 decode；再看 LiteRT-LM 如何把这份状态做成一个能搬运的对象，换来会话克隆、检查点、回退这些能力。

第 5 章末尾留了个账没算：那条 25 tokens/s 的纯权重上限，实测为什么还够不着？欠的这笔，就欠在 KV cache 上。这一章把它补齐，顺带解开第 2 章"二十个问题"里的第 13、14 问。

## 先算一笔账：KV cache 有多大

先说它是什么。注意力机制里，每生成一个新 token，都要让它去"看"前面所有 token——具体说，是拿它的 query 去和前面每个 token 的 key、value 做运算。如果不缓存，第 1000 步就要把前 999 个 token 的 key/value 全部重算一遍，第 1001 步再重算一遍……这是第 1 章带宽墙的最坏放大。

KV cache 就是把这些 key/value 存下来，算过一次就不再重算。**用内存，换掉重复计算。**

它有多大？一个可以写下来的公式：

$$ \text{KV 字节} = 2 \times L \times H_{kv} \times D \times S \times b $$

其中 $L$ 是层数，$H_{kv} \times D$ 是每层 KV 的投影维度，$S$ 是序列长度（缓存了多少个 token），$b$ 是每个元素的字节数，最前面的 2 是 key 和 value 各一份。

代入一组示例量级（层数 30、KV 投影维度 1024、fp16 即 $b=2$）：每个 token 约

$$ 2 \times 30 \times 1024 \times 2 = 122880 \text{ 字节} \approx 120 \text{ KiB/token} $$

到 4096 个 token 的上下文，就是约 480 MiB。（这里的 $L$、$H_{kv}$、$D$ 是示例；某个具体模型的真实数字可以从它的元数据读出，第 7 章的 `litertlm_print` 会带你看。）

把这个数字接回第 1 章的内存墙：权重约 2 GiB，KV cache 又要几百 MiB，而且它随对话变长一直涨。这解释了第 2 问——为什么 8 GiB 的手机跑 2B 模型也紧张：吃内存的不止权重。

## 它也占带宽：补上第 5 章欠的账

现在补第 5 章那笔账。decode 每一步，除了把 2 GiB 权重读一遍（第 1 章的账），**还要读一遍到目前为止的整个 KV cache，并把新 token 的 key/value 追加进去。** 上下文越长，这部分越重。

于是那条 25 tokens/s 的纯权重上限，只是"权重"这一项的账；实测更慢，慢的那部分正是被 KV cache 的读写吃掉的。上下文短时差距小，上下文长到几千 token 时，KV cache 的带宽开销变得不可忽略——decode 会随对话变长而变慢。本书基准里这条曲线清楚可见：上下文从 256 拉到 4096，decode 从 24.8 掉到 20.7 tok/s（cpu，-17%），gpu 也从 50.6 掉到 45.6〔基准 D〕。这不是实现的锅，是自回归 + 注意力的固有代价。

这也顺带解释了第 13 问，一个真实的上游现象（`LiteRT-LM#2568`）：`--max-num-tokens` 这个参数为什么会影响解码速度？因为它设定的是 KV cache 能容纳多少 token——也就是要预留多大的一块缓存。设得越大，预留和管理的内存越多，触及的读写与分配代价也不同。所以它不是越大越好：设定值直接决定预留的内存，以及随之而来的读写量。

## 双缓冲：一个朴素的工程手艺

KV cache 要频繁读写，这在 GPU 上撞见一个具体麻烦：部分 GPU 后端不允许对同一块缓冲同时读和写（源码注释就是这么写的，`runtime/executor/llm_litert_compiled_model_executor.h:327 @ v0.13.1`）。

LiteRT-LM 的解法不玄乎：**备两套缓冲**（`kv_cache_buffers_1_` 和 `kv_cache_buffers_2_`，`llm_litert_compiled_model_executor.h:329-330 @ v0.13.1`），再用两个指针分别指向"当前读的"和"当前写的"（`input_kv_cache_buffers_` / `output_kv_cache_buffers_`，`:331-333`）。每一步，从旧缓冲读、往新缓冲写，然后把两个指针一交换——下一步的"新"就成了"旧"。全程不需要把数据从一块缓冲拷到另一块，只是换个指针指向。

<figure>

{{#include figs/fig-6-1.svg}}

<figcaption>图 6-1　KV cache 双缓冲。每步读旧缓冲、写新缓冲，然后交换两个指针——用一次指针交换避开了"同缓冲读写"的限制，也避开了数据拷贝。</figcaption>
</figure>

这个手艺没有高深算法，但它是端侧工程的典型样子：一个来自硬件的具体约束（GPU 不能同缓冲读写），一个来自成本的具体考量（不想拷贝几百 MiB），凑出一个朴素解法（两套缓冲 + 指针交换）。全书这类"约束逼出手艺"的例子会一再出现。

## 状态即对象

到这里，一次会话的核心状态已经清楚了：一堆 KV cache，加上"当前到第几步"这样的元信息。LiteRT-LM 把它们打包成一个对象——`LlmContext`（`runtime/executor/llm_executor_io_types.h:92 @ v0.13.1`），里面装着已处理的 token（含 KV cache）、运行配置、以及记录 `current_step` 的运行状态（`RuntimeState`，`:78`）。

把状态做成一个能整体搬运的对象，是第 2 章那条"状态即对象"原则的兑现。它一旦成立，几件原本很难的事就顺理成章：

- **克隆会话**（`Clone`，`runtime/engine/engine.h:245 @ v0.13.1`；还有异步版 `CloneAsync`，`:263`）：复制这个对象，就得到一个独立的会话分支。
- **存检查点、回退**（`SaveCheckpoint`，`engine.h:270 @ v0.13.1`；配套的 `RewindToCheckpoint`）：给当前状态打个标记，之后能退回来。

克隆的价值，第 3 章其实已经埋过：头文件里那个例子——session1 先 prefill 一段公共前缀，Clone 出 session2，两个分支各接不同的后续，公共前缀只算一次（`engine.h:238 @ v0.13.1`）。这就是第 14 问的答案：克隆分叉不必重算公共前缀，因为那段 KV cache 被整个复制了过去，而不是重新 prefill。

## KV cache 的搬运接口

克隆、回退这些能力，底层要求 KV cache 本身能被搬运。所以它有一个专门的接口 `KVCacheInterface`（`runtime/executor/kv_cache_interface.h:28 @ v0.13.1`），几个方法各有用途：

- `Serialize` / `Load`（`:39` / `:42`）：把 KV cache 存成字节串、再读回来——持久化的基础；
- `DeepCopy`（`:61`）：深拷贝一份。注释直言这是"昂贵操作，慎用"——几百 MiB 的深拷贝不便宜，克隆的代价就在这里；
- `SelectAndCopyFrom`（`:50`）：从一个多分支的 KV cache 里挑一条出来。这在需要从多个候选里收敛到一个时用得上；
- `BroadcastAndCopyFrom`（`:58`）：反过来，把一条广播成多份。

这几个方法凑齐了 KV cache 的"搬运工具箱"。它们就是 Clone 和 SaveCheckpoint 底下真正在搬数据的那一层。

## 顺带一手：把思考从缓存里择出去

最后一个应用，把前面的机制串起来。有些模型会先"想"再答，把思考过程也吐出来（第 10 章的 channel 机制会细讲）。思考内容对用户不必展示，更重要的是——它不该占着 KV cache，否则接下来每一步 decode 都要连着这段思考一起读，白白付带宽（第 2 节的账）。

LiteRT-LM 的处理是：把思考这段 channel 内容从 KV cache 里"择"出去。做法正是回退——退到思考开始前的那个位置，丢掉这段的 KV cache。前面搭的 `RewindToCheckpoint` 到这里派上了用场：一个为"回退"造的机制，顺手也解决了"别让思考污染上下文"的问题。

## 小结

KV cache 是"用内存换计算"的经典权衡：它省掉了重复的注意力计算，代价是吃内存、也吃带宽（decode 变慢的一部分来自它）。围绕它，LiteRT-LM 做了两件事——一个朴素的双缓冲手艺应对 GPU 的读写约束，一套"状态即对象"的设计换来克隆、检查点、回退，还顺手解决了思考污染上下文的问题。

下一章，我们去看这份账的另一半：模型本身怎么被压小、装箱、变体——量化、`.litertlm` 格式与 LoRA。

---

## 参考

- KV cache 接口：`runtime/executor/kv_cache_interface.h @ v0.13.1`（`KVCacheInterface`:28；`Serialize`:39；`Load`:42；`SelectAndCopyFrom`:50；`BroadcastAndCopyFrom`:58；`DeepCopy`:61）。
- 双缓冲：`runtime/executor/llm_litert_compiled_model_executor.h @ v0.13.1`（注释:327；`kv_cache_buffers_1_/2_`:329-330；读写指针:331-333）。
- 会话状态：`runtime/engine/engine.h @ v0.13.1`（`Clone`:245；`CloneAsync`:263；`SaveCheckpoint`:270）；`runtime/executor/llm_executor_io_types.h @ v0.13.1`（`RuntimeState`:78；`LlmContext`:92）。

<!-- 实验（--max-num-tokens 扫描解释 #2568、Clone 分叉、get_token_count 增长）数字待基准 D 回填〔基准 D〕。KV cache 公式为示例量级；具体模型 L/H_kv/D 待第 7 章 litertlm_print 读出后可补精确值。图 6-2(增长)、图 6-3(状态分叉) 与表 6-1(内存账) 规格见 notes.md，本轮先出签名图 6-1(双缓冲)。 -->
