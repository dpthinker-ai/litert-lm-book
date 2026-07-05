# 第 6 章 KV cache 与会话状态

> 使命：把 KV cache 算清楚——它有多大、为什么 decode 离不开它、又为什么它拖慢了 decode；再看 LiteRT-LM 如何把这份状态做成一个能搬运的对象，换来会话克隆、检查点、回退这些能力。

第 5 章末尾留了个账没算：那条 25 tokens/s 的纯权重上限，实测为什么还够不着？欠的这笔，就欠在 KV cache 上。这一章把它补齐，顺带解开第 2 章"二十个问题"里的第 13、14 问。

## 先算一笔账：KV cache 有多大

先说它是什么。注意力机制里，每生成一个新 token，都要让它去"看"前面所有 token——具体说，是拿它的 query 去和前面每个 token 的 key、value 做运算。如果不缓存，第 1000 步就要把前 999 个 token 的 key/value 全部重算一遍，第 1001 步再重算一遍……这是第 1 章带宽墙的最坏放大。

KV cache 就是把这些 key/value 存下来，算过一次就不再重算。**用内存，换掉重复计算。**

它有多大？一个可以写下来的公式：

$$ \text{KV 字节} = 2 \times L \times H_{kv} \times D \times S \times b $$

其中 L 是层数，H_kv × D 是每层 KV 的投影维度，S 是序列长度（缓存了多少个 token），b 是每个元素的字节数，最前面的 2 是 key 和 value 各一份。

代入一组示例量级（层数 30、KV 投影维度 1024、fp16 即每元素 2 字节）：每个 token 约

$$ 2 \times 30 \times 1024 \times 2 = 122880 \text{ 字节} \approx 120 \text{ KiB/token} $$

到 4096 个 token 的上下文，就是约 480 MiB。（这里的 L、H_kv、D 都是示例；某个具体模型的真实数字可以从它的元数据读出，第 7 章的 `litertlm_print` 会带你看。）

把这个数字接回第 1 章的内存墙：权重约 2 GiB，KV cache 又要几百 MiB，而且它随对话变长一直涨。这解释了第 2 问——为什么 8 GiB 的手机跑 4B 模型也紧张：吃内存的不止权重。

## 它也占带宽：补上第 5 章欠的账

现在补第 5 章那笔账。decode 每一步，除了把 2 GiB 权重读一遍（第 1 章的账），**还要读一遍到目前为止的整个 KV cache，并把新 token 的 key/value 追加进去。** 上下文越长，这部分越重。

于是那条 25 tokens/s 的纯权重上限，只是"权重"这一项的账；实测更慢，慢的那部分有很大一头是被 KV cache 的读写吃掉的（还有采样、非注意力算子等零头）。顺带防一个误读：本机 cpu 实测 24.8 与那个"25"几乎相等纯属巧合：上限要按各自机器的带宽重算，对账见第 2 章。上下文短时差距小，上下文长到几千 token 时，KV cache 的带宽开销变得不可忽略——decode 会随对话变长而变慢。本书基准里这条曲线清楚可见：上下文从 256 拉到 4096，decode 从 24.8 掉到 20.7 tok/s（cpu，-17%），gpu 也从 50.6 掉到 45.6〔基准 D〕。这不是实现的锅，是自回归 + 注意力的固有代价。

这也顺带解释了第 13 问，一个真实的上游现象（`LiteRT-LM#2568`）：`--max-num-tokens` 这个参数为什么会影响解码速度？因为它设定的是 KV cache 能容纳多少 token——也就是要预留多大的一块缓存。在固定形状的路径上（第 4 章），decode 的注意力按整个预留长度计算，没用到的位置靠 mask 屏蔽，但读写并不因此免单。所以它不是越大越好：预留多大，每步就要为多大的缓存付账。

## 双缓冲：一个朴素的工程手艺

KV cache 要频繁读写，这在 GPU 上撞见一个具体麻烦：部分 GPU 后端不允许对同一块缓冲同时读和写。这不是猜的，源码注释直接写在成员声明上方（`runtime/executor/llm_litert_compiled_model_executor.h:327-333 @ v0.13.1`）：

```cpp
// KV cache double buffers because some GPU backends can't allocate one buffer
// for both read and write at the same time.
absl::flat_hash_map<absl::string_view, TensorBuffer> kv_cache_buffers_1_;      // (1)
absl::flat_hash_map<absl::string_view, TensorBuffer> kv_cache_buffers_2_;      // (2)
absl::flat_hash_map<absl::string_view, TensorBuffer>* input_kv_cache_buffers_;  // (3)
absl::flat_hash_map<absl::string_view, TensorBuffer>*
    output_kv_cache_buffers_;
```

(1)(2) 是两套实打实的缓冲，各持一份 KV。(3) 起的两个指针才是关键：它们不拥有数据，只指向那两套缓冲之一——`input_kv_cache_buffers_` 指向"这一步读哪套"，`output_kv_cache_buffers_` 指向"这一步写哪套"。构造时（`:199-200`）两者分别指向 `kv_cache_buffers_1_` 和 `kv_cache_buffers_2_`，一读一写、错开。

交换动作发生在每次跑完模型之后。prefill 的路径上（`llm_litert_compiled_model_executor.cc:738 @ v0.13.1`）：

```cpp
if (!gpu_optimized_single_buffer_cache_) {   // (1)
  std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);   // (2)
}
```

(2) 只交换两个指针的指向，不动缓冲里的字节——这一步刚往 `output` 那套写了新 token 的 key/value，交换后它就变成下一步的 `input`，上一步读过的旧缓冲则腾出来当下一步的写入目标。几百 MiB 的 KV 数据一次也没搬。(1) 是个例外闸门：`gpu_optimized_single_buffer_cache_` 为真时（某些 GPU 后端反而支持同缓冲读写）跳过交换，两个指针始终指同一套缓冲（`:1960-1961` 就是这么初始化的），省掉第二套的内存。同一份 `std::swap` 在 decode 路径也各来一次（`:947`），逻辑一致。

<figure>
{{#include figs/fig-6-1.svg}}
<figcaption>图 6-1　KV cache 双缓冲。每步读旧缓冲、写新缓冲，然后交换两个指针——用一次指针交换避开了"同缓冲读写"的限制，也避开了数据拷贝。</figcaption>
</figure>

这个手艺没有高深算法，但它是端侧工程的典型样子：一个来自硬件的具体约束（GPU 不能同缓冲读写），一个来自成本的具体考量（不想拷贝几百 MiB），凑出一个朴素解法（两套缓冲 + 指针交换）。全书这类"约束逼出手艺"的例子会一再出现。

## 状态即对象

到这里，一次会话的核心状态已经清楚了：一堆 KV cache，加上"当前到第几步"这样的元信息。LiteRT-LM 把它们打包成一个对象——`LlmContext`（`runtime/executor/llm_executor_io_types.h:92 @ v0.13.1`）：

```cpp
struct RuntimeState {
  int current_step = 0;   // (1)
  std::shared_ptr<std::default_random_engine> rand_gen;
  bool ran_decode = false;
};

class LlmContext {
 public:
  // ...
  ProcessedContext& processed_context() { return *processed_context_; };   // (2)
  RuntimeConfig& runtime_config() { return *runtime_config_; };
  RuntimeState& runtime_state() { return *runtime_state_; };
 private:
  std::unique_ptr<ProcessedContext> processed_context_;   // (3)
  std::unique_ptr<RuntimeConfig> runtime_config_;
  std::unique_ptr<RuntimeState> runtime_state_;
};
```

三样东西各管一摊。(3) 的 `processed_context_` 装已处理的 token 和它们的 KV cache，KV 那几百 MiB 就在这下面。`runtime_config_` 是运行配置。(1) 的 `runtime_state_` 记运行状态，其中 `current_step` 就是"当前到第几步"这个游标；`RuntimeState` 上方那行源码注释特意声明它"不含直接与 KVCache 相关的状态"（`:75-76`），所以搬运时 KV cache 走 `processed_context_` 那一路，步数游标走 `runtime_state_` 这一路，两者分开。(2) 三个访问器返回引用，底下全用 `unique_ptr` 独占持有——想搬运整个上下文，只要把这三个 `unique_ptr` 一起打包。

把状态做成一个能整体搬运的对象，是第 2 章那条"状态即对象"原则的兑现。它一旦成立，几件原本很难的事就顺理成章：

- **克隆会话**（`Clone`，`runtime/engine/engine.h:245 @ v0.13.1`；还有异步版 `CloneAsync`，`:263`）：复制这个对象，就得到一个独立的会话分支。
- **存检查点、回退**（`SaveCheckpoint`，`engine.h:270 @ v0.13.1`；配套的 `RewindToCheckpoint`）：给当前状态打个标记，之后能退回来。

克隆的价值，头文件注释里有个现成的例子（`engine.h:238 @ v0.13.1`）：session1 先 `Prefill("What is the tallest building ")`，`Clone` 出 session2，之后 session1 接 `"in the world?"`、session2 接 `"in France?"`，那段公共前缀的 prefill 只跑一次。这就是第 14 问的答案：克隆分叉不必重算公共前缀，因为那段 KV cache 被整个复制了过去，而不是重新 prefill。

复制到底发生在哪一行？`Clone` 一路走到执行器的 `CloneContext`（`llm_litert_compiled_model_executor.cc:1288 @ v0.13.1`）：

```cpp
std::optional<uint32_t> lora_id;
ASSIGN_OR_RETURN(auto kv_cache_buffers, CloneKVCacheBuffers());   // (1)
ProcessedTokens new_processed_tokens =
    llm_context_->processed_context().processed_tokens();
auto new_processed_context = std::make_unique<LlmProcessedContext>(
    std::move(lora_id), std::move(kv_cache_buffers),
    std::move(new_processed_tokens));
auto new_runtime_config =
    std::make_unique<RuntimeConfig>(llm_context_->runtime_config());
auto new_runtime_state =
    std::make_unique<RuntimeState>(llm_context_->runtime_state());
return std::make_unique<LlmContext>(std::move(new_processed_context),
                                    std::move(new_runtime_config),
                                    std::move(new_runtime_state));   // (2)
```

(2) 就是上一节那个 `LlmContext` 的新实例，克隆的产物正是一个装齐三样东西的新上下文对象，"状态即对象"在这里收口。真正搬字节的是 (1) 的 `CloneKVCacheBuffers`（`:1243`），它逐个遍历 `input_kv_cache_buffers_`，对每块调 `CopyTensorBuffer` 复制一份。这就是 `DeepCopy` 注释里那句"昂贵操作"的具体来源：4096 token 上下文下按第 1 节的账约 480 MiB，克隆一次就要原样拷这么多字节——克隆的代价，全在这个循环里。

## KV cache 的搬运接口

克隆、回退这些能力，底层要求 KV cache 本身能被搬运。所以它有一个专门的接口 `KVCacheInterface`（`runtime/executor/kv_cache_interface.h:28 @ v0.13.1`），一组纯虚函数划定了"搬运工具箱"的边界：

```cpp
class KVCacheInterface {
 public:
  // ...
  virtual absl::StatusOr<std::string> Serialize() const = 0;               // (1)
  virtual absl::Status Load(absl::string_view serialized_kv_cache) = 0;    // (2)
  virtual absl::Status SelectAndCopyFrom(KVCacheInterface& other,
                                         int batch_index) = 0;             // (3)
  virtual absl::Status BroadcastAndCopyFrom(KVCacheInterface& other) = 0;  // (4)
  virtual absl::StatusOr<std::unique_ptr<KVCacheInterface>> DeepCopy()     // (5)
      const = 0;
};
```

(1)(2) 是持久化的一对：`Serialize` 把 KV cache 压成一个 `std::string` 字节串，`Load` 反向读回，存盘、跨进程传都靠这两个。(5) 的 `DeepCopy` 返回一个新的 `unique_ptr<KVCacheInterface>`，源码注释就写在这行上方：`This is an expensive operation. Use sparingly.`（`:60`）——上一节 `CloneKVCacheBuffers` 那个逐块拷贝的循环，正是这条注释在具体实现里的兑现。(3)(4) 是一对方向相反的批处理搬运，注释里各带一个 shape 例子：`SelectAndCopyFrom` 从 `[3, ...]` 的多分支缓存里挑第 `batch_index` 条拷到自己这个 `[1, ...]`；`BroadcastAndCopyFrom` 反过来，把 `[1, ...]` 的一条广播到 `[3, ...]` 的每一路。它们服务于并行采样这类"一进多出、多进一出"的场景。

注意这五个方法全是 `= 0` 的纯虚声明——`KVCacheInterface` 只定契约，不含一行实现。真正的字节搬运落在具体后端里，比如 `LitertKVCache::DeepCopy`（`runtime/executor/litert/kv_cache.cc:380 @ v0.13.1`）。接口与实现分离，是 LiteRT-LM 支持多后端（cpu/gpu/npu）的一贯手法，后面几章还会反复见到。

## 顺带一手：把思考从缓存里择出去

最后一个应用，把前面的机制串起来。有些模型会先"想"再答，把思考过程也吐出来（第 10 章的 channel 机制会细讲）。思考内容对用户不必展示，也不该占着 KV cache，否则接下来每一步 decode 都要连着这段思考一起读，白白付带宽（第 2 节的账）。

LiteRT-LM 的处理是：把思考这段 channel 内容从 KV cache 里"择"出去。做法正是回退——退到思考开始前的那个位置，丢掉这段的 KV cache。看 `RewindToCheckpoint` 是怎么退的（`runtime/core/session_advanced.cc:467 @ v0.13.1`）：

```cpp
int target_step = it->second.step;    // (1)
session_state_ = it->second.state;
absl::erase_if(checkpoint_map_, [target_step](const auto& pair) {   // (2)
  return pair.second.step > target_step;
});
// ...
return execution_manager_lock->SetCurrentStep(*session_info_, target_step);   // (3)
```

回退没有拷贝、没有删除任何 KV 字节。它做的是 (3)：把游标 `current_step` 调回 `target_step`，也就是上一节 `RuntimeState` 里那个整数。`SaveCheckpoint`（`:455`）存的也不过是 `{current_step, session_state_}` 这一对（`CheckpointInfo`，`session_advanced.h:257-260`），一个步数加一份会话状态引用，几十字节。之后继续 prefill 时，新 token 从 `target_step` 这个位置往后写，直接覆盖掉原来那段思考占的 KV 槽位——旧字节没被主动清除，而是被下一轮写入盖过。(2) 顺手把 `target_step` 之后的所有检查点从 map 里抹掉，让检查点集合始终是当前时间线的一条前缀，避免退回后残留一个指向"未来"的悬空标记。

一个只在 `RuntimeState` 上加减整数、连内存都不释放的回退，就顺手解决了"别让思考污染上下文"的问题。代价与克隆恰成对照：`Clone` 要深拷几百 MiB（上一节那个循环），`RewindToCheckpoint` 只动一个 `int`。

## 小结

KV cache 是"用内存换计算"的经典权衡：它省掉了重复的注意力计算，代价是吃内存、也吃带宽（decode 变慢的一部分来自它）。围绕它，LiteRT-LM 做了两件事——一个朴素的双缓冲手艺应对 GPU 的读写约束，一套"状态即对象"的设计换来克隆、检查点、回退，还顺手解决了思考污染上下文的问题。

下一章，我们去看这份账的另一半：模型本身怎么被压小、装箱、变体——量化、`.litertlm` 格式与 LoRA。

---

## 参考

- KV cache 接口：`runtime/executor/kv_cache_interface.h @ v0.13.1`（`KVCacheInterface`:28；`Serialize`:39；`Load`:42；`SelectAndCopyFrom`:50；`BroadcastAndCopyFrom`:58；`DeepCopy`:61，"expensive operation"注释:60）；具体实现 `LitertKVCache::DeepCopy`:`runtime/executor/litert/kv_cache.cc:380 @ v0.13.1`。
- 双缓冲：`runtime/executor/llm_litert_compiled_model_executor.h @ v0.13.1`（注释:327-328；`kv_cache_buffers_1_/2_`:329-330；读写指针:331-333；构造初始化:199-200）；指针交换 `runtime/executor/llm_litert_compiled_model_executor.cc @ v0.13.1`（prefill `std::swap`:738；decode `std::swap`:947；单缓冲初始化:1960-1961）。
- 克隆：`runtime/core/session_advanced.cc @ v0.13.1`（`Clone`:389；`CloneAsyncLocked`:412）；执行器侧 `CloneContext`:`llm_litert_compiled_model_executor.cc:1288 @ v0.13.1`；`CloneKVCacheBuffers`:1243。
- 检查点与回退：`runtime/core/session_advanced.cc @ v0.13.1`（`SaveCheckpoint`:455；`RewindToCheckpoint`:467）；`CheckpointInfo` 结构:`session_advanced.h:257-260 @ v0.13.1`。
- 会话状态：`runtime/engine/engine.h @ v0.13.1`（`Clone`:245，用法示例注释:238；`CloneAsync`:263；`SaveCheckpoint`:270；`RewindToCheckpoint`:277）；`runtime/executor/llm_executor_io_types.h @ v0.13.1`（`RuntimeState`:78，"不含 KVCache 状态"注释:75-76；`LlmContext`:92）。

<!-- 实验（--max-num-tokens 扫描解释 #2568、Clone 分叉、get_token_count 增长）数字待基准 D 回填〔基准 D〕。KV cache 公式为示例量级；具体模型 L/H_kv/D 待第 7 章 litertlm_print 读出后可补精确值。图 6-2(增长)、图 6-3(状态分叉) 与表 6-1(内存账) 规格见 notes.md，本轮先出签名图 6-1(双缓冲)。 -->
