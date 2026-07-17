# 第 6 章 KV cache 与会话状态

> 使命：把 KV cache 的容量与带宽两笔开销算清楚，说明 decode 为什么离不开它、又为什么被它拖慢；再看 LiteRT-LM 如何把这份状态封装成可拷贝、可序列化、可回退的对象，换来会话克隆、检查点、回退等能力。

第 5 章末尾留了一个未算完的数：那条 25 tokens/s 的纯权重带宽上限，实测为什么还够不着。差额的主要来源是 KV cache。本章把这笔账补齐，同时回答第 2 章「二十个问题」里的第 13、14 问。

术语约定：本章反复出现的 decode step，指一次「前向计算加采样」的迭代，每个 decode step 产出一个 token。prefill 指把输入 prompt 一次性喂入模型、批量填充 KV cache 的阶段。两者的机制展开见第 4 章与第 5 章。

## KV cache 的内存占用估算

先说它是什么。注意力机制里，每生成一个新 token，都要拿它的 query 与前文每个 token 的 key、value 做注意力计算（attention over previous tokens）。如果不缓存，第 1000 个 decode step 就要把前 999 个 token 的 key/value 全部重算一遍，第 1001 步再重算一遍。这正是第 1 章内存带宽约束的最坏放大：计算量随上下文长度平方增长。

KV cache 就是把这些 key/value 存下来，算过一次不再重算，以内存容量换掉重复计算。

它有多大，可以写下一个封闭公式：

$$ \text{KV 字节} = 2 \times L \times H_{kv} \times D \times S \times b $$

其中 L 是层数，H_kv × D 是每层 KV 的投影维度，S 是序列长度（缓存了多少个 token），b 是每个元素的字节数，最前面的 2 对应 key 和 value 各一份。

代入一组示例量级（层数 30、KV 投影维度 1024、fp16 即每元素 2 字节），每个 token 约

$$ 2 \times 30 \times 1024 \times 2 = 122880 \text{ 字节} \approx 120 \text{ KiB/token} $$

到 4096 个 token 的上下文，就是约 480 MiB。这里的 L、H_kv、D 是便于复算的示例量级；真实数字可以从模型文件里读出。本书对基准模型 Gemma 4 E4B 做了实剖（方法与完整数据见附录 D）：decode signature 的 KV 输入共 48 个张量，即 24 层各一对 K/V；其中 20 层形状为 `[1, 2, 32003, 256]`（H_kv = 2、D = 256），4 层为 `[1, 2, 32003, 512]`；**数据类型全部是 INT8**，即 b = 1。代入公式：每 token 2 × 2 × (20 × 256 + 4 × 512) × 1 = 28672 字节 = 28 KiB，4096 上下文合计 112 MiB。比示例小一个量级的原因有两个，都值得记住：真实模型的 KV 投影维度比示例小（GQA，grouped-query attention，分组查询注意力：多个查询头共享少量 KV 头，此模型只留 2 个），以及 KV cache 被量化到了 int8。后者是量化的第三个独立维度——第 7 章讲了权重量化与激活精度，KV cache 的存储精度是又一个可以单独选择的旋钮，它直接把本节的内存账和下一节的带宽账都除以二。

<div class="aside-compare">

这个旋钮交给谁拧，两家给了不同答案。llama.cpp 把 KV cache 的存储类型做成运行时命令行参数：`--cache-type-k` / `--cache-type-v` 可以在启动时选 f16、q8_0 等类型（`llama.cpp/common/arg.cpp:2174 @ b9873`），同一个模型文件，部署者按设备内存自行决定 KV 精度。LiteRT-LM 的 int8 KV 则是烘进模型文件的静态事实（本章实剖），选择权在模型发布者手里，运行时不可调。前者灵活、但把「精度损失是否可接受」的判断推给了每个部署者；后者把这个判断做在发布前的验证里，端上拿到的组合是被测过的。

</div>

把这个数字接回第 1 章的内存容量约束：权重约 2 GiB，KV cache 又要几百 MiB，而且它随对话变长持续增加。这解释了第 2 问——为什么 8 GiB 的手机跑 4B 模型也紧张，占内存的不止权重。

## KV cache 对解码带宽的影响

现在补第 5 章那笔账。每个 decode step，除了把 2 GiB 权重读一遍（第 1 章的账），还要读一遍到目前为止的整个 KV cache，并把新 token 的 key/value 追加进去。上下文越长，这部分访存越重。

于是那条 25 tokens/s 的纯权重上限，只计入了「权重」这一项；实测更慢，慢下来的部分有很大一头是 KV cache 的读写带宽（还有采样、非注意力算子等零头）。这里防一个误读：本机 cpu 实测 24.8 与那个「25」几乎相等纯属巧合，带宽上限要按各自机器的带宽重新计算，方法见第 2 章。上下文短时差距小，一旦长到几千 token，KV cache 的带宽开销就不可忽略，decode 会随对话变长而变慢。本书基准里这条曲线清晰可见：上下文从 256 拉到 4096，decode 从 24.8 降到 20.7 tok/s（cpu，-17%），gpu 也从 50.6 降到 45.6〔基准 D〕。这不是实现的问题，而是自回归加注意力的固有代价。

### `--max-num-tokens` 如何决定预留大小

这也顺带解释了第 13 问，一个真实的上游现象（`LiteRT-LM#2568`）：`--max-num-tokens` 这个参数为什么会影响解码速度。它设定的是 KV cache 能容纳多少 token，也就是要预留多大的一块缓存。在固定形状的执行路径上（第 4 章），decode 的注意力按整个预留长度计算，没用到的位置靠 attention mask 屏蔽，但这些位置的读写并不因 mask 而省去。预留多大，每个 decode step 就要为多大的缓存付出访存代价。

这个参数并非直接来自命令行填多少就是多少，它有一段默认值推导。当用户没有显式指定时（`GetMaxNumTokens()` 返回 0），LiteRT-LM 按 prompt 长度推一个默认值（`runtime/engine/engine_settings.cc:293-301`）：

```cpp
if (main_executor_settings_.GetMaxNumTokens() == 0) {
  // ...
  int max_num_tokens = ((num_prompt_tokens + 1023) / 4096 + 1) * 4096;   // (1)
  if (metadata.max_num_tokens() > 0) {                                   // (2)
    max_num_tokens = metadata.max_num_tokens();
  }
  main_executor_settings_.SetMaxNumTokens(max_num_tokens);
}
```

(1) 是核心：它先给 prompt 的 token 数加上 1023，做整数除以 4096 再向上取一格，最后乘回 4096。效果是把预留长度对齐到 4096 的整数倍，并且总留出至少一格（那个 `+ 1`）给 decode 输出，注释里说的 1024 默认解码长度就藏在这个 `+ 1023` 里。举例：prompt 有 100 个 token，`(100 + 1023) / 4096` 整数除得 0，`+ 1` 得 1，乘 4096 得 4096；prompt 涨到 4000 个 token，`(4000 + 1023) / 4096` 整数除仍得 1，`+ 1` 得 2，预留跳到 8192。这解释了 `#2568` 里观察到的现象：预留长度按 4096 的粒度阶梯式跳变，而不是随 prompt 平滑变化。(2) 是覆盖闸门：如果模型元数据里写了 `max_num_tokens`，就用元数据的值，模型作者对自己的上下文窗口有最终发言权。

预留长度确定后，它就是第 1 节公式里的 S。预留 4096，KV cache 按 4096 个槽位分配内存、每个 decode step 按 4096 的宽度做注意力，哪怕当前只填了 100 个 token。这就是 `#2568` 里参数影响速度的直接原因：调大 `--max-num-tokens` 不改变 prompt，却把每步 decode 要读写的 KV cache 宽度整体放大。

预留长度还是解码循环的终止条件之一。`ShouldStop` 判定何时停止解码（`runtime/core/tasks.cc:99-100`）：

```cpp
} else if (current_step >= max_num_tokens) {
  // Reaching maximum number of kv-cache size.
  return true;
```

`current_step` 是已经填到第几个槽位。一旦它顶到 `max_num_tokens`，KV cache 没有空位再追加，解码必须停。这条判定的上方，`TryGetMaxNumTokens`（`tasks.cc:73`）在执行器设置取不到时回退到 `kDefaultMaxNumTokens = 4096`（`:72`），并挂了一条 `TODO(b/423364170)`——目标是让所有 LLM 执行器都遵守模型返回的最大 token 数、届时移除这个默认回退。所以 4096 这个数字在代码里出现两次，含义一致：它既是默认预留粒度，也是取不到配置时的兜底上限。

### 固定形状与动态 KV cache

上面说「固定形状路径上，未用到的位置靠 mask 屏蔽、读写不省」，这句话背后有一个分支。LiteRT-LM 的 KV cache 支持两种形态：固定形状预留满 S 个槽位，动态形状则随实际长度增长。区分它们的是 KV 张量里有没有一个动态维度。推断上下文宽度的代码写在这里（`runtime/executor/litert/kv_cache.cc:302-311`）：

```cpp
LITERT_ASSIGN_OR_RETURN(const SimpleTensor& mask_tensor,
                        signature.InputTensor(mask_input_name));
// ...
auto dims = mask_tensor_type.Layout().Dimensions();
// Expect [1, 1, Sequence, KV Length]
RET_CHECK_EQ(dims.size(), 4);
const bool is_dynamic_kv_cache = k_dynamic_dim.has_value();   // (1)
context_size = is_dynamic_kv_cache ? 1 : dims[3];             // (2)
```

(1) 用 key 张量有没有动态维度（`k_dynamic_dim.has_value()`）判定是不是动态 KV cache。(2) 是关键分叉：动态时 `context_size` 记为 1，固定时取 attention mask 的最后一维 `dims[3]`，也就是完整的 KV Length。注释里写明 mask 的形状是 `[1, 1, Sequence, KV Length]`；代码宁可从 mask 推断上下文宽度，因为 key、value 张量的内部布局各不相同、不能直接读出这个宽度。

固定形状下 `context_size` 等于整个预留宽度，这一句就把上一段的带宽账坐实了：注意力算子看到的序列宽度是 S，不是当前已填的长度。屏蔽由 attention mask 完成——`FillAttentionMask` 每个 decode step 只把当前步之前的合法位置置为可见（`runtime/executor/litert_compiled_model_executor_utils.cc:363-366`）：

```cpp
for (int b = 0; b < batch_size; ++b) {
  for (int i = 0; i < steps; ++i) {
    int current_step = start_timestep + i;
    int offset = b * batch_offset + i * channel_size;
    // For current step = n, we fill (n+1) positions for the mask sequence.
```

注释那行说得很直白：当前步为 n 时，mask 只填 n+1 个可见位置，其余位置保持屏蔽值。屏蔽让被 mask 的位置不贡献注意力结果，却不改变张量的物理宽度——kernel 仍按 S 的宽度遍历、读取整段 KV。这正是「固定预留付固定带宽」的机制来源：mask 管的是数值正确性（不让未来 token 泄漏进注意力），管不了访存量。动态 KV cache 把 `context_size` 收到实际长度，才能让访存量随上下文增长而非一步到位付满，代价是形状动态会限制部分后端的图优化（第 4 章讨论过固定形状对编译期优化的价值）。

<figure>
{{#include figs/fig-6-1.svg}}
<figcaption>图 6-1　KV cache 双缓冲。每个 decode step 从旧缓冲读取、向新缓冲写入，随后交换两个指针，以一次指针交换规避同缓冲读写限制并消除数据拷贝。</figcaption>
</figure>

## 双缓冲

KV cache 每步都要读写，这在 GPU 上撞见一个具体约束：部分 GPU 后端不允许对同一块缓冲同时读和写。这不是推断，源码注释直接写在成员声明上方（`runtime/executor/llm_litert_compiled_model_executor.h:327-333`）：

```cpp
// KV cache double buffers because some GPU backends can't allocate one buffer
// for both read and write at the same time.
absl::flat_hash_map<absl::string_view, TensorBuffer> kv_cache_buffers_1_;      // (1)
absl::flat_hash_map<absl::string_view, TensorBuffer> kv_cache_buffers_2_;      // (2)
absl::flat_hash_map<absl::string_view, TensorBuffer>* input_kv_cache_buffers_;  // (3)
absl::flat_hash_map<absl::string_view, TensorBuffer>*
    output_kv_cache_buffers_;
```

(1)(2) 是两套实打实的缓冲，各持一份 KV。(3) 的两个指针是关键：它们不拥有数据，只指向那两套缓冲之一。`input_kv_cache_buffers_` 指向「这一步从哪套读」，`output_kv_cache_buffers_` 指向「这一步向哪套写」。构造时（`:199-200`）两者分别指向 `kv_cache_buffers_1_` 和 `kv_cache_buffers_2_`，一读一写、错开。这是 double buffering（双缓冲），一种以指针交换规避读写别名的常见工程手法，图形学和并发编程里都有它。

交换动作发生在每次跑完模型之后。prefill 的路径上（`llm_litert_compiled_model_executor.cc:737-738`）：

```cpp
if (!gpu_optimized_single_buffer_cache_) {   // (1)
  std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);   // (2)
}
```

(2) 只交换两个指针的指向，不动缓冲里的字节。这一步刚往 `output` 那套写了新 token 的 key/value，交换后它就成为下一步的 `input`；上一步读过的旧缓冲腾出来当下一步的写入目标。几百 MiB 的 KV 数据一次也没拷贝。(1) 是个例外闸门：`gpu_optimized_single_buffer_cache_` 为真时跳过交换。同一份 `std::swap` 在 decode 路径也有一次（`:946-947`），逻辑一致。

### 单缓冲路径的额外代价

`gpu_optimized_single_buffer_cache_` 这个闸门牵动的不止一次 swap。某些 GPU 后端反而支持同缓冲原地更新（inplace update），启用后可以省掉第二套缓冲，把 KV 内存对半砍。这条路径值得摊开看，因为「省一半内存」不是白得的，它换来了一处额外机制。

启用的判据是模型签名里有没有一个 int32 参数张量（`llm_litert_compiled_model_executor.cc:428-429`）：

```cpp
if (signatures_.input_int32_param.has_value()) {
  gpu_optimized_single_buffer_cache_ = true;
```

有这个参数张量，就认定后端走单缓冲原地更新路径。之后两套缓冲退化为一套，两个指针始终指向同一块，swap 被前一节那个 `if (!gpu_optimized_single_buffer_cache_)` 跳过。省下的是第二份几百 MiB 的 KV 内存。

换来的额外机制是：既然读写落在同一块缓冲上，kernel 必须知道「这一步该往哪个槽位区间写」,否则会覆盖到旧数据。双缓冲不需要这个信息，因为写入目标整块都是空的旧缓冲。单缓冲路径就得每步额外填一个参数张量告诉 kernel 当前的写入区间。prefill 与 decode 两侧各填一次（`llm_litert_compiled_model_executor.cc:673-677`，decode 侧在 `:904-907`）：

```cpp
if (gpu_optimized_single_buffer_cache_) {
  LITERT_RETURN_IF_ERROR(signatures_.input_int32_param.has_value());
  RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
      prefill_input_buffers[signatures_.input_int32_param.value()],
      start_step, ids.size()));
}
```

填的内容是什么，看 `FillSingleBufferCacheParamTensor`（`runtime/executor/litert_compiled_model_executor_utils.cc:332-335`）：

```cpp
int end_index = start_index + update_length;
int32_t params[] = {start_index, end_index, end_index};   // (1)
LITERT_RETURN_IF_ERROR(sizeof(params) <= packed_size);
std::memcpy(param_tensor_lock_and_addr.second, params, sizeof(params));
```

(1) 就是那个写入区间：起点 `start_index`（当前 step）、终点 `end_index`。函数上方注释交代了这三个 int32 的分工——前两个给 `add_values_to_cache` kernel 用来定位写入区间，第三个给 `runtime_batched_matmul` kernel 检查通道结束位置。两种方案的差别就此显形：双缓冲用「换一整块空缓冲写」这个简单办法免掉了区间管理；单缓冲省了内存，但把区间管理的责任交回给了 host，每个 decode step 多一次 memset 加 memcpy 的小填充，并要求模型签名带上这个 int32 参数张量。这是一个具体的空间与复杂度的对换：双缓冲多一份内存、逻辑更简单；单缓冲省内存、但 host 侧和 kernel 都要多认一个「当前写到哪」的参数。

这个双缓冲方案没有高深算法，它是端侧工程的典型形态：一个来自硬件的约束（部分 GPU 不能同缓冲读写），一个来自成本的考量（不想拷贝几百 MiB），组合出一个直接的解法（两套缓冲加指针交换），再为愿意付出额外复杂度的后端保留一条省内存的单缓冲支路。

## 会话状态封装为可搬运对象

到这里，一次会话的核心状态已经清楚：一份 KV cache，加上「当前到第几步」这样的元信息。LiteRT-LM 把它们封装成一个对象 `LlmContext`（`runtime/executor/llm_executor_io_types.h:92`）：

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

三样东西各管一摊。(3) 的 `processed_context_` 装已处理的 token 和它们的 KV cache，那几百 MiB 就在这下面。`runtime_config_` 是运行配置。(1) 的 `runtime_state_` 记运行状态，其中 `current_step` 是「当前到第几步」的游标。`RuntimeState` 上方那行源码注释特意声明它「不含直接与 KVCache 相关的状态」（`:75-77`）：搬运时 KV cache 走 `processed_context_` 一路，步数游标走 `runtime_state_` 一路，两者分开。(2) 三个访问器返回引用，底下全用 `unique_ptr` 独占持有；要拷贝整个上下文，只需把这三个 `unique_ptr` 各复制一份。

把状态封装成一个可整体拷贝、可序列化的对象，是第 2 章那条「会话状态即对象」原则的兑现。严格地说，这条原则的技术含义是：一次会话的全部可变状态都归拢进单一对象、外部只通过它的接口读写，从而这份状态能被整体 clone、serialize、restore。它一旦成立，几件原本很难的事就顺理成章：

- 克隆会话（`Clone`，`runtime/engine/engine.h:245`；异步版 `CloneAsync`，`:263`）：复制这个对象，就得到一个独立的会话分支。
- 存检查点、回退（`SaveCheckpoint`，`engine.h:270`；配套的 `RewindToCheckpoint`）：给当前状态打个标记，之后能退回来。

克隆的价值，头文件注释里有个现成的例子（`engine.h:238`）：session1 先 `Prefill("What is the tallest building ")`，`Clone` 出 session2，之后 session1 接 `"in the world?"`、session2 接 `"in France?"`，那段公共前缀的 prefill 只跑一次。这就是第 14 问的答案：克隆分叉不必重算公共前缀，因为那段 KV cache 被整份复制了过去，而不是重新 prefill。

复制到底发生在哪一行。`Clone` 一路走到执行器的 `CloneContext`（`llm_litert_compiled_model_executor.cc:1288`）：

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

(2) 就是上一节那个 `LlmContext` 的新实例：克隆的产物正是一个装齐三样东西的新上下文对象，「会话状态即对象」在这里收口。真正拷贝字节的是 (1) 的 `CloneKVCacheBuffers`（`:1243`），它逐块深拷贝——遍历 `input_kv_cache_buffers_`，对每块调 `CopyTensorBuffer` 复制一份（`:1246-1248`）：

```cpp
for (const auto& [name, buffer] : *input_kv_cache_buffers_) {
  LITERT_ASSIGN_OR_RETURN(auto buffer_copy, CopyTensorBuffer(env_, buffer));
  kv_cache_buffers[name] = std::move(buffer_copy);
}
```

这就是接口里 `DeepCopy` 注释那句「昂贵操作」的具体来源：4096 token 上下文下按第 1 节的实剖数字约 112 MiB，克隆一次就要原样拷这么多字节。克隆的代价，全在这个循环里。

### 克隆之后：装载是零拷贝的 move

克隆的账面代价是一次深拷贝，那把它装载回执行器要不要再拷一次？答案是不要。`RestoreContext`（`llm_litert_compiled_model_executor.cc:1305`）直接以 move 接收这份上下文：只有 `current_step > 0` 时才把 KV 缓冲 move 进 `input_kv_cache_buffers_`（`:1311-1315`），step 0 的空 KV cache 允许直接复用现有缓冲、省掉一次搬移。字节从头到尾只在 `CloneKVCacheBuffers` 里拷了一次。

仓库里还躺着一个写好了、却没接线的拷贝实现，值得一看，因为它标着这条链路的演进方向（`llm_litert_compiled_model_executor.cc:1253-1263`）：

```cpp
absl::Status LlmLiteRtCompiledModelExecutorBase::RestoreKVCacheBuffers(
    const absl::flat_hash_map<absl::string_view, TensorBuffer>&
        kv_cache_buffers) {
  // TODO: b/452977992: Instead of copying, consider replacing our kv cache
  // buffers the caller's.
  if (!gpu_optimized_single_buffer_cache_) {
    for (const auto& [name, buffer] : kv_cache_buffers) {
      RETURN_IF_ERROR(CopyBuffer(buffer, (*input_kv_cache_buffers_)[name]));   // (1)
    }
  }
  return absl::OkStatus();
}
```

(1) 对每块 KV 缓冲逐块拷贝。这个函数在 v0.13.1 没有任何调用方——它是一份备用的拷贝式装载实现，挂着 TODO `b/452977992`：与其拷贝，不如直接用调用方传入的缓冲替换掉自己的（move 而非 copy）。而这条优化方向，现行 `RestoreContext` 的 move 装载已经兑现了。函数里那个 `if` 也有信息量：单缓冲路径（`gpu_optimized_single_buffer_cache_` 为真）跳过整个拷贝块，因为它没有可替换的第二套缓冲。

## KV cache 的拷贝与序列化接口

克隆、回退这些能力，底层要求 KV cache 本身能被拷贝、能被序列化。所以它有一个专门的接口 `KVCacheInterface`（`runtime/executor/kv_cache_interface.h:28`），一组纯虚函数划定了拷贝与序列化的边界：

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

(1)(2) 是序列化与反序列化的一对：`Serialize` 把 KV cache 压成一个 `std::string` 字节串，`Load` 反向读回，用于落盘或跨进程传递。(5) 的 `DeepCopy` 返回一个新的 `unique_ptr<KVCacheInterface>`，源码注释就写在这行上方：`This is an expensive operation. Use sparingly.`（`:60`），它正是上一节 `CloneKVCacheBuffers` 那个逐块深拷贝循环的接口对应物。(3)(4) 是一对方向相反的批处理拷贝，注释里各带一个 shape 例子：`SelectAndCopyFrom` 从 `[3, ...]` 的多分支缓存里挑第 `batch_index` 条拷到自己这个 `[1, ...]`；`BroadcastAndCopyFrom` 反过来，把 `[1, ...]` 的一条广播到 `[3, ...]` 的每一路。它们服务于并行采样这类批量场景。

这五个方法全是 `= 0` 的纯虚声明，`KVCacheInterface` 只定契约、不含实现。真正的字节拷贝落在具体后端里，比如 `LitertKVCache`。接口与实现分离，是 LiteRT-LM 支持多后端（cpu/gpu/npu）的一贯手法，后面几章还会反复见到。

### 序列化在 LiteRT 后端尚未实现

这里有一处容易误读的地方，值得当面点破。接口把 `Serialize`/`Load` 列为契约，但 LiteRT 后端目前并没有兑现它们（`runtime/executor/litert/kv_cache.h:45-51`）：

```cpp
absl::StatusOr<std::string> Serialize() const override {
  return absl::UnimplementedError("Not implemented");
}

absl::Status Load(absl::string_view serialized_kv_cache) override {
  return absl::UnimplementedError("Not implemented");
}
```

两个都是返回 `UnimplementedError` 的桩实现。也就是说，在 v0.13.1 的 LiteRT 后端上，「把 KV cache 落盘、跨进程传」这条路径接口上留了口子、实现上还没打通；配套的单测 `SerializeNotSupported`（`kv_cache_test.cc:117`）恰恰断言它返回未实现错误。当前真正能用的 KV cache 搬运，是内存内的 `DeepCopy` 与下面两个批处理拷贝，而非序列化。把三者放到同一张代价表上看得更清楚：

| 操作 | 实现 | 字节搬运量（4096 上下文，基准模型） | 用途 |
|---|---|---|---|
| `RewindToCheckpoint` | 改一个 `int` 游标 | 0（不动 KV 字节） | 回退到检查点 |
| `DeepCopy` / `CloneKVCacheBuffers` | 逐块 `CopyTensorBuffer` | 约 112 MiB（装载走 move，零拷贝） | 克隆会话分支 |
| `Serialize` / `Load` | 未实现（`UnimplementedError`） | 不适用 | 计划中的落盘、跨进程传递 |

表 6-1 把「会话状态即对象」的收益兑现成一张可查的代价账：同一份 KV cache，回退几乎零成本、克隆是一次百余 MiB 的深拷、序列化则还在路上。

### 并行采样的批处理拷贝

`SelectAndCopyFrom` 与 `BroadcastAndCopyFrom` 服务于并行采样：从一条上下文广播出 N 条并行走，各自采样后再择回其中一条。LiteRT 后端的实现把批处理的约束写在了一组 `RET_CHECK` 里（`runtime/executor/litert/kv_cache.cc:351-360`）：

```cpp
absl::Status LitertKVCache::BroadcastAndCopyFrom(KVCacheInterface& other) {
  auto other_litert = dynamic_cast<LitertKVCache*>(&other);
  RET_CHECK(other_litert != nullptr) << "Only support LitertKVCache.";
  RET_CHECK(!bank_2_key_cache_buffers_.has_value());               // (1)
  RET_CHECK(!other_litert->bank_2_key_cache_buffers_.has_value()); // (1)
  RET_CHECK_EQ(other_litert->batch_size_, 1);                      // (2)
  RET_CHECK_GT(batch_size_, other_litert->batch_size_);            // (3)
```

(2)(3) 划定方向：源必须是 batch size 为 1 的单条（`other_litert->batch_size_ == 1`），目标的 batch size 必须更大，广播才有意义。(1) 是一处关键边界：源和目标都不能持有第二套缓冲（`bank_2` 必须为空）。`bank_2` 就是双缓冲里的第二套。这条断言等于说，批处理拷贝不能在双缓冲激活的状态下做——并行采样与双缓冲这两种用途在这里互斥。真正搬字节的是 `BroadcastAndCopyBuffer`（`:179`），它对每块缓冲把源的内容 memcpy 复制 `dst_batch_size` 份，铺满整个 batch 维度（`:194-196`）：

```cpp
for (int i = 0; i < dst_batch_size; ++i) {
  memcpy(dst_buffer_ptr, src_buffer_ptr, src_buffer_size);
  dst_buffer_ptr += src_buffer_size;
}
```

反方向的 `SelectAndCopyFrom` 断言对称（`:328-330`）：目标的 batch size 必须更小（`other.batch_size > batch_size`），并从源里挑出第 `batch_index` 条。它的搬运靠 `SelectAndCopyBuffer`，用一句指针偏移定位到要挑的那一条（`:174`）：

```cpp
src_buffer_ptr += batch_index * dst_buffer_size;
memcpy(dst_buffer_ptr, src_buffer_ptr, dst_buffer_size);
```

这段拷贝依赖一个布局假设，注释写在偏移上方：KV cache 的布局是 `[batch * X, ...]` 或 `[1, batch * X, ...]`，同一 batch 的数据在内存里连续、不与其他 batch 交错，所以「挑第 batch_index 条」可以简化成一次线性偏移加一次 memcpy。这个假设对当前所有后端的 LLM 模型成立，注释也如实标明它是假设而非普遍保证。

## 从 KV cache 中丢弃思考内容

最后一个应用把前面的机制串起来。有些模型会先「想」再答，把思考过程也一并生成。思考内容对用户不必展示，也不该占着 KV cache，否则接下来每个 decode step 都要连着这段思考一起读，付出额外访存开销（第 2 节的账）。

LiteRT-LM 的处理是把思考这段 channel 内容从 KV cache 里丢弃（discard）。做法正是回退：退到思考开始前的那个位置，丢掉这段的 KV cache。看 `RewindToCheckpoint` 怎么退（`runtime/core/session_advanced.cc:467`）：

```cpp
int target_step = it->second.step;    // (1)
session_state_ = it->second.state;
absl::erase_if(checkpoint_map_, [target_step](const auto& pair) {   // (2)
  return pair.second.step > target_step;
});
// ...
return execution_manager_lock->SetCurrentStep(*session_info_, target_step);   // (3)
```

回退没有拷贝、没有删除任何 KV 字节。它做的是 (3)：把游标 `current_step` 调回 `target_step`，也就是上一节 `RuntimeState` 里那个整数。`SaveCheckpoint`（`:455`）存的也不过是 `{current_step, session_state_}` 这一对（`CheckpointInfo`，`session_advanced.h:257-260`），一个步数加一个会话状态枚举，几个字节。之后继续 prefill 时，新 token 从 `target_step` 这个位置往后写，直接覆盖掉原来那段思考占的 KV 槽位。旧字节没被主动清除，而是被下一轮写入盖过。(2) 顺手把 `target_step` 之后的所有检查点从 map 里删掉，让检查点集合始终是当前时间线的一条前缀，避免退回后残留一个指向「未来」的悬空标记。

一个只在 `RuntimeState` 上加减整数、连内存都不释放的回退，就解决了「别让思考污染上下文」的问题。代价与克隆恰成对照：`Clone` 要深拷一百多 MiB（4096 上下文，上一节那个循环），`RewindToCheckpoint` 只动一个 `int`。这也是表 6-1 那三行代价的两端。

## 小结

KV cache 是用内存换计算的经典权衡：它省掉重复的注意力计算，代价是占内存、也占带宽（decode 变慢的一部分来自它）。`--max-num-tokens` 决定预留多大，进而决定每个 decode step 的固定访存宽度，这解释了它为何按 4096 的粒度影响速度。围绕这份状态，LiteRT-LM 做了几件事：一个双缓冲方案应对 GPU 的读写约束（并为部分后端保留省内存的单缓冲支路）；一套「会话状态即对象」的封装换来克隆、检查点、回退；具体的字节搬运则由 `DeepCopy` 与批处理拷贝承担，序列化路径在 v0.13.1 尚未实现。回退还顺带解决了思考内容污染上下文的问题。

下一章去看这份账的另一半：模型本身怎么被压小、装箱、变体——量化、`.litertlm` 格式与 LoRA。

---

## 练习与自查

1. **真实参数复算。** 用实剖参数（每 token 28 KiB）算 8192 上下文的 KV cache 占用；再算静态槽位 32003 全预留是多少。
2. **成本对比。** 4096 上下文时，Clone 一次要拷贝约多少字节？RewindToCheckpoint 一次要改动什么？为什么两者代价差这么多？
3. **代码定位。** 双缓冲交换为什么不搬数据？找出 prefill 与 decode 路径上各自执行交换的那一行。
4. **旋钮推演。** 把 `--max-num-tokens` 调大一倍，内存占用与 decode 速度各受什么影响？与 `LiteRT-LM#2568` 的现象对上。
5. **三维度归纳。** 权重量化、激活精度、KV cache 精度是三个独立维度。为本书基准模型写出它的三元组，并再举一个合法但不同的组合。


<!-- 实验缺口：--max-num-tokens 扫描、Clone 分叉验证、get_token_count 增长三项未单测（带宽曲线已有实测：24.8→20.7、50.6→45.6〔基准 D〕）。KV 公式已用实剖真值（24 层/int8/28 KiB per token，附录 D 第六节）。图 6-3（Clone/Rewind 状态分叉）未出，是下半章最该补的图。2026-07-16 评审修订：克隆链路按 v0.13.1 实况重写（CloneKVCacheBuffers 一次深拷 + RestoreContext move 装载；RestoreKVCacheBuffers 为零调用方死代码）。 -->
