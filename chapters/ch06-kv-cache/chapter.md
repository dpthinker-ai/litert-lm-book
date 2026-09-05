# 第 6 章 KV cache：容量、带宽与会话生命周期

> 本章计算 KV cache 的容量与带宽成本，并说明 LiteRT-LM 的双缓冲、会话克隆、检查点回退与序列化的能力边界。

第 1 章估算的 25 tokens/s 只计算了权重读取，KV cache、采样和其他算子均未计入；第 2 章 2.4.1 节又用实测的减速折算出每个上下文 token 约 59 到 107 KiB 的等效附加字节，并把详细计算留给了本章。本章先按张量形状把 KV cache 每 token 的真实字节数算出来（6.1 节），再看它在 decode 每一步的带宽开销中占多大比例，以及 `--max-num-tokens` 为什么会改变吞吐（6.2 节）。后半章转向 KV cache 的生命周期：部分 GPU 后端要求的双缓冲（6.3 节），会话克隆的写时复制与它真正复制的字节数（6.4 至 6.5 节），序列化接口的现状（6.6 节），检查点与 channel 过滤（6.7 至 6.8 节）。这些内容对应第 2 章表 2-2 的问题 13 与 14。

## 6.1　KV cache 的内存占用估算

注意力机制生成新 token 时，要用该位置的 query 与此前所有 token 的 key/value 计算。不缓存这些 key/value，第 1000 步就要重算前 999 个 token 的 key/value，第 1001 步再算一遍：单步开销随上下文长度线性增加，生成整段序列的累计开销呈二次增长。KV cache 把已经算出的 key/value 存在内存里，后续 decode step 直接读取，以内存容量为代价省去这部分重复计算。

KV cache 的大小沿用 1.3 节的理论公式：

$$ \text{KV 字节} = 2 \times L \times n_{kv} \times d_{head} \times S \times b $$

其中 \\(L\\) 是层数，\\(n_{kv}\\) 是 KV 头数，\\(d_{head}\\) 是每头维度，\\(S\\) 是缓存的 token 数，\\(b\\) 是每个元素的字节数；最前面的 2 表示 key 和 value 各一份。1.3 节的示意参数（\\(L=32\\)、\\(n_{kv}=8\\)、\\(d_{head}=128\\)、FP16）给出 128 KiB/token；真实参数应从模型文件读取，下面直接对本书基准模型代入。

附录 D 记录了 Gemma 4 E4B 的模型检查结果：decode signature 有 48 个 KV 输入张量，即 24 层各一对 K/V。20 层的 K 张量形状为 `[1, 2, 32003, 256]`，V 张量为 `[1, 2, 256, 32003]`；其余 4 层的最后两维分别为 `[32003, 512]` 与 `[512, 32003]`。数据类型均为 INT8，因此 \\(b=1\\)；转置布局不改变元素数。形状里的 2 是 KV 头数，256 或 512 是每头维度，32003 是文件里写下的序列宽度（它是一个占位值，实际宽度由运行时决定，见 6.2.2 节）。代入公式，每 token 占用 \\(2 \times 2 \times (20 \times 256 + 4 \times 512) \times 1 = 28672\\) 字节，即 28 KiB。4096 个 token 合计 112 MiB。

这个数不到 1.3 节 FP16 示意值的四分之一，原因有二：该模型采用 GQA，KV 头数只有 2；KV cache 的存储类型又是 INT8，同一形状的容量和传输字节数相对 FP16 减半。KV cache 精度、权重精度与激活精度是三项独立的配置，不能由其中一项推断另一项。

<div class="aside-compare">

llama.cpp 与 LiteRT-LM 在不同阶段确定 KV cache 精度。llama.cpp 提供运行时参数 `--cache-type-k` 和 `--cache-type-v`，部署者可选择 f16、q8_0 等类型。[^ch06-llamacpp-kvtype] Gemma 4 E4B 的 LiteRT-LM 模型则把 INT8 KV 类型固定在导出产物中，运行时不提供对应参数。前者把选择留给部署时，后者由模型发布者确定模型与 KV 精度的组合。本节不比较两种方案的精度结果。

</div>

内存预算不能只计入权重。以本书基准模型为例，4096 个 token 的 KV cache 为 112 MiB；若宽度取到这个模型允许的默认上限 32000（6.2.2 节），则为 875 MiB。其他模型的形状和精度不同，应按本节公式分别计算。

附录 D 第十四节的手机实验固定 4096-token 上限，实际输入为 77 token，并分别记录了会话释放与引擎释放。这里的 112 MiB 仍是按模型形状算出的单份 KV 大小，不能直接等同于某个阶段的进程 RSS 差值：阶段读数还受驻留、页面换入换出、运行时对象与其他缓冲变化的影响，GPU 分配也未必全部进入进程统计。要单独核对 KV 的实际分配，应记录 KV 缓冲的大小与数量；只改变预留宽度的成对实验可以观察总占用如何变化，但还要区分 mask、工作区等随宽度变化的开销。

## 6.2　KV cache 对解码带宽的影响

标准全注意力的每个 decode step 都要读取模型权重和此前所有位置的 KV cache，还要写入新 token 的 key/value。上下文越长，这部分数据访问越多。

第 1 章的 25 tokens/s 只包含权重读取，不能单独用它解释实测吞吐：主矩阵中 cpu/256 档的 24.8 tokens/s 与它接近，但两者的带宽条件和未计成本都不同，接近不构成验证。同一矩阵里，上下文从 256 增至 4096 后，cpu 的 decode 从 24.8 降到 20.7 tokens/s，gpu 从 50.6 降到 45.6 tokens/s（Apple M5 Pro，Gemma 4 E4B，decode 128 token；完整数据见附录 D）。2.4.1 节把这组减速折算成每个上下文 token 的等效附加字节，cpu 约 107 KiB、gpu 约 59 KiB，都比 6.1 节算出的 28 KiB 大得多。差距说明减速不只来自多读 KV：注意力计算量、mask 与 KV 张量的静态宽度、缓冲布局都随上下文或预留宽度一起变化，而本书没有用性能计数器把它们分开测量。本节先把 KV 读取本身的理论量算清，再单独看预留宽度这一项。

### 6.2.1　从容量公式推导理论扫描量

容量公式还能给出注意力侧的理论扫描量。设每个 token 的 K/V 条目合计为 \\(E\\) 字节，当前上下文长度为 \\(S\\)。若标准全注意力的每一层都访问此前全部位置，一次 decode step 扫描的 K/V 工作集约为 \\(E\\times S\\) 字节。decode 吞吐为 \\(T\\) tokens/s 时，对应的理论扫描速率为：

$$ B_{KV,\mathrm{scan}} \approx E \times S \times T $$

这个量既不是 DRAM 带宽实测值，也不是严格的总线流量上界。片上缓存可能减少主存读取，重复加载和中间布局转换则可能增加流量。输出侧也取决于实现：原地更新只需写入新增 K/V；非原地（out-of-place）kernel 还要写出另一份输出 cache。仅凭 host 侧的缓冲绑定无法确定这些操作产生的总线字节数。

把权重也纳入同一个估算。若每个 decode step 从主存读取一次 \\(W\\) 字节的权重和 \\(E\\times S\\) 字节的 KV，写回与其他流量忽略，带宽侧的乐观上限就是 \\(B_{\mathrm{eff}}/(W+E\\times S)\\)，其中 \\(B_{\mathrm{eff}}\\) 是 1.4 节定义的有效带宽。对基准模型，\\(W\\) 取主 decode 段的 2.26 GB：\\(S=4096\\) 时 \\(E\\times S\\) 为 112 MiB，约为 \\(W\\) 的 5%；\\(S\\) 取 32000 时为 875 MiB，约为 \\(W\\) 的 40%。短上下文下 KV 读取在带宽开销中占比很小，上下文接近模型允许的上限时才与权重同量级。该式的每项假设都要随设备和后端重新核对。

基准模型的 \\(E\\) 为 28 KiB/token。暂取 \\(T=25\\) tokens/s，可得到以下理论扫描量：

| 上下文 S | 有效 KV 容量 | 每步理论扫描量 | 25 tokens/s 对应的 KV 理论扫描速率 |
|---:|---:|---:|---:|
| 512 | 14 MiB | 约 14 MiB | 约 0.37 GB/s |
| 4096 | 112 MiB | 约 112 MiB | 约 2.94 GB/s |
| 8192 | 224 MiB | 约 224 MiB | 约 5.87 GB/s |
| 32000 | 875 MiB | 约 875 MiB | 约 22.9 GB/s |

> 表 6-1　KV 理论扫描量按 28 KiB/token 和固定的 25 tokens/s 复算，不是 Gemma 4 E4B 各层 kernel 的实测流量。带宽按 GB/s（\\(10^9\\) 字节每秒）计算；32000 是该模型静态宽度的默认上限（6.2.2 节）。

上下文从 4096 增至 8192，KV 容量和注意力侧的理论扫描量均翻倍。双缓冲（6.3 节）不改变注意力需要读取的已缓存 K/V 集合，但可能增加输出 cache 的写入量和常驻容量；表 6-1 不把这部分计入扫描速率。

权重那一项的量级可以同样复算。2.4 节识别出的主 decode 段 payload 为 2.26 GB；若每个 decode step 恰好读取该段一次，25 tokens/s 对应的权重读取速率为 56.5 GB/s，而 4096 上下文的 KV 理论扫描速率只有 2.94 GB/s，到 32000 时才增至约 22.9 GB/s。`.litertlm` 整文件为 3.66 GB，但其中包含 10 个模型段，不能把整文件大小当作每步读取的权重字节数。

### 6.2.2　`--max-num-tokens` 如何决定预留大小

问题 13 的后半句来自 `LiteRT-LM#2568` 报告的现象：`--max-num-tokens` 会影响 decode 吞吐。[^ch06-issue-2568] 该参数设定 KV cache 可容纳的 token 数，本书称为预留宽度。在固定形状路径上，attention mask 屏蔽尚未使用的位置，张量的宽度却由预留宽度决定；表 6-2 的同 prompt 对照显示，扩大预留宽度会降低吞吐。这一节回答两个问题：预留宽度如何确定，它又是怎样变成张量宽度的。

未显式指定该参数时，LiteRT-LM 根据 prompt 长度计算默认值：

```cpp
// runtime/engine/engine_settings.cc:293-302
if (main_executor_settings_.GetMaxNumTokens() == 0) {
  // ...
  int max_num_tokens = ((num_prompt_tokens + 1023) / 4096 + 1) * 4096;   // (1)
  if (metadata.max_num_tokens() > 0) {                                   // (2)
    max_num_tokens = metadata.max_num_tokens();
  }
  main_executor_settings_.SetMaxNumTokens(max_num_tokens);
}
```

代码行 `(1)` 等价于把 prompt token 数加 1024 后向上对齐到 4096 的整数倍：prompt 为 100 个 token 时得到 4096，4000 个 token 时得到 8192。默认预留宽度因此按 4096 分段变化，而不是随 prompt 长度连续变化。`(2)` 表示模型元数据中的 `max_num_tokens` 可以覆盖这项推导结果。

预留宽度通过 LiteRT 的 magic number 机制变成张量宽度。6.1 节看到的 32003 不是真实容量：模型转换时把序列宽度写成一个大于 10 的素数作占位值；运行时创建 LiteRT 环境时扫描 prefill signature，把 mask 输入的最后一维识别为上下文宽度的占位值，再连同目标值一起作为环境选项交给编译器，编译后张量的宽度就是目标值。目标值的规则如下：

```cpp
// runtime/executor/magic_number_configs_helper.cc:254-263
int64_t GetTargetNumber(int64_t magic_number, int64_t target_number_hint) {
  if (target_number_hint > 0 && target_number_hint < magic_number) {
    return target_number_hint;
  }
  int64_t default_target_number = GetDefaultTargetNumber(magic_number);
  // ...
  return default_target_number;
}
```

`--max-num-tokens` 小于占位值时直接采用；为 0 或不小于占位值时改用占位值以下最大的 256 的倍数，对 32003 即 32000，并打印一条警告。所以 `--max-num-tokens 4096` 得到宽度为 4096 的 KV 张量与 mask，8192 得到 8192；表 6-1 末行的 32000 也由此而来。预留宽度就是 6.1 节公式里的 \\(S\\)：固定形状模型预留 4096 时，KV 张量含 4096 个位置，当前上下文可能只用其中一部分；增大 `--max-num-tokens` 不改变 prompt，只扩大这些张量的静态宽度。

对照实验的条件为 cpu 后端、decode 128 个 token、磁盘缓存已预热；完整记录见附录 D（附录 D 称之为 `--max-num-tokens` 扫描）：

| 预留宽度 | prompt / prefill 分块计划 | decode tokens/s | 比较口径 |
|---|---|---:|---|
| 1024 | prompt 100 / `[128]` | 33.9 | 独立短 prompt 样本，不与下两行计算降幅 |
| 4096 | prompt 256 / `[128]` | 26.4 | 同 prompt 对照基线 |
| 8192 | prompt 256 / `[128]` | 21.5 | 仅放宽预留；相对 4096 下降约 19% |
| 1024 | prompt 256 / `[1024]` | 运行失败 | prefill 报 `dynamic_update_slice` 越界，用于检查容量边界 |

> 表 6-2　`--max-num-tokens` 对照实验（Gemma 4 E4B，cpu，各 2 次）。只有 4096 与 8192 两行使用相同 prompt 和 prefill 分块计划；预留宽度加倍后，decode 从 26.4 降至 21.5 tokens/s，下降约 19%。

1024 宽度、prompt 100 的 33.9 tokens/s 来自不同输入，只能作为单独样本，不能参与 4096 到 8192 的百分比计算。容量设置过小则可能直接失败：表中第 4 行的 prompt 为 256、prefill 分块计划为 `[1024]`，prefill 阶段触发 `dynamic_update_slice` 越界，而不是截断输入。排查 decode 吞吐时，应同时记录 prompt、prefill 分块计划和最终预留宽度。

预留宽度还是 decode 循环的终止条件之一：

```cpp
// runtime/core/tasks.cc:99-101
} else if (current_step >= max_num_tokens) {
  // Reaching maximum number of kv-cache size.
  return true;
```

`current_step` 是当前写入位置；达到 `max_num_tokens` 后 KV cache 没有剩余槽位，decode 停止，任务状态记为 `kMaxNumTokensReached`。读取这个上限时，若执行器没有提供配置，则回退到常量 `kDefaultMaxNumTokens = 4096`；源码的 TODO 注释说明该回退会在所有执行器都能返回上限后移除。代码里因此有两个 4096，作用不同：默认值推导里的 4096 是对齐粒度，回退常量里的 4096 是读取配置失败时的上限。

### 6.2.3　固定形状与动态 KV cache

LiteRT-LM 的 KV cache 有固定形状和动态形状两条路径，对应 4.2 节的两个执行器子类。执行器工厂创建时检查 prefill signature 的 KV cache 输入张量是否含动态维度（维度值为 -1），含则选动态子类，否则选固定形状子类；KV 张量与位置张量必须同为动态或同为固定，否则拒绝创建。固定形状路径下，KV 张量与 attention mask 的宽度在编译时就等于预留宽度（6.2.2 节），当前上下文只使用其中一部分；动态路径下缓冲随实际长度增长，首次 prefill 时按输入长度分配，之后按增量扩容。

固定形状下，mask 的静态宽度等于完整预留宽度 \\(S\\)，而不是当前已写入的长度。每个 decode step 只把当前步之前的位置设为可见：

```cpp
// runtime/executor/litert_compiled_model_executor_utils.cc:363-367
for (int b = 0; b < batch_size; ++b) {
  for (int i = 0; i < steps; ++i) {
    int current_step = start_timestep + i;
    int offset = b * batch_offset + i * channel_size;
    // For current step = n, we fill (n+1) positions for the mask sequence.
```

当前步为 n 时，mask 设置 n+1 个可见位置，其余位置保留屏蔽值。mask 决定数值上哪些位置参与注意力，但不缩小静态张量形状；具体后端是否跳过被屏蔽的位置，取决于编译结果和 kernel，不能仅由这段 host 代码判定。表 6-2 的同 prompt 数据表明，至少在 cpu 后端上，更大的静态宽度确实增加了端到端成本。动态形状让张量长度随实际上下文增长，但可能减少部分编译期优化机会，权衡见 4.2 节。

## 6.3　双缓冲

KV cache 每步都要读写，而部分 GPU 后端不允许同一块缓冲同时作为输入和输出。执行器成员声明处的注释记录了这条约束：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.h:327-333
// KV cache double buffers because some GPU backends can't allocate one buffer
// for both read and write at the same time.
absl::flat_hash_map<absl::string_view, TensorBuffer> kv_cache_buffers_1_;      // (1)
absl::flat_hash_map<absl::string_view, TensorBuffer> kv_cache_buffers_2_;      // (2)
absl::flat_hash_map<absl::string_view, TensorBuffer>* input_kv_cache_buffers_;  // (3)
absl::flat_hash_map<absl::string_view, TensorBuffer>*
    output_kv_cache_buffers_;
```

代码行 `(1)` 和 `(2)` 是两套 KV 缓冲，`(3)` 所示的输入、输出指针不拥有数据，只分别指向其中一套。当前 step 从一套读取、向另一套写入，完成后交换两者角色，这种方式称为双缓冲（double buffering）。

交换发生在模型执行完成之后。prefill 路径的代码：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:737-739
if (!gpu_optimized_single_buffer_cache_) {   // (1)
  std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);   // (2)
}
```

代码行 `(2)` 只交换指针，不复制缓冲内容：本步写入的输出缓冲成为下一步的输入缓冲，原输入缓冲成为下一步的输出目标。`(1)` 在单缓冲标志为真时跳过交换，原因见 6.3.1 节。decode 路径执行同样的操作。

两个 map 不一定对应两块独立分配。创建缓冲时，CPU 后端用 `TensorBuffer::Duplicate()` 把同一块输入缓冲登记进输出 map，源码注释称之为单缓冲，目的是节省内存和复制；只有 GPU 后端才为输入、输出分别创建缓冲。因此不能仅凭成员名称就把 KV 常驻容量固定乘以二。

<figure>
{{#include figs/fig-6-1.svg}}
<figcaption>图 6-1　GPU out-of-place KV 路径分别读写两套缓冲；执行结束后交换输入、输出指针。指针交换本身不复制张量数据。</figcaption>
</figure>

### 6.3.1　单缓冲路径的额外代价

部分模型的 signature 支持在同一缓冲上原地更新（in-place update）KV cache。判据是 signature 里有没有一个 int32 参数张量：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:428-429
if (signatures_.input_int32_param.has_value()) {
  gpu_optimized_single_buffer_cache_ = true;
```

该标志为真时，执行器不再交换指针；创建缓冲时，GPU 后端也不再创建输入侧的 KV 缓冲，只保留一套输出侧缓冲。在 CPU 后端上，缓冲仍按 6.3 节的复制登记方式创建，但只要 signature 含该参数张量，执行时同样填充参数、不交换指针。Gemma 4 E4B 的记录中存在 `param_tensor[1,1,1,7]`〔基准 D〕，所以它在 GPU 上使用单缓冲路径，不能按两套独立 KV 缓冲估算。

同一缓冲既读又写时，kernel 需要本步更新的槽位区间作为参数，所以单缓冲路径在每次 prefill 与 decode 前都要填充这个参数张量。prefill 路径的调用：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:673-678
if (gpu_optimized_single_buffer_cache_) {
  LITERT_RETURN_IF_ERROR(signatures_.input_int32_param.has_value());
  RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
      prefill_input_buffers[signatures_.input_int32_param.value()],
      start_step, ids.size()));
}
```

decode 路径在每一步前做同样的事，区间长度为 1。`FillSingleBufferCacheParamTensor` 先清零参数缓冲，再写入三个 int32：

```cpp
// runtime/executor/litert_compiled_model_executor_utils.cc:332-335
int end_index = start_index + update_length;
int32_t params[] = {start_index, end_index, end_index};   // (1)
LITERT_RETURN_IF_ERROR(sizeof(params) <= packed_size);
std::memcpy(param_tensor_lock_and_addr.second, params, sizeof(params));
```

`start_index` 和 `end_index` 标明本步更新区间；源码注释说明前两个参数供 `add_values_to_cache` kernel 使用，第三个供 `runtime_batched_matmul` kernel 检查 channel 结束位置。

两条路径的取舍可以归纳如下。双缓冲用两套缓冲和指针交换避开原地更新，代价是多一套 KV 常驻容量；单缓冲只保留一套 KV 数据，代价是模型 signature、host 代码和 kernel 要共同维护更新区间。CPU 后端的缓冲复用是第三种情况：创建时两个 map 的句柄共享底层缓冲，不能套用 GPU 的两套分配口径。是否交换 map 仍由参数张量标志决定：无参数张量时 CPU 也会交换指针，这一步不复制数据；signature 含参数张量时，CPU 上同样每步填充它。

## 6.4　会话克隆与写时复制

3.1.1 节按实现路径说明了 `Session::Clone`：克隆任务执行时，资源管理器让新旧两个 context handler 指向同一个 `SharedProcessedContext`，运行配置与运行状态按值复制，KV cache 不复制任何字节。本节接着回答问题 14 剩下的部分：共享之后，什么时候必须复制，复制多少字节，各后端有什么差别。

### 6.4.1　共享的上下文与各自的运行状态

每个会话在资源管理器里对应一个 context handler，它管理两类状态。第一类是已处理上下文（processed context）：已经 prefill 或 decode 过的 token 序列，以及它们在执行器里的 KV cache。这一类通过 `SharedProcessedContext` 共享：共享对象记录关联到它的全部 handler，也就是同一条上下文链；handler 处于活动状态时，实际的已处理上下文对象移入执行器，共享对象只保留链的成员关系。第二类是运行配置与运行状态（`RuntimeConfig`、`RuntimeState`），包括采样参数、当前步数 `current_step` 和随机数生成器，每个 handler 各有一份，活动时同样由执行器持有。

克隆得到的 handler 就是这样组成的：它复用原 handler 的共享对象，配置与状态按值复制，两个会话从相同的 `current_step` 开始。有两点容易误判。`RuntimeState` 里的随机数生成器是 `shared_ptr`，按值复制 `RuntimeState` 复制的是指针，两个会话此后共用同一个生成器对象；带音频上下文的会话会另行克隆音频状态，那不属于本章讨论的 LLM KV cache。

### 6.4.2　写时分离的触发条件

共享让第一个分支可以从公共前缀末尾继续执行而不复制 KV；只有当某个分支要在较早位置改写共享历史时，资源管理器才把它分离出去。判断发生在 prefill 入口。资源管理器先比较执行器里已处理 token 的数量与本次请求的当前步：两者相等，说明请求正好接在已处理序列末尾，直接执行；当前步更早，说明请求要从历史中间重新开始，此时先把新输入里与已处理 token 逐位相同的前缀删掉（这些 token 不必重算），把当前步推进到第一个不同的位置。删完仍有输入要处理，就必须改写历史，资源管理器再判断一件事：当前 handler 是不是共享链里最长的那个。

```cpp
// runtime/framework/resource_management/resource_manager.cc:322-332
ASSIGN_OR_RETURN(
    int largest_time_step,
    current_handler_->shared_processed_context()->LongestHandlerTimeStep(
        *llm_executor_));
if (largest_time_step != current_step) {
  // ...
  RETURN_IF_ERROR(SaveProcessedContextAndSeparateLoadedHandler(
      current_handler_, llm_executor_));  // (1)
}
```

`LongestHandlerTimeStep` 遍历共享链，读取每个 handler 的 `current_step`（活动 handler 的步数从执行器读取），取最大值。代码行 `(1)` 只在当前 handler 不是最长分支时执行：最长分支改写自己末尾之后的槽位不会影响其他分支，直接覆盖即可；较短分支要覆盖的槽位却正被更长的分支使用，必须先分离。decode 在准备截断已处理 token 时执行同一检查。因此复制时机取决于分支的执行顺序和步数：从末尾续写的分支不触发复制，较短分支需要分叉或截断时才触发。

分离由 `SaveProcessedContextAndSeparateLoadedHandler` 完成。它把执行器里当前的已处理上下文保存回原共享对象，再让当前 handler 改用一个新的共享对象：

```cpp
// runtime/framework/resource_management/resource_manager.cc:82-92
ASSIGN_OR_RETURN(auto llm_context, llm_executor->CloneContext());  // (1)
ASSIGN_OR_RETURN(auto current_processed_context,
                 llm_context->RetrieveProcessedContext());
RETURN_IF_ERROR(
    context_handler->shared_processed_context()->SetProcessedContext(
        std::move(current_processed_context)));

auto new_shared_processed_context =
    std::make_shared<ContextHandler::SharedProcessedContext>(nullptr);
RETURN_IF_ERROR(context_handler->UpdateSharedProcessedContext(
    new_shared_processed_context));  // (2)
```

代码行 `(1)` 从执行器取得一份独立快照，KV 字节的复制就发生在这里，复制多少见 6.4.3 节；`(2)` 把即将改写历史的 handler 从原共享链移出。此后执行器回到该 handler 的 `current_step` 继续 prefill 或 decode；原共享链中较长的上下文保存在快照里，当前分支可以覆盖分叉点之后的槽位。

### 6.4.3　复制粒度与后端差异

常规 LiteRT compiled executor 的 `CloneContext` 只遍历当前活动的输入 KV map，不遍历双缓冲的两套 map；每块缓冲交给 `CopyTensorBuffer` 复制，复制量由缓冲的 `PackedSize()` 决定：

```cpp
// runtime/util/tensor_buffer_util.cc:51-80
LITERT_ASSIGN_OR_RETURN(auto size, tensor_buffer.PackedSize());  // (1)
// ...
std::memcpy(dst_lock_and_addr.second, src_lock_and_addr.second, size);  // (2)
```

`PackedSize()` 是缓冲的完整容量，不按 `current_step` 裁剪有效前缀。按基准模型的形状，宽度为 4096 的一套 K/V 缓冲约为 112 MiB；只要活动输入 map 里有这套完整缓冲，`CloneContext` 就复制这么多字节。CPU 后端和没有 int32 参数张量的 GPU out-of-place 路径满足这一前提。GPU 单缓冲路径则不然：6.3.1 节说明它不创建输入侧缓冲，输入 map 为空，`CloneContext` 遍历不到任何 KV 缓冲，快照里不含 KV 字节；恢复快照时，单缓冲标志为真的执行器也跳过 KV map 的替换。我们据此推断，signature 含参数张量的模型在会话分支后，KV 状态并不随快照保存与恢复；本书没有在真机上验证这条路径的行为。动态形状路径的 `PackedSize()` 对应当时已分配的张量形状，同样要另算。

常规 compiled executor 恢复快照时不复制张量：它把保存的 KV map 移入活动输入 map，接管所有权。文件中另有一个按缓冲复制的 `RestoreKVCacheBuffers`，当前没有调用点。

NPU executor 的 `CloneContext` 扫描 prefill 输入缓冲，只复制名称以 `kv_cache_k_`、`kv_cache_v_` 或 `kv_cache_c_` 开头的项（第三类是部分 NPU 模型额外携带的 cache，本书按名称称为 C cache），每块同样由 `CopyTensorBuffer` 按 `PackedSize()` 完整复制；张量形状由对应模型决定，基准模型的 112 MiB 不能沿用。恢复时它不像常规 executor 那样移动 map，而是检查源、目标缓冲的 `PackedSize()` 相等后，把快照内容 `memcpy` 回固定的输入缓冲。

写时分离不是调用 `CloneContext` 的唯一位置。两个 handler 一旦分属不同的 `SharedProcessedContext`，之后每次在它们之间切换，资源管理器都会先用 `CloneContext` 保存当前执行器状态，再恢复目标状态；仍共享同一个对象的 handler 之间切换时，只交换运行配置与运行状态，不涉及 KV。`Session::Clone`、写时分离与上下文切换是三层不同的操作，不能合并成一次立即发生的 KV 拷贝。

## 6.5　容量案例：公共前缀分出两条会话

用一个条件化案例同时计算容量、双缓冲与写时分离。KV 张量形状取自本书基准模型，存储类型为 INT8，上下文宽度为 4096。执行器假定为 GPU out-of-place 路径，signature 不含 int32 参数张量。该条件与 Gemma 4 E4B 的实际 GPU signature 不同，案例只核算双缓冲路径，计算值不代表实测结果。

一套 K/V 缓冲约为 112 MiB。out-of-place 路径分别创建输入、输出缓冲，因此两套常驻 KV 数据约为 224 MiB。这里只计算 LLM KV 张量，不含权重、激活、采样器、后端工作区和内存分配器开销。

设会话 A 已处理 2048 个 token。应用调用 `A.Clone()` 创建会话 B，然后让 B 从公共前缀末尾继续生成 256 个 token。Clone 只建立共享关系，LLM KV 复制量为 0。B 从末尾续写时也不需要保存 A 的旧状态，因为 B 此时是共享链中最长的分支。

随后应用切回 A。切换时 A、B 仍共享同一个对象，只交换运行状态，不复制 KV。应用不沿 B 的结果续写，而是在第 2048 步加入另一段输入。资源管理器发现共享链的最长位置已经是 2304，而 A 仍在 2048；A 即将覆盖执行器中 2048 之后的槽位，因此触发 `SaveProcessedContextAndSeparateLoadedHandler`。compiled executor 此时复制活动输入 map 的完整 `PackedSize()`，增量约为 112 MiB。

逻辑有效前缀只有 2048 个 token，按 28 KiB/token 计算约为 56 MiB。实际复制量仍约为 112 MiB，因为固定形状缓冲的容量是 4096 个槽位。写时分离按缓冲容量复制，不按有效 token 数截短。图 6-2 把三个阶段和对应的 KV 字节变化列在一起。

<figure>
{{#include figs/fig-6-2.svg}}
<figcaption>图 6-2　条件化 out-of-place 路径的容量变化：Clone 与最长分支续写不额外复制 KV；较短分支改写历史时，compiled executor 才复制一套完整活动输入缓冲。</figcaption>
</figure>

| 阶段 | 共享关系与执行动作 | 新增的 LLM KV 复制量 | 4096 槽位下的容量口径 |
|---|---|---:|---:|
| A 位于第 2048 步 | 执行器持有输入、输出两套 KV 缓冲 | 不适用 | 双缓冲约 224 MiB |
| `A.Clone()` 得到 B | 两个 handler 共享 `SharedProcessedContext` | 0 | 不因 Clone 再分配一套 KV |
| B 从末尾续写到 2304 | B 成为共享链最长分支 | 0 | 仍使用当前执行器缓冲 |
| A 在 2048 后改写 | 保存 B 所在共享链，A 改用新共享对象 | 约 112 MiB | 复制活动输入 map 的完整容量 |
| A、B 此后交替运行 | 已分属不同共享对象：每次切换保存当前上下文、恢复目标上下文 | 每次切换约 112 MiB | 不能只按首次写时分离估算 |

> 表 6-3　表中的 112 MiB 和 224 MiB 只适用于上述张量形状。计算条件还包括 4096 槽位与 GPU out-of-place 路径。Gemma 4 E4B 的实际 GPU 路径不同。两项数字也不是整个进程的内存占用，不能用于 NPU 的 K/V/C cache 集合。

相同计算还能说明 `--max-num-tokens` 如何增加分支成本。若其他条件不变，把固定宽度从 4096 增至 8192，一套 KV 缓冲由约 112 MiB 增至约 224 MiB。双缓冲由约 224 MiB 增至约 448 MiB；一次相同路径的写时分离也由约 112 MiB 增至约 224 MiB。预留宽度同时影响常驻双缓冲与后续快照增量。

容量不能只按“模型最大支持长度”设置。部署侧需要先给出 prompt 上限、单轮输出上限、会话保留策略和允许的分支数，再确定预留宽度。设置过大增加内存与固定形状执行成本；设置过小会让 prefill 越界，或在达到 `max_num_tokens` 时结束 decode。表 6-2 已展示这两类结果。

### 6.5.1　诊断案例：Clone 之后才出现的内存峰值

若部署采用上述 out-of-place 路径，仍不能把分支后的内存增长直接归因于 `Session::Clone`。验证时应把 API 调用和首次改写分开计时。先在 Clone 前后记录 `current_step` 与进程内存，再让新分支从公共前缀末尾续写。最后切回较短分支，在分叉点后输入不同 token。内存增长若出现在最后一步而非 Clone 调用处，才与写时分离路径一致。若模型 signature 含 int32 参数张量（基准模型即如此），快照路径与本案例不同（6.4.3 节），要先确认这一点再套用上述预期。

进程 RSS 不能直接等同于 112 MiB 的理论增量。后端可能延迟分配，内存分配器也可能保留已经释放的页。诊断记录至少要包含以下信息：

- 模型文件及其摘要、后端和 `--max-num-tokens`；
- Clone 时两个会话的 `current_step`；
- 哪个分支先续写，以及发生不同输入的位置；
- 内存采样点位于 Clone 前后、首次续写前后，还是上下文切换前后；
- 是否使用单缓冲路径，以及观测值是 RSS、设备内存还是后端分配统计。

如果只记录“Clone 后内存增加”，无法区分 Clone、写时分离、执行器上下文切换和分配器缓存。上述顺序把四个事件拆开，并允许用源码中的触发条件解释观测位置。

## 6.6　`KVCacheInterface` 的能力边界

`KVCacheInterface` 是 KV cache 的后端抽象，声明了序列化、批量选择与复制等操作：

```cpp
// runtime/executor/kv_cache_interface.h:28-63
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

这些方法都是纯虚函数，接口只规定调用形式。`Serialize` 与 `Load` 表达把 KV cache 转成字节串再载入的目标；`SelectAndCopyFrom` 从较大的 batch 中选出一条，`BroadcastAndCopyFrom` 把单条 cache 复制到较大的 batch；`DeepCopy` 要求返回拥有独立缓冲的新对象。

这个接口当前只有一个实现 `LitertKVCache`，而这个实现没有任何执行器使用：仓库里引用它的只有它自己的源文件和单元测试。6.3 节的 compiled executor 用的是自己的两套缓冲 map，6.4 节的 `CloneContext` 调用的是 `CloneKVCacheBuffers`，NPU executor 也直接调用 `CopyTensorBuffer`，都不经过这个接口。因此本节讨论的是接口的设计目标和它唯一实现的行为，不是实际执行路径上已经具备的能力。

### 6.6.1　`Serialize` 与 `Load` 是尚未实现的接口目标

`LitertKVCache` 对两个序列化方法都返回 `UnimplementedError`：

```cpp
// runtime/executor/litert/kv_cache.h:45-51
absl::StatusOr<std::string> Serialize() const override {
  return absl::UnimplementedError("Not implemented");
}

absl::Status Load(absl::string_view serialized_kv_cache) override {
  return absl::UnimplementedError("Not implemented");
}
```

配套测试 `SerializeNotSupported` 也要求返回未实现错误。加上没有执行器使用这个类这一事实，结论是：当前没有任何路径能把 LiteRT 后端的 KV cache 保存到磁盘或跨进程传递。接口声明表示设计目标，不能当作已提供的运行时能力。

| 操作 | 实际路径 | KV 字节复制量 | 状态或用途 |
|---|---|---|---|
| `Session::Clone` | 新旧 handler 共享 `SharedProcessedContext`；分别持有按值复制的 `RuntimeConfig`、`RuntimeState` | LLM KV 为 0 | 建立可写时分离的会话分支；音频上下文另行克隆 |
| compiled executor 写时分离 | `CloneContext` 复制活动输入 KV map 中每块缓冲的 `PackedSize()` | 该 map 含一套宽度为 4096 的基准模型 K/V 时约 112 MiB；不按有效前缀裁剪；GPU 单缓冲路径的 map 为空 | 较短分支需要截断或改写共享历史时触发；非单缓冲恢复通过移动 map 接管所有权 |
| NPU executor 写时分离 | `CloneContext` 复制 prefill 输入中的 K、V、C cache；恢复时再次按完整缓冲复制 | 取决于对应模型各缓冲的 `PackedSize()` | 与 compiled executor 共用上层 COW 条件，但缓冲集合和恢复方式不同 |
| `RewindToCheckpoint` | 获取会话对应执行器，再恢复 `current_step` 与会话状态 | 检查点记录本身不含 KV；获取执行器时的独立上下文切换可能另有复制 | 回到已保存位置，后续由 refill 覆盖旧槽位 |
| `LitertKVCache::DeepCopy` | 复制 bank 1，并复制可选的 bank 2（该类为不支持原地更新的后端准备的第二套缓冲） | 取决于张量形状与 bank 数量 | 独立接口实现；没有执行器调用 |
| `Serialize` / `Load` | 返回 `UnimplementedError` | 不适用 | 接口目标，LiteRT 后端尚不可用 |

> 表 6-4　会话克隆先共享 LLM 上下文。后续写时分离或独立上下文切换进入 executor `CloneContext` 时，才按具体后端的缓冲集合复制 KV 字节。

### 6.6.2　批量分支的选择与广播

`SelectAndCopyFrom` 与 `BroadcastAndCopyFrom` 提供 batch 之间的选择和广播。`LitertKVCache` 的实现先检查类型与 batch 形状：

```cpp
// runtime/executor/litert/kv_cache.cc:351-357
absl::Status LitertKVCache::BroadcastAndCopyFrom(KVCacheInterface& other) {
  auto other_litert = dynamic_cast<LitertKVCache*>(&other);
  RET_CHECK(other_litert != nullptr) << "Only support LitertKVCache.";
  RET_CHECK(!bank_2_key_cache_buffers_.has_value());               // (1)
  RET_CHECK(!other_litert->bank_2_key_cache_buffers_.has_value()); // (1)
  RET_CHECK_EQ(other_litert->batch_size_, 1);                      // (2)
  RET_CHECK_GT(batch_size_, other_litert->batch_size_);            // (3)
```

代码行 `(2)` 要求源的 batch size 为 1，`(3)` 要求目标 batch 更大；`(1)` 要求源和目标都没有 bank 2，即当前实现只接受单套缓冲的对象。通过检查后，源缓冲被整块复制目标 batch 数那么多次。反方向的 `SelectAndCopyFrom` 要求源 batch 大于目标，按 `batch_index` 乘以目标缓冲大小做指针偏移，再复制一次。这段偏移依赖源码注释写明的布局假设：缓冲按 `[batch * X, ...]` 或 `[1, batch * X, ...]` 排列，同一 batch 的数据连续且不与其他 batch 交错。注释说明该假设对当时各后端的 LLM 模型成立，但它不是接口对任意布局的普遍保证。

## 6.7　检查点只保存逻辑位置

`SaveCheckpoint` 的名称容易让人联想到完整状态快照。`SessionAdvanced` 实际保存的 `CheckpointInfo` 只有两个字段：整数 `step` 和三态枚举 `SessionState`。它不包含 token 数组、KV 张量或采样器缓冲。

保存时，Session 先从 execution manager 读取当前步数，再把 `{current_step, session_state_}` 写入以字符串为键的 map：

```cpp
// runtime/core/session_advanced.cc:461-463
ASSIGN_OR_RETURN(int current_step,
                 execution_manager_lock->GetCurrentStep(*session_info_));
checkpoint_map_[label] = {current_step, session_state_};  // (1)
```

代码行 `(1)` 使用下标赋值：相同 label 再保存一次会覆盖旧条目，而不是建立同名历史栈。应用若需要多个恢复点，应使用不同 label，并明确其生命周期。

`CheckpointInfo` 也不保存 `RuntimeState::rand_gen`。因此，checkpoint 表示逻辑位置和 Session 状态，不是随机采样状态的快照；使用非零温度回退后重新 decode，不能据此保证生成与上次相同。

回退时，代码先查找 label，不存在则返回 `NotFoundError`；找到后恢复 step 与 `SessionState`，删除目标 step 之后的检查点，最后把执行器的 `current_step` 设回目标位置：

```cpp
// runtime/core/session_advanced.cc:475-488
int target_step = it->second.step;    // (1)
session_state_ = it->second.state;
// ...
absl::erase_if(checkpoint_map_, [target_step](const auto& pair) {   // (2)
  return pair.second.step > target_step;
});
// ...
return execution_manager_lock->SetCurrentStep(*session_info_, target_step);   // (3)
```

代码行 `(1)` 取出目标步数并恢复会话状态；`(2)` 的删除条件是 `step > target_step`，位于同一步的其他 label 不会被删除；`(3)` 请求 execution manager 设置步数，步数赋值本身不复制也不清零 KV 张量。

保存时读取步数、回退时设置步数，都要先获取该 Session 对应的执行器；若需要从另一个独立上下文切换过来，资源管理器会按 6.4.3 节的路径保存并恢复上下文，可能复制 KV。因此“检查点不包含 KV”描述的是检查点数据，不能据此把整个 API 调用的复制量一概记为 0。

| 操作 | checkpoint map 的变化 | LLM KV 字节操作（不含上下文切换） |
|---|---|---:|
| 在 step 100 保存 `base` | `base → {100, state}` | 0 |
| 在 step 200 保存 `candidate` | 增加 `candidate → {200, state}` | 0 |
| 再用 `base` 于 step 220 保存 | `base` 被覆盖为 step 220 | 0 |
| 回退到 step 200 的 `candidate` | 恢复逻辑 step；删除 step 大于 200 的条目 | 0 |
| 回退后 refill 新输入 | 从目标位置继续写入 | 由后续 prefill 决定 |

> 表 6-5　检查点数据不含 KV，也不保留同名 label 的多个版本；获取执行器时的上下文切换成本另计。

这个语义影响错误恢复设计。若应用先保存 `safe`，随后在更晚位置再次使用同名 label，第一次恢复点便已丢失。若需要保留树状分支，应使用 `Session::Clone` 管理独立会话，而不是把 checkpoint map 当成持久版本库。若需要跨进程恢复，当前 `Serialize` / `Load` 又尚未实现；应用不能把 checkpoint label 当成可持久化状态。

回退也不会把目标位置之后的 KV 缓冲置零，它只把逻辑位置设回保存点，后续 refill 或 decode 从该处继续写入。6.8 节的 channel 过滤正是利用这一点：先回退到检查点，再用移除 channel 内容后的历史覆盖后续槽位。

## 6.8　按配置过滤 channel 内容

channel 是模型输出里与可见回复并列的字段，源码注释举的例子是 reasoning 内容；部分模型会在其中放不需要保留到后续上下文的内容。`ConversationConfig::filter_channel_content_from_kv_cache` 控制是否把这部分内容从 KV cache 中过滤掉，默认值为 `false`。关闭时，下面的检查点与 refill 流程不会执行。

assistant 消息生成结束时不会立即清除 KV。收到非追加式 user 消息后，`Conversation::SendMessage` 才检查配置和待过滤标记：

```cpp
// runtime/conversation/conversation.cc:460-472
std::vector<InputData> refill_session_inputs;
if (config_.filter_channel_content_from_kv_cache() &&
    IsUserMessage(message) && !is_appending_message_) {
  if (channel_content_since_last_user_message_) {
    ASSIGN_OR_RETURN(refill_session_inputs,
                     RewindAndGetInputDataVector(optional_args));
    channel_content_since_last_user_message_ = false;
  }

  if (refill_session_inputs.empty()) {
    // ...
    RETURN_IF_ERROR(session_->SaveCheckpoint(kChannelContentCheckpoint));
  }
```

第一次符合条件的 user 消息到达时，待过滤标记尚未设置，代码先保存检查点，并记录这条消息在历史中的位置。assistant 完成一条含 channel 字段的消息后，回调只设置待过滤标记：

```cpp
// runtime/conversation/conversation.cc:537-540
if (config_.filter_channel_content_from_kv_cache() &&
    complete_message.contains(kChannelsKey)) {
  channel_content_since_last_user_message_ = true;
}
```

此时 channel 对应的 KV 仍在当前会话中。下一条非追加式 user 消息到达后，上述条件再次成立，`RewindAndGetInputDataVector` 才会执行：会话先回到上次保存的检查点，再根据消息历史重建 refill 输入，重建范围排除刚到达的这条 user 消息。配套单元测试用 channel 内容 `"hmm"` 验证了这一点：refill 输入包含上一条 user 消息和 assistant 的可见内容，不包含 `"hmm"`。

refill 输入不为空时，`Conversation` 先 prefill 清理后的历史，再保存新检查点，之后才 prefill 当前 user 消息并进入 decode。完整时序为：assistant 完成时设置标记；下一条 user 消息到达时 rewind；refill 无 channel 的历史；保存检查点；处理当前 user 消息。底层的回退就是 6.7 节那段 `RewindToCheckpoint`：恢复步数与会话状态，删除之后的检查点，不清零 KV 张量；若获取执行器时需要切换独立上下文，仍可能产生该节所述的复制。refill 随后从目标位置继续写入，覆盖旧 channel 内容占用的槽位。

<figure>
{{#include figs/fig-6-3.svg}}
<figcaption>图 6-3　Clone 先共享公共前缀。较短分支需要改写历史时才复制并分离上下文。channel 过滤通过 rewind 与 refill 覆盖原有槽位。</figcaption>
</figure>

## 小结

KV cache 以容量和带宽换取较少的重复计算。其占用由层数、KV 头维度、序列长度和存储类型共同决定，本书基准模型为 28 KiB/token。固定形状路径中，`--max-num-tokens` 经 magic number 机制成为张量的静态宽度；本章的同 prompt 对照显示，预留宽度从 4096 增至 8192，decode 吞吐从 26.4 降至 21.5 tokens/s，降幅约 19%。

LiteRT-LM 用双缓冲处理部分 GPU 后端的输入输出别名限制，signature 含 int32 参数张量的模型则走单缓冲的原地更新路径。`Session::Clone` 只共享 `SharedProcessedContext`，按值复制运行配置与状态，不复制 KV 字节；较短分支需要截断或改写共享历史时，资源管理器才调用 executor `CloneContext` 完成写时分离，此后两个分支之间每次切换都要再保存一次。

常规 compiled executor 按活动输入 map 中各缓冲的 `PackedSize()` 复制完整容量，宽度 4096 的基准模型 K/V 约 112 MiB；GPU 单缓冲路径的输入 map 为空，快照不含 KV。NPU executor 同样按完整缓冲复制，缓冲集合还可能包含 C cache，恢复时把保存内容复制回固定输入缓冲。`KVCacheInterface` 及其唯一实现 `LitertKVCache` 不在任何执行器的调用链上，`Serialize` 与 `Load` 也只返回未实现错误。检查点只保存逻辑位置；channel 过滤在下一条非追加式 user 消息到达后才回退并 refill 清理后的历史。

---

## 练习与自查

1. 模型参数复算。每 token 按 28 KiB 计。先计算 8192 个 token 的 KV cache 占用；再说明 `--max-num-tokens 32768` 时该模型实际得到的静态宽度是多少（提示：6.2.2 节的目标值规则），并计算对应占用。
2. 成本对比。调用 `Session::Clone` 后，LLM KV 立即复制多少字节？新旧 handler 共享什么、分别复制什么？若 compiled executor 的活动输入 map 含一套宽度为 4096 的基准模型 K/V 缓冲，后续写时分离约复制多少字节？
3. 代码定位。双缓冲交换为什么不复制张量内容？找出 prefill 与 decode 路径执行交换的位置。
4. 参数推演。`--max-num-tokens` 从 4096 增至 8192 时，容量和本章实测 decode 吞吐分别如何变化？
5. 接口辨析。比较 compiled executor 与 NPU executor 的 `CloneContext`。两者选择哪些缓冲，复制粒度如何，恢复方式有何不同？再说明它们与 `LitertKVCache::DeepCopy` 是否属于同一调用链。
6. 时序复述。启用 channel 过滤后，依次说明相关操作。起点是 assistant 输出含 channel 字段。终点是下一条 user 消息开始 decode。

[^ch06-issue-2568]: Yegorsh，[*`--max-num-tokens` unreasonably affects decoding speed*](https://github.com/google-ai-edge/LiteRT-LM/issues/2568)，LiteRT-LM issue #2568，2026-06-13；访问日期：2026-07-18。
[^ch06-llamacpp-kvtype]: ggml-org，[*llama.cpp 源码 common/arg.cpp:2174*](https://github.com/ggml-org/llama.cpp/blob/b9873/common/arg.cpp#L2174)，版本 b9873；访问日期：2026-08-31。
