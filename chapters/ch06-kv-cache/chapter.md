# 第 6 章 KV cache：容量、带宽与会话生命周期

> 本章计算 KV cache 的容量与带宽成本。说明 LiteRT-LM 的双缓冲、会话克隆、检查点回退和序列化能力边界。

第 1 章估算的 25 tokens/s 只计算了权重读取，KV cache、采样和其他算子均未计入。本章补上 KV cache 这部分的成本，并回答第 2 章第 13、14 问。

## 6.1　KV cache 的内存占用估算

注意力机制生成新 token 时，要用该位置的 query 与此前所有 token 的 key/value 计算。不使用缓存时，第 1000 步会重算前 999 个 token 的 key/value。第 1001 步又会重复这项计算。单步开销随上下文长度线性增加，生成整段序列的累计开销呈二次增长。

KV cache 保存已经计算出的 key/value。后续 decode step 可以直接读取这些缓存，用内存容量减少重复计算。

KV cache 的大小沿用 1.3 节的理论公式：

$$ \text{KV 字节} = 2 \times L \times n_{kv} \times d_{head} \times S \times b $$

其中 \\(L\\) 是层数，\\(n_{kv}\\) 是 KV 头数，\\(d_{head}\\) 是每头维度，\\(S\\) 是缓存的 token 数，\\(b\\) 是每个元素的字节数；最前面的 2 表示 key 和 value 各一份。1.3 节的示意参数（\\(L=32\\)、\\(n_{kv}=8\\)、\\(d_{head}=128\\)、FP16）给出 128 KiB/token；真实参数应从模型文件读取，下面直接对本书基准模型代入。

附录 D 记录了 Gemma 4 E4B 的模型检查结果。decode signature 有 48 个 KV 输入张量，即 24 层各一对 K/V。20 层的 K 张量为 `[1, 2, 32003, 256]`，V 张量为 `[1, 2, 256, 32003]`。其余 4 层的最后两维分别为 `[32003, 512]` 与 `[512, 32003]`。数据类型均为 INT8，因此 \\(b=1\\)。转置布局不改变元素数。代入公式，每 token 占用 \\(2 \times 2 \times (20 \times 256 + 4 \times 512) \times 1 = 28672\\) 字节，即 28 KiB。4096 个 token 合计 112 MiB。

该结果小于 1.3 节的 FP16 示意值：该模型采用 GQA，KV cache 的存储类型又是 INT8，同一形状的容量和传输字节数相对 FP16 减半。KV cache 精度、权重精度与激活精度是三个不同的配置维度。

<div class="aside-compare">

llama.cpp 与 LiteRT-LM 在不同阶段确定 KV cache 精度。llama.cpp 提供运行时参数 `--cache-type-k` 和 `--cache-type-v`，部署者可选择 f16、q8_0 等类型。[^ch06-llamacpp-kvtype]Gemma 4 E4B 的 LiteRT-LM 模型把 INT8 KV 类型固定在导出产物中。运行时不提供对应参数。前者保留部署时选择；后者由模型发布者确定模型与 KV 精度的组合。本节不比较两种方案的精度结果。

</div>

内存预算不能只计入权重。以本书基准模型为例，4096 个 token 的 KV cache 为 112 MiB；若 32003 个静态槽位全部预留，则约为 875 MiB。其他模型的形状和精度不同，应按本节公式分别计算。

## 6.2　KV cache 对解码带宽的影响

标准全注意力的每个 decode step 都要读取模型权重和此前位置的 KV cache。它还要写入新 token 的 key/value。上下文越长，这部分数据访问越多。

第 1 章的 25 tokens/s 只包含权重读取，不能单独用它来解释实测吞吐。主基准 CPU 的 24.8 tokens/s 与它接近，但这不构成验证：两者的带宽条件和未计成本都不同。附录 D 记录了同组基准。测试将上下文从 256 增至 4096。CPU decode 从 24.8 降到 20.7 tokens/s；GPU 从 50.6 降到 45.6 tokens/s。这组数据与 KV 访问量随上下文增长的机制一致，但没有分别测出 KV、采样和其他算子的贡献。

### 6.2.1　从容量公式推导理论扫描量

容量公式还能给出注意力侧的理论扫描量。设每个 token 的 K/V 条目合计为 \\(E\\) 字节，当前上下文长度为 \\(S\\)。若标准全注意力的每一层都访问此前全部位置，一次 decode step 扫描的 K/V 工作集约为 \\(E\\times S\\) 字节。decode 吞吐为 \\(T\\) tokens/s 时，对应的理论扫描速率为：

$$ B_{KV,\mathrm{scan}} \approx E \times S \times T $$

这个量既不是 DRAM 带宽实测值，也不是严格的总线流量上界。片上缓存可能减少主存读取，重复加载和中间布局转换则可能增加流量。输出侧也取决于实现：原地更新只需写入新增 K/V；非原地（out-of-place）kernel 还要形成另一份输出 cache。仅凭 host 侧的缓冲绑定无法确定这些操作产生的总线字节数。

若进一步假定每个 decode step 从主存读取一次 \\(W\\) 字节的权重和 \\(E\\times S\\) 字节的 KV，可以继续估算吞吐。计算还忽略写回与其他流量。此时带宽侧的乐观上限为 \\(B_{\mathrm{eff}}/(W+E\\times S)\\)。该式的每项假设都必须随设备和后端重新核对。

基准模型的 \\(E\\) 为 28 KiB/token。暂取 \\(T=25\\) tokens/s，可得到以下理论扫描量：

| 上下文 S | 有效 KV 容量 | 每步理论扫描量 | 25 tokens/s 对应的 KV 理论扫描速率 |
|---:|---:|---:|---:|
| 512 | 14 MiB | 约 14 MiB | 约 0.37 GB/s |
| 4096 | 112 MiB | 约 112 MiB | 约 2.94 GB/s |
| 8192 | 224 MiB | 约 224 MiB | 约 5.87 GB/s |
| 32003 | 约 875 MiB | 约 875 MiB | 约 22.9 GB/s |

> 表 6-1　KV 理论扫描量使用 28 KiB/token 和固定的 25 tokens/s 复算。它不是 Gemma 4 E4B 各层 kernel 的实测流量。带宽按 GB/s（\\(10^9\\) 字节每秒）计算。

上下文从 4096 增至 8192，KV 容量和注意力侧的理论扫描量均翻倍。双缓冲不改变注意力需要访问的已缓存 K/V 集合，但它可能增加输出 cache 的写入量和常驻容量。表 6-1 不把这部分计入扫描速率。

附录 D 识别出的主 decode 模型段 payload 为 2.26 GB。若进一步假定每个 decode step 恰好读取该段一次，则 25 tokens/s 对应的权重读取速率为 56.5 GB/s。4096 上下文的 KV 理论扫描速率为 2.94 GB/s；到 32003 时增至约 22.9 GB/s。`.litertlm` 整文件为 3.66 GB，但其中包含 10 个模型段，不能把整文件大小当作每步读取的权重字节数。

### 6.2.2　`--max-num-tokens` 如何决定预留大小

第 13 问涉及 `LiteRT-LM#2568` 报告的现象：`--max-num-tokens` 会影响 decode 吞吐。[^ch06-issue-2568] 该参数设定 KV cache 可容纳的 token 数。在固定形状路径上，attention mask 屏蔽尚未使用的位置，但输入张量的宽度仍由预留宽度决定。表 6-2 的同 prompt 对照显示，扩大预留宽度会降低吞吐。

未显式指定该参数时（`GetMaxNumTokens()` 返回 0），LiteRT-LM 根据 prompt 长度计算默认值。对应代码：

```cpp
// runtime/engine/engine_settings.cc:293-301
if (main_executor_settings_.GetMaxNumTokens() == 0) {
  // ...
  int max_num_tokens = ((num_prompt_tokens + 1023) / 4096 + 1) * 4096;   // (1)
  if (metadata.max_num_tokens() > 0) {                                   // (2)
    max_num_tokens = metadata.max_num_tokens();
  }
  main_executor_settings_.SetMaxNumTokens(max_num_tokens);
}
```

`(1)` 等价于把 `prompt token 数 + 1024` 向上对齐到 4096 的整数倍。prompt 为 100 个 token 时得到 4096；prompt 为 4000 个 token 时得到 8192。因此，默认预留宽度按 4096 分段变化，而不是随 prompt 长度连续变化。`(2)` 表示模型元数据中的 `max_num_tokens` 可以覆盖这项推导结果。

预留宽度对应 6.1 节公式中的 S。固定形状模型预留 4096 时，KV cache 张量包含 4096 个位置。当前上下文可能只使用其中一部分。增大 `--max-num-tokens` 不会改变 prompt，但会扩大这些张量的静态宽度。

扫描条件为 CPU、`-d 128`，且磁盘缓存已经预热。完整记录见附录 D：

| 预留宽度 | prompt / prefill 分块计划 | decode tokens/s | 比较口径 |
|---|---|---:|---|
| 1024 | prompt 100 / `[128]` | 33.9 | 独立短 prompt 样本，不与下两行计算降幅 |
| 4096 | prompt 256 / `[128]` | 26.4 | 同 prompt 对照基线 |
| 8192 | prompt 256 / `[128]` | 21.5 | 仅放宽预留；相对 4096 下降约 19% |
| 1024 | prompt 256 / `[1024]` | 运行失败 | prefill 报 `dynamic_update_slice` 越界，用于检查容量边界 |

> 表 6-2　`--max-num-tokens` 扫描（Gemma 4 E4B，CPU，各 2 次）。只有 4096 与 8192 两行使用相同 prompt 和 prefill 分块计划；预留宽度加倍后，decode 从 26.4 降至 21.5 tokens/s，下降约 19%。

1024 宽度、prompt 100 的 33.9 tokens/s 来自不同输入，只能作为单独样本。它不能参与 4096 到 8192 的百分比计算。容量设置过小则可能直接失败：表中第 4 行的 prompt 为 256、prefill 分块计划为 `[1024]`，prefill 阶段触发 `dynamic_update_slice` 越界，而不是截断输入。排查 decode 吞吐时，应同时记录 prompt、prefill 分块计划和最终预留宽度。

预留宽度还是解码循环的终止条件之一。`ShouldStop` 判定何时停止解码：

```cpp
// runtime/core/tasks.cc:99-100
} else if (current_step >= max_num_tokens) {
  // Reaching maximum number of kv-cache size.
  return true;
```

`current_step` 表示当前写入位置。达到 `max_num_tokens` 后，KV cache 没有剩余槽位，decode 停止。执行器未提供配置时，`TryGetMaxNumTokens` 使用 `kDefaultMaxNumTokens = 4096`。旁边的 `TODO(b/423364170)` 计划在所有执行器都返回最大 token 数后移除该回退。代码中的两个 4096 作用不同：前者是默认预留宽度的对齐粒度，后者是读取配置失败时的回退上限。

### 6.2.3　固定形状与动态 KV cache

LiteRT-LM 的 KV cache 有固定形状和动态形状两条路径。固定形状预留 S 个槽位，动态形状随实际长度增长。代码根据 KV 张量是否包含动态维度来区分二者，并据此推断上下文宽度。对应实现：

```cpp
// runtime/executor/litert/kv_cache.cc:302-311
LITERT_ASSIGN_OR_RETURN(const SimpleTensor& mask_tensor,
                        signature.InputTensor(mask_input_name));
// ...
auto dims = mask_tensor_type.Layout().Dimensions();
// Expect [1, 1, Sequence, KV Length]
RET_CHECK_EQ(dims.size(), 4);
const bool is_dynamic_kv_cache = k_dynamic_dim.has_value();   // (1)
context_size = is_dynamic_kv_cache ? 1 : dims[3];             // (2)
```

(1) 根据 key 张量是否包含动态维度（`k_dynamic_dim.has_value()`）判断路径。(2) 动态路径把 `context_size` 设为 1；固定路径读取 attention mask 的最后一维 `dims[3]`，即完整的 KV Length。mask 的形状为 `[1, 1, Sequence, KV Length]`。key 与 value 张量的内部布局不同，因此代码从 mask 推断上下文宽度。

固定形状下，`context_size` 等于完整预留宽度 S，而不是当前已写入的长度。`FillAttentionMask` 在每个 decode step 只把当前步之前的位置设为可见：

```cpp
// runtime/executor/litert_compiled_model_executor_utils.cc:363-366
for (int b = 0; b < batch_size; ++b) {
  for (int i = 0; i < steps; ++i) {
    int current_step = start_timestep + i;
    int offset = b * batch_offset + i * channel_size;
    // For current step = n, we fill (n+1) positions for the mask sequence.
```

当前步为 n 时，mask 设置 n+1 个可见位置，其余位置保留屏蔽值。mask 决定数值上哪些位置参与注意力，但不会缩小静态张量形状。源代码能确认这一形状与 mask 语义；具体后端是否跳过部分被屏蔽位置，仍取决于编译结果和 kernel，不能仅由这段 host 代码判定。

表 6-2 的同 prompt 数据表明，当前实验路径的较大预留宽度增加了端到端成本。动态 KV cache 让张量长度随实际上下文增长。动态形状可能减少部分编译期优化机会，相关权衡见第 4 章。

## 6.3　双缓冲

KV cache 每步都要读写，而部分 GPU 后端不允许同一块缓冲同时作为输入和输出。成员声明处的源码注释记录了该约束：

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

`(1)` 和 `(2)` 是两套 KV 缓冲。`(3)` 所示的输入、输出指针不拥有数据，只指向其中一套。构造时，`input_kv_cache_buffers_` 指向 `kv_cache_buffers_1_`，`output_kv_cache_buffers_` 指向 `kv_cache_buffers_2_`。这种读写方式称为 double buffering（双缓冲）。当前 step 从一套读取并向另一套写入，完成后交换两者角色。

模型执行完成后交换输入、输出指针。prefill 路径的代码：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:737-738
if (!gpu_optimized_single_buffer_cache_) {   // (1)
  std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);   // (2)
}
```

`(2)` 只交换指针，不复制缓冲内容。本步写入的输出缓冲成为下一步的输入缓冲，原输入缓冲成为下一步的输出目标。`(1)` 在 `gpu_optimized_single_buffer_cache_` 为真时跳过交换。decode 路径执行同样的操作。

这两个 map 不一定对应两块独立分配。创建执行器时，CPU 路径用 `TensorBuffer::Duplicate()` 把输入缓冲放入输出 map。源码注释将其称为单缓冲。GPU 在 signature 没有 int32 参数张量时分别创建输入和输出缓冲；signature 包含该参数时则进入原地更新路径。因此，不能仅凭成员名称就将 KV 常驻容量固定乘以二。

<figure>
{{#include figs/fig-6-1.svg}}
<figcaption>图 6-1　GPU out-of-place KV 路径分别读写两套缓冲；执行结束后交换输入、输出指针。指针交换本身不复制张量数据。</figcaption>
</figure>

### 6.3.1　单缓冲路径的额外代价

部分模型 signature 支持同缓冲原地更新（in-place update）。GPU 创建路径在 signature 含 int32 参数张量时不再创建独立的输入 KV 缓冲；CPU 另行通过 `Duplicate()` 复用缓冲。执行阶段还需要提供本步的写入区间。

启用的判据是模型 signature 里有没有一个 int32 参数张量：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:428-429
if (signatures_.input_int32_param.has_value()) {
  gpu_optimized_single_buffer_cache_ = true;
```

模型 signature 包含该参数张量时，执行器启用原地更新，并由前述条件跳过指针交换。Gemma 4 E4B 的记录中存在 `param_tensor[1,1,1,7]`〔基准 D〕，所以不能把它的 GPU 路径按两块独立 KV 缓冲估算。

同一缓冲既读又写时，kernel 需要知道本步更新的槽位区间。单缓冲路径因此在每次 prefill 和 decode 前填充参数张量。两个调用位置：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:673-677
// runtime/executor/llm_litert_compiled_model_executor.cc:904-907
if (gpu_optimized_single_buffer_cache_) {
  LITERT_RETURN_IF_ERROR(signatures_.input_int32_param.has_value());
  RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
      prefill_input_buffers[signatures_.input_int32_param.value()],
      start_step, ids.size()));
}
```

`FillSingleBufferCacheParamTensor` 写入以下三个 int32 参数：

```cpp
// runtime/executor/litert_compiled_model_executor_utils.cc:332-335
int end_index = start_index + update_length;
int32_t params[] = {start_index, end_index, end_index};   // (1)
LITERT_RETURN_IF_ERROR(sizeof(params) <= packed_size);
std::memcpy(param_tensor_lock_and_addr.second, params, sizeof(params));
```

`start_index` 和 `end_index` 标明更新区间。源码注释说明前两个参数供 `add_values_to_cache` kernel 使用。第三个参数供 `runtime_batched_matmul` kernel 检查 channel 结束位置。辅助函数先清零参数缓冲，再复制这三个值。双缓冲通过两套缓冲和指针交换避免原地更新；单缓冲减少一份 KV 内存，但要求模型 signature、host 代码和 kernel 共同处理更新区间。

对 GPU 而言，out-of-place 路径需要独立的输入、输出缓冲，不需要原地更新参数；参数化的 in-place 路径只保留一套 KV 数据，但增加更新区间的接口与执行约束。CPU 的缓冲复用由 `Duplicate()` 路径处理，不能直接套用 GPU 的两套分配口径。

## 6.4　`SharedProcessedContext` 与会话克隆

`SessionInterface::Clone` 的接口语义是：新会话继承调用时的设置与上下文。接口语义不等于立即深拷贝 KV cache。实际实现中，`SessionAdvanced::Clone` 先登记克隆任务并等待任务完成，异步路径在 `CloneAsyncLocked` 中注册新会话和任务依赖。任务进入 `ThreadedExecutionManager` 后，在 `AddCloneSessionTask` 中调用 `ResourceManager::CloneContextHandler`。

### 6.4.1　Clone 调用建立共享关系

`ContextHandler` 把 LLM 状态分成两部分。`SharedProcessedContext` 记录实际 `ProcessedContext` 的共享所有权。多个 handler 借此实现写时复制（copy-on-write，COW）。handler 活动时，实际对象会移入执行器。共享对象仍标识这些 handler 属于同一条上下文链。每个 handler 分别持有自己的 `RuntimeConfig` 与 `RuntimeState`。`SharedProcessedContext` 还记录共享它的 handler 链。

`CloneContextHandler` 直接复用原 handler 的 `shared_ptr<SharedProcessedContext>`，并为运行配置与状态创建新对象：

```cpp
// runtime/framework/resource_management/resource_manager.cc:615-648
RuntimeConfig runtime_config;
RuntimeState runtime_state;
// ...
auto processed_context = llm_context_handler->shared_processed_context();  // (1)
// ...
return ContextHandler::Bundle(
    processed_context, std::make_unique<RuntimeConfig>(runtime_config),
    std::make_unique<RuntimeState>(runtime_state), std::move(audio_context));  // (2)
```

(1) 让新旧 handler 指向同一个 `SharedProcessedContext`。两者共享已处理 token 与 KV cache 的逻辑上下文。(2) 按值复制 `RuntimeConfig` 和 `RuntimeState`，再分别放入新的 `unique_ptr`。两个会话从相同 `current_step` 开始。`Session::Clone` 本身没有调用 LLM 执行器的 `CloneContext`，也没有复制 KV 字节。`RuntimeState::rand_gen` 是 `shared_ptr`。复制 `RuntimeState` 会复制这个智能指针，不会复制随机数生成器对象。若会话含音频上下文，`CloneContextHandler` 会另行克隆音频状态。这不属于本节的 LLM KV cache。

### 6.4.2　写时分离的触发条件

共享状态允许第一个分支从公共前缀末尾继续执行，而不先复制 KV。需要在较早位置改写共享历史时，资源管理器才分离上下文。prefill 路径先比较 `ProcessedTokens::TokenCount()` 与当前步数。若当前步已经位于已处理序列末尾，便直接执行，。若当前步较早，代码会移除仍然匹配的 token。仍有新输入需要处理时，代码再比较共享链中最长 handler 的步数。两处实现：

```cpp
// runtime/framework/resource_management/resource_manager.cc:228-239
// runtime/framework/resource_management/resource_manager.cc:250-260
// runtime/framework/resource_management/resource_manager.cc:316-331
ASSIGN_OR_RETURN(
    int largest_time_step,
    current_handler_->shared_processed_context()->LongestHandlerTimeStep(
        *llm_executor_));
if (largest_time_step != current_step) {
  RETURN_IF_ERROR(SaveProcessedContextAndSeparateLoadedHandler(
      current_handler_, llm_executor_));  // (1)
}
```

`LongestHandlerTimeStep` 遍历共享链，并从 handler 或当前执行器读取各自的 `current_step`。当当前 handler 不是最长分支时，(1) 才触发写时分离。decode 也在准备截断已处理 token 时执行同一检查。复制时机取决于分支执行顺序和步数。从末尾续写的分支不触发这次复制；较短分支需要分叉或截断时才触发。

`SaveProcessedContextAndSeparateLoadedHandler` 调用执行器的 `CloneContext`。该函数把当前已加载的 `ProcessedContext` 保存回原共享对象，再让当前 handler 改用新的共享对象：

```cpp
// runtime/framework/resource_management/resource_manager.cc:69-93
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

(1) 生成独立的执行器快照。(2) 把即将改写历史的 handler 从原共享链移出。执行器回到该 handler 的 `current_step` 后继续 prefill 或 decode。原共享链中的较长上下文得以保留，当前分支可以覆盖分叉点之后的槽位。

### 6.4.3　复制粒度与后端差异

常规 LiteRT compiled executor 的 `CloneContext` 调用 `CloneKVCacheBuffers`。后者只遍历当前的 `input_kv_cache_buffers_`，不遍历双缓冲的两套 map。每个缓冲都传递给 `CopyTensorBuffer`。两处调用。底层复制函数读取 `PackedSize()`，分配同样大小的目标缓冲。该函数按同一字节数执行 `memcpy`：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:1243-1250
// runtime/executor/llm_litert_compiled_model_executor.cc:1288-1302
// runtime/util/tensor_buffer_util.cc:47-82
LITERT_ASSIGN_OR_RETURN(auto size, tensor_buffer.PackedSize());  // (1)
// ...
std::memcpy(dst_lock_and_addr.second, src_lock_and_addr.second, size);  // (2)
```

复制量由缓冲的当前完整容量决定，不按 `current_step` 裁剪有效前缀。按本书基准模型的形状计算，宽度为 4096 的一套 K/V 缓冲约为 112 MiB。活动输入 map 含有这套完整缓冲时，`CloneContext` 才会复制相应字节。CPU 路径和没有 int32 参数张量的 GPU out-of-place 路径满足这一前提。优化后的 GPU in-place 创建路径会跳过独立输入缓冲的创建，不能直接套用 112 MiB。这个数字也不是 `Session::Clone` 的固定成本。动态形状路径的 `PackedSize()` 对应当时已经分配的张量形状，同样需要另算。

常规 compiled executor 的非单缓冲路径需要恢复快照。执行器把保存的 KV map 移入活动输入 map，不再复制张量字节。单缓冲路径会跳过这次 map 替换。文件中另有 `RestoreKVCacheBuffers`，但 v0.13.1 没有调用点。

NPU executor 的 `CloneContext` 扫描 prefill 输入缓冲。它只复制名称以 `kv_cache_k_`、`kv_cache_v_` 或 `kv_cache_c_` 开头的项。每个匹配缓冲仍由 `CopyTensorBuffer` 按 `PackedSize()` 完整复制。NPU 可能包含额外的 C cache，张量形状也由对应模型决定。基准模型的 112 MiB 不能沿用到这条路径。恢复时，NPU executor 检查源、目标的 `PackedSize()` 相等，再按完整 `src_size` 执行一次 `memcpy`。这与常规 compiled executor 的所有权移动路径不同。

写时分离不是调用 `CloneContext` 的唯一位置。两个 handler 已经拥有不同的 `SharedProcessedContext` 后，资源管理器还要处理上下文切换。切换活动 handler 时，它先用 `CloneContext` 保存当前执行器状态，再恢复目标状态。`Session::Clone`、写时分离与执行器快照是三层不同操作，不能把它们合并成一次立即发生的 KV 拷贝。

## 6.5　容量案例：公共前缀分出两条会话

用一个条件化案例同时计算容量、双缓冲与写时分离。KV 张量形状取自本书基准模型，存储类型为 INT8，上下文宽度为 4096。执行器假定为 GPU out-of-place 路径，signature 不含 int32 参数张量。该条件与 Gemma 4 E4B 的实际 GPU signature 不同。案例只核算双缓冲路径，计算值不代表实测结果。

一套 K/V 缓冲约为 112 MiB。out-of-place 路径分别创建输入、输出缓冲，因此两套常驻 KV 数据约为 224 MiB。这里只计算 LLM KV 张量，不含权重、激活、采样器、后端工作区和内存分配器开销。

设会话 A 已处理 2048 个 token。应用调用 `A.Clone()` 创建会话 B，然后让 B 从公共前缀末尾继续生成 256 个 token。Clone 只建立共享关系，LLM KV 搬运量为 0。B 从末尾续写时也不需要保存 A 的旧状态，因为 B 此时是共享链中最长的分支。

随后调度器切回 A。应用不沿 B 的结果续写，而是在第 2048 步加入另一段输入。资源管理器发现共享链的最长位置已经是 2304，而 A 仍在 2048。A 即将覆盖公共执行器中 2048 之后的槽位，因此触发 `SaveProcessedContextAndSeparateLoadedHandler`。compiled executor 此时复制活动输入 map 的完整 `PackedSize()`，增量约为 112 MiB。

逻辑有效前缀只有 2048 个 token，按 28 KiB/token 计算约为 56 MiB。实际复制量仍约为 112 MiB，因为固定形状缓冲的容量是 4096 个槽位。写时分离按缓冲容量复制，不按有效 token 数截短。图 6-2 把三个阶段和对应的 KV 字节变化列在一起。

<figure>
{{#include figs/fig-6-2.svg}}
<figcaption>图 6-2　条件化 out-of-place 路径的容量变化：Clone 与最长分支续写不额外复制 KV；较短分支改写历史时，compiled executor 才复制一套完整活动输入缓冲。</figcaption>
</figure>

| 阶段 | 共享关系与执行动作 | 新增的 LLM KV 搬运量 | 4096 槽位下的容量口径 |
|---|---|---:|---:|
| A 位于第 2048 步 | 执行器持有输入、输出两套 KV 缓冲 | 不适用 | 双缓冲约 224 MiB |
| `A.Clone()` 得到 B | 两个 handler 共享 `SharedProcessedContext` | 0 | 不因 Clone 再分配一套 KV |
| B 从末尾续写到 2304 | B 成为共享链最长分支 | 0 | 仍使用当前执行器缓冲 |
| A 在 2048 后改写 | 保存 B 所在共享链，A 改用新共享对象 | 约 112 MiB | 复制活动输入 map 的完整容量 |
| A、B 此后交替运行 | 保存当前上下文并恢复目标上下文 | 取决于后端与切换次数 | 不能只按首次写时分离估算 |

> 表 6-3　表中的 112 MiB 和 224 MiB 只适用于上述张量形状。计算条件还包括 4096 槽位与 GPU out-of-place 路径。Gemma 4 E4B 的实际 GPU 路径不同。两项数字也不是整个进程的内存占用，不能用于 NPU 的 K/V/C cache 集合。

相同计算还能说明 `--max-num-tokens` 如何放大分支成本。若其他条件不变，把固定宽度从 4096 增至 8192，一套 KV 缓冲由约 112 MiB 增至约 224 MiB。双缓冲由约 224 MiB 增至约 448 MiB；一次相同路径的写时分离也由约 112 MiB 增至约 224 MiB。预留宽度同时影响常驻双缓冲与后续快照增量。

容量不能只按“模型最大支持长度”设置。部署侧需要先给出 prompt 上限、单轮输出上限、会话保留策略和允许的分支数，再确定预留宽度。设置过大增加内存与固定形状执行成本；设置过小会让 prefill 越界，或在达到 `max_num_tokens` 时结束 decode。表 6-2 已展示这两类结果。

### 6.5.1　诊断案例：Clone 之后才出现的内存峰值

若部署采用上述 out-of-place 路径，仍不能把分支后的内存增长直接归因于 `Session::Clone`。验证时应把 API 调用和首次改写分开计时。先在 Clone 前后记录 `current_step` 与进程内存，再让新分支从公共前缀末尾续写。最后切回较短分支，在分叉点后输入不同 token。内存增长若出现在最后一步而非 Clone 调用处，才与写时分离路径一致。

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
// runtime/executor/kv_cache_interface.h:28-62
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

这些方法都是纯虚函数，接口只规定调用形式，能力是否可用取决于具体实现。`Serialize` 与 `Load` 表达把 KV cache 转换为字节串并重新载入的接口目标。`SelectAndCopyFrom` 从较大的 batch 中选择一条，`BroadcastAndCopyFrom` 把单条 cache 复制到较大的 batch。`DeepCopy` 则要求返回一个拥有独立 KV 缓冲的新对象。

LiteRT 实现中的 `LitertKVCache::DeepCopy` 会复制 bank 1 的 key/value 缓冲。若 bank 2 存在，它也会复制 bank 2 的两组缓冲，并构造新的 `LitertKVCache`。这是接口层的一条独立实现路径。6.4 节的 compiled executor 使用另一套缓冲类型和 `CloneKVCacheBuffers`，没有调用该方法。NPU executor 也直接调用 `CopyTensorBuffer`。

### 6.6.1　`Serialize` 与 `Load` 是尚未实现的接口目标

v0.13.1 的 `LitertKVCache` 对两个序列化方法都返回 `UnimplementedError`：

```cpp
// runtime/executor/litert/kv_cache.h:45-51
absl::StatusOr<std::string> Serialize() const override {
  return absl::UnimplementedError("Not implemented");
}

absl::Status Load(absl::string_view serialized_kv_cache) override {
  return absl::UnimplementedError("Not implemented");
}
```

配套测试 `SerializeNotSupported` 也要求返回未实现错误。因此，本版本不能通过这两个方法把 LiteRT KV cache 保存到磁盘或跨进程传递。接口声明表示设计目标，不能当作已提供的运行时能力。

| 操作 | v0.13.1 的实际路径 | KV 字节搬运量 | 状态或用途 |
|---|---|---|---|
| `Session::Clone` | 新旧 handler 共享 `SharedProcessedContext`；分别持有按值复制的 `RuntimeConfig`、`RuntimeState` | LLM KV 为 0 | 建立可写时分离的会话分支；音频上下文另行克隆 |
| compiled executor 写时分离 | `CloneContext` 复制活动输入 KV map 中每块缓冲的 `PackedSize()` | 该 map 含一套宽度为 4096 的基准模型 K/V 时约 112 MiB；不按有效前缀裁剪 | 较短分支需要截断或改写共享历史时触发；非单缓冲恢复通过移动 map 接管所有权 |
| NPU executor 写时分离 | `CloneContext` 复制 prefill 输入中的 K、V、C cache；恢复时再次按完整缓冲复制 | 取决于对应模型各缓冲的 `PackedSize()` | 与 compiled executor 共用上层 COW 条件，但缓冲集合和恢复方式不同 |
| `RewindToCheckpoint` | 恢复 `current_step` 与会话状态 | 0；不复制 KV 张量 | 回到已保存位置，后续由 refill 覆盖旧槽位 |
| `LitertKVCache::DeepCopy` | 复制 bank 1，并复制可选的 bank 2 | 取决于张量形状与 bank 数量 | 独立接口实现；上述 executor `CloneContext` 不调用 |
| `Serialize` / `Load` | 返回 `UnimplementedError` | 不适用 | 接口目标，LiteRT 后端尚不可用 |

> 表 6-4　会话克隆先共享 LLM 上下文。后续写时分离或独立上下文切换进入 executor `CloneContext` 时，才按具体后端的缓冲集合搬运 KV 字节。

### 6.6.2　批量分支的选择与广播

`SelectAndCopyFrom` 与 `BroadcastAndCopyFrom` 提供 batch 之间的选择和广播。LiteRT 实现先检查源、目标类型和 batch 形状：

```cpp
// runtime/executor/litert/kv_cache.cc:351-362
absl::Status LitertKVCache::BroadcastAndCopyFrom(KVCacheInterface& other) {
  auto other_litert = dynamic_cast<LitertKVCache*>(&other);
  RET_CHECK(other_litert != nullptr) << "Only support LitertKVCache.";
  RET_CHECK(!bank_2_key_cache_buffers_.has_value());               // (1)
  RET_CHECK(!other_litert->bank_2_key_cache_buffers_.has_value()); // (1)
  RET_CHECK_EQ(other_litert->batch_size_, 1);                      // (2)
  RET_CHECK_GT(batch_size_, other_litert->batch_size_);            // (3)
```

`(2)` 要求源的 batch size 为 1，`(3)` 要求目标 batch 更大。`(1)` 要求源和目标都没有 bank 2；换言之，当前实现不接受分配了第二组缓冲的 `LitertKVCache`。通过检查后，`BroadcastAndCopyBuffer` 把源缓冲复制 `dst_batch_size` 次：

```cpp
// runtime/executor/litert/kv_cache.cc:179-198
for (int i = 0; i < dst_batch_size; ++i) {
  memcpy(dst_buffer_ptr, src_buffer_ptr, src_buffer_size);
  dst_buffer_ptr += src_buffer_size;
}
```

反方向的 `SelectAndCopyFrom` 要求源 batch 大于目标，并检查 `batch_index` 范围。`SelectAndCopyBuffer` 通过指针偏移定位源 batch 中的一段连续数据，再执行一次 `memcpy`：

```cpp
// runtime/executor/litert/kv_cache.cc:322-334
// runtime/executor/litert/kv_cache.cc:156-176
src_buffer_ptr += batch_index * dst_buffer_size;
memcpy(dst_buffer_ptr, src_buffer_ptr, dst_buffer_size);
```

这段偏移依赖源码写出的布局假设。缓冲按 `[batch * X, ...]` 或 `[1, batch * X, ...]` 排列，同一 batch 的数据连续且不与其他 batch 交错。该注释说明此假设适用于当时各后端的 LLM 模型，但它不是接口对任意布局的普遍保证。

## 6.7　检查点只保存逻辑位置

`SaveCheckpoint` 的名称容易让人联想到完整状态快照。`SessionAdvanced` 实际保存的 `CheckpointInfo` 只有两个字段：整数 `step` 和三态枚举 `SessionState`。它不包含 token 数组、KV 张量或采样器缓冲。

保存时，Session 先从 execution manager 读取当前步数。随后把 `{current_step, session_state_}` 写入以字符串为键的 map：

```cpp
// runtime/core/session_advanced.cc:455-464
ASSIGN_OR_RETURN(int current_step,
                 execution_manager_lock->GetCurrentStep(*session_info_));
checkpoint_map_[label] = {current_step, session_state_};  // (1)
```

`(1)` 使用下标赋值。相同 label 再保存一次会覆盖旧条目，而不是建立同名历史栈。应用若需要多个恢复点，应使用不同 label，并明确其生命周期。

`CheckpointInfo` 也不保存 `RuntimeState::rand_gen`。因此，checkpoint 表示逻辑位置和 Session 状态，不是随机采样状态的快照。使用非零温度回退后重新 decode，不能据此保证生成与上次相同。

回退时，代码先查找 label。不存在则返回 `NotFoundError`。找到后恢复 step 与 `SessionState`，并删除目标 step 之后的检查点。删除条件是 `pair.second.step > target_step`。位于同一步的其他 label 不会被删除。

| 操作 | checkpoint map 的变化 | LLM KV 字节操作 |
|---|---|---:|
| 在 step 100 保存 `base` | `base → {100, state}` | 0 |
| 在 step 200 保存 `candidate` | 增加 `candidate → {200, state}` | 0 |
| 再用 `base` 于 step 220 保存 | `base` 被覆盖为 step 220 | 0 |
| 回退到 step 200 的 `candidate` | 恢复逻辑 step；删除 step 大于 200 的条目 | 0 |
| 回退后 refill 新输入 | 从目标位置继续写入 | 由后续 prefill 决定 |

> 表 6-5　检查点是 Session 层的逻辑书签。它既不复制 KV，也不保留同名 label 的多个版本。

这个语义影响错误恢复设计。若应用先保存 `safe`，随后在更晚位置再次使用同名 label，第一次恢复点便已丢失。若需要保留树状分支，应使用 `Session::Clone` 管理独立会话，而不是把 checkpoint map 当成持久版本库。若需要跨进程恢复，当前 `Serialize` / `Load` 又尚未实现；应用不能把 checkpoint label 当成可持久化状态。

回退也不会将目标位置之后的 KV 缓冲置零。它只把逻辑位置设回保存点；后续 refill 或 decode 从该处继续写入。channel 过滤正是利用这一点：先回退到检查点，再用移除 channel 内容后的历史覆盖后续槽位。

## 6.8　按配置过滤 channel 内容

部分模型会在输出消息的 channel 字段中携带不需要保留到后续上下文的内容。`ConversationConfig::filter_channel_content_from_kv_cache` 控制是否从 KV cache 中过滤这部分内容，默认值为 `false`。关闭时，下面的检查点与 refill 流程不会执行。

assistant 消息生成结束时不会立即清除 KV。收到非追加式 user 消息后，`Conversation::SendMessage` 才检查配置和待过滤标记：

```cpp
// runtime/conversation/conversation.cc:457-475
std::vector<InputData> refill_session_inputs;
if (config_.filter_channel_content_from_kv_cache() &&
    IsUserMessage(message) && !is_appending_message_) {
  if (channel_content_since_last_user_message_) {
    ASSIGN_OR_RETURN(refill_session_inputs,
                     RewindAndGetInputDataVector(optional_args));
    channel_content_since_last_user_message_ = false;
  }

  if (refill_session_inputs.empty()) {
    RETURN_IF_ERROR(session_->SaveCheckpoint(kChannelContentCheckpoint));
  }
```

第一次符合条件的 user 消息到达时，待过滤标记尚未设置。代码先保存检查点，并记录这条消息在历史中的位置。assistant 完成一条含 channel 字段的消息后，回调只设置 `channel_content_since_last_user_message_`：

```cpp
// runtime/conversation/conversation.cc:529-540
if (config_.filter_channel_content_from_kv_cache() &&
    complete_message.contains(kChannelsKey)) {
  channel_content_since_last_user_message_ = true;
}
```

此时 channel 对应的 KV 仍在当前会话中。下一条非追加式 user 消息到达后，上述条件再次成立。`RewindAndGetInputDataVector` 此时才会执行。会话先回到上次保存的检查点，再根据消息历史重建 refill 输入。重建范围排除了刚到达的 user 消息。测试中的 refill 包含上一条 user 消息和 assistant 的可见内容。channel 中的 `"hmm"` 未被包含。

refill 输入不为空时，`Conversation` 先 prefill 清理后的历史，再保存新检查点。之后才 prefill 当前 user 消息并进入 decode。完整时序为：assistant 完成时设置标记；下一条 user 消息到达时 rewind；refill 无 channel 的历史；保存检查点；处理当前 user 消息。

底层的 `RewindToCheckpoint` 恢复检查点记录的步数和会话状态，并删除目标位置之后的检查点：

```cpp
// runtime/core/session_advanced.cc:467-480
int target_step = it->second.step;    // (1)
session_state_ = it->second.state;
absl::erase_if(checkpoint_map_, [target_step](const auto& pair) {   // (2)
  return pair.second.step > target_step;
});
// ...
return execution_manager_lock->SetCurrentStep(*session_info_, target_step);   // (3)
```

`SaveCheckpoint` 保存 `{current_step, session_state_}`。回退时，(1) 取出目标步数并恢复会话状态；(2) 删除目标位置之后的检查点；(3) 把执行器的 `current_step` 设回目标位置。该过程不复制或清零 KV 张量。refill 从目标位置继续写入，覆盖旧 channel 内容占用的槽位。

<figure>
{{#include figs/fig-6-3.svg}}
<figcaption>图 6-3　Clone 先共享公共前缀。较短分支需要改写历史时才复制并分离上下文。channel 过滤通过 rewind 与 refill 覆盖原有槽位。</figcaption>
</figure>

## 小结

KV cache 以容量和带宽换取较少的重复计算。其占用由层数、KV 头维度、序列长度和存储类型共同决定；固定形状路径中的 `--max-num-tokens` 还会改变预留张量的宽度。本章的同 prompt 对照显示：预留宽度从 4096 增至 8192，decode 吞吐从 26.4 降至 21.5 tokens/s，降幅约 19%。

LiteRT-LM 用双缓冲处理部分 GPU 后端的输入输出别名限制。支持原地更新的模型还可以使用单缓冲路径。`Session::Clone` 初始共享 `SharedProcessedContext`，并按值复制运行配置与状态对象；它不立即搬运 LLM KV 字节。较短分支需要截断或改写共享历史时，资源管理器才调用 executor `CloneContext` 完成写时分离。

常规 compiled executor 按活动输入 map 中各缓冲的 `PackedSize()` 复制完整容量。该 map 含有一套宽度为 4096 的基准模型 K/V 缓冲时，复制量约为 112 MiB；执行还必须实际进入快照路径。优化后的 GPU in-place 路径不能直接套用这个数字。NPU executor 同样按完整缓冲复制，但缓冲集合还可能包含 C cache。恢复快照时，它还会把保存内容复制回固定输入缓冲。`LitertKVCache::DeepCopy` 是另一条接口实现，不能与 executor `CloneContext` 视为同一调用链。

`Serialize` 与 `Load` 仅有接口声明，LiteRT 后端在 v0.13.1 返回未实现错误。assistant 消息结束时，channel 过滤不会立即删除 KV 字节。下一条非追加式 user 消息到达后，系统才回退并 refill 清理后的历史。

---

## 练习与自查

1. 模型参数复算。每 token 按 28 KiB 计。先计算 8192 个 token 的 KV cache 占用；再计算 32003 个静态槽位全部预留时的占用。
2. 成本对比。调用 `Session::Clone` 后，LLM KV 立即搬运多少字节？新旧 handler 共享什么、分别复制什么？若 compiled executor 的活动输入 map 含一套宽度为 4096 的基准模型 K/V 缓冲，后续写时分离约复制多少字节？
3. 代码定位。双缓冲交换为什么不搬运张量内容？找出 prefill 与 decode 路径执行交换的位置。
4. 参数推演。`--max-num-tokens` 从 4096 增至 8192 时，容量和本章实测 decode 吞吐分别如何变化？
5. 接口辨析。比较 compiled executor 与 NPU executor 的 `CloneContext`。两者选择哪些缓冲，复制粒度如何，恢复方式有何不同？再说明它们与 `LitertKVCache::DeepCopy` 是否属于同一调用链。
6. 时序复述。启用 channel 过滤后，依次说明相关操作。起点是 assistant 输出含 channel 字段。终点是下一条 user 消息开始 decode。

[^ch06-issue-2568]: Yegorsh，[*`--max-num-tokens` unreasonably affects decoding speed*](https://github.com/google-ai-edge/LiteRT-LM/issues/2568)，LiteRT-LM issue #2568，2026-06-13；访问日期：2026-07-18。
[^ch06-llamacpp-kvtype]: ggml-org，[*llama.cpp 源码 common/arg.cpp:2174*](https://github.com/ggml-org/llama.cpp/blob/b9873/common/arg.cpp#L2174)，版本 b9873；访问日期：2026-08-31。
