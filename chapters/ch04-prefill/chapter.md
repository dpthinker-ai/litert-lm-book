# 第 4 章 Prefill：并行处理提示词

> 本章说明 prefill 的算术强度和固定 signature 对吞吐测量的影响，比较静态与动态路径的权衡。同时明确任务调度、回调与取消的边界。

上一章末尾，输入文本已经变成一串 token id。`Prefill` 把整段提示词成批送入模型前向，注意力产生的 K/V 写入 KV cache；prefill 耗时是 TTFT 的主要组成之一。本章回答三件事：prefill 快在哪、固定形状会怎样影响吞吐读数（4.1、4.2 节），执行中的 prefill 能不能取消（4.3 节），以及任务在哪个线程上如何排队执行（4.4 节）。

## 4.1　prefill 的资源约束

prefill 比 decode 快一个量级，原因在于一次前向同时处理整段提示词：读一遍权重，服务所有位置的计算，算术强度因此通常高于一次只处理一个 token 的 decode step（Roofline 分析见 1.4 节与 2.4 节）。不过“通常”不是“必然”：序列长度、模型结构和后端都可能使瓶颈变为内存访问或 host 侧开销，确认是否受算力约束，要靠目标设备的性能计数器或受控实验。

差距有多大，主矩阵（条件见 2.2 节）给了对照：上下文为 1024 时，gpu 后端 prefill 999.1 tokens/s、decode 50.6 tokens/s，相差约 20 倍；cpu 后端分别为 259.2 与 24.7 tokens/s，相差约 10 倍〔基准 D〕。不过这组数据是端到端吞吐，不能单独证明两段分别受算力和带宽约束：测量路径从任务层入口进入 executor，包含模型执行、固定形状填充和 host 侧开销。

`Tasks::Prefill` 的编排入口依次完成最大长度校验、等待参数设置、计时和 executor 调用：

```cpp
// runtime/core/tasks.cc:544
  auto num_tokens = token_id_tensor_type.Layout().Dimensions().back();
  if (num_tokens >= max_num_tokens) {  // (1)
    return absl::InvalidArgumentError(absl::StrCat(
        "Input token ids are too long. Exceeding the maximum number of tokens "
        "allowed: ",
        num_tokens, " >= ", max_num_tokens));
  }
// ...
  ExecutorPrefillParams params;
  // Wait for prefill to complete if benchmark mode is enabled.
  params.SetWaitForCompletion(wait_for_completion | benchmark_info.has_value());  // (2)
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimePrefillTurnStart());  // (3)
  }
  ABSL_RETURN_IF_ERROR(executor.Prefill(inputs, params));  // (4)
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimePrefillTurnEnd(num_token_ids));
```

代码行 `(1)` 越界判断用 `>=` 而非 `>`：上下文窗口要留一个位置给 pending token（每次 prefill 都会留下最后一个 token 待下一步处理，机制见 4.2 节末），所以 token 数必须严格小于 `max_num_tokens`。`(2)` `wait_for_completion` 与 benchmark 开关做按位或。启用基准模式后强制同步等待，避免在异步执行完成前结束计时。`(3)` 的 `TimePrefillTurnStart` 与后面的 `TimePrefillTurnEnd` 包围 `(4)` 的 `executor.Prefill` 调用。任务层负责校验、计时和编排；executor 负责形状选择、模型执行与 KV cache 更新。

这两个探针只记录起止时间。`TimePrefillTurnStart` 以 `prefill:<turn_index>` 为键保存 `absl::Now()`。`TimePrefillTurnEnd` 再取一次时间并求差，把结果和 token 数存入 `prefill_turns_`。测量范围覆盖整个 `executor.Prefill`，包括工作组循环、掩码填充、embedding 装配与 KV 缓冲交换。附录 D 的 prefill 吞吐由这对探针产生。

### 4.1.1　固定 signature 如何影响吞吐曲线

以下长度扫描使用 v0.13.1 的工作组策略，不能视为当前版本的分块或性能结果。固定 signature 会让吞吐曲线产生误导：吞吐的分子是真实 token 数，分母却是选中入口的执行时间，两者并不同步变化。本书在同一台 Mac 的 cpu 后端扫描了 100 至 4000 token 的输入长度验证这一点（热磁盘缓存，decode 长度 32，每点运行一次，完整结果见附录 D 第七节）；基准模型只有 `prefill_128` 和 `prefill_1024` 两个固定入口。

100 token 由 `prefill_128` 处理。按 token 数除以吞吐复算，prefill 墙钟时间约为 2.07 s。250、500 和 1000 token 都只调用一次 `prefill_1024`。三者的 prefill 墙钟时间分别约为 3.91、3.90 和 3.89 s，近似恒定；吞吐依次为 64.0、128.4 和 257.0 tokens/s〔基准 D〕。吞吐近似按真实 token 数成比例增长，因为分母都是一次 1024 signature 的执行时间。

gpu 主基准也有相同现象。256 与 1024 token 的 prefill 墙钟时间约为 0.99 s 和 1.02 s，吞吐从 259.8 增至 999.1 tokens/s〔基准 D〕。这些点的吞吐差异主要来自填充率口径，不能据此判断向量单元的利用率。

2000 和 3000 token 分别调用两个、三个 1024 signature。prefill 墙钟时间约为 7.67 s 和 11.48 s，折合每组约 3.83 s。4000 token 需要四组，墙钟时间增至约 17.45 s，吞吐降到 229.3 tokens/s〔基准 D〕。后续工作组开始时，已处理上下文更长。但仅凭这次扫描，还无法区分注意力计算、掩码写入、热状态和调度抖动各自的影响。若要判断瓶颈，需要分别记录各 signature 的耗时并读取硬件性能计数器。

按探针定义，prefill 吞吐等于 token 数除以墙钟时间。cpu/4096 档的 prefill 时间约为 \\(4096 \div 226.5 \approx 18.08\\) s〔基准 D〕。`GetTimeToFirstToken` 在此基础上加一次平均 decode step 的耗时，得到表中的 18.13 s。因此，这项复算只能检查指标定义和表中数字，不能证明模型加载或采样开销可以忽略。在相同模型和条件下，减少 prefill token 数或选择实测时间更短的后端，都能降低 TTFT。本书测得 gpu/4096 的 TTFT 为 4.45 s〔基准 D〕。

## 4.2　设计权衡：固定形状还是动态形状

executor 需要把长度可变的提示词映射到模型可执行的输入形状：编译后的模型可能只接受固定序列长度，提示词却可能包含 20 或 2000 个 token。LiteRT-LM 提供两条路径，对应 `LlmLiteRtCompiledModelExecutorStatic` 与 `LlmLiteRtCompiledModelExecutorDynamic` 两个 executor 子类。

静态形状路径预先编译若干个固定长度的 prefill 入口（signature），例如 128、512、1024。工作组选择先排除超过剩余状态容量的入口，再按长度从大到小处理。完整块优先使用当前入口；不足一块的余数如何处理，由后端与入口间距决定：

```cpp
// runtime/executor/litert_compiled_model_executor_utils.cc:536
    if (use_greedy_chunking) {  // (1)
      continue;
    }

    int next_seq_len = next_it->first;
// ...
    if (next_seq_len * 2 >= cur_seq_len && input_length <= next_seq_len) {  // (2)
// ...
      continue;
    } else if (input_length * 2 >= cur_seq_len &&  // (3)
               cur_seq_len <= current_remaining_capacity) {
// ...
      work_groups.push_back(std::make_pair(it->second, input_length));  // (4)
      current_remaining_capacity -= cur_seq_len;
      input_length = 0;
      break;
    }
```

代码行 `(1)` 在 CPU 路径上把余数继续交给较小入口，以减少填充。其他后端使用带阈值的选择：`(2)` 两个相邻入口的长度相差不超过 2 倍，且余数能装入较小入口时，继续向下选择；否则，`(3)``(4)` 余数至少达到当前入口一半、容量也足够时，就用当前入口覆盖余数。到最小入口仍有余数时，使用一个最小入口补齐。

以下填充量按工作组的 signature 总长度减分配的输入 id 数计算，用于比较分块策略。它不是实际有效 position 的逐项统计；每次 prefill 留下 pending token 的处理在本节末说明。

以入口集 {1024, 512, 128}、1300-token 输入和充足剩余容量为例。CPU 选择 1024、128、128、128，最后一组只有 20 个真实 token，总共填充 108 个位置。GPU 选择 1024、512，第二组处理 276 个真实 token，填充 236 个位置。前者少算填充，后者少调用两次模型。哪条路径更快，不能仅凭填充量判断。

本书基准模型 Gemma 4 E4B 的入口集是 {1024, 128}。同样处理 1300 个 token，当前 CPU 与 GPU 策略都会选择 1024 加三个 128，整体填充率为 \\(108/1408 \approx 7.7\%\\)。这里是按源码推演的工作组，附录 D 的旧版测量没有验证这组执行时间。

<div class="aside-compare">

llama.cpp 把提示词分成不超过 `n_ubatch` 的物理微批，并按实际批长构建计算图。[^ch04-llamacpp-ubatch] 它不需要在固定 signature 末尾填充，也不保存多个预编译序列长度入口。相应地，形状直到运行时才确定。两种方案对性能和产物体积的影响取决于编译器与后端，需要实测比较。

</div>

填充量仍随输入长度变化，但不能再用统一的“最大块加一个容纳余数的入口”推导曲线。入口集 {1024, 128} 下，CPU 处理 700 个 token 时使用六个 128，填充 68 个位置；GPU 的余数达到 1024 的一半，因而使用一个 1024，填充 324 个位置。两者分别减少填充计算与模型调用次数，策略本身没有查询各 signature 的实测成本。

容量约束还可能改变结果。算法按每个工作组的 signature 长度扣减剩余容量，真实 token 放得下并不保证补齐后的工作组也放得下。若所有入口都超过剩余容量，或选完工作组仍有未处理输入，函数返回 `FailedPreconditionError`。比较性能时，除了模型入口集与输入长度，也要记录后端和已占用上下文。

`Static::Prefill` 取得工作组后逐组调用内部实现：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:1635
  int remaining_capacity =
      state_->GetNumEntries() - llm_context_->runtime_state().current_step;

  const bool is_cpu = executor_settings_.GetBackend() == Backend::CPU;
  ABSL_ASSIGN_OR_RETURN(auto work_groups, GetOptimizedPrefillWorkGroups(
                                              prefill_signature_map_,
                                              ids.size(), remaining_capacity,
                                              /*use_greedy_chunking=*/is_cpu));
  for (int i = 0; i < work_groups.size(); ++i) {
    const auto& prefill_signature = work_groups[i].first;
    int prefill_length = work_groups[i].second;
// ...
    bool async = !*do_prefill_sync_ &&  // (1)
                 (i < work_groups.size() - 1 || !params.GetWaitForCompletion());
    ABSL_RETURN_IF_ERROR(PrefillInternal(
        prefill_signature, prefill_input_buffers_[prefill_signature],
        prefill_output_buffers_[prefill_signature],
        ids.subspan(/*pos=*/0, prefill_length), async));
    ids = ids.subspan(/*pos=*/prefill_length);  // (2)
  }
```

代码行 `(1)` `do_prefill_sync_` 为 false 时，除最后一组外的工作组都使用异步执行。最后一组是否等待，由 `wait_for_completion` 决定。Metal 缓冲会使 `do_prefill_sync_` 为 true，此时各组同步执行（源码把 Metal 的异步 prefill 标为待办）。`(2)` 每处理一组就缩短 `ids` span。循环后的 `RET_CHECK_EQ(ids.size(), 0)` 验证工作组覆盖了全部输入。

动态形状路径允许输入长度变化，KV cache 也按需增长。长提示词被切成不超过配置上限的块：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:1945
  if (prefill_chunk_size_ <= 0) {  // (1)
    return PrefillInternal(ids, params);
  }

  while (!ids.empty()) {
    int chunk_size =
        std::min(static_cast<int>(ids.size()), prefill_chunk_size_);
    absl::Span<int> chunk_ids = ids.first(chunk_size);  // (2)
    ids = ids.subspan(chunk_size);
    ABSL_RETURN_IF_ERROR(PrefillInternal(chunk_ids, params));  // (3)
  }
```

代码行 `(1)` 若未配置 chunk size，整段输入直接进入内部实现。`(2)``(3)` 否则按 `prefill_chunk_size_` 分块，每块调用一次 `PrefillInternal`。静态路径必须选择能容纳当前 token 数的预编译入口；动态路径的块长只受上限约束，最后一块通过 `std::min` 取得余数，不需要补到固定 signature 长度。动态 executor 目前只接受 cpu 后端。

表 4-1 汇总两条路径的差异。

| 维度 | 静态形状（`Static`） | 动态形状（`Dynamic`） |
|---|---|---|
| 形状 | 预编译固定长度入口（基准模型为 {1024, 128}） | 块长只受 `prefill_chunk_size` 上限约束 |
| 填充 | 有：最后一组可能填充，数量呈锯齿状变化 | 无：最后一块取余数 |
| KV cache | 按最大上下文一次性预留 | 首次按 prefill 长度分配，后续按需扩容 |
| 形状已知时机 | 编译期 | 运行时 |
| 运行条件 | 模型需包含对应 signature | 仅 cpu 后端 |

> 表 4-1　静态与动态两条 prefill 路径的权衡。第 8 章继续说明 executor 的分派条件。

静态路径调用基类的 `PrefillInternal`；动态路径有自己的同名实现。以下看静态路径。该函数写入输入缓冲，推进 `current_step`，更新已处理 token 记录，再由 LiteRT 执行 signature。输出写入 KV cache，供后续 decode 使用：

```cpp
// runtime/executor/llm_litert_compiled_model_executor.cc:489
    // We always hold one pending token in the input ids for the next
    // prefill or decode step.
    int prefill_length = ids.size() - 1;  // (1)
// ...
      std::transform(prefill_input_pos_ptr,
                     prefill_input_pos_ptr + prefill_length,
                     prefill_input_pos_ptr, [&](int token) mutable {
                       return llm_context_->runtime_state().current_step++;  // (2)
                     });
```

代码行 `(1)` 这里把处理长度减一：每次 prefill 都留下最后一个 token，作为下一次 prefill 或 decode 的 pending token。若已有上一组留下的 pending token，当前组先补处理它，再处理本组除末尾外的输入。因此，没有旧 pending token 的首组比工作组算账口径少处理一个位置；后续组则用旧 pending 补回这个位置。`(2)` `current_step` 随每个已处理 token 递增，用于填充 position 张量并确定 KV cache 位置。decode 延续同一个计数器，详见第 5 章。

### 4.2.1　prefill 的 host 侧 CPU 开销

`PrefillInternal` 除了调用 LiteRT 执行 signature，还包含三类 host 侧工作。这些操作不计入模型 FLOPs，但会进入墙钟时间。

掩码填充。signature 包含注意力 mask 时，prefill 会初始化并填充 4D 缓冲。`FillAttentionMask` 根据模型的全局、局部及环形缓存参数写入对应位置。写入量由 mask 形状、本段长度和具体分支共同决定；初始化还可能覆盖整个缓冲。这些操作发生在 host 侧并计入墙钟时间。

embedding 装配。除了 token 到 embedding 的查表，带 per-layer embeddings（部分模型在各层另有随层注入的附加 embedding）的模型还要准备逐层数据。同一个循环调用 `per_layer_embedding_lookup_->LookupPrefill`，按偏移写入本段结果。查表和写入的数据量随 prefill 长度增长。第 3 章介绍 embedding lookup。

KV 缓冲交换。双缓冲状态在执行前选择读写 bank，不复制 KV 数据。原地写入状态在执行前填写 int32 参数张量。`FillSingleBufferCacheParamTensor` 的实现见第 6 章。

现有探针没有分别记录这三类操作的耗时，附录 D 的曲线不能给出各项占比。旧版扫描中，250、500 和 1000 token 都执行同一个 1024 signature，短输入的低吞吐主要反映填充率口径。若要判断 host 侧开销，需要在 `PrefillInternal` 内分别插桩。

<figure>
{{#include figs/fig-4-1.svg}}
<figcaption>图 4-1　静态与动态 compiled executor 分别执行各自的 prefill 实现，最终都更新 KV cache。正在执行的 prefill 不会在工作组或 chunk 之间中断；decode 在下一次迭代开始前检查取消标志。</figcaption>
</figure>

## 4.3　取消语义：prefill 与 decode 的检查点

`TaskController::Cancel()` 将共享的原子变量设为 true。任务是否立即停止，取决于执行路径在何处读取该原子量。prefill 与 decode 的检查位置不同。

接口层为合作式取消（执行方主动轮询标志、在安全点自行停下）预留了字段：`ExecutorPrefillParams` 声明了取消标志与最大长度两个字段。但当前任务层只设置 `wait_for_completion`，两种 executor 也都不读取这两个字段，运行中的 prefill 因此尚无合作式取消。

<figure>
{{#include figs/fig-4-2.svg}}
<figcaption>图 4-2　本章 compiled executor 的取消请求不会中断已经开始的模型调用：prefill 要等整次调用返回后再改写状态，decode 则在下一次循环开始时停止。</figcaption>
</figure>

任务准备阶段有几个取消检查点：取得 executor 前后、组装输入前后都会读一次标志。但进入 prefill 本体后，要等整次调用返回才再次检查：

```cpp
// runtime/framework/resource_management/threaded_execution_manager.cc:912
    if (cancelled != nullptr && cancelled->load()) {
      llm_executor.value().reset();
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }
// ...
    auto responses =
        Tasks::Prefill(*llm_executor.value(), *executor_inputs,
                       /*wait_for_completion=*/true,
                       /*benchmark_info=*/session_info->benchmark_info);
// ...
    if (cancelled != nullptr && cancelled->load()) {
      responses = Responses(TaskState::kCancelled);
```

若取消发生在 `Tasks::Prefill` 之前，任务可以跳过模型执行。executor 开始运行后，当前 prefill 仍会完成，随后响应状态才改为 `kCancelled`。静态工作组循环和动态 chunk 循环都没有读取取消标志，因而分块边界不是取消点。

decode 在每次迭代开始前检查同一个标志：

```cpp
// runtime/core/tasks.cc:659
  while (true) {
    if (cancelled != nullptr && cancelled->load()) {  // (1)
// ...
      return absl::CancelledError("Process cancelled.");  // (2)
    }
```

代码行 `(1)` 在当前迭代开始时读取原子标志，为 true 时由 `(2)` 返回 `CancelledError`。内部采样也会把取消指针传入执行器参数，但本章的 compiled executor 不读取它。若 `Cancel()` 发生在该执行器的一次 decode step 中，当前 step 不会中止；任务循环在下一次迭代开始前检查取消。外部采样路径观察到取消后，还会执行一次 prefill，把最后一个采样 token 留作 pending token。因此，取消延迟除当前 step 与回调外，也可能包含这次状态整理。第 5 章继续分析 decode 循环。

## 4.4　任务调度：在工作线程执行 prefill

prefill 不在调用方线程上运行：会话把 prefill、decode 与克隆包装成任务，交给 `ThreadedExecutionManager` 调度。本节回答三个问题：任务在哪个线程执行，按什么顺序执行，回调会不会阻塞后续任务。任务之间的排队与依赖属于 inter-op 顺序；算子内部的线程数和核心绑定属于 intra-op 并行度，见第 8 章。

framework 里还有一个名字相近的独立原语 `ExecutionQueue`——单工作线程的顺序队列，会话路径并不经过它。它的移除语义需要说明一下以免混淆：`Remove` 只删除尚未执行的任务体，工作线程取到已删除的 id 时跳过；这是对排队中任务的撤销，与 4.3 节的运行中取消无关。它同样把任务体放在互斥锁外执行，防止任务内再次入队或移除时死锁。

### 4.4.1　按需扩容的 ThreadPool

真正执行任务的底层原语是按需扩容的线程池 `ThreadPool`。`Schedule` 提交任务时判断要不要再创建一个工作线程：

```cpp
// runtime/framework/threadpool.cc:87
  // If all worker threads are (supposed to be) busy, instantiates a new worker
  // thread to run the task.
  size_t num_threads = threads_.size();
  if (num_threads < max_num_threads_) {
    size_t num_tasks = num_active_tasks_ + tasks_.size();
    if (num_threads <= num_tasks) {  // (1)
      auto thread = WorkerThread::Create(this, name_prefix_);
      if (thread.ok()) {
        threads_.push_back(std::move(*thread));
// ...
      } else if (num_threads == 0) {
// ...
        return thread.status();  // (2)
// ...
      }
    }
  }

  tasks_.push_back(std::move(callback));  // (3)
```

代码行 `(1)` 任务数不少于现有线程数，且尚未达到 `max_num_threads_` 时，线程池尝试创建一个工作线程。线程池从零个线程开始按需增长。`(2)` 第一个工作线程创建失败时返回错误。后续扩容失败只记录警告，任务仍由现有线程处理。`(3)` 只要没有提前返回错误，无论是否扩容，任务都会进入队列。`RunWorker` 也在锁外执行任务。

### 4.4.2　两个单线程池

`ThreadedExecutionManager` 创建两个线程池，并把各自的上限设为 1：

```cpp
// runtime/framework/resource_management/threaded_execution_manager.cc:85
  execution_thread_pool_ =
      std::make_unique<ThreadPool>(/*name_prefix=*/"execution_thread_pool",
                                   /*max_num_threads=*/1);  // (1)
  callback_thread_pool_ =
      std::make_unique<ThreadPool>(/*name_prefix=*/"callback_thread_pool",
                                   /*max_num_threads=*/1);  // (2)
```

两个池各自至多有一个工作线程，首次提交任务时才按需创建。`(1)` prefill、decode 与克隆任务进入同一个执行池。因此，同一 `ThreadedExecutionManager` 中的模型任务串行运行。`(2)` 终态回调（任务结束时的那一次回调，区别于每段文本的流式回调，细分见 4.4.3 节）投递到单独的回调池，但执行线程仍会等待它完成。

### 4.4.3　回调线程边界与背压

流式回调与终态回调使用不同路径。`Tasks::Decode` 产生文本后直接调用流式回调，此时仍在执行池线程上；慢回调会延长当前 decode 迭代。循环结束后的 `Flush` 也可能产生一次流式更新：

```cpp
// runtime/core/tasks.cc:737
    if (is_streaming && any_updates) {
      callback(Responses(TaskState::kProcessing, std::move(step_texts),
                         std::move(step_scores), /*token_lengths=*/{},
                         std::move(step_token_ids)));
    }
```

任务结束时，`FinishTask` 把终态回调投递到回调池，随后等待它完成：

```cpp
// runtime/framework/resource_management/threaded_execution_manager.cc:513
    if (callback_thread_pool_ != nullptr) {
      ABSL_RETURN_IF_ERROR(callback_thread_pool_->Schedule(
          [callback = std::move(callback), responses = std::move(responses),
           task_id = task_id, next_task_state = std::move(next_task_state),
           this]() mutable {
            callback(std::move(responses));  // (1)
            absl::MutexLock lock(session_and_task_lookup_mutex_);
            auto status = UpdateTaskState(task_id, next_task_state);
// ...
          }));
      ABSL_RETURN_IF_ERROR(
          UpdateTaskState(task_id, TaskState::kLastCallbackQueued));
    } else {
// ...
  if (callback_thread_pool_ != nullptr) {
// ...
    ABSL_RETURN_IF_ERROR(
        callback_thread_pool_->WaitUntilDone(absl::Seconds(10)));  // (2)
  }
```

代码行 `(1)` 终态回调在回调池线程执行。`(2)` `FinishTask` 随后调用 `WaitUntilDone`，最长等待 10 s。`FinishTask` 此时仍占用单线程执行池，所以后续模型任务无法开始。回调池改变了终态回调的执行线程，但没有消除背压。源码 TODO 提出异步处理并移除这次等待。流式回调不经过该回调池，仍直接占用执行线程。

终态回调可以异步提交新任务并返回。如果回调在提交新任务后同步等待结果，执行线程与回调线程可能相互等待，直至其中一侧超时。

`ThreadedExecutionManager` 在 `QueueTask` 中检查依赖。`dependent_tasks` 非空的任务不能进入执行池。条件满足后，`Schedule` 才把任务加入执行池。第 3 章 `Clone` 使用的 `last_task_ids_` 记录了这组依赖。因此，decode 会在所依赖的 prefill 完成后开始。实际响应时间还包括模型执行和回调背压。

## 小结

prefill 通常比单 token decode 具有更高的算术强度，具体瓶颈仍需在目标设备上测量。静态路径按预编译 signature 组织工作组，可能产生填充。动态路径按 chunk 处理余数，目前仅在 cpu 后端使用。

正在执行的 prefill 不会轮询取消标志，工作组和 chunk 之间也没有取消检查。decode 在下一次迭代开始前检查。任务依赖使 decode 排在所需 prefill 之后；流式回调与终态回调都可能对单线程执行池形成背压。

prefill 完成后，KV cache 中已有提示词对应的 K/V。decode 如何读取这些 K/V，见第 5 章。

---

## 练习与自查

1. 工作组计算。本书基准模型的入口集是 {1024, 128}。输入 700 个 token，假设剩余容量充足，分别按 CPU 与 GPU 策略算出工作组与填充率。
2. 参数推演。把动态路径的 `prefill_chunk_size` 从 -1 改为 512。代码能确认 2048 token 会被拆成几块？激活峰值和 TTFT 的变化为什么还需要实测？
3. 异步边界。静态路径在什么条件下异步提交中间工作组？Metal 缓冲为何构成例外？
4. 曲线解读。附录 D 第七节中，250、500 和 1000 token 的 TTFT 近似相同，吞吐却接近按 token 数成比例增长。结合入口集 {1024, 128} 解释原因。
5. 取消语义。若取消发生在一次 prefill signature 执行期间和一次 decode step 执行期间，运行时分别会在何时观察到取消标志？

[^ch04-llamacpp-ubatch]: ggml-org，[*llama.cpp 源码 include/llama.h:342*](https://github.com/ggml-org/llama.cpp/blob/b9873/include/llama.h#L342)，版本 b9873；访问日期：2026-08-31。
