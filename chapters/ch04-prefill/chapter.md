# 第 4 章 Prefill：并行处理提示词

> 本章目标：说清 prefill 为什么是计算受限（compute-bound）的一步、静态与动态两条实现路径各自的权衡、每次 prefill 调用背后的隐性 CPU 开销（注意力掩码、embedding 装配、KV cache 缓冲交换），以及支撑异步执行与合作式取消的那层线程调度底座。

上一章末尾，输入文本已经变成一串 token id。`Prefill` 接过这串 id，一次前向处理整段提示词，把注意力中间结果写进 KV cache。这一步的耗时直接决定首 token 时延（TTFT，time-to-first-token）。

## prefill 为什么是计算受限的一步

沿用第 2 章的 Roofline 分析。prefill 一次前向处理整段提示词，几百上千个 token 复用同一批已读入的权重，算术强度（每字节权重承担的浮点运算数）高，落在 Roofline 的计算受限区。它与 decode 的差距不是实现优劣，是两类操作被不同资源顶住：prefill 受算力约束，decode 受内存带宽约束。

这一差距在本书基准上是实测可见的。附录 D 的主基准（Gemma 4 E4B，公开权重，`litert-lm benchmark` 中位数）里，gpu 后端上下文 1024 时 prefill 吞吐 999.1 tokens/s、decode 仅 50.6 tokens/s，相差约 20 倍；cpu 后端上下文 1024 时 prefill 259.2 tokens/s、decode 24.7 tokens/s，相差约 10 倍〔基准 D〕。同一模型、同一后端，两个数字差一到两个数量级，因为一个吃算力、一个吃带宽。本章从第 2 层的编排入口（`Tasks::Prefill`）下探到第 3 层的 executor，把这条路径上的每一处开销摊开。

编排入口很短。它先取模型能接受的最大 token 数，做一次越界校验，再把整段 token 交给 executor（`runtime/core/tasks.cc:412 @ v0.13.1`）：

```cpp
absl::StatusOr<Responses> Prefill(
    LlmExecutor& executor, ExecutorInputs& inputs, bool wait_for_completion,
    std::optional<BenchmarkInfo>& benchmark_info) {
  const int max_num_tokens = TryGetMaxNumTokens(executor);
  // ...
  auto num_tokens = token_id_tensor_type.Layout().Dimensions().back();
  if (num_tokens >= max_num_tokens) {                                    // (1)
    return absl::InvalidArgumentError(absl::StrCat(
        "Input token ids are too long. ...", num_tokens, " >= ", max_num_tokens));
  }
  // ...
  ExecutorPrefillParams params;
  params.SetWaitForCompletion(wait_for_completion | benchmark_info.has_value());  // (2)
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimePrefillTurnStart());            // (3)
  }
  RETURN_IF_ERROR(executor.Prefill(inputs, params));                    // (4)
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimePrefillTurnEnd(ids_buffer_span.size()));
  }
  return Responses(TaskState::kDone);
}
```

四处值得说明。(1) 越界判断用 `>=` 而非 `>`。模型的上下文窗口要留一个位置给随后的 pending token（下一节展开这个 token 的来历），所以 prefill 能写入的 token 数必须严格小于 `max_num_tokens`。(2) `wait_for_completion` 与 benchmark 开关做了一次按位或：只要在跑基准就强制同步等待，否则计时会把还在异步执行的 prefill 提前算完，拿到偏短的数字。(3)(4) 计时探针 `TimePrefillTurnStart` / `TimePrefillTurnEnd` 把 `executor.Prefill` 夹在中间，`TimePrefillTurnEnd` 收到本轮 token 数用于算吞吐。真正的计算下沉给 `executor.Prefill`：这个函数不碰模型、不选形状、不管 KV cache。第 2 层只做校验、计时与编排，形状怎么选、算子怎么跑是第 3 层 executor 的事。

这两个探针的实现极薄。`TimePrefillTurnStart` 以 `prefill:<turn_index>` 为键记一个 `absl::Now()` 起点，`TimePrefillTurnEnd` 再取一次 `absl::Now()` 求差，连同 token 数存进 `prefill_turns_`（`runtime/engine/io_types.cc:296` 与 `:306 @ v0.13.1`）。它量的是整个 `executor.Prefill` 的墙钟时间，包含下面要讲的分块循环、掩码填充、embedding 装配与 KV cache 缓冲交换的全部开销。附录 D 那张 prefill 吞吐表，每个数字都出自这一对探针。

### 用探针复算一次 prefill 曲线

拿这对探针可以复现 notes 里列的实验：把提示词长度从 100 扫到 4000，画 prefill 耗时与吞吐曲线（附录 C 复现表第 4 项，`litert-lm benchmark` 扫 `-p 100…4000`）。曲线的形状能直接印证「计算受限」这个判断。

短提示词区间吞吐偏低。附录 D 里 cpu 后端上下文 256 时 prefill 只有 65.6 tokens/s，到 1024 时跳到 259.2 tokens/s〔基准 D〕。原因是几百个 token 还喂不满 CPU 的向量单元，算术强度不够高，一部分时间花在读权重而非做乘加，此时更接近带宽受限。随着长度增加，同一批权重被更多 token 复用，算术强度上升，吞吐爬向该后端的算力上限。gpu 后端同样从 256 档的 259.8 tokens/s 升到 1024 档的 999.1 tokens/s〔基准 D〕。

长提示词区间吞吐回落。cpu 从 1024 档的 259.2 降到 4096 档的 226.5，gpu 从 999.1 降到 925.2〔基准 D〕。回落来自注意力本身：因果注意力的计算量随序列长度平方增长，长上下文里注意力占的比重变大，而下一节会看到，掩码填充也是 O(L²) 的 CPU 开销。这条「先升后回落」的曲线不是实现瑕疵，是算术强度上升与注意力二次项此消彼长的自然结果。

TTFT 与吞吐互为倒数，可当面对账。cpu/4096 的 TTFT 实测 18.13 s，而 4096 ÷ 226.5 tokens/s ≈ 18.08 s，两者吻合到小数点后一位〔基准 D〕。这条对账说明 TTFT 在长上下文下几乎全部是 prefill 耗时，加载与采样的固定开销可忽略。工程含义直接：要压 TTFT，要么减少 prefill 的 token 数（模板 diff 增量渲染，见第 3 章），要么换到 prefill 吞吐更高的后端（gpu/4096 的 TTFT 只有 4.45 s）。

## 设计权衡：固定形状还是动态形状

executor 如何处理这段 token，是一处端侧特有的设计权衡。

模型编译成能在硬件上运行的形式后，它接受的输入形状（序列长度）往往是固定的。而提示词长度千变万化，有时 20 个 token，有时 2000 个。LiteRT-LM 给出两条路径，对应两个 executor 子类：`LlmLiteRtCompiledModelExecutorStatic` 与 `LlmLiteRtCompiledModelExecutorDynamic`。

静态形状路径预先编译好若干个固定长度的 prefill 入口（signature），比如 128、512、1024 各一个。这组入口按长度排序存放（`SortedPrefillSignatureMap`，`runtime/executor/litert_compiled_model_executor_utils.h:43 @ v0.13.1`）：

```cpp
using SortedPrefillSignatureMap =
    absl::btree_map<int, std::string, std::greater<int>>;                // (1)
```

键是序列长度，值是 signature 名，比较器 `std::greater<int>` 让它从大到小排列，`begin()` 即最长的入口 (1)。选路策略是一个贪心切分（`GetOptimizedPrefillWorkGroups`，`runtime/executor/litert_compiled_model_executor_utils.cc:248 @ v0.13.1`）：

```cpp
int max_seq_len = prefill_runner_set.begin()->first;                    // (1)
while (input_length >= max_seq_len) {
  work_groups.push_back(
      std::make_pair(prefill_runner_set.begin()->second, max_seq_len)); // (2)
  input_length -= max_seq_len;
}
if (input_length > 0) {
  for (auto it = prefill_runner_set.begin(); it != prefill_runner_set.end(); ++it) {
    if (std::next(it) != prefill_runner_set.end() &&
        std::next(it)->first >= input_length) {                        // (3)
      continue;
    }
    work_groups.push_back(std::make_pair(it->second, input_length));    // (4)
    break;
  }
}
```

这段代码把一段输入切成一串「用哪个 signature、处理多长」的工单。(1) 先取最大的固定长度；(2) 只要剩余长度还够，就反复用最大入口覆盖，每次消耗 `max_seq_len` 个位置；(3)(4) 剩余长度不足一个最大入口时，从大往小找到第一个「再小一档就装不下」的入口收尾，即恰好覆盖剩余长度的最小入口。

举一个可复算的例子。入口集是 {1024, 512, 128}、输入 1300 个 token：第一步用 1024 覆盖一次，剩 276；276 装不进 128、装得进 512，于是最终工单是 [1024, 512]，第二段用 512 的入口处理 276 个真实 token、余下 236 个位置填充。这就是固定形状的代价：换来编译器把 kernel 充分优化的确定性，付出的是填充浪费和预编译多个入口带来的产物体积。

**填充浪费定量。** 这个策略的浪费量随输入长度呈锯齿状波动，可以对固定入口集算一条曲线。仍用 {1024, 512, 128}。设输入长度 L，贪心切分先用若干个 1024 铺满，剩余 r = L mod 1024，再用恰好覆盖 r 的最小入口收尾；浪费量等于该收尾入口的长度减去 r。当 r 落在 (512, 1024) 区间时，收尾用 1024，最坏浪费接近 512（例如 L = 1025，r = 1，用 1024 收尾，浪费 1023，占该段的 99.9%）；r 恰等于某个入口长度时浪费为 0（例如 L = 1536，工单 [1024, 512]，零填充）。对整段而言，L = 1300 的例子填了 236 个零、总位置 1536，填充率约 15%；而 L 取 (1024, 1536] 区间的下沿时，例如 L = 1025，整段填充率高达约 40%（1024 + 512 = 1536 个位置里只有 1025 个是真实 token）。这条锯齿曲线的最坏点，恰好落在每一档入口长度刚被跨过的位置。

源码在这个函数上留了一条 TODO：`b/378772479 - Improve this strategy once we have benchmarked costs`（`runtime/executor/litert_compiled_model_executor_utils.cc:255 @ v0.13.1`）。它承认当前策略只是「先用最大入口铺、剩余用最小可容入口收尾」的朴素贪心，尚未按实测成本调优。据此推断，更优的切分会权衡两件事：大入口的算术强度更高、单位 token 更省时，但一旦最后一段填充率高，被填充的零位置也要走一遍前向计算，白付算力。在填充率高的区间，用两个较小入口拼可能比一个大入口更省——但这需要 benchmark 各入口的实际 kernel 成本才能定夺，也正是 TODO 想做的事。这属于基于代码与注释的推断，v0.13.1 尚未实现。

`Static::Prefill` 拿到工单后逐段调用底层（`runtime/executor/llm_litert_compiled_model_executor.cc:1537 @ v0.13.1`）：

```cpp
ASSIGN_OR_RETURN(auto work_groups, GetOptimizedPrefillWorkGroups(
                                       prefill_signature_map_, ids.size()));
for (int i = 0; i < work_groups.size(); ++i) {
  const auto& prefill_signature = work_groups[i].first;
  int prefill_length = work_groups[i].second;
  // ...
  bool async = !*do_prefill_sync_ &&
               (i < work_groups.size() - 1 || !params.GetWaitForCompletion()); // (1)
  RETURN_IF_ERROR(PrefillInternal(
      prefill_signature, prefill_input_buffers_[prefill_signature],
      ids.subspan(/*pos=*/0, prefill_length), async));
  ids = ids.subspan(/*pos=*/prefill_length);                           // (2)
}
```

(1) 这行决定每一段是异步派发还是同步等：除最后一段外的所有工单都可以异步（因为后面还要接着提交，不必等），只有最后一段在调用方要求 `wait_for_completion` 时才同步落地。这就是编排层那个 `wait_for_completion` 参数真正生效的地方。(2) 每铺完一段就把 span 往前推，循环结束时全部消化完，函数末尾有一句 `RET_CHECK_EQ(ids.size(), 0)` 兜底，确保工单不多不少覆盖整段输入。

**动态形状**：输入长度可变，KV cache 也按需增长。长提示词被切成固定大小的块，一块一块 prefill（`runtime/executor/llm_litert_compiled_model_executor.cc:1855 @ v0.13.1`）：

```cpp
if (prefill_chunk_size_ <= 0) {
  return PrefillInternal(ids, params);                                  // (1)
}
while (!ids.empty()) {
  int chunk_size = std::min(static_cast<int>(ids.size()), prefill_chunk_size_);
  absl::Span<int> chunk_ids = ids.first(chunk_size);                    // (2)
  ids = ids.subspan(chunk_size);
  RETURN_IF_ERROR(PrefillInternal(chunk_ids, params));                  // (3)
}
```

(1) 若没配 chunk size，就一把梭：整段直接进内部实现，全靠动态形状消化任意长度。(2)(3) 否则按 `prefill_chunk_size_` 切块，每块单独一次 `PrefillInternal`。与静态路径的关键区别是：静态的段长必须命中预编译入口之一，动态的块长只受一个上限约束，最后一块由 `std::min` 自然取到不足一个 chunk 的余数——不需要填充。这更灵活，桌面和服务端常见，代价是失去了固定形状带来的一部分编译期优化。

两条路径都通过基类的同一个内部函数落地（`PrefillInternal`，`runtime/executor/llm_litert_compiled_model_executor.cc:543 @ v0.13.1`）。它把 token 写进输入缓冲、推进 `current_step`、更新已处理 token 记录，再交给 LiteRT 跑 signature，注意力中间结果就此写进 KV cache，为 decode 铺好底：

```cpp
// We always hold one pending token in the input ids for the next
// prefill or decode step.
int prefill_length = ids.size() - 1;                                   // (1)
// ...
std::transform(prefill_input_pos_ptr, prefill_input_pos_ptr + prefill_length,
               prefill_input_pos_ptr, [&](int token) mutable {
                 return llm_context_->runtime_state().current_step++;  // (2)
               });
```

(1) 这里把要处理的长度减一：每次 prefill 都刻意留一个 token 不处理，把它当作"pending token"挂到下一次 prefill 或 decode 的第一步。这解释了上一节 `>=` 的越界判断——那个被留下的位置，正是给这个 pending token 的。(2) `current_step` 是一个随每个 token 自增的计数器，它填进 position 张量、也标记 KV cache 里这段 token 的落位。这个 step 计数器会一直用到 decode（第 5 章），是 prefill 与 decode 之间的接力棒。

> 对照视野
> "预编译多个固定长度入口"这个取舍在端侧很典型：宁可多占一点编译产物和填充浪费，换取运行时的确定性与峰值性能。云端更倾向动态形状（灵活、省显存），因为它不缺重新编译的算力，也不在乎多留几个 kernel。同一个问题，两端因约束不同给出相反的默认答案——这类"因地制宜"贯穿全书。

<figure>
{{#include figs/fig-4-1.svg}}
<figcaption>图 4-1　prefill 的两条路径与异步底座。静态路径按长度挑固定 signature，动态路径分块吞入；两者都经 PrefillInternal 落到 LiteRT，并把结果写进 KV cache。任务经队列异步执行，取消在 decode 循环每步与分块边界处生效。</figcaption>
</figure>

## 生成中途，为什么能立刻停

第 2 章那份"二十个问题"里有一问：生成中途取消，为什么能立刻停下（第 9 问）？答案的一半在 prefill 这一层——但要说清楚，得先分清"API 上有"和"路径上真正生效"两件事。

`ExecutorPrefillParams` 上确实挂着两个为取消准备的开关（`runtime/executor/llm_executor_io_types.h:376 @ v0.13.1`）：一个取消标志 `GetCancelFlag()`（一个 `const std::atomic_bool*`），一个限长开关 `GetMaxPrefillSequenceLength()`。前者让调用方传进一个原子布尔量的指针，后者限制单次能用多长的 prefill signature。设计意图很清楚：一次跑太长的 signature 就是一段不可打断的时间，把单次长度限住，取消标志才有足够密的机会被检查到。

不过在 v0.13.1 里要诚实：这两个字段在整个 prefill 运行路径上还只是**声明和存取器**，`llm_litert_compiled_model_executor.cc` 里既没读 `GetCancelFlag()`、也没读 `GetMaxPrefillSequenceLength()`（在源码里 grep 这两个符号，命中的只有 `llm_executor_io_types.*` 自身与其测试）。换句话说，prefill 内部目前不会在算到一半时主动瞄取消标志。这是我们据代码得出的判断，不是文档说法。

那"立刻停"靠什么？靠两处真正落地的机制。其一，**分块本身就是取消点**：动态路径每块、静态路径每个工单之间，控制权都回到循环里，长 prefill 被拆成若干段短调用，天然给了打断的缝隙。其二，取消真正生效的地方在 decode 循环——生成每一步开头都在看这个标志（`runtime/core/tasks.cc:487 @ v0.13.1`）：

```cpp
while (true) {
  if (cancelled != nullptr && cancelled->load()) {                     // (1)
    // ...
    return absl::CancelledError("Process cancelled.");                 // (2)
  }
  // ... run one decode step ...
}
```

(1) 每次循环先读原子标志，(2) 一旦为真就立刻返回 `CancelledError`。注意编排层的 `Prefill`（tasks.cc:413）签名里根本没有 `cancelled` 参数，只有 `Decode` 有，这印证了前面的判断：v0.13.1 里 prefill 的可打断性来自"被切成小段"，decode 的可打断性才来自"每步查标志"。第 9 问答案的这一半，落在 decode 的 `ShouldStop` 路径（第 5 章展开）。会话层把两者串起来：`SessionAdvanced::Cancel()` 只做一件事——`cancelled_->store(true)`（`runtime/core/session_advanced.h:68 @ v0.13.1`），把这个共享原子量置真，正在跑的 decode 循环下一步就会读到。

## 异步底座：把 prefill 挪出主线程

prefill 任务并不总在主线程同步跑。framework 层提供了一个最小的异步原语——一个单工作线程的顺序队列（`ExecutionQueue`，`runtime/framework/execution_queue.cc @ v0.13.1`）。它的 `Enqueue` 把任务塞进两个结构：

```cpp
int id = next_id_++;
pending_tasks_[id] = std::move(task);                                  // (1)
task_order_.push(id);                                                  // (2)
return id;
```

(1) `pending_tasks_` 存任务体、(2) `task_order_` 存 FIFO 顺序，两者分开是为了让取消变简单：`Remove(id)` 只需从 `pending_tasks_` 里 `erase` 掉任务体，不用去队列里翻找。工作线程取任务时若发现 id 已不在表里，就知道这个任务被取消了，直接跳过。工作线程循环里有一处关键决定（`runtime/framework/execution_queue.cc:97 @ v0.13.1`）：

```cpp
// Execute the task OUTSIDE the mutex lock.
// This prevents deadlocks if a task itself calls Enqueue() or Remove().
if (current_task) {
  std::move(current_task)();                                          // (1)
}
```

(1) 任务在锁**外**执行。注释点明了原因：任务自己可能回头调 `Enqueue` 或 `Remove`（比如一个 prefill 任务完成后排入 decode），若持锁执行就会自锁死。这是并发代码里常见的一处纪律——临界区只碰共享结构，真正的活儿在锁外干。

要说清楚的是，在 v0.13.1 里 `ExecutionQueue` 是 framework 提供的独立原语，会话路径实际用的是另一套 `ThreadedExecutionManager`。它才是 prefill/decode 任务真正的调度者，下面把它拆开。先划一条边界：这里讲的是**任务层**的排队与依赖（inter-op，谁先谁后）；算子内部用几个线程、绑哪个核（intra-op 并行度）是第 8 章的事，两者互不越界。

### 会自己扩容的 ThreadPool

底层原语是一个按需扩容的线程池。`Schedule` 提交任务时顺带决定要不要加人手（`runtime/framework/threadpool.cc:78 @ v0.13.1`）：

```cpp
  // If all worker threads are (supposed to be) busy, instantiates a new worker
  // thread to run the task.
  size_t num_threads = threads_.size();
  if (num_threads < max_num_threads_) {
    size_t num_tasks = num_active_tasks_ + tasks_.size();
    if (num_threads <= num_tasks) {                       // (1)
      auto thread = WorkerThread::Create(this, name_prefix_);
      if (thread.ok()) {
        threads_.push_back(std::move(*thread));
      } else if (num_threads == 0) {
        // ...
        return thread.status();                           // (2)
      }
      // ...
    }
  }

  tasks_.push_back(std::move(callback));                  // (3)
```

(1) 扩容判据：在跑的加在排的任务数不少于现有线程数，且未到 `max_num_threads_` 上限，就再建一个工作线程。池从零起步、按压力生长，空闲时不预留线程。(2) 错误容忍分两档：第一个线程都建不出来是致命错误，直接上抛；后续扩容失败只记警告，任务仍然入队，由现有线程消化。(3) 无论扩容成败，任务都进队列。工作线程的主循环则是和 `ExecutionQueue` 一样的锁外执行纪律（`RunWorker`，`:180`）：取任务、`mutex_.unlock()`、执行、再 `mutex_.lock()`，任意用户代码都不在持锁状态下运行。

### 两个只有一个线程的池

`ThreadedExecutionManager` 的构造函数把这套线程池用得很反直觉（`runtime/framework/resource_management/threaded_execution_manager.cc:74 @ v0.13.1`）：

```cpp
  execution_thread_pool_ =
      std::make_unique<ThreadPool>(/*name_prefix=*/"execution_thread_pool",
                                   /*max_num_threads=*/1);       // (1)
  callback_thread_pool_ =
      std::make_unique<ThreadPool>(/*name_prefix=*/"callback_thread_pool",
                                   /*max_num_threads=*/1);       // (2)
```

两个池、各限一个线程。(1) 执行池单线程不是省资源，是**用构造保证串行**：执行器与 KV cache 都不是线程安全的，所有 prefill/decode/克隆任务都排进同一个单线程池，天然互斥，不需要给执行器内部加一把锁。多会话（第 3 章）也因此天然安全：不同 Session 的任务最终都汇进这一个线程。(2) 回调池单独一个线程，用途在下一段。

### 回调走另一条线程，别堵住执行

任务完成后要通知调用方（流式回调、完成回调）。回调是用户代码，跑多久、干什么都不可控。若在执行线程上直接调它，一个慢回调就会卡住后面所有排队的 decode step。`ThreadedExecutionManager` 的处理是把回调改投到回调池（`threaded_execution_manager.cc:438 @ v0.13.1`）：

```cpp
    if (callback_thread_pool_ != nullptr) {
      RETURN_IF_ERROR(callback_thread_pool_->Schedule(
          [callback = std::move(callback), responses = std::move(responses),
           task_id = task_id, next_task_state = std::move(next_task_state),
           this]() mutable {
            callback(std::move(responses));                      // (1)
            absl::MutexLock lock(session_and_task_lookup_mutex_);
            auto status = UpdateTaskState(task_id, next_task_state);
            // ...
          }));
```

(1) 用户回调在回调池的线程上执行，执行池不等它。这同时规避了一类死锁：回调若反过来调引擎接口（例如收到完整回复后立刻发起下一轮 prefill），新任务排进执行池即可，不会出现「执行线程等回调、回调等执行线程」的环。回调池同样单线程，保证回调按任务完成的顺序逐个送达，流式文本不会乱序。这一段的收尾处有一个 `WaitUntilDone(absl::Seconds(10))`，等回调池清空后 `FinishTask` 才返回，源码上方挂着一条 TODO（b/476205457）说计划改成全异步——这是 v0.13.1 里回调路径尚存的一处同步点。

回到主线：`ThreadedExecutionManager` 把任务 `Schedule` 到执行池（`:311`），并在 `QueueTask` 里检查依赖，一个任务若还有 `dependent_tasks` 没跑完就拒绝入列（`:301`）。第 3 章 `Clone` 那节看到的 `last_task_ids_` 串链，串的正是这张依赖图。这条依赖链保证 decode 一定在它依赖的 prefill 完成之后才开始。异步、分块、依赖三者合起来，端侧才有"响应跟手"的体感。

## 小结

prefill 是算力受限的一步，快，且有静态/动态两条实现路径应对端侧固定形状的约束：静态按预编译入口贪心切工单、代价是填充；动态按 chunk 切、不用填充但少了编译期优化。它随时可被打断——但在 v0.13.1，这份"可打断"来自把长 prefill 切成小段，以及紧接其后的 decode 循环每步查取消标志，而非 prefill 内部主动轮询。异步底座把任务挪出主线程，依赖链保证 decode 接在 prefill 之后。

提示词已经吞进去了，KV cache 也填好了第一段。下一章，最核心的一步：decode 循环，逐字的心跳。

---

## 参考

- prefill 编排入口：`runtime/core/tasks.cc:413 @ v0.13.1`（`Prefill`，含 `>=` 越界校验与 `wait_for_completion` 处理）。
- 静态路径：`runtime/executor/llm_litert_compiled_model_executor.cc:1503 @ v0.13.1`（`Static::Prefill`，工单循环与 async 判定在 :1537）。
- 动态路径分块：`runtime/executor/llm_litert_compiled_model_executor.cc:1841 @ v0.13.1`（`Dynamic::Prefill`，:1855 起为分块循环）。
- 内部实现与 pending token / current_step：`runtime/executor/llm_litert_compiled_model_executor.cc:543 @ v0.13.1`（基类 `PrefillInternal`）。
- signature 排序与选路：`runtime/executor/litert_compiled_model_executor_utils.h:43 @ v0.13.1`（`SortedPrefillSignatureMap`）；`runtime/executor/litert_compiled_model_executor_utils.cc:248 @ v0.13.1`（`GetOptimizedPrefillWorkGroups`）。
- 取消与限长参数：`runtime/executor/llm_executor_io_types.h:376 @ v0.13.1`（`ExecutorPrefillParams`；`GetCancelFlag` / `GetMaxPrefillSequenceLength` 在 v0.13.1 尚未被 executor 消费）。取消实际生效在 decode 循环 `runtime/core/tasks.cc:487 @ v0.13.1`，会话侧置标志在 `runtime/core/session_advanced.h:68 @ v0.13.1`。
- 异步队列原语：`runtime/framework/execution_queue.cc @ v0.13.1`（`ExecutionQueue`，锁外执行在 :97）。
- 任务层调度：`ThreadPool::Schedule` 弹性扩容 `runtime/framework/threadpool.cc:78`、`RunWorker` 锁外执行 `:180`；`ThreadedExecutionManager` 双单线程池构造 `runtime/framework/resource_management/threaded_execution_manager.cc:74`、回调改投回调池 `:438`（含 TODO b/476205457）、`Schedule` 到执行池 `:311`、`QueueTask` 依赖检查 `:301`——均 @ v0.13.1。
