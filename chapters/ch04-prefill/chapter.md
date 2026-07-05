# 第 4 章 Prefill：吞下提示词

> 使命：理解 prefill 为什么快、它的两条实现路径各自在权衡什么，以及支撑"生成中途能立刻取消"的那层异步底座。

上一章，你的输入变成了一串 token id。现在 `Prefill` 接过它，把它一口吞进模型。这一步决定了"第一个字多久出来"。

## prefill 快在哪

回到第 2 章的 Roofline 眼镜：prefill 一次处理一整段提示词，几百上千个 token 共用同一批读进来的权重——算术强度高、算力受限，这就是它快的全部原因。它与 decode 的差距（本书基准上 10-20 倍〔基准 D〕）不是实现好坏，是物理分工。本章会从第 2 层的编排入口（`Tasks::Prefill`）一路下探到第 3 层的 executor。

编排入口很短，也很直白。它先算出模型能吃的最大 token 数，做一次越界校验，再把整段 token 交给 executor（`runtime/core/tasks.cc:413 @ v0.13.1`）：

```cpp
absl::StatusOr<Responses> Prefill(
    LlmExecutor& executor, ExecutorInputs& inputs, bool wait_for_completion,
    std::optional<BenchmarkInfo>& benchmark_info) {
  const int max_num_tokens = TryGetMaxNumTokens(executor);
  ASSIGN_OR_RETURN(auto text_data, inputs.GetTextDataPtr());
  // ...
  auto num_tokens = token_id_tensor_type.Layout().Dimensions().back();
  if (num_tokens >= max_num_tokens) {                                    // (1)
    return absl::InvalidArgumentError(absl::StrCat(
        "Input token ids are too long. ...", num_tokens, " >= ", max_num_tokens));
  }
  // ...
  ExecutorPrefillParams params;
  params.SetWaitForCompletion(wait_for_completion | benchmark_info.has_value());  // (2)
  // ...
  RETURN_IF_ERROR(executor.Prefill(inputs, params));                    // (3)
  return Responses(TaskState::kDone);
}
```

三行值得点名。(1) 这里的越界判断用的是 `>=` 而非 `>`：模型的上下文窗口要留一个位置给 decode 阶段的第一个 pending token，所以 prefill 能塞进去的 token 必须严格小于 `max_num_tokens`。(2) `wait_for_completion` 与 benchmark 开关做了一次按位或——只要在跑基准，就强制同步等待，否则拿不到干净的 prefill 计时。(3) 真正的活儿一行就交出去了：这个函数不碰模型、不选形状、不管 KV cache，全部下沉给 `executor.Prefill`。这是本书反复出现的"接口隔离"：第 2 层只做校验与编排，形状怎么选、算子怎么跑是第 3 层的事。

## 一道选择题：固定长度还是可变长度

executor 怎么"吞"这段 token，是一道端侧特有的选择题。

模型编译成能在硬件上跑的形式后，它接受的输入形状（多长的序列）往往是**固定**的。可提示词长度千变万化，有时 20 个 token，有时 2000 个。怎么办？LiteRT-LM 给了两条路径，对应两个 executor 子类：`LlmLiteRtCompiledModelExecutorStatic` 与 `LlmLiteRtCompiledModelExecutorDynamic`。

**静态形状**：预先编译好若干个固定长度的 prefill 入口（signature），比如 128、512、1024 各一个。这组入口按长度排好序存着（`SortedPrefillSignatureMap`，`runtime/executor/litert_compiled_model_executor_utils.h:43 @ v0.13.1`）：

```cpp
using SortedPrefillSignatureMap =
    absl::btree_map<int, std::string, std::greater<int>>;                // (1)
```

键是序列长度，值是 signature 名，比较器 `std::greater<int>` 让它**从大到小**排——`begin()` 就是最长的那个入口 (1)。选路的策略贪心而朴素（`GetOptimizedPrefillWorkGroups`，`runtime/executor/litert_compiled_model_executor_utils.cc:248 @ v0.13.1`）：

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

这段代码把一段输入切成一串「用哪个 signature、跑多长」的工单。(1) 先取最大的固定长度；(2) 只要剩余长度还够，就反复用最大入口铺过去，每次消掉 `max_seq_len`；(3)(4) 剩下的尾巴，从大往小找到第一个「再小一档就装不下」的入口收尾——即恰好覆盖尾巴的最小入口。举个可验算的账：入口集是 {1024, 512, 128}、输入 1300 个 token，则第一步用 1024 铺一次剩 276，尾巴 276 装不进 128、装得进 512，于是最终工单是 [1024, 512]，第二段用 512 的入口跑 276 个 token、余下 236 个位置填充。这就是固定形状的代价：换来编译器把 kernel 尽力优化的确定性，付出的是填充浪费和预编译多个入口的产物体积。

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

要说清楚的是，在 v0.13.1 里 `ExecutionQueue` 是 framework 提供的独立原语，会话路径实际用的是另一套 `ThreadedExecutionManager`，它把任务 `Schedule` 到一个线程池上跑（`runtime/framework/resource_management/threaded_execution_manager.cc:311 @ v0.13.1`），并在 `QueueTask` 里检查依赖：一个任务若还有 `dependent_tasks` 没跑完就拒绝入列（同文件:301）。这条依赖链保证 decode 一定在它依赖的 prefill 完成之后才开始。异步、分块、依赖三者合起来，端侧才有"响应跟手"的体感。这层线程模型第 8 章会展开。

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
- 异步队列原语：`runtime/framework/execution_queue.cc @ v0.13.1`（`ExecutionQueue`，锁外执行在 :97）；会话路径实用的线程池与依赖检查在 `runtime/framework/resource_management/threaded_execution_manager.cc:311 @ v0.13.1`。
