# 模块素材：并发框架 (Framework)  `framework`

> 来源：litert-lm-guide/data.json（多智能体源码分析）。**这是素材，不是正文**。
> 引用进正文前，须按 CLAUDE.md 第二节逐条 Read 源码核验、补 `@ v0.13.1`。

**一句话**：runtime/framework/ 是 LiteRT-LM 的并发与资源编排底座。它提供线程池(ThreadPool)、任务队列(ExecutionQueue)、线程抽象(WorkerThread + pthread/std::thread 双实现)、类型安全的资源注册表(ResourceRegistry),以及把这些原语组合起来、管理 LLM 推理任务调度/依赖/取消/会话生命周期的 ExecutionManager(线程版与串行版)。它向上游的 Engine/Session 屏蔽了多线程细节，让 prefill/decode/clone/scoring 等任务能异步、有依赖地执行。

**在架构中的位置**：位于 Engine/Session 层(runtime/core/, runtime/engine/)与底层 Executor(runtime/executor/ 的 LLM/Vision/Audio 推理器)之间，是二者的"调度与资源中枢"。上游 EngineImpl 在创建时根据后端/配置二选一地构造 ThreadedExecutionManager 或 SerialExecutionManager(见 runtime/core/engine_advanced_impl.cc:353-377),Session 通过 ExecutionManager 接口提交 AddPrefillTask/AddDecodeTask 等。下游它持有 ResourceManager，后者独占地包装并发访问共享的 LlmExecutor/VisionExecutor/AudioExecutor(用 MovableMutexLock 实现"取出即上锁"),并通过 ContextHandler 在多会话间做 KV-cache(ProcessedContext) 的 copy-on-write 共享。ThreadPool/WorkerThread/ThreadOptions 是最底层、可独立复用的并发原语。

## 关键文件
- `runtime/framework/threadpool.cc` — 线程池实现,含懒扩容、RunWorker 循环、安全关闭。
- `runtime/framework/resource_management/threaded_execution_manager.cc` — 多线程任务调度与依赖/状态机核心实现。
- `runtime/framework/resource_management/resource_manager.cc` — 共享 Executor 并发包装与 KV-cache 写时复制优化。

## 核心抽象
- **ThreadPool** (class) 〔`runtime/framework/threadpool.h`〕：线程池。Schedule(absl::AnyInvocable<void()&&>) 入队任务并按需懒创建 worker(只有当现有线程数 <= 待处理+活跃任务数且未达上限时才新建,见 threadpool.cc:90-116);WaitUntilIdle 等队列空、WaitUntilDone 等队列空且 num_active_tasks_==0;析构时置 stopped_、swap 出线程并逐个 Join。全程用单把 absl::Mutex + absl::Condition 做条件等待,不用裸 condition_variable。
- **WorkerThread** (class) 〔`runtime/framework/worker_thread.h`〕：工作线程抽象基类。静态 Create() 是工厂(由具体 .cc 提供 pthread 或 std::thread 实现);Join() 用原子 joined_ 保证幂等;子类只需实现 JoinImpl() 并在自己的 ThreadBody 中调用基类 RunWorker(),后者转发到 ThreadPool::RunWorker() 进入取任务循环。
- **ExecutionManager** (interface) 〔`runtime/framework/resource_management/execution_manager.h`〕：执行管理器接口,是 Framework 对上游暴露的核心抽象。提供 RegisterNewSession/ReleaseSession/CancelAllTasksInSession 管理会话;AddPrefillTask/AddDecodeTask/AddCloneSessionTask/AddTextScoringTask 提交带依赖(dep_tasks)与取消标志(shared_ptr<atomic<bool>>)的任务;WaitUntilDone/WaitUntilSessionDone/WaitUntilAllDone 同步等待。SessionInfo 持有 ContextHandler/Sampler/StopTokenDetector/active_tasks;TaskInfo 持有依赖集合与回调。
- **ThreadedExecutionManager** (class) 〔`runtime/framework/resource_management/threaded_execution_manager.h`〕：ExecutionManager 的多线程实现。内部用两个 max_num_threads=1 的 ThreadPool:execution_thread_pool_ 串行跑推理、callback_thread_pool_ 跑回调,从而保证对 Executor 的访问天然串行又不阻塞提交方。维护任务状态机(kCreated→kQueued→kProcessing→kDone/...)与依赖图:CreateTask 解析依赖、QueueTask 入池、StartTask 取会话上下文、FinishTask 触发后继任务并传播失败/取消。所有 task/session 表由 session_and_task_lookup_mutex_ 保护。
- **SerialExecutionManager** (class) 〔`runtime/framework/resource_management/serial_execution_manager.h`〕：ExecutionManager 的单线程实现,不创建任何线程、非线程安全。用 ready_queue_(std::deque) 代替线程池,在 WaitUntil* 被调用时同步地 RunNextTask() 把队列里就绪任务跑完。适合 IoT/单线程或确定性测试场景,与 ThreadedExecutionManager 共享几乎相同的任务状态机与依赖逻辑。
- **ResourceManager / LockedLlmExecutor** (class) 〔`runtime/framework/resource_management/resource_manager.h`〕：共享 Executor 的并发守门人。AcquireExecutor()/AcquireExecutorWithContextHandler() 在 executor_mutex_ 下取出 LlmExecutor 并用 MovableMutexLock 包成 LockedLlmExecutor 返回——句柄存活期间一直持锁,析构即解锁。AcquireExecutorWithContextHandler 还封装上下文切换:同会话直接返回;共享 ProcessedContext 则只换 RuntimeConfig/State;否则 clone 并 RestoreContext。LockedLlmExecutor 在 Prefill/Decode 前做 token 复用与 KV-cache copy-on-write 优化。Vision/Audio Executor 同理由 LockedVision/AudioExecutor 守护,各自一把 mutex。
- **ResourceRegistry / ResourceScopedLock** (class) 〔`runtime/framework/resource_registry.h`〕：通用类型安全资源注册表(模板)。Register<T>(id, unique_ptr<T>) 用类型擦除的 ResourceHolder<T> 存入,每个 ResourceNode 自带一把 absl::Mutex;Acquire<T>(id) 做 dynamic_cast 校验类型后返回 ResourceScopedLock<T>(资源指针 + 该资源专属锁),提供 operator->/* 的 RAII 独占访问;View<T> 提供只读引用;HasResource 查询存在性。
- **ExecutionQueue** (class) 〔`runtime/framework/execution_queue.h`〕：轻量级 FIFO 串行任务队列,单后台 std::thread。Enqueue 返回自增 id 并把任务存进 pending_tasks_ map、id 进 task_order_ queue;Remove(id) 只需从 map 删除即可 O(1) 取消(worker pop 到的 id 若在 map 里缺失就跳过);任务一律在锁外执行以避免任务内再调用 Enqueue/Remove 造成自死锁。
- **ThreadOptions** (class) 〔`runtime/framework/thread_options.h`〕：线程配置的链式 setter 值对象:stack_size(栈大小)、nice_priority_level(nice 优先级)、cpu_set(CPU 亲和性核集合)、name_prefix(线程名前缀)。被 pthread 变体在线程启动时读取并应用。
- **MovableMutexLock** (class) 〔`runtime/framework/resource_management/utils/movable_mutex_lock.h`〕：可移动版的作用域锁(absl::MutexLock 本身不可移动)。构造即 lock、析构若仍持有则 unlock,移动构造把对方 mutex_ 置空。它是 ResourceManager '取资源即上锁、把锁随返回值移交调用方' 模式的关键支撑;带 ABSL_SCOPED_LOCKABLE 注解配合静态线程安全分析。
- **ContextHandler / SharedProcessedContext** (class) 〔`runtime/framework/resource_management/context_handler/context_handler.h`〕：会话上下文句柄。封装一个会话的 RuntimeConfig、RuntimeState、AudioContext,以及通过 shared_ptr 指向的 SharedProcessedContext(真正的 KV-cache/ProcessedContext)。SharedProcessedContext 维护一条 handlers_ 链(handlers[i] 由 handlers[i-1] 克隆而来),配合 LongestHandlerTimeStep 判断当前会话是否最长前缀,从而决定是否需要 copy-on-write,实现多会话共享前缀 KV-cache。

## 数据流
1. 上游 Session 调用 ExecutionManager::AddPrefillTask/AddDecodeTask(...),传入 session_id、自动分配的 task_id、依赖任务集合 dep_tasks、共享取消标志 cancelled 和回调 callback。
2. ThreadedExecutionManager 把真正干活的逻辑打包成一个 task lambda,交给 CreateTask:在 session_and_task_lookup_mutex_ 保护下登记 TaskInfo,逐个解析 dep_tasks——已结束的依赖直接剔除并据其状态决定本任务初始状态(失败/取消会传染),未结束的依赖则把本任务登记进其 following_tasks。
3. 若任务无未完成依赖且状态为 kCreated,CreateTask 立即 QueueTask:把 task lambda 通过 execution_thread_pool_->Schedule 投入执行线程池,状态置 kQueued,并通过 callback 回报 kQueued。
4. execution_thread_pool_ 的 worker 在 RunWorker 循环中取出 task,放锁后执行;task lambda 调用 StartTask(置 kProcessing、取出 SessionInfo/cancelled/callback),期间多次检查 cancelled->load() 以便尽早中止。
5. task lambda 调用 resource_manager_->AcquireExecutorWithContextHandler(context_handler) 取得 LockedLlmExecutor——此处在 executor_mutex_ 下完成上下文切换(KV-cache 共享/克隆),并把锁随句柄移交;随后调用 Tasks::Prefill/Decode 等真正驱动底层 LlmExecutor 推理。
6. 推理产出 Responses 后,task lambda 调用 FinishTask:更新任务状态,把就绪的 following_tasks 通过 QueueTask 推入执行池,失败/取消则沿依赖链用 UpdateAllTasksToState 传播;最终把 callback+responses 投入 callback_thread_pool_ 异步回调上游,并将任务从 session 的 active_tasks 中移除。
7. 上游通过 WaitUntilDone/WaitUntilSessionDone/WaitUntilAllDone(底层是 absl::Mutex::AwaitWithTimeout 或 ThreadPool::WaitUntilDone)阻塞等待任务/会话/全部完成。

## 概念
- **线程池 (Thread Pool)**：预先或按需创建一组工作线程,反复从共享任务队列里取任务执行,避免为每个任务频繁创建/销毁线程的开销。本模块的 ThreadPool 是'懒扩容'式:初始 0 线程,只有在所有现有线程都可能忙时才新建,直到 max_num_threads 上限。
- **任务/工作窃取与依赖图 (Task DAG)**：一个推理请求常被拆成多个有先后顺序的任务(如先 prefill 再 decode)。ExecutionManager 用 dependent_tasks(我依赖谁)和 following_tasks(谁依赖我)两个集合构成有向无环图;只有当一个任务的所有前置依赖完成,它才会被放入队列执行,从而正确表达'prefill 完成后才能 decode'这类约束。
- **异步回调 (Asynchronous Callback)**：任务完成后不直接在执行线程里调用上游回调,而是把回调丢进单独的 callback_thread_pool_。这样推理线程能立刻去做下一个任务,回调里即使有耗时操作也不拖慢推理,同时回调彼此串行、顺序可控。
- **RAII 作用域锁 (Scoped Lock)**：C++ 用对象生命周期管理锁:构造时加锁、析构时自动解锁,杜绝忘记解锁。ResourceScopedLock 和 MovableMutexLock 都是这一思想;MovableMutexLock 进一步让锁可被'移动'给函数返回值,实现'拿到资源句柄=持有锁,句柄销毁=释放锁'。
- **CPU 亲和性与线程优先级 (CPU Affinity / nice)**：在 Linux 上,sched_setaffinity 可把线程绑定到指定 CPU 核(如绑大核以提升推理吞吐),nice() 可调整线程调度优先级。这些是端侧性能调优手段,但只在 Linux/Android 生效,iOS/macOS/Windows 上会被忽略或仅设线程名。
- **类型擦除 (Type Erasure)**：ResourceRegistry 要在同一张表里存放不同类型的资源,却又想在取出时保证类型安全。它用基类 ResourceBase + 模板派生 ResourceHolder<T> 把具体类型'擦除'成统一指针存储,Acquire<T> 时再用 dynamic_cast 还原并校验,类型不符则返回错误。
- **写时复制 (Copy-on-Write) 的 KV-cache**：多个会话可能共享同一段历史 token 的 KV-cache(ProcessedContext)。只要大家只读就共享同一份;当某会话要写入(继续生成)且自己不是最长前缀持有者时,才克隆一份独立的上下文,避免互相污染。ContextHandler/SharedProcessedContext + LockedLlmExecutor 共同实现这一优化以省内存、省重复 prefill。

## 优化
- **线程池懒扩容 (Lazy worker creation)**：ThreadPool 初始不创建任何线程,Schedule 时仅当'现有线程数 <= 活跃+排队任务数'且未达上限才新建一个 worker(threadpool.cc:89-116)。这样空闲时不占线程资源,负载来时才逐步扩容,且第一个 worker 创建失败会作为致命错误上报、后续失败则容忍(已有线程仍能消费)。
- **任务执行时释放锁 (Run task outside the lock)**：ThreadPool::RunWorker 在 mutex_.unlock() 后才执行用户任务、执行完再 lock(threadpool.cc:181-183);ExecutionQueue::WorkerThread 同样在锁外执行(execution_queue.cc:97-102)。既提高并发(执行期间别的线程可继续 Schedule),也避免任务内部回调 Schedule/Enqueue 造成的自死锁。
- **O(1) 任务取消 (Map-based cancellation)**：ExecutionQueue 用 task_order_(只存 id 的 FIFO) 与 pending_tasks_(id→任务体的 hash map) 分离。Remove 只需从 map erase 即 O(1);worker pop 出 id 后若在 map 中找不到就跳过,无需在队列里线性搜索删除(execution_queue.cc:59-94)。
- **可移动锁实现锁的所有权移交 (MovableMutexLock)**：absl::MutexLock 不可移动,ResourceManager 无法把'已加锁状态'随返回的 Locked*Executor 句柄一起传出。MovableMutexLock 通过支持移动语义解决此问题,使调用方'持有句柄即持有锁、句柄析构即解锁',无需手动配对加解锁(movable_mutex_lock.h)。
- **KV-cache 写时复制与 token 复用 (CoW + RemoveMatchingTokens)**：LockedLlmExecutor::Prefill 会先比较待 prefill 的 input_ids 与已处理 token,调用 RemoveMatchingTokens 跳过已处理前缀、只 prefill 增量;并通过 LongestHandlerTimeStep 判断当前会话是否最长前缀持有者,不是则克隆 ProcessedContext 再写,避免污染共享同一前缀的其他会话(resource_manager.cc:208-340)。这显著减少多轮对话/多会话场景下的重复 prefill 与内存拷贝。
- **双单线程池分离执行与回调 (Execution vs Callback pools)**：ThreadedExecutionManager 用两个 max_num_threads=1 的池:一个串行执行推理(保证对同一 Executor 的访问天然有序、无需更细粒度锁),一个串行执行用户回调。推理线程提交完回调即可返回继续下一个任务,回调耗时不阻塞推理流水线(threaded_execution_manager.cc:75-80, 477-489)。
- **按平台编译线程实现 (pthread vs std::thread)**：CMake 根据 WIN32/LITERTLM_USE_STD_THREAD 选择编译 worker_thread_pthread.cc 或 worker_thread_std_thread.cc(CMakeLists.txt:39-43)。pthread 变体可用 CPU 亲和性与 nice 优先级做端侧调优(绑大核等),std::thread 变体保证 Windows 等平台可移植。

## 关键代码片段（待核验 @ v0.13.1）
**ThreadPool 懒创建 worker:仅当线程都可能忙且未达上限时才扩容** — 待核验：`runtime/framework/threadpool.cc:89-118`
```cpp
size_t num_threads = threads_.size();
if (num_threads < max_num_threads_) {
  size_t num_tasks = num_active_tasks_ + tasks_.size();
  if (num_threads <= num_tasks) {
    auto thread = WorkerThread::Create(this, name_prefix_);
    if (thread.ok()) {
      threads_.push_back(std::move(*thread));
    } else if (num_threads == 0) {
      return thread.status();  // 首个线程失败=致命
    }
    // 其余失败容忍:已有 worker 仍可消费
  }
}
tasks_.push_back(std::move(callback));
```
**worker 取任务后释放锁执行,执行完再加锁——提高并发并避免重入死锁** — 待核验：`runtime/framework/threadpool.cc:176-185`
```cpp
auto task_to_run = std::move(tasks_.front());
tasks_.pop_front();
++num_active_tasks_;

// Execute the task with mutex released.
mutex_.unlock();
std::move(task_to_run)();
mutex_.lock();

--num_active_tasks_;
```
**pthread 变体在线程启动时应用 CPU 亲和性(仅 Linux)** — 待核验：`runtime/framework/worker_thread_pthread.cc:116-132`
```cpp
if (!selected_cpus.empty()) {
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  for (const int cpu : selected_cpus) {
    CPU_SET(cpu, &cpu_set);
  }
  if (sched_setaffinity(syscall(SYS_gettid), sizeof(cpu_set_t),
                        &cpu_set) != -1 || errno == 0) {
    ABSL_LOG(INFO) << "Pinned the thread pool executor to processor "
                   << absl::StrJoin(selected_cpus, ", processor ");
  }
}
```
**ExecutionQueue:双结构实现 O(1) 取消 + 锁外执行任务** — 待核验：`runtime/framework/execution_queue.cc:84-101`
```cpp
int id = task_order_.front();
task_order_.pop();
auto it = pending_tasks_.find(id);
if (it != pending_tasks_.end()) {
  current_task = std::move(it->second);
  pending_tasks_.erase(it);
} else {
  continue;  // 已被 Remove() 删除,跳过
}
// Execute the task OUTSIDE the mutex lock.
// This prevents deadlocks if a task itself calls Enqueue() or Remove().
if (current_task) { std::move(current_task)(); }
```
**ResourceManager 用 MovableMutexLock 把锁随句柄移交给调用方** — 待核验：`runtime/framework/resource_management/resource_manager.cc:669-681`
```cpp
absl::StatusOr<std::unique_ptr<LlmExecutor>>
ResourceManager::AcquireExecutor() {
  MovableMutexLock lock(&executor_mutex_);  // 加锁
  if (llm_executor_ == nullptr) {
    return absl::InvalidArgumentError("Llm executor should not be null...");
  }
  // 锁随 LockedLlmExecutor 一起移出:句柄存活=持锁,析构=解锁
  return std::make_unique<LockedLlmExecutor>(llm_executor_, std::move(lock));
}
```

## 入手顺序
- 先读 runtime/framework/thread_options.h(最简单的配置值对象),建立 stack_size/nice/cpu_set 概念。
- 再读 runtime/framework/threadpool.h 与 threadpool.cc:重点看 Schedule 的懒扩容判断(line 90-116)、RunWorker 的取任务-放锁执行循环(line 155-187)、析构关闭流程(line 45-76)。这是整个模块的并发基石。
- 接着读 worker_thread.h 与两个变体 worker_thread_pthread.cc / worker_thread_std_thread.cc,理解 Create 工厂如何按平台分流,以及 pthread 变体里 CPU 亲和性/优先级/线程名的设置(ThreadBody)。
- 读 runtime/framework/execution_queue.{h,cc} 和 resource_registry.h 这两个相对独立的小组件,体会'锁外执行任务避免重入死锁'与'类型擦除 + 每资源一锁'两种模式。
- 读 resource_management/execution_manager.h,把 ExecutionManager 接口、SessionInfo、TaskInfo、依赖/取消字段看明白——这是上层契约。
- 重点精读 resource_management/threaded_execution_manager.cc:跟着 CreateTask→QueueTask→StartTask→FinishTask 的任务状态机和依赖传播走一遍,再看 AddPrefillTask/AddDecodeTask 里 task lambda 如何反复检查 cancelled 并调用 ResourceManager。
- 最后读 resource_management/resource_manager.cc(LockedLlmExecutor + AcquireExecutorWithContextHandler)和 context_handler.h、movable_mutex_lock.h,理解共享 Executor 的加锁移交与 KV-cache 写时复制优化。对照 serial_execution_manager.h 看单线程版如何用 ready_queue_ 替换线程池。
