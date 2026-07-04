# 模块素材：核心调度 (Core Pipeline)  `core-pipeline`

> 来源：litert-lm-guide/data.json（多智能体源码分析）。**这是素材，不是正文**。
> 引用进正文前，须按 CLAUDE.md 第二节逐条 Read 源码核验、补 `@ v0.13.1`。

**一句话**：runtime/core/ 是 LiteRT-LM 的推理编排核心：把一次生成请求拆解成 prefill（一次性吞入提示词）和 decode（逐 token 自回归生成）两个阶段，并把 tokenizer、LlmExecutor、sampler、logits processor、stop token detector 串成一条完整的推理流水线。SessionAdvanced 提供有状态的多轮会话/克隆/检查点能力，Tasks 实现真正的 prefill/decode/score 算法，pipeline.h 则是这套算法对外的稳定函数式封装。

**在架构中的位置**：它处于「上层 Engine/Session 接口」与「下层 LlmExecutor（实际跑模型）」之间的中间编排层。上游：EngineAdvancedImpl 创建 Tokenizer、LlmExecutor、ExecutionManager 并实例化 SessionAdvanced；用户通过 SessionInterface（GenerateContent/RunPrefill/RunDecode）发起请求。本层：SessionAdvanced 负责会话状态机与输入预处理，把任务投递给 ExecutionManager；ExecutionManager（serial/threaded）在工作线程里回调 Tasks::Prefill / Tasks::Decode / Tasks::Score 真正执行算法。下游：Tasks 调用 LlmExecutor 的 Prefill/Decode/DecodeLogits 跑模型，调用 Tokenizer 做编解码，调用 Sampler/LogitsProcessor/StopTokenDetector 做采样与停止判定。pipeline.h 是一层与 ExecutionManager 解耦的纯函数封装，主要给测试和需要直接编排的调用方用。

## 关键文件
- `/Users/dpthinker/workspace/LiteRT-LM/runtime/core/tasks.cc` — 流水线算法核心实现（DecodeOneStep + Prefill/Decode/Score）
- `/Users/dpthinker/workspace/LiteRT-LM/runtime/core/session_advanced.cc` — 会话级 prefill/decode 编排与状态机、克隆、检查点
- `/Users/dpthinker/workspace/LiteRT-LM/runtime/core/session_utils.cc` — 输入预处理：套对话模板、文本转 token、内容归一化
- `/Users/dpthinker/workspace/LiteRT-LM/runtime/core/pipeline.h` — 对外稳定的函数式流水线 API
- `/Users/dpthinker/workspace/LiteRT-LM/runtime/core/engine_advanced_impl.cc` — 引擎组装与 SessionAdvanced 实例化入口

## 核心抽象
- **Tasks::Decode** (function)：decode 阶段的核心驱动函数（tasks.cc:462）。维护 while(true) 自回归循环：每轮调用 DecodeOneStep::Run 得到新 token，累积/流式上报文本与 token_ids，调用 GetCurrentStep 计步，再用 ShouldStop 判断是否结束。通过参数组合支持四种模式：内部采样/外部采样 × 阻塞返回 Responses/流式 callback。还负责 benchmark 打点、cancelled 原子量轮询、外部采样结束时补一次 prefill 把停止 token 写回 KV cache。
- **DecodeOneStep** (class)：tasks.cc:114 的私有类，封装「一个 decode step」的全部细节。构造时按需装配 RepetitionPenaltyProcessor 和 ConstrainedDecoder 到 logits_processors_。Run() 内部先 DecodeAndSample（内部采样走 executor_.Decode；外部采样走 executor_.DecodeLogits→logits processor→sampler→tokenizer），再把 token id 经 StopTokenDetector::ProcessTokens 检测停止、用 MergeTokenIds 处理跨 step 的 BPE 半截 token、用队列缓冲处理「部分匹配停止词」的回吐逻辑。RunScoreStep 则为打分场景计算逐 token 的对数似然。
- **SessionAdvanced** (class)：SessionInterface 的高级实现（session_advanced.h:47），代表一次有状态会话。持有会话状态机 SessionState、与 ExecutionManager 的 weak_ptr、tokenizer 指针和 last_task_ids_（任务依赖链）。RunPrefillAsync 做模板化+预处理后投递 AddPrefillTask；RunDecodeAsync 在真正 decode 前先补一次空内容的 kLast 模板 prefill，再投 AddDecodeTask。GenerateContent = RunPrefill + RunDecode。还实现 Clone（会话克隆）、SaveCheckpoint/RewindToCheckpoint/RewindToStep（按 step 回退）等高级特性。所有可变状态用 mutex_ 保护。
- **Tasks::Prefill** (function)：prefill 阶段实现（tasks.cc:429）。校验输入 token 数不超过 max_num_tokens，构造 ExecutorPrefillParams（benchmark 时强制 wait_for_completion），调用 executor.Prefill 把整段提示词一次性吞进模型并填充 KV cache，记录 benchmark 时间，返回 TaskState::kDone。pipeline.h 的 Prefill 在此基础上额外返回 prefill 最后一个 token id 作为后续 decode 的起点。
- **Tasks::Score** (function)：文本打分实现（tasks.cc:644）。在 prefill 之后，对给定 target_text 逐 token 喂入 executor 并用 RunScoreStep 取 logits，累加每个 token 的负对数概率得到困惑度类分数。用「padding 到批内最长目标 + 跳过 padding 步」的技巧支持批量目标，可选返回 token_lengths 与逐 token 分数。
- **ApplyPromptTemplates / PreprocessContents** (function)：session_utils.cc 的两个预处理主力。ApplyPromptTemplates 按 ContentType{kFirst,kMiddle,kLast,kNA} 给内容套对话模板（user.prefix/suffix + model.prefix），首轮补 BOS，并禁止用户输入里夹带 BOS 控制 token。PreprocessContents 把文本经 StringToProcessedInputText 转成 token-id TensorBuffer，把已预处理的图像/音频 TensorBuffer 透传，统一成可投递给 executor 的 InputData 序列。
- **AdvancedTaskController** (class)：SessionInterface::TaskController 的实现（session_advanced.h:49）。持有 task_id、共享的 cancelled 原子量、ExecutionManager 的 weak_ptr。WaitUntilDone 转发到 ExecutionManager 等待该任务完成；Cancel 把 cancelled 置 true，让 Tasks::Decode 的循环在下一轮检测到并提前退出。它是异步 API 返回给调用方的句柄。
- **ShouldStop** (function)：tasks.cc:89 的纯函数，集中表达 decode 循环的所有停止条件：命中停止 token、达到 benchmark 指定步数、current_step 触及 KV cache 上限 max_num_tokens、num_decoded_steps 触及用户 max_output_tokens。把停止判定从主循环里抽出来，便于阅读与测试。

## 数据流
1. 用户调用 Session::GenerateContent(contents) 或 GenerateContentStream(contents, callback)，contents 是 InputText/InputImage/InputAudio 的序列。
2. SessionAdvanced::RunPrefillAsync 判断是否首轮（SessionState==kFresh）决定 ContentType，调用 ApplyPromptTemplates 套对话模板，再用 PreprocessContents 把文本经 tokenizer 转成 token-id TensorBuffer。
3. 预处理后的 InputData 经 ExecutionManager::AddPrefillTask 投递；会话状态置为 kPrefilled，last_task_ids_ 更新为本次 prefill 任务，形成任务依赖链。
4. ExecutionManager 在工作线程取出任务，AcquireExecutor 后调用 Tasks::Prefill，executor.Prefill 把整段提示词一次性写入 KV cache，记录最后一个 token id（作为 decode 起点）。
5. prefill 完成回调触发后，SessionAdvanced::RunDecodeAsync 先（若启用模板）补投一次 kLast 空内容 prefill 收尾提示词，状态置 kDecoded，再调用 AddDecodeTask 把 RepetitionPenaltyConfig/Constraint/max_output_tokens 一起传下。
6. Tasks::Decode 构造 DecodeOneStep（按需挂上 RepetitionPenaltyProcessor、ConstrainedDecoder），进入 while 循环：每轮先查 cancelled，再 Run 一步。
7. DecodeOneStep::Run 内部：内部采样走 executor.Decode(decode_params) 一步出 token；外部采样走 executor.DecodeLogits→各 LogitsProcessor.ProcessLogits→sampler.SampleToIdAndScoreBuffer→tokenizer 反查 token id。
8. 得到新 token 后：StopTokenDetector.ProcessTokens 检测停止词；tokenizer.MergeTokenIds 拼接跨 step 的 BPE 半截 token；TokenIdsToTexts 解码为文本；用队列缓冲处理「部分停止词」的延迟输出/回吐。
9. Tasks::Decode 把本步文本/token_ids（流式时 step 级、阻塞时累加）整理好；流式模式立即 callback(Responses(kProcessing,...))，阻塞模式累积到 final_texts。
10. 每轮末用 GetCurrentStep 计步并调用 ShouldStop 判断；满足任一停止条件则跳出循环。外部采样在结束/取消时再补一次 prefill 把停止 token 写回 KV cache。
11. 循环结束返回 Responses（kDone 或 kMaxNumTokensReached）；阻塞路径的 RunDecode 同步回调里累加各 step 文本/分数并按 token 数归一化，最终把完整结果返回给用户。

## 概念
- **Prefill 与 Decode 两阶段**：自回归 LLM 推理分两步：Prefill 把整段输入提示词一次性喂进模型，并行计算所有位置、填满 KV cache，是计算密集型；Decode 之后每次只喂入上一步生成的 1 个 token，逐个吐字，是访存密集型。本模块的 Tasks::Prefill 对应前者，Tasks::Decode 的循环对应后者，这是整个流水线编排的主干。
- **内部采样 vs 外部采样 (internal/external/custom sampling)**：内部采样指采样逻辑由 LlmExecutor（通常在硬件/编译模型里）自己完成，调用 executor.Decode 直接拿到 token；外部采样（custom sampling）指 executor 只吐 logits（DecodeLogits），由运行时这边的 Sampler 在 CPU 上采样。后者更灵活（可插自定义采样、拿到分数），但需要把采到的 token 用一次额外 prefill 写回模型，所以代码里多处出现『结束/取消时补一次 Prefill』。
- **KV Cache 与 current_step**：Transformer 推理会缓存历史 token 的 Key/Value 向量避免重复计算，这块缓存有最大容量（max_num_tokens）。executor 的 current_step 表示当前缓存里已有多少 token。模块用 GetCurrentStep/SetCurrentStep 来计步、判断是否撑爆缓存（停止条件之一），以及在提前停止或回退检查点（RewindToStep）时回滚缓存位置。
- **BPE 半截 token 与停止词的流式难题**：BPE 分词下一个字/词可能由多个 token 拼成，单独解码某个 token 可能得到不完整的 UTF-8/词片。代码用 IsIncompleteBpeSequence + MergeTokenIds 把跨 step 的半截 token 攒起来再解码。停止词检测同理：一个停止短语可能横跨多个 token，所以用 pending_stop_tokens_ 队列缓冲『可能是停止词前缀』的文本，确认不是才吐出，避免把停止词泄漏给用户。
- **会话状态机与任务依赖链**：SessionAdvanced 用 SessionState{kFresh,kPrefilled,kDecoded} 约束调用顺序（必须先 prefill 才能 decode），用 last_task_ids_ 把连续任务串成依赖关系交给 ExecutionManager，从而在多线程下保证同一会话内 prefill→decode 的执行顺序，同时允许不同会话并行。
- **Prompt Template（对话模板）**：指令微调模型期望输入按固定格式包裹，如『<user>...內容...</user><model>』。ApplyPromptTemplates 在首/中/末片段插入 user.prefix/suffix 与 model.prefix，并在首轮补 BOS（句首特殊 token）。decode 前还会补一个 kLast 模板来闭合用户回合、开启模型回合。

## 优化
- **流式回调与逐 step 增量上报**：Tasks::Decode 检测到非空 callback 即进入流式模式，每个 decode step 生成内容后立刻通过 callback 上报 Responses(kProcessing)，让上层边生成边显示，而非等全部生成完。阻塞模式则在内部累加，二者复用同一份循环代码。
- **部分停止词的延迟缓冲**：用 pending_stop_tokens_ / pending_stop_token_ids_ 队列只保留最近 max_partial_stop_token_length 个 token 做缓冲，超出部分才真正吐出。既能在停止词跨多 token 时正确截断，又避免缓冲整段输出，内存占用受限于停止词长度。
- **提前停止时回滚 executor 步数**：DecodeOneStep::Run 中若在一批 token 处理到一半就 AllDone，会计算 diff 并 SetCurrentStep(current_step - diff)，把多算进 KV cache 的步数回退，保证缓存状态与实际输出一致，便于后续续写或打分。
- **tokenizer 与 executor 并行加载**：engine_advanced_impl.cc 在新格式模型（有 LlmModelType）下用 std::async 把 create_tokenizer 放到独立线程，与模型/executor 加载并行（受 GetParallelFileSectionLoading 控制），缩短引擎初始化时间。
- **benchmark 细粒度打点**：Tasks 各阶段用 BenchmarkInfo::TimeMarkDelta 对 executor_decode、sampling、executor_decode_and_sample 等子阶段分别计时，并支持 num_decode_tokens 固定步数压测，方便定位是模型前向还是采样成为瓶颈。
- **外部采样 logits 的 fp16/fp32 自适应**：RunScoreStep/DecodeAndSample 检测 logits 的 ElementType，fp32 优先零拷贝 ReferTensorBufferAsSpan，fp16 则 ConvertFp16ToFp32，兼容不同后端输出精度且尽量减少拷贝。

## 关键代码片段（待核验 @ v0.13.1）
**decode 主循环：逐步采样、流式上报、停止判定** — 待核验：`runtime/core/tasks.cc:500-592`
```cpp
DecodeOneStep run_one_step(&executor, &tokenizer, num_output_candidates,
                           stop_token_detector, benchmark_info, sampler,
                           repetition_penalty_config, constraint);
while (true) {
  if (cancelled != nullptr && cancelled->load()) { /* ...收尾... */ return absl::CancelledError("Process cancelled."); }
  absl::StatusOr<bool> all_done = run_one_step.Run(std::move(decoded_ids_to_use));
  // ...整理 step 文本/token_ids...
  if (is_streaming && any_updates) {
    callback(Responses(TaskState::kProcessing, std::move(step_texts), ...));
  }
  ASSIGN_OR_RETURN(int current_step, executor.GetCurrentStep());
  int num_decode_steps = current_step - executor_step_before_decode;
  if (ShouldStop(*all_done, benchmark_decode_token_count, num_decode_steps,
                 current_step, max_num_tokens, max_output_tokens)) { break; }
}
```
**一步 decode：内部采样直接出 token，外部采样走 logits→processor→sampler** — 待核验：`runtime/core/tasks.cc:334-397`
```cpp
if (sampler_) {  // External sampling path
  ExecutorInputs inputs(ExecutorTextData(std::move(duplicate_decoded_ids)), std::nullopt, std::nullopt);
  ASSIGN_OR_RETURN(auto output_logits, executor_.DecodeLogits(inputs));
  for (LogitsProcessor* lp : logits_processors_) { RETURN_IF_ERROR(lp->ProcessLogits(output_logits)); }
  RETURN_IF_ERROR(sampler_.value()->SampleToIdAndScoreBuffer(output_logits, decoded_ids.value(), &scores_tensor_));
  return tokenizer_.TensorBufferToTokenIds(decoded_ids.value());
} else {  // Internal sampling path
  if (!logits_processors_.empty()) {
    auto decode_params = ExecutorDecodeParams();
    decode_params.SetLogitsProcessorList(logits_processors_);
    ASSIGN_OR_RETURN(output_tokens, executor_.Decode(decode_params));
  } else { ASSIGN_OR_RETURN(output_tokens, executor_.Decode()); }
  return output_tokens;
}
```
**会话编排：prefill 套模板+预处理后投递任务，并更新状态机与依赖链** — 待核验：`runtime/core/session_advanced.cc:109-132`
```cpp
bool is_first_turn = session_state_ == SessionState::kFresh;
// ...根据 first_turn / 模板配置确定 content_type...
ASSIGN_OR_RETURN(std::vector<InputData> templated_contents,
    ApplyPromptTemplates(contents, content_type, session_info_->session_config, *tokenizer_, is_first_turn));
ASSIGN_OR_RETURN(preprocessed_contents,
    PreprocessContents(templated_contents, session_info_->session_config, *tokenizer_, session_info_->benchmark_info));
ASSIGN_OR_RETURN(auto task_id, execution_manager_lock->GetNewTaskId());
RETURN_IF_ERROR(execution_manager_lock->AddPrefillTask(
    session_id_, task_id, std::move(preprocessed_contents), last_task_ids_, cancelled, std::move(callback)));
session_state_ = SessionState::kPrefilled;
last_task_ids_ = {task_id};
```
**集中表达的 decode 停止条件** — 待核验：`runtime/core/tasks.cc:89-110`
```cpp
bool ShouldStop(bool hit_stop_tokens, int benchmark_decode_token_count,
                int num_decoded_steps, int current_step, int max_num_tokens,
                int max_output_tokens) {
  if (hit_stop_tokens && benchmark_decode_token_count == 0) return true;
  else if (benchmark_decode_token_count > 0 && num_decoded_steps >= benchmark_decode_token_count) return true;
  else if (current_step >= max_num_tokens) return true;          // KV cache 用尽
  else if (num_decoded_steps >= max_output_tokens) return true;  // 达到用户上限
  return false;
}
```
**GenerateContent = Prefill 然后 Decode 的最简编排** — 待核验：`runtime/core/session_advanced.cc:351-355`
```cpp
absl::StatusOr<Responses> SessionAdvanced::GenerateContent(
    const std::vector<InputData>& contents) {
  RETURN_IF_ERROR(RunPrefill(contents));
  return RunDecode();
}
```

## 入手顺序
- 先读 runtime/engine/engine.h 里的 SessionInterface，了解 GenerateContent/RunPrefill/RunDecode 这些对外契约——这是本模块要实现的目标。
- 读 runtime/core/session_advanced.cc 的 GenerateContent → RunPrefillAsync → RunDecodeAsync，看一次请求如何被拆成 prefill/decode 两个任务并投给 ExecutionManager，注意 SessionState 状态流转和 last_task_ids_ 依赖链。
- 读 runtime/core/session_utils.cc 的 ApplyPromptTemplates 与 PreprocessContents，理解文本如何套模板、经 tokenizer 变成 token-id TensorBuffer。
- 进入 runtime/core/tasks.cc，先看自由函数 Prefill（最简单），再看 Decode 的 while 主循环骨架，重点看 ShouldStop 的停止条件。
- 深入 DecodeOneStep 类：先看 DecodeAndSample 区分内部/外部采样两条路径，再看 Run() 里 BPE 合并与停止词缓冲的处理。
- 最后回看 runtime/core/pipeline.h / pipeline.cc，理解它是 Tasks::* 的薄封装；并扫一眼 serial_execution_manager.cc 中 AddPrefillTask/AddDecodeTask 如何在工作线程里调用 Tasks::Prefill/Tasks::Decode，把整条链路闭合。
- 想看完整用法可参考 runtime/core/pipeline_test.cc 与 session_advanced_test.cc，里面有大量端到端的调用示例。
