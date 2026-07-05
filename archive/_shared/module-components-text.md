# 模块素材：文本组件: 分词与采样 (Text Components: Tokenization & Sampling)  `components-text`

> 来源：litert-lm-guide/data.json（多智能体源码分析）。**这是素材，不是正文**。
> 引用进正文前，须按 CLAUDE.md 第二节逐条 Read 源码核验、补 `@ v0.13.1`。

**一句话**：该模块负责 LLM 文本流水线的两端: 入口的 tokenization(文本↔token id 互转, 支持 SentencePiece 与 HuggingFace 两种后端、Jinja 提示模板渲染), 以及出口的 token 选择(top-k/top-p/temperature/greedy 采样、logits 处理如重复惩罚与约束解码、停止符检测)。它把模型输出的 logits 张量转化为最终生成的 token 序列, 再解码回文本。

**在架构中的位置**：位于 runtime/components/ 下, 是 Engine 与 Executor 之间的可复用基础组件层。上游是 Engine/Pipeline(组装会话、读取模型资源里的 tokenizer 与提示模板), 下游被 Executor(如 llm_litert_compiled_model_executor.cc)在 decode 循环中调用: Executor 每步产出 logits → LogitsProcessor 就地改写 logits → Sampler 采样出 token id → StopTokenDetector 判断是否停止 → Tokenizer 把 token id 流式解码成文本返回给上层。采样既有纯 CPU 实现(TopPSampler), 也通过 sampler_factory 经 C API 动态加载 GPU(OpenCL/WebGPU/Metal)实现, GPU 不可用时回退 CPU。

## 关键文件
- `runtime/components/tokenizer.h` — Tokenizer 抽象基类, 定义 TextToTokenIds / TokenIdsToText 等接口, 并提供 TokenIds↔TensorBuffer 转换、批处理解码、BPE 不完整序列检测(HasBpeSuffix)等静态工具。
- `runtime/components/sentencepiece_tokenizer.cc` — 基于 SentencePiece 的分词实现, 重点在 TokenIdsToText 里处理 byte token 的分块解码以支持流式输出与前导空格保留。
- `runtime/components/huggingface_tokenizer.cc` — 基于 tokenizers-cpp(Rust 绑定)的 HuggingFace 分词实现, 从 JSON 加载, 用 mmap 读取文件。
- `runtime/components/sampler.h` — Sampler 抽象接口, 核心方法 SampleToIdAndScoreBuffer 把 logits 张量采样为 id 张量(可选输出 score), 并支持 sampler 自行填充输入张量的 input-handling 模式。
- `runtime/components/sampling_cpu_util.cc` — 采样的纯算法核心: TopKTokenIds(分三档优化的 top-k)、Softmax(数值稳定 + 边界处理)、TopKTopPSampling(top-k→softmax→top-p 截断→按概率采样)。
- `runtime/components/top_p_cpu_sampler.cc` — Sampler 接口的 CPU 实现 TopPSampler, 负责张量校验、FP16→FP32 转换, 并调用 sampling_cpu_util 完成采样, 把 score 转成 log 概率写回。
- `runtime/components/sampler_factory.cc` — CreateSampler 工厂, 按 Backend 与 SamplerParameters 分派 CPU/GPU; GPU 端通过 dlopen 动态加载 OpenCL/WebGPU/Metal 的 C API, 失败时回退 CPU。
- `runtime/components/stop_token_detector.cc` — StopTokenDetector, 对每个 batch 流独立跟踪多条停止 token 序列的匹配进度, 支持多 token 停止序列与部分匹配长度查询。
- `runtime/components/prompt_template.cc` — PromptTemplate 用 Rust 实现的 Minijinja 渲染聊天模板; EditTemplateForMinijinja 用 RE2 把 Python Jinja 习惯用法重写成 Minijinja 兼容语法。
- `runtime/components/logits_processor/logits_processor.h` — LogitsProcessor 抽象接口, 在采样前就地改写 logits(ProcessLogits)、采样后更新内部状态(UpdateState), 是重复惩罚与约束解码的共同基类。
- `runtime/components/logits_processor/repetition_penalty_processor.cc` — RepetitionPenaltyProcessor, 同时支持乘法重复惩罚(HF 风格)与加法 presence/frequency 惩罚(OpenAI 风格), 用滑动窗口 + 计数 hashmap 跟踪历史。
- `runtime/components/logits_processor/constrained_decoding/constrained_decoder.cc` — ConstrainedDecoder, 用 Constraint 状态机为每步计算允许 token 的 bitmap, 把不允许的 token logit 置为 -inf, 实现语法/JSON Schema 约束解码。
- `runtime/components/token_id_util.cc` — token id 工具: PreprocessTokenIds(前置 start token 并做上下文长度检查)与 StopTokenFound(简单的单 token 停止判断)。

## 核心抽象
- **Tokenizer** (class) 〔`runtime/components/tokenizer.h`〕：分词器抽象基类。纯虚接口: TextToTokenIds(编码)、TokenIdsToText(解码, 不完整 BPE 序列返回 DataLossError)、TokenToId(原始查表)、GetVocabSize/GetTokens。还提供一组 static 工具: TokenIdsToTensorBuffer / TensorBufferToTokenIds 做 TokenIds 与 [batch, len] TensorBuffer 的互转, TokenIdsToTexts 做批量解码, HasBpeSuffix/IsIncompleteBpeSequence 用 U+FFFD 替换字符判断 BPE 是否需要更多 token 才能解码(流式解码的关键)。
- **Sampler** (class) 〔`runtime/components/sampler.h`〕：采样器抽象接口。核心 SampleToIdAndScoreBuffer(logits[batch,seq,vocab] → ids[batch,seq], 可选 scores 为 log 概率)。UpdateConfig 支持运行时更新 k/p/temperature 与共享随机引擎。CanHandleInput/HandlesInput/SetInputTensorsAndInferenceFunc 是为 GPU 后端设计的扩展: sampler 可自行填充下一步输入张量并异步触发推理, 从而把采样与推理流水线化。
- **TopPSampler** (class) 〔`runtime/components/top_p_cpu_sampler.h`〕：Sampler 的 CPU 实现。Create 时校验 k>0、p∈[0,1]、temperature≥0。SampleToIdAndScoreBuffer 负责: 校验张量维度/batch、把 FP32 或 FP16 logits 取到 host span(FP16 走 ConvertFp16ToFp32)、调用 TopKTopPSampling、把扁平化结果写回 ids_tensor、并对 score 取 std::log 写回 scores_tensor。logits_data_ 作为成员缓存避免每步重分配。
- **TopKTopPSampling** (function) 〔`runtime/components/sampling_cpu_util.h`〕：采样算法主入口。流程: TopKTokenIds 选出 top-k 候选 → Softmax 在候选上求带温度的概率 → 按概率降序排序 top-k → 累加概率做 top-p 截断(cumulative_prob≥p 停止)→ 在 [0,cumulative_prob] 上均匀采样选 token。k==1 直接走贪心(score=1.0); cumulative_prob 近 0 时回退到最高概率 token。返回 [batch,seq] 的采样 id, sampled_scores 输出近似概率。
- **Softmax / TopKTokenIds** (function) 〔`runtime/components/sampling_cpu_util.h`〕：TopKTokenIds 按 k 大小分三档: k==1 用 max_element 贪心; k≤1024 用大小为 k 的最小堆 O(N log k); 否则用 nth_element O(N) 部分排序。Softmax 在 top-k 候选上做减最大值的数值稳定 softmax, temperature 钳到 epsilon 模拟贪心, 并处理 sum 为 0(退化为均匀分布)和 sum 为 inf(温度极小时塌缩到 argmax)两种边界。
- **LogitsProcessor** (class) 〔`runtime/components/logits_processor/logits_processor.h`〕：logits 后处理抽象接口, 在采样前后串入 decode 循环。ProcessLogits 就地改写 logits(支持 TensorBuffer / float span / half span 三种重载, 要求 seq_len==1), UpdateState 在 token 被选中后更新内部状态。GetConstraintDecoder 暴露底层约束解码器。RepetitionPenaltyProcessor 与 ConstrainedDecoder 都实现该接口。
- **RepetitionPenaltyProcessor** (class) 〔`runtime/components/logits_processor/repetition_penalty_processor.h`〕：重复惩罚处理器。每个 batch 维护 BatchState{token_counts 计数 hashmap, token_history 环形缓冲}。ProcessLogitsImpl 对计数>0 的 token 先做乘法 repetition_penalty(正 logit 除、负 logit 乘), 再减去 presence_penalty + count*frequency_penalty。UpdateState 把新 token 入环形窗口, 并把滑出窗口的旧 token 计数减 1, 实现固定 window_size 的滑动惩罚。
- **ConstrainedDecoder / Constraint** (class) 〔`runtime/components/logits_processor/constrained_decoding/constrained_decoder.h`〕：约束解码器, LogitsProcessor 的实现。持有一个 Constraint(状态机)并为每个 batch 维护 Constraint::State。ProcessLogits 调用 Constraint::ComputeBitmap 得到允许 token 的 Bitmap, 把不允许的 token logit 置 -inf(FP16 用 half::min)。UpdateState 调 ComputeNext 推进状态, 到达终态则重置回 Start。Constraint 提供 Start/IsEnded/ComputeNext/ComputeBitmap, 由 ConstraintProvider 创建(具体实现包括基于 llguidance 的 LlgConstraint, 用于 JSON Schema / 语法约束)。
- **StopTokenDetector** (class) 〔`runtime/components/stop_token_detector.h`〕：停止序列检测器。支持动态 AddStopTokenSequence 添加多条(可多 token)停止序列, 为每个 batch 项 × 每条序列独立跟踪匹配进度 batch_item_match_progress_。ProcessTokens 逐 token 推进匹配, 完全匹配则置 stop_token_found_; AllDone 判断全 batch 是否都停止。GetStepsBeforeStopTokens / MaxPartialStopTokenLength 供解码层裁掉停止序列对应的尾部 token。
- **PromptTemplate** (class) 〔`runtime/components/prompt_template.h`〕：聊天提示模板渲染器, 底层是 Rust 的 Minijinja(经 cxx 桥接)。Apply 接收 PromptTemplateInput(messages/tools/add_generation_prompt/bos_eos_token/now 等)渲染成最终 prompt 字符串。构造时 EditTemplateForMinijinja 用 RE2 把 Python Jinja 习惯(startswith/split/strip/items 等)重写为 Minijinja 兼容写法, 并从模板源自动推断 PromptTemplateCapabilities(是否支持 tools/system role/单轮等)。
- **CreateSampler** (function) 〔`runtime/components/sampler_factory.h`〕：采样器工厂函数。按 Backend 分派: GPU 走 CreateGpuSampler(按环境选项在 Metal/WebGPU/OpenCL 间选择, 经 dlopen 加载 LiteRtTopK*Sampler C API), GPU 不可用(kUnavailable)时 fallthrough 到 CPU; CPU 走 CreateCpuSampler, 目前 TOP_P 类型映射到 TopPSampler。SamplerParameters 类型未指定时返回 nullptr, 表示由 Executor 自带采样。

## 数据流
1. 输入侧: 上层把对话 messages/tools 交给 PromptTemplate.Apply, 经 Minijinja 渲染成 prompt 字符串。
2. Tokenizer.TextToTokenIds 把 prompt 编码为 TokenIds(SentencePiece 或 HuggingFace); 必要时 PreprocessTokenIds 前置 start token 并检查上下文长度。
3. TokenIdsToTensorBuffer 把 TokenIds 转成 [1, num_tokens] 的 TensorBuffer 送入 Executor 做 prefill。
4. Decode 循环每步: Executor 产出 logits 张量 [batch, 1, vocab]。
5. (可选)LogitsProcessor 链就地改写 logits: RepetitionPenaltyProcessor 惩罚重复, ConstrainedDecoder 把违反约束的 token 置 -inf。
6. Sampler.SampleToIdAndScoreBuffer 采样: TopPSampler 取 logits(FP16 先转 FP32)→ TopKTokenIds → Softmax → top-p 截断 → 概率采样, 得到 token id 与 log 概率 score。
7. 采样后 LogitsProcessor.UpdateState 用选中 token 更新内部状态(计数窗口 / 约束状态机)。
8. StopTokenDetector.ProcessTokens 用新 token 推进停止序列匹配; AllDone 为真则结束生成。
9. 输出侧: Tokenizer.TokenIdsToText 把 token id 流式解码成文本; 若返回 DataLossError(BPE 未完整)则等待更多 token 再解码, 并按 GetStepsBeforeStopTokens 裁掉停止序列尾部。

## 概念
- **Tokenization / 分词**：把自然语言文本切成模型能处理的离散 token, 并映射为整数 id 的过程。LLM 不直接看字符, 而是看 token id 序列。本模块支持两种主流方案: SentencePiece(常见于 Gemma 等)和 HuggingFace tokenizers(JSON 描述的 BPE/WordPiece)。
- **BPE 与不完整序列**：BPE(Byte-Pair Encoding)会把一个 UTF-8 字符拆成多个字节 token。流式解码时, 若当前 token 只是某个多字节字符的一部分, 解码结果会以替换字符 U+FFFD 结尾。代码用 HasBpeSuffix 检测到这种情况就返回 DataLossError, 让调用方先缓存、等更多 token 到齐再解码, 避免输出乱码。
- **Logits 与 Softmax**：logits 是模型最后一层对每个词表 token 输出的未归一化分数。Softmax 把 logits 指数化并归一化成概率分布。Softmax 前通常先减去最大 logit(数值稳定), 并除以 temperature 调节分布尖锐程度。
- **Temperature(温度)**：缩放 logits 的超参数。温度越低分布越尖锐(越接近贪心), 越高越平坦(越随机)。代码把温度钳到一个极小正数来逼近贪心, 并专门处理温度极小导致 exp 溢出为 inf 的情况(直接塌缩到 argmax)。
- **Top-k 与 Top-p(nucleus)采样**：Top-k 只在概率最高的 k 个 token 中采样。Top-p(核采样)按概率降序累加, 取累计概率刚超过阈值 p 的最小 token 集合再采样。本模块先做 top-k 再在其内做 top-p 截断, 兼顾质量与多样性; k==1 即退化为 greedy(贪心, 总取最高分)。
- **重复惩罚 (Repetition / Presence / Frequency Penalty)**：为抑制模型反复输出相同内容而对已生成 token 的 logit 施加惩罚。乘法 repetition penalty(HF 风格)按是否出现过缩放 logit; 加法 presence penalty 对出现过的 token 减一个常数; frequency penalty 按出现次数线性减。可配滑动窗口只惩罚最近 N 个 token。
- **约束解码 (Constrained Decoding)**：强制模型输出符合某种形式(如合法 JSON、特定语法、工具调用格式)。做法是每步用一个状态机(Constraint)算出当前合法的 token 集合(bitmap), 把非法 token 的 logit 设成 -inf, 这样采样永远只会选到合法 token。本仓库底层接入了 llguidance 来支持 JSON Schema/语法。
- **停止符检测 (Stop Token Detection)**：判断生成何时该结束。除了单个 EOS token, 还可能是多 token 的停止序列(如某段特殊标记)。StopTokenDetector 为每个 batch 流独立跟踪每条停止序列的匹配进度, 支持部分匹配, 完全匹配后通知上层停止并裁掉对应尾部 token。
- **Prompt Template(聊天模板)**：把结构化的多轮对话(role/content)和工具定义按模型约定的格式拼成单个输入字符串的模板, 通常用 Jinja2 写。本模块用 Rust 的 Minijinja 渲染, 并把 HuggingFace 模板里的 Python 习惯用法重写成 Minijinja 兼容语法。

## 优化
- **Top-k 三档算法选择**：TopKTokenIds 按 k 大小切换策略: k==1 用 std::max_element 走纯贪心; k≤1024 用大小为 k 的最小堆(O(N log k), 只为 top-k 维护排序); k 很大时用 std::nth_element 做平均 O(N) 的部分排序。避免对整个词表(数万到十几万)做全排序。
- **数值稳定 Softmax 与边界处理**：Softmax 先减去每行最大 logit 再 exp, 防止溢出; temperature 钳到 epsilon 模拟贪心; 专门处理 sum==0(退化为均匀分布)与 sum==inf(温度极小时塌缩到 argmax)两种数值病态, 避免 NaN/inf 传播。
- **logits 缓冲复用**：TopPSampler 把 logits_data_ 作为成员变量缓存, 跨多次采样调用复用同一块内存, 避免每个 decode step 都重新分配 vector(在逐 token 解码的高频循环里显著降分配开销)。
- **重复惩罚滑动窗口 + 增量计数**：RepetitionPenaltyProcessor 用环形缓冲(token_history)+ 计数 hashmap(token_counts), UpdateState 时新 token 计数加 1、滑出窗口的旧 token 计数减 1(减到 0 即从 map 删除)。窗口惩罚做到 O(1) 摊还更新, 无需每步重扫历史。
- **零拷贝引用 logits 张量**：TopPSampler 优先用 ReferTensorBufferAsSpan 直接引用 host 内存里的 logits, 只有当数据不在 host(如 GPU 内存)时才 CopyFromTensorBuffer 下载, 减少不必要的拷贝。
- **GPU 采样动态加载与回退**：sampler_factory 通过 SharedLibrary/dlopen 按需加载 OpenCL/WebGPU/Metal 采样 C API(也支持静态符号), 任一不可用(kUnavailable)就逐级回退, 最终回退到 CPU TopPSampler, 保证在缺少 GPU 库的环境也能运行。
- **mmap 加载分词器文件**：HuggingFaceTokenizer.CreateFromFile 用 MemoryMappedFile 内存映射读取 tokenizer JSON, 避免一次性整文件读入的开销。

## 关键代码片段（待核验 @ v0.13.1）
**采样主流程: top-k → softmax → top-p 截断 → 概率采样** — 待核验：`runtime/components/sampling_cpu_util.cc:285-326`
```cpp
double cumulative_prob = 0.0;
int final_sample_size = 0;
for (int i = 0; i < k; ++i) {
  cumulative_prob += (*probabilities)[b][s * k + index_of_topk[i]];
  final_sample_size = i + 1;
  if (cumulative_prob >= p) break;  // 找到满足 top-p 的最小集合
}
// ... 在 [0, cumulative_prob] 上均匀采样
std::uniform_real_distribution<double> dist(0.0, cumulative_prob);
double random_sample = dist(*rng);
double current_cumulative = 0.0;
for (int i = 0; i < final_sample_size; ++i) {
  current_cumulative += (*probabilities)[b][s * k + index_of_topk[i]];
  if (random_sample <= current_cumulative) {
    sampled_ids[b][s] = flat_topk_token_ids[topk_offset + index_of_topk[i]];
    sampled_scores[b][s] = (*probabilities)[b][s * k + index_of_topk[i]];
    break;
  }
}
```
**Top-k 按 k 大小分三档优化(贪心 / 最小堆 / nth_element)** — 待核验：`runtime/components/sampling_cpu_util.cc:48-102`
```cpp
if (k == 1) {  // 贪心: 直接 max_element
  auto max_iterator = std::max_element(logits.begin()+..., logits.begin()+...);
  output_indices[b][s] = std::distance(..., max_iterator);
} else if (k <= 1024) {  // 大小为 k 的最小堆, O(N log k)
  absl::c_make_heap(min_heap, min_heap_comp);
  for (int i = actual_k; i < vocab_size; ++i) {
    if (val > min_heap.front().logit) { /* 替换堆顶 */ }
  }
} else {  // 大 k: nth_element 部分排序, 平均 O(N)
  std::nth_element(indices.begin(), indices.begin()+k, indices.end(), desc_prob_comp);
}
```
**SentencePiece 流式解码: byte token 分块与 BPE 不完整检测** — 待核验：`runtime/components/sentencepiece_tokenizer.cc:94-118`
```cpp
if (processor_->IsByte(token_id)) {
  std::string decoded = processor_->DecodeIds({token_id});
  if (Tokenizer::HasBpeSuffix(decoded)) {
    chunk_byte_token_ids.push_back(token_id);  // 部分 BPE, 等更多 token
  } else {
    absl::StrAppend(&text, decoded);
  }
} else {
  if (!chunk_byte_token_ids.empty()) {
    absl::StrAppend(&text, processor_->DecodeIds(chunk_byte_token_ids));
    chunk_byte_token_ids.clear();
  }
  // 用 IdToPiece 以保留前导空格, 否则流式拼接会丢空格
  absl::StrAppend(&text, processor_->IdToPiece(token_id));
}
```
**重复惩罚: 乘法(HF) + 加法 presence/frequency(OpenAI)** — 待核验：`runtime/components/logits_processor/repetition_penalty_processor.cc:170-182`
```cpp
// 1. 乘法重复惩罚
if (config_.repetition_penalty() > 1.0f) {
  if (val > 0.0f) val /= config_.repetition_penalty();
  else            val *= config_.repetition_penalty();
}
// 2 & 3. 加法 presence + frequency 惩罚
val -= (config_.presence_penalty() + count * config_.frequency_penalty());
logits[batch_offset + vocab_idx] = static_cast<T>(val);
```
**约束解码: 用 bitmap 把非法 token 的 logit 置为 -inf** — 待核验：`runtime/components/logits_processor/constrained_decoding/constrained_decoder.cc:67-78`
```cpp
for (int b = 0; b < batch_size; ++b) {
  auto& constraint_state = constraint_states_[b];
  ASSIGN_OR_RETURN(auto bitmap, constraint_->ComputeBitmap(*constraint_state));
  for (int i = 0; i < vocab_size; ++i) {
    if (!bitmap->Get(i)) {
      logits.data()[b * vocab_size + i] = std::numeric_limits<float>::lowest();
    }
  }
}
```

## 入手顺序
- 先读 tokenizer.h 与 sampler.h 两个抽象接口, 建立‘文本进 / token 出 → logits 进 / token 出’的整体心智模型。
- 读 sentencepiece_tokenizer.cc 的 TokenIdsToText, 理解 byte token 分块与 BPE 不完整序列(HasBpeSuffix/DataLossError)如何支撑流式解码。
- 读 sampling_cpu_util.cc, 按 TopKTokenIds → Softmax → TopKTopPSampling 的顺序看完整采样算法, 这是本模块的算法核心。
- 再读 top_p_cpu_sampler.cc, 看 Sampler 接口如何包装上述算法(张量校验、FP16 转换、score 取 log)。
- 读 sampler_factory.cc 的尾部 CreateSampler/CreateCpuSampler/CreateGpuSampler, 理解 CPU/GPU 后端分派与 GPU 不可用时回退 CPU 的逻辑。
- 读 logits_processor/logits_processor.h 接口, 再看 repetition_penalty_processor.cc(惩罚)与 constrained_decoding/constrained_decoder.cc(约束)两个实现如何就地改写 logits。
- 最后读 stop_token_detector.cc 与 prompt_template.cc, 补齐输出侧停止判断与输入侧模板渲染。
