# 模块素材：核心概念与性能优化 (Core Concepts & Optimizations)  `concepts`

> 来源：litert-lm-guide/data.json（多智能体源码分析）。**这是素材，不是正文**。
> 引用进正文前，须按 CLAUDE.md 第二节逐条 Read 源码核验、补 `@ v0.13.1`。

**一句话**：这是一个横切主题，系统梳理 LiteRT-LM 端侧 LLM 推理涉及的关键概念与工程优化：Prefill/Decode 两阶段、KV Cache、Speculative Decoding (MTP)、量化、CPU/GPU/NPU 后端、采样策略、约束解码、Tool Use、多模态、LoRA、以及线程/CPU 亲和性等内存与并发优化。它不是某个独立目录，而是贯穿 executor/components/schema/docs 多个模块的"知识地图"。

**在架构中的位置**：在整体架构中，这些概念跨越三层：上游是 Engine/Session/Conversation 高层 API（负责对话编排、Tool Use、约束解码配置），中游是 components（采样器、tokenizer、embedding lookup、logits processor、LoRA），下游是 executor（LlmExecutorBase 抽象 + LiteRT CompiledModel 后端实现，承载 Prefill/Decode/KV Cache/量化模型）。schema/capabilities 负责从 .litert_lm 文件探测模型能力（如是否带 MTP drafter）。这些概念是理解任何其它模块行为的前提，因此被单独抽出作为"概念库"。

## 关键文件
- `runtime/executor/llm_executor_base.h` — LLM Executor 的核心抽象基类，定义 Prefill/Decode/DecodeLogits/Reset/CreateNewContext 等接口，是 Prefill-Decode 两阶段推理与多后端抽象的总入口。
- `runtime/executor/kv_cache_interface.h` — KV Cache 的抽象接口，定义 Serialize/Load/SelectAndCopyFrom/BroadcastAndCopyFrom/DeepCopy，体现 KV Cache 的序列化、批次广播与拷贝语义。
- `runtime/executor/llm_executor_settings.h` — 承载 CpuConfig/GpuConfig/NpuConfig/GpuArtisanConfig 与 AdvancedSettings（含 enable_speculative_decoding、prefill_chunk_size、kv_increment_size、max_top_k 等），是后端选择与各类优化开关的集中地。
- `runtime/executor/llm_litert_compiled_model_executor.cc` — 基于 LiteRT CompiledModel 的 CPU/GPU 主执行器实现，包含 KV Cache 双缓冲 ping-pong、分块 prefill、异步 RunAsync 等关键优化的真实实现。
- `runtime/executor/llm_litert_mtp_drafter.h` — Multi-Token Prediction (MTP) drafter 的接口，实现 Speculative Decoding 的 draft+verify 两阶段；头文件清晰描述了 drafter/verifier 各自的输入输出 TensorBuffer 布局。
- `runtime/executor/llm_litert_mtp_drafter.cc` — MTP 推测解码的核心逻辑：RunDraftingLoop 草拟 G 个 token，RunVerification 用 base model 一次性验证，Draft() 比对接受并产生 bonus token。
- `runtime/components/sampler.h` — 采样器抽象接口 Sampler，定义 SampleToIdAndScoreBuffer / CanHandleInput / SetInputTensorsAndInferenceFunc，支持采样器自填输入张量以与后端异步推理重叠。
- `runtime/components/top_p_cpu_sampler.cc` — CPU 上 Top-K/Top-P/temperature 采样的具体实现，依赖 sampling_cpu_util 的 TopKTokenIds/Softmax/TopKTopPSampling。
- `runtime/components/sampling_cpu_util.h` — 采样数学原语：TopKTokenIds、Softmax（含 temperature 缩放）、TopKTopPSampling，是 greedy/top-k/top-p 策略的底层实现。
- `runtime/proto/sampler_params.proto` — SamplerParameters proto，定义 TOP_K/TOP_P/GREEDY 类型与 k/p/temperature/seed/backend 字段，是采样策略的配置载体。
- `runtime/components/logits_processor/logits_processor.h` — LogitsProcessor 抽象：在采样前就地修改 logits（约束解码 mask、重复惩罚），采样后 UpdateState 推进状态机。
- `runtime/components/logits_processor/constrained_decoding/constraint.h` — Constraint 接口（Start/IsEnded/ComputeNext/ComputeBitmap），约束解码的状态机抽象，LLGuidance 与自定义约束都实现它。
- `docs/api/cpp/constrained-decoding.md` — 约束解码官方文档：Tool Calling 约束 vs 自定义约束（LLGuidance 的 Regex/JSON Schema/Lark，或外部 Constraint）。
- `docs/api/cpp/tool-use.md` — Tool Use / Function Calling 的端到端流程文档：tool 声明、模型输出 tool call、解析为 JSON、应用执行后回填。
- `runtime/components/lora.h` — LoRA 适配器：从 LoraData 加载低秩权重并填充为 LiteRT TensorBuffer，支持运行时切换微调适配。
- `runtime/engine/cpu_affinity_utils.h` — CPU 亲和性工具：检测 Pixel Tensor SoC 并把推理线程绑定到大/中核，是端侧延迟优化手段。
- `runtime/framework/threadpool.h` — 通用 ThreadPool（Schedule/WaitUntilDone），支撑多线程任务调度与并发执行。
- `schema/capabilities/speculative_decoding.cc` — 从 .litert_lm 文件 header 探测是否含 tf_lite_mtp_drafter 子模型，决定能否启用推测解码。
- `runtime/components/embedding_lookup/embedding_lookup_manager.h` — 多模态 embedding 管理器：把 vision/audio 特殊 token 替换为对应模态的 embedding，在 prefill 前注入。

## 核心抽象
- **LlmExecutorBase** (class) 〔`runtime/executor/llm_executor_base.h`〕：所有 LLM 执行器的抽象基类，是 Prefill-Decode 两阶段与多后端抽象的核心。关键方法：Prefill(ExecutorInputs, ExecutorPrefillParams) 处理输入前缀并填充 KV Cache；Decode()/Decode(ExecutorDecodeParams) 自回归生成下一 token（返回每候选一个 token id）；DecodeLogits() 仅输出 logits 不采样；Reset() 清空 KV Cache 等内部状态；CreateNewContext/CloneContext/RestoreContext 管理可序列化的会话上下文。多数高级方法用 UnimplementedError 默认实现，由各后端按需重写——体现可移植、最小依赖的设计。
- **KVCacheInterface** (interface) 〔`runtime/executor/kv_cache_interface.h`〕：KV（键值）缓存的抽象接口。KV Cache 缓存了已处理 token 在每个注意力层的 Key/Value 张量，使 Decode 阶段每步只需计算 1 个新 token 而非重算整个序列。关键方法：Serialize/Load 支持会话持久化；SelectAndCopyFrom 从多批次缓存中选出某一批复制（用于多候选收敛）；BroadcastAndCopyFrom 把 batch=1 的缓存广播到 batch>N（用于一次 prefill 多候选 decode）；DeepCopy 深拷贝（昂贵，慎用）。
- **LlmLiteRtMtpDrafter** (class) 〔`runtime/executor/llm_litert_mtp_drafter.h`〕：Multi-Token Prediction 推测解码器，实现 Speculative Decoding。Draft() 一次产出多个 token：RunDraftingLoop 用轻量 MTP drafter 模型连续草拟 num_draft_steps 个 token，PrepareVerifierInputBuffers 把 [输入token + 草稿] 拼成一批，RunVerification 让 base model 用 'verify' 签名一次性并行验证（async RunAsync），再逐位比对：第一个不匹配处用 base model 的结果作 bonus token 截断。接受率统计 num_verified_tokens_/num_drafted_tokens_。核心收益：用一次 base model 前向验证多个 token，摊薄 decode 的内存带宽瓶颈。
- **Sampler** (interface) 〔`runtime/components/sampler.h`〕：采样器抽象。SampleToIdAndScoreBuffer(logits[B,S,V] -> ids[B,S] + 可选 scores) 从 logits 采样 token；UpdateConfig 注入 SamplerParameters 与随机数引擎。CanHandleInput/HandlesInput/SetInputTensorsAndInferenceFunc 是关键优化点：当采样器'能处理输入'时，它可自行用上一步输出 token 填充下一步的 input tokens/positions/mask，并在内部调用 run_inference_func 启动下一步推理——从而把采样与 GPU 异步推理重叠，隐藏延迟。
- **SamplerParameters** (struct) 〔`runtime/proto/sampler_params.proto`〕：采样策略配置 proto。Type 枚举 TOP_K / TOP_P / GREEDY（argmax）；k 控制 top-k 候选数；p 是 top-p 累积概率阈值；temperature 缩放 logits（越低越确定、越高越随机）；seed 控制可复现；Backend 枚举（UNSPECIFIED/CPU/GPU）选择采样在哪执行。注意 GPU 路径默认组合 top-k 与 top-p。
- **LogitsProcessor** (interface) 〔`runtime/components/logits_processor/logits_processor.h`〕：在采样前就地修改 logits 的处理器，按序运行：ProcessLogits 修改 logits[B,1,V]（约束解码把禁止 token 置为 -inf、重复惩罚降低已出现 token 概率），采样后 UpdateState(token) 推进内部状态机。GetConstraintDecoder 暴露底层约束解码器。RepetitionPenaltyProcessor 与约束解码都通过它接入解码管线。
- **Constraint** (interface) 〔`runtime/components/logits_processor/constrained_decoding/constraint.h`〕：约束解码的状态机抽象。Start 返回起始状态；ComputeBitmap(state) 给出当前允许的 token 位图（Bitmap）；ComputeNext(state, token) 消费一个 token 并推进状态；IsEnded 判断结束。LLGuidance（Regex/JSON Schema/Lark）和用户自定义 C++ 约束都实现此接口，保证输出严格符合语法/schema（用于 Function Calling、结构化抽取）。
- **LlmExecutorSettings / AdvancedSettings** (class) 〔`runtime/executor/llm_executor_settings.h`〕：执行器配置中枢。max_num_tokens_ 等于 KV Cache 容量；backend_config_ 是 variant<GpuArtisanConfig,GpuConfig,CpuConfig,NpuConfig> 持有各后端专属参数（如 CpuConfig.kv_increment_size 动态扩容步长、prefill_chunk_size 分块大小、number_of_threads；NpuConfig.use_hw_cache_update_for_npu 等硬件加速开关）。AdvancedSettings 集中了 enable_speculative_decoding、clear_kv_cache_before_prefill、sampler_handles_input、share_constant_tensors 等性能/正确性开关。
- **Backend (enum)** (enum) 〔`runtime/executor/executor_settings_base.h`〕：后端枚举：CPU/GPU（LiteRT 路径）、CPU_ARTISAN/GPU_ARTISAN（手写优化路径）、GOOGLE_TENSOR_ARTISAN、NPU。配合 ActivationDataType（FLOAT32/FLOAT16/INT16/INT8）和 FakeWeightsMode（INT8 / INT4 attn8+ffn4+emb4 量化）描述模型精度与加速器选择。
- **EmbeddingLookupManager** (class) 〔`runtime/components/embedding_lookup/embedding_lookup_manager.h`〕：多模态 embedding 注入器。UpdateMultiModalEmbeddings 在 prefill 前把文本中的 vision/audio 特殊 token 占位替换为对应模态编码器输出的 embedding；LookupTextEmbedding 为 MTP/外部 embedding 流程提供单 token 查表。是 vision/audio 多模态接入 LLM 主干的桥梁。
- **LoRA** (class) 〔`runtime/components/lora.h`〕：LoRA 低秩适配器。Create 从 LoraData 加载适配权重并为 CompiledModel 创建后端 TensorBuffer；GetLoRABuffers 返回所有 LoRA 张量。底层是 shared_ptr 共享数据，支持运行时挂载不同微调适配而无需重载整个基座模型；GpuArtisanConfig.supported_lora_ranks 声明支持的秩。

## 数据流
1. 文件能力探测：从 .litert_lm 读取 header，schema/capabilities 判断是否含 tf_lite_mtp_drafter（speculative_decoding）、是否多模态等，决定创建哪种 executor 与是否启用推测解码。
2. Prefill（预填充阶段）：ExecutorInputs（文本 token + 可选 vision/audio 数据）进入 LlmExecutorBase::Prefill；EmbeddingLookupManager 先把多模态特殊 token 替换为模态 embedding；executor 把整段前缀一次性（或按 prefill_chunk_size 分块）喂入 CompiledModel，计算并写满 KV Cache。此阶段是计算密集型（compute-bound），可异步 RunAsync。
3. Decode（解码阶段）：从 KV Cache 出发，每步只前向 1 个新 token；输出 logits[B,1,V]。此阶段是内存带宽密集型（memory-bound），是推测解码优化的目标。
4. Logits 后处理：LogitsProcessor 链就地修改 logits——RepetitionPenalty 降低重复、Constraint(LLGuidance) 把不合法 token mask 成 -inf。
5. 采样：Sampler 按 SamplerParameters（GREEDY/TOP_K/TOP_P + temperature）从修改后的 logits 选出 token id；CPU 路径走 sampling_cpu_util 的 Softmax+TopKTopP，GPU 路径在设备上采样。若采样器 HandlesInput，则顺带填好下一步输入并触发异步推理。
6. KV Cache 推进：新 token 的 K/V 追加进缓存；实现上用 input/output 两套缓冲做 ping-pong（std::swap）避免拷贝。
7. 推测解码分支（若启用 MTP）：drafter 一次草拟 G 个 token，base model 用 verify 签名一次性验证整批，逐位比对接受前缀并加 1 个 bonus token，单次 base 前向即可前进多个 token。
8. Tool Use / 终止检测：StopTokenDetector 检测停止符；若检测到 tool call 语法，tool_use 解析器将其解析为 JSON 交给应用执行，结果回填后从 Prefill 重新进入循环。

## 概念
- **Prefill vs Decode 两阶段推理**：自回归 LLM 推理分两阶段。Prefill（预填充/prefix）：把整段输入提示一次性并行喂入模型，计算所有位置的注意力并填满 KV Cache，是计算密集型，能充分利用并行硬件。Decode（解码）：逐 token 自回归生成，每步只算 1 个新 token，依赖前面所有 token 的 KV，是内存带宽密集型（每步都要读全部权重却只算 1 个 token，硬件利用率低）。LiteRT-LM 的 LlmExecutorBase 用 Prefill() 和 Decode() 两组 API 明确区分二者；首 token 延迟主要来自 prefill，吐字速度（tokens/s）主要受 decode 的内存带宽限制——这正是推测解码、量化要攻克的瓶颈。
- **KV Cache（键值缓存）**：Transformer 注意力中，每个已生成 token 在每一层都会产生 Key 和 Value 向量。若不缓存，生成第 n 个 token 要重算前 n-1 个的 K/V，复杂度 O(n²)。KV Cache 把这些 K/V 存下来，decode 每步只需为新 token 算 K/V 并追加，复杂度降为 O(n)。代价是显存/内存随上下文长度线性增长——max_num_tokens 就等于 KV Cache 容量。LiteRT-LM 的 KVCacheInterface 还支持序列化（会话保存/恢复）、批次广播（一次 prefill 供多候选 decode）和动态扩容（CpuConfig.kv_increment_size）。
- **Speculative Decoding / MTP（推测解码 / 多 token 预测）**：针对 decode 内存带宽瓶颈的核心优化。思路：用一个又小又快的 drafter 连续草拟多个候选 token，再用大的 base model 在一次前向里并行验证这一整批草稿（验证 N 个 token 的成本≈生成 1 个，因为都是内存带宽受限）。逐位比对：连续匹配的前缀全部接受，第一个不匹配处用 base model 的正确结果作为 bonus token。最好情况一次 base 前向前进 N+1 个 token，吐字速度成倍提升，且输出分布与 base model 完全一致（无质量损失）。LiteRT-LM 的 MTP（Multi-Token Prediction）drafter 与 base 共享 KV Cache，drafter 是带 'tf_lite_mtp_drafter' 标记的子模型。
- **量化（int4 / int8 Quantization）**：把模型权重从 FP32/FP16 压缩成低比特整数（INT8/INT4），主要为减小模型体积和降低内存带宽——而带宽正是 decode 的瓶颈，所以量化既省内存又提速。LiteRT-LM 中 ActivationDataType 控制激活精度（FP32/FP16/INT16/INT8），FakeWeightsMode 揭示了典型量化方案：全 INT8，或 attention 用 INT8、feed-forward 与 embedding 用 INT4（FAKE_WEIGHTS_ATTN_8_FFN_4_EMB_4），因为 FFN/embedding 占参数大头且对精度更鲁棒。量化后的模型由 LiteRT CompiledModel 加载执行，部分 GPU 还支持 src quantized fc/conv 算子进一步加速。
- **LiteRT CompiledModel**：LiteRT（前身 TensorFlow Lite）的已编译模型对象，是 LiteRT-LM 所有后端执行器的底层引擎。它封装了针对目标硬件（CPU 走 XNNPACK、GPU 走 ML Drift/OpenCL/WebGPU/Metal、NPU 走 dispatch 库）编译好的算子图，提供 Run（同步）与 RunAsync（异步）执行，以及输入/输出 TensorBuffer 的创建。executor 把 token、position、mask、KV Cache 全部组织成 TensorBuffer 字典传给它执行。编译产物可缓存（weight cache / program cache）以加速二次启动。
- **CPU / GPU / NPU 后端选择**：端侧三类加速器各有取舍。CPU（XNNPACK）：兼容性最好、随处可用，靠多线程（number_of_threads）和 SIMD/NEON，适合小模型或无 GPU 设备。GPU（ML Drift/OpenCL/Vulkan/WebGPU/Metal）：并行度高、吞吐大，但有初始化开销（shader 编译、权重上传）和 UI 抢占问题，需谨慎调度（低优先级、kernel 批量 flush）。NPU：专用神经网络加速器，能效比最高，但需硬件专属逻辑（NpuConfig 里的 hw masking / hw cache update / hw PLE / NEON greedy sampling 等开关）。Backend 枚举还区分 LiteRT 通用路径与手写优化路径（_ARTISAN）。
- **采样策略（temperature / top-k / top-p / greedy）**：决定如何从 logits 概率分布里挑下一个 token。Greedy（argmax）：永远选概率最高的，确定性强但易重复。Temperature：先用 T 缩放 logits 再 softmax，T<1 更尖锐(确定)、T>1 更平坦(随机)。Top-K：只在概率最高的 K 个候选里采样，截断长尾。Top-P（nucleus）：按概率从高到低累加，取累积达到 P 的最小集合内采样，自适应候选数量。LiteRT-LM 用 SamplerParameters 配置，CPU 实现见 sampling_cpu_util 的 TopKTopPSampling（先 top-k 再 softmax 再 top-p 采样）。
- **约束解码（Constrained Decoding）**：强制模型输出符合特定结构（正则、JSON Schema、上下文无关文法）。原理：每个 decode 步根据当前文法状态计算一个'允许 token 的位图'，把不合法 token 的 logit 置为 -inf，使采样只能在合法集合里选；选出 token 后推进文法状态机。LiteRT-LM 支持两种：(1) Tool Calling 内置约束，自动保证 function call 语法正确；(2) 自定义约束，用 LLGuidance（支持 Regex/JSON Schema/Lark）或实现 Constraint 接口的 C++ 状态机。通过 LogitsProcessor 接入解码管线，是可靠 Function Calling 与结构化抽取的关键。
- **Tool Use / Function Calling（工具调用）**：让 LLM 能调用外部函数。流程：应用声明工具（name/parameters/description 的 JSON Schema，放进 Preface.tools）→ 用户提问 → 模型输出表示工具调用的特殊字符串 → LiteRT-LM 检测并解析成 JSON（tool_use 目录里的 fc/json/antlr 解析器）→ 应用执行真实函数并回填结果 → 模型基于结果继续回答或再次调用。约束解码常与之配合，保证模型输出的 tool call 语法 100% 合法可解析。
- **多模态（Vision / Audio Embedding）**：让纯文本 LLM 理解图像/音频。做法：图像/音频先经各自的编码器（preprocessor + vision/audio executor）转成与文本 token 同维度的 embedding 向量；在文本序列里用特殊占位 token（kVisionSpecialToken 等）标记插入位置；prefill 前由 EmbeddingLookupManager 把这些占位 token 的 embedding 替换成真实模态 embedding，从而图文/音文混合输入到同一 LLM 主干。还支持 per-layer embeddings（部分模型每层都注入模态信息）。
- **LoRA 微调适配（Low-Rank Adaptation）**：一种高效微调技术：冻结基座模型权重，只训练每个权重矩阵旁挂的两个低秩小矩阵（秩 r 远小于原维度），微调成本和存储都极小。推理时把低秩增量加到基座输出上。端侧价值：一个基座模型 + 多个小 LoRA 适配器即可服务多任务/多风格，切换适配器无需重载基座。LiteRT-LM 的 LoRA 类负责加载低秩权重并填充为 TensorBuffer，GpuArtisanConfig.supported_lora_ranks 声明引擎支持的秩集合。
- **内存 / 线程优化（ThreadPool & CPU Affinity）**：端侧 CPU 是异构的（big.LITTLE：大核高性能、小核省电）。CPU Affinity（亲和性）把推理线程绑定到大/中核，避免被调度到慢的小核——cpu_affinity_utils 为 Pixel Tensor G3~G6 硬编码了各自的高性能核 ID。ThreadPool 提供任务级并发（prefill 分块、多模态编码可并行）。配合 CpuConfig.number_of_threads（XNNPACK 算子内并行），共同压低延迟。内存侧还有 KV Cache 双缓冲复用、常量张量共享、权重/程序缓存等手段。

## 优化
- **KV Cache 双缓冲 ping-pong**：executor 持有 input_kv_cache_buffers_ 和 output_kv_cache_buffers_ 两套缓冲，每次 prefill/decode 后用 std::swap 交换指针（llm_litert_compiled_model_executor.cc:757、966），让本步输出直接成为下步输入，避免每步深拷贝整个 KV Cache。GPU 优化模式 (gpu_optimized_single_buffer_cache_) 下甚至共用单缓冲省一半内存。
- **分块 Prefill（Chunked Prefill）**：CpuConfig.prefill_chunk_size 控制每次 prefill 处理的最大 token 数（executor.cc:1891 起循环按 chunk 切分）。小 chunk 降低峰值内存、便于及时取消超长输入；-1 表示不分块一次处理完。对动态导出模型尤其重要。
- **异步推理与采样重叠**：prefill/decode 用 compiled_model_->RunAsync 异步执行（executor.cc:749、962）；ExecutorPrefillParams.wait_for_completion=false 时不阻塞等待。配合 Sampler.SetInputTensorsAndInferenceFunc——采样器可在 GPU 异步算下一步时，先返回上一步采样结果，把采样与推理重叠隐藏延迟（AdvancedSettings.sampler_handles_input）。
- **Speculative Decoding 摊薄带宽**：MTP drafter 用小模型草拟 G 个 token，base model 用 'verify' 签名一次 RunAsync 并行验证整批（mtp_drafter.cc:439），逐位接受 + bonus token（Draft() 473-484 行）。decode 是内存带宽受限，验证 N 个 token 成本≈生成 1 个，故一次 base 前向最多前进 G+1 个 token，吐字成倍加速且无质量损失；drafter 与 base 共享 KV Cache。
- **权重 / 程序缓存**：ExecutorSettingsBase 支持 XNNPACK weight cache、ML Drift program/weight cache（kXnnpackCacheSuffix / kMlDriftCacheSuffix）。首次初始化时把重排后的权重和编译好的 shader/program 落盘，二次启动直接加载，大幅缩短冷启动时间。AdvancedSettings 还可只缓存编译 shader (cache_compiled_shaders_only)。
- **量化降低体积与带宽**：INT8/INT4 量化（FakeWeightsMode：attn8+ffn4+emb4）把权重压到 1/4~1/8，既省内存又因 decode 是带宽受限而直接提速。GPU 上 allow_src_quantized_fc_conv_ops 允许量化 fc/conv 算子进一步加速（以少量质量为代价）。
- **CPU 亲和性绑大核**：cpu_affinity_utils 检测 Pixel Tensor SoC，把推理线程通过 sched_setaffinity 绑到硬编码的高性能核（如 G5/G6 用 core 2-7），避免被调度到节能小核，稳定降低 decode 延迟。
- **GPU UI 平滑优化**：针对移动端 GPU 与 UI 抢占：gpu_context_low_priority 降 GPU 上下文优先级、hint_kernel_batch_size 周期性 flush 命令队列（OSS 通用模型默认 4）、share_constant_tensors 共享常量张量、optimize_shader_compilation 等，权衡推理速度与界面流畅度。
- **采样器内存复用**：TopPSampler 把 logits_data_ 作为成员复用，避免每步重新分配大 vector；MTP drafter 预分配 active_*_buffers_ 与临时 id_tensor，避免每步重建 map/张量（mtp_drafter.h:188-199）。

## 关键代码片段（待核验 @ v0.13.1）
**Prefill/Decode 两阶段的核心抽象 API** — 待核验：`runtime/executor/llm_executor_base.h:45-61`
```cpp
// Basic API to trigger the "prefill" or "prefix" process.
// Input is token ids with shape `[batch, sequence_length]`
virtual absl::Status Prefill(const ExecutorInputs& inputs) = 0;
...
// Basic API to trigger the "decode" process. On success, will return a vector
// of token ids of generated output tokens, one per candidate.
virtual absl::StatusOr<std::vector<std::vector<int>>> Decode() = 0;
```
**KV Cache 的批次广播与深拷贝接口** — 待核验：`runtime/executor/kv_cache_interface.h:53-62`
```cpp
// Broadcasts the source KV with batch size 1 to this KV cache with batch size
// > 1. (other[0,:,...] -> this[0,:,...], this[1,:,...], this[2,:,...].)
virtual absl::Status BroadcastAndCopyFrom(KVCacheInterface& other) = 0;

// Deep copies the KV cache. This is an expensive operation. Use sparingly.
virtual absl::StatusOr<std::unique_ptr<KVCacheInterface>> DeepCopy() const = 0;
```
**Speculative Decoding 的接受逻辑：逐位比对草稿 + bonus token** — 待核验：`runtime/executor/llm_litert_mtp_drafter.cc:471-493`
```cpp
int num_correct_tokens = 0;
int bonus_token = -1;
for (int i = 0; i < num_draft_steps_; ++i) {
  last_verified_token_id_idx_ = i;
  if (verifier_id_vector[i] != drafted_tokens[i]) {
    bonus_token = verifier_id_vector[i];  // 第一个不匹配处用 base 的结果
    break;
  }
  num_correct_tokens++;
}
if (bonus_token == -1) {  // 全部草稿都被接受
  last_verified_token_id_idx_ = num_draft_steps_;
  bonus_token = verifier_id_vector[num_draft_steps_];
}
drafted_tokens.resize(num_correct_tokens);
drafted_tokens.push_back(bonus_token);
```
**KV Cache 双缓冲 ping-pong：用 swap 避免拷贝** — 待核验：`runtime/executor/llm_litert_compiled_model_executor.cc:748-758`
```cpp
if (async) {
  LITERT_RETURN_IF_ERROR(compiled_model_->RunAsync(
      prefill_signature, input_buffers, output_buffers, async));
} else {
  LITERT_RETURN_IF_ERROR(
      compiled_model_->Run(prefill_signature, input_buffers, output_buffers));
}
if (!gpu_optimized_single_buffer_cache_) {
  std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);
}
```
**采样策略与量化精度的配置枚举** — 待核验：`runtime/proto/sampler_params.proto:26-43`
```protobuf
enum Type {
  TYPE_UNSPECIFIED = 0;
  TOP_K = 1;   // 在 top-k 候选里按概率采样
  TOP_P = 2;   // top-k 之后再取累积概率 >= p 的核
  GREEDY = 3;  // argmax，取最大 logit
}
Type type = 1;
int32 k = 2;
float p = 3;
float temperature = 4;  // 缩放 logits 后再 softmax
```

## 入手顺序
- 先读 runtime/executor/llm_executor_base.h，理解 Prefill/Decode 两阶段 API 与执行器抽象——这是所有概念的骨架。
- 再读 runtime/executor/kv_cache_interface.h 与 runtime/executor/llm_executor_settings.h，搞懂 KV Cache 语义和各后端 Config / AdvancedSettings 里的优化开关。
- 读 runtime/components/sampler.h + runtime/proto/sampler_params.proto + runtime/components/sampling_cpu_util.h，掌握 greedy/top-k/top-p/temperature 采样策略及其 CPU 实现。
- 读 runtime/components/logits_processor/logits_processor.h 与 constrained_decoding/constraint.h，配合 docs/api/cpp/constrained-decoding.md，理解约束解码如何 mask logits。
- 读 runtime/executor/llm_litert_mtp_drafter.h 再看 .cc 的 Draft()/RunDraftingLoop()/RunVerification()，完整理解 Speculative Decoding 的 draft-verify-accept 流程。
- 跳到 runtime/executor/llm_litert_compiled_model_executor.cc，搜 input_kv_cache_buffers_ / std::swap / prefill_chunk_size_ / RunAsync，看 KV Cache 双缓冲、分块 prefill、异步推理的真实落地。
- 最后浏览 docs/api/cpp/tool-use.md、embedding_lookup/embedding_lookup_manager.h、components/lora.h、engine/cpu_affinity_utils.cc、framework/threadpool.h，补齐 Tool Use、多模态、LoRA、线程/亲和性等横切优化。
