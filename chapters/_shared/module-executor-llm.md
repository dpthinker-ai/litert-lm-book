# 模块素材：LLM 执行器 (Executor)  `executor-llm`

> 来源：litert-lm-guide/data.json（多智能体源码分析）。**这是素材，不是正文**。
> 引用进正文前，须按 CLAUDE.md 第二节逐条 Read 源码核验、补 `@ v0.13.1`。

**一句话**：真正“跑模型”的一层。Executor 把不同格式/不同硬件后端的 LLM 包装成统一的 Prefill / Decode 接口，屏蔽 CPU/GPU/NPU 差异。核心实现 LlmLiteRtCompiledModelExecutor 用 LiteRT 的 CompiledModel 执行 Transformer，管理 KV cache 双缓冲，并集成片上采样与 MTP 推测解码。它是整个运行时性能的发动机。

**在架构中的位置**：位于 Core Pipeline 之下、LiteRT/硬件之上的“推理执行层”。上游：Tasks::Prefill / Tasks::Decode 调用本层的 Prefill()/Decode()/DecodeLogits()。下游：通过 LiteRT CompiledModel 调度到具体后端（XNNPack CPU、OpenCL/Metal/WebGPU GPU、QNN NPU），并向 components 层借用 Sampler、EmbeddingLookupManager。它对上只暴露 LlmExecutorBase 抽象，对下用工厂按 Backend 选择具体实现。

## 关键文件
- `runtime/executor/llm_executor_base.h` — Executor 抽象基类，定义 Prefill/Decode/DecodeLogits/上下文管理等全部虚接口（多数带 UnimplementedError 默认实现）。
- `runtime/executor/llm_executor.h` — 一行别名 using LlmExecutor = LlmExecutorBase，对外的稳定类型名。
- `runtime/executor/llm_litert_compiled_model_executor.h` — 核心实现：基类 Base + 静态形状 Static + 动态形状 Dynamic 三个类，封装 CompiledModel、KV cache 双缓冲、采样器、MTP drafter。
- `runtime/executor/llm_litert_compiled_model_executor.cc` — Prefill/Decode 的实际算法：绑定输入输出张量、跑 signature、KV cache 滚动、片上采样。
- `runtime/executor/llm_litert_compiled_model_executor_factory.cc` — 工厂 CreateLlmLiteRtCompiledModelExecutor：按 Backend(CPU/GPU/NPU) 分派创建对应 executor。
- `runtime/executor/kv_cache_interface.h` — KVCacheInterface：KV 缓存的序列化/加载/选批拷贝/广播/深拷贝接口，支撑会话克隆与 checkpoint。
- `runtime/executor/llm_litert_mtp_drafter.h` — MTP（Multi-Token Prediction）推测解码 drafter：用小 drafter 模型一次草拟多 token，再用 base 模型一次性 verify。
- `runtime/executor/llm_litert_npu_compiled_model_executor.h` — NPU 后端 executor（QNN 等），含 SpeculativeDecodingType/KVCacheUpdateMethod 等 NPU 特有枚举与 embedder 上下文。
- `runtime/executor/llm_executor_io_types.h` — I/O 类型：ExecutorInputs(文本/视觉/音频)、ExecutorPrefillParams、ExecutorDecodeParams、LlmContext/RuntimeState/RuntimeConfig。
- `runtime/executor/llm_executor_settings.h` — LlmExecutorSettings：max_num_tokens、后端配置等；GetBackendConfig<T>() 取具体后端参数。
- `runtime/executor/executor_settings_base.h` — Backend 枚举(CPU/GPU/NPU/ARTISAN...) 与 ActivationDataType(FP32/FP16/INT16/INT8)。
- `runtime/executor/fake_llm_executor.h` — 测试用假 executor，预置脚本化的 token 序列，让上层算法可脱离真实模型做单测。

## 核心抽象
- **LlmExecutorBase (= LlmExecutor)** (class) 〔`runtime/executor/llm_executor_base.h`〕：执行器抽象。输入侧 Prefill(inputs[, params])；输出侧 Decode()（内部采样，直接出 token）与 DecodeLogits(inputs)（只出 logits，交给上层外部采样）。还有 GetCurrentStep/SetCurrentStep、GetVocabSize、FillVisionEmbeddings（多模态 embedding 注入）、Reset、以及 CreateNewContext/CloneContext/RestoreContext 一组上下文管理接口。绝大多数方法是带 UnimplementedError 默认实现的虚函数，后端按需重写。
- **LlmLiteRtCompiledModelExecutorBase** (class) 〔`runtime/executor/llm_litert_compiled_model_executor.h`〕：GPU/CPU 共享的核心实现基类（本身不可实例化，Create 未实现）。持有 CompiledModel、decode 输入/输出张量、两套 KV cache 缓冲(kv_cache_buffers_1_/2_ 指针互换实现读写双缓冲)、Sampler、MTP drafter、EmbeddingLookupManager。关键内部方法：PrefillInternal/BindTensorsAndRunPrefill、DecodeInternal/BindTensorsAndRunDecode、SampleLogits、PrepareFirstDecode（把 KV cache 广播到 output_batch_size）、PrepareFirstPrefillAfterDecode。
- **LlmLiteRtCompiledModelExecutorStatic** (class) 〔`runtime/executor/llm_litert_compiled_model_executor.h`〕：静态形状变体：模型有一组固定长度的 prefill signature（SortedPrefillSignatureMap），按输入长度挑最合适的 signature。常见于 GPU/移动端预编译模型。
- **LlmLiteRtCompiledModelExecutorDynamic** (class) 〔`runtime/executor/llm_litert_compiled_model_executor.h`〕：动态形状变体：输入序列长度与 KV cache 大小可变，用 prefill_chunk_size 分块 prefill，按 kv_increament_size 增量扩展 KV cache（key/value 各有动态维度索引）。
- **LlmLiteRtNpuCompiledModelExecutor** (class) 〔`runtime/executor/llm_litert_npu_compiled_model_executor.h`〕：NPU 后端实现（如 Qualcomm QNN）。含 NPU 特有的 SpeculativeDecodingType、KVCacheUpdateMethod、MaskUpdateMethod 枚举，以及把 token→embedding 的 embedder 子模型上下文（CreateEmbedderContextWithBufferSharing）。
- **LlmLiteRtMtpDrafter** (class) 〔`runtime/executor/llm_litert_mtp_drafter.h`〕：推测解码核心。Draft(position, token_id, activations, in_kv, out_kv)：先用 drafter 小模型 RunDraftingLoop 草拟 num_draft_steps 个候选 token，再 PrepareVerifierInputBuffers 把候选+1 喂给 base 模型的 verify signature 一次性 RunVerification，接受前缀匹配的部分。统计 num_drafted_tokens_/num_verified_tokens_ 衡量接受率。drafter 用贪心采样(batch=1)，verifier 用 batch=G+1 贪心。
- **KVCacheInterface** (class) 〔`runtime/executor/kv_cache_interface.h`〕：KV 缓存抽象：Serialize/Load（持久化）、SelectAndCopyFrom（从多 batch 选一条，用于 beam/candidate 收敛）、BroadcastAndCopyFrom（batch=1 广播到多 batch）、DeepCopy（昂贵，会话克隆用）。
- **LlmContext / RuntimeState / RuntimeConfig** (class) 〔`runtime/executor/llm_executor_io_types.h`〕：执行器的可迁移状态容器。LlmContext 打包 ProcessedContext(含 KV cache 与已处理 token)、RuntimeConfig(output_heads 等)、RuntimeState(current_step 等)。CloneContext/RestoreContext 即是搬运它，支撑 Session 的 Clone 与 checkpoint。
- **ExecutorInputs / ExecutorPrefillParams / ExecutorDecodeParams** (struct) 〔`runtime/executor/llm_executor_io_types.h`〕：ExecutorInputs 是 文本(token id TensorBuffer)+视觉embedding+音频embedding 的容器。PrefillParams 带 current_step、wait_for_completion(同步/异步)、cancel(外部原子取消标志)、max_prefill_sequence_length(限长以便及时取消)。DecodeParams 携带 LogitsProcessor 列表与约束解码器。
- **Backend / ActivationDataType** (enum) 〔`runtime/executor/executor_settings_base.h`〕：Backend：UNSPECIFIED / CPU_ARTISAN / GPU_ARTISAN / CPU / GPU / GOOGLE_TENSOR_ARTISAN / NPU（ARTISAN=手写算子路径，非 ARTISAN=LiteRT 编译路径）。ActivationDataType：FLOAT32/FLOAT16/INT16/INT8，决定激活精度与 GPU 采样张量类型。
- **CreateLlmLiteRtCompiledModelExecutor** (function) 〔`runtime/executor/llm_litert_compiled_model_executor_factory.cc`〕：工厂入口：读 executor_settings.GetBackend()，CPU/GPU 走 CompiledModel 路径(Static/Dynamic)，NPU 走 NPU executor，其它后端报 InvalidArgument。

## 数据流
1. 工厂 CreateLlmLiteRtCompiledModelExecutor 按 Backend 创建具体 executor：CPU/GPU → CompiledModel Static/Dynamic；NPU → NPU executor。创建时加载 .litertlm 权重、建 CompiledModel、分配 KV cache 双缓冲与 decode 张量、初始化 Sampler（GPU 片上采样）和可选 MTP drafter。
2. Prefill：上层把整段提示词 token（必要时拼上视觉/音频 embedding）封进 ExecutorInputs，调用 Prefill(inputs, params)。PrefillInternal 选合适长度的 prefill signature（Static）或按 chunk 分块（Dynamic），BindTensorsAndRunPrefill 跑模型，一次性把整段写进 KV cache。params.wait_for_completion 控制同步/异步，params.cancel 支持中途取消。
3. Decode（内部采样路径）：Decode(decode_params) → GetTokenToDecode 取上一个 token → DecodeInternal 跑 decode signature 得 logits → SampleLogits 用片上/CPU 采样器出新 token id → 把新 token 的 K/V 追加进 KV cache（双缓冲指针互换）→ current_step+1 → 返回 token。
4. Decode（外部采样路径）：DecodeLogits(inputs) 只返回 logits TensorBuffer，由 Core Pipeline 的 LogitsProcessor（重复惩罚/约束解码）+ Sampler 在外部完成采样，再回灌。
5. MTP 推测解码：drafter 一次草拟 G 个 token，base 模型用 verify signature 一次前向验证 G+1 个，接受最长匹配前缀；一次大模型前向产出多 token，显著提速。
6. 上下文迁移：CloneContext/RestoreContext 搬运 LlmContext（KV cache + step）；KVCacheInterface 的 DeepCopy/SelectAndCopyFrom/BroadcastAndCopyFrom 支撑会话克隆、候选收敛与 batch 广播。

## 概念
- **Prefill / Decode 两阶段**：Prefill 把整段提示词一次性并行喂入、批量填充 KV cache（计算密集、可并行）；Decode 逐 token 自回归、每步只算 1 个新 token（访存密集、串行）。两者复用同一套 KV cache，是所有自回归 LLM 的标准结构。
- **CompiledModel 与 Signature**：LiteRT 把模型编译成针对目标后端优化的 CompiledModel；模型暴露多个具名 signature（如不同长度的 prefill、decode、verify）。executor 按场景选 signature、绑定输入输出 TensorBuffer 后调用。
- **KV cache 双缓冲**：代码里 kv_cache_buffers_1_/2_ 两套缓冲用指针 input_/output_ 互换：因为部分 GPU 后端不能对同一缓冲同时读写，所以读旧缓冲、写新缓冲再交换，避免拷贝。
- **静态形状 vs 动态形状**：Static：预编译若干固定长度 signature，按输入长度选最近的（移动/GPU 常见，避免重编译）。Dynamic：序列长度与 KV cache 可增长，prefill 分块、KV 按需扩容（更灵活，桌面/服务端常见）。
- **片上采样 (on-device GPU sampling)**：采样器可直接在 GPU 上对 logits 做 top-k/top-p（gpu_sampler_max_top_k_），省去 logits 从 GPU 拷回 CPU 的开销——decode 每步都做，省下的带宽很可观。
- **Speculative Decoding / MTP**：用一个便宜的 drafter 一次猜多 token，再用大模型一次性验证；只要猜对，就用一次大模型前向换来多 token 输出。MTP 是其单模型变体（drafter 共享主干）。是 Gemma 4 提速 ~3x 的关键。
- **异步执行与取消**：Prefill 可 async 提交到工作线程；ExecutorPrefillParams.cancel 是一个外部 atomic_bool，配合 max_prefill_sequence_length（限制单次 signature 长度）让长 prefill 也能及时响应取消。
- **上下文 (LlmContext) 可迁移**：把 KV cache + step + 配置打包成 LlmContext，可 Clone/Restore/Serialize——这是多轮会话、会话分叉(Clone)、保存检查点(checkpoint)、回退(rewind)的底层机制。

## 优化
- **KV cache 双缓冲免拷贝**：用两套缓冲+指针互换规避 GPU 读写同缓冲限制，decode 每步零拷贝滚动 KV cache。
- **GPU 片上采样**：logits 不回传 CPU，直接在 GPU 上完成 top-k/top-p 采样，去掉 decode 热路径上的 device→host 拷贝。
- **Prefill 分块 + 异步 + 可取消**：动态后端按 chunk 分批 prefill；异步提交不阻塞主线程；限制单次 signature 长度保证取消的实时性。
- **MTP 推测解码**：一次大模型前向验证多 token，接受率高时成倍提升 decode 吞吐（Gemma 4 提速约 3x）。
- **权重缓存 (weight_cache_path)**：首次编译/转换的权重缓存到磁盘，二次启动直接复用，缩短冷启动。
- **量化 + FP16 激活**：权重 int4/int8 量化压体积、提带宽利用率；use_fp16_precision 用半精度激活进一步省显存与算力。
- **Embedding/PLE 子模型缓冲共享**：NPU 路径下 embedder 与主模型共享缓冲（BufferSharing），减少 token→embedding 的额外拷贝。

## 关键代码片段（待核验 @ v0.13.1）
**Executor 抽象的核心 I/O 接口** — 待核验：`runtime/executor/llm_executor_base.h:47`
```cpp
// Prefill：整段提示词一次性写入 KV cache
virtual absl::Status Prefill(const ExecutorInputs& inputs) = 0;
// Decode：内部采样，直接返回每个候选的新 token id
virtual absl::StatusOr<std::vector<std::vector<int>>> Decode() = 0;
// DecodeLogits：只出 logits，交给上层做外部采样/约束解码
virtual absl::StatusOr<::litert::TensorBuffer> DecodeLogits(
    const ExecutorInputs& inputs);
```
**KV cache 双缓冲（读旧写新再交换）** — 待核验：`runtime/executor/llm_litert_compiled_model_executor.h:327`
```cpp
// KV cache double buffers because some GPU backends can't allocate
// one buffer for both read and write at the same time.
absl::flat_hash_map<absl::string_view, TensorBuffer> kv_cache_buffers_1_;
absl::flat_hash_map<absl::string_view, TensorBuffer> kv_cache_buffers_2_;
absl::flat_hash_map<absl::string_view, TensorBuffer>* input_kv_cache_buffers_;
absl::flat_hash_map<absl::string_view, TensorBuffer>* output_kv_cache_buffers_;
```
**按 Backend 分派创建 executor** — 待核验：`runtime/executor/llm_litert_compiled_model_executor_factory.cc:168`
```cpp
Backend backend = executor_settings.GetBackend();
switch (backend) {
  case Backend::CPU:
  case Backend::GPU:
    return CreateCpuOrGpuLlmLiteRtCompiledModelExecutor(...);
  case Backend::NPU:
    return CreateNpuLlmLiteRtCompiledModelExecutor(...);
  default:
    return absl::InvalidArgumentError(...);
}
```
**MTP 推测解码：草拟 + 验证** — 待核验：`runtime/executor/llm_litert_mtp_drafter.h:71`
```cpp
// 用 drafter 小模型一次草拟多 token，再用 base 模型 verify
absl::StatusOr<std::vector<std::vector<int>>> Draft(
    int position, int token_id, std::optional<TensorBuffer> activations,
    absl::flat_hash_map<absl::string_view, TensorBuffer>& input_kv_cache_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer>& output_kv_cache_buffers);
int num_drafted_tokens_ = 0;   // 草拟数
int num_verified_tokens_ = 0;  // 被接受数（接受率=verified/drafted）
```
**Prefill 参数：同步/取消/限长** — 待核验：`runtime/executor/llm_executor_io_types.h:378`
```cpp
class ExecutorPrefillParams {
  int current_step_ = -1;
  bool wait_for_completion_ = false;       // 同步 or 异步
  const std::atomic_bool* cancel_ = nullptr; // 外部取消标志
  std::optional<int> max_prefill_sequence_length_; // 限长保证可取消
};
```

## 入手顺序
- 先读 runtime/executor/llm_executor_base.h，把握 Prefill/Decode/DecodeLogits + 上下文管理这组对外契约。
- 再读 llm_executor_io_types.h，搞清 ExecutorInputs / PrefillParams / DecodeParams / LlmContext 这几个数据载体。
- 看 llm_litert_compiled_model_executor.h 的 Base/Static/Dynamic 三个类与成员（KV cache 双缓冲、sampler、mtp_drafter）。
- 顺着 llm_litert_compiled_model_executor_factory.cc 看后端如何按 Backend 分派创建。
- 想理解性能优化，重点读 llm_litert_mtp_drafter.h（推测解码）与 kv_cache_interface.h（缓存迁移）。
- 想跑通单测，看 fake_llm_executor.h 如何用脚本化 token 替身脱离真实模型。
