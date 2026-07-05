# 模块素材：多模态执行器 (Multimodal Executor: Vision/Audio)  `executor-multimodal`

> 来源：litert-lm-guide/data.json（多智能体源码分析）。**这是素材，不是正文**。
> 引用进正文前，须按 CLAUDE.md 第二节逐条 Read 源码核验、补 `@ v0.13.1`。

**一句话**：该模块负责把图像和音频编码成可以直接喂给 LLM 的 soft token embedding（软令牌嵌入）。它通过 LiteRT CompiledModel 运行视觉/音频的「encoder + adapter」两段子模型，把原始像素张量或音频频谱图（spectrogram）转换成与文本词嵌入维度对齐的向量序列，并以 ExecutorVisionData / ExecutorAudioData 的形式返回，最终由上层在 token 序列中用占位符（kSpecialToken）的位置替换进去。

**在架构中的位置**：它处在「预处理 → 多模态编码 → LLM 主模型推理」链路的中间一环，是文本 executor（LlmLiteRtCompiledModelExecutor）的并列旁路。上游是 ImagePreprocessor / 音频特征提取（生成 image patch 张量、spectrogram + mask），本模块把它们编码成 embedding；下游是主 LLM executor 的 prefill 阶段：上层（runtime/framework/resource_management/resource_manager.cc 中的 ResourceManager 通过 LockedVisionExecutor / LockedAudioExecutor 包装并加锁调用）把编码结果塞进 ExecutorInputs 的 vision_data_/audio_data_，并在文本 token 序列里用 ExecutorVisionData::kSpecialToken(-1) 和 ExecutorAudioData::kSpecialToken(-2) 作为占位，主模型 prefill 时按占位符位置把这些向量行注入到词嵌入矩阵中，从而实现「图文/音文混合」上下文。VisionExecutor/AudioExecutor 只是纯接口别名，本模块的具体实现是 *LiteRtCompiledModelExecutor，与文本 executor 共享同一套 LiteRT CompiledModel / TensorBuffer / Backend 抽象。

## 关键文件
- `/Users/dpthinker/workspace/LiteRT-LM/runtime/executor/vision_executor_base.h` — VisionExecutorBase 抽象接口：定义 Encode(单张量) / Encode(张量map) / GetExpectedInputDimension / GetVisionExecutorProperties。vision_executor.h 仅把 VisionExecutor 设为它的别名。
- `/Users/dpthinker/workspace/LiteRT-LM/runtime/executor/audio_executor_base.h` — AudioExecutorBase 抽象接口：除 Encode 外还带 Reset / CreateNewContext / CloneContext / RestoreContext / LoadLoRA / UseLoRA，体现音频比视觉多了「流式有状态」和「LoRA」两条能力线。
- `/Users/dpthinker/workspace/LiteRT-LM/runtime/executor/vision_litert_compiled_model_executor.cc` — 视觉执行器实现核心。内含 VisionEncoder + VisionAdapter 两个内部类；两个 Encode 重载分别处理「固定尺寸 CNN 编码器」和「ViT 多签名(multi-signature) patch 编码器」，是理解视觉编码流程的主文件。
- `/Users/dpthinker/workspace/LiteRT-LM/runtime/executor/audio_litert_compiled_model_executor.cc` — 音频执行器实现核心(约1.1k行)。包含 AudioStaticEncoder / AudioStreamingEncoder / AudioAdapter；Encode 做分块滑窗(chunking)，EncodeInternal 跑单块 encoder+adapter，SwapInternalStateBuffers 实现流式状态在 input/output buffer 间的乒乓交换。
- `/Users/dpthinker/workspace/LiteRT-LM/runtime/executor/llm_executor_io_types.h` — 定义 ExecutorVisionData / ExecutorAudioData（embedding + per_layer_embeddings + valid_tokens）以及它们的 kSpecialToken/kEndToken 占位符常量，是本模块的输出契约；还有 AudioContext 基类与 ExecutorInputs 容器。
- `/Users/dpthinker/workspace/LiteRT-LM/runtime/executor/vision_executor_utils.cc` — 通过检视模型 signature/输入输出张量形状，反推 VisionExecutorProperties（num_tokens_per_image、patch_num_shrink_factor），用于判定是否为 ViT 及计算图像 token 数。
- `/Users/dpthinker/workspace/LiteRT-LM/runtime/executor/audio_executor_utils.cc` — 用启发式(检测 prev_ 前缀、是否有 adapter)判定音频模型是否为流式，并算出 streaming_chunk_size / overlap_size / audio_shrink_factor 等属性。
- `/Users/dpthinker/workspace/LiteRT-LM/runtime/executor/vision_executor_settings.cc` — VisionExecutorSettings：分别配置 encoder/adapter 的 backend 与权重/程序缓存文件，按文件名后缀(.vision_encoder/.vision_adapter)区分两套子模型的缓存。audio_executor_settings.cc 同理，并额外管理 max_sequence_length、num_threads、LoRA rank。

## 核心抽象
- **VisionLiteRtCompiledModelExecutor** (class) 〔`runtime/executor/vision_litert_compiled_model_executor.cc`〕：视觉执行器的具体实现，实现 VisionExecutorBase。Create() 加载 kTfLiteVisionEncoder + kTfLiteVisionAdapter 两个子模型并探测属性；Encode(TensorBuffer) 走固定输入路径(encoder→adapter)；Encode(map) 走 ViT 路径：含 images/positions_xy 两个输入、按 patch 数选 signature、用 mask 统计有效 patch 数、最后只取 num_patches 行写入输出。内部持有 VisionEncoder/VisionAdapter 两个子对象。
- **AudioLiteRtCompiledModelExecutor** (class) 〔`runtime/executor/audio_litert_compiled_model_executor.cc`〕：音频执行器具体实现。Create() 据 is_streaming_model 选择 AudioStaticEncoder 或 AudioStreamingEncoder，adapter 可选(无则直接用 encoder 输出)。Encode(spectrogram,mask) 把整段音频按 window_size/overlap_size 滑窗切块，逐块调用 EncodeInternal 并累加有效 token；EncodeInternal 负责 fp32↔fp16 转换、跑 encoder+adapter、用 output mask 数出 chunk_valid_tokens，并在流式下交换状态 buffer。
- **VisionEncoder / VisionAdapter** (class) 〔`runtime/executor/vision_litert_compiled_model_executor.h`〕：VisionLiteRtCompiledModelExecutor 的两个私有内部类，各包一个 CompiledModel 与输入/输出 TensorBuffer。Encoder 把像素/patch 编成高维 feature，Adapter 把 feature 投影/下采样成与 LLM 词嵌入维度对齐的 soft token。非 ViT 模型在 Initialize 时就预建 buffer；ViT 模型按 signature 在 Encode 时按需创建。
- **AudioEncoder / AudioStaticEncoder / AudioStreamingEncoder** (class) 〔`runtime/executor/audio_litert_compiled_model_executor.h`〕：AudioEncoder 是抽象基类(IsStreaming/Reset/LoadLoRA/UseLoRA 等)。AudioStaticEncoder 一次吃完整音频；AudioStreamingEncoder 维护大量状态张量(prev_features/prev_mask/各层 prev_q/k/v、conv_padding、feature_states)，靠 SwapInternalStateBuffers 把本次输出状态变成下次输入状态，实现跨 chunk 的因果注意力与卷积重叠。
- **ExecutorVisionData / ExecutorAudioData** (class) 〔`runtime/executor/llm_executor_io_types.h`〕：编码输出的数据契约。各含可选 embeddings([tokens,model_dim]) 与 per_layer_embeddings([layers,tokens,dim])；Audio 还含 valid_tokens 表示有效 token 数。关键常量 kSpecialToken(vision=-1,audio=-2)/kEndToken(-3/-4) 是塞进文本 token 序列里的占位符，标记每行 embedding 该插入的位置。Duplicate() 做底层 TensorBuffer 的浅拷贝。
- **AudioContext / AudioStreamingContext** (class) 〔`runtime/executor/audio_litert_compiled_model_executor.h`〕：流式音频的可保存/克隆状态快照。AudioContext 是抽象基类(Clone)；AudioStreamingContext 持有一份 state_buffers(KV cache+卷积特征/掩码)。配合 CreateNewContext/CloneContext/RestoreContext，支持多会话并发时保存与恢复音频编码器内部状态。
- **VisionExecutorProperties / AudioExecutorProperties** (struct) 〔`runtime/engine/io_types.h`〕：由 *_executor_utils.cc 通过检视模型张量形状自动推断的属性。Vision: num_tokens_per_image、patch_num_shrink_factor(ViT 标志)。Audio: is_streaming_model、streaming_chunk_size、streaming_chunk_overlap_size、audio_shrink_factor。这些值驱动 token 数计算与流式滑窗策略。

## 数据流
1. 视觉(固定输入)：上游 ImagePreprocessor 产出 [batch,H,W,C] 图像 TensorBuffer → VisionLiteRtCompiledModelExecutor::Encode(单张量) 把数据 Write 进 encoder 输入 buffer → 运行 VisionEncoder.CompiledModel → encoder 输出 feature 直接作为 VisionAdapter 输入 → 运行 VisionAdapter → 得到 [1,num_vision_tokens,model_dim] 的 embedding，包成 ExecutorVisionData 返回。
2. 视觉(ViT 多签名)：上游产出 images(patch) + positions_xy 两个张量的 map → 据 patch 数经 GetVitSignatureIndex 选最合适的 signature → 写入 encoder buffer(并对 padding 清零、对 position 初始化为 -1) → 运行 encoder → 用 mask 输出或 patch_num_shrink_factor 算出真实 num_patches → 只取前 num_patches 行 feature 喂给 adapter → 运行 adapter → 截取 num_patches 行重建为最终 embedding 张量返回。
3. 音频：上游产出 spectrogram[...,frame,feat_bins] 与可选 mask → Encode 据 GetValidCount(mask) 得有效帧数，按 window_size/overlap/stride 切成 N 个 chunk → 逐 chunk 调 EncodeInternal：写入(必要时 fp32→fp16) spectrogram/mask → 运行 AudioEncoder → 据 output mask 数出 chunk_valid_tokens → (若有)运行 AudioAdapter 投影到 model_dim → Read 出该 chunk 的 embedding 拼接 → 流式下 SwapInternalStateBuffers 把状态传给下个 chunk。
4. 汇总：所有 chunk 的有效 token 拼成 [1,total_valid_tokens,audio_dim] 的 ExecutorAudioData(带 valid_tokens) 返回。
5. 注入 LLM：上层(ResourceManager / engine_advanced_impl)把 ExecutorVisionData/ExecutorAudioData 放入 ExecutorInputs，并在文本 token 序列对应位置填 kSpecialToken(-1/-2)；主 LLM executor prefill 时按占位符把这些 embedding 行替换进词嵌入序列，实现多模态 prefill。

## 概念
- **Encoder + Adapter 两段式编码**：多模态编码拆成两个子模型：Encoder(视觉/音频骨干网络)把原始信号编成高维特征；Adapter(投影/下采样头)把特征对齐到 LLM 词嵌入的维度与 token 粒度。拆开的好处是 Adapter 通常很轻，可单独用不同 backend/精度，且能复用同一 Encoder 接不同 LLM。
- **Soft token / 软令牌 embedding**：普通文本 token 要先查 embedding 表才变成向量；而图像/音频编码器直接产出向量(soft token)，不经过词表。它们在 token 序列里用占位符(kSpecialToken)标记位置，prefill 时把对应行向量直接注入，等价于一段「没有离散 id 的词」。
- **ViT 与 patch / multi-signature**：Vision Transformer 把图像切成 patch，序列长度随图像分辨率变化。为支持不同 patch 数，模型导出多个 signature(如 vision_xxx_256/512…)，运行时按实际 patch 数选「刚好够长」的那个签名，避免一律按最大长度浪费算力。patch_num_shrink_factor 表示输入 patch 数与输出 token 数的压缩比。
- **Spectrogram / 流式分块编码**：音频先转成频谱图(spectrogram，[帧, 频率bin])再喂给编码器。长音频用滑动窗口分块(chunk)处理：window_size 是窗口长度，overlap_size 是相邻窗口重叠帧数(给卷积/注意力提供上下文)，stride=window-overlap。流式编码器还需在块之间传递内部状态。
- **音频流式状态(KV cache / conv state)**：流式编码器为保证跨块的因果一致性，需要把上一块的注意力 K/V、卷积重叠特征(prev_features/conv_padding/feature_states)等状态带到下一块。本模块用 input/output 两套状态 buffer，每块结束后 SwapInternalStateBuffers 互换，实现零拷贝的状态接力；AudioContext 则把这套状态打包以便保存/克隆/恢复。
- **valid_tokens 与 mask**：因为按固定窗口/最大长度分配 buffer，实际有效内容往往更短。mask 标记哪些帧/patch 是真实数据，编码后用 output mask 数出有效 token 数(valid_tokens)，只取前 valid_tokens 行作为真正注入 LLM 的 embedding，避免把 padding 当成内容。

## 优化
- **按需 signature 选择 (ViT multi-signature)**：GetVitSignatureIndex 据实际 patch 数选「>= 所需长度且最短」的 signature，短图用短模型推理，避免一律按最长序列计算，显著省算力；非 ViT 单签名模型则在 Initialize 时预建 buffer 复用。
- **valid_tokens 截断**：视觉用 mask 统计真实 num_patches、音频用 output mask 数出 chunk_valid_tokens，最终只写入/注入有效行，避免把 padding 当内容传给 LLM，既省 prefill 长度又保证正确性。
- **流式状态零拷贝乒乓交换**：AudioStreamingEncoder 用 input/output 两套状态 buffer，每块结束 std::swap 指针式互换而非深拷贝，实现 KV cache / 卷积特征在 chunk 间的低开销接力。
- **fp16/fp32 就地转换与精度可配**：encoder 输出 fp16、adapter 需要 fp32 时在 EncodeInternal 内用 ScopedLock 直接逐元素转换，省去中间 buffer；GPU backend 还可据 ActivationDataType 选 fp16/fp32 精度。
- **权重/程序缓存分离 (encoder vs adapter)**：Settings 用 .vision_encoder/.vision_adapter(音频含 .static/.streaming_audio_encoder) 后缀区分两套子模型的 weight cache 与 program cache(GPU shader)，避免 XNNPACK/GPU 重复编译，加速冷启动。
- **后端专属调优**：SetGpuOptions 开启 constant tensor sharing、按平台选 Metal argument buffers / texture weights、GPU 上转换权重；CPU 开 XNNPACK dynamic fully-connected;NPU 设 HTP/GoogleTensor 的 Burst 性能模式。
- **WebGPU/Metal 输出 buffer 重建规避锁冲突**：视觉 Encode 检测到 encoder 输出在 WebGPU/Metal 内存时重新 CreateOutputBuffers，规避第二次调用时 lock TensorBuffer 失败(见 b/457483190)。

## 关键代码片段（待核验 @ v0.13.1）
**视觉编码的两段式核心：encoder 输出直接当 adapter 输入，得到 soft token embedding** — 待核验：`runtime/executor/vision_litert_compiled_model_executor.cc:458-467`
```cpp
LITERT_RETURN_IF_ERROR(vision_encoder_->GetCompiledModel().Run(
    /*input_buffers=*/vision_encoder_->GetInputBuffers(),
    /*output_buffers=*/encoder_outputs));

LITERT_RETURN_IF_ERROR(vision_adapter_->GetCompiledModel().Run(
    /*input_buffers=*/encoder_outputs,
    /*output_buffers=*/output_tensor_buffers));

return ExecutorVisionData(std::move(output_tensor_buffers[0]),
                          /*per_layer_embeddings=*/std::nullopt);
```
**ViT 多签名：按实际 patch 数挑选「刚好够长」的 signature，避免按最大长度浪费算力** — 待核验：`runtime/executor/vision_litert_compiled_model_executor.cc:155-177`
```cpp
const int max_num_tokens =
    num_patches / vision_executor_properties.patch_num_shrink_factor.value();
for (int i = 0; i < model.GetNumSignatures(); ++i) {
  ...
  if (current_length >= max_num_tokens && current_length < best_length) {
    best_length = current_length;
    best_signature_index = i;
  }
}
```
**占位符契约：图像/音频 embedding 用 kSpecialToken 标记在文本 token 序列里的注入位置** — 待核验：`runtime/executor/llm_executor_io_types.h:218-286`
```cpp
class ExecutorVisionData {
 public:
  static constexpr int kSpecialToken = -1;
  static constexpr int kEndToken = -3;
...
class ExecutorAudioData {
 public:
  static constexpr int kSpecialToken = -2;
  static constexpr int kEndToken = -4;
```
**音频滑窗分块：按 window/overlap/stride 切块，逐块 EncodeInternal 并累加有效 token** — 待核验：`runtime/executor/audio_litert_compiled_model_executor.cc:974-995`
```cpp
while (pos == 0 || pos + overlap_size < total_frames) {
  int chunk_len = std::min(window_size, total_frames - pos);
  auto spectrogram_slice = absl::MakeSpan(spectrogram_host_buffer)
      .subspan(pos * feature_dim, chunk_len * feature_dim);
  ...
  ASSIGN_OR_RETURN(int chunk_valid_tokens,
                   EncodeInternal(spectrogram_slice, spectrogram_mask_slice,
                                  audio_embeddings_slice));
  total_valid_tokens += chunk_valid_tokens;
  pos += stride;
}
```
**流式状态接力：把本次输出状态与下次输入状态做乒乓交换，实现跨 chunk 因果上下文** — 待核验：`runtime/executor/audio_litert_compiled_model_executor.cc:494-499`
```cpp
for (const auto& input_name : all_input_names) {
  if (output_buffers_map_.contains(input_name)) {
    std::swap(input_buffers_map_[input_name],
              output_buffers_map_[input_name]);
  }
}
```

## 入手顺序
- 先读 vision_executor_base.h 和 audio_executor_base.h，搞清两个执行器对外承诺的最小接口(Encode + 属性查询)，注意 vision_executor.h/audio_executor.h 只是别名。
- 读 llm_executor_io_types.h 中的 ExecutorVisionData / ExecutorAudioData 及其 kSpecialToken/kEndToken 注释(180-331 行)，理解输出契约和它如何嵌进文本 token 序列——这是理解整个模块目的的关键。
- 读 vision_litert_compiled_model_executor.cc 的两个 Encode 实现(430-679 行)：先看简单的固定输入版，再看 ViT map 版的 signature 选择与 num_patches 截断逻辑。
- 读 vision_executor_utils.cc + io_types.h 里的 VisionExecutorProperties，理解 patch_num_shrink_factor / num_tokens_per_image 是怎么从模型张量形状推断出来的。
- 读 audio_litert_compiled_model_executor.cc 的 Encode(896-1037) 与 EncodeInternal(781-894)，重点看滑窗分块、fp16 转换、valid_tokens 统计；再看 SwapInternalStateBuffers(490-501) 与 AudioStreamingContext 的 Create/Clone/Restore，理解流式状态管理。
- 最后到 runtime/framework/resource_management/resource_manager.cc 看 LockedVisionExecutor/LockedAudioExecutor(98-176 行) 和编码结果如何被放进 ExecutorInputs，串起它与文本 executor、主 LLM 推理的关系。
