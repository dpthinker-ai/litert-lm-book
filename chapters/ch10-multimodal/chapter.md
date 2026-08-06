# 第 10 章 多模态与工具调用：视觉/音频编码、约束解码与函数调用

> 本章目标：说明图像、音频输入的 embedding 转换路径，以及约束解码的 token 屏蔽过程。前者扩展 prefill 输入，后者限制 decode 输出；文法只保证其中编码的结构条件，工具校验与执行仍由应用负责。

文本推理以 token 序列为输入，并逐 token 输出文本。多模态路径在 prefill 前增加模态编码与 embedding 查找。图像和音频由此转换为主干模型接收的向量序列。工具调用路径则在 decode 采样时增加约束状态与 logit 掩码，以产生可解析的结构化文本。

## 10.1　从 Message 到 InputData：入口类型与所有权

Conversation API 接收的 `Message` 是 `nlohmann::ordered_json`。一条消息可以同时包含文本、图像和音频项。`JsonPreface` 中的 `messages`、`tools` 与 `extra_context` 也使用同一 JSON 类型（`runtime/conversation/io_types.h:26-59`）。这层接口表达的是消息语义，还不是主干模型的张量输入。

图像和音频项有两种数据来源。`path` 指向本地文件，`blob` 保存 base64 字符串。`LoadItemData` 对前者创建 `MemoryMappedFile`，对后者先解码 base64，再创建 `InMemoryFile`。两者都没有时，函数返回 `InvalidArgumentError`；未知的 `type` 返回 `UnimplementedError`（`runtime/conversation/model_data_processor/data_utils.cc:30-56`）。因此，文件读取、base64 解码和模态推理是三个不同的失败阶段。

模型数据处理器（ModelDataProcessor）把渲染后的 prompt 与原始消息一起转换为 `std::vector<InputData>`。这个步骤不可由聊天模板单独完成。模板只产生模型约定的模态标记；处理器还要按消息顺序取出二进制对象、执行预处理，并把文本片段与模态对象重新排成一条输入序列。`TypeSafeModelDataProcessor` 在进入具体实现前检查逐轮参数类型（`runtime/conversation/model_data_processor/model_data_processor.h:144-172`）。

以 Gemma 4 路径为例，处理器先扫描所有消息，把图像与音频文件分别放入两个队列。随后，它遍历渲染 prompt 中的模态标记。遇到图像标记便从图像队列头部取一个对象，预处理后插入 `InputImage` 和 `InputImageEnd`；音频路径相应插入 `InputAudio` 和 `InputAudioEnd`（`runtime/conversation/model_data_processor/gemma4_data_processor.cc:348-455`）。队列顺序决定“第几个标记”对应“第几个二进制对象”，而不是文件名或内容哈希。

下面的精简代码展示了输入层的类型边界（`runtime/engine/io_types.h:93-238`）：

```cpp
class InputImage {
 public:
  explicit InputImage(
      std::variant<std::string, absl::string_view, TensorBuffer,
                   absl::flat_hash_map<std::string, TensorBuffer>> data);
  InputImage(const InputImage&) = delete;
  InputImage(InputImage&&) = default;
};

class InputAudio {
 public:
  explicit InputAudio(
      std::variant<std::string, TensorBuffer, std::vector<float>> data);
  InputAudio(const InputAudio&) = delete;
  InputAudio(InputAudio&&) = default;
};

using InputData = std::variant<InputText, InputImage, InputAudio,
                               InputImageEnd, InputAudioEnd>;
```

`InputImage` 和 `InputAudio` 都禁止复制并允许移动。这一约束使二进制数据或 `TensorBuffer` 的转移在类型上可见。两者的 `CreateCopy` 仍可显式复制：字符串做字节复制，`TensorBuffer` 调用 `Duplicate`；后者复制句柄，底层缓冲为浅拷贝（`runtime/engine/io_types.cc:105-164`、`runtime/executor/llm_executor_io_types.h:251-254`）。调用方不能把 `Duplicate` 理解为重新分配并复制全部张量内容。

`InputImage` 的 `std::string` 分支持有自己的字节；`absl::string_view` 分支只保存视图。后者的源缓冲必须至少存活到预处理完成。这一点来自成员类型和 `GetRawImageBytes` 的实现，而不是额外的生命周期管理（`runtime/engine/io_types.h:93-146`、`runtime/engine/io_types.cc:75-83`）。Gemma 4 的 `path/blob` 路径在调用预处理器前构造 `std::string`，因此不会把文件映射的裸视图继续传入异步 prefill（`runtime/conversation/model_data_processor/gemma4_data_processor.cc:414-440`）。

预处理完成后，图像通常变为单个 `TensorBuffer` 或名为 `images`、`positions_xy` 的张量表；音频变为频谱 `TensorBuffer`。Conversation 把这组 `InputData` 传递给 `RunPrefillAsync`（`runtime/conversation/conversation.cc:506-515`）。该接口接收 const 引用，并在返回前生成 `preprocessed_contents`。张量通过 `CreateCopy` 调用 `Duplicate`，随后这份内部向量被移动到执行任务（`runtime/core/session_utils.cc:168-218`、`runtime/core/session_advanced.cc:85-135`）。调用方不必让原始 `InputData` 存活到异步任务结束，但底层张量复制仍是句柄级浅拷贝。

| 阶段 | 输入表示 | 输出表示 | 所有权或有效期 | 典型检查 |
|---|---|---|---|---|
| 消息入口 | `Message` 中的 `path/blob` | 文件映射或内存文件 | 由处理器局部队列持有 | 类型、路径、base64 |
| 模型数据处理 | prompt 标记 + 二进制对象 | `vector<InputData>` | 模态对象按值移动 | 标记数与对象数 |
| 图像预处理 | 编码图像字节 | `TensorBuffer` 或张量表 | `InputImage` 持有缓冲句柄 | 解码、通道、patch 配置 |
| 音频预处理 | 编码音频或 PCM | 频谱 `TensorBuffer` | `InputAudio` 持有缓冲句柄 | 格式、采样配置、帧数 |
| 模态编码 | 预处理张量 | `ExecutorVisionData` / `ExecutorAudioData` | 执行器输出随后被移动或合并 | signature、shape、数据类型 |
| prefill 组合 | 文本 token + 模态 embedding | `ExecutorInputs` | 任务闭包持有直至 prefill 完成 | 占位符数、embedding 行数 |

> 表 10-1　多模态输入在各阶段改变表示；排查悬空视图和 shape 错误时，应先确定对象处于哪一层。

## 10.2　图像输入的编码与 embedding 替换

Transformer 主干处理 token 序列（第 3 章），不能直接接收像素。图像需要先转换为与文本 embedding 兼容的向量序列。

文本 token 通过词嵌入（embedding）查表得到向量。图像没有可直接查表的 token ID，需由视觉编码器与适配器生成同维度的 embedding。执行管理器随后插入视觉占位符，prefill 查找器再把这些向量写入对应位置。

图像预处理通过 patchify（切块）把图像划分为 patch。执行 patchify 前，`GetAspectRatioPreservingSize` 根据原图宽高、patch 大小和 patch 数上限计算目标尺寸（`runtime/components/preprocessor/image_preprocessor_utils.cc:26-75`）：

```cpp
float total_px = width * height;
float target_px =
    patchify_config.max_num_patches *
    (patchify_config.patch_width * patchify_config.patch_height);  // (1)
float factor = std::sqrt(target_px / total_px);                    // (2)
float ideal_height = factor * height;
float ideal_width = factor * width;
int side_mult =
    patchify_config.pooling_kernel_size * patchify_config.patch_width;  // (3)
int target_height =
    static_cast<int>(std::floor(ideal_height / side_mult)) * side_mult;  // (4)
int target_width =
    static_cast<int>(std::floor(ideal_width / side_mult)) * side_mult;
```

(1) `target_px` 是 patch 数上限对应的像素面积。(2) `factor` 是保持长宽比的缩放系数；未取整时，缩放后的面积等于 `target_px`。(3)(4) 目标宽高还要向下对齐到 `side_mult` 的整数倍。`side_mult` 等于 `pooling_kernel_size × patch_width`，因此 patch 网格的每一边都能按池化核尺寸分组。函数只接受正方形 patch。若极端长宽比使某一边向下取整为 0，代码把该边设为一个 `side_mult`，再按原始比例计算另一边，并限制其不超过 `max_side_length`。

目标尺寸确定后，记 patch 数为 \\(N_{patch}\\)：

\\[
N_{patch}=\frac{\text{target\_height}\times\text{target\_width}}
{\text{patch\_height}\times\text{patch\_width}}.
\\]

该值不是通用的 visual token 数公式。对带 `patch_num_shrink_factor` 的 ViT 路径，编码器可能不返回 mask。此时，执行器按 \\(\lceil N_{patch}/\text{patch\_num\_shrink\_factor}\rceil\\) 计算有效输出行数（`runtime/executor/vision_litert_compiled_model_executor.cc:604-624`）。序列中的 `kSpecialToken` 数最终以视觉 embedding 的行数为准。`pooling_kernel_size` 影响目标尺寸对齐，不能单独用来推导占位符数量。

上面的函数只计算目标尺寸。`MaybeResizeImageWithSameAspectRatio` 负责实际重采样（`runtime/components/preprocessor/stb_image_preprocessor.cc:56-115`）：

```cpp
ASSIGN_OR_RETURN(auto size,
                 GetAspectRatioPreservingSize(
                     width, height, parameter.GetPatchifyConfig().value()));
int new_height = size.first;
int new_width = size.second;

if (new_height == height && new_width == width) {
  resized_image_data = std::move(image_data);
  return absl::OkStatus();                                  // (1)
}
// ...
for (int i = 0; i < batch_size; ++i) {
  // ...
  if (stbir_resize(input_data, width, height, 0, output_data, new_width,
                   new_height, 0,
                   static_cast<stbir_pixel_layout>(channels),
                   STBIR_TYPE_UINT8_SRGB, STBIR_EDGE_CLAMP,
                   STBIR_FILTER_CATMULLROM) == 0) {          // (2)
    return absl::InternalError("Failed to resize image.");
  }
}
```

(1) 若目标尺寸与原图相同，函数移动原始缓冲并直接返回。(2) 否则，它对 batch 中的图像逐一调用 `stbir_resize`，采用 `STBIR_FILTER_CATMULLROM`、`STBIR_TYPE_UINT8_SRGB` 与 `STBIR_EDGE_CLAMP`。这段代码在当前函数内同步执行，没有并行调度或与视觉编码器重叠的逻辑。重采样在端到端时延中的占比仍需在目标设备上测量，不能仅由滤波器类型或输入分辨率推定。

预处理器先把像素除以 255，再由 `PatchifyImage` 重排为 patch 序列（`runtime/components/preprocessor/stb_image_preprocessor.cc:265-283`、`runtime/components/preprocessor/stb_image_preprocessor.cc:120-220`）。输出 `images` 的形状是 `[batch, num_patches, patch_dim]`，`positions_xy` 的形状是 `[batch, num_patches, 2]`。核心是一段六重循环：

```cpp
for (int b = 0; b < batch_size; ++b) {
  for (int h = 0; h < num_patches_h; ++h) {
    for (int w = 0; w < num_patches_w; ++w) {
      int patch_idx = h * num_patches_w + w;
      int global_patch_idx = b * num_patches + patch_idx;
      positions_ptr[global_patch_idx * 2] = w;
      positions_ptr[global_patch_idx * 2 + 1] = h;              // (1)
      for (int ph = 0; ph < patch_height; ++ph) {
        for (int pw = 0; pw < patch_width; ++pw) {
          for (int c = 0; c < channels; ++c) {
            int src_h = h * patch_height + ph;
            int src_w = w * patch_width + pw;
            int src_idx =
                ((b * height + src_h) * width + src_w) * channels + c;  // (2)
            int dest_idx = global_patch_idx * patch_dim +
                           ((ph * patch_width + pw) * channels + c);    // (3)
            patches_ptr[dest_idx] = static_cast<float>(image_data[src_idx]);
          }
        }
      }
    }
  }
}
```

(1) 外三层遍历 batch 与 patch 网格，并把每个 patch 的 \\((w,h)\\) 坐标写入 `positions_xy`。(2) `src_idx` 按 HWC 行主序读取原图。(3) `dest_idx` 使同一 patch 的 `patch_dim = patch_width × patch_height × channels` 个值连续存放。源码使用逐元素标量循环，没有显式 SIMD。它是否构成预处理热点，需要结合编译器向量化结果与真机 profile 判断。

patchify 输出由视觉执行器编码。`Encode` 的声明位于 `runtime/executor/vision_litert_compiled_model_executor.h:51-62`，单张量重载的实现位于 `runtime/executor/vision_litert_compiled_model_executor.cc:454-491`。该函数先运行视觉编码器，再运行视觉适配器，并返回 `ExecutorVisionData`：

```cpp
absl::StatusOr<ExecutorVisionData> VisionLiteRtCompiledModelExecutor::Encode(
    const litert::TensorBuffer& input_image_tensor) {
  // ...
  LITERT_ASSIGN_OR_RETURN(auto input_image_data,
                          ReferTensorBufferAsSpan<float>(input_image_tensor));
  LITERT_RETURN_IF_ERROR(
      vision_encoder_->GetMutableInputBuffers()[0].Write<float>(
          input_image_data));                                        // (1)
  // ...
  LITERT_RETURN_IF_ERROR(vision_encoder_->GetCompiledModel().Run(
      /*input_buffers=*/vision_encoder_->GetInputBuffers(),
      /*output_buffers=*/encoder_outputs));                          // (2)
  LITERT_RETURN_IF_ERROR(vision_adapter_->GetCompiledModel().Run(
      /*input_buffers=*/encoder_outputs,
      /*output_buffers=*/output_tensor_buffers));                    // (3)
  return ExecutorVisionData(std::move(output_tensor_buffers[0]),
                            /*per_layer_embeddings=*/std::nullopt);   // (4)
}
```

(1) 预处理后的图像张量被写入编码器输入 buffer。(2) 视觉编码器产生中间特征。(3) 视觉适配器将中间特征映射为主干接收的 embedding。(4) 该重载返回常规 embedding，并把 `per_layer_embeddings` 设为 `std::nullopt`。被省略的分支在 WebGPU 或 Metal 输出 buffer 上每次重新创建缓冲。源码给出的原因是，复用会使第二次 `Encode` 无法锁定 TensorBuffer（`runtime/executor/vision_litert_compiled_model_executor.cc:471-480`）。

视觉 embedding 通过占位符写入组合序列。视觉占位符 `ExecutorVisionData::kSpecialToken` 的值为 -1；头文件同时规定，视觉 embedding 的行数必须等于输入 token 序列中的视觉占位符数量（`runtime/executor/llm_executor_io_types.h:189-220`）：

```cpp
// token_ids = [2, kSpecialToken, kSpecialToken, kSpecialToken, 106, 77, (other
// text token ids)...] (contains 3 vision tokens)
//
// Then, the vision embeddings should have shape [3,
// model_dimension]:
// [[0.1, ...],  // Embedding for the 1st kVisionSpecialToken
//  [0.5, ...],  // Embedding for the 2nd kVisionSpecialToken
//  [0.9, ...]]  // Embedding for the 3rd kVisionSpecialToken
```

执行管理器从视觉 embedding 张量的倒数第二维读取 `image_token_num`，再向组合 token 序列插入同样数量的 `kSpecialToken`（`runtime/framework/resource_management/serial_execution_manager.cc:592-600`）。prefill 查表时，非负 token 由文本 embedding 查找器处理。`EmbeddingLookupMultiModal` 则依次把视觉 embedding 行复制到 `kSpecialToken` 对应位置（`runtime/components/embedding_lookup/embedding_lookup_manager.cc:147-165`、`runtime/components/embedding_lookup/embedding_lookup_multi_modal.cc:145-158`）。因此，占位符数量与 embedding 行数必须一致。多张图的 embedding 会先合并，相应的占位符也按各图输出行数插入序列。

<figure>
{{#include figs/fig-10-1.svg}}
<figcaption>图 10-1　图像经 patchify、视觉编码器与适配器得到 embedding；prefill 查表再按 `kSpecialToken` 位置写入对应行。</figcaption>
</figure>

### 10.2.1　变分辨率的视觉编码

另一个 `Encode` 重载接收含 `images` 与 `positions_xy` 的 `input_maps`。它从 `images` 张量的第 1 维读取实际 patch 数，并分别为视觉编码器和适配器选择 signature（`runtime/executor/vision_litert_compiled_model_executor.cc:499-535`）：

```cpp
const auto& images_dimensions = images_tensor_type.Layout().Dimensions();
const int num_patches_from_input = images_dimensions[1];               // (1)
ASSIGN_OR_RETURN(auto encoder_signature_index,
                 GetVitSignatureIndex(vision_encoder_->GetModel(),
                                      vision_executor_properties_,
                                      num_patches_from_input));         // (2)
```

(1) patch 数取自输入张量第 1 维，即 patchify 产生的 `num_patches_h × num_patches_w`。(2) `GetVitSignatureIndex` 根据该数值选择 signature。完整逻辑位于 `runtime/executor/vision_litert_compiled_model_executor.cc:138-195`：

```cpp
const int max_num_tokens =
    num_patches / vision_executor_properties.patch_num_shrink_factor.value();  // (1)
for (int i = 0; i < model.GetNumSignatures(); ++i) {
  // ...
  if (current_length >= max_num_tokens && current_length < best_length) {  // (2)
    best_length = current_length;
    best_signature_index = i;
  }
}
```

(1) 代码用整数除法把输入 patch 数除以 `patch_num_shrink_factor`，得到用于选择 signature 的 `max_num_tokens`。(2) 随后遍历所有以 `kVisionLengthPrefix` 开头的 signature，从名称末尾解析长度，选择不小于 `max_num_tokens` 的最短入口。若没有入口满足长度，函数返回错误，并在消息中说明继续执行会截断输入图像。

多 signature 让运行时按输入长度选择较近的固定入口，而不是始终使用最长入口。模型因此需要暴露多组 signature 定义，后端也要能为这些入口创建相应 buffer。signature 是命名入口；其数量不能证明模型文件复制了同样数量的 encoder 权重。判断权重是否共享时，还须检查各 signature 指向的 subgraph 与常量张量。编译产物增加多少，同样不能从 `GetNumSignatures()` 推断。

当模型只有一个 signature 时，`GetVitSignatureIndex` 直接返回索引 0（`runtime/executor/vision_litert_compiled_model_executor.cc:138-144`）。这只能说明无需比较入口长度；模型是否支持变分辨率，还要结合输入张量与预处理配置判断。

附录 D 对 Gemma 4 E4B 模型文件的检查记录显示，视觉编码器段包含 `vision_70`、`vision_140`、`vision_280`，适配器段包含对应的 `vision_adapter_70/140/280`。关键输入张量 `images[1, 1260, 768]` 表明，该入口的每个 patch 有 768 个值。结合预处理配置可得 \\(768=16\times16\times3\\)，对应 16 × 16 的 RGB patch。`[1, 1260, 768]` 只是一条已记录的 signature 形状，不能据此把 1260 当成所有入口的统一容量上限。

### 10.2.2　visual token budget 如何作用于 patchify

Gemma 4 的逐轮参数可以设置 `visual_token_budget`。处理器先要求该值为正，再乘以 9，最后与模型配置中的 `max_num_patches` 取较小值。得到的 patch 上限写入 `PatchifyConfig`（`runtime/conversation/model_data_processor/gemma4_data_processor.cc:381-400`）：

```cpp
int max_num_patches = config_.max_num_patches;
if (args.visual_token_budget) {
  int visual_token_budget = args.visual_token_budget.value();
  if (visual_token_budget <= 0) {
    return absl::InvalidArgumentError(
        "Visual token budget must be positive.");
  }
  max_num_patches =
      std::min(max_num_patches, visual_token_budget * 9);
}
```

这里的 9 来自 3 × 3 patch 池化约定。设模型配置的 patch 上限为 \\(P_{cfg}\\)，调用方给出的 visual token budget 为 \\(T_{budget}\\)，则预处理实际使用

\\[
P_{limit}=\min(P_{cfg},9T_{budget}).
\\]

这个式子限制的是进入视觉编码器的 patch 数，而不是直接截取适配器输出。`GetAspectRatioPreservingSize` 还会保持长宽比，并把宽高向下对齐到 `pooling_kernel_size × patch_width` 的整数倍。因此，实际 \\(N_{patch}\\) 一般不超过 \\(P_{limit}\\)，但不一定等于它。

若编码器没有输出 mask，运行时再按 `patch_num_shrink_factor` 对实际 patch 数向上取整，得到视觉 embedding 的有效行数。对 3 × 3 池化且 shrink factor 为 9 的模型，预算与输出行数通常一致或更小；“通常”不构成接口保证。最终行数仍由编码器输出 mask 或执行器的取整分支确定（`runtime/executor/vision_litert_compiled_model_executor.cc:604-636`）。

预算过小与预算过大导致的问题不同。小预算减少图像空间采样点，可能丢失细节；这是模型质量问题，运行时不会把它报告为错误。大预算若超过模型最长视觉 signature，`GetVitSignatureIndex` 会返回错误，并报告可用的最大长度，避免静默截断（`runtime/executor/vision_litert_compiled_model_executor.cc:181-195`）。本书尚未完成同一图像在不同预算下的真机质量与时延对照，因此不提供推荐数值。

### 10.2.3　多图组合不是 batch

一条消息可以包含多幅图。每幅图分别执行预处理与视觉编码，得到各自的 `ExecutorVisionData`。执行管理器随后调用 `CombineExecutorVisionData`，沿 token 维拼接 embedding（`runtime/framework/resource_management/serial_execution_manager.cc:640-648`、`runtime/util/executor_data_util.cc:35-126`）。

假设两幅图分别产生形状 `[1,T_1,D]` 和 `[1,T_2,D]`。组合结果不是 `[2,T,D]`，而是 `[1,1,T_1+T_2,D]`。若输入已经是四维张量，则保留前两维，只累加倒数第二维。对应的占位符也按消息顺序插入同一文本序列。

这一区别影响三个判断。第一，多图不会自动变成可并行的图像 batch；当前组合函数先逐图编码，再拼接结果。第二，主干看到的是一条更长的上下文，而不是多条独立样本。第三，每幅图的边界要由模型约定的开始/结束 token 表达，不能从拼接后的 embedding 张量恢复。

组合一个对象时，函数直接移动原有 `ExecutorVisionData`。组合多个对象时，它创建新的 host `TensorBuffer`，依次锁定每个源缓冲并复制其 packed bytes（`runtime/util/executor_data_util.cc:40-104`）。因此，单图和多图经过的内存路径不同。源码只能证明多图存在一次组合复制，不能据此估计它在端到端时延中的占比。

### 10.2.4　视觉 token 折算成的 prefill 与 KV cache 开销

视觉编码器输出的 visual token 数决定占位符数量。它还会增加 prefill 序列长度与有效 KV 数据量。visual token 写入 embedding 序列后，与文本 token 一样通过主干前向。各层都会为这些位置产生 K/V。

一幅图像产生 \\(T_{vis}\\) 个 visual token 时，prefill 序列会增加 \\(T_{vis}\\) 个位置。对应的 KV 数据量不能用 `model_dimension` 计算。对各层 KV 形状可能不同的模型，增量为

\\[
\Delta B_{KV}=2T_{vis}\sum_{l=1}^{L}\left(H_{kv,l}D_l b_l\right),
\\]

其中，2 表示 K 和 V。\\(H_{kv,l}\\) 是第 \\(l\\) 层的 KV 头数，\\(D_l\\) 是每个 KV 头的维度；\\(b_l\\) 是每个元素的字节数。若所有层形状和类型相同，公式简化为 \\(2L H_{kv}D T_{vis}b\\)。`model_dimension` 是主干 embedding 宽度，不一定等于 \\(H_{kv}\times D\\)。

附录 D 记录的 Gemma 4 E4B 有 24 层 INT8 KV。20 层为 \\(H_{kv}=2,D=256\\)，其余 4 层为 \\(H_{kv}=2,D=512\\)。若一次图像输入增加 256 个 visual token，活动 KV 数据量增加

\\[
2\times256\times2\times(20\times256+4\times512)\times1\ \text{B}
=7\ \text{MiB}.
\\]

三次同样的输入会增加 768 个序列位置，对应 21 MiB 的活动 KV 数据。这里计算的是有效上下文对应的数据量或容量需求。固定宽度实现可能已经按最大序列长度预分配 KV buffer。此时，插入图像未必使进程驻留内存再增长 7 MiB。prefill 时延还受注意力、后端与 padding 影响，需要真机测量。图像预算应同时考虑 `max_num_patches` 与 `patch_num_shrink_factor`，前者不能直接等同于 visual token 数。

## 10.3　音频输入的频谱编码与 embedding 替换

音频 embedding 采用另一种占位符：`ExecutorAudioData::kSpecialToken` 为 -2，且 embedding 行数应与音频占位符数量一致（`runtime/executor/llm_executor_io_types.h:263-297`）。执行管理器根据 `GetValidTokens()` 插入相同数量的 -2（`runtime/framework/resource_management/serial_execution_manager.cc:615-625`），prefill 查表再由 `EmbeddingLookupMultiModal` 复制音频 embedding。文本、视觉和音频由不同查找器提供向量，但主干接收的都是 embedding 序列。

### 10.3.1　DSP（数字信号处理）前端与分块编码

编码器接收的不是原始波形，而是 log-mel 频谱。预处理器先对分帧信号加窗，调用 `kiss_fftr` 计算实数 FFT，再保存复数结果的平方幅度（`runtime/components/preprocessor/audio_preprocessor_miniaudio.cc:224-274`）。`MelFilterbank` 把每个平方幅度谱切片转换为三角 mel 加权的滤波器组输出；初始化参数包括 FFT bin 数、采样率、mel 通道数和频率上下限（`runtime/components/preprocessor/mel_filterbank.h:25-53`）。随后，代码按配置在取对数前加 `mel floor`，或在取对数后用该阈值截断，并可继续做标准化（`runtime/components/preprocessor/audio_preprocessor_miniaudio.cc:277-308`）。最终张量形状为 `[1, num_frames, num_mel_bins]`（`runtime/components/preprocessor/audio_preprocessor_miniaudio.cc:347-365`）。这里的 `num_frames` 是 log-mel 频谱的时间帧数。

音频执行器按固定的 `sequence_length_` 分块处理频谱。`Encode` 先检查频谱与 mask 的序列长度，再进入循环（`runtime/executor/audio_litert_compiled_model_executor.cc:941-1010`）：

```cpp
  // Chunk the spectrogram into smaller pieces and encode them one by one.
  int total_valid_tokens = 0;
  int pos = 0;
  while (pos < input_sequence_length) {                        // (1)
    int end = std::min(pos + sequence_length_, input_sequence_length);
```

(1) 每轮截取至多 `sequence_length_` 个频谱时间位置。`EncodeInternal` 返回该块的有效输出 token 数；没有输出 mask 时，这个数按 `ceil(input_valid_tokens / encoder_shrinking_factor_)` 计算（`runtime/executor/audio_litert_compiled_model_executor.cc:866-874`）。各块结果顺序写入输出，`total_valid_tokens` 记录总有效长度。循环没有并行提交多个块，因而块间按顺序执行。

音频预处理包含分帧、FFT 与 mel 滤波，编码器还可能运行多个顺序块。视觉与音频进入主干后都增加 prefill 序列位置，并按上一节的公式增加有效 KV 数据量。两种预处理在端到端时延中的占比不能只由代码结构比较，仍需在相同设备和后端上 profile。

附录 D 的模型文件检查记录显示，音频编码器与 `audio_adapter` 分属两个模型段；适配器输入 `features` 的形状为 `[1, 204, 1536]`。其中 1536 是每个位置的特征维度，204 是编码器输出的特征序列位置数。204 不是原始波形帧数，也不能仅凭该张量形状还原 log-mel 输入帧数；二者还隔着编码器的缩减过程。该记录只能支持“编码器输出再进入适配器”。适配器输出形状仍须检查输出张量，不能由 1536 维的输入直接推出。

### 10.3.2　音频状态、分块边界与有效 token

音频预处理器可能保存跨调用状态。`AudioPreprocessorMiniAudio::Reset` 会清空 `input_queue_`，再按是否采用半因果 padding 重设下一帧所需样本数（`runtime/components/preprocessor/audio_preprocessor_miniaudio.h:69-77`）。Gemma 4 数据处理器每处理完一个音频对象便调用 `Reset`，然后插入 `InputAudioEnd`（`runtime/conversation/model_data_processor/gemma4_data_processor.cc:425-440`）。因此，同一消息中的两个完整音频文件不会在 DSP 输入队列中直接相接。

这个 reset 只作用于 DSP 预处理器。音频执行器还可以有流式编码器状态。`EncodeInternal` 在每个块完成后检查 `IsStreaming()`，若为真则交换内部状态 buffer（`runtime/executor/audio_litert_compiled_model_executor.cc:933-937`）。Session 创建与克隆另行创建或复制音频 context，不能用一次预处理器 reset 代表整个音频执行器已重置（`runtime/framework/resource_management/resource_manager.cc:591-648`）。

设有效频谱长度为 \\(S\\)，编码器固定块长为 \\(C\\)，缩减因子为 \\(R\\)。代码按 `[pos,min(pos+C,S))` 顺序处理，因此块数为

\\[
N_{chunk}=\left\lceil\frac{S}{C}\right\rceil.
\\]

没有输出 mask 时，第 \\(i\\) 个块的有效输出长度是 \\(\lceil S_i/R\rceil\\)，总 audio token 数为各块结果之和，而不一定等于 \\(\lceil S/R\rceil\\)。只有当块边界与缩减因子对齐时，两式才相等。实现按块累加 `chunk_valid_tokens`，并用总和创建 `[1,total_valid_tokens,audio_embedding_dimensions]` 张量（`runtime/executor/audio_litert_compiled_model_executor.cc:985-1027`）。

这个细节会影响离线复算。若只知道整段频谱长度和缩减因子，却不知道 `sequence_length_`，便不能在所有情况下还原输出 token 数。若编码器提供输出 mask，则有效长度直接由 mask 中的有效项计数，不再使用上述取整公式（`runtime/executor/audio_litert_compiled_model_executor.cc:866-874`）。

执行器在编码前检查频谱与 mask 的序列长度是否相等，并检查频谱特征宽度是否符合模型期望（`runtime/executor/audio_litert_compiled_model_executor.cc:941-975`）。这两项检查发生在模型运行前。收到“sequence length must match”时，应先核对预处理张量和 mask；收到“feature dimension must match”时，应核对 mel 配置或是否把其他模型的预处理结果传入当前执行器。

图像和音频的 shape 账本如下。表中符号只表达各层关系，不替代模型元数据或运行时张量检查。

| 模态 | 阶段 | 典型 shape | 该 shape 能证明什么 | 不能据此推出什么 |
|---|---|---|---|---|
| 图像 | 解码后像素 | `[H,W,3]` | 已转换为三通道像素 | visual token 数 |
| 图像 | patchify | `[1,N_patch,P_hP_w3]` | patch 数与每块展开宽度 | 适配器输出行数 |
| 图像 | patch 位置 | `[1,N_patch,2]` | 每个 patch 有 `(x,y)` 坐标 | encoder 权重是否共享 |
| 图像 | 视觉 embedding | `[...,T_vis,D_model]` | 占位符数应为 `T_vis` | KV 每 token 字节数 |
| 音频 | DSP 输出 | `[1,S,F_mel]` | 频谱帧数与特征宽度 | 原始音频时长，除非采样与分帧参数已知 |
| 音频 | 编码器特征 | `[1,S_enc,F_enc]` | 编码器输出序列与特征宽度 | audio token 数，除非适配器输出已核对 |
| 音频 | 音频 embedding | `[1,T_aud,D_model]` | 可插入 `T_aud` 个音频占位符 | 是否保留全部声学细节 |

> 表 10-2　多模态 shape 只能支撑相邻阶段的结论；跨过编码器、池化或适配器反推时需要额外配置。

## 10.4　约束解码：按文法屏蔽 token

自由采样可能生成缺少引号、括号不闭合或分隔符错误的结构化文本，下游解析器会因此拒绝输入。约束解码把文法状态加入采样过程，限制每一步可选择的 token 集合。

每个采样 step 开始前，约束解码器先计算允许 token 的位图。实现把其余 token 的 logit 设为类型可表示的最小值。采样器随后只能从当前约束允许的候选中选择。

`ConstrainedDecoder` 的类注释给出了调用顺序（`runtime/components/constrained_decoding/constrained_decoder.h:32-59`）：

```cpp
//   ConstrainedDecoder decoder(constraint, batch_size);
//   while (!done) {
//     TensorBuffer logits = Decode(...);
//     RETURN_IF_ERROR(decoder.MaskLogits(logits));            // (1)
//     TensorBuffer next_tokens = sampler.Sample(logits);      // (2)
//     RETURN_IF_ERROR(decoder.UpdateConstraintState(next_tokens));  // (3)
//   }
```

(1) `MaskLogits` 屏蔽当前状态不允许的 token。(2) 采样器从剩余候选中选出下一个 token。(3) `UpdateConstraintState` 用该 token 推进状态。构造函数为 batch 中的每条序列分别调用 `constraint_->Start()`，因此各序列维护独立状态。

### 10.4.1　状态推进发生在下一次采样之前

约束状态与模型 KV 状态不是同一个对象。KV cache 保存主干前向所需的 K/V；`Constraint::State` 保存文法解析进度。`Constraint` 接口只定义 `Start`、`IsEnded`、`ComputeNext` 和 `ComputeBitmap`，没有访问模型张量或 KV cache 的方法（`runtime/components/constrained_decoding/constraint.h:25-56`）。

第一个生成 token 的允许集合由起始状态计算。采样得到 \\(y_0\\) 后，运行时不会在同一步再次提交它。下一次 decode 开始前，才以 \\(y_0\\) 调用 `ComputeNext`，得到状态 \\(s_1\\)，然后根据 \\(s_1\\) 屏蔽产生 \\(y_1\\) 的 logits。对第 \\(t\\) 个输出 token，可写为

\\[
A_t=\operatorname{Bitmap}(s_t),\qquad
y_t=\operatorname{Sample}(z_t\mid A_t),\qquad
s_{t+1}=\operatorname{Next}(s_t,y_t).
\\]

这里 \\(z_t\\) 是模型 logits，\\(A_t\\) 是允许 token 集合。公式中的三个动作分属约束引擎、采样器和状态推进，顺序不能交换。若先采样再屏蔽，非法 token 已经选出；若漏掉 `Next`，每一步都会重复使用起始状态。

外部采样路径显式跳过第一个 decode step 的状态更新，因为这时传入执行器的是 prefill 最后一个 token，而不是约束生成出的 token。后续 step 才用上一轮生成 token 更新状态（`runtime/core/tasks.cc:319-350`）。内部采样路径用 `last_run_is_decode` 做同样区分（`runtime/executor/llm_litert_compiled_model_executor.cc:1146-1178`）。

`UpdateConstraintState` 按 batch 索引逐一提交 token。若某条序列到达约束终态，代码立即为该序列创建新的起始状态（`runtime/components/constrained_decoding/constrained_decoder.cc:40-53`）。这意味着约束对象可以继续处理下一段受约束输出；它不表示宿主函数已经执行，也不会自动清空 Conversation 历史。

batch 中的状态彼此独立，但 `UpdateConstraintState` 要求 token 数恰好等于 `batch_size`。`MaskLogits` 还要求 logits 形状为 `[batch_size,1,vocab_size]`（`runtime/components/constrained_decoding/constrained_decoder.cc:73-103`）。这两个条件把“每条序列一个状态”与模型输出 shape 对齐。不能把一个序列的位图广播到整个 batch。

v0.13.1 同时支持外部采样和内部采样。外部路径在 `Tasks::DecodeOneStep` 中取得 logits、调用 `MaskLogits`，再调用外部 sampler（`runtime/core/tasks.cc:319-364`）；内部路径把同一个 `ConstrainedDecoder` 放入 `ExecutorDecodeParams`（`runtime/core/tasks.cc:365-378`），执行器再更新状态并屏蔽 logits（`runtime/executor/llm_litert_compiled_model_executor.cc:1152-1208`）。

float32 路径的 `MaskLogits` 使用双重循环（`runtime/components/constrained_decoding/constrained_decoder.cc:73-103`）：

```cpp
for (int b = 0; b < batch_size; ++b) {
  auto& constraint_state = constraint_states_[b];
  ASSIGN_OR_RETURN(auto bitmap,
                   constraint_->ComputeBitmap(*constraint_state));   // (1)
  for (int i = 0; i < vocab_size; ++i) {
    if (!bitmap->Get(i)) {                                           // (2)
      logits.data()[b * vocab_size + i] =
          std::numeric_limits<float>::lowest();                      // (3)
    }
  }
}
```

(1) `ComputeBitmap` 为每条序列返回一张允许 token 位图；`Bitmap::Get` 的接口定义见 `runtime/components/constrained_decoding/bitmap.h:20-27`。(2)(3) 实现遍历整个模型词表，把位图为 0 的 float32 logit 写成 `std::numeric_limits<float>::lowest()`。float16 重载则写入 `tflite::half::min()`（`runtime/components/constrained_decoding/constrained_decoder.cc:106-135`）。函数还要求 logits 形状为 `[batch_size, 1, vocab_size]`，并检查模型词表不大于约束词表。该循环每个 decode step、每条序列执行 \\(O(V)\\) 次位图查询与条件写；本书基准模型的 \\(V=262{,}144\\)。这只是操作量，是否成为瓶颈需要分项 profile。

允许集合由 llguidance 计算，LiteRT-LM 通过 C 接口调用它。`LlgConstraint::Start` 调用 `llg_clone_constraint`，`ComputeNext` 调用 `llg_commit_token`，`ComputeBitmap` 调用 `llg_compute_mask`（`runtime/components/constrained_decoding/llg_constraint.cc:64-105`）。返回的掩码以 32 位字打包，C++ 再展开为布尔位图：

```cpp
mask_vector.push_back(sample_mask[i / 32] & (1 << (i % 32)));  // (1)
```

(1) 第 `i` 个 token 对应第 `i / 32` 个字中的第 `i % 32` 位。llguidance 负责维护约束状态并计算掩码，`ConstrainedDecoder` 负责把掩码应用到 logits。

### 10.4.2　工具声明生成的文法覆盖到哪里

解析器文法与采样文法用途不同。`AntlrFcParser.g4` 是模型生成完之后使用的通用 FC parser，它把任意合法 `ID` 解析为函数名或参数名。约束生成器则读取本次 Conversation 的工具声明，构造更窄的 llguidance 文法。二者不能混为同一层保证。

FC 约束生成器逐个读取工具的 `name`。它只为声明过的函数名生成 `"call:" "tool_name" ...` 分支（`runtime/components/constrained_decoding/llg_fc_tool_calls.cc:104-136`）。对参数，它读取 `parameters.required` 和 `parameters.properties`，为必需参数与可选参数分别建立规则（`runtime/components/constrained_decoding/llg_tool_call_utils.cc:30-68`）。因此，在这条 tools-derived 路径中，函数名和已声明参数名不只是任意 `ID`。

类型约束只覆盖代码显式转换的部分。字符串映射到 FC 字符串规则，`number/integer` 映射到 `NUMBER`，布尔、数组、对象与 null 也有对应规则。属性带 `enum` 时，生成器把字符串、数字和布尔枚举值写入候选规则（`runtime/components/constrained_decoding/llg_fc_tool_calls.cc:32-67`）。

这不是完整 JSON Schema 验证器。当前生成代码没有读取 `minimum`、`maximum`、字符串长度、正则 pattern 或跨字段关系。数组与对象映射到通用递归规则，没有继续展开 `items` 或嵌套 `properties`。这些结论来自该文件实际访问的 schema 字段；没有被读取的字段不能形成采样约束。

`constraint_mode` 还决定输出范围。`kFunctionCallsOnly` 只接受一个或多个函数调用；`kTextAndOr` 允许普通文本、函数调用或二者组合。若后者没有工具声明，生成器退化为禁止出现函数调用 fence 的文本规则（`runtime/components/constrained_decoding/llg_fc_tool_calls.cc:138-169`）。因此，“开启约束解码”并不总意味着“必须调用工具”。

| 条件 | tools-derived FC 约束可保证 | 仍需宿主验证 |
|---|---|---|
| 函数选择 | 名称来自已生成的工具分支 | 当前用户是否有权调用、工具是否可用 |
| 顶层参数名 | 来自声明的 `properties` | 别名、弃用字段、业务互斥关系 |
| 必需参数 | 文法包含 `required` 参数 | 非空、取值范围、资源是否存在 |
| 基本类型 | 字符串、数字、布尔、数组、对象、null 的语法形态 | 数值单位、精度、编码、领域语义 |
| 枚举 | 代码覆盖的字符串、数字和布尔枚举值 | 枚举对应的权限或设备能力 |
| 嵌套结构 | 通用数组与对象可以保持可解析 | `items`、嵌套 schema、跨字段约束 |
| 输出结束 | 能到达文法终态 | 外部调用成功、结果可信、是否需要重试 |

> 表 10-3　约束把部分工具声明编译进 token 允许集合，但不会完成授权、业务校验或外部执行。

<div class="aside-compare">

llama.cpp 在 b9873 中把视觉投影器作为独立 mmproj 文件，由 `--mmproj` 指定（`llama.cpp/common/arg.cpp:2315 @ b9873`）；LiteRT-LM 的 `.litertlm` 文件可包含视觉编码器与适配器模型段。前者允许单独替换投影器，但需要应用自行保证模型与投影器匹配；后者把相关模型段一并分发。约束解码方面，llama.cpp 的 `llama_grammar_accept` 位于 `llama.cpp/src/llama-grammar.cpp:1042 @ b9873`；LiteRT-LM 则通过 C 接口使用 llguidance。两种实现的外部依赖边界不同。

</div>

约束解码只能保证采样结果满足当前约束所编码的条件。通用 ANTLR FC parser 中的 `ID` 只规定词法形式；由工具声明生成的采样文法更窄，可以限制函数名、顶层参数名和部分类型。两者都不能证明当前用户有权限调用函数，也不验证未编码的 schema 条件、外部可用性和执行结果。应用仍须在解析后重新核对工具白名单、完整参数 schema 与授权策略，并处理执行错误。

附录 D 记录了一组 Python SDK 对照，模型为 Gemma 4 E4B，后端为 GPU。`enable_constrained_decoding` 开启与关闭时各生成 6 次，共 12 次。单参数工具在温度 0 下生成 4 次；三参数工具在温度 0 和 1.0 下各生成 4 次。12 个样本都得到结构可解析的调用，未观察到开关差异。每种开关设置仅 6 次，样本量不足以估计失败率，也不能支持“关闭约束同样可靠”的结论。实验没有覆盖多工具混淆、嵌套 JSON、参数语义和真实函数执行。

<figure>
{{#include figs/fig-10-2.svg}}
<figcaption>图 10-2　约束解码按当前文法状态屏蔽 token；图中只表示 FC 结构约束，不表示函数存在、参数语义正确或调用可执行。</figcaption>
</figure>

## 10.5　按数据边界定位多模态输入失败

多模态请求报错时，最后出现的 `prefill failed` 往往只说明任务没有完成，不能定位最初的错误。有效的排查顺序应沿数据变换方向进行：消息对象、模板标记、预处理张量、模态执行器、embedding 组合、主干 prefill。每跨一层，数据表示和责任方都会变化。

以一条包含一幅图和一段文本的消息为例。这个案例只用于演示排查过程，不代表真机故障统计。假设自定义模板在渲染结果中产生了两个图像标记。处理器扫描消息后，图像队列长度为 1。第一个标记消耗该对象，第二个标记发现队列为空，直接返回 `Provided less images than expected in the prompt.`（`runtime/conversation/model_data_processor/gemma4_data_processor.cc:402-423`）。视觉编码器和主干此时都没有运行。

这个错误首先要检查两份材料：原始 `Message` 中图像项的数量与顺序，以及本轮渲染 prompt 中模态标记的数量与顺序。增加 GPU 日志、替换后端或调整 decode 参数都不会改变结果。相反，若队列处理完后仍有图像，函数会返回 `Provided more images than expected in the prompt.`（`runtime/conversation/model_data_processor/gemma4_data_processor.cc:443-445`）。两条错误分别对应“标记多于对象”和“对象多于标记”。

修正模板后，请求可能在图像解码阶段失败。`StbImagePreprocessor` 调用 `stbi_load_from_memory`。返回空指针时，状态中包含 `stbi_failure_reason()`（`runtime/components/preprocessor/stb_image_preprocessor.cc:237-263`）。此处应保存原始字节长度、文件类型和解码错误，不应记录完整用户图片。若使用 `blob`，还要先确认错误是否来自 base64 解码；该错误发生在图像解码之前（`runtime/conversation/model_data_processor/data_utils.cc:42-48`）。

解码成功后，patchify 配置仍可能拒绝输入。patch 宽高不相等会立即返回错误；目标宽高同时取整为 0，或计算结果超过 patch 上限，也有各自的错误状态（`runtime/components/preprocessor/image_preprocessor_utils.cc:26-74`）。这些错误属于预处理配置，不表示模型不支持图片内容。

再下一层是视觉 signature 选择。多 signature 模型没有 `patch_num_shrink_factor`、signature 名称无法解析，或不存在足够长的入口，都会在 `GetVitSignatureIndex` 返回错误（`runtime/executor/vision_litert_compiled_model_executor.cc:138-195`）。最后一种错误会报告 `max_available_length`。应用可以据此降低 visual token budget 或换用匹配的模型产物，但不能静默选择较短入口，因为源码明确把这种情况视为会截断图像。

变分辨率路径还要求张量表同时含 `images` 和 `positions_xy`。缺少任一键时，视觉执行器在运行模型前返回 `InvalidArgumentError`（`runtime/executor/vision_litert_compiled_model_executor.cc:499-509`）。如果张量存在但类型不是 float32 或 int32，输入复制阶段返回 `Unsupported input tensor type`（`runtime/executor/vision_litert_compiled_model_executor.cc:552-593`）。

视觉编码成功后，执行管理器从 embedding 的倒数第二维读取 visual token 数，并据此插入 -1 占位符。标准 Conversation 路径因此由同一个张量决定行数和占位符数（`runtime/framework/resource_management/serial_execution_manager.cc:592-600`）。若应用绕过该路径自行构造 `ExecutorInputs`，则要自行维持二者相等。`EmbeddingLookupMultiModal` 在 embedding 数据不足以覆盖下一个特殊 token 时返回错误（`runtime/components/embedding_lookup/embedding_lookup_multi_modal.cc:145-158`）。

音频可以用同一方法分层。消息与模板数量错误发生在数据处理器；音频格式解码、PCM 分帧和 log-mel 计算发生在预处理器；频谱宽度与 mask 长度检查发生在音频执行器；最后才是 audio embedding 与 -2 占位符组合。`ProcessAndCombineContents` 若收到未经预处理的 `InputAudio`，会返回 `The audio is not a preprocessed tensor.`（`runtime/framework/resource_management/serial_execution_manager.cc:604-625`）。

### 10.5.1　一份可复现的最小诊断记录

多模态问题需要同时保存控制面和数据面信息。控制面包括模型文件哈希、后端、模型类型、模板配置、visual token budget 和是否启用音频模态。数据面至少包括各阶段 shape、元素类型、模态项与标记计数、选中的 signature 名称，以及完整的第一条错误状态。

原始图像、音频和完整 prompt 可能包含隐私。诊断包可以记录字节数、内容哈希、宽高、采样率和裁剪后的错误上下文，避免默认保存原始媒体。是否允许采集原始输入应由产品的隐私策略决定，不由 LiteRT-LM 推理接口代为决定。

若错误可以在预处理器单测中复现，就不必加载主干模型。若预处理张量已经稳定，再单独验证视觉或音频执行器。只有前两层均通过后，才需要运行完整 prefill。这样的最小化顺序减少变量，但不保证一次定位；后端编译与设备 buffer 错误仍可能只在完整执行路径出现。

<figure>
{{#include figs/fig-10-3.svg}}
<figcaption>图 10-3　多模态输入依次经过六个数据边界；每层应记录输入表示、输出 shape 与第一条错误。</figcaption>
</figure>

## 10.6　Tool Use：结构解析与应用执行

Tool Use（工具调用或函数调用）把模型生成的结构化文本转换为函数名和参数。函数由应用实现并执行，LiteRT-LM 负责格式化工具声明、检测输出并解析调用。

一次工具调用包含以下环节：

1. 声明工具。第 3 章 `Preface` 的 `tools` 字段携带工具声明。`Preface` 当前是以 `JsonPreface` 为成员的 `variant`；`JsonPreface` 的 `messages`、`tools` 和 `extra_context` 都使用 `nlohmann::ordered_json`（`runtime/conversation/io_types.h:26-59`）。`ordered_json` 保留插入顺序，格式化时不会按键名重新排序。
2. 格式化 prompt。模型数据处理器把工具描述写入提示词。FC（function call）格式中的键不加引号，字符串用 `<escape>` 包围；`FormatValueAsFc` 的接口与示例见 `runtime/components/tool_use/fc_tool_format_utils.h:26-59`。这种表示与 JSON 的 token 数差异取决于 tokenizer 和具体声明。本章没有测量，因此不比较两种表示的 token 数。
3. 生成调用。模型可输出 `call:tool_name{param_1:7,param_2:<escape>foo<escape>}`。约束解码只能保证该输出满足所用约束；函数名是否在工具白名单内、参数是否符合应用 schema，仍需解析后检查。
4. 解析与执行。FC 路径的 Rust lexer/parser 由 ANTLR 生成。C++ 通过 CXX bridge 接收结果（`runtime/components/tool_use/rust/fc_parser.rs:156-173`、`runtime/components/tool_use/fc_parser_utils.cc:33-47`）。FC parser grammar 包含六条规则（`runtime/components/tool_use/antlr/AntlrFcParser.g4:23-40`）：

```antlr
start : functionCall EOF;
functionCall: CALL COLON ID object?;            // (1)
object : OPEN_BRACE ( pair (COMMA pair)* )? CLOSE_BRACE;
pair : ID COLON value;
value
    : ESCAPED_STRING
    | NUMBER
    | BOOLEAN
    | NULL_LITERAL
    | object
    | array
    ;                                             // (2)
array: OPEN_BRACKET ( value (COMMA value)* )? CLOSE_BRACKET;
```

(1) `functionCall` 要求 `call`、冒号、`ID`，参数对象可选。(2) `value` 递归引用 `object` 和 `array`，因此能表示嵌套参数。`ParseFcExpression` 把示例还原为 `{"name":"tool_name","arguments":{"param_1":7,"param_2":"foo"}}`；头文件给出了相同的输入输出（`runtime/components/tool_use/fc_parser_utils.h:24-42`）。FC 与 Python 风格调用使用 ANTLR parser；JSON 路径在 v0.13.1 中调用 `serde_json::from_str`（`runtime/components/tool_use/rust/json_parser.rs:54-60`）。

解析成功后，应用核对函数与参数，执行函数，再把结果作为消息发回模型。执行函数是应用的责任，不是语法约束或 parser 的责任（`docs/api/cpp/tool-use.md:22-38`）。

| 环节 | 职责 | 代码锚点 |
|---|---|---|
| 1. 声明工具 | `Preface.tools` 携带可用工具描述 | `runtime/conversation/io_types.h:30-40` |
| 2. 格式化 prompt | JSON 值转换为目标模型使用的 FC 格式 | `runtime/components/tool_use/fc_tool_format_utils.h:26-59` |
| 3. 约束生成 | 按当前约束状态屏蔽 token | `runtime/components/constrained_decoding/constrained_decoder.cc:73-103` |
| 4. 解析与执行 | parser 返回函数名和参数；应用校验、执行并回填结果 | `runtime/components/tool_use/fc_parser_utils.cc:33-47`、`docs/api/cpp/tool-use.md:22-38` |

> 表 10-4　Tool Use 各环节的职责边界；约束生成、结构解析与应用执行是三个不同阶段。

### 10.6.1　parser 输出仍位于信任边界之外

模型输出不是函数调用对象。`FunctionGemmaDataProcessor::ToMessageImpl` 先取得完整响应文本，只有 `Preface` 中存在工具时才调用 `ParseTextAndToolCalls`。解析成功后，它分别把普通文本与 `tool_calls` 放进 assistant message（`runtime/conversation/model_data_processor/function_gemma_data_processor.cc:305-330`）。

`ParseTextAndToolCalls` 先用模型配置的起止 fence 切分文本与工具代码块，再根据 `syntax_type` 选择 Python、JSON 或 FC parser。默认策略在工具块解析失败时返回错误；若关闭 `return_error_on_parse_failure`，函数把原代码块作为文本返回，并附加 `error` 字段（`runtime/components/tool_use/parser_utils.cc:88-166`）。因此，应用必须区分三种结果：没有工具块、工具块解析失败、解析成功并得到 `tool_calls`。

解析成功只表示字符串满足 parser 接受的语法。parser 不访问用户身份、设备状态、网络权限或外部服务。FC Rust listener 把 `ID` 保存为 `name`，把可选对象保存为 `arguments`；没有参数对象时填入空对象（`runtime/components/tool_use/rust/fc_parser.rs:130-173`）。这些字段仍来自模型输出，不能直接作为函数指针、文件路径或 SQL 片段使用。

工具声明本身也不等于当前执行能力。声明通常在 Conversation 创建时进入 `Preface.tools`，而权限、登录状态和设备连接可能在后续轮次变化。宿主应在每次执行前重新查询授权和可用性。若工具已不可用，应返回受控错误结果，而不是因为它曾出现在 prompt 中就继续执行。

### 10.6.2　端到端案例：受控地修改设备模式

下面的 `set_device_mode` 是应用侧集成示例，不是 LiteRT-LM 内置工具。它修改外部设备状态，因而适合展示结构约束与执行授权之间的距离。工具声明可以写成：

```json
{
  "name": "set_device_mode",
  "description": "Set an enrolled device to an allowed operating mode.",
  "parameters": {
    "type": "object",
    "properties": {
      "device_id": {"type": "string"},
      "mode": {"type": "string", "enum": ["normal", "eco"]}
    },
    "required": ["device_id", "mode"]
  }
}
```

FunctionGemma 的 `FormatTools` 会把工具声明转换为模型使用的 FC 格式。启用约束后，`CreateConstraint` 把工具 JSON 与 fence、引号等格式选项传递给 Gemma constraint provider（`runtime/conversation/model_data_processor/function_gemma_data_processor.cc:333-397`）。模型可能生成：

```text
call:set_device_mode{
  device_id:<escape>lab-7<escape>,
  mode:<escape>eco<escape>
}
```

tools-derived 文法可以把函数名限制为 `set_device_mode`，并把 `mode` 限制在声明的枚举中。parser 随后产生：

```json
{
  "type": "function",
  "function": {
    "name": "set_device_mode",
    "arguments": {"device_id": "lab-7", "mode": "eco"}
  }
}
```

此时仍不能调用设备 SDK。宿主至少要确认 `lab-7` 属于当前账户、当前用户具有写权限、设备在线且支持 `eco`。这些条件不在工具 schema 中，也不在语言模型上下文中形成可靠事实。

应用可以先把解析结果转换为内部命令，再调用设备适配器。调用方只把 `tool_call["function"]` 传入下面的 `call` 参数：

```cpp
absl::StatusOr<ordered_json> ExecuteToolCall(
    const ordered_json& call, const RequestContext& ctx) {
  if (call.at("name") != "set_device_mode") {
    return absl::PermissionDeniedError("tool is not allowed");
  }
  ASSIGN_OR_RETURN(auto args, ValidateSetDeviceMode(call.at("arguments")));
  if (!ctx.policy.CanControl(args.device_id)) {
    return absl::PermissionDeniedError("device is not authorized");
  }
  ASSIGN_OR_RETURN(auto capability,
                   ctx.devices.GetCapability(args.device_id));
  if (!capability.modes.contains(args.mode)) {
    return absl::InvalidArgumentError("mode is not supported");
  }
  ASSIGN_OR_RETURN(auto receipt,
                   ctx.devices.SetMode(args.device_id, args.mode,
                                       ctx.idempotency_key));
  return ordered_json{{"tool_name", "set_device_mode"},
                      {"status", "accepted"},
                      {"receipt_id", receipt.id}};
}
```

这段代码刻意不让模型提供权限主体和幂等键。两者来自宿主的 `RequestContext`。函数名采用显式比较，不用模型字符串动态查找任意符号。`ValidateSetDeviceMode` 应拒绝未知字段、错误类型、空标识符和超长字符串；外部适配器还应设置超时，并把取消与重试策略固定在应用层。

| 边界 | 输入是否可信 | 宿主需要执行的检查 | 失败后的处理 |
|---|---|---|---|
| 工具声明 → prompt | 应用配置可信，模型解释不可信 | 声明版本与当前能力一致 | 更新声明或拒绝旧会话 |
| 模型文本 → parser | 不可信 | fence、语法、输出大小 | 返回解析错误，不执行 |
| parser JSON → 命令对象 | 仍不可信 | allowlist、完整 schema、长度 | 返回参数错误 |
| 命令对象 → 权限系统 | 不可信请求 | 用户、资源、操作级授权 | 返回拒绝，记录审计事件 |
| 权限通过 → 外部适配器 | 已授权但外部状态可变 | 能力、幂等、超时、取消 | 返回可分类执行错误 |
| 外部结果 → tool message | 外部数据不默认可信 | 裁剪、转义、脱敏、大小限制 | 返回最小错误对象 |

> 表 10-5　parser 只跨过语法边界；真正的信任边界位于宿主的命令校验与授权入口。

执行成功后，应用把最小结果作为 `role: "tool"` 的消息发回 Conversation。FunctionGemma 的格式化代码可从 `tool_name` 或 `name` 取工具名，再把其余字段转换为 FC 对象（`runtime/conversation/model_data_processor/function_gemma_data_processor.cc:91-127`）。返回数据同样会进入模型上下文，因此不应包含访问令牌、内部异常栈或不必要的设备属性。

外部调用失败时，也应回填结构化错误，而不是伪造成功结果。例如 `{"tool_name":"set_device_mode","error":{"code":"DEVICE_OFFLINE"}}` 足以让模型解释当前状态。是否允许模型重试由宿主决定。对有副作用的操作，未确认上次请求是否生效前，不能仅凭模型再次生成同一调用就重放。

工具循环还需要两个上限：每个用户请求允许的工具轮数，以及单次工具结果进入上下文的最大尺寸。LiteRT-LM 负责维护对话和解析工具块，但源码中的 parser 不提供业务级循环上限。应用必须在调用 Conversation 的外层维护计数，并在达到上限时终止或转人工处理。

<figure>
{{#include figs/fig-10-4.svg}}
<figcaption>图 10-4　模型文本和 parser JSON 均属于不可信输入；宿主校验、授权与受控适配器共同构成执行边界。</figcaption>
</figure>

> 版本注记
> 工具调用格式与 parser 会随版本变化，本节描述的是 v0.13.1。上游 `LiteRT-LM#2418` 记录过特定模型的嵌套 JSON 参数解析问题。[^ch10-issue-2418] 排查时先保存模型原始输出。确认生成文本符合文法后，再检查 parser 是否覆盖该格式，并核对应用是否因参数校验而拒绝调用。

## 小结

图像和音频从 `Message` 进入模型数据处理器，依次变为预处理张量、模态 embedding 和主干输入。执行管理器插入模态占位符，prefill 查找器再写入对应向量。排错时应沿相同顺序检查对象数量、shape、signature 与 embedding 行数。

视觉输入长度还会增加 prefill 序列与有效 KV 数据量。计算时必须使用 \\(H_{kv}\times D\\)，不能用 `model_dimension` 代替。多图在 token 维拼接，不等于 batch；音频分块后的有效 token 数要按每块结果累加。

约束解码按状态计算允许 token 位图，并在采样前屏蔽其他候选。tools-derived 文法可以编码函数名、顶层参数和部分类型，但不覆盖完整 schema。Tool Use 还需要 parser 把文本还原为函数名和参数。parser 结果仍是不可信输入；应用随后完成白名单、完整 schema、权限、幂等与实际执行。

---

## 练习与自查

1. patchify 计算。一张 768 × 512 的图，patch 16 × 16、`max_num_patches = 256`、`pooling_kernel_size = 1`。根据源码算出目标尺寸与 patch 数。若 `patch_num_shrink_factor = 4`，再算无输出 mask 路径的 visual token 数。
2. 占位符替换。视觉和音频的 `kSpecialToken` 分别是什么值？执行管理器如何保证占位符数与 embedding 行数一致？
3. KV 数据量。沿用附录 D 的混合 KV 形状，计算 280 个 visual token 对应的活动 KV 数据量。说明为什么不能代入 `model_dimension = 2560`。
4. 音频张量。`features[1, 204, 1536]` 的三个维度分别能支持哪些结论？为什么不能把 204 直接称为原始频谱帧数？
5. 约束边界。给定本章的 FC 文法，列出它能保证的结构条件，以及应用仍需验证的函数、参数、权限和执行条件。
6. 状态时序。画出连续生成三个 token 时 `MaskLogits`、`Sample` 与 `UpdateConstraintState` 的调用顺序。说明首个 decode step 为什么不提交 prefill 的最后一个 token。
7. 故障定位。一条消息含两幅图，而渲染 prompt 只有一个图像标记。写出最先返回错误的模块，并说明为什么无需运行视觉执行器。
8. 执行边界。为一个具有文件写入副作用的工具设计宿主检查项。至少覆盖路径范围、授权、幂等、超时和结果回填。

[^ch10-issue-2418]: schwartz1375，*Gemma 4 tool call parser fails on nested JSON string parameters (`<|"|>` tokens)*，LiteRT-LM issue #2418，2026-05-31，<https://github.com/google-ai-edge/LiteRT-LM/issues/2418>（访问 2026-07-18）。
