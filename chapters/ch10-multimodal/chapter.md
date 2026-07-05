# 第 10 章 多模态输入、约束解码与工具调用

> 本章走出纯文本，考察两个方向的扩展。输入端：模型如何把图像、音频编码为可处理的表示，接入以文本为设计前提的推理流水线。输出端：如何在采样层约束生成，使输出恒为合法 JSON 或可执行的函数调用。

前三部把纯文本生成讲透了：一段文本进入模型，一串 token 逐步生成。真实应用要的往往不止对话。它要处理图像和音频输入，要模型给出的不是自由文本而是一个可直接执行的函数调用。本章讲这两件事。二者方向相反，一个作用于输入、一个作用于输出，但都建立在前几章的推理流水线之上，核心的 prefill 与 decode 一字未改。

## 图像如何进入模型

先看输入端。Transformer 主干处理的是 token 序列（第 3 章），它不接受像素。图像怎么接进去？

思路是把图像也编码成 embedding，注入到 token 序列中。第 3 章讲过，文本先经词嵌入（embedding）查表才进入主干；图像走同一条路，区别只在于「如何得到 embedding」。整条链路分三步：切块、编码、注入。

第一步，切块。一张图先被切成许多小块（patch），这一步叫 patchify。切之前要先决定缩放到多大——`GetAspectRatioPreservingSize`（`runtime/components/preprocessor/image_preprocessor_utils.cc:26 @ v0.13.1`）算的就是这个目标尺寸。这段算法把「patch 数不超过上限」这条约束落成了几行浮点运算：

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

(1) 先算允许的总像素上限：`max_num_patches` 个 patch，每个 patch 占 `patch_width × patch_height` 像素。(2) 拿这个上限除以原图像素数、开方，得到一个各向同性的缩放系数 `factor`。这个系数乘在宽高上，缩放后的总面积恰好压到上限，同时保持长宽比不变。(3)(4) 处理网格对齐：缩放后的宽高必须是 `side_mult` 的整数倍，`side_mult` 等于池化核尺寸乘 patch 边长。代码先除以 `side_mult`、向下取整、再乘回去，把宽高对齐到网格。对齐的原因在于下游要按 `patch_width` 均匀切块、再按 `pooling_kernel_size` 做池化，宽高不是这个乘积的整数倍就无法整除切分。函数开头有一句前置检查：`patch_width != patch_height` 直接返回错误，patch 必须为正方形。向下取整可能把某一边压成 0（极端细长的图），代码对此单列了一个分支：把为 0 的那一边设成一个 `side_mult`，另一边按原始长宽比放大且不超过 `max_side_length`。

这里可以当面把 visual token 数算清。给定 `max_num_patches`、`patch_width`、原图尺寸，$\text{target\_height} \times \text{target\_width} / (\text{patch\_width} \times \text{patch\_height})$ 就是这张图切出的 patch 数，也就是它在序列里占用的 visual token 数——序列里要为此预留的 `kSpecialToken` 占位符槽位数量。这个数字并非无关紧要，本节末尾会把它连到 prefill 计算量与 KV cache 占用上，量化「看一张图」的真实代价。

上面这段只算目标尺寸，真正的重采样发生在另一处。`MaybeResizeImageWithSameAspectRatio`（`runtime/components/preprocessor/stb_image_preprocessor.cc:56 @ v0.13.1`）拿到目标尺寸后，先判断是否需要缩放，再逐张调重采样：

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

(1) 短路分支：若目标尺寸与原图相等，直接 `move` 走原数据、不做任何重采样。已经符合 patch 约束的图（例如上游已按网格裁好）在此零开销通过。(2) 逐张调 `stbir_resize`（stb_image_resize2 提供），滤波器选 `STBIR_FILTER_CATMULLROM`（Catmull-Rom 三次插值），色彩空间按 `STBIR_TYPE_UINT8_SRGB` 处理，边缘按 `STBIR_EDGE_CLAMP` 钳位。Catmull-Rom 每个输出像素要采样源图一个邻域并做加权，是纯 CPU 上的密集浮点运算。缩放本身在图像预处理里往往是耗时占比最大的一步：一张 4K 图缩到几百像素见方，输出像素虽少，但每个输出像素的插值核要覆盖较大的源邻域，代价随源图分辨率上升。这也是短路分支值得单列的原因——能跳过就跳过一整趟重采样。重采样在 CPU 上串行完成，与后续视觉编码器可能跑在 GPU/NPU 上形成一段串行前缀，端到端时延里这段无法与主干计算重叠。

重采样之后是归一化与切块。第二步真正把像素搬成 patch 序列的是 `PatchifyImage`（`stb_image_preprocessor.cc:120 @ v0.13.1`）。这一步不做数学变换，只做数据布局重排：把 HWC 连续排列的像素重新打包成 `[batch, num_patches, patch_dim]`，同时为每个 patch 生成它在网格里的 (w, h) 坐标。核心是一段六重嵌套循环：

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

(1) 外三层遍历每个 patch 的网格位置，顺带把它的 (w, h) 坐标写进 `positions_xy` 张量。这份坐标随后要连同像素一起喂给编码器，用于恢复 patch 的空间位置。(2) 内三层遍历 patch 内的每个像素通道，`src_idx` 是标准的 HWC 行主序索引 `((b·H+src_h)·W+src_w)·C+c`。(3) `dest_idx` 是 patch 主序索引：同一 patch 的 `patch_dim = patch_width·patch_height·channels` 个值在目的缓冲里连续排布。这段循环的访存模式值得留意。目的端 `patches_ptr` 严格顺序写入，cache 友好；源端 `image_data` 的读取则在行内连续（`pw` 与 `c` 变化时），但每换一个 patch 行（`ph` 递增）就跳一整个图像宽度 `W·C` 个元素：源读取有周期性的大 stride 跳跃，patch 越大、图越宽，跳跃越频繁，L1/L2 命中率越低。这段是逐元素标量拷贝，未做 SIMD 向量化；实际负载不算大（一张图的像素总量），但它落在缩放之后、编码之前的串行 CPU 段里，和重采样一样属于「喂进编码器之前」必须付的前处理成本。`positions_xy` 之所以显式外传而非由编码器内部推算，是因为下一步的变分辨率编码要按它对齐可变数量的 patch。

第二步的后半段，切好的图交给视觉执行器编码成 embedding。执行器的 `Encode`（声明 `runtime/executor/vision_litert_compiled_model_executor.h:57`，实现 `vision_litert_compiled_model_executor.cc:454 @ v0.13.1`）返回一个装着 embedding 的 `ExecutorVisionData`，它的实现是两级串联：

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

(1) 把预处理好的图像张量写进编码器的输入 buffer。(2) 跑视觉编码器（vision encoder）——一个独立的 LiteRT 编译模型，产出编码器自身特征空间里的向量。(3) 编码器的输出直接喂给视觉适配器（vision adapter）再跑一次：适配器把编码器特征投影到 LLM 的 embedding 空间，维度对齐到 `model_dimension`，此后 Transformer 主干可与文本 embedding 一并处理。两个模型都是编译好的 `CompiledModel`，视觉编码器提取视觉特征，视觉适配器将其投影到 LLM 嵌入空间。(4) 返回时 `per_layer_embeddings` 传 `std::nullopt`，该路径仅输出常规 embedding，不产出 per-layer embeddings（逐层嵌入，供特定模型架构使用，此处不需要）。中间被 `// ...` 省去的代码处理一个后端相关的约束：WebGPU 与 Metal 内存下输出 buffer 不可复用，第二次调 `Encode` 会触发 lock 失败，需每次新建（源码注释挂着内部编号 b/457483190）。

第三步，注入——整件事最巧的地方在"预留位置"。序列里先放一串占位的特殊 token，`kSpecialToken` 值为 -1（`runtime/executor/llm_executor_io_types.h:220 @ v0.13.1`）。头文件里给了直观的例子（`:202`）：

```cpp
// token_ids = [2, kSpecialToken, kSpecialToken, kSpecialToken, 106, 77, ...]
// (contains 3 vision tokens)
// Then, the vision embeddings should have shape [3, model_dimension]:
// [[0.1, ...],  // Embedding for the 1st kVisionSpecialToken
//  [0.5, ...],  // Embedding for the 2nd kVisionSpecialToken
//  [0.9, ...]]  // Embedding for the 3rd kVisionSpecialToken
```

三个 `kSpecialToken` 就是给视觉 embedding 预留的三个占位符槽位，`ExecutorVisionData` 里 embedding 的行数必须严格等于序列中 `kSpecialToken` 的个数——上一步 patchify 算出的 visual token 数，在这里必须对得上。prefill 之前，`FillVisionEmbeddings`（`runtime/executor/llm_executor_base.h:178 @ v0.13.1`）把这些 embedding 按行填进对应槽位。它的签名带一个 `image_index` 参数：一次对话可含多张图，每张图占一段连续的 `kSpecialToken`，`image_index` 指定这批 embedding 覆盖哪一张。填完，序列里一部分槽位是文本 embedding、一部分是视觉 embedding，对主干而言都是同一空间的向量，统一处理。

<figure>
{{#include figs/fig-10-1.svg}}
<figcaption>图 10-1　图像输入的三步链路：图片先 patchify 切块、经视觉编码器与适配器编码为 embedding，再填入 token 序列中由特殊 token（kSpecialToken）预留的占位符槽位。对主干而言，视觉 embedding 与文本 embedding 同属一个嵌入空间。</figcaption>
</figure>

### 变分辨率的视觉编码

上面那个 `Encode` 重载接受单个图像张量，隐含假设编码器只有一种固定的输入尺寸。实际的视觉编码器往往内置多个 signature，按输入 patch 数在运行时择一派发。第二个 `Encode` 重载（`vision_litert_compiled_model_executor.cc:499 @ v0.13.1`）就是走这条路：它接受 `input_maps`（含 `images` 与 `positions_xy` 两个键），从 `images` 张量的形状里读出实际 patch 数，再选签名：

```cpp
const auto& images_dimensions = images_tensor_type.Layout().Dimensions();
const int num_patches_from_input = images_dimensions[1];               // (1)
ASSIGN_OR_RETURN(auto encoder_signature_index,
                 GetVitSignatureIndex(vision_encoder_->GetModel(),
                                      vision_executor_properties_,
                                      num_patches_from_input));         // (2)
```

(1) patch 数直接取自输入张量的第 1 维——就是 patchify 那步 `num_patches_h × num_patches_w` 的结果。(2) 把它交给 `GetVitSignatureIndex` 选签名。选择逻辑（`:138 @ v0.13.1`）是「够用的最小者」：

```cpp
const int max_num_tokens =
    num_patches / vision_executor_properties.patch_num_shrink_factor.value();  // (1)
for (int i = 0; i < model.GetNumSignatures(); ++i) {
  // ... 从签名名末尾解析出该签名支持的 length ...
  if (current_length >= max_num_tokens && current_length < best_length) {  // (2)
    best_length = current_length;
    best_signature_index = i;
  }
}
```

(1) 先把 patch 数按 `patch_num_shrink_factor` 折算成编码后 token 数：池化会缩减序列长度，编码器 signature 是按输出 token 数命名的。(2) 遍历所有以 `kVisionLengthPrefix` 打头的签名，从名字末尾解析出各自支持的长度，取「大于等于所需长度、且最短」的那个。若没有任何签名够长，返回错误并提示把 `max_num_tokens` 降到 `max_available_length`——宁可报错也不静默截断图像。

这套多签名设计是对「静态 shape」与「动态 shape」的一种折中。纯动态 shape（一个签名吃任意长度）在移动端后端上代价高：GPU/NPU 常要按具体形状预编译 kernel，运行时才定形状会触发重编译或走慢速回退路径。纯单一静态 shape（把所有图 pad 到最大长度）则浪费算力：一张小图也得按最大 patch 数跑一遍编码器。多签名相当于预编译好若干档固定长度，运行时按实际 patch 数向上取整到最近一档。代价是模型文件里要多带几套编码器权重的 signature（同一权重、不同输入形状的多个入口），换来的是每张图只按贴近实际的档位付算力。这也解释了为什么 `positions_xy` 要在 patchify 阶段显式生成并外传：可变数量的 patch 需要显式的位置信息，编码器无法从固定网格假设里反推。

单张量重载则是这套机制的退化情形：当 `GetNumSignatures() == 1` 时 `GetVitSignatureIndex` 直接返回 0（`:142 @ v0.13.1`），跳过全部选择逻辑。固定分辨率模型走这条快路径，变分辨率模型走上面那条。

这套多签名机制在本书基准模型里就是现役的。实剖 Gemma 4 E4B 的模型文件（附录 D），视觉编码器段里恰好是三档签名 `vision_70`、`vision_140`、`vision_280`，配套的适配器段也是对应的三档 `vision_adapter_70/140/280`；编码器输入 `images` 的形状是 `[1, 1260, 768]`，其中 768 = 16 × 16 × 3，即 patch 是 16 像素见方的 RGB 块，1260 是输入槽位的 patch 容量上限。签名名末尾的 70/140/280 正是上文「按输出 token 数命名」的那串数字。

### 视觉 token 折算成的 prefill 与 KV cache 开销

回到本节开头留下的账。patchify 公式算出的 visual token 数，不只决定序列里预留多少槽位，它直接放大 prefill 的计算量与 KV cache 的占用。视觉 token 一旦填进序列，对 prefill 而言与文本 token 无差别（第 4 章）：每个 token 都要过一遍完整的 Transformer 前向，都要在每一层写入一份 K/V 进 cache。

把数字代进去。设某视觉编码器 `patch_width = patch_height = 14`，`max_num_patches = 256`，`pooling_kernel_size = 1`（不额外池化）。一张接近上限的图折算成约 256 个 visual token。这 256 个 token 等价于 256 个文本 token 的 prefill 成本：prefill 的计算量随 token 数近似线性（注意力那部分随序列长度平方增长，但在几百 token 尺度下线性项主导），KV cache 占用则严格线性：`num_layers × 2 × model_dimension × num_tokens × sizeof(dtype)`。以一个 26 层、`model_dimension = 2048`、fp16 KV 的配置粗算，256 个 token 的 KV cache 约为 $26 \times 2 \times 2048 \times 256 \times 2 \approx 54\,\text{MiB}$（按 $2^{20}$ 计）。也就是说，「看一张图」在上下文里的占位，相当于一次几百 token 的额外 prefill，外加数十 MiB 的 KV cache 常驻。多图对话会成倍放大这两项：三张图就是约 768 个 visual token、约 160 MiB KV cache。这解释了 `max_num_patches` 为何是端侧多模态的关键预算旋钮——它是「图像细节」与「prefill 时延／显存」之间的直接兑换比。〔本节字面值为按公式的量级估算，真机端到端时延与显存占用待附录 D 基准回填〕

音频走同一条路，只是特殊 token 换成 -2（`ExecutorAudioData::kSpecialToken`，`llm_executor_io_types.h:281 @ v0.13.1`；对齐约定在 `:263` 的注释里，与视觉逐字对应）。三个模态各用一个 `kSpecialToken` 值（文本无、视觉 -1、音频 -2）区分各自的占位符槽位，填充时各填各的。这就是多模态的统一之处：各模态各有编码器把输入编码成同一嵌入空间的向量，一旦成为 embedding，后续的 prefill、decode（第 4、5 章）无需任何改动——它们本就工作在 embedding 上，不关心这些向量原本来自文本、图像还是音频。第 2 章那套分层架构因此得以保持干净：多模态是在输入端多接一个编码器，而非改动整条流水线。

### 音频链路：DSP 前端与分块编码

音频链路比视觉多两层结构。第一层在编码器之前：喂给音频编码器的不是波形，而是 log-mel 频谱图，这是语音处理的经典前端。预处理器（`runtime/components/preprocessor/audio_preprocessor_miniaudio.cc @ v0.13.1`）把波形切帧后逐帧做实数 FFT（`kiss_fftr`，`:265`，用的是 kissfft 库），取平方幅度谱，再交给 `MelFilterbank` 加权。这个类的注释一句话说清了它做的变换：把平方幅度谱的一个切片转换为三角 mel 加权的线性幅度滤波器组（`runtime/components/preprocessor/mel_filterbank.h:25 @ v0.13.1`），初始化参数就是教科书上那几个：FFT bin 数、采样率、mel 通道数、频率上下限（`:38`）。滤波器组输出取对数前还加一个下限保护（`audio_preprocessor_miniaudio.cc:295` 一带，对数值加 floor 防止 log(0)），得到最终的 log-mel 频谱。到这里音频已经变成一个 `[帧数, mel 通道数]` 的浮点矩阵，后续才轮到神经网络。

第二层在编码器内部：频谱图不是一次性喂进编码器，而是分块编码。`Encode`（`runtime/executor/audio_litert_compiled_model_executor.cc:941 @ v0.13.1`）先校验频谱与掩码的形状一致性，然后进入分块循环（`:987`）：

```cpp
  // Chunk the spectrogram into smaller pieces and encode them one by one.
  int total_valid_tokens = 0;
  int pos = 0;
  while (pos < input_sequence_length) {                        // (1)
    int end = std::min(pos + sequence_length_, input_sequence_length);
```

(1) 每轮取至多 `sequence_length_` 帧（编码器 signature 的固定输入长度），编码器对每块 Run 一次，输出按 `encoder_shrinking_factor_` 缩减后的 token 数拼接进结果，`total_valid_tokens` 累计有效 token。这与第 4 章 prefill 的分块是同一个约束的两次出现：编译好的模型入口是定长的，任意长度的输入只能切块喂。一段几十秒的音频有几千帧频谱，分块让编码器的输入 buffer 尺寸有界，代价同样是块间串行。

两层加起来，音频的成本结构与视觉不同：视觉的预处理大头在重采样（纯 CPU 浮点），音频的预处理是 FFT 加滤波器组（同样纯 CPU，但随音频时长线性增长），编码阶段则多了分块循环的串行段。落到序列里之后二者归一：音频 token 同样按第 4 章的规则参与 prefill、按第 6 章的规则占 KV cache，上一节的预算算式对它同样适用。

实剖同样能看到这条链路的两级：基准模型文件里音频编码器与 `audio_adapter` 各占一段，适配器输入 `features` 形状 `[1, 204, 1536]`，即编码器输出的 1536 维特征、每块至多 204 帧，再由适配器投影到主干的 2560 维（附录 D）。

## 让输出守规矩：约束解码

现在转到输出端。聊天场景下，模型爱怎么说怎么说。但如果你要它输出一段 JSON、或者一个格式严格的函数调用，"爱怎么说"就成了问题——它可能漏个引号、多个逗号，让下游解析崩掉。

约束解码的思路是：**在每一步采样前，把不合法的 token 全部掐掉。** 回忆第 5 章的 decode 循环，采样是从 logits（每个 token 的分数）里挑一个。约束解码在采样前插一手，把当前语法不允许的 token 的分数全设成负无穷——它们的概率就成了零，永远不会被选中。

头文件把这套流程写在类注释里（`ConstrainedDecoder`，`runtime/components/constrained_decoding/constrained_decoder.h:48 @ v0.13.1`），是逐字从源码抄下来的循环：

```cpp
//   ConstrainedDecoder decoder(constraint, batch_size);
//   while (!done) {
//     TensorBuffer logits = Decode(...);
//     RETURN_IF_ERROR(decoder.MaskLogits(logits));            // (1)
//     TensorBuffer next_tokens = sampler.Sample(logits);      // (2)
//     RETURN_IF_ERROR(decoder.UpdateConstraintState(next_tokens));  // (3)
//   }
```

(1) `MaskLogits` 把非法 token 打成 -inf。(2) 采样此时只可能选到合法 token。(3) `UpdateConstraintState` 根据选中的 token 推进语法状态，决定下一步哪些合法。三步咬合成一个循环，插在第 5 章 decode 循环的采样前后。注意这里的 `decoder` 是有状态的——它在构造时给 batch 里每条序列各调一次 `constraint_->Start()` 存下初始语法状态，后面每步都在这份状态上推进。

`MaskLogits` 的实现（`runtime/components/constrained_decoding/constrained_decoder.cc:73 @ v0.13.1`）核心就是一个双重循环：

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

(1) 对每条序列，问当前语法状态："此刻哪些 token 合法？"答案是一张按词表大小铺开的位图（`Bitmap`，`runtime/components/constrained_decoding/bitmap.h:21 @ v0.13.1`）。(2)(3) 遍历整个词表，位图里为 0 的位置（不合法的 token）把它的 logit 直接压到 `float` 的最小值。这就是"负无穷"的工程写法：不是数学上的 $-\infty$，而是 `std::numeric_limits<float>::lowest()`。经过 softmax 后概率无限趋近 0，采样再也选不到它。这段前面还有一串 `RET_CHECK`：要求 logits 形状是 `[batch_size, 1, vocab_size]`（序列长度必须是 1，一次只掩一步），并且模型词表不能大于约束词表——约束词表可以更大，多出来的位当未用 token 处理。代价也在这段里明摆着：每步要对整个词表跑一遍，`vocab_size` 量级在几万到十几万，这是约束解码相比裸采样多付的每步开销。

"哪些合法"最终由谁算？底层的语法引擎是 llguidance，一个 Rust 库，通过 C bridge 接进来（见仓库的 `PATCH.llguidance*` 与 `docs/api/cpp/constrained-decoding.md`）。C++ 侧的 `LlgConstraint`（`runtime/components/constrained_decoding/llg_constraint.cc @ v0.13.1`）只是把三个动作转成三次 FFI 调用：`Start` 调 `llg_clone_constraint` 克隆一份初始状态，`ComputeNext` 调 `llg_commit_token` 推进，`ComputeBitmap` 调 `llg_compute_mask` 取掩码。取回的掩码是 llguidance 打包的 32 位字数组，C++ 侧再解包成布尔位图：

```cpp
mask_vector.push_back(sample_mask[i / 32] & (1 << (i % 32)));  // (1)
```

(1) 第 `i` 个 token 的允许位，藏在第 `i / 32` 个 32 位字的第 `i % 32` 位——用位运算把它取出来。语法怎么写、GBNF 或 JSON Schema 怎么编译成状态机，都在 Rust 那边；C++ 这层只管"每步取一张位图、按位掩 logits"。分工清楚：约束的表达力归 llguidance，掩码的执行归这三十行 C++。

<div class="aside-compare">

本章两大主题在 llama.cpp 里各有对照。多模态：视觉投影器是独立的 mmproj 文件，运行时用 `--mmproj` 指定（`llama.cpp/common/arg.cpp:2315 @ b9873`），与 `.litertlm` 把视觉编码器、适配器全打包进主文件形成两种分发哲学——组件独立分发便于混搭升级，单文件打包免除版本错配。约束解码：llama.cpp 内置自研的 GBNF 文法引擎（`src/llama-grammar.cpp`，示例文法在 `grammars/*.gbnf`），文法直接在 C++ 里逐 token 推进；LiteRT-LM 则经 C bridge 借 Rust 的 llguidance。自研引擎少一层依赖、文法方言自己定义；借力 llguidance 拿到的是跨项目共享的语法生态与优化。

</div>

约束解码的意义是把"结构合法"从"祈祷模型别出错"变成"从机制上不可能出错"——只要语法写对了，输出就一定合法。这对下一节的工具调用是刚需。

## Tool Use：让模型调用函数

把感知输入和受控输出接起来，就是 Tool Use（工具调用/函数调用）——让模型不只是回话，而是能调外部函数：查天气、算数、搜数据库。

一次工具调用走一条完整的链路：

1. **声明工具**。你在对话的开场白里告诉模型有哪些工具可用，就是第 3 章 `Preface` 里的 `tools` 字段。`Preface` 是个 `variant`，实际装的是 `JsonPreface`（`runtime/conversation/io_types.h:30 @ v0.13.1`），里面三个 `nlohmann::ordered_json` 字段并排：`messages` 是对话历史，`tools` 是可用工具列表（`:36`），`extra_context` 留给模型特定的模板渲染。用 `ordered_json` 而非普通 `json` 是有意的——工具和参数的书写顺序要保住，格式化进 prompt 时不能被容器重排。
2. **格式化进 prompt**。这些工具描述被按模型认得的格式写进提示词。`FormatValueAsFc`（`runtime/components/tool_use/fc_tool_format_utils.h @ v0.13.1`，`fc` 即 function call）把标准 JSON 转成一种更省 token 的 FC 格式，头文件里的例子把差异讲明白了：键不加引号（`"string_value"` 变 `string_value`），字符串用 `<escape>` 标签包起来而不是双引号（`"foo"` 变 `<escape>foo<escape>`）。去掉成对的引号，是为了让同样的工具声明少占 token——上下文窗口寸土寸金，工具声明又常年占在 prompt 开头。
3. **生成调用**。模型决定要用某个工具时，输出一段结构化的函数调用文本，长这样：`call:tool_name{param_1:7,param_2:<escape>foo<escape>}`。这里约束解码派上用场：开着它，模型吐出的调用就一定是结构合法的（第 3 章 `ConversationConfig` 那个开关的用途之一）。
4. **解析回填**。运行时把这段文本解析回结构化的函数名和参数。解析用的是 ANTLR 语法，而不是拿正则去凑。FC 格式的整个文法只有六条规则（`runtime/components/tool_use/antlr/AntlrFcParser.g4 @ v0.13.1`）：

```antlr
start : functionCall EOF;
functionCall: CALL COLON ID object?;            // (1)
object : OPEN_BRACE ( pair (COMMA pair)* )? CLOSE_BRACE;
pair : ID COLON value;
value
    : ESCAPED_STRING | NUMBER | BOOLEAN | NULL_LITERAL | object | array;  // (2)
array: OPEN_BRACKET ( value (COMMA value)* )? CLOSE_BRACKET;
```

(1) 一次调用就是 `call` 关键字、冒号、函数名（`ID`）、后面跟一个可选的参数 `object`，`call:tool_name{...}` 逐字对上。(2) `value` 是递归定义的：一个值可以是字符串、数字、布尔、null，也可以再嵌一个 `object` 或 `array`。这条递归是正则做不到的——正则识别不了任意深度的嵌套括号，而文法解析器天生能。`ParseFcExpression`（`runtime/components/tool_use/fc_parser_utils.h:41 @ v0.13.1`）跑完这套文法，把 `call:tool_name{param_1:7,param_2:<escape>foo<escape>}` 还原成 `{"name":"tool_name","arguments":{"param_1":7,"param_2":"foo"}}`：`<escape>` 标签脱掉、`7` 还原成数字而非字符串。解析出的调用交给你的函数执行，结果再作为一条消息喂回模型，继续对话。同目录下另有 `AntlrJson`、`AntlrPython` 两套文法，对应不同模型偏好的调用格式（有的吐 JSON、有的吐 Python 风格的函数调用）。

这条链路把前面几章的零件串了起来：Preface（第 3 章）声明工具，约束解码（本章上一节）保证输出合法，ANTLR 文法把文本解析回结构。四步下来，一个只会输出文本的模型，就有了调用真实函数的能力。（各环节职责与完整时序，另见 `docs/api/cpp/tool-use.md`。）

> 版本注记
> 工具调用的文法与解析细节（ANTLR 那几个 `.g4`、不同模型的函数调用格式差异）在版本间有演进，本节只讲稳定的四步骨架。某些模型在嵌套 JSON 参数上的解析边界曾有过问题（上游 `LiteRT-LM#2418`），属实现细节，不在本节的骨架之列。

## 小结

这一章讲了两个方向的扩展，它们共享同一套地基。输入端，多模态靠"万物皆 embedding"——图片经 patchify、视觉执行器编码、再填进特殊 token 占的坑，之后的流水线一字不改。输出端，约束解码靠"每步掐掉非法 token"把结构合法从祈祷变成保证，Tool Use 再把它和 Preface、ANTLR 文法串成完整的函数调用链路。两个方向都印证了第 2 章那句"接口隔离"：新能力是在输入端多接一个编码器、在输出端多插一道约束长出来的，核心流水线一字未动。

下一章是最后一块工程拼图：这套 C++ 核心，怎么服务从 Python 到 Web 的六种语言（这笔账，下一章开头数清）。

---

## 参考

- 多模态：`runtime/components/preprocessor/image_preprocessor_utils.cc:26 @ v0.13.1`（`GetAspectRatioPreservingSize` 缩放/切块，头文件声明 `image_preprocessor_utils.h:28`）；`runtime/executor/vision_litert_compiled_model_executor.cc:454 @ v0.13.1`（`Encode` 两级串联，头文件声明 `.h:57`）；`runtime/executor/llm_executor_io_types.h @ v0.13.1`（`ExecutorVisionData`:216；`kSpecialToken`:220；对齐示例:202；`ExecutorAudioData`:277，注释:263，`kSpecialToken`:281）；`runtime/executor/llm_executor_base.h:178 @ v0.13.1`（`FillVisionEmbeddings`，带 `image_index`）；`runtime/executor/audio_litert_compiled_model_executor.cc:941 @ v0.13.1`（音频 `Encode`）。
- 约束解码：`runtime/components/constrained_decoding/constrained_decoder.h:48 @ v0.13.1`（循环用法注释:39-47）；`constrained_decoder.cc:73 @ v0.13.1`（`MaskLogits` 双重循环 + `float` 最小值）；`bitmap.h:21`（`Bitmap::Get`）；`llg_constraint.cc @ v0.13.1`（`Start`/`ComputeNext`/`ComputeBitmap` 三次 FFI、掩码解包:57）；llguidance（`PATCH.llguidance*`）；`docs/api/cpp/constrained-decoding.md`。
- Tool Use：`runtime/conversation/io_types.h:30 @ v0.13.1`（`JsonPreface`，`tools`:36，`Preface` variant:59）；`runtime/components/tool_use/fc_tool_format_utils.h @ v0.13.1`（`FormatValueAsFc`，FC 格式）；`fc_parser_utils.h:41`（`ParseFcExpression`）；`antlr/AntlrFcParser.g4 @ v0.13.1`（六条规则文法，另有 `AntlrJson`/`AntlrPython`）；`docs/api/cpp/tool-use.md`。

<!-- 补读：vision/audio executor 已贴 .cc（Encode 两级串联 encoder→adapter），patchify 贴 .cc（GetAspectRatioPreservingSize 缩放对齐公式），约束解码贴 MaskLogits 双重循环 + llg_constraint FFI 三调用 + 掩码位解包，tool_use 贴 AntlrFcParser.g4 六条文法 + FC 格式差异 + ParseFcExpression。实测（图片端到端、visual token 计数、约束解码开/关工具调用成功率）待基准 D 回填〔基准 D〕。双主题章，两半已切干净。图 10-2(约束解码逐步屏蔽) 表 10-1(Tool Use 各环节) 规格见 notes.md，本轮出签名图 10-1。 -->
