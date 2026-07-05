# 模块素材：模型资源与扩展组件 (Model Resources & Extension Components)  `components-resources`

> 来源：litert-lm-guide/data.json（多智能体源码分析）。**这是素材，不是正文**。
> 引用进正文前，须按 CLAUDE.md 第二节逐条 Read 源码核验、补 `@ v0.13.1`。

**一句话**：该模块负责把端侧 LLM 运行所需的全部"资源"装载进内存并按需提供给执行器，包括：从 .litertlm / .task 容器中懒加载 TFLite 子模型、tokenizer 与元数据；LoRA 适配器的加载/切换/填充；词嵌入(text/multimodal)的查表；图像/音频预处理(含 patchify)；以及 tool use(function calling)的输出解析与工具声明格式化。它是连接"磁盘上的模型文件"与"实际推理执行器"之间的资源供给层。

**在架构中的位置**：位于架构的"资源供给层"，处于模型文件(.litertlm/.task)与 executor/engine 之间。上游是 runtime/util 中的容器加载器(LitertLmLoader、ModelAssetBundleResources、LoraData、ScopedFile/MemoryMappedFile)，本模块通过统一的 ModelResources 接口把这些原始字节包装为可用的 litert::Model、Tokenizer、LlmMetadata。下游是 executor(如 llm_litert_executor)与 engine：执行器通过 ModelResources 取出各类子模型并编译，EmbeddingLookupManager 在 prefill/decode 时把 token 转成 embedding 向量，LoraManager 在推理前把 LoRA 权重填进 CompiledModel 的输入 TensorBuffer，preprocessor 把原始图像/音频转成模型输入张量，tool_use 在生成结束后把文本解析成结构化 function call。

## 关键文件
- `runtime/components/model_resources.h` — 定义 ModelResources 抽象接口与 ModelType 枚举(prefill_decode/embedder/vision/audio/mtp 等子模型类型)，以及 ModelType<->字符串互转工具，是整个资源层的核心契约。
- `runtime/components/model_resources_litert_lm.cc` — .litertlm 容器的 ModelResources 实现：通过 LitertLmLoader mmap 各 section，懒加载并缓存 litert::Model，支持 file-backed(从 fd+offset 直接建模)与 mmap buffer 两种装载路径，并选择 SentencePiece/HuggingFace tokenizer。
- `runtime/components/model_resources_task.cc` — .task(MediaPipe 资产包)格式的 ModelResources 实现：按 ModelType 字符串名从 ModelAssetBundleResources 取文件，tokenizer 固定用 SentencePiece，metadata 经 ExtractOrConvertLlmMetadata 兼容旧格式。
- `runtime/components/lora_manager.cc` — 按 lora_id 管理多个 LoRA 适配器：LoadLoRA 仅读取 LoraData(轻量)，UseLoRA 时才惰性创建 LoRA(在后端分配 TensorBuffer)，实现切换不同适配器。
- `runtime/components/lora.cc` — 单个 LoRA 适配器实现：遍历 decode signature 的输入，识别 LoRA 输入名，把 LoraData 中对应张量 memcpy 进后端 TensorBuffer，缺失则填零。
- `runtime/components/embedding_lookup/embedding_lookup_manager.cc` — 统一的 embedding 查表入口：根据 token 正负分流到文本 / 多模态 / end-of-multimodal 查表，prefill 时叠加多模态 embedding，decode 时只走文本。
- `runtime/components/preprocessor/image_preprocessor_utils.cc` — 图像预处理核心算法：GetAspectRatioPreservingSize 在保持长宽比下计算 patch 友好的目标尺寸；PatchifyImage 把 RGB 浮点图切成 patch 张量并生成 positions_xy 位置张量。
- `runtime/components/tool_use/parser_utils.cc` — tool use 解析总入口 ParseTextAndToolCalls：用 RE2 提取 code fence 之间的工具调用块，按 python/json/fc 三种语法分派到对应解析器，产出 OpenAI 风格的 content + tool_calls JSON。
- `runtime/components/tool_use/rust/parsers.rs` — 通过 cxx::bridge 暴露给 C++ 的 Rust 解析器 FFI：parse_python/fc/json_expression 把模型输出解析为 JsonValue 树，C++ 侧再转回 nlohmann::json。
- `runtime/util/lora_util.cc` — LoRA 输入名正则识别(IsLoRAInputName)与带自动对齐的 mmap 包装(MemoryMappedFileWithAutoAlignment)，被 LoRA 装载复用。

## 核心抽象
- **ModelResources** (interface)：资源层的统一抽象基类(model_resources.h)。关键纯虚方法：GetTFLiteModel(按 ModelType 懒加载并缓存 litert::Model)、GetTFLiteModelBuffer(返回原始字节视图)、GetTokenizer、GetLlmMetadata、GetScopedFile、GetWeightsSectionOffset(外部权重的 section 偏移)、GetTFLiteModelBackendConstraint/PreferActivationType。设计为'懒加载+缓存'且明确标注非线程安全。
- **ModelType** (enum)：标识容器内不同子模型(model_resources.h)：kTfLitePrefillDecode(主模型)、kTfLiteEmbedder/kTfLitePerLayerEmbedder(词嵌入)、kTfLiteVisionEncoder/Adapter/EndOfVision、kTfLiteAudioFrontend/EncoderHw/Adapter/EndOfAudio、kArtisanTextDecoder、kTfLiteMtpDrafter/Aux(投机解码)。配套 StringToModelType/ModelTypeToString 做大小写无关的双向转换。
- **ModelResourcesLitertLm** (class)：.litertlm 实现。GetTFLiteModel 优先走 enable_file_backed_model_loading 路径(CreateModelFromFileSection 用 dup 后的 fd + begin/end offset 直接建模，物理内存按需分配)，否则用 LitertLmLoader mmap 出的 BufferRef 建模；用 model_map_ 缓存。GetTokenizer 在 SentencePiece 与 HuggingFace 间按编译宏择优。
- **ModelResourcesTask** (class)：.task 实现。子模型以 ModelType 的字符串名作为 bundle 内文件名查取；GetTokenizer 固定读取 'TOKENIZER_MODEL' 用 SentencePiece；GetLlmMetadata 读取 'METADATA' 经 ExtractOrConvertLlmMetadata 转换。GetScopedFile/GetWeightsSectionOffset/BackendConstraint 等均返回 Unimplemented/nullopt(task 格式不支持外部权重)。
- **ModelResourcesStreaming** (class)：占位实现(model_resources_streaming.cc)。当模型权重以流式方式喂入而非从资源随机读取时使用，几乎所有方法返回 UnimplementedError，因为流式不支持对模型文件的随机访问。
- **LoraManager** (class)：LoRA 适配器管理器(lora_manager.h)。按 uint32 lora_id 维护 lora_data_(已读数据)与 loras_(已实例化到后端)两张表；LoadLoRA 只读 LoraData，UseLoRA 惰性把 LoraData 提升为 LoRA 并设为当前；GetLoRABuffers 返回当前适配器所有 LoRA 张量名->TensorBuffer 的副本。
- **LoRA** (class)：单个 LoRA 适配器(lora.h/lora.cc)。Init 遍历 decode signature 输入名，用 IsLoRAInputName 过滤出 LoRA 张量，CreateInputBuffer 在后端建 TensorBuffer 并把 LoraData 张量 memcpy 进去(缺失填零、并校验 size 一致)。TensorBuffer 本质是共享指针，Get*Buffer 返回 Duplicate 副本以正确管理引用计数。
- **EmbeddingLookupManager / EmbeddingLookup** (class)：词嵌入查表(embedding_lookup_manager.h / embedding_lookup.h)。EmbeddingLookup 是抽象接口(LookupPrefill/LookupDecode)。Manager 持有 text、multi_modal、end_of_multi_modal 三类查表器：token>=0 走文本表，token<0(多模态占位)在 prefill 期叠加多模态 embedding；decode 期禁止多模态;fully_supports_multi_modal 为 false 时用 0 号 entry 的默认向量填充。
- **ImagePreprocessor / PatchifyImage** (class)：图像预处理(image_preprocessor.h / image_preprocessor_utils.cc)。ImagePreprocessor 是抽象基类，Create 按编译选项返回 Skia 或 stb 实现。PatchifyConfig 描述 patch 宽高/最大 patch 数/pooling kernel。PatchifyImage 把 RGB 浮点图切成 [batch,num_patches,patch_dim] 张量并生成 [batch,num_patches,2] 的 positions_xy 位置张量(供视觉 ViT 用)。
- **ParseTextAndToolCalls / SyntaxType** (function)：tool use 解析(parser_utils.h/.cc)。SyntaxType 区分 kPython/kJson/kFc 三种工具调用语法。ParseTextAndToolCalls 用 RE2 反复 Consume 出 code fence 间的代码块，分派到 ParsePython/Json/FcExpression(底层为 Rust FFI)，产出 {content:[...], tool_calls:[{type:function,function:...}]} 的结构化 JSON。
- **FormatToolAsFc / FormatValueAsFc** (function)：FC(function call)工具声明格式化(fc_tool_format_utils.h/.cc)。把 JSON Schema 风格的工具声明转成模型偏好的 FC 文本格式(键不加引号、字符串用 <escape> 包裹)，用于在 prompt 中向模型描述可用工具。

## 数据流
1. 装载阶段：engine/executor 拿到模型文件 -> 据格式构造 LitertLmLoader(.litertlm) 或 ModelAssetBundleResources(.task) -> 据此 Create 出对应的 ModelResources 实现。
2. 取子模型：executor 调用 ModelResources::GetTFLiteModel(ModelType::kTfLitePrefillDecode 等) -> 首次调用时从 mmap buffer 或 fd+offset 创建 litert::Model 并缓存进 model_map_ -> 返回指针给 executor 去 CompiledModel 编译。
3. 取 tokenizer / 元数据：GetTokenizer()(SentencePiece 或 HuggingFace) 和 GetLlmMetadata()(解析 LlmMetadata proto) 供分词与配置使用。
4. LoRA 流程：LoraManager.LoadLoRA(id, assets) 读出 LoraData -> UseLoRA(id) 惰性构造 LoRA，在 CompiledModel 上为每个 LoRA 输入名建 TensorBuffer 并 memcpy 权重 -> 执行器调 GetLoRABuffers() 取副本，连同其他输入一起跑 decode。
5. Embedding 流程(prefill)：tokens 进入 EmbeddingLookupManager.LookupPrefill -> 文本查表写入 output_tensor -> 若支持多模态，对 token<0 的占位再叠加 multimodal/end-of-multimodal embedding。decode 时逐 token 走 LookupDecode。
6. 多模态输入：原始图像字节 -> ImagePreprocessor.Preprocess(resize 到目标尺寸) -> PatchifyImage 切 patch 并生成 positions_xy -> 作为 vision encoder 的输入张量。
7. Tool use(生成后)：模型输出文本 -> ParseTextAndToolCalls 用 RE2 抽取 code fence 间内容 -> 按 SyntaxType 经 Rust FFI 解析成 JsonValue -> 转回 nlohmann::json，输出 content + tool_calls 结构。

## 概念
- **.litertlm 与 .task 两种模型容器**：.litertlm 是 LiteRT-LM 自有的单文件容器(带 FlatBuffer header schema 描述各 section 偏移)，支持 mmap 随机访问与外部权重 section;.task 是 MediaPipe 资产包格式(类似 zip，里面按文件名放各子模型/tokenizer/metadata)。本模块用同一个 ModelResources 接口屏蔽两者差异。
- **懒加载 + mmap**：模型很大，一次性读进内存代价高。ModelResources 在真正调用 Get*() 时才把对应 section 通过内存映射(mmap)建成 litert::Model；mmap 让物理内存按访问页惰性分配，并把结果缓存进 model_map_ 复用，避免重复建模。
- **LoRA(Low-Rank Adaptation)**：一种轻量微调技术：不改基座大模型权重，只额外训练一组低秩矩阵(rank 通常很小)。推理时把这些低秩增量权重作为额外输入张量注入模型。本模块的 LoRA/LoraManager 负责加载这些增量权重、按 id 切换不同适配器，并填进模型的 LoRA 输入。
- **词嵌入查表(Embedding Lookup)**：把离散 token id 映射成连续向量。理想情况嵌入应是主模型的一部分，但大词表嵌入会占满加速器内存，因此这里把嵌入查表单独放到 CPU 上做。多模态场景中负数 token 是图像/音频的占位符，需用对应模态的 embedding 替换。
- **Patchify(图像分块)**：ViT 类视觉编码器不直接吃整张图，而是把图切成固定大小的小块(patch)展平成序列，再配上每块的位置坐标(positions_xy)。PatchifyImage 完成切块并生成位置张量;切块前还要把图 resize 到能被 patch 整除、且 patch 数不超上限的尺寸。
- **Tool use / Function calling**：让 LLM 输出可被程序执行的'函数调用'。模型按约定语法(Python 调用式 / JSON / 自有 FC 格式)在 code fence 里写出要调的函数与参数；本模块负责把这段文本解析成结构化的 tool_calls，并能反向把工具声明格式化进 prompt 让模型知道有哪些工具。
- **cxx::bridge(Rust/C++ FFI)**：tool use 的语法解析(Python/FC/JSON 表达式)用 Rust 写(借助成熟的解析库)，通过 cxx 这个 Rust<->C++ 桥把 Rust 函数(parse_*_expression)暴露成 C++ 可调用接口，并用 JsonValue 类型在两边传递解析结果。chat template 渲染同样用 Rust(minijinja)实现。

## 优化
- **懒加载 + 模型缓存(lazy loading + model_map_)**：ModelResources 只在首次 Get*() 时才创建 litert::Model 并存入 model_map_/llm_metadata_，后续命中缓存直接返回，避免未用到的子模型(如 vision/audio)被无谓装载。
- **mmap 与 file-backed 建模**：.litertlm 用内存映射读取，物理内存按页惰性分配;enable_file_backed_model_loading 时用 dup 出的 fd + begin/end offset 直接 CreateFromFd 建模，权重不必整体 mmap 进进程地址空间，进一步降低常驻内存。
- **LoRA 两阶段惰性实例化**：LoadLoRA 只把 LoraData 读进来(轻量、可 mmap)，真正在后端(可能是 GPU)分配 TensorBuffer 的昂贵操作推迟到 UseLoRA 才做;切到某适配器后会 erase 掉对应 LoraData 释放 CPU 端内存。
- **TensorBuffer 引用计数与 Duplicate**：LoRA 张量本质是共享指针，GetLoRABuffer(s) 返回 Duplicate 副本而非裸引用，由调用方负责释放，保证多处使用时底层数据生命周期正确。
- **CPU 侧嵌入查表分离**：把大词表 embedding 放到 CPU 查表而非加速器，避免大嵌入表撑爆加速器内存;多模态占位 token 在 prefill 时才叠加，decode 路径保持精简。
- **对齐感知的 mmap 包装**：MemoryMappedFileWithAutoAlignment 自动把请求 offset 向下对齐到平台 mmap 对齐边界、size 向上对齐，再用内部 offset/size 还原真实范围，使 LoRA 等任意偏移的 section 也能安全 mmap。
- **Rust 实现解析/模板的工程取舍**：把易错的语法解析(Python/FC/JSON 表达式)与 chat template 渲染放到 Rust(生态成熟、内存安全)，通过 cxx::bridge 与 C++ 互操作，兼顾安全与性能。

## 关键代码片段（待核验 @ v0.13.1）
**ModelResources 接口：懒加载式资源供给的核心契约** — 待核验：`runtime/components/model_resources.h:151-187`
```cpp
// 据 ModelType 懒加载并缓存 litert::Model(mmap 文件，物理内存按需分配)
virtual absl::StatusOr<const litert::Model*> GetTFLiteModel(ModelType model_type) = 0;
// 返回原始 TFLite 字节视图(生命周期随 ModelResources)
virtual absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(ModelType model_type) = 0;
// 外部权重 section 的起止偏移
virtual absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(ModelType model_type) = 0;
virtual absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() = 0;
virtual absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() = 0;
```
**.litertlm 的 GetTFLiteModel：file-backed 优先 + 缓存复用** — 待核验：`runtime/components/model_resources_litert_lm.cc:91-112`
```cpp
auto it = model_map_.find(model_type);
if (it != model_map_.end()) return it->second.get();  // 命中缓存
if (enable_file_backed_model_loading_) {
  auto scoped_file = litert_lm_loader_->GetScopedFile();
  auto section_location = litert_lm_loader_->GetSectionLocation(
      BufferKey(schema::AnySectionDataType_TFLiteModel, model_type));
  if (scoped_file.ok() && section_location.ok()) {
    LITERT_ASSIGN_OR_RETURN(auto model_from_section,
        CreateModelFromFileSection(scoped_file->get(),
            section_location->first, section_location->second));
    auto& model = model_map_[model_type];
    model = std::make_unique<litert::Model>(std::move(model_from_section));
    return model.get();
  }
}
```
**LoRA::Init：识别 LoRA 输入名并把权重填进后端 TensorBuffer** — 待核验：`runtime/components/lora.cc:70-102`
```cpp
for (const auto& input_name : input_names) {
  if (!IsLoRAInputName(input_name)) continue;
  LITERT_ASSIGN_OR_RETURN(litert::TensorBuffer tensor_buffer,
      compiled_model_.CreateInputBuffer(signature_name_, input_name));
  // ... lock buffer ...
  if (lora_data_->HasTensor(input_name)) {
    ASSIGN_OR_RETURN(auto lora_tensor_data, lora_data_->ReadTensor(input_name));
    RET_CHECK_EQ(tensor_buffer_size, lora_tensor_data->Size());  // size 校验
    std::memcpy(lock_and_addr.second, lora_tensor_data->Data(), lora_tensor_data->Size());
  } else {
    std::memset(lock_and_addr.second, 0, tensor_buffer_size);  // 缺失填零
  }
  lora_buffers_[input_name] = std::move(tensor_buffer);
}
```
**LoraManager::UseLoRA：两阶段惰性实例化并释放 LoraData** — 待核验：`runtime/components/lora_manager.cc:61-68`
```cpp
if (!loras_.contains(lora_id)) {
  ASSIGN_OR_RETURN(auto lora, LoRA::Create(std::move(lora_data_[lora_id]),
                                           compiled_model_, signature_name_));
  loras_[lora_id] = std::move(lora);
  lora_data_.erase(lora_id);  // 提升为后端 LoRA 后释放 CPU 端数据
}
current_lora_id_ = lora_id;
```
**ParseTextAndToolCalls：RE2 切 code fence + 按语法分派解析** — 待核验：`runtime/components/tool_use/parser_utils.cc:109-156`
```cpp
while (RE2::Consume(&response_str, regex, &text, &code_block)) {
  if (!text.empty())
    result["content"].push_back({{"type", "text"}, {"text", text}});
  if (!code_block.empty()) {
    absl::StatusOr<nlohmann::ordered_json> tool_calls;
    if (syntax_type == SyntaxType::kPython)      tool_calls = ParsePythonExpression(code_block);
    else if (syntax_type == SyntaxType::kJson)   tool_calls = ParseJsonExpression(code_block);
    else if (syntax_type == SyntaxType::kFc)     tool_calls = ParseFcExpression(code_block);
    for (const auto& tool_call : *tool_calls)
      result["tool_calls"].push_back({{"type", "function"}, {"function", tool_call}});
  }
}
```

## 入手顺序
- 先读 model_resources.h：理解 ModelResources 接口的全部纯虚方法和 ModelType 枚举，这是整个模块的契约。
- 再读 model_resources_litert_lm.cc 的 GetTFLiteModel 与 GetTokenizer：看懂懒加载+缓存、file-backed vs mmap 两条装载路径，以及 tokenizer 选择逻辑;对照 model_resources_task.cc 看另一种容器格式如何实现同一接口。
- 读 lora_manager.cc + lora.cc + util/lora_util.cc(IsLoRAInputName)：理清 Load(只读数据)/Use(惰性实例化)/GetLoRABuffers 的两阶段惰性加载与 TensorBuffer 引用计数。
- 读 embedding_lookup/embedding_lookup.h 接口，再读 embedding_lookup_manager.cc 的 LookupPrefill/LookupDecode：看清 token 正负分流与多模态叠加逻辑。
- 读 preprocessor/image_preprocessor.h + image_preprocessor_utils.cc 的 GetAspectRatioPreservingSize 和 PatchifyImage：理解 resize 约束与切块/位置张量生成。
- 最后读 tool_use/parser_utils.cc(ParseTextAndToolCalls 总入口) + python_parser_utils.cc + rust/parsers.rs：从 C++ 入口追到 Rust FFI，理解三种语法的统一解析框架;有余力再看 fc_tool_format_utils.h 的工具声明格式化。
