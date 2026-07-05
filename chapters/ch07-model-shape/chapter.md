# 第 7 章 模型的形态：量化、.litertlm 格式与 LoRA

> 使命：理解一个模型在端侧被"压小、装箱、变体"的完整链路——int4 到底省了什么，为什么要自造一个文件格式，以及不动基座怎么给模型换个"人格"。

前一章算的是运行时的账（KV cache）。这一章算模型本身的账：它以什么形态存在磁盘上、怎么被压小、加载时怎么省时间、又怎么在不重训的前提下适配新任务。第 2 章"二十个问题"里的第 15、16 问，在这里结清。

## int4 省的到底是什么

第 1 章我们用过 int4：一个参数压到 4 比特，0.5 字节。第 15 问要问的是——这 0.5 字节，省的是体积、带宽，还是算力？答案不一样，值得掰开。

- **省体积**：这是最直接的。同一个模型，权重从 fp16（2 字节/参数）压到 int4（0.5 字节/参数），小到四分之一。第 1 章的内存墙因此松动——原本放不下的模型放得下了。
- **省带宽**：这是最容易被忽略、却对 decode 最要命的。回忆第 1 章那条上限公式：decode 速度 = 带宽 ÷ 每 token 要读的字节。权重小到四分之一，每 token 要搬的字节也小到四分之一，**decode 的上限也随之约抬高到四倍**（受 KV cache 等其他开销约束，实际增幅略小）。量化不只是"装得下"，更是"跑得快"。
- **算力：不一定省**。int4 权重在参与计算前，往往要先反量化回浮点，这一步有开销。但 decode 是带宽受限的（第 1 章），访存才是瓶颈，反量化那点算力代价在瓶颈之外——所以净效果仍是快。

一句话：int4 主要省的是**体积和带宽**，并借带宽间接把 decode 提了速；算力不是它的主场。这也是为什么端侧默认用低比特量化，而不是图省事用 fp16。

## 精度不是一个开关，是一条谱系

"量化"容易被当成一个二选一的开关，其实端侧的精度是一条谱系，而且权重和激活是两回事。

权重压到 int4 是一码事；模型跑起来时，中间的激活值用什么精度算，是另一码事。LiteRT-LM 把激活精度做成一个可配置项 `ActivationDataType`（`runtime/executor/executor_settings_base.h:62 @ v0.13.1`）：

```cpp
enum class ActivationDataType {
  // Use float32 as the activation data type.
  FLOAT32,           // (1)
  // Use float16 as the activation data type.
  FLOAT16,           // (2)
  // Use int16 as the activation data type.
  INT16,
  // Use int8 as the activation data type.
  INT8,              // (3)
};
```

四档从上到下越来越省。(1) FLOAT32 是最保真的基准，一个激活值占 4 字节；(2) FLOAT16 折半到 2 字节，是端侧最常见的默认；(3) INT8 压到 1 字节，比 fp32 省四分之三，但对数值范围敏感的层可能掉精度。这是一个 `enum class` 而非松散的整数常量——传错档在编译期就会被拦下，配错激活精度不会静默跑出一个精度更低的模型。

这给了工程师一个权衡空间：激活用 fp16 比 fp32 省一半显存和带宽，通常质量损失可忽略；进一步压到 int8 更省，但对某些模型精度影响变大。注意激活精度与权重量化是彼此独立的两个旋钮：同一份 int4 权重，激活可以配 fp16 也可以配 int8。权重量化管"模型多大"，激活精度管"跑起来占多少、多快、多准"。旁边紧挨着的 `Backend` 枚举（`:34`–`:54`）列出这些精度实际落地的执行后端：`CPU`、`GPU`、`NPU`，外加两条手写优化路径 `GPU_ARTISAN` 与 `GOOGLE_TENSOR_ARTISAN`——哪档激活精度在哪个后端上真正快，取决于后端算子的支持，这条线留到第 8 章接。

## 为什么要自造一个文件格式

第 16 问：一个 `.litertlm` 文件里装了什么，为什么不直接用现成格式？

因为端侧要装的东西不止权重。要跑起来一个模型，你需要：权重、tokenizer、聊天模板、停止符、能力声明（比如支不支持推测解码，第 9 章）……如果这些散成一堆文件，分发、版本对齐、加载都是麻烦。`.litertlm` 把它们打包进**一个文件**。

它的结构是"头 + 分段"。头是一个 FlatBuffer，用类型化的键值对（`KeyValuePair`，`schema/core/litertlm_header_schema.fbs:56 @ v0.13.1`）记录元数据。"类型化"是关键——键是字符串，值是一个 `union`，能装十二种基本类型：

```cpp
// Union for the value in KeyValuePair
union VData {
  UInt8,
  // ...
  StringValue,      // (1)
  // ...
  Double,
}

table KeyValuePair {
  key:string (required);      // (2)
  value:VData (required);
}
```

union 里每个成员都是一个各含单字段的 `table`（如 `table UInt8 { value: ubyte; }`），(1) 连字符串也包成 `StringValue`——这样 union 里放的是带类型标签的对象，读取方能问"这个值到底是什么类型"而不是自己猜。(2) `key` 和 `value` 都标了 `(required)`，FlatBuffer 在校验期强制它们非空，一个没有值的元数据项进不了合法文件。相比塞一段裸 JSON，这套 schema 把"元数据长什么样"钉死在编译期：新增字段要动 `.fbs` 并升版本号，schema 头部注释写明了规矩——纯加段是 minor 版本，段的顺序和删除要 bump major 版本。

正文是若干**段**（section），每段装一样东西——一段权重、一段 tokenizer、一段元数据。段的类型由 `AnySectionDataType` 枚举声明（`:72`）：`TFLiteModel`、`SP_Tokenizer`、`LlmMetadataProto`、`HF_Tokenizer_Zlib`（zlib 压缩的 HuggingFace tokenizer）、`TFLiteWeights`（与 `TFLiteModel` 配对的外挂权重）。每段在文件里的位置由 `SectionObject` 记录（`:91`）：

```cpp
// ...
// In the LitRT-LM file format, BLOCK_SIZE = 16 * 1024.
table SectionObject {
  items:[KeyValuePair]; // (optional)
  begin_offset:ulong;
  end_offset:ulong;
  data_type:AnySectionDataType; // Enum to indicate the type of 'data'
}
```

每段就是文件里 `[begin_offset, end_offset)` 的一段字节。上面注释里那条约束是这套设计的骨架：下一段的起点被对齐到 `BLOCK_SIZE = 16 KiB` 的整数倍。为什么对齐到 16 KiB？因为下一节的 mmap 要按页映射——段边界落在页边界的整数倍上，才能对单独一段做零拷贝映射，而不必牵连相邻段。写文件的一侧则用几个流类把不同来源的数据填进段里：文件字节（`FileBackedSectionStream`，`schema/core/litertlm_section.h:98 @ v0.13.1`）、protobuf（`ProtoBufSectionStream`，`:189`）、zlib 压缩流（`ZlibBackendedSectionStream`，`:252`）。这几个是**写路径**的构件——`FileBackedSectionStream::Prepare()` 把整个文件读进内存缓冲再吐出去；读路径不走它们，走的是下一节的 mmap。

<figure>
{{#include figs/fig-7-1.svg}}
<figcaption>图 7-1　.litertlm 文件结构：一个 FlatBuffer 头（类型化键值对元数据）+ 若干段。段可以是文件字节、protobuf 或 zlib 压缩流。权重、tokenizer、模板、能力声明打包进同一个文件。</figcaption>
</figure>

这个"单文件 + 分段"的设计不是为了好看，是为了两件实事：**分发只需给一个文件**，以及下一节要讲的——**加载可以只挑需要的段、还能并行**。第 2 章那个 `litertlm_print` 工具（`schema/core/litertlm_print.cc`）做的，就是把这些段列出来给你看。

## 加载：mmap 与分段并行，凿冷启动的墙

模型文件常有几 GB。如果加载时老老实实把整个文件读进内存，冷启动要等很久——这是第三堵墙之外的又一个真实体感问题（用户点开 App，等模型加载的那几秒）。

`.litertlm` 的分段结构在这里第二次发力。读文件分两步：先读头，再按需读段。读头是廉价的——`ReadHeaderFromLiteRTLM` 只读文件开头那一小段 FlatBuffer 元数据（`schema/core/litertlm_read.h:116 @ v0.13.1`），拿到每段的 `begin_offset`/`end_offset`/`data_type`，这时几 GB 的权重一个字节都没碰。真正的省时间在读段那一步，它用 **mmap** 把某一段直接映射进地址空间（`schema/core/litertlm_read.cc:238 @ v0.13.1`）：

```cpp
absl::Status ReadSectionIntoTFLiteMappedFile(
    const std::string& litertlm_path, uint64_t begin_offset,
    uint64_t end_offset, ...) {
  size_t model_size = end_offset - begin_offset;      // (1)
  auto model_file = lm::ScopedFile::Open(litertlm_path);
  // ...
  absl::StatusOr<std::unique_ptr<MemoryMappedFile>> mmap_status =
      litert::lm::MemoryMappedFile::Create(platform_file, begin_offset,
                                           model_size, "section");   // (2)
  // ...
  *tflite_model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>((*mapped_file)->data()),
      (*mapped_file)->length());                      // (3)
}
```

(1) 段的大小就是两个 offset 之差，上一节 schema 里那对 `ulong` 在这里被消费。(2) `MemoryMappedFile::Create` 从 `begin_offset` 起映射 `model_size` 字节，正是上一节 16 KiB 对齐要保证的：这段能独立映射，不牵连邻段。映射建立时并不真正读盘——操作系统只是在页表里登记了这段虚拟地址到文件的对应关系，物理页要等 CPU 首次访问才按缺页中断逐页填进来。(3) `BuildFromBuffer` 直接拿映射出来的指针构建 TFLite 模型，没有一次把权重拷进堆内存的操作。冷启动省的就是这次拷贝：几 GB 的权重不进 read 缓冲、不占堆，用到哪一页读哪一页。`read.h` 的接口注释也点明了这个约定：调用方拿到的 `mapped_file` 持有的是 "mmapped buffer"，它的生命周期要活到模型不再用为止（`schema/core/litertlm_read.h:151 @ v0.13.1`）。

分段还带来并行的机会：各段互相独立，可以同时加载。这由一个开关控制（`litert_lm_engine_settings_set_parallel_file_section_loading`，`c/engine.h:295 @ v0.13.1`，默认开）。此外，编译后的模型产物也能缓存到磁盘（第 2 章 benchmark 见过的 `--cache disk`），二次启动直接复用、省掉重新编译。mmap 按需分页、分段并行、编译产物缓存——三招合起来对付的就是冷启动。其中编译缓存的效果在本书基准里直接可见：GPU 后端首次运行（缓存未热）Init 5.29 s，其后稳定在约 1.77 s〔基准 D〕；分段并行加载的开关对比本书未单测。

## LoRA：不动基座，换个人格

最后一块拼图：变体。你有一个通用基座模型，想让它在某个专门任务上更好——写代码、医疗问答、特定语气。重新训练或全量微调一个几 GB 的模型，端侧存不下也换不起。

LoRA 的思路是：**基座权重一个字节都不动，另外挂一小份"增量权重"**。推理时把增量叠加到基座上，模型行为就偏向新任务。增量很小（相比基座是零头），存得下、也能热加载。LiteRT-LM 用两个类支撑它：`LoRA`（`runtime/components/lora.h:40 @ v0.13.1`）持有一份增量权重的后端资源，`LoraManager`（`runtime/components/lora_manager.h:39 @ v0.13.1`）按 id 管理多份。多份靠两张以 id 为键的表并存（`:75`–`:76`）：

```cpp
absl::flat_hash_map<uint32_t, std::unique_ptr<LoraData>> lora_data_;   // (1)
absl::flat_hash_map<uint32_t, std::unique_ptr<LoRA>> loras_;           // (2)
```

(1) 存"从磁盘读进来的原始 LoRA 权重"，(2) 存"在后端上建好的 LoRA 对象"。分成两张表，是因为加载被拆成了两拍。`LoadLoRA` 只做第一拍（`runtime/components/lora_manager.cc:46 @ v0.13.1`）：

```cpp
absl::Status LoraManager::LoadLoRA(uint32_t lora_id,
                                   const ModelAssets& model_assets) {
  if (lora_data_.contains(lora_id)) {
    return absl::AlreadyExistsError("LoRA ID already exists");  // (1)
  }
  ASSIGN_OR_RETURN(auto scoped_file, model_assets.GetOrCreateScopedFile());
  ASSIGN_OR_RETURN(auto lora_data, LoraData::CreateFromScopedFile(scoped_file));
  lora_data_[lora_id] = std::move(lora_data);                   // (2)
  return absl::OkStatus();
}
```

(1) 同一个 id 不许重复加载，防止静默覆盖。(2) 它只把权重读进 `lora_data_`，没碰后端——真正在 GPU 上建资源的第二拍留给 `UseLoRA(lora_id)`，那时才 `LoRA::Create` 并填进 `loras_`（`:57`–`:62`，类注释写明这是 "lazily, only when UseLoRA() is called"）。这个拆分对端侧的意义很直接：可以一次把多份 LoRA 的权重都读进内存待命，但只为当前真正激活的那一份付出建后端资源、占显存的代价，切换任务时换 id 即可，不必重新读盘。

LoRA 的价值恰好呼应本章主题：它是"变体"的最省成本形态——一个基座 + 若干小增量，就能覆盖多个任务，而不必为每个任务存一个完整模型。放到端侧的存储约束下，这个省法尤其值钱。

## 小结

这一章讲的是模型的三种"形态变化"：量化把它压小（省体积、更省带宽，间接提 decode 速度），`.litertlm` 把它连同 tokenizer、模板、能力声明装进一个可 mmap、可分段并行加载的文件，LoRA 用小增量让它在不动基座的前提下适配新任务。三者共同回答一个问题：一个几十亿参数的模型，怎么以端侧扛得住的形态存在。

模型的形态清楚了。下一章回到运行时：同一个模型，为什么在 CPU、GPU、NPU 上跑起来速度甚至输出都不一样。

---

## 参考

- 激活精度与后端：`runtime/executor/executor_settings_base.h @ v0.13.1`（`ActivationDataType`:62，FLOAT32/16、INT16/8 于 `:64`–`:73`；`Backend`:34，`GPU_ARTISAN`/`CPU`/`GPU`/`GOOGLE_TENSOR_ARTISAN`/`NPU` 于 `:42`–`:54`）。
- `.litertlm` 格式：`schema/core/litertlm_header_schema.fbs @ v0.13.1`（`union VData`:39；`KeyValuePair`:56；`AnySectionDataType`:72；`SectionObject`:91，`begin_offset`/`end_offset` 于 `:93`–`:94`，`BLOCK_SIZE=16*1024` 见 `:90` 注释）；`schema/core/litertlm_section.h @ v0.13.1`（写路径流类：`FileBackedSectionStream`:98；`ProtoBufSectionStream`:189；`ZlibBackendedSectionStream`:252）。
- mmap 加载：`schema/core/litertlm_read.h @ v0.13.1`（`ReadHeaderFromLiteRTLM`:116；按段读的 mmapped buffer 约定:151）；`schema/core/litertlm_read.cc @ v0.13.1`（`ReadSectionIntoTFLite`:219 用 `MMAPAllocation`；`ReadSectionIntoTFLiteMappedFile`:238 用 `MemoryMappedFile::Create`）。
- 并行加载：`c/engine.h:295 @ v0.13.1`（`litert_lm_engine_settings_set_parallel_file_section_loading`，默认 true）。
- LoRA：`runtime/components/lora.h:40 @ v0.13.1`（`LoRA`）；`runtime/components/lora_manager.h @ v0.13.1`（`LoraManager`:39；`LoadLoRA`:57；`lora_data_`/`loras_` 两张表:75–76）；`runtime/components/lora_manager.cc:46 @ v0.13.1`（`LoadLoRA` 只填 `lora_data_`，后端资源由 `UseLoRA` 于 `:57` 惰性创建）。

<!-- 实测（int4 vs int8 三角、并行加载 on/off 冷启动、litertlm_print 实剖）待基准 D 回填〔基准 D〕；量化内部(分组/scale)未展开，仅到"权重压 4bit + 激活精度谱系"层面，未臆测未核验的细节。图 7-2(mmap/并行加载) 与表 7-1(量化三角) 规格见 notes.md，本轮出签名图 7-1。 -->
