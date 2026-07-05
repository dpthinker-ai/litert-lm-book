# 第 7 章 模型的形态：量化、.litertlm 格式与 LoRA

> 本章目标：理清一个模型在端侧从磁盘到显存的完整形态——量化到底省了什么，为什么 LiteRT-LM 要定义一套专用文件格式，加载时如何用 mmap 与并行降低冷启动延迟，以及如何在不改动基座权重的前提下用 LoRA 适配新任务。

前一章分析的是运行时的内存开销（KV cache）。这一章转向模型本身：它以什么形态存在于磁盘上、如何被量化压小、加载时如何减少延迟与拷贝、又如何在不重训的前提下适配新任务。第 2 章「二十个问题」里的第 15、16 问，本章回答。

本章比前几章更贴近系统层。量化那一节是一笔可以当面算清的带宽账；格式与加载两节要一路读到 mmap 的系统调用、页对齐的运行时断言、macOS 与其他平台相反的预取策略；LoRA 一节要看清增量权重从磁盘到后端缓冲的两阶段拷贝。这些机制平时被一句「模型加载好了」盖过，但它们决定了用户点击到首字输出之间那几秒的长短，也决定了多任务切换时的显存占用。

## 量化省的是体积、带宽还是算力

第 1 章用过 int4：一个参数压到 4 比特，即 0.5 字节。第 15 问要分清的是，这 0.5 字节省下的到底是磁盘体积、内存带宽，还是计算量。三者的答案并不相同。

先看体积。同一个模型，权重从 fp16（2 字节/参数）压到 int4（0.5 字节/参数），落盘体积降到四分之一。第 1 章讲的内存容量约束（后文有时称内存墙）因此松动：原本放不进设备内存的模型，量化后放得下。

再看带宽，这是对 decode 阶段影响最大、也最容易被忽略的一项。回忆第 1 章的上限估算：decode 速度受限于内存带宽除以每个 decode step 要读取的权重字节数。权重体积降到四分之一，每步要从内存搬到计算单元的字节数也降到四分之一，decode 吞吐的理论上限随之抬高到约四倍。「约」是因为 KV cache 的读写、激活的搬运等开销并未同比缩小，实际增幅小于四倍。量化在这里的意义不止是「装得下」，还有「读得快」。

最后看算力，这一项量化不一定省。int4 权重在参与矩阵乘之前，通常要先反量化回浮点或提升到更高位宽，这一步本身有计算开销。但 decode 是内存带宽受限的（第 1 章的 Roofline 分析），瓶颈在访存而非计算，反量化多出的那点算力落在瓶颈之外，不构成净损失。综合下来，低比特量化对 decode 的净效果仍是加速。

量化主要省的是体积和带宽，并通过带宽间接提升 decode 吞吐；算力不是它的收益点。这也解释了为什么端侧默认采用低比特量化，而不是直接用 fp16。下面把「精度」这个常被当作开关的概念拆成两个独立维度，再看代码里量化方案的具体形态。

## 精度是一条谱系，权重与激活是两个独立维度

「量化」常被理解成一个二选一的开关。在端侧，精度更接近一条连续谱系，而且要分清两个彼此独立的维度：权重量化到 int4 是一个维度，模型运行时中间激活值的计算精度是另一个维度。

LiteRT-LM 把激活精度做成一个配置项 `ActivationDataType`（`runtime/executor/executor_settings_base.h:62 @ v0.13.1`）：

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

四个档位按此顺序对内存与带宽的占用递减。(1) FLOAT32 是保真基准，一个激活值占 4 字节；(2) FLOAT16 折半到 2 字节，是端侧最常见的默认；(3) INT8 压到 1 字节，比 fp32 省四分之三，但对数值范围敏感的层可能损失精度。这里用的是 `enum class` 而非松散的整数常量，传入非法档位在编译期即被类型系统拦截，不会静默退化成一个精度更低的配置。

激活精度给了一个权衡空间：激活用 fp16 比 fp32 省一半内存和带宽，多数模型上质量损失可忽略；进一步压到 int8 更省，但对某些模型精度影响变大。激活精度与权重量化相互独立：同一份 int4 权重，激活既可以配 fp16 也可以配 int8。权重量化决定模型的落盘与常驻体积，激活精度决定运行时的内存占用、带宽压力与数值质量。紧挨着的 `Backend` 枚举（`:34`–`:54`）列出这些精度实际落地的执行后端：`CPU`、`GPU`、`NPU`，外加两条手写优化路径 `GPU_ARTISAN` 与 `GOOGLE_TENSOR_ARTISAN`。哪一档激活精度在哪个后端上真正快，取决于后端算子的支持情况，这条线留到第 8 章展开。

### 混合精度的一处实证：不同层量化到不同位宽

前面说「对数值范围敏感的层可能损失精度」是一个抽象论断。代码里有一处地方把它落到了具体的量化方案上。同一个文件往下几行，有一个 `FakeWeightsMode` 枚举（`runtime/executor/executor_settings_base.h:82 @ v0.13.1`）：

```cpp
// Fake weights mode.
enum class FakeWeightsMode {
  // Don't use fake weights, read real weights from disk.
  FAKE_WEIGHTS_NONE,

  // Replace all weights with INT8 fakes.
  FAKE_WEIGHTS_8BITS_ALL_LAYERS,     // (1)

  // Replace feedforward and embedding weights with INT4 fakes and replace
  // attention weights with INT8 fakes.
  FAKE_WEIGHTS_ATTN_8_FFN_4_EMB_4,   // (2)
};
```

「fake」指这些权重不是从磁盘读来的真实训练结果，而是按指定位宽合成的占位数据，用于在没有真实模型产物时测量某个量化方案的内存与速度表现，属于 benchmark 与开发用途。它的价值在于把两种量化策略写成了具名档位。(1) `FAKE_WEIGHTS_8BITS_ALL_LAYERS` 是均匀方案，所有层一律 int8。(2) `FAKE_WEIGHTS_ATTN_8_FFN_4_EMB_4` 是混合方案：注意力层保守地留在 int8，前馈层（FFN）与嵌入层激进地压到 int4。

这个混合方案正好实证了上面的抽象论断。注意力层的权重直接参与 query、key、value 的投影与输出投影，量化误差会通过注意力分数放大，因此保守用 8 比特；前馈层与嵌入层的参数量占模型主体，对量化误差的容忍度更高，压到 4 比特换取更大的体积与带宽收益。一个模型内部不同层用不同位宽，这不是特例而是端侧量化的常规做法。真实的量化产物在打包进 `.litertlm` 文件时也遵循类似的按层差异化策略，`FakeWeightsMode` 只是把这套策略显式命名，供性能测量复现。

## 为什么要定义一套专用文件格式

第 16 问：一个 `.litertlm` 文件里装了什么，为什么不直接复用现成格式？

因为端侧要打包的内容不止权重。要在设备上运行一个模型，至少需要权重、tokenizer、聊天模板、停止符、以及能力声明（例如是否支持 speculative decoding，见第 9 章）。这些内容若散成一堆独立文件，分发、版本对齐、加载都会变得繁琐。`.litertlm` 把它们容纳进一个文件。

它的结构是「头 + 分段」。头是一个 FlatBuffer，用类型化的键值对（`KeyValuePair`，`schema/core/litertlm_header_schema.fbs:56 @ v0.13.1`）记录元数据。类型化是这里的要点：键是字符串，值是一个 `union`，可容纳十二种基本类型：

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

union 里每个成员都是一个各含单字段的 `table`（例如 `table UInt8 { value: ubyte; }`）。(1) 连字符串也包成 `StringValue`，于是 union 里存的是带类型标签的对象，读取方可以查询「这个值是什么类型」而不必自行推断。(2) `key` 与 `value` 都标了 `(required)`，FlatBuffer 在校验期强制它们非空，一个缺值的元数据项无法通过校验进入合法文件。相比直接嵌入一段无 schema 的 JSON，这套 schema 在编译期固定了元数据的结构：新增字段要修改 `.fbs` 并调整版本号。schema 头部注释写明了版本规则——纯粹追加新段构成 minor 版本变更，段的重排或删除必须提升 major 版本（`schema/core/litertlm_header_schema.fbs:69 @ v0.13.1`）。这条规则把「向后兼容」写进了格式本身：老版本读取新文件时，只要没有段被重排或删除，就仍能按已知偏移定位到自己认识的段。

正文是若干段（section），每段容纳一类内容：一段权重、一段 tokenizer、一段元数据。段的类型由 `AnySectionDataType` 枚举声明（`:72`）：`TFLiteModel`、`SP_Tokenizer`、`LlmMetadataProto`、`HF_Tokenizer_Zlib`（zlib 压缩的 HuggingFace tokenizer）、`TFLiteWeights`（与 `TFLiteModel` 配对的外挂权重）。每段在文件里的位置由 `SectionObject` 记录（`:91`）：

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

每段是文件里 `[begin_offset, end_offset)` 范围内的一段字节。注释里那条约束支撑着整套加载设计：下一段的起点被对齐到 `BLOCK_SIZE = 16 KiB` 的整数倍。对齐到 16 KiB 的原因在下一节的 mmap：段边界落在页边界的整数倍上，才能对单独一段做零拷贝映射而不牵连相邻段。这条对齐约束在读路径上不是一句注释，而是一个硬断言，下一节会看到它落到 `offset % getpagesize() == 0` 的运行时检查。

写文件的一侧用几个流类把不同来源的数据写入段：文件字节（`FileBackedSectionStream`，`schema/core/litertlm_section.h:98 @ v0.13.1`）、protobuf（`ProtoBufSectionStream`，`:189`）、zlib 压缩流（`ZlibBackendedSectionStream`，`:252`）。这几个是写路径的构件，`FileBackedSectionStream::Prepare()` 把整个文件读进内存缓冲再输出。读路径不走这些流类，走的是下一节的 mmap。

<figure>
{{#include figs/fig-7-1.svg}}
<figcaption>图 7-1　.litertlm 文件结构：一个 FlatBuffer 头（类型化键值对元数据）+ 若干段。段可以是文件字节、protobuf 或 zlib 压缩流。权重、tokenizer、模板、能力声明打包进同一个文件。</figcaption>
</figure>

单文件加分段的设计服务于两件事：分发只需交付一个文件，以及下一节要讲的按需加载与并行加载。第 2 章介绍过的 `litertlm_print` 工具（`schema/core/litertlm_print.cc`）就是把这些段逐一列出来。本章末尾会走读它的实现，看它如何按类型分派打印各类元数据与段。

## 加载：mmap、按需分页与并行

模型文件常有几 GB。如果加载时把整个文件读进内存，冷启动会等待很久。这是一个直接影响用户体验的冷启动延迟（cold-start latency）问题：从点击到模型可用之间的等待时间，与前面讨论的内存容量、带宽、功耗几类约束并列，是端侧另一项要优化的成本。

`.litertlm` 的分段结构在加载阶段再次发挥作用。读文件分两步：先读头，再按需读段。读头很廉价，`ReadHeaderFromLiteRTLM` 只读文件开头那一小段 FlatBuffer 元数据（`schema/core/litertlm_read.h:116 @ v0.13.1`），取得每段的 `begin_offset`、`end_offset`、`data_type`，此时几 GB 的权重一个字节都没有读入。这个头结构禁止拷贝与移动，调用方持有它就持有了整份文件的段索引。省时间的关键在读段这一步。LiteRT-LM 有两条读段路径，先看用自研内存映射类的那条（`schema/core/litertlm_read.cc:238 @ v0.13.1`）：

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

(1) 段的大小是两个 offset 之差，上一节 schema 里那对 `ulong` 在这里被消费。(2) `MemoryMappedFile::Create` 从 `begin_offset` 起映射 `model_size` 字节，正是上一节 16 KiB 对齐所保证的：这段能独立映射而不牵连邻段。映射建立时并不真正读盘，操作系统只是在页表里登记了这段虚拟地址到文件的对应关系，物理页要等 CPU 首次访问时经缺页中断逐页填入。(3) `BuildFromBuffer` 直接用映射出来的指针构建 TFLite 模型，没有把权重整体拷进堆内存的操作。冷启动省下的正是这次拷贝：几 GB 的权重不进读缓冲、不占堆，访问到哪一页才载入哪一页。接口注释点明了配套约定：调用方拿到的 `mapped_file` 持有这块 mmapped buffer，其生命周期必须延续到模型不再使用为止（`schema/core/litertlm_read.h:151 @ v0.13.1`）。这个「生命周期交回调用方」的约定，正是下面要对比的两条读段路径的分野所在。

### mmap 背后：页对齐的硬断言与平台相反的预取策略

上一节说 16 KiB 对齐是为了让段能独立映射，这一节看它在运行时如何被强制。`MemoryMappedFile::Create` 的开头就是一个断言（`runtime/util/memory_mapped_file_posix.cc:101 @ v0.13.1`）：

```cpp
absl::StatusOr<std::unique_ptr<MemoryMappedFile>> MemoryMappedFile::Create(
    int file, uint64_t offset, uint64_t length, absl::string_view key) {
  RET_CHECK_EQ(offset % GetOffsetAlignment(), 0)      // (1)
      << "Offset must be a multiple of page size : " << offset << ", "
      << GetOffsetAlignment();
  // ...
  void* data =
      mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_PRIVATE, file, offset); // (2)
  // ...
#ifdef __APPLE__
  // Mark it not needed to avoid unnecessary page loading on MacOS or iOS.
  if (madvise(data, length, MADV_DONTNEED) != 0) {    // (3)
    ABSL_LOG(WARNING) << "madvise failed: " << strerror(errno);
  }
#else
  if (madvise(data, length, MADV_WILLNEED) != 0) {    // (4)
    ABSL_LOG(WARNING) << "madvise failed: " << strerror(errno);
  }
#endif
  // ...
}
```

(1) `GetOffsetAlignment()` 在 POSIX 平台返回 `getpagesize()`（`:91`），断言要求映射起点是页大小的整数倍。这就是 schema 里 16 KiB 对齐的运行时兑现：常见页大小是 4 KiB 或 16 KiB，16 KiB 是它们的公倍数，任何一段的起点都能通过这个断言。若某段起点没对齐，这里会直接失败，而不是给出一个错位的映射。

(2) `MAP_PRIVATE` 建立的是私有写时复制映射，加上 `PROT_READ | PROT_WRITE`：读取直接命中文件页，一旦写入某页，内核复制该页供本进程私有修改，不回写原文件。权重加载几乎全是读取，这个组合让多个进程映射同一个模型文件时能共享同一批只读物理页，只在极少写入时才分裂出私有页。

(3)(4) 是这段代码里对性能影响最微妙、也最容易被忽略的一处：`madvise` 在两类平台上给出相反的提示。macOS 与 iOS 上是 `MADV_DONTNEED`，注释写明用途是「避免不必要的页加载」；其他平台上是 `MADV_WILLNEED`。`MADV_WILLNEED` 提示内核提前把整段预读进物理内存，赌的是这些页很快都会用到，用一次顺序读盘换取后续零缺页；`MADV_DONTNEED` 相反，提示内核不要预取、甚至丢弃已缓存的页，把物理内存的填充完全推迟到真正的缺页中断。两种策略指向不同的取舍：预读适合确定整段都要用、且希望首次访问不卡顿的场景；不预读适合内存紧张、且未必访问整段的场景。我们据此推断，苹果平台选择 `MADV_DONTNEED` 与其内存受限、且倾向按需分页的系统特性一致，但代码注释只给了「避免不必要的页加载」这一句，未展开具体权衡，此处不做超出注释的断言。这个平台分叉直接决定了冷启动时页是「提前预读」还是「按需缺页」，是冷启动性能账里一处真实的分支。

### 两条读段路径：谁持有 mmap 句柄，谁负责释放

前面走读的 `ReadSectionIntoTFLiteMappedFile` 不是默认路径。同一个文件里还有一条更常用的 `ReadSectionIntoTFLite`（`schema/core/litertlm_read.cc:219 @ v0.13.1`）：

```cpp
absl::Status ReadSectionIntoTFLite(
    const std::string& litertlm_path, uint64_t begin_offset,
    uint64_t end_offset,
    std::unique_ptr<tflite::FlatBufferModel>* tflite_model) {
  size_t model_size = end_offset - begin_offset;

  // Create the MMappedAllocation
  std::unique_ptr<tflite::Allocation> mmap_alloc =
      std::make_unique<tflite::MMAPAllocation>(litertlm_path.c_str(),  // (1)
                                               begin_offset, model_size,
                                               tflite::DefaultErrorReporter());

  // Move the allocation into the FlatBufferModel and build the TFLite
  *tflite_model =
      tflite::FlatBufferModel::BuildFromAllocation(std::move(mmap_alloc)); // (2)
  return absl::OkStatus();
}
```

两条路径都用 mmap 实现零拷贝，区别在于谁持有映射句柄。(1) 这条路径用的是 TFLite 自带的 `tflite::MMAPAllocation`，它内部完成映射。(2) `BuildFromAllocation` 把这个 allocation 移动进 `FlatBufferModel`，映射的生命周期随 `FlatBufferModel` 一同管理：模型析构时映射自动解除，调用方不必额外持有任何句柄。相比之下，前一节 `ReadSectionIntoTFLiteMappedFile` 用 `BuildFromBuffer` 从裸指针建模型，`FlatBufferModel` 并不知道这块内存是 mmap 出来的，也不负责解除映射，于是映射句柄以 `mapped_file` 出参的形式交回调用方，由调用方保证它活得比模型久。

为什么要有这个把生命周期交回调用方的变体？因为有些调用场景需要在模型之外长期持有这块 buffer，或需要对映射做模型层看不到的额外控制（例如统一记账、按段缓存）。默认权重路径走 `MMAPAllocation`，省去调用方的心智负担；需要更强所有权控制的路径走 `MemoryMappedFile` 变体。两条路径在 `ReadTFLiteFileFromSection` 的重载里按调用方是否传入 `mapped_file` 出参分派（`schema/core/litertlm_read.h:151 @ v0.13.1`）。

### 并行加载到底并行了什么

分段结构常被认为「各段可以同时读盘」，实际实现要具体得多。这由一个开关控制（`litert_lm_engine_settings_set_parallel_file_section_loading`，`c/engine.h:295 @ v0.13.1`，默认开）。这个 C-API 设置一路传到 `EngineSettings`，其成员默认为真（`runtime/engine/engine_settings.h:173 @ v0.13.1`：`bool parallel_file_section_loading_ = true;`）。真正读取这个开关做分支的地方在加载流程里（`runtime/core/engine_advanced_impl.cc:249 @ v0.13.1`）：

```cpp
    if (engine_settings.GetParallelFileSectionLoading()) {
      // Launch the tokenizer creation in a separate thread in parallel with the
      // model loading.
      tokenizer_future = std::async(std::launch::async, create_tokenizer);   // (1)
    } else {
      // Launch the tokenizer creation in the same thread.
      tokenizer_future = std::async(std::launch::deferred, create_tokenizer); // (2)
    }
```

开关打开时并行的不是「逐段读盘」，而是 tokenizer 的创建与模型的加载这两件重活。(1) `std::launch::async` 把 `create_tokenizer` 放到独立线程，与主线程的模型加载真正并发；(2) `std::launch::deferred` 则把它推迟到 `future` 被取值时才在当前线程同步执行，等价于串行。收益点落在 tokenizer 与模型这两处，而非逐段并行，原因在于前一节的 mmap 设计：段的物理页是按需缺页载入的，读段本身不是一次大块的顺序读盘，把段两两并行读盘并不会带来相称的收益；真正耗时的是 tokenizer 的构建（HuggingFace tokenizer 还要 zlib 解压）与模型的编译准备，让这两件重活重叠才划算。图 7-1 底注据此表述为「tokenizer 与模型加载可并行」，而非「各段并行读」。

除并行外，编译后的模型产物还能缓存到磁盘（第 2 章 benchmark 用过的 `--cache disk`），二次启动直接复用、省去重新编译。按需分页、tokenizer 与模型并行、编译产物缓存，三者共同降低冷启动延迟。其中编译缓存的效果在本书基准里可见：GPU 后端首次运行（缓存未热）Init 5.29 s，其后稳定在约 1.77 s〔基准 D〕。并行加载开关开/关的冷启动差异本书未单独测量。

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
