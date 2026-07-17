# 第 7 章 模型的形态：量化、.litertlm 格式与 LoRA

> 本章目标：理清一个模型在端侧从磁盘到显存的完整形态——量化到底省了什么，为什么 LiteRT-LM 要定义一套专用文件格式，加载时如何用 mmap 与并行降低冷启动延迟，以及如何在不改动基座权重的前提下用 LoRA 适配新任务。

前一章分析的是运行时的内存开销（KV cache）。这一章转向模型本身：它以什么形态存在于磁盘上、如何被量化压小、加载时如何减少延迟与拷贝、又如何在不重训的前提下适配新任务。第 2 章「二十个问题」里的第 15、16 问，本章回答。

本章比前几章更贴近系统层。量化那一节是一笔可以当面算清的带宽账；格式与加载两节要一路读到 mmap 的系统调用、页对齐的运行时断言、macOS 与其他平台相反的预取策略；LoRA 一节要看清增量权重从磁盘到后端缓冲的两阶段拷贝。这些机制平时被一句「模型加载好了」盖过，但它们决定了用户点击到首字输出之间那几秒的长短，也决定了多任务切换时的显存占用。

## 量化省的是体积、带宽还是算力

第 1 章用过 int4：一个参数压到 4 比特，即 0.5 字节。第 15 问要分清的是，这 0.5 字节省下的到底是磁盘体积、内存带宽，还是计算量。三者的答案并不相同。

先看体积。同一个模型，权重从 fp16（2 字节/参数）压到 int4（0.5 字节/参数），落盘体积降到四分之一。第 1 章讲的内存容量约束因此松动：原本放不进设备内存的模型，量化后放得下。

再看带宽，这是对 decode 阶段影响最大、也最容易被忽略的一项。回忆第 1 章的上限估算：decode 速度受限于内存带宽除以每个 decode step 要读取的权重字节数。权重体积降到四分之一，每步要从内存搬到计算单元的字节数也降到四分之一，decode 吞吐的理论上限随之抬高到约四倍。「约」是因为 KV cache 的读写、激活的搬运等开销并未同比缩小，实际增幅小于四倍。量化在这里的意义不止是「装得下」，还有「读得快」。

最后看算力，这一项量化不一定省。int4 权重在参与矩阵乘之前，通常要先反量化回浮点或提升到更高位宽，这一步本身有计算开销。但 decode 是内存带宽受限的（第 1 章的 Roofline 分析），瓶颈在访存而非计算，反量化多出的那点算力落在瓶颈之外，不构成净损失。综合下来，低比特量化对 decode 的净效果仍是加速。

量化主要省的是体积和带宽，并通过带宽间接提升 decode 吞吐；算力不是它的收益点。这也解释了为什么端侧默认采用低比特量化，而不是直接用 fp16。下面把「精度」这个常被当作开关的概念拆成两个独立维度，再看代码里量化方案的具体形态。

| 档位 | 落盘体积 | decode 每步读取 | 算力 | 质量风险 |
|---|---|---|---|---|
| fp16 | 2 字节/参数（基准） | 基准 | 无额外开销（基准） | 基准 |
| int8 | 减半 | 减半 | 反量化小开销，落在带宽瓶颈之外 | 低 |
| int4 | 四分之一 | 四分之一 | 反量化开销，仍落在瓶颈之外 | 视模型而定，端侧默认接受 |

> 表 7-1　量化精度的收益账：体积与带宽随比特数线性缩，算力不是收益点，质量是唯一的变量。

## 精度是一条谱系，权重与激活是两个独立维度

「量化」常被理解成一个二选一的开关。在端侧，精度更接近一条连续谱系，而且要分清两个彼此独立的维度：权重量化到 int4 是一个维度，模型运行时中间激活值的计算精度是另一个维度。

LiteRT-LM 把激活精度做成一个配置项 `ActivationDataType`（`runtime/executor/executor_settings_base.h:62`）：

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

四个档位按此顺序对内存与带宽的占用总体递减（FLOAT16 与 INT16 同为 2 字节，这一档是打平）。(1) FLOAT32 是保真基准，一个激活值占 4 字节；(2) FLOAT16 折半到 2 字节，是端侧最常见的默认；(3) INT8 压到 1 字节，比 fp32 省四分之三，但对数值范围敏感的层可能损失精度。这里用的是 `enum class` 而非松散的整数常量，传入非法档位在编译期即被类型系统拦截，不会静默退化成一个精度更低的配置。

激活精度给了一个权衡空间：激活用 fp16 比 fp32 省一半内存和带宽，多数模型上质量损失可忽略；进一步压到 int8 更省，但对某些模型精度影响变大。激活精度与权重量化相互独立：同一份 int4 权重，激活既可以配 fp16 也可以配 int8。权重量化决定模型的落盘与常驻体积，激活精度决定运行时的内存占用、带宽压力与数值质量。紧挨着的 `Backend` 枚举（`:34`–`:54`）列出这些精度实际落地的执行后端：`CPU`、`GPU`、`NPU`，外加 `CPU_ARTISAN`、`GPU_ARTISAN` 两条手写算子路径，以及 Pixel 专用的 `GOOGLE_TENSOR_ARTISAN`。哪一档激活精度在哪个后端上真正快，取决于后端算子的支持情况，这条线留到第 8 章展开。

### 混合精度的一处实证：不同层量化到不同位宽

前面说「对数值范围敏感的层可能损失精度」是一个抽象论断。代码里有一处地方把它落到了具体的量化方案上。同一个文件往下几行，有一个 `FakeWeightsMode` 枚举（`runtime/executor/executor_settings_base.h:82`）：

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

它的结构是「头 + 分段」。头是一个 FlatBuffer，用类型化的键值对（`KeyValuePair`，`schema/core/litertlm_header_schema.fbs:56`）记录元数据。类型化是这里的要点：键是字符串，值是一个 `union`，可容纳十二种基本类型：

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

union 里每个成员都是一个各含单字段的 `table`（例如 `table UInt8 { value: ubyte; }`）。(1) 连字符串也包成 `StringValue`，于是 union 里存的是带类型标签的对象，读取方可以查询「这个值是什么类型」而不必自行推断。(2) `key` 与 `value` 都标了 `(required)`，FlatBuffer 在校验期强制它们非空，一个缺值的元数据项无法通过校验进入合法文件。相比直接嵌入一段无 schema 的 JSON，这套 schema 在编译期固定了元数据的结构：新增字段要修改 `.fbs` 并调整版本号。schema 头部注释写明了版本规则——纯粹追加新段构成 minor 版本变更，段的重排或删除必须提升 major 版本（`schema/core/litertlm_header_schema.fbs:69`）。这条规则把「向后兼容」写进了格式本身：老版本读取新文件时，只要没有段被重排或删除，就仍能按已知偏移定位到自己认识的段。

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

写文件的一侧用几个流类把不同来源的数据写入段：文件字节（`FileBackedSectionStream`，`schema/core/litertlm_section.h:98`）、protobuf（`ProtoBufSectionStream`，`:189`）、zlib 压缩流（`ZlibBackendedSectionStream`，`:252`）。这几个是写路径的构件，`FileBackedSectionStream::Prepare()` 把整个文件读进内存缓冲再输出。读路径不走这些流类，走的是下一节的 mmap。

<figure>
{{#include figs/fig-7-1.svg}}
<figcaption>图 7-1　.litertlm 文件结构：一个 FlatBuffer 头（类型化键值对元数据）+ 若干段。段可以是文件字节、protobuf 或 zlib 压缩流。权重、tokenizer、模板、能力声明打包进同一个文件。</figcaption>
</figure>

单文件加分段的设计服务于两件事：分发只需交付一个文件，以及下一节要讲的按需加载与并行加载。第 2 章介绍过的 `litertlm_print` 工具（`schema/core/litertlm_print.cc`）就是把这些段逐一列出来。本章末尾会走读它的实现，看它如何按类型分派打印各类元数据与段。

<div class="aside-compare">

同类问题的另一份答卷是 llama.cpp 的 GGUF：同样是单文件、键值元数据加张量数据、为 mmap 设计。一处对照很能说明「对齐为谁服务」：GGUF 的默认对齐是 32 字节（`GGUF_DEFAULT_ALIGNMENT`，`llama.cpp/ggml/include/gguf.h:46 @ b9873`），服务的是张量数据的访问对齐；`.litertlm` 的 16 KiB 对齐服务的是「每一段能独立按页 mmap」。粒度差了五百倍，因为二者优化的层不同：GGUF 把整个文件当一块映射、张量在其中寻址，`.litertlm` 要支持段级的按需映射与释放（本章 loader 一节）。LoRA 的挂法也成对照：llama.cpp 用运行时参数 `--lora` 挂独立适配器文件（`common/arg.cpp:2648`），与 LiteRT-LM 的独立 LoRA 文件加 `LoadLoRA`/`UseLoRA` 两拍装载思路相通，殊途同归。

</div>

## 加载：mmap、按需分页与并行

模型文件常有几 GB。如果加载时把整个文件读进内存，冷启动会等待很久。这是一个直接影响用户体验的冷启动延迟（cold-start latency）问题：从点击到模型可用之间的等待时间，与前面讨论的内存容量、带宽、功耗几类约束并列，是端侧另一项要优化的成本。

`.litertlm` 的分段结构在加载阶段再次发挥作用。读文件分两步：先读头，再按需读段。读头很廉价，`ReadHeaderFromLiteRTLM` 只读文件开头那一小段 FlatBuffer 元数据（`schema/core/litertlm_read.h:116`），取得每段的 `begin_offset`、`end_offset`、`data_type`，此时几 GB 的权重一个字节都没有读入。这个头结构禁止拷贝（移动仍放行，用于所有权转移），调用方持有它就持有了整份文件的段索引。省时间的关键在读段这一步。LiteRT-LM 有两条读段路径，先看用自研内存映射类的那条（`schema/core/litertlm_read.cc:238`）：

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

(1) 段的大小是两个 offset 之差，上一节 schema 里那对 `ulong` 在这里被消费。(2) `MemoryMappedFile::Create` 从 `begin_offset` 起映射 `model_size` 字节，正是上一节 16 KiB 对齐所保证的：这段能独立映射而不牵连邻段。映射建立时并不真正读盘，操作系统只是在页表里登记了这段虚拟地址到文件的对应关系，物理页要等 CPU 首次访问时经缺页中断逐页填入。(3) `BuildFromBuffer` 直接用映射出来的指针构建 TFLite 模型，没有把权重整体拷进堆内存的操作。冷启动省下的正是这次拷贝：几 GB 的权重不进读缓冲、不占堆，访问到哪一页才载入哪一页。接口注释点明了配套约定：调用方拿到的 `mapped_file` 持有这块 mmapped buffer，其生命周期必须延续到模型不再使用为止（`schema/core/litertlm_read.h:151`）。这个「生命周期交回调用方」的约定，正是下面要对比的两条读段路径的分野所在。

### mmap 背后：页对齐的硬断言与平台相反的预取策略

上一节说 16 KiB 对齐是为了让段能独立映射，这一节看它在运行时如何被强制。`MemoryMappedFile::Create` 的开头就是一个断言（`runtime/util/memory_mapped_file_posix.cc:101`）：

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

前面走读的 `ReadSectionIntoTFLiteMappedFile` 不是默认路径。同一个文件里还有一条更常用的 `ReadSectionIntoTFLite`（`schema/core/litertlm_read.cc:219`）：

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

为什么要有这个把生命周期交回调用方的变体？因为有些调用场景需要在模型之外长期持有这块 buffer，或需要对映射做模型层看不到的额外控制（例如统一记账、按段缓存）。默认权重路径走 `MMAPAllocation`，省去调用方的心智负担；需要更强所有权控制的路径走 `MemoryMappedFile` 变体。两条路径在 `ReadTFLiteFileFromSection` 的重载里按调用方是否传入 `mapped_file` 出参分派（`schema/core/litertlm_read.h:151`）。

### 并行加载到底并行了什么

分段结构常被认为「各段可以同时读盘」，实际实现要具体得多。这由一个开关控制（`litert_lm_engine_settings_set_parallel_file_section_loading`，`c/engine.h:295`，默认开）。这个 C-API 设置一路传到 `EngineSettings`，其成员默认为真（`runtime/engine/engine_settings.h:173`：`bool parallel_file_section_loading_ = true;`）。真正读取这个开关做分支的地方在加载流程里（`runtime/core/engine_advanced_impl.cc:249`）：

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

<figure>
{{#include figs/fig-7-2.svg}}
<figcaption>图 7-2　mmap 与并行加载：映射不搬字节、页按访问调入；tokenizer 构建与模型加载两件重活并行，缩短冷启动。</figcaption>
</figure>

除并行外，编译后的模型产物还能缓存到磁盘（`--cache disk`），二次启动直接复用、省去重新编译。按需分页、tokenizer 与模型并行、编译产物缓存，三者共同降低冷启动延迟。其中编译缓存的效果在本书基准里可见：GPU 后端首次运行（缓存未热）Init 5.29 s，其后稳定在约 1.77 s〔基准 D〕。并行加载开关开/关的冷启动差异本书未单独测量。

### loader 层：按段缓存、双检锁与对齐补偿

前面两条读段路径的更下面，还有一层本章尚未揭开的实现：`LitertLmLoader`（`runtime/util/litert_lm_loader.h:100`）。它在初始化时只记录每个段的 `(begin_offset, end_offset)` 位置表，真正的映射推迟到第一次有人要这个段的数据。取段入口 `GetSectionBuffer` 是一个标准的双检锁（`runtime/util/litert_lm_loader.cc:270`）：

```cpp
  {
    absl::ReaderMutexLock lock(section_buffers_mutex_);        // (1)
    auto section_buffer_it = section_buffers_.find(buffer_key);
    if (section_buffer_it != section_buffers_.end()) {
      return section_buffer_it->second;
    }
  }

  absl::MutexLock lock(section_buffers_mutex_);
  // Check again in case another thread has mapped it.
  auto section_buffer_it = section_buffers_.find(buffer_key);  // (2)
  if (section_buffer_it != section_buffers_.end()) {
    return section_buffer_it->second;
  }
  // ...
  absl::Status status = MapSection(buffer_key, offset_begin, offset_end); // (3)
```

(1) 快路径持读锁查缓存，段已映射过就直接返回，多个线程可以并发走这条路。(2) 未命中才升级为写锁，并且再查一次：上一节的并行加载意味着 tokenizer 线程与模型加载线程可能同时来要各自的段，第二次检查防止两个线程都没查到、各映射一遍。(3) 确认没人抢先后才真正 `MapSection`。每段只映射一次、按需映射、映射后共享，这一层把「分段」从文件布局变成了运行时行为。

`MapSection` 里还藏着一道跨平台的缝。`.litertlm` 的段按 16 KiB 对齐（本章前文），POSIX 的 mmap 要求偏移对齐到页（macOS 上 16 KiB，恰好整除），但 Windows 的映射偏移必须是分配粒度的整数倍，通常是 64 KiB（`runtime/util/memory_mapped_file_win.cc:95`）。16 KiB 对齐的段偏移未必是 64 KiB 的倍数，loader 的补偿是往前多映一段（`litert_lm_loader.cc:141`）：

```cpp
    size_t alignment = MemoryMappedFile::GetOffsetAlignment();
    uint64_t alignment_gap = begin_offset % alignment;          // (1)
    uint64_t aligned_begin_offset = begin_offset - alignment_gap;
    // ...
    data = static_cast<uint8_t*>(memory_mapped_file->data()) + alignment_gap; // (2)
```

(1) 算出段偏移与平台对齐粒度的差，把映射起点回退到对齐边界；(2) 返回给上层的指针再前移同样的差值，指向真正的段数据。多映的那一小段（至多一个对齐粒度）只是地址空间的浪费，不产生额外读盘。文件格式定一个对齐值、各平台各有一个对齐值，这道缝总要有人缝，缝在 loader 层比缝在文件格式里便宜：文件不必为最挑剔的平台把对齐提到 64 KiB。

### 一笔内存账：mmap 之下，「占了多少内存」怎么读

mmap 让「模型占多少内存」这个问题变得需要口径。CLI 在退出时打印的内存报告就分了好几行（`LogMemoryUsage`，`runtime/engine/litert_lm_lib.cc:427`）：峰值系统内存、physical footprint（物理驻留）、非 mmap 的堆分配总量、in-use 堆、private footprint（私有驻留）。区分的原因正是 mmap：3.4 GB 的权重文件映射进地址空间后，虚拟内存立即增加 3.4 GB，但物理内存只在页被真正访问后才占用，且这些页是文件后备的，内存紧张时操作系统可以直接丢弃、下次访问再从文件读回，不占交换空间。于是「模型放得下吗」（第 1 章内存容量约束）在 mmap 语义下的精确问法是：**私有驻留（权重之外的堆、KV cache、激活）加上权重的常驻工作集，是否放得进物理内存**。decode 每步都要读全部权重，权重的工作集就是全量，mmap 省不掉这部分物理占用，省掉的是加载时的一次性拷贝和内存紧张时的换出成本。第 1 章的内存预算账按这个口径读才准确。

## LoRA：不动基座，换个人格

最后一块拼图：变体。你有一个通用基座模型，想让它在某个专门任务上更好——写代码、医疗问答、特定语气。重新训练或全量微调一个几 GB 的模型，端侧存不下也换不起。

LoRA 的思路是：**基座权重一个字节都不动，另外挂一小份"增量权重"**。推理时把增量叠加到基座上，模型行为就偏向新任务。增量很小（相比基座是零头），存得下、也能热加载。LiteRT-LM 用两个类支撑它：`LoRA`（`runtime/components/lora.h:40`）持有一份增量权重的后端资源，`LoraManager`（`runtime/components/lora_manager.h:39`）按 id 管理多份。多份靠两张以 id 为键的表并存（`:75`–`:76`）：

```cpp
absl::flat_hash_map<uint32_t, std::unique_ptr<LoraData>> lora_data_;   // (1)
absl::flat_hash_map<uint32_t, std::unique_ptr<LoRA>> loras_;           // (2)
```

(1) 存"从磁盘读进来的原始 LoRA 权重"，(2) 存"在后端上建好的 LoRA 对象"。分成两张表，是因为加载被拆成了两拍。`LoadLoRA` 只做第一拍（`runtime/components/lora_manager.cc:46`）：

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

「增量是零头」可以算出来。LoRA 对一个 `d_in × d_out` 的投影矩阵不训练全量增量，而是训练两个窄矩阵 A（`d_in × r`）与 B（`r × d_out`），r 就是 rank（秩），增量参数量为 `r × (d_in + d_out)`，与全量的比值约为 `2r / d`（方阵情形，教科书级结论）。代入 r = 16、d = 2048，比值约 1.6%：几 GB 的基座，增量在几十 MB 量级。真实的 r 不用猜，从 LoRA 文件自己的元数据读出（`LoraData::GetLoRARank`，`runtime/util/lora_data.h:54`）；`LoraData` 的类注释也点明它「以最小拷贝方式读取（如 mmap）、以只读视图提供数据」（`:28`），与主模型的加载哲学一致。

第二拍在后端上建资源的动作，具体是一次逐张量的拷贝（`LoRA::Init`，`runtime/components/lora.cc:70`）：

```cpp
  for (const auto& input_name : input_names) {
    if (!IsLoRAInputName(input_name)) {
      continue;                                                  // (1)
    }
    // Create the input buffer for the LoRA tensor.
    LITERT_ASSIGN_OR_RETURN(
        litert::TensorBuffer tensor_buffer,
        compiled_model_.CreateInputBuffer(signature_name_, input_name));
    // ...
    if (lora_data_->HasTensor(input_name)) {
      // ...
      RET_CHECK_EQ(tensor_buffer_size, lora_tensor_data->Size())  // (2)
          << "LoRA tensor size mismatch between model input and Lora Data: "
          << tensor_buffer_size << " vs. " << lora_tensor_data->Size();
      std::memcpy(lock_and_addr.second, lora_tensor_data->Data(),
                  lora_tensor_data->Size());                      // (3)
    } else {
      // Fill the buffer with zeros if the tensor is not in LoraData.
      std::memset(lock_and_addr.second, 0, tensor_buffer_size);   // (4)
    }
```

(1) 只处理名字标记为 LoRA 输入的张量：支持 LoRA 的模型在编译期就为增量权重预留了具名输入位，运行时按名对号入座。(2) 尺寸必须与模型预留的输入位完全一致，不一致直接报错，这里没有任何形状适配。(3) 增量权重从 `LoraData` 的只读视图逐字节拷进后端 buffer，「占显存」发生在这一行。(4) 是一处安静的兜底：LoRA 文件里缺某个张量时填零，而零增量恰好等价于「这一层不加改动」，模型退回基座行为，不会崩也不会错。加载完成后，取用侧的 `GetLoRABuffers` 用 `Duplicate()` 发出的是引用计数增量而非再拷贝一份（`lora.cc:117`），多处使用同一份 LoRA 不叠加显存。

LoRA 的价值恰好呼应本章主题：它是"变体"的最省成本形态——一个基座 + 若干小增量，就能覆盖多个任务，而不必为每个任务存一个完整模型。放到端侧的存储约束下，这个省法尤其值钱。

## litertlm_print：把格式知识变成一次实剖

本章开头承诺走读 `litertlm_print` 的实现，现在格式的各个部件都讲过了，正好收尾。这个工具做的事只有两件：打印头部的键值元数据，再逐段打印段目录。第一件事落在 `PrintKeyValuePair`（`schema/core/litertlm_print.cc:61`），它就是对本章前文那个 `VData` union 的 tag 逐类分派：

```cpp
  switch (kvp->value_type()) {
    case VData::VData_StringValue: {
      output_stream << ANSI_BOLD << "Value" << ANSI_RESET << " (String): "
                    << kvp->value_as_StringValue()->value()->c_str() << "\n";
      break;
    }
    case VData::VData_Int32: {
      output_stream << ANSI_BOLD << "Value" << ANSI_RESET
                    << " (Int32): " << kvp->value_as_Int32()->value() << "\n";
      break;
    }
    // ... Float32 / Bool / UInt64 同构分支 ...
```

每个分支用 `value_as_<类型>()` 取出对应的 table 再读 `value()`。前文说 schema 把元数据的类型「钉死在编译期」，这个 switch 是它在读取侧的镜像：能打印的类型就是 union 里声明过的那几种，多一种都编译不出来。

第二件事是段目录遍历（`litertlm_print.cc:156`）：对每个 `SectionObject` 打印它的键值项、`begin_offset` 与 `end_offset`（本章 16 KiB 对齐一节里那对偏移）、段类型名。其中有一个特判：段类型是 `LlmMetadataProto` 时，不满足于打印偏移，而是当场调 `ReadLlmMetadataFromSection` 把这段解析成 proto、以 `DebugString` 逐行打出（`:181`）。所以第 2 章看到的那份 dump 里，别的段只有一行类型加一对偏移，唯独元数据段展开成了几十行的 start_token、stop_tokens、聊天模板：不是格式对它特殊，是打印工具对它多走了一步解析。

读懂这个工具的意义在于验证：拿到任何一个 `.litertlm` 文件，`litertlm_print` 的每一行输出现在都能对回本章的某一节——键值对回到 `VData` union，偏移回到 16 KiB 对齐，段类型回到 `AnySectionDataType` 枚举，元数据段回到 proto。格式的每一项知识由此都有了可动手核对的出口。

## 小结

这一章讲的是模型的三种"形态变化"：量化把它压小（省体积、更省带宽，间接提 decode 速度），`.litertlm` 把它连同 tokenizer、模板、能力声明装进一个可 mmap、可分段并行加载的文件，LoRA 用小增量让它在不动基座的前提下适配新任务。三者共同回答一个问题：一个几十亿参数的模型，怎么以端侧扛得住的形态存在。

模型的形态清楚了。下一章回到运行时：同一个模型，为什么在 CPU、GPU、NPU 上跑起来速度甚至输出都不一样。

---

## 练习与自查

1. **对齐计算。** 某段 `begin_offset = 49152`。在偏移须为 64 KiB 整数倍的平台上，`MapSection` 的 `alignment_gap` 是多少？实际映射从哪里开始？
2. **LoRA 账。** rank = 16、`d_in = d_out = 2560`，一个投影矩阵的 LoRA 增量参数是多少？fp16 存储占多少字节？
3. **口径辨析。** mmap 加载后，进程的虚拟内存增加了 3.66 GB，但物理驻留远小于此。解释两者差异，以及内存紧张时这些页的去向。
4. **工具理解。** `litertlm_print` 对 `LlmMetadataProto` 段比对其他段多做了什么？为什么只对它多做？
5. **两拍装载。** `LoadLoRA` 与 `UseLoRA` 各占用什么资源？为什么切换任务时不必重新读盘？


<!-- litertlm_print 实剖已完成（附录 D 第六节，自研扫描替代）。仍开放：int4 vs int8 三角（无同模型两种量化产物）、并行加载 on/off 冷启动（CLI 未暴露开关）；量化内部（分组/scale）未展开，仅到"权重压 4bit + 激活精度谱系"层面。图 7-2、表 7-1 未出；weight cache 章卡项未兑现。2026-07-16 评审修订：图 7-1 底注已改为与正文一致的「tokenizer 与模型加载可并行」。 -->
