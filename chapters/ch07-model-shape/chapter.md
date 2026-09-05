# 第 7 章 模型文件与权重：量化、容器格式与 LoRA

> 本章区分低比特表示的理论收益与实测收益，说明 `.litertlm` 的文件布局与主加载路径，再分析 weight cache 与 LoRA 的资源生命周期；这些口径也是第 8 章讨论后端内存的基础。

第 6 章讨论随会话推进而变化的 KV 状态；本章转向保存计算图与权重的模型文件。Engine 加载文件后建立运行资源，资源由 Engine 及其内部对象管理。Engine 可以在进程结束前销毁，磁盘上的模型文件则可供后续 Engine 重新加载。

围绕这个文件有三个问题要分开分析。第一个是量化：权重从 FP16 改成 INT4，理想权重大小降至四分之一，主存流量与计算开销会如何变化（7.1 节，对应第 2 章表 2-2 的问题 15）。第二个是容器：`.litertlm` 单文件包含哪些分段，运行时怎样按需取出，weight cache 与 mmap 各自意味着什么（7.2 至 7.3 节，问题 16）。第三个是 LoRA：增量权重与基座权重分开保存后，运行时实际保留了多少资源（7.4 节）。7.5 与 7.6 节给出检查工具与一个诊断案例，7.7 节把这些检查组织成一条可回滚的部署流程。

## 7.1　量化收益取决于表示、算子与瓶颈

权重大小可以直接计算。同一组参数以 FP16 保存时每个占 2 字节，改成紧密打包的 INT4 后理想占用是 0.5 字节，比例为 4。这个比例只算权重本身：分组 scale、zero point 和对齐填充不在其中，文件头与 tokenizer 也不在其中，所以它不能直接用于整个 `.litertlm` 文件。

权重读取流量能否按位宽比下降，取决于读取次数和主存中的实际表示。若每步读取次数相同，且读取时仍保持紧密打包的 INT4，理想权重字节数可降至 FP16 的四分之一。若启动时已展开成更高位宽，decode 就不再按文件中的 INT4 位宽读取权重。

流量减少能否转化为吞吐收益，还要看运行瓶颈。decode 主要受权重带宽限制时，减少这项流量才可能带来接近位宽比的加速（1.4 节）。即使受计算限制，权重读取字节数也可能下降，但吞吐未必同比提高。KV cache、激活、量化参数与调度开销也没有按位宽比同步缩小，端到端吞吐仍须测量。

计算量的变化没有固定方向。原生低比特矩阵 kernel 可能同时减少运算和访存，动态反量化（计算前把 INT4 还原成可运算数值）则会增加指令。实际结果取决于算子融合、向量指令、线程划分和后端实现。

质量是另一项独立指标。INT8 或 INT4 的位宽本身不能限定质量损失，评估时必须给出模型、量化方法、校准数据与评测集。本书没有同一模型的 FP16、INT8、INT4 对照实验，因此既不报告质量收益，也不把质量损失分成“低”或“可接受”。

| 表示变化 | 可直接计算的结论 | 仍需确认的条件 | 本章不据此断言 |
|---|---|---|---|
| FP16 → INT8 | 不计额外元数据时，权重大小减半 | 打包格式、后端是否保持 INT8、运行瓶颈 | decode 吞吐提高 2 倍、质量基本不变 |
| FP16 → INT4 | 不计额外元数据时，权重大小降至四分之一 | scale/zero point、对齐、低比特算子与设备支持 | decode 吞吐提高 4 倍、质量风险可接受 |
| 激活类型变化 | 单个标量的表示宽度随类型改变 | 图中实际张量类型、后端支持、转换开销 | 峰值内存和吞吐按位宽同比变化 |

> 表 7-1　低比特表示能直接确定的是理想权重大小比例；端到端内存、速度与质量都需要在完整条件下测量。

### 7.1.1　权重位宽与激活类型是两项独立配置

运行时设置里的激活类型（activation type，前向计算中各层中间张量的数据类型）由一个枚举表示，可选 FLOAT32、FLOAT16、INT16、INT8 四个值。这个枚举只说明设置能区分这四种类型；它不保证每个后端都按该类型保存所有中间张量，也不说明转换成本和质量。

激活设置到编译选项的映射随后端而变。GPU 编译选项只分两档：激活类型为 FLOAT32 时用 FP32 精度，其余三个值都按 FP16 精度编译。

权重量化也不能从激活枚举推断。设置里另有一个合成权重模式（fake weights）枚举，声明了两种测试配置：所有层用 INT8，或者注意力用 INT8、FFN 与 embedding 用 INT4。它只用来生成待测的位宽组合，没有给出真实模型的分组方式、scale、校准过程或质量结果；枚举名也不能证明真实 `.litertlm` 文件采用同一策略。

### 7.1.2　从低比特文件到低比特 kernel 要经过四层

判断一个 INT4 模型能否在目标设备上运行，至少要区分四层。第一层是文件中的权重表示，包括位宽、分组、scale 和对齐。第二层是 TFLite 图如何描述这些常量与算子。第三层是编译时选定的后端及其选项。第四层才是后端生成或选择的 kernel，以及它是否保留压缩权重、何时转换布局。

这四层不能互相替代。文件中存在 INT4 常量，不代表图中的每个矩阵乘都能由同一个低比特算子处理。图能表达某种量化形式，也不代表 CPU、GPU 与 NPU 都接受它。后端完成编译后，权重还可能转换成设备相关布局；此时文件大小不再等于运行时权重缓冲大小。

GPU 编译选项提供了两个可观察的例子：`convert_weights_on_gpu` 控制 OpenCL 与 WebGPU 路径是否在 GPU 上转换权重，其他后端忽略该开关；`allow_src_quantized_fc_conv_ops` 决定是否允许源量化的 FC/Conv 算子，默认允许。两个开关说明“量化权重存在”和“量化算子被采用”是两个判断，源码也没有给出它们在每款 GPU 上对应的 kernel 清单。CPU 路径则配置 XNNPACK 的 weight cache、动态 fully connected 标志和量化 zero point 压缩，再把硬件加速器设为 CPU；这里同样没有一个通用的“INT4 已启用”布尔值，能否编译由模型图、LiteRT、XNNPACK 版本和目标 CPU 共同决定。

<figure>
{{#include figs/fig-7-1.svg}}
<figcaption>图 7-1　权重位宽只有依次通过图表示、后端编译和 kernel 支持，才可能转化为运行时收益。</figcaption>
</figure>

| 检查层 | 可检查对象 | 通过后能说明什么 | 仍不能说明什么 |
|---|---|---|---|
| 文件表示 | 权重张量类型、分组参数、scale、外挂权重段 | 模型产物保存了低比特数据 | 后端会保持该布局 |
| 图与算子 | TFLite tensor、operator、signature | 图能够表达所需计算和输入输出 | 目标 delegate 支持全部节点 |
| 编译选项 | backend、激活类型、量化 FC/Conv 与权重转换开关 | 运行时向选定后端提出了对应要求 | 编译器一定选择原生低比特 kernel |
| 编译结果 | delegate 日志、生成的 cache、节点的执行后端 | 当前设备完成了这次编译或加载 | 吞吐、质量和峰值内存达到目标 |
| 受控实测 | 模型、设备、上下文、cache 状态一致的对照 | 给定条件下的端到端差异 | 差异可无条件推广到其他设备 |

> 表 7-2　量化部署需要逐层取证；任何一层的信息都不足以单独证明端到端收益。

部署时，先检查模型产物和后端约束，再做受控实测。若编译阶段已经报不支持的算子，继续比较 tokens/s 没有意义。若编译成功但吞吐没有变化，应先检查权重是否被展开及工作负载是否受带宽约束；反量化或同步也可能抵消流量收益。

### 7.1.3　可复算案例：2B 参数模型的部署容量

这组部署预算不对应某个已发布模型。假定模型共有 20 亿个参数，其中 90% 使用 INT4，10% 使用 INT8。INT4 部分每 32 个权重共用一个 FP16 scale，不使用 zero point。TFLite 图、tokenizer、元数据等其余内容合计 80 MiB，容器共有 5 个 section。

INT4 权重大小为：

$$
1.8\times 10^9\times 0.5\ \mathrm{B}=900{,}000{,}000\ \mathrm{B}.
$$

INT8 权重大小为：

$$
0.2\times 10^9\times 1\ \mathrm{B}=200{,}000{,}000\ \mathrm{B}.
$$

INT4 分组共有 \\(1.8\times 10^9/32=56{,}250{,}000\\) 组，每个 FP16 scale 占 2 字节，所以 scale 共 112,500,000 B。加上 80 MiB 的其余内容，文件在对齐填充前约为：

$$
900{,}000{,}000+200{,}000{,}000+112{,}500{,}000+80\times 2^{20}
=1{,}296{,}386{,}080\ \mathrm{B}\approx1.21\ \mathrm{GiB}.
$$

打包工具把每个 section 的起点向上对齐到 16 KiB。5 个 section 在本题中最多引入不足 \\(5\times16\ \mathrm{KiB}=80\ \mathrm{KiB}\\) 的起点填充，不会改变 1.21 GiB 的两位小数结果。这里没有计入文件系统块、签名包或应用安装包的额外开销。

文件能写入存储空间，不等于进程内存足够。再假定模型有 28 层、8 个 KV 头、每头维度 128，KV cache 使用 FP16，最大上下文为 4096。按 6.1 节的公式，每个 token 的 KV 为 \\(2\times28\times8\times128\times2\ \mathrm{B}=112\\) KiB，单会话 4096 个 token 为：

$$
2\times28\times8\times128\times4096\times2\ \mathrm{B}
=469{,}762{,}048\ \mathrm{B}=448\ \mathrm{MiB}.
$$

若激活、后端工作区和普通堆合计预留 0.40 GiB，再留 0.25 GiB 安全余量，且驻留权重接近 1.21 GiB，则预算约为 \\(1.21+0.44+0.40+0.25=2.30\\) GiB。假定系统只能稳定提供 3.0 GiB，这个口径尚能容纳。若后端编译时另建一份接近 1.21 GiB 的转换权重，预算会升至约 3.51 GiB，超过题设上限。

这个案例解释了为什么部署评审不能只写“模型文件 1.21 GiB”。需要同时确认外挂权重是否映射、后端是否复制或转换权重、KV cache 上限，以及冷启动阶段是否出现旧布局和新布局并存。各项实际值仍要用目标设备测量。

## 7.2　`.litertlm`：头部记录 section 目录

一个 `.litertlm` 文件由若干 section 组成，每个 section 有类型和字节范围。类型枚举除去“未设置”与“已废弃”两个占位值，实际有六种：TFLite 模型、外挂 TFLite 权重（与模型段配对使用）、SentencePiece tokenizer、HuggingFace tokenizer（JSON 配置经 zlib 压缩）、`LlmMetadataProto` 与通用二进制。也就是说，端侧运行所需的数据不只有权重，tokenizer 与运行参数也在同一个文件里。

`LlmMetadata` 这个 protobuf 保存起始 token、停止 token、prompt 模板和默认采样参数，模型族类型也在同一消息内。它们不是各自独立的 section，而是一个段里的字段。

文件头用 FlatBuffer 二进制格式描述这些 section。头部的键值对里，键是字符串，值是带类型标签的 `VData` union，允许 12 种标量或字符串包装类型。这个设计固定了值的表示类型，但没有固定键名集合：增加一个字符串键通常不需要改 schema，增加 union 值类型或 section 类型才需要。schema 注释还规定了版本演进约定：新增 section 类型提升 minor 版本，重排或删除已有类型提升 major 版本；它没有声称所有 minor 版本都能由旧读取器无条件处理。

段目录由一组 `SectionObject` 给出，每项记录可选的键值属性、`begin_offset`、`end_offset` 和 `data_type`，数据范围是 `[begin_offset, end_offset)`。注释规定下一段不得早于按 16 KiB 计算的对齐边界。schema 里标为 `(required)` 的字段是结构约束，读取路径并不据此对损坏文件做完整验证，这一点留到 7.7.2 节展开。

### 7.2.1　固定前缀与两阶段写入

当前 Python 打包工具生成的格式版本是 1.5.0。文件开头 8 字节是 `LITERTLM`，随后依次是 4 字节的 major、minor、patch；字节 20-23 是填充，字节 24-31 保存 FlatBuffer 头的结束偏移，头部元数据从字节 32 开始。section 的起点按 16 KiB 对齐。

打包工具无法在写 section 之前一次性算出所有结束偏移，所以分两步。先用非零占位值打包头部，得到头部长度；再从对齐后的第一个 section 起点开始写数据，逐段回填起止偏移；全部写完后回到文件开头，写入魔数、版本、头部结束偏移和更新后的 FlatBuffer。这要求第二次打包的元数据长度与第一次相同，代码用断言检查这一条件。偏移值是固定宽度的 `ulong`，从占位值改成真实值不会改变字段宽度，前提才成立；若以后把可变长内容加入回填过程，就要重新检查这一前提。

TFLite 模型段的属性里有三项会影响加载行为。`model_type` 区分主干、视觉、音频等资源角色；`backend_constraint` 声明允许的后端集合；`prefer_activation_type` 给出首选激活类型。打包工具禁止覆盖前两项，并把后端与激活字符串转成小写。加载器读取时对键名不区分大小写；缺少 `model_type` 的旧文件回退为主干模型；后端约束与激活提示则存入 section 提示表，交给 Engine：创建 executor 前先检查后端约束，调用者没有显式指定激活类型时采用 `prefer_activation_type`。

两者语义不同。`backend_constraint` 是硬约束，不包含当前后端时 Engine 返回 `InvalidArgumentError`；`prefer_activation_type` 只是默认选择，不会覆盖调用者已经设置的激活类型。排查初始化失败时，先分清这两种。

<figure>
{{#include figs/fig-7-2.svg}}
<figcaption>图 7-2　.litertlm 的头部记录 section 类型与字节范围；prompt 模板和停止 token 等运行参数位于 LlmMetadataProto 段内。</figcaption>
</figure>

<div class="aside-compare">

llama.cpp 的 GGUF 也在单文件中保存元数据与张量数据。它的默认数据对齐是 32 字节[^ch07-llamacpp-gguf]；`.litertlm` 的 section 布局使用 16 KiB 边界。二者的对齐对象和粒度不同，不能只凭数值判断格式优劣。

两个项目都允许从独立文件接入 LoRA。llama.cpp 暴露 `--lora` 参数[^ch07-llamacpp-lora]；LiteRT-LM 由 `LoadLoRA` 和 `UseLoRA` 分别处理数据登记与后端资源创建。

</div>

## 7.3　当前加载路径：`LitertLmLoader` 按请求取段

Engine 创建时按文件格式选择资源构建方式。`.litertlm` 格式使用 `LitertLmLoader`：它解析头部、记录每个 section 的位置，再被包装成模型资源对象，供 executor 与 tokenizer 按需取段。这是本章所说的主加载路径。schema 工具层另有按 section 提取 TFLite 模型的辅助函数，一个用 TFLite 的 `MMAPAllocation`，另一个返回独立的映射句柄；Engine 主路径不经过它们，所以不能把其中任一称为“默认权重路径”。

### 7.3.1　只在初始化时建立 section 索引

初始化时 loader 最多读取文件开头 16 KiB 作为头部：来源是文件句柄（`ScopedFile`）时为这段建立映射，来源已经是整文件的 `MemoryMappedFile` 时直接用基址。解析后它遍历 section，记录每个 `BufferKey` 对应的偏移范围，但不提前建立各 section 的缓冲视图。调用方请求模型、tokenizer 或元数据时，`GetSectionBuffer` 才处理相应范围：

```cpp
// runtime/util/litert_lm_loader.cc:272-295
  {
    absl::ReaderMutexLock lock(section_buffers_mutex_);
    auto section_buffer_it = section_buffers_.find(buffer_key);
    if (section_buffer_it != section_buffers_.end()) {
      return section_buffer_it->second;
    }
  }

  absl::MutexLock lock(section_buffers_mutex_);
  // Check again in case another thread has mapped it.
  auto section_buffer_it = section_buffers_.find(buffer_key);
  if (section_buffer_it != section_buffers_.end()) {
    return section_buffer_it->second;
  }
  // ...
  absl::Status status = MapSection(buffer_key, offset_begin, offset_end);
```

第一次检查持读锁，已有的 section 缓冲直接返回；未命中再取写锁并复查，避免两个线程重复映射同一 section。取得的缓冲视图留在表中复用。这里的“按需”指 section 视图在第一次请求时建立：来源是文件句柄时包含一次 section 映射，整文件已映射时只按基址加偏移建立视图。它不保证物理页只在首次访问时才读盘。

### 7.3.2　对齐补偿与平台页建议

`MapSection` 先判断模型来源：整文件已映射时直接用基址加 `begin_offset`，只有文件句柄分支才为 section 新建映射。新建映射有一个平台约束：`MemoryMappedFile::Create` 要求映射偏移是平台对齐值的整数倍，POSIX 取 `getpagesize()`，Windows 取系统的 allocation granularity，而 16 KiB 的 section 边界不一定满足它。于是文件句柄分支先把起点向前调整到平台对齐边界，映射后再把指针前移 `alignment_gap`：

```cpp
// runtime/util/litert_lm_loader.cc:141-154
    size_t alignment = MemoryMappedFile::GetOffsetAlignment();
    uint64_t alignment_gap = begin_offset % alignment;
    uint64_t aligned_begin_offset = begin_offset - alignment_gap;
    // ...
    data = static_cast<uint8_t*>(memory_mapped_file->data()) + alignment_gap;
```

这段补偿属于当前 loader，不是文件格式的一部分。向前多映射的范围扩大了虚拟映射区，哪些页真正进入物理内存仍由后续访问与操作系统策略决定。

POSIX 实现用 `MAP_PRIVATE` 建立映射，随后给内核一条页使用建议：Apple 平台调用 `MADV_DONTNEED`，其他 POSIX 平台调用 `MADV_WILLNEED`。`madvise` 只是建议，既不保证同步读盘完成，也不保证禁止预读。所以不能笼统地说“mmap 后一律不预读”：两类平台的代码给出了相反的建议。

### 7.3.3　并行的是 tokenizer 创建与模型加载

设置项 `parallel_file_section_loading` 默认为真，名字容易让人以为所有 section 会同时读盘，实际范围窄得多。对带模型族类型的新格式文件，开关为真时 Engine 用 `std::launch::async` 在另一个线程创建 tokenizer，当前线程继续设置模型并创建 executor；开关为假时改用 `std::launch::deferred`，等到需要时在当前线程创建。旧格式文件在进入这个分支前就同步创建了 tokenizer。

<figure>
{{#include figs/fig-7-3.svg}}
<figcaption>图 7-3　当前加载链路先建立 section 索引，再按请求映射 section；新格式模型可让 tokenizer 创建与后续模型加载重叠，实际冷启动差异需单独测量。</figcaption>
</figure>

本书没有对这个开关做对照实验，因而不报告并行加载的毫秒收益。附录 D 有一组相关记录：Apple M5 Pro、Gemma 4 E4B、GPU（Metal）后端、CLI 使用 `--cache disk`，同一批次里首次 Init 聚合值为 5.29 s，后续约为 1.77 s。这个 Init 值不是端到端墙钟时间：C API 遍历初始化阶段表，把各阶段时长相加，而总阶段与其中的模型资源、元数据子阶段在时间上重叠。这组数据只能说明同一口径下的批内变化，既没有 cache 开关对照，也没有外部墙钟，不能单独量化编译缓存或 tokenizer 并行的贡献。

### 7.3.4　weight cache 的标识不是内容哈希

CPU 与 GPU 后端在编译阶段会用到派生的缓存文件：CPU 是 XNNPACK 的 weight cache，GPU 是 MlDrift 的 program cache 与 weight cache。它们不是 `.litertlm` 中某个 section 的原样副本，而是后端根据模型生成的派生物。LiteRT-LM 只负责派生路径或接受调用者提供的文件描述符，再把配置交给后端；入口模式与责任划分见 7.7.3 节。

主文本 executor 的创建顺序是：先构造带 cache 参数的编译选项；若 `.litertlm` 有独立的外挂权重段，再把该段的偏移与长度作为外部权重交给 GPU，非 GPU 后端遇到外挂权重段会直接返回错误；最后才调用 `CompiledModel::Create`。外挂权重、weight cache 与 compiled model 是三个对象，分别属于模型输入产物、后端派生物和本进程中的可执行对象。CPU 和 GPU 的 cache 也不能互换，删除或禁用其中一种只能隔离对应后端的初始化变量。

缓存名里的模型标识来自 `GetFileCacheIdentifier`。路径版本读取文件的最后修改时间（秒）和字节数，拼成 `<mtime_seconds>_<size>`；文件描述符版本同样返回修改时间与大小；头文件注释写的也是“timestamp + file size”。结果还会在进程内按路径缓存，同一进程第一次查询后就复用。

这不是内容哈希。大小或秒级修改时间变了，派生路径通常跟着变；内容变了但大小与时间戳相同，标识不变。自动清理旧 cache 只在路径模式下执行，且要同时满足三个条件：调用方启用了检查，按当前标识派生出的文件不存在，目录可遍历。清理函数按模型文件名与后缀匹配旧文件后直接删除，没有进程间锁。两个盲区由此产生：模型内容改变但标识没变时，当前 cache 路径仍存在，清理不会运行；模型在同一进程内被原位替换时，即使新文件的 mtime 或大小不同，进程内保存的旧标识也可能继续被使用。文件描述符模式绕过自动清理，调用者自行管理旧文件。

<figure>
{{#include figs/fig-7-4.svg}}
<figcaption>图 7-4　weight cache 由模型路径、秒级 mtime 和文件大小派生；进程内保存的旧标识与同大小同时间替换构成失效盲区。</figcaption>
</figure>

| 变化 | 新进程中的派生标识 | 同一进程已查询过该路径 | 自动清理旧 cache |
|---|---|---|---|
| 文件大小改变 | 通常改变 | 仍可能沿用旧标识 | 新路径不存在时尝试清理 |
| mtime 跨秒改变 | 通常改变 | 仍可能沿用旧标识 | 新路径不存在时尝试清理 |
| 内容改变，大小与秒级 mtime 相同 | 不变 | 不变 | 不触发 |
| 文件改名，内容不变 | 标识的数值部分可相同，basename 改变 | 作为新路径重新查询 | 在新 basename 范围内判断 |
| `cache_dir = :nocache` | 不派生可用 cache | 不适用 | 不执行 cache 清理 |

> 表 7-3　weight cache 的失效依据是路径元数据和进程状态，不是模型内容摘要。

部署系统若会原位更新模型，较稳妥的做法是使用不可变版本文件名，并在更新后创建新进程。若必须复用路径，应由应用在替换时显式管理对应 cache，不能依赖同大小文件自动失效。这里的结论针对当前的命名与清理逻辑，不代表底层 cache 格式在不同 LiteRT 版本之间兼容。

### 7.3.5　mmap 下的内存口径

mmap 先增加虚拟地址映射，物理驻留随后受页面访问、`madvise` 和内核回收策略影响。它可以避免把整个模型再复制到一块普通堆缓冲，但不会自动缩小推理阶段需要访问的权重工作集。对稠密模型而言，每个 decode step 通常会访问大部分权重；实际驻留规模还取决于后端预打包、缓存和系统内存压力。

LiteRT-LM 的内存日志本身就提供多种口径：peak system RAM、physical footprint、非 mmap 堆、in-use heap 和 private footprint。判断模型能否运行时，应同时看私有内存、KV cache、激活、后端工作区和文件映射的驻留工作集。虚拟映射大小不是峰值物理内存，二者之差也不能全部视为节省量。

## 7.4　LoRA：基座权重与增量权重分离

LoRA 用两个低秩矩阵表示某个线性层的权重增量。对于 \\(d_{in} \times d_{out}\\) 的原矩阵，秩（rank）为 \\(r\\) 时，两个矩阵共有 \\(r \times (d_{in} + d_{out})\\) 个参数。若 \\(d_{in} = d_{out} = d\\)，相对全量矩阵的参数比例是 \\(2r/d\\)。代入 \\(r = 16\\)、\\(d = 2048\\)，比例约为 1.6%。

这个数字只描述单个方阵投影。完整适配器与基座模型的大小比还取决于覆盖哪些层、保存类型和文件元数据，不能直接写成 1.6%。

### 7.4.1　离线合并与运行时适配是两条路径

设 LoRA 的增量为 \\(\Delta W=sBA\\)，其中 \\(A\\) 与 \\(B\\) 是低秩矩阵，\\(s\\) 是训练或导出约定的缩放系数。离线合并先计算 \\(W^{\prime}=W+\Delta W\\)，再把 \\(W^{\prime}\\) 量化并导出为新的模型产物。运行时适配则保留基座权重不变，让模型图把 LoRA 张量作为额外输入参与计算。

离线合并不需要运行时切换接口，但每个适配器都会产生一份新的完整模型。若基座权重已经量化，先合并还是先量化会改变数值结果，不能把 FP16 合并后的差分直接等同于量化模型上的差分。运行时适配只分发增量文件，却要求基座模型的 signature 预先暴露匹配的 LoRA 输入，后端还要能为这些输入创建 buffer。

LiteRT-LM 仓库里能追踪到的 LoRA 组件都属于运行时适配路径，而且当前只有音频编码器调用了它们。主文本 executor 能识别 LoRA 输入名并跳过这些输入的普通缓冲创建，但没有创建或调用 `LoraManager` 的代码；上层资源管理收到文本侧的 LoRA 文件时直接返回 `Lora is not supported.`。仓库内能完整追踪登记、切换与执行的调用点只在音频编码器。所以不能仅凭主文本图中存在 LoRA 输入，就断定文本生成 API 已经支持热切换。仓库里也没有由这些组件执行 \\(W^{\prime}=W+\Delta W\\) 原位合并的路径；若产品选择离线合并，应把它当作模型导出流程，量化、后端约束、cache 标识和质量都要重新验证，不能把 `LoadLoRA` 当作合并工具。

运行时适配由三个类分工。`LoraData` 是 CPU 侧的只读数据视图：适配器文件本身也是 TFLite FlatBuffer，它从名为 `lora_rank` 的 metadata 读取 rank，按 tensor 名找到相应 buffer；来源可以是路径、文件句柄或已有缓冲，路径分支用 mmap 提供视图。`LoRA` 对象持有为一份适配器创建的后端 buffer。`LoraManager` 用两张表管理它们：`lora_data_` 保存已登记但尚未创建后端对象的 ID，`loras_` 保存已创建的对象，`current_lora_id_` 记录当前选用哪一份。`LoadLoRA` 只建立 `LoraData` 并加入待用表；第一次 `UseLoRA(id)` 才把数据移入 `LoRA::Create`，把对象写入 `loras_`，删除待用表中的同一项，再更新当前 ID。

`LoRA::Init` 遍历基座 signature 的输入，为每个 LoRA 输入创建 `TensorBuffer`：适配器里有同名 tensor 时检查字节数并复制，没有时把整个 buffer 置零。buffer 的实际存放位置由编译模型和后端决定，不能统一称为 GPU 显存。名称和尺寸共同构成兼容边界：`IsLoRAInputName` 只接受两组固定命名模式，并要求名称以层号结尾；基座需要的输入在适配器里缺失时填零，同名但字节数不一致时初始化失败，适配器多出的 tensor 不会被基座 signature 请求。所以只比较 rank 不足以判断兼容性，还要比较命名、覆盖层和每个输入的形状。

| 兼容情况 | `LoRA::Init` 的行为 | 结果 |
|---|---|---|
| 基座有输入，适配器有同名且同尺寸 tensor | 复制到后端 `TensorBuffer` | 该增量参与执行 |
| 基座有输入，适配器缺少同名 tensor | 对整个输入 buffer 填零 | 该处不施加增量 |
| 基座有输入，适配器同名但尺寸不同 | `RET_CHECK_EQ` 失败 | 适配器不能启用 |
| 适配器有额外 tensor，基座没有对应输入 | 初始化循环不会请求 | 额外 tensor 不参与执行 |
| 输入名不匹配两组正则模式 | 不视为 LoRA 输入 | 不会由 `LoRA::Init` 管理 |

> 表 7-4　运行时适配的兼容性由基座 signature 驱动，rank 只是其中一个条件。

`current_lora_id_` 只决定当前返回哪组 buffer，不会删除以前用过的对象：`loras_` 保留每个执行过 `UseLoRA` 的 ID，公开接口没有逐项卸载方法，切回旧 ID 时直接复用已创建的对象。用过的不同 ID 越多，管理器保留的源数据与后端 buffer 越多，直到管理器销毁。ID 还必须由调用者保证在管理器生命周期内唯一：`LoadLoRA` 的重复检查只查待用表，某个 ID 物化后再用同一 ID 登记新数据不会被拒绝，之后的 `UseLoRA` 仍选择旧对象，新数据不会替换它。应用不能通过复用 ID 更新适配器。

音频侧的 `UseLoRA(std::nullopt)` 也不是卸载操作：实现直接返回成功，相邻的 TODO 注释说明尚未支持清除 LoRA buffer。停用当前适配器、释放某个 ID 和销毁管理器是三个不同的操作，前两个目前都不能回收单个适配器占用的内存。取 buffer 时，`GetLoRABuffers` 只从当前 ID 的对象取，返回共享底层数据的 `Duplicate()` 句柄，不再复制张量；调用方仍要按接口约定释放取得的句柄。

## 7.5　`litertlm_print`：检查头部与 section 目录

`litertlm_print` 先读取版本与系统元数据，再遍历 `SectionObject`，打印每项的键值、起止偏移和 section 类型。遇到 `LlmMetadataProto` 段时，它额外读出 protobuf 并输出 `DebugString()`，于是 start token、stop token 和 prompt 模板会展开显示；其他段只显示位置与类型。

这个工具并不覆盖所有合法的 `VData` 类型：union 声明了 12 种值类型，打印函数只显式处理 `StringValue`、`Int32`、`Float32`、`Bool` 和 `UInt64`，其余合法类型显示为 `Unknown Type`。解析元数据段时，它保存了读取函数的返回状态，却没有在输出 `DebugString()` 前检查它。因此，该工具能显示目录和常见元数据，但不能完整展示所有 union 类型，单次输出也不能替代文件验证。

## 7.6　故障诊断案例：同一模型换后端后初始化失败

假设应用收到一个名为 `chat-int4.litertlm` 的文件。GPU 初始化成功，改用 CPU 后在 executor 创建前返回 `Main backend constraint mismatch`。开发者删除 XNNPACK cache，错误仍然存在。

第一步应检查容器目录，而不是根据 `int4` 文件名推断 CPU 支持。用 `litertlm_print` 查看主 `TFLiteModel` 段的 `model_type`、`backend_constraint` 与 `prefer_activation_type`。若 `backend_constraint` 只有 `gpu`，loader 会把该属性传给 Engine，`ValidateBackendConstraint` 在编译前返回 `InvalidArgumentError`。此时 cache 尚未决定算子能否执行，删除 cache 不会解除模型声明的后端约束。

若约束同时包含 `cpu,gpu`，再进入第二层诊断：检查是否存在独立的 `TFLiteWeights` 段。当前主 executor 只允许 GPU 使用该段的偏移表，非 GPU 路径会直接返回 `InvalidArgumentError`。这种失败同样不是 kernel 性能问题，而是当前加载接口不接受该产物布局。

若没有外挂权重段，且后端约束允许 CPU，才进入编译与 cache 层。可以用 `:nocache` 做一次隔离运行。无 cache 仍失败，优先检查 TFLite 图和 CPU delegate 的算子支持；无 cache 成功、启用 cache 失败，再核对 cache 路径、文件权限和模型是否被原位替换。若替换后的文件大小与秒级 mtime 都相同，或者同一进程已查询过旧路径，`GetFileCacheIdentifier` 可能继续给出旧标识。

| 失败位置 | 可观察线索 | 首要检查 | 不应先归因于 |
|---|---|---|---|
| Engine 设置校验 | `backend constraint mismatch` | section 的后端约束与请求后端 | INT4 kernel 性能 |
| 外挂权重接入 | 非 GPU 使用 `TFLiteWeights` 时返回错误 | 容器是否拆分模型与权重 | cache 命中率 |
| `CompiledModel::Create` | delegate 或算子编译错误 | 图、算子、激活类型、目标后端 | tokenizer 并行 |
| cache 启用后才失败 | 无 cache 可成功初始化 | 路径、权限、mtime、size、进程重启 | 模型质量 |
| 初始化成功但内存超限 | RSS、private footprint 或后端分配上升 | 转换权重、KV cache、工作区 | 文件体积就是峰值内存 |

> 表 7-5　初始化诊断按容器契约、产物布局、后端编译和资源占用逐层推进。

这个案例没有假定 CPU 一定支持或不支持 INT4，它只利用错误发生的位置缩小范围。后端真正采用何种 kernel，仍要结合 delegate 日志、编译结果和目标设备实测判断。

## 7.7　部署核验：从模型文件到可回滚版本

量化、容器与 cache 影响 Engine 创建，LoRA 则在后续的会话资源建立中接入。部署系统仍要按同一版本管理这些对象。只检查模型能否打开，既不能确认模型段完整，也不能证明 cache 属于当前版本；编译产物是否复用、适配器切换后保留多少资源，还需分别核验。本节把这些检查组织成一条部署流程。

### 7.7.1　真实容器案例：3.66 GB 文件中的十个 TFLite payload

附录 D 使用的 Gemma 4 E4B `.litertlm` 文件为 3.66 GB。静态分析先在 16 KiB 边界查找 TFLite 标识，再用 TFLite FlatBuffer 绑定读取 signature 与张量形状，由此识别出 10 个 TFLite payload。这里的“段”是该扫描方法识别出的 payload，不是 `litertlm_print` 导出的 section 目录。

| 起点（字节） | 大小 | signature | 资源用途或关键输入 |
|---:|---:|---|---|
| 4,734,976 | 171 MB | `embedder` | `token_ids[1,1]` |
| 175,669,248 | 837 MB | `per_layer_embedder` | `token_ids[1,1]` |
| 1,012,449,280 | 94 MB | `serving_default` | 音频编码器，`mask[1,1,816]` |
| 1,106,509,824 | 16 MB | `audio_adapter` | `features[1,204,1536]` |
| 1,122,254,848 | 16 KB | `eoa` | 音频结束标记 |
| 1,122,271,232 | 224 MB | `vision_70/140/280` | `images[1,1260,768]` |
| 1,346,420,736 | 8 MB | `vision_adapter_70/140/280` | `soft_tokens[1,140,768]` |
| 1,354,317,824 | 16 KB | `eoi` | 图像结束标记 |
| 1,354,334,208 | 2,260 MB | `decode`、`prefill_1024`、`prefill_128`、`verify` | `embeddings[1,1,2560]` |
| 3,614,392,320 | 45 MB | `mtp_drafter` | `activations[1,1,5120]` |

> 表 7-6　同一个 `.litertlm` 可以同时携带文本、音频、视觉和 MTP 资源；文件总大小不能当作单个 decode step 的权重读取量。

主文本段约为 2,260 MB，约占文件的 62%。两个 embedding 段合计约 1,008 MB，音频、视觉与 MTP 占据其余空间。表中的近似大小按十进制 MB 记录，不宜与前文按二进制计算的 GiB 预算直接相加；需要换算时，应回到起止偏移计算字节数。

这个文件同时说明了两个部署问题。第一，下载、签名验证和磁盘配额处理的是完整的 3.66 GB 文件；运行时内存则取决于实际请求哪些 section、后端是否转换权重，以及映射页的驻留情况。第二，文本请求不等于所有段都会在每个 decode step 参与计算；把整文件体积代入“每 token 权重流量”，会把多模态与 MTP 资源一并算进主干模型。

section 的资源角色由目录属性中的 `model_type` 区分，例如主文本、视觉编码器与 MTP drafter。它与 `LlmMetadata.llm_model_type` 不同：后者记录 Gemma、Qwen 等模型族，参与默认 prompt 与部分编译设置的选择。部署清单应分别记录这两个字段，避免把“模型族”和“容器中的资源角色”混为一项。

### 7.7.2　目录可解析不等于容器完整

固定前缀、FlatBuffer 头、section 目录和 section 内容是四层不同的结构。固定前缀正确，只能说明读取器找到了预期魔数和版本字段。当前读取器比较 major 版本，minor 与 patch 会被读出，但不参与兼容性拒绝；字节 20-23 也会被直接跳过。当前格式常量是 1.5.0。

字节 24-31 给出 FlatBuffer 头部的结束位置。普通读取器要求结束位置不小于 32，并检查相应字节能否读出。Engine 的主 loader 最多映射文件开头 16 KiB，再把这段传递给读取器；流式 loader 另有显式的 32 字节至 16 KiB 范围检查。两个入口的防护位置不同，损坏文件不一定返回同一种错误。

头部读完后，读取器直接取得生成的根对象访问器，这条路径没有建立 `flatbuffers::Verifier`。schema 中的 `(required)` 字段定义了合法文件应有的结构，但不能据此认为主 loader 已经验证了所有 vector 边界、必需字段和 union 类型。外部取得的模型文件若不受发布链信任，应在调用 Engine 前执行独立的 FlatBuffer 与 section 范围校验。这是由当前边界推导出的应用要求，不是运行时已提供的验证接口。

section 目录还要做全局检查。打包工具按 16 KiB 倍数计算各段起点，并在写完一段后记录其结束位置；主 loader 建索引时只显式拒绝 `begin_offset > end_offset`。这段循环没有统一检查零长度、文件末尾、头部重叠、section 之间的重叠和 16 KiB 对齐。`MapSection` 为文件句柄调整 mmap 起点，解决的是操作系统映射 API 的对齐要求，不是容器格式校验；底层文件映射会拒绝超出文件长度的范围，整文件已映射的分支则直接执行基址加偏移。因此，越界目录项不应留到首次取段时再发现。

<figure>
{{#include figs/fig-7-5.svg}}
<figcaption>图 7-5　模型产物要依次通过前缀、FlatBuffer、section 布局、payload、Engine 语义和后端编译检查；上一层通过不能替代下一层。</figcaption>
</figure>

| 校验层 | 当前检查 | 发布前还应检查 | 典型失败位置 |
|---|---|---|---|
| 固定前缀 | 魔数、版本字节可读、major 相等 | padding、允许的 minor/patch 范围 | 读取头部 |
| 头部范围 | 结束位置不小于 32、指定字节可读 | 分配前的大小上限、FlatBuffer verifier | 读取或访问头部字段 |
| section 局部范围 | `begin_offset <= end_offset` | 非零长度、末尾不越界、不得覆盖头部 | 建索引或首次取段 |
| section 全局布局 | loader 无统一检查；builder 按 16 KiB 排布 | 段间不重叠、键唯一、起点符合发布格式 | 建索引或取错资源 |
| payload | 交给 TFLite、tokenizer 或 protobuf 消费者 | 子格式完整性、内容摘要与签名 | 创建具体资源 |
| Engine 语义 | 资源角色、后端约束与激活提示 | 必需角色齐全、约束字符串规范化 | 创建 executor 前 |
| 后端编译 | delegate 尝试创建 compiled model | 算子的执行后端、cache 代际、目标设备冒烟测试 | `CompiledModel::Create` |

> 表 7-7　容器检查要覆盖字节范围、目录全局关系和资源语义，不能以 `litertlm_print` 能输出目录作为完整性结论。

现有测试能复现两个层级的损坏。流式 loader 的 `HeaderTooSmall` 用例保留正确魔数，把头部结束偏移设为 8，读取头部即失败；主 loader 的 `InitializeWithInvalidOffsets` 用例构造 `begin_offset=100`、`end_offset=50` 的模型段，头部可以读取，失败发生在建索引阶段。两种错误都早于后端编译，删除 cache 或切换 CPU/GPU 不会修复文件结构。

还应补三类回归测试：`begin == end`、`end > file_size`，以及两个 section 生成相同的 `BufferKey`。`BufferKey` 由 `data_type` 与可选的 `model_type` 组成，`backend_constraint` 不在键中。据索引表的赋值方式推断，两段若键相同而后端约束不同，后遍历到的范围和提示会覆盖前一项，而不会形成 CPU/GPU 自动选段。这是代码分析结论，当前测试没有把它固定成格式错误。

发布侧校验器应使用头部目录，而不是扫描 `TFL3` 标识来恢复 section。表 7-6 的扫描用于研究现有文件；它可能找到嵌套数据中的相同字节，也无法得到 `model_type`、后端约束和 tokenizer 等非 TFLite section。面向发布的校验流程可以先读取文件长度，再用有上界的缓冲解析前缀与 FlatBuffer；只有头部通过 verifier，才遍历目录。

目录遍历时，为每个 section 建立半开区间 `[begin,end)` 和 `BufferKey`。先检查 `32 <= header_end <= 16 KiB`，再要求 `header_end <= begin < end <= file_size`。若发布产物只允许由当前打包工具生成，还可把 16 KiB 起点对齐列入发布规则，这比主 loader 的兼容范围更严格。所有区间按起点排序，若前一项的 `end` 大于后一项的 `begin`，文件即有重叠。`BufferKey` 集合则用于拒绝重复资源角色。

结构检查通过后，再按 `data_type` 验证 payload。TFLite 段要在自己的字节范围内完成模型验证，protobuf 和 tokenizer 也使用各自解析器。最后核对产品需要的角色，例如主文本模型、metadata 和 tokenizer 是否齐全。这样得到的错误可以指向“目录第 4 项越界”或“缺少主文本角色”，而不是在后端编译时只留下一个宽泛的初始化失败。

校验结果最好保存成机器可比较的制品清单。下面是应用层报告的示例结构，不是 `.litertlm` schema 的新增字段：

```json
{
  "artifact_sha256": "…",
  "format_version": "1.5.0",
  "file_size": 3660000000,
  "header_end": 4731,
  "sections": [
    {
      "index": 0,
      "data_type": "TFLiteModel",
      "model_type": "TF_LITE_PREFILL_DECODE",
      "begin": 16384,
      "end": 123456,
      "payload_sha256": "…"
    }
  ],
  "required_roles_ok": true
}
```

其中的数值只是字段示例，不描述表 7-6 的真实头部与 section。内容摘要算法、签名和密钥管理由发布系统决定。v41 与 v42 的报告可以在上线前直接比较：文件版本、角色集合、后端约束或某个 payload 发生变化时，发布审批不必等到设备初始化后才发现。

### 7.7.3　cache 入口决定文件由谁管理

`GetWeightCacheFile` 与 `GetProgramCacheFile` 都可能返回派生路径或调用者已经打开的 `ScopedFile`。`cache_dir = :nocache` 的判断最先执行，连已经提供的 scoped cache file 也会被禁用。未禁用时，scoped file 优先；只有缺少它，helper 才根据模型路径计算标识并拼接 cache 路径。

主 CPU 路径把 `.xnnpack_cache` 附加到模型名后。设置独立目录时，典型路径为：

```text
<cache_dir>/<model_basename>.xnnpack_cache_<mtime_seconds>_<size>
```

这一路径或 fd 被写入 XNNPACK 编译选项。MTP drafter 在 CPU 后缀前增加 `.mtp_drafter`，以免与主模型共用名称。

GPU 的 program cache 候选路径以 `_mldrift_program_cache.bin` 结尾，模型 cache key 使用 `<basename>_<mtime_seconds>_<size>`。在路径模式下，GPU 选项接收序列化目录与这个 key；在文件描述符模式下，才分别接收 weight cache 和 program cache 的 fd。路径模式下 GPU weight cache 的实际文件名、格式与命中判断属于下层 LiteRT 后端，不能从 LiteRT-LM 的候选路径单独确定。基类还提供按模型组件区分名称的 `GetCacheSuffix`，但主 LLM 编译路径没有调用它，主路径直接使用上述常量。

| 入口 | LiteRT-LM 的处理 | 交给后端的值 | 目录与旧文件责任 |
|---|---|---|---|
| CPU，cache 目录 | 派生 XNNPACK 文件路径 | weight cache path | 应用应先创建目录；路径模式可触发匹配清理 |
| CPU，scoped file | 复制 fd | weight cache fd | 调用者创建、授权并回收文件 |
| GPU，cache 目录 | 派生 program 候选路径与模型 key | serialization dir、model cache key | 实际 weight 文件由下层后端决定 |
| GPU，scoped files | 分别复制两个 fd | weight/program cache fd | 调用者负责两个文件的代际 |
| `:nocache` | helper 返回错误，调用链按无 cache 配置继续 | CPU 不设置 cache；GPU 关闭序列化 | 不执行自动清理 |

> 表 7-8　路径模式由 LiteRT-LM 派生名称，文件描述符模式由应用决定文件位置和生命周期。

路径 helper 只拼接名称、检查文件并尝试清理旧项，不会创建 cache 目录。应用应在创建 Engine 前建立目录并验证写权限。scoped file 测试采用的顺序也是先创建空文件、以可写方式打开，再把 fd 传递给 Engine。

清理规则还带来并发边界。`DeleteStaleCaches` 没有进程间锁；若两个进程使用相同 basename 和同一 cache 目录，却计算出不同标识，其中一个进程可能把另一个进程的文件识别为旧项。删除失败只记录警告，helper 仍会返回当前路径。不可变模型名配合独立版本目录可以避免这一竞争；代价是旧版本 cache 不会自动被新版本清理，需要发布系统按保留期回收。

### 7.7.4　发布案例：用新进程切换不可变版本

假设进程 P41 正在使用 `/models/chat-v41.litertlm` 和 `/cache/v41/`。待发布产物为 v42。若直接覆盖 `/models/current.litertlm`，同一长生命周期进程可能已经保存了旧文件的 `<mtime>_<size>`。即使另起进程，只要新旧文件的字节数和秒级 mtime 相同，内置标识仍无法区分内容。

发布系统可以先把 v42 写入临时文件，校验内容摘要、容器目录和必需资源，再改成最终的不可变名称 `/models/chat-v42.litertlm`。内容摘要属于应用的完整性证据，不能由 LiteRT-LM 的修改时间与大小组合替代。新进程 P42 使用独立的 `/cache/v42/` 创建 Engine。固定输入的 prefill 与 decode 冒烟测试通过后，应用层才切换请求路由。

<figure>
{{#include figs/fig-7-6.svg}}
<figcaption>图 7-6　新模型在独立进程和独立 cache 目录中完成初始化与冒烟测试后再接收请求；旧版本保留到回滚窗口结束。</figcaption>
</figure>

切换后，P41 不再接收新请求，但继续处理已经进入的会话。v41 的模型、进程镜像和 cache 目录保留到回滚窗口结束。观察期内若 v42 触发质量、内存或兼容性问题，路由重新指向 P41；旧进程已经退出时，则用 v41 的不可变路径和 cache 目录启动替代进程。

若应用要求原子切换，应在路由或 Engine 句柄层实现；这条调用链不提供模型发布事务。GPU 模型 key 没有加入运行时版本、设备型号、驱动、激活类型和 `cache_compiled_shaders_only`。下层后端是否另行校验 cache 兼容性，不能从这段代码确定。这些条件变化时，发布系统应建立新的 cache 代际。

| 发布关口 | 核验对象 | 通过条件 | 失败后的动作 |
|---|---|---|---|
| 产物落盘 | 内容摘要、字节数、只读权限 | 与制品清单一致 | 删除临时产物，不修改线上版本 |
| 容器检查 | 头部、section 范围、角色、payload | 所有必需项通过表 7-7 | 阻止创建 Engine |
| Engine 创建 | backend、激活类型、cache 目录 | 目标设备完成编译或加载 | 保留日志与 cache 文件清单 |
| 冒烟测试 | 固定 token、prefill、decode、停止原因 | 结构与基准输出满足发布阈值 | 销毁 P42，不切路由 |
| 资源阈值 | 外部墙钟、峰值内存、cache 增量 | 不超过设备与服务预算 | 调整产物或后端配置 |
| 路由切换 | 新请求入口 | P42 健康，P41 可继续排空 | 立即切回 P41 |
| 保留期结束 | v41 进程、模型与 cache | 回滚窗口结束且指标稳定 | 按明确清单回收旧代际 |

> 表 7-9　发布步骤把容器完整性、后端可执行性和在线回滚分成独立关口，避免用一次初始化成功替代全部验证。

### 7.7.5　冷启动与 cache 命中要分开测量

冷启动至少受四类状态影响：进程是否新建、后端 cache 是否存在、模型页是否已进入操作系统 page cache，以及设备温度和频率是否稳定。删除 cache 文件不能清空 page cache；只重建 Engine 也不能排除同一进程保存的文件标识。

| 组别 | 进程 | backend cache | 运行配置 | 回答的问题 |
|---|---|---|---|---|
| N：无 cache 对照 | 每轮新建 | 不使用 | `:nocache` | 不读写后端 cache 时的初始化成本 |
| P：首次生成 | 每轮新建 | 空的版本目录 | 启用 cache | 生成 cache 增加多少时间和存储量 |
| W：已有 cache | 每轮新建 | 复用 P 生成的受控副本 | 启用 cache | 新进程复用 cache 后的初始化成本 |
| S：同进程重建 | 同一进程 | 已存在 | 启用 cache | 进程内复用现象，不作独立冷启动结论 |

> 表 7-10　N、P、W 三组用于分离 cache 的生成成本与跨进程复用收益；S 组只暴露进程内状态。

每轮应从 Engine 创建调用之前开始记录外部墙钟，在创建返回后结束。首次 prefill 和首次 decode 另行计时，避免把权重转换、kernel 编译与首 token 时延合成一个数字。模型内容摘要、字节数、mtime、LiteRT-LM commit、后端、设备与驱动、激活类型、CPU 线程数、cache 选项、目录前后文件清单和设备温度都要随原始结果保存。

N 与 W 可以交替运行，降低温度和后台负载随时间单向变化造成的偏差。若要控制操作系统 page cache，应使用目标平台允许且可复现的方法，并把操作记录写入实验数据。没有这项控制时，只能把结果称为“新进程、无 backend cache”或“新进程、已有 backend cache”，不能称为物理磁盘冷读。

cache 文件存在本身不能证明命中；scoped-file 模式甚至要求调用者先创建文件。这条调用链没有统一的 cache-hit 计数器，判断命中还需结合后端日志、文件是否重写和受控初始化时间差。现有测试验证了同一 cache 可再次创建 Engine 并得到非空输出，但没有量化初始化收益。附录 D 的 5.29 s 与约 1.77 s 也缺少 N/P/W 对照，不能直接归因于 cache。

### 7.7.6　多 LoRA 案例：文件稀疏不等于后端缓冲稀疏

对形状为 \\(d_{\mathrm{out}}\times d_{\mathrm{in}}\\) 的投影，若 LoRA rank 为 \\(r\\)，每个元素占 \\(b\\) 字节，两组低秩矩阵的理论数据量为：

$$
B_{\mathrm{pair}}=r(d_{\mathrm{in}}+d_{\mathrm{out}})b.
$$

这个公式只能计算矩阵 payload。运行时不拿 `lora_rank` metadata 直接判断兼容性：`LoRA::Init` 先按基座 signature 创建输入 buffer，再比较 `TensorBuffer::PackedSize()` 与适配器 tensor 的实际字节数，不同就返回错误。两个适配器即使 rank 都是 32，也可能因隐藏维度、GQA 投影宽度、元素类型或导出命名不同而不兼容。

仓库测试资产 `litert_dummy_lora32_f16_model.tflite` 提供了可复算的规模。本书对该二进制的静态解析表明，`decode` signature 有 35 层、280 个 FP16 LoRA 输入；完整命令与逐类统计见附录 C。单元测试另行确认 rank 为 32，一个 `32 × 2048` 的 query tensor 占 \\(32 \times 2048 \times 2 = 131072\\) 字节，即 128 KiB；物化后返回的 buffer 数量也是 280。

| 投影与矩阵 | 每层个数 | 单个形状 | 单个大小 | 35 层合计 |
|---|---:|---:|---:|---:|
| query left / right | 2 | `32 × 2048` | 各 128 KiB | 8.75 MiB |
| post left / right | 2 | `32 × 2048` | 各 128 KiB | 8.75 MiB |
| key left | 1 | `32 × 2048` | 128 KiB | 4.375 MiB |
| key right | 1 | `32 × 512` | 32 KiB | 1.09375 MiB |
| value left | 1 | `32 × 2048` | 128 KiB | 4.375 MiB |
| value right | 1 | `32 × 512` | 32 KiB | 1.09375 MiB |
| 合计 | 8 | 每层 832 KiB | 不适用 | 28.4375 MiB |

> 表 7-11　静态解析得到的 280 个 LoRA 输入按 shape 计算为 28.4375 MiB 理论大小；物化后的 `PackedSize()`、allocator 对齐和运行时开销仍需实测。

同一份静态解析记录显示，配套适配器 `test_lora_rank32_f16_all_ones.tflite` 只保存 220 个匹配 tensor：前 20 层包含 query、key、value 与 post，后 15 层只包含 query 和 post，因此文件 tensor payload 为：

$$
20\times832\ \mathrm{KiB}+15\times512\ \mathrm{KiB}=23.75\ \mathrm{MiB}.
$$

后 15 层缺少 60 个 key/value tensor，共 4.6875 MiB。`LoRA::Init` 仍按基座 signature 创建全部 280 个 buffer，找不到同名 tensor 时把对应 buffer 清零；单元测试检查了缺失的 `value_w_prime_left_20`，返回内容全为零。按基座 shape 计算，这 280 个输入对应 28.4375 MiB 理论大小，其中 4.6875 MiB 是清零的输入；后端实际分配仍应读取 `PackedSize()` 并测量。

清零是字节层面的已验证行为。它是否在任意导出图中都等价于“不施加增量”，还取决于图如何使用该输入，不能只根据 `memset` 推广。尺寸不匹配会使首次 `UseLoRA` 失败；目前没有对应单元测试，部署前应加入适配器与基座 signature 的逐 tensor 兼容性检查。

### 7.7.7　懒物化推迟分配，但不限制累计数量

7.4.1 节说明了管理器的规则：`LoadLoRA` 只登记数据，首次 `UseLoRA` 才创建后端 buffer；切换当前 ID 不释放已物化的对象，ID 也不能当作可覆盖的槽位。这一节把这些规则换算成容量。源数据不会在复制结束后释放：`LoRA` 对象同时持有 `LoraData` 与创建出的后端 buffer，所以一个已启用的适配器通常同时持有文件映射或原始缓冲视图、后端 `TensorBuffer` 与管理器记录。文件映射范围不等于常驻物理内存，后端 buffer 的位置也不能统一写成 GPU 显存。

<figure>
{{#include figs/fig-7-7.svg}}
<figcaption>图 7-7　首次 UseLoRA 把待用数据物化为后端 buffer；切换当前 ID 不会释放已经物化的适配器。</figcaption>
</figure>

假设音频服务依次加载 A、B、C 三个适配器，三者都采用表 7-11 的测试形状。状态与理论大小如下：

| 操作结束后 | `lora_data_` | `loras_` | 当前 ID | 输入 buffer 理论大小 |
|---|---|---|---|---:|
| `Load(A/B/C)` | A、B、C | 空 | 无 | 0 |
| `Use(A)` | B、C | A | A | 28.4375 MiB |
| `Use(B)` | C | A、B | B | 56.875 MiB |
| 再次 `Use(A)` | C | A、B | A | 56.875 MiB |
| `Use(C)` | 空 | A、B、C | C | 85.3125 MiB |

> 表 7-12　切回已物化 ID 只更新 `current_lora_id_`；管理器仍保留此前创建的对象和 buffer。

测试覆盖了 `0 → 1 → 0` 的切换，并验证切回后仍能读取 ID 0 的原内容。若同形状的 8 个适配器都至少使用过一次，仅 LoRA 输入 buffer 的理论大小就是 \\(8\times28.4375=227.5\\) MiB，还要计入 8 份源数据视图与运行时开销。管理器没有逐 ID 卸载接口，容量预算应按“生命周期内启用过的不同 ID 数量”计算，而不是只按当前 ID 计算。

### 7.7.8　回到设备预算：快速回滚需要两类峰值

图 7-6 的双版本方案把回滚时间缩短到一次路由切换，但代价发生在存储和运行内存两个口径。以表 7-6 的 3.66 GB 文件为例，仅同时保留 v41 与 v42 两个模型，就需要约 7.32 GB 十进制存储。两个版本的 cache、下载临时文件、文件系统预留和日志还要另算。

存储峰值可以写成：

$$
D_{\mathrm{peak}}=D_{\mathrm{old\ model}}+D_{\mathrm{new\ model}}
+D_{\mathrm{old\ cache}}+D_{\mathrm{new\ cache}}+D_{\mathrm{staging}}+D_{\mathrm{margin}}.
$$

这里的 \\(D_{\mathrm{staging}}\\) 取决于更新器。若新文件直接下载到同一文件系统的临时名称，再以 rename 变成最终名称，临时文件与最终文件是同一份字节，不能重复相加；若更新器另外保留压缩包或补丁，则要把那份文件计入。\\(D_{\mathrm{margin}}\\) 用于文件系统元数据、日志增长和发布过程中的并发写入，由产品策略设定。

Engine 也有重叠期。P42 完成初始化和冒烟测试前，P41 仍在处理请求。此时内存预算包含两个 Engine 的映射、后端转换权重、工作区与会话状态。两个版本的文件不同，不能预设操作系统会共享其映射页；后端私有 buffer 更不能按文件页共享。峰值也不必恰好等于两个单 Engine 峰值之和，因为页驻留、初始化临时量和旧会话数量随时间变化。

运行内存要先固定统计范围。令 \\(W\\) 为 P42 开始初始化到 P41 释放的观察窗口，\\(M_{\mathrm{total}}(t)\\) 是同一口径下时刻 \\(t\\) 的总量：

$$
\begin{aligned}
M_{\mathrm{total}}(t)={}&M_{\mathrm{base}}(t)+M_{\mathrm{old\ engine}}(t)+M_{\mathrm{new\ engine}}(t)\\
&+M_{\mathrm{sessions}}(t)+M_{\mathrm{init\ temp}}(t),\\
M_{\mathrm{peak}}={}&\max_{t\in W}M_{\mathrm{total}}(t).
\end{aligned}
$$

其中，Engine 项不含 Session 状态，\\(M_{\mathrm{init\ temp}}\\) 单列编译和权重转换期间的临时量。发布前再检查 \\(M_{\mathrm{peak}}+M_{\mathrm{margin}}\le M_{\mathrm{budget}}\\)。这些量不能从 `.litertlm` 文件大小推算。应在目标设备上记录外部峰值、private footprint、文件映射驻留和后端设备内存，但有重叠的操作系统指标不能直接相加。采样从 P42 初始化前开始，到 P41 排空并释放后结束。若服务允许多个并发 Session，切换点的旧会话数量也要进入测试矩阵。

<figure>
{{#include figs/fig-7-8.svg}}
<figcaption>图 7-8　双 Engine 重叠窗口覆盖 P42 初始化至 P41 排空；峰值须实测，Engine 数量为 2 不表示内存严格翻倍。</figcaption>
</figure>

| 发布方式 | 模型与 cache 存储 | 同时存活的 Engine 上限 | 旧会话处理 | 回滚路径 |
|---|---|---:|---|---|
| 独立进程，双版本并行 | 两代都保留 | 2，直到旧进程排空 | P41 继续完成 | 切回仍存活的 P41 |
| 同进程，两个 Engine 句柄 | 两代都保留 | 2，且共享同一进程故障域 | 由应用分别路由 | 切换句柄；进程级故障无法隔离 |
| 保留旧文件，排空后重启 | 两代都保留 | 1 | 先排空，再释放 P41 | 重新创建 v41 Engine |
| 覆盖文件后重启 | 通常只留一代，更新期仍可能有临时文件 | 1 | 需要中断或迁移 | 依赖备份或重新下载旧产物 |

> 表 7-13　发布方式在存储峰值、运行内存、会话连续性与回滚时间之间取舍；LiteRT-LM 不替应用选择其中一种。

移动设备若无法同时容纳两个 Engine，可以保留 v41 文件和 cache，停止接收新会话，等待旧会话结束，然后销毁 P41 再创建 P42。这样仍有两个版本的存储峰值，但 Engine 工作集维持一份。代价是切换期间不能立即服务新请求；P42 初始化失败时，也要重新创建 v41 Engine，回滚时间包含一次冷启动或 cache 复用启动。

同进程持有两个 Engine 不会消除运行内存峰值。销毁旧 Engine 也不会清空路径标识的进程内静态 map；若复用原路径，后续查询仍可能取得第一次保存的标识。因此，同进程方案仍应使用不可变模型路径。它能省去跨进程路由，却失去进程级故障隔离。是否采用这一方式，要看应用能否在一个进程内明确串行创建、切换和销毁 Engine，并完成目标平台的峰值验证。

若磁盘空间不足以同时存放两个模型文件，就不能同时满足“旧产物本地可用”和“新产物完整落盘后再切换”。此时需要在发布前明确选择：缩减安装包中的可选模态资源、使用系统提供的增量分发能力，或者接受回滚时重新下载。`.litertlm` 是单文件容器；本章没有证据表明发布器能在两个版本之间自动复用相同 section 的磁盘块。

### 7.7.9　验收证据包：保存可复核的发布记录

Engine 初始化成功，只能说明某个进程在当时的配置下完成了这次调用。发布审批还需要知道它打开了哪个文件、选了哪个后端、cache 目录原来有什么、峰值发生在哪个阶段，以及回滚是否真正执行过。这些信息若分散在终端历史和临时日志中，下一次升级无法复用同一判断标准。

可以为每个不可变版本保存一份应用层证据包。例如：

```text
release-v42/
├── manifest.json
├── container-report.json
├── init/
│   ├── runtime-config.json
│   ├── cache-before.txt
│   ├── cache-after.txt
│   └── timing-n-p-w.csv
├── smoke/
│   ├── input.json
│   ├── output.json
│   └── backend.log
├── memory/
│   ├── dual-engine.csv
│   └── measurement-notes.md
├── lora/
│   └── compatibility-report.json
└── rollback.md
```

`manifest.json` 记录模型内容摘要、字节数、来源、LiteRT-LM commit 与构建标识。`container-report.json` 保存前文的头部、section 和必需角色检查。`runtime-config.json` 则固定 backend、激活类型、线程数、最大 token 数和 cache 设置。三类文件描述的是不同对象，不能只保留其中一份。

| 证据 | 至少记录 | 可识别或排除的问题 | 仍不能单独证明 |
|---|---|---|---|
| 制品清单 | 内容摘要、文件大小、来源、签名结果 | 拿错文件、传输损坏 | 目标后端可执行 |
| 容器报告 | 格式版本、section 范围、角色、payload 校验 | 目录越界、重复键、必需角色缺失 | 算子全部受支持 |
| 运行配置 | commit、设备、驱动、backend、激活类型、线程与 token 上限 | 配置漂移导致结果不可比 | 性能满足阈值 |
| cache 快照 | 模式、路径或 fd、运行前后文件名与摘要 | 把旧目录状态误写成新产物 | 本次一定命中 cache |
| N/P/W 原始时间 | 每轮外部墙钟、顺序、温度与异常 | 用一次批内 Init 值代表 cache 收益 | page cache 已被完全控制 |
| 冒烟输入输出 | token 化输入、采样参数、停止原因、后端日志 | 只创建 Engine 而未执行 prefill/decode | 质量已覆盖真实分布 |
| 内存时间线 | 采样口径、P42 初始化和 P41 释放时刻、峰值 | 用文件大小代替运行峰值 | 未测并发下仍安全 |
| LoRA 兼容报告 | 输入名、shape、payload、缺项与 ID 规则 | 只比 rank、忽略 signature | 文本路径已支持热切换 |
| 回滚记录 | 触发条件、路由步骤、旧版本启动时间、恢复结果 | 只写回滚方案而未演练 | 所有现场故障都能恢复 |

> 表 7-14　证据包把制品、运行配置、原始测量和回滚演练分开归档；每项证据只回答表中限定的问题。

`cache-before.txt` 与 `cache-after.txt` 应列出文件名、字节数和修改时间；必要时再保存摘要。它们与后端日志和 N/P/W 时间共同用于判断 cache 行为，不能只凭新文件出现下结论。内存 CSV 需要带单调时钟，并在相同时间轴上标记 Engine 创建、路由切换与旧 Engine 释放，否则无法把峰值归到图 7-8 的具体阶段。

LoRA 报告只在产品启用该能力时生成。对当前的文本生成路径，报告应明确写“不支持上层热切换”，而不是用一份空结果表示通过。回滚记录也要保存实际命令、耗时和恢复后的固定输入输出；尚未演练时，发布状态应标为“方案已写，恢复未验证”。证据包不改变 LiteRT-LM 的运行行为，它只保证每个结论能追溯到产物、配置和原始记录。

证据包还要区分“检查通过”和“没有失败记录”。若字段只允许 true/false，缺失的实验很容易被默认成 false 后忽略，或者被空文件误判为成功。应用的发布系统可以使用下面五种治理状态；它们不是 LiteRT-LM 的状态码：

| 状态 | 含义 | 必需记录 | 示例 |
|---|---|---|---|
| `PASS` | 已按指定方法执行，结果满足阈值 | 原始证据、阈值、执行时间 | 容器范围校验全部通过 |
| `BLOCK` | 已检查或执行，但结果不满足阈值或要求 | 失败值、日志、阻断原因 | 9.14 GB 超过 9.0 GB 配额 |
| `NOT_MEASURED` | 适用但尚未执行或数据不足 | 缺失项、负责人、计划时间 | 尚无 N/P/W 外部墙钟对照 |
| `NOT_APPLICABLE` | 当前产物或发布方式不经过该路径 | 判定条件与依据 | 未配置 LoRA 的产品不做适配器兼容检查 |
| `WAIVED` | 已知不满足或未测，经明确审批暂时放行 | 风险、责任人、截止日期与撤销条件 | 经审批仅在内部设备试用，双 Engine 峰值仍未测 |

> 表 7-15　`NOT_MEASURED`、`NOT_APPLICABLE` 与 `WAIVED` 都不是 `PASS`；例外放行也要保留责任与失效条件。

状态必须绑定具体环境。某款 GPU 上的 `PASS` 不能覆盖另一款驱动，CPU 的 cache 对照也不能替代 GPU program cache。条件变化时，旧记录仍可保留作对照，但新组合要生成独立条目。对本书当前证据而言，N/P/W cache 收益和双 Engine 峰值应是 `NOT_MEASURED`；文本 LoRA 热切换则应记录为“当前上层路径不支持”，不能用 `NOT_APPLICABLE` 掩盖产品原本要求该能力的事实。

## 小结

低比特权重能按位宽算出理想的权重大小比例，但带宽、吞吐和质量要结合图表示、后端编译、kernel 与实测判断。`.litertlm` 用 FlatBuffer 头部描述 section；`model_type`、`backend_constraint` 与 `prefer_activation_type` 参与资源选择和设置校验。当前 Engine 通过 `LitertLmLoader` 建立索引并按请求取段，schema 工具层的 TFLite 提取函数只是辅助 API。

weight cache 由 LiteRT-LM 配置、底层后端解释。其模型标识使用修改时间和文件大小，不是内容哈希，进程内保存的旧标识还会扩大失效盲区。LoRA 的离线合并与运行时适配有不同的产物和验证要求；当前管理器按 ID 懒创建后端对象，已使用 ID 的资源会保留，调用者不能依赖复用 ID 更新适配器，而且完整的运行时适配路径目前只在音频编码器中实现。

CPU、GPU 与 NPU 的执行路径对照见第 8 章。该章继续区分配置项、代码路径与实测性能。

---

## 练习与自查

1. 收益偏离因素。某模型权重从 FP16 改为 INT4。列出至少三项会使整个文件大小或 decode 吞吐偏离 4 倍比例的因素。
2. 对齐偏移计算。某 section 的 `begin_offset = 49152`，平台对齐值为 65536。计算 `alignment_gap`、实际映射起点，以及返回给上层的指针偏移。
3. 缓存标识盲区。两个模型文件大小相同、修改时间精确到秒也相同，但内容不同。说明 `GetFileCacheIdentifier` 能否区分它们，并分析同一进程内的路径缓存会带来什么影响。
4. LoRA 状态追踪。依次对 ID 0 和 ID 1 调用 `LoadLoRA`、`UseLoRA`，再切回 ID 0。说明 `lora_data_`、`loras_` 与 `current_lora_id_` 的变化，以及哪些资源仍被保留。
5. 元数据打印路径。`litertlm_print` 对 `LlmMetadataProto` 多做了哪一步？遇到合法的 UInt8 元数据值时会打印什么？
6. 分组参数重算。把本章 2B 参数案例的 INT4 分组从 32 改为 64，其他条件不变。重新计算 scale、模型文件与 3.0 GiB 运行预算，并说明哪一项结论没有变化。
7. 后端约束诊断。某 `.litertlm` 的 `backend_constraint` 为 `cpu,gpu`，含独立 `TFLiteWeights`，GPU 可初始化而 CPU 返回错误。根据本章加载链路指出失败层级，并说明为什么删除 XNNPACK cache 不能解决该问题。
8. 容器校验判断。某容器的魔数、版本和 FlatBuffer 头均可读取，但两个 section 的字节范围互相重叠。说明主 loader 是否会在建索引时统一拒绝，并给出发布前校验器应增加的判断。
9. 缓存实验设计。为同一模型设计 N、P、W 三组 cache 实验。说明每组的进程与 cache 初始状态、外部墙钟的起止点，以及为什么“cache 文件存在”还不足以证明 W 组命中。
10. LoRA 容量累计。表 7-11 的测试形状下，A、B、C 三个 LoRA 都物化后，仅输入 buffer 的理论大小是多少？若切回 A，数值是否减少？再说明将 8 个适配器都物化后的 payload。
11. 发布配额推演。某设备给模型目录的配额是 9.0 GB。旧、新模型各 3.66 GB，两个 cache 上限分别为 0.28 GB 和 0.44 GB；更新器另存 0.60 GB 压缩包，并要求 0.50 GB 余量。计算发布存储峰值。若改为排空旧 Engine 后再创建新 Engine，存储峰值与运行内存峰值分别如何变化？运行内存能否仅凭 Engine 数量写成减半？

[^ch07-llamacpp-gguf]: ggml-org，[*llama.cpp 源码 ggml/include/gguf.h:46*](https://github.com/ggml-org/llama.cpp/blob/b9873/ggml/include/gguf.h#L46)，版本 b9873；访问日期：2026-08-31。
[^ch07-llamacpp-lora]: ggml-org，[*llama.cpp 源码 common/arg.cpp:2648*](https://github.com/ggml-org/llama.cpp/blob/b9873/common/arg.cpp#L2648)，版本 b9873；访问日期：2026-08-31。
