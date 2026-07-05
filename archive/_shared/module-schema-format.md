# 模块素材：模型格式与 Schema (Model Format & Schema)  `schema-format`

> 来源：litert-lm-guide/data.json（多智能体源码分析）。**这是素材，不是正文**。
> 引用进正文前，须按 CLAUDE.md 第二节逐条 Read 源码核验、补 `@ v0.13.1`。

**一句话**：本模块定义并实现 .litertlm 端侧模型容器文件格式: 固定二进制前缀(magic+SemVer) + FlatBuffer header(系统元数据与分段索引) + 各 section 原始数据, 并提供按段读取 TFLite/tokenizer/LlmMetadata 与探测 speculative decoding 的工具。

**在架构中的位置**：运行时模型装载的最底层格式定义层。上游是打包工具 litertlm_export_main, 下游是 runtime/util/litert_lm_loader.cc 与 litert_lm_streaming_loader.cc, 它们调用 ReadHeaderFromLiteRTLM 解析 header 后按 section offset 把权重/tokenizer mmap 进内存交给推理引擎。capabilities::HasSpeculativeDecodingSupport 也被上层用于探测文件能力。

## 关键文件
- `/Users/dpthinker/workspace/LiteRT-LM/schema/core/litertlm_header_schema.fbs` — FlatBuffer schema 源文件, 格式权威定义: KeyValuePair/VData union、SystemMetadata、SectionObject(含 begin/end_offset 与 data_type)、AnySectionDataType 枚举、根表 LiteRTLMMetaData。编译生成 litertlm_header_schema_generated.h。
- `/Users/dpthinker/workspace/LiteRT-LM/schema/core/litertlm_read.cc` — 格式读取核心实现: 解析二进制 header 前缀、校验 magic 与 major version、定位 FlatBuffer, 用模板 ReadValueTFromSection/ReadAnyT 按 section 类型读出 TFLite/SP tokenizer/LlmMetadata/HF tokenizer/二进制数据, 含 zlib 解压。
- `/Users/dpthinker/workspace/LiteRT-LM/schema/core/litertlm_read.h` — 读取层公共接口: IsLiteRTLMFile 嗅探、LitertlmHeader 结构体、三种重载 ReadHeaderFromLiteRTLM, 以及 ReadAny/ReadFromSection 系列声明。
- `/Users/dpthinker/workspace/LiteRT-LM/schema/core/litertlm_section.h` — 写入侧段数据源抽象: SectionStreamBase 接口及 File/ProtoBuf/String/Zlib 四种实现, 把不同来源统一成可顺序读取的 stream。
- `/Users/dpthinker/workspace/LiteRT-LM/schema/core/litertlm_utils.h` — 含 MemoryStreamBuf: 把指针+长度内存块包装成可 seek 的 std::streambuf(替代 C++23 std::spanstream), 让 header 读取能从 mmap 内存直接进行; 及 AnySectionDataTypeToString 声明。
- `/Users/dpthinker/workspace/LiteRT-LM/schema/core/litertlm_print.cc` — ProcessLiteRTLMFile 实现: 读 header 后以 ASCII 边框+ANSI 加粗打印版本、SystemMetadata 与各 section 的 items/offset/类型, 对 LlmMetadata 段反序列化打印 DebugString。
- `/Users/dpthinker/workspace/LiteRT-LM/schema/capabilities/speculative_decoding.cc` — HasSpeculativeDecodingSupport 实现: 读 header 后遍历 TFLiteModel section 的 items, 若 model_type 属于已知 drafter 白名单(tf_lite_mtp_drafter)则判定支持投机解码。
- `/Users/dpthinker/workspace/LiteRT-LM/schema/core/litertlm_header.h` — 写元数据辅助: CreateKeyValuePair<T> 模板用 if constexpr + ValueTypeTraits 把标量/字符串构造成 FlatBuffer KeyValuePair; 并定义 LITERTLM_MAJOR/MINOR/PATCH_VERSION(当前 1.5.0)。

## 核心抽象
- **LiteRTLMMetaData** (flatbuffer table) 〔`schema/core/litertlm_header_schema.fbs`〕：FlatBuffer 根表(root_type), 是整个 header 的入口。包含 system_metadata(对外可见键值元数据)和 section_metadata(内部分段索引表)。读取时用生成函数 GetLiteRTLMMetaData(buffer) 把字节缓冲解释成该表, 无需反序列化。
- **SectionObject / AnySectionDataType** (flatbuffer table + enum) 〔`schema/core/litertlm_header_schema.fbs`〕：SectionObject 描述一个数据段: items(键值属性如 model_type/quantized)、begin_offset、end_offset(数据落在 [begin,end) 区间)、data_type。data_type 取自 AnySectionDataType 枚举(TFLiteModel/SP_Tokenizer/LlmMetadataProto/HF_Tokenizer_Zlib/GenericBinaryData/TFLiteWeights 等)。枚举顺序不可重排或删除, 否则需升 major 版本。
- **LitertlmHeader** (struct) 〔`schema/core/litertlm_read.h`〕：解析后的 header 内存表示。持有 unique_ptr<uint8_t[]> buffer(拥有原始 FlatBuffer 字节)和指向其内部的 const LiteRTLMMetaData* metadata, 外加 major/minor/patch_version。禁拷贝、支持移动; reset() 替换 buffer 时重新调用 GetLiteRTLMMetaData 更新 metadata 指针。
- **ReadHeaderFromLiteRTLM** (function (3 overloads)) 〔`schema/core/litertlm_read.cc`〕：格式解析核心入口。istream 版是真正实现: 依次读 8 字节 magic、3 个 uint32 版本号、4 字节 padding、uint64 的 header_end_offset, 据此算出 header 大小并读入 buffer。另两个重载分别接收文件路径(开 ifstream)和 void*+length(经 MemoryStreamBuf 包装)。会校验 magic 与 major version 一致性。
- **ReadValueTFromSection / ReadAnyT** (function template) 〔`schema/core/litertlm_read.cc`〕：读取层两个泛型骨架。ReadValueTFromSection<SectionT,T> 校验 section_idx 合法性与 data_type 是否等于期望类型, 算出 [begin,end) 后委派给具体 read_section_into_t 回调。ReadAnyT 遍历所有 section 找第一个匹配类型的段再读。对外暴露为 ReadTFLiteFileFromSection/ReadSPTokenizerFromSection/ReadLlmMetadataFromSection/ReadHfTokenizerJsonFromSection/ReadBinaryDataFromSection 及对应 ReadAny 便捷版。
- **SectionStreamBase** (abstract class) 〔`schema/core/litertlm_section.h`〕：写入侧统一数据源接口, 含 Prepare()/GetStream()/IsReady()/Finalize()/BufferSize()。四个实现: FileBackedSectionStream(读文件入内存)、ProtoBufSectionStream<T>(直接序列化 protobuf 到流)、StringBackedSectionStream(字符串)、ZlibBackendedSectionStream(包装另一个流并 zlib 压缩, 前置 8 字节未压缩长度)。让打包工具用同一套逻辑写不同类型 section。
- **MemoryStreamBuf** (class) 〔`schema/core/litertlm_utils.h`〕：继承 std::streambuf, 把已知 buffer(char*+length)暴露成只读输入流, 重写 underflow 与 seekoff 以支持 tellg/seekg。用于 ReadHeaderFromLiteRTLM(void*,length) 版本, 使其能在不复制的前提下从 mmap 内存解析 header。注释明确说是 C++23 std::spanstream 的临时替代。
- **HasSpeculativeDecodingSupport** (function) 〔`schema/capabilities/speculative_decoding.cc`〕：能力探测函数(istream 版与路径版)。读 header 后遍历每个 TFLiteModel section, 查其 items 中 key 为 model_type 的字符串值, 若命中白名单 tf_lite_mtp_drafter 即认为文件内含 MTP drafter、支持投机解码。体现了用 section 的 KeyValuePair 元数据声明能力的设计。
- **CreateKeyValuePair / DecompressData** (function template / function) 〔`schema/core/litertlm_header.h`〕：CreateKeyValuePair<T>(litertlm_header.h) 用 if constexpr + ValueTypeTraits 把任意 C++ 标量/字符串构造成 FlatBuffer 的 KeyValuePair(选对 VData union 分支), 是写元数据的辅助。DecompressData(litertlm_read.cc) 读取前 8 字节未压缩长度后用 zlib uncompress 解压, 带 1GB 上限防护, 服务于 HF_Tokenizer_Zlib 段。

## 数据流
1. 写入(打包)方向: litertlm_export_main 把 tokenizer/tflite/llm_metadata/binary 各自包成一个 SectionStreamBase(文件->FileBacked、proto->ProtoBuf、需压缩->Zlib 包装), Prepare() 后顺序写入文件, 每段按 16KB 块对齐, 记录 begin/end_offset。
2. 构建 header: 用 CreateKeyValuePair 把每段属性写成 KeyValuePair, 组装成 SectionObject(含 offset 与 data_type)与 SystemMetadata, 序列化成 FlatBuffer LiteRTLMMetaData。
3. 落盘前缀: 文件开头写 8 字节 magic LITERTLM + major/minor/patch 各 uint32 + 4 字节 padding + uint64 header_end_offset, 紧接着是 FlatBuffer header 字节, 再后面是各 section 的实际数据。
4. 读取方向: ReadHeaderFromLiteRTLM 校验 magic 与 major version, 跳过 padding, 读 header_end_offset 算出 header 大小并读入 LitertlmHeader.buffer, GetLiteRTLMMetaData 把它解释为 metadata。
5. 按段取数据: 调用 ReadAnyTFLiteFile/ReadAnySPTokenizer/ReadAnyLlmMetadata/ReadAnyHfTokenizerJson 等, ReadAnyT/ReadValueTFromSection 在 section_metadata 中按 data_type 定位段, 校验类型后用 begin/end_offset。
6. 实际加载: TFLite 段经 MMAPAllocation/MemoryMappedFile 零拷贝 mmap 进 FlatBufferModel; SP tokenizer 用 LoadFromSerializedProto; LlmMetadata 用 ParseFromString; HF tokenizer 段先 ReadSectionIntoBinaryData 再 DecompressData(zlib)还原 JSON。
7. 下游消费: runtime/util/litert_lm_loader.cc 在 Initialize() 里调用 ReadHeaderFromLiteRTLM, 遍历 sections 记录每段 (begin,end) 偏移, 后续 GetSectionBuffer 按需 mmap 交给推理引擎。

## 概念
- **FlatBuffer**：Google 的零拷贝序列化库。与 Protobuf 不同, FlatBuffer 序列化后的字节本身就是可直接访问的内存布局, 读取时无需解析/反序列化即可用偏移直接取字段, 非常适合端侧加载大模型 header(读 header 时几乎不产生额外内存与 CPU 开销)。.fbs 是它的 schema 定义语言, 编译后生成 generated.h。
- **.litertlm 容器文件格式**：一种自描述的打包文件: 开头是固定二进制前缀(magic + 版本 + header 偏移), 接着是 FlatBuffer 写的 header(描述有哪些段、每段类型与字节区间), 最后是各段连续排布的原始数据(TFLite 权重、tokenizer、元数据等)。一个文件即可承载一个端侧 LLM 所需的全部产物。
- **Section(分段)与 offset 对齐**：文件被切成若干段, 每段在 header 里有 begin_offset/end_offset 标明数据落在 [begin,end) 字节区间。schema 注释规定下一段起点须对齐到 BLOCK_SIZE=16*1024 的整数倍, 使大段(如模型权重)能页对齐、便于 mmap。
- **SemVer 与向后兼容规则**：文件格式用语义化版本 MAJOR.MINOR.PATCH(当前 1.5.0)。规则写在 .fbs 注释里: 纯新增 section 类型只升 MINOR; 重排或删除 section 必须升 MAJOR。读取器只在 major version 匹配时才继续, 否则返回 UnimplementedError, 保证旧 reader 不会误读不兼容文件。
- **Magic number(幻数)**：文件最前面的 8 字节固定标识 LITERTLM, 用来快速判定一个文件/流是否是 LiteRT-LM 容器(IsLiteRTLMFile), 避免把任意文件当成模型解析。
- **VData union 与 KeyValuePair**：FlatBuffer 的 union 让一个值能是多种类型之一(UInt8..Double/StringValue)。KeyValuePair 把字符串 key 与一个 VData 值绑定, 用来给文件和每个 section 附带灵活的属性(如 model_type=tf_lite_mtp_drafter、quantized=true), 是能力声明与元数据的通用载体。
- **Speculative decoding(投机解码)**：一种 LLM 推理加速技术: 用一个小而快的 draft 模型一次猜出多个 token, 再由大模型一次性并行验证、接受其中正确的前缀, 从而减少大模型的串行步数。本模块不实现该算法, 只通过 section 元数据(model_type 是否为已知 drafter)来声明/探测文件是否带有可用于投机解码的 drafter 模型。

## 优化
- **FlatBuffer 零拷贝读 header**：header 用 FlatBuffer 表示, GetLiteRTLMMetaData 直接把字节缓冲解释成 LiteRTLMMetaData* 而无需反序列化, 读取大量 section 元数据几乎不产生 CPU/内存开销。
- **section 16KB 块对齐 + mmap 零拷贝加载**：.fbs 规定每段起点对齐到 BLOCK_SIZE=16*1024, 使模型权重等大段页对齐; 读取 TFLite 时用 MMAPAllocation/MemoryMappedFile 按 [begin,end) 直接内存映射进 FlatBufferModel, 避免把大权重整体读入堆内存。
- **只 mmap header 前缀**：下游 litert_lm_loader.cc 只映射 min(kLitertLmHeaderMaxSize=16KB, file_size) 字节解析 header, 而非整文件, 随后各 section 按需 lazily mmap(GetSectionBuffer + 双重检查锁), 降低启动内存。
- **HF tokenizer zlib 压缩存储**：HuggingFace tokenizer 的 JSON 体积大, 用 HF_Tokenizer_Zlib 段以 zlib 压缩存放(写入侧 ZlibBackendedSectionStream 按 16KB 分块压缩、前置 8 字节未压缩长度), 读取时 DecompressData 解压并带 1GB 上限防御异常长度。
- **if constexpr 编译期分发的 CreateKeyValuePair**：用模板 + if constexpr + ValueTypeTraits 在编译期为不同标量类型选择正确的 VData union 分支, 既类型安全又无运行时分发开销。
- **MemoryStreamBuf 复用同一套解析逻辑**：通过自定义 streambuf 把内存缓冲伪装成 istream, 使路径版/内存版/流版三种 ReadHeaderFromLiteRTLM 共享同一份解析实现, 避免重复代码(临时替代 C++23 std::spanstream)。

## 关键代码片段（待核验 @ v0.13.1）
**二进制 header 前缀的精确布局与版本校验(magic->3 个 uint32 版本->4 字节 padding->uint64 header_end_offset)** — 待核验：`schema/core/litertlm_read.cc:73-111`
```cpp
litertlm_stream.read(magic_number, 8);
if (... std::string(magic_number, 8) != "LITERTLM") { return InvalidArgumentError(...); }
litertlm_stream.read((char*)&header->major_version, sizeof(uint32_t));
litertlm_stream.read((char*)&header->minor_version, sizeof(uint32_t));
litertlm_stream.read((char*)&header->patch_version, sizeof(uint32_t));
if (header->major_version != LITERTLM_MAJOR_VERSION) {
  return UnimplementedError(... "doesn't support version %d" ...);
}
litertlm_stream.ignore(4);  // skip padding
uint64_t header_end_offset;
litertlm_stream.read((char*)&header_end_offset, sizeof(uint64_t));
```
**按 section 类型读取的泛型骨架: 校验索引、校验 data_type、用 [begin,end) 委派给具体读取回调** — 待核验：`schema/core/litertlm_read.cc:190-215`
```cpp
const SectionObject* section = sections->Get(section_idx);
if (section->data_type() != SectionT) {
  return InvalidArgumentError(... "is not the expected type" ...);
}
size_t end_offset = section->end_offset();
size_t begin_offset = section->begin_offset();
if (begin_offset > end_offset) { return InvalidArgumentError(...); }
size_t data_size = end_offset - begin_offset;
if (data_size == 0) { return InvalidArgumentError(...); }
return read_section_into_t(litertlm_path, begin_offset, end_offset, data, ...);
```
**FlatBuffer schema: SectionObject 用 begin/end_offset + AnySectionDataType 描述每个数据段; 注释规定 16KB 块对齐** — 待核验：`schema/core/litertlm_header_schema.fbs:86-108`
```text
// Section i+1 begins no sooner than K * BLOCK_SIZE ... BLOCK_SIZE = 16 * 1024.
table SectionObject {
  items:[KeyValuePair]; // (optional)
  begin_offset:ulong;
  end_offset:ulong;
  data_type:AnySectionDataType;
}
table LiteRTLMMetaData {
  system_metadata:SystemMetadata;
  section_metadata:SectionMetadata;
}
root_type LiteRTLMMetaData;
```
**投机解码能力探测: 遍历 TFLiteModel 段的 model_type 元数据匹配 drafter 白名单** — 待核验：`schema/capabilities/speculative_decoding.cc:54-72`
```cpp
if (section_object->data_type() == AnySectionDataType_TFLiteModel) {
  const auto* items = section_object->items();
  for (size_t j = 0; j < items->size(); ++j) {
    const KeyValuePair* item = items->Get(j);
    if (item->key()->string_view() == "model_type") {
      const auto* value = item->value_as_StringValue();
      if (std::find(speculative_decoding_model_types.begin(),
                    speculative_decoding_model_types.end(),
                    value->value()->string_view()) != ...end()) {
        return true;
      }
    }
  }
}
```
**TFLite 段零拷贝加载: 用 MMAPAllocation 按 section 偏移直接 mmap 进 FlatBufferModel** — 待核验：`schema/core/litertlm_read.cc:226-233`
```cpp
std::unique_ptr<tflite::Allocation> mmap_alloc =
    std::make_unique<tflite::MMAPAllocation>(litertlm_path.c_str(),
                                             begin_offset, model_size,
                                             tflite::DefaultErrorReporter());
*tflite_model =
    tflite::FlatBufferModel::BuildFromAllocation(std::move(mmap_alloc));
```

## 入手顺序
- 先读 schema/core/litertlm_header_schema.fbs: 这是格式的权威定义, 弄清 KeyValuePair/SectionObject/AnySectionDataType/LiteRTLMMetaData 的字段含义, 以及关于 offset 对齐与 SemVer 的注释。
- 再看 schema/core/litertlm_read.h 的 LitertlmHeader 结构体与函数声明, 建立 header 在内存里长什么样、提供哪些读取入口 的整体印象。
- 然后读 litertlm_read.cc 的 ReadHeaderFromLiteRTLM(istream 版, 第 70-140 行), 对照真实文件的 hexdump 理解二进制前缀(magic+版本+padding+header_end_offset)的精确布局。
- 接着看同文件的模板 ReadValueTFromSection 与 ReadAnyT(第 170-216、471-500 行), 理解按类型定位 section 再委派读取 的统一套路, 以及各 ReadSectionInto 具体读法(mmap TFLite、Parse proto、zlib 解压 HF tokenizer)。
- 看 litertlm_utils.h 的 MemoryStreamBuf, 理解 void*+length 重载如何复用 istream 版逻辑; 再看 litertlm_section.h 的 SectionStreamBase 家族, 理解写入侧如何产出这些段。
- 最后读 schema/capabilities/speculative_decoding.cc, 看一个基于 section 元数据做能力探测的完整实例; 并扫一眼 runtime/util/litert_lm_loader.cc 的 Initialize() 看下游如何消费 header。
