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

权重压到 int4 是一码事；模型跑起来时，中间的激活值用什么精度算，是另一码事。LiteRT-LM 把激活精度做成一个可配置项 `ActivationDataType`（`runtime/executor/executor_settings_base.h:62 @ v0.13.1`），四档：FLOAT32、FLOAT16、INT16、INT8（`:64`–`:73`）。

这给了工程师一个权衡空间：激活用 fp16 比 fp32 省一半显存和带宽，通常质量损失可忽略；进一步压到 int8 更省，但对某些模型精度影响变大。权重量化管"模型多大"，激活精度管"跑起来占多少、多快、多准"，两个旋钮各调各的。端侧调优，很大一部分就是在这两个旋钮之间找那个够快又够准的平衡。

## 为什么要自造一个文件格式

第 16 问：一个 `.litertlm` 文件里装了什么，为什么不直接用现成格式？

因为端侧要装的东西不止权重。要跑起来一个模型，你需要：权重、tokenizer、聊天模板、停止符、能力声明（比如支不支持推测解码，第 9 章）……如果这些散成一堆文件，分发、版本对齐、加载都是麻烦。`.litertlm` 把它们打包进**一个文件**。

它的结构是"头 + 分段"。头是一个 FlatBuffer，用类型化的键值对（`KeyValuePair`，`schema/core/litertlm_header_schema.fbs:56 @ v0.13.1`）记录元数据；正文是若干**段**（section），每段装一样东西——一段权重、一段 tokenizer、一段元数据。段的存储方式还挺灵活：可以直接指向文件里的一块字节（`FileBackedSectionStream`，`schema/core/litertlm_section.h:98 @ v0.13.1`），可以是一个 protobuf（`ProtoBufSectionStream`，`:189`），也可以是 zlib 压缩过的（`ZlibBackendedSectionStream`，`:252`）。

<figure>
{{#include figs/fig-7-1.svg}}
<figcaption>图 7-1　.litertlm 文件结构：一个 FlatBuffer 头（类型化键值对元数据）+ 若干段。段可以是文件字节、protobuf 或 zlib 压缩流。权重、tokenizer、模板、能力声明打包进同一个文件。</figcaption>
</figure>

这个"单文件 + 分段"的设计不是为了好看，是为了两件实事：**分发只需给一个文件**，以及下一节要讲的——**加载可以只挑需要的段、还能并行**。第 2 章那个 `litertlm_print` 工具（`schema/core/litertlm_print.cc`）做的，就是把这些段列出来给你看。

## 加载：mmap 与分段并行，凿冷启动的墙

模型文件常有几 GB。如果加载时老老实实把整个文件读进内存，冷启动要等很久——这是第三堵墙之外的又一个真实体感问题（用户点开 App，等模型加载的那几秒）。

`.litertlm` 的分段结构在这里第二次发力。读取时，LiteRT-LM 用 **mmap** 把文件映射进地址空间（`ReadHeaderFromLiteRTLM` 与按段读取的接口，`schema/core/litertlm_read.h:116`、`:151 @ v0.13.1`，注释里明说读到的是"mmapped buffer"）。mmap 的好处是：不必先把几 GB 拷进内存，操作系统按需分页——用到哪段才真正读哪段。

分段还带来并行的机会：各段互相独立，可以同时加载。这由一个开关控制（`litert_lm_engine_settings_set_parallel_file_section_loading`，`c/engine.h:295 @ v0.13.1`，默认开）。此外，编译后的模型产物也能缓存到磁盘（第 2 章 benchmark 见过的 `--cache disk`），二次启动直接复用、省掉重新编译。mmap 按需分页、分段并行、编译产物缓存——三招合起来对付的就是冷启动。其中编译缓存的效果在本书基准里直接可见：GPU 后端首次运行（缓存未热）Init 5.29 s，其后稳定在约 1.77 s〔基准 D〕；分段并行加载的开关对比本书未单测。

## LoRA：不动基座，换个人格

最后一块拼图：变体。你有一个通用基座模型，想让它在某个专门任务上更好——写代码、医疗问答、特定语气。重新训练或全量微调一个几 GB 的模型，端侧存不下也换不起。

LoRA 的思路是：**基座权重一个字节都不动，另外挂一小份"增量权重"**。推理时把增量叠加到基座上，模型行为就偏向新任务。增量很小（相比基座是零头），存得下、也能热加载。LiteRT-LM 用两个类支撑它：`LoRA`（`runtime/components/lora.h:40 @ v0.13.1`）持有一份增量权重，`LoraManager`（`runtime/components/lora_manager.h:39 @ v0.13.1`）负责按 id 加载、管理多份（`LoadLoRA`，`:57`）。

LoRA 的价值恰好呼应本章主题：它是"变体"的最省成本形态——一个基座 + 若干小增量，就能覆盖多个任务，而不必为每个任务存一个完整模型。放到端侧的存储约束下，这个省法尤其值钱。

## 小结

这一章讲的是模型的三种"形态变化"：量化把它压小（省体积、更省带宽，间接提 decode 速度），`.litertlm` 把它连同 tokenizer、模板、能力声明装进一个可 mmap、可分段并行加载的文件，LoRA 用小增量让它在不动基座的前提下适配新任务。三者共同回答一个问题：一个几十亿参数的模型，怎么以端侧扛得住的形态存在。

模型的形态清楚了。下一章回到运行时：同一个模型，为什么在 CPU、GPU、NPU 上跑起来速度甚至输出都不一样。

---

## 参考

- 激活精度：`runtime/executor/executor_settings_base.h:62 @ v0.13.1`（`ActivationDataType`，FLOAT32/16、INT16/8 于 `:64`–`:73`）。
- `.litertlm` 格式：`schema/core/litertlm_header_schema.fbs:56 @ v0.13.1`（`KeyValuePair`）；`schema/core/litertlm_section.h @ v0.13.1`（`FileBackedSectionStream`:98；`ProtoBufSectionStream`:189；`ZlibBackendedSectionStream`:252）；`schema/core/litertlm_read.h @ v0.13.1`（`ReadHeaderFromLiteRTLM`:116；按段读:151）。
- 并行加载：`c/engine.h:295 @ v0.13.1`。
- LoRA：`runtime/components/lora.h:40 @ v0.13.1`；`runtime/components/lora_manager.h @ v0.13.1`（`LoraManager`:39；`LoadLoRA`:57）。

<!-- 实测（int4 vs int8 三角、并行加载 on/off 冷启动、litertlm_print 实剖）待基准 D 回填〔基准 D〕；量化内部(分组/scale)未展开，仅到"权重压 4bit + 激活精度谱系"层面，未臆测未核验的细节。图 7-2(mmap/并行加载) 与表 7-1(量化三角) 规格见 notes.md，本轮出签名图 7-1。 -->
