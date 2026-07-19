# 第 1 章 端侧 LLM：三类物理约束与运行时版图

> 本章目标：在明确假设的前提下，估算 batch=1、稠密模型 decode 的带宽侧吞吐上限。这个结果是分析基线，不是具体设备的性能承诺。

## 为什么要在端侧部署

以离线邮件改写为例：模型和输入都在手机上。用户需要等待片刻才能看到第一个 token，之后以每秒几个 token 的速度生成。模型结构可以完全一致，但手机的内存、算力和功耗预算远不及服务器，因此响应速度可能相差很大。

选择端侧部署，通常出于以下几方面考虑：

- 数据边界：当模型、输入与后处理均在设备上执行时，业务数据不必发送给推理服务。具体应用仍需单独审查日志、同步和遥测行为。
- 网络时延：本地执行可以去掉推理请求的网络往返。模型加载、prefill 和 decode 本身的耗时仍然存在。
- 服务成本：一部分推理计算与能耗转移到用户设备，可减少服务端 GPU 请求。商业模式、下载流量和设备能耗仍是独立成本。
- 离线可用性：模型文件、运行时和所需数据均已在本地时，推理可以不依赖网络连接。

需求成立后，还要判断设备是否具备运行条件。服务器可以通过增加加速卡或分布式执行扩展资源；移动设备的内存容量、内存通道和散热条件在运行时基本固定。端侧 LLM 因而同时受到内存容量、内存带宽、功耗与异构后端三类约束。

本章使用便于验算的示例参数，每处假设都会标出。模型文件大小不等于进程峰值内存，硬件标称带宽不等于有效带宽，单项能耗估算也不等于整机续航。附录 D 给出本书的实测数据，第 6 至第 9 章再结合 LiteRT-LM 的实现分析这些差异。

<figure>
{{#include figs/fig-1-1.svg}}
<figcaption>图 1-1　端侧 LLM 的三类约束分别影响配置可行性、batch=1 稠密 decode 的带宽上限和持续性能。</figcaption>
</figure>

## 第一类约束：内存容量

模型能否运行，取决于进程可用内存能否容纳权重、KV cache、激活张量、后端工作区和运行时开销。设备的物理内存还要供操作系统与其他进程使用，不能全部计入模型预算。

权重 payload 的理想字节数等于参数量乘以每个参数的位宽。FP32 每参数占 4 字节，FP16 占 2 字节，int8 占 1 字节。低比特量化可以减少权重 payload，但端侧模型并不必然采用同一种格式。本章用 int4 作示例，每个参数按 4 bit，即 0.5 字节计算：

$$ 4 \times 10^{9} \text{ 参数} \times 0.5 \text{ 字节/参数} = 2 \times 10^{9} \text{ 字节} \approx 1.86 \text{ GiB} $$

其中 1 GiB = \\(2^{30}\\) 字节。这个 1.86 GiB 只表示理想的 int4 权重 payload。实际文件还可能包含量化 scale、zero point、对齐填充和元数据，也可能包含其他模型段。参数量乘位宽不能直接代表文件大小。

LiteRT-LM 可以从只读 `MemoryMappedFile` 创建模型，接口见 `runtime/util/memory_mapped_file.h:46`。`ModelResources` 的注释说明，内存映射模型在实际使用时才分配物理页。对应代码见 `runtime/components/model_resources.h:148`。它还为“不应 mmap 的外部权重”保留独立文件接口，见 `runtime/components/model_resources.h:163`。

mmap 建立虚拟地址映射，不会保证 1.86 GiB 始终全部常驻，也不会消除运行期工作集。在 batch=1 的稠密 decode 中，如果模型远大于片上缓存，每一步通常会访问大部分有效权重。后端还可能复制或重排权重，并分配编译产物与工作区。因此，模型文件大小、虚拟映射大小、RSS 和进程峰值内存是四个不同口径。是否满足内存预算必须以目标后端的运行时测量为准。

自回归注意力会复用历史 token 的 Key 和 Value。KV cache 保存这些张量，避免每一步重新计算全部历史 K/V。忽略填充、量化元数据和多缓冲时，每个会话的逻辑容量可写成

$$ \text{KV cache 字节} = 2 \times L \times n_{kv} \times d_{head} \times T \times b $$

其中 2 表示 K、V 两份张量。\\(L\\) 是层数，\\(n_{kv}\\) 是 KV 头数，\\(d_{head}\\) 是每头维度。\\(T\\) 是已缓存 token 数，\\(b\\) 是每个元素的字节数。取一组示意参数：\\(L=32\\)、\\(n_{kv}=8\\)、\\(d_{head}=128\\)，KV 采用 FP16（\\(b=2\\)）。每个 token 的逻辑容量为

$$ 2 \times 32 \times 8 \times 128 \times 2 = 131072 \text{ 字节} = 128 \text{ KiB} $$

上下文为 4096 个 token 时，逻辑容量约为

$$ 128 \text{ KiB} \times 4096 = 0.5 \text{ GiB} $$

按同一组参数，上下文增至 8192 个 token，逻辑容量增至 1 GiB。真实模型的层数、GQA 配置、数据类型、预留长度与缓冲策略可能不同。128 KiB/token 不能外推到其他模型。

LiteRT-LM 对动态导出的 CPU 模型提供增量扩容参数。decode 时按 `CpuConfig::kv_increment_size` 增加 KV cache 大小，见 `runtime/executor/llm_executor_settings.h:107`。其默认值为 16（`runtime/executor/llm_executor_settings.h:110`）。较小的增量会增加 `Resize` 调用机会，较大的增量则可能提前预留更多空间。重分配次数、内存碎片和峰值占用仍取决于具体后端，需要测量。

激活与工作区的峰值无法用一个包含 \\(L\\) 和 \\(d_{model}\\) 的通用公式算出。FFN 中间维度、算子调度和张量生命周期都会影响它。buffer 复用、数据类型和 backend scratch buffer 也会改变结果。batch=1 的 decode 每次只处理一个新 token，激活张量通常小于长序列 prefill。未实测时不应给它指定固定数值，也不应忽略后端工作区。

`CpuConfig::prefill_chunk_size` 适用于动态导出的模型。缩小 chunk 可以降低峰值内存，见 `runtime/executor/llm_executor_settings.h:114`。默认值 -1 表示不分块（`runtime/executor/llm_executor_settings.h:117`）。例如，2048-token prompt 取 chunk=512 时，每次最多处理 512 个 token。这种切分最少需要四次调用。与序列长度相关的中间张量可能因此缩小。固定工作区、buffer 复用与后端调度会改变比例，不能断言峰值恰好降到四分之一。额外调用会使 TTFT 升高还是降低、变化多大，也应在目标设备上实测。

以下预算示例假定设备的物理内存为 8 GiB。操作系统和其他进程合计占用 4 GiB，模型进程可以使用余下 4 GiB。4B 模型的理想 int4 权重 payload 为 1.86 GiB；按本节示意参数，4096-token KV cache 为 0.5 GiB。两项已知 payload 合计 2.36 GiB，余下约 1.64 GiB。余量还要容纳量化附加数据、激活、后端工作区、权重复制或重排、运行时对象与分配器碎片。仅凭前两项不能判定模型一定能运行。

在相同假设下，理想 int8 权重约为 3.73 GiB。加上 0.5 GiB KV cache 后共约 4.23 GiB，尚未计入工作区就已超过 4 GiB。若仍采用 128 KiB/token 的示意参数，32K 上下文的 KV cache 为 4 GiB。它与 int4 权重合计约 5.86 GiB。这里得到的是条件判断：低比特权重和较短上下文更容易满足该内存预算，不是“int4 必然可用”的结论。

模型规模也应按同一方法判断。70B 权重按理想 int4 payload 计算约为 32.6 GiB。这还不包含量化附加数据、KV cache 和工作区，因而不满足上述 8 GiB 设备的题设。可行参数规模随设备可用内存、模型结构、量化格式与后端实现变化。这个示例不能规定端侧模型只能是 2B 或 4B。

> 对照视野
> 服务器可以通过更大显存或模型并行扩展容量；固定移动设备无法在运行时增加物理内存。端侧运行时因此需要同时管理权重表示、KV cache 预留和后端工作区。第 6、7 章分别分析前两项在 LiteRT-LM 中的实现。

## 第二类约束：内存带宽

LLM 生成包含 prefill 与 decode。prefill 一次处理多个输入 token，同一份权重可服务于多个 token。因此，它的算术强度通常高于 decode。具体工作点仍可能受内存、signature 填充率和 kernel 效率影响。decode 按自回归顺序生成 token。第 \\(N+1\\) 步依赖第 \\(N\\) 步的结果，单个会话不能任意并行这些步骤。

这里先限定 batch=1、稠密且所有参数均参与计算的 Transformer。若有效权重远大于片上缓存，每个 decode step 通常需要从主存访问大部分权重。MoE、结构化稀疏、权重常驻片上缓存或分层卸载会改变这项假设。

用 Roofline 记号可以同时写出两类上限。设每个 token 的数据搬运量为 \\(D\\) 字节，计算量为 \\(F\\) 次运算。工作负载可用的有效带宽记为 \\(B_{\mathrm{eff}}\\)，对应数据类型的有效计算吞吐记为 \\(P_{\mathrm{eff}}\\)。则

$$ R \leq \min\left(\frac{B_{\mathrm{eff}}}{D},\frac{P_{\mathrm{eff}}}{F}\right) $$

其中 \\(R\\) 的单位是 tokens/s。\\(B_{\mathrm{eff}}/D\\) 是带宽侧上限，\\(P_{\mathrm{eff}}/F\\) 是算力侧上限。只有前者更低时，增加计算吞吐才不会改变这个工作点的上限。

数值示例采用 4B 的理想 int4 稠密 payload，上下文较短，暂时忽略 KV cache。同时假定低比特 kernel 足够有效，量化解包、同步和采样没有先成为瓶颈。该工作负载可用的有效内存带宽设为 \\(B_{\mathrm{eff}}=50\\) GB/s。50 GB/s 只是题设，不对应某款 SoC 的规格或实测值。

短上下文时取 \\(D \approx 2\times10^9\\) 字节，因此带宽侧上限为

$$ R_{\mathrm{bw}}=\frac{50 \times 10^{9}\ \text{字节/s}}{2 \times 10^{9}\ \text{字节/token}}=25\ \text{tokens/s} $$

25 tokens/s 只在上述假设下成立。算力侧上限、低比特 kernel、同步或采样开销若更低，实际吞吐还会下降。提高 TOPS 不会改变 \\(B_{\mathrm{eff}}/D\\) 这一项，但在工作点尚未进入带宽约束区时仍可能提高实测吞吐。

上下文增长后，还要计算 KV cache。对普通全上下文注意力，先假设每步从 DRAM 读取全部逻辑 K/V。4096-token KV cache 会为本例增加约 0.5 GiB 的读取量。若后端只回写新增 K/V，写入量为 128 KiB，本例可暂时忽略。示例分母从 1.86 GiB 增至 2.36 GiB：

$$ \frac{50 \times 10^{9}}{2.36 \times 2^{30}} \approx 19.7\ \text{tokens/s} $$

这个计算仍是上限估算。缓存命中、预留区是否实际读取、注意力 kernel 和 KV 数据类型都会改变 DRAM 流量。附录 D 给出了 Gemma 4 E4B 的对应基准。上下文从 256 增至 4096 时，CPU decode 从 24.8 降至 20.7 tokens/s，约下降 17%。该现象与 KV 读写量和注意力工作量随上下文增长的分析一致。不能仅凭端到端吞吐就把降幅全部归于 KV cache。

有效带宽也不是硬件峰值的固定比例。访问模式、共享内存流量和缓存行为都会影响 \\(B_{\mathrm{eff}}\\)。解包计算和同步同样会影响吞吐，不能套用固定利用率。若公式从硬件标称峰值起算，结果只是更宽松的带宽侧上界。目标工作负载的有效带宽应通过硬件计数器或可比的基准测量获得。

KV cache 的缓冲策略还会影响容量。支持原地更新的后端可以让输入、输出指向同一组 KV buffer。不支持时，LiteRT-LM 返回两组 buffer 并在调用间交换。对应接口见 `runtime/executor/litert/kv_cache.h:70`。张量形状相同时，后一条路径可能需要接近两倍的 KV buffer 容量。整个进程的内存占用并不会因此翻倍。

硬件峰值带宽由设备决定，软件仍可提高有效带宽利用率，或减少每步读取的字节量。prefill 常因权重复用而更接近算力约束区，batch=1 的稠密 decode 则通常主要受带宽约束；二者都需要用实际 Roofline 工作点确认。量化与投机解码能否提高吞吐，还取决于 kernel 实现、接受率和额外计算开销，不能只看方法名称就下判断。

> 版本注记
> 4B、理想 int4 payload 与 50 GB/s 有效带宽均为示意假设，用于展示计算方法。附录 D 的 Mac、模型与后端参数不同；其实测值不能用来验证“25 tokens/s”这个点值，只能用于检验同一分析方法。

## 第三类约束：功耗与异构后端

峰值吞吐不等于持续吞吐。移动设备依赖电池供电，持续高负载可能提高芯片温度并触发系统的功耗或温控策略。评估端侧推理时，需要记录稳态吞吐、功率和温度随时间的变化，而不能只报告短时峰值。

DRAM 搬运能耗可以写成符号形式。设每个 token 的 DRAM 流量为 \\(D\\)，每字节能耗为 \\(e_{\mathrm{byte}}\\)，则搬运部分的能量近似为

$$ E_{\mathrm{mem/token}} \approx D \times e_{\mathrm{byte}} $$

\\(e_{\mathrm{byte}}\\) 随工艺、内存类型、访问局部性、功耗状态和测量边界变化，没有跨设备通用的常数。下面只展示代入方法，假定 \\(e_{\mathrm{byte}}=20\\) pJ/byte。再假定每 token 的 DRAM 流量为 \\(2\times10^9\\) 字节，则

$$ 2 \times 10^{9}\ \text{字节} \times 20 \times 10^{-12}\ \text{J/字节} = 0.04\ \text{J/token} $$

20 pJ/byte 是示意假设，不是本书设备的实测值。0.04 J/token 也只包含题设中的 DRAM 搬运。它不含算术运算、量化解包、显示、操作系统、无线电和电源转换的能耗，不能据此推算手机续航或可生成的 token 总数。整机能耗需要在目标设备上测量。

减少 DRAM 流量可以同时降低带宽占用和搬运能耗，但只有在这些成本占主要部分时，收益才接近线性。线程数、核心绑定和异步调度会改变吞吐、功率与温度，参数选择应依据持续负载实验。

LiteRT-LM 的 Android 路径提供一个设备相关的绑核实例。`cpu_affinity_utils.cc` 读取 SoC 厂商与型号属性，逻辑见 `runtime/engine/cpu_affinity_utils.cc:64`。固定的中核和大核编号定义在 `kTensorAffinities` 中，见 `runtime/engine/cpu_affinity_utils.cc:56`。例如 G4 返回 `{4,5,6,7}`，G5/G6 返回 `{2,3,4,5,6,7}`。`SetCpuAffinity` 再调用 `sched_setaffinity`，见 `runtime/engine/cpu_affinity_utils.cc:103`。源码注释只说明该选择用于性能优化，没有给出能耗或温控测量。不能据此断言该配置在所有负载下最省电。

异构后端的可用性还取决于设备和模型。CPU 的通用性较强；GPU 与 NPU 的可用算子、驱动和内存路径随设备而变。专用加速器在受支持的图上可能具有能效优势。离开模型、量化格式、delegate 和设备测量，不能给 CPU、GPU、NPU 作固定排序。

不同后端的算子实现和浮点路径可能产生数值差异。在自回归生成中，早期差异会影响后续 token。数值路径不同可能导致输出差异，但不能据此认定所有差异都不是缺陷。异常结果仍需用容差测试、逐层对比和已知 issue 排查。第 8 章讨论相关的 `LiteRT-LM#2281`。[^ch01-issue-2281]

LiteRT-LM 定义了 `Backend` 枚举，见 `runtime/executor/executor_settings_base.h:34`。通用名称包括 `CPU`、`GPU` 和 `NPU`，另有 `UNSPECIFIED`。其余三项是 `CPU_ARTISAN`、`GPU_ARTISAN` 和 `GOOGLE_TENSOR_ARTISAN`。源码把 CPU/GPU Artisan 标为 hand-written path。Google Tensor 项标为 Emission Graph。枚举只能证明运行时暴露了多条可选执行路径，不能证明某条路径具有更高性能或能效。具体实现与测量见第 8 章。

配置评估必须同时检查内存预算、Roofline 工作点、持续功耗和后端支持。量化减少权重 payload，也可能增加解包计算或改变后端支持范围。投机解码试图用一次目标模型调用确认多个 token，但 drafter 与验证也有额外成本；`LiteRT-LM#2227` 记录了特定 GPU 条件下的负收益。[^ch01-issue-2227]

## 同类项目对照

多个开源项目都支持在受限设备上运行 LLM。表 1-1 按各项目的官方仓库与文档记录接口侧重、后端和模型产物。

| 运行时 | 接口与实现 | 文档侧重 | 官方列出的执行后端 | 典型模型产物 |
|---|---|---|---|---|
| **llama.cpp**[^ch01-llama] | C/C++ 本地运行时 | 本地推理与跨平台部署 | CPU 与多种 GPU 后端 | GGUF |
| **MLC-LLM**[^ch01-mlc] | 编译栈与运行时 | 经 TVM 编译后部署 | CUDA、Vulkan、Metal、WebGPU 等 | 按目标生成的编译产物 |
| **ExecuTorch**[^ch01-executorch] | PyTorch 导出与 C++ 端侧运行时 | PyTorch 模型的端侧部署 | portable CPU 与设备 delegate | `.pte` |
| **LiteRT-LM**[^ch01-litertlm] | C++ LLM 运行时 | Engine/Session 编排与 LiteRT 执行 | `Backend` 枚举中的 CPU、GPU、NPU 与 Artisan 路径 | `.litertlm` |

> 表 1-1　端侧 LLM 运行时对照：各项目的接口、后端抽象和分发产物不同，表中不作性能排序。

LiteRT-LM 在 LiteRT 之上提供面向 LLM 的模型加载、Engine/Session 和生成编排。Google Developers Blog 列出的部署包括 Chrome、Chromebook Plus 与 Pixel Watch。[^ch01-litertlm-deploy] 仓库 README 还提供 Google AI Edge Gallery 的应用入口。[^ch01-edge-gallery] 这些材料说明项目已有公开部署，但不能替代特定设备、模型和版本的基准测试。

本书选择 LiteRT-LM，是因为其核心代码公开，且可以把文字说明锚定到冻结的 `v0.13.1`。对其他项目的讨论只用于比较实现选择，不据此给出统一的优劣结论。

## 小结

内存预算必须同时覆盖权重 payload、KV cache、激活、后端工作区和运行时开销。低比特量化是减少权重 payload 的常用选择，但是否可运行仍要以目标后端的峰值内存为准。

在 batch=1、稠密模型、权重远大于片上缓存且 kernel 有效的条件下，decode 通常主要受内存带宽约束。本章的题设是 4B 理想 int4 payload 和 50 GB/s 有效带宽。据此算得的 25 tokens/s 是带宽侧上限。完整 Roofline 上限还要与算力侧比较，真实上下文还要加入 KV cache 与其他流量。

功耗与后端决定峰值能否持续。20 pJ/byte 只是示意假设，不能推导整机续航；CPU、GPU 与 NPU 的性能和能效也必须针对模型与设备测量。

## 后续章节

第 2 章用 benchmark 与 Roofline 解读实测数据。第 3 至第 5 章说明从输入到输出的推理流水线。第 6 至第 9 章再分析 KV cache、模型加载与量化、异构后端和投机解码。

---

## 练习与自查

1. **内存预算复算。** 一台 12 GiB 手机，题设假定操作系统与其他 App 合计占用 5 GiB。按 KV cache 每 token 128 KiB 计算。求理想 int4 7B 权重与 8K 上下文 KV 的已知 payload。仅凭这两项能否判定可运行？上下文增至 32K 时呢？
2. **带宽侧上限复算。** 一款 SoC 使用 LPDDR5X-9600 与 64 bit 总线。先由数据率与总线宽度推导理论峰值带宽，再估算 int8 4B 稠密模型的带宽侧 decode 上限。说明为什么它不是持续性能承诺。
3. **能耗假设变体。** 题设明确假定 \\(e_{\mathrm{byte}}=20\\) pJ/byte。若 int4 2B 模型每 token 产生约 1 GB DRAM 流量，计算搬运能耗。再计算把 15 Wh 全部用于这一项时的算术上界，并说明它为什么不代表设备续航。
4. **Roofline 判断。** 解释为什么 batch=1 的稠密模型通常表现为 prefill 算术强度较高、decode 算术强度较低。再列出一个会破坏该简化判断的条件。
5. **条件判断。** 某 decode 工作点已确认受带宽约束。若芯片计算吞吐翻倍而有效内存带宽不变，带宽侧上限是否变化？若尚未确认瓶颈，为什么不能直接作答？

[^ch01-issue-2281]: 4ntoine，[*Different inference result depending on backend*](https://github.com/google-ai-edge/LiteRT-LM/issues/2281)，LiteRT-LM issue #2281，2026-05-15；访问日期：2026-07-18。
[^ch01-issue-2227]: Shoolife，[*MTP / speculative decoding regresses decode tok/s on PowerVR GPU (Tensor G6) — even with GPU sampler fully loaded*](https://github.com/google-ai-edge/LiteRT-LM/issues/2227)，LiteRT-LM issue #2227，2026-05-11；访问日期：2026-07-18。
[^ch01-llama]: ggml-org，[llama.cpp](https://github.com/ggml-org/llama.cpp/tree/b9873)，tag `b9873`；访问日期：2026-07-18。
[^ch01-mlc]: MLC LLM，[*Welcome to MLC LLM*](https://llm.mlc.ai/docs/)，文档版本 0.1.0；访问日期：2026-07-18。
[^ch01-executorch]: PyTorch，[ExecuTorch](https://github.com/pytorch/executorch)，GitHub 仓库；访问日期：2026-07-18。
[^ch01-litertlm]: Google AI Edge，[LiteRT-LM README](https://github.com/google-ai-edge/LiteRT-LM/tree/v0.13.1)，版本 v0.13.1；访问日期：2026-07-18。
[^ch01-litertlm-deploy]: Google Developers Blog，Yu-hui Chen、Ram Iyengar，[*On-device GenAI in Chrome, Chromebook Plus, and Pixel Watch with LiteRT-LM*](https://developers.googleblog.com/on-device-genai-in-chrome-chromebook-plus-and-pixel-watch-with-litert-lm/)，2025-09-24；访问日期：2026-07-18。
[^ch01-edge-gallery]: Google AI Edge，[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)，GitHub 仓库；访问日期：2026-07-18。
