# 尾声 · 实践入口与待验证问题

> 本章给出实践入口、参与项目的注意事项，以及仍需继续验证的问题。

正文从三类物理约束出发组织端侧推理问题：第 1 章量化约束并给出分析基线，第 2 至 5 章分析推理流水线，第 6 至 9 章讨论 KV cache、模型格式、异构后端和投机解码，第 10、11 章讨论多模态、工具调用与多语言绑定。涉及 LiteRT-LM 实现的机制统一核对 v0.17.0 源码，实验数字注明测量条件。来自官方文档或 issue 的结论保留各自的版本与来源边界。

## 1. 四种使用入口

源码编译不是使用 LiteRT-LM 的前置条件，按目标选择入口即可：

- 运行模型并做实验：使用第 2 章介绍的 Python CLI `litert-lm`。安装并运行模型后，可以改变参数并记录结果；需要从 Python 程序调用时，再使用 Python SDK。[^epilogue-cli]
- Android / JVM：使用 Kotlin SDK（第 11 章）与预编译 Maven 产物。[^epilogue-android] 依赖版本应显式固定，并与应用验证过的 LiteRT-LM 版本对应；不要使用 `latest.release`。`Engine`、`Session` 与 `Conversation` 的接口关系见第 3、11 章。
- iOS / macOS：使用 Swift package。[^epilogue-swift] 生命周期限制以冻结版头文件为准。第 11 章分别讨论 Conversation 与 Engine 的显式释放问题，不能把两个 issue 合并为同一接口结论。
- Web：使用 Web SDK，通过 npm 安装 `@litert-lm/core`，或从 CDN 以 `+esm` 导入。[^epilogue-web] 核心编译为 WASM 并在浏览器执行。数 GiB 的模型文件需随应用分发或在首次启动时下载，部署时要规划网络流量与缓存空间。

四种入口均可先从语言层 API 开始。出现部署失败、吞吐下降或模型加载错误时，可按模型文件、编排层、执行器和后端四个位置依次定位。

## 2. 建立可比较的最小基线

“模型可以运行”只能证明当前组合完成了一次调用。它不能作为版本升级、后端切换或参数调整的比较基线。可比较的基线需要固定输入、产物、运行时与测量方法。

| 类别 | 必须记录的字段 | 不记录时的歧义 |
|---|---|---|
| 运行时 | LiteRT-LM tag 或 commit、语言 SDK 版本、构建类型 | 无法判断接口与执行路径是否相同 |
| 模型 | 文件名、来源、SHA-256、量化类型、是否含 drafter、vision/audio 等 section | 同名文件可能不是同一导出产物 |
| 设备 | SoC、内存、操作系统、驱动或厂商运行时版本 | 后端能力与性能无法复现 |
| 后端 | cpu/gpu/npu、线程数、delegate 与相关开关 | “同一设备”仍可能使用不同执行器 |
| 输入 | 原始文本或 token id、图片/音频尺寸、对话模板 | tokenizer 与预处理差异会改变实际负载 |
| prefill | prompt token 数、signature 分块、预留上下文长度 | TTFT 与容量错误无法解释 |
| decode | 输出 token 数、采样参数、停止序列、MTP 和约束解码开关 | 输出内容和吞吐不具可比性 |
| 测量 | 预热次数、重复次数、统计量、温度与电源条件 | 单次墙钟时间容易混入冷启动和热降频 |

> 表 1　最小基线记录的是一组完整条件，而不是一个吞吐数字。附录 D 按同样原则保存本书实验。

建立基线时应先检查正确性，再测性能。第一阶段只要求加载成功、prefill 成功、decode 能停止，并保存首轮输出与错误日志。第二阶段固定第一阶段的模型和输入，只改变一个待测因素，例如后端或 `--max-num-tokens`。若同时改变模型、后端和 prompt，结果无法归因。

第 6 章的预留宽度对照实验（表 6-2）说明了这种控制方法。`4096/256` 这样的记号依次表示 `max_num_tokens` 与 prompt token 数：4096/256 与 8192/256 两组使用相同 prompt 和 prefill 计划，可以比较 decode 吞吐；1024/256 组在 `dynamic_update_slice` 处失败，而成功的 1024/100 记录改用了另一条 prompt 和 prefill 计划，只能作为独立样本。实验记录能否比较，取决于条件是否对齐，而不是表格中是否都有数字。

## 3. 从失败现象定位到执行层

端侧推理的调用链跨越模型容器、语言 API、会话编排、执行器和硬件后端，只保留最外层异常，通常不足以确定失败位置。图 1 按依赖关系给出一条从产物到调用边界的定位路径。

<figure>
{{#include figs/fig-e-1.svg}}
<figcaption>图 1　部署诊断按模型产物、Engine、prefill、decode 和语言边界逐层缩小范围；每一层都要求保留能区分上下游的证据。</figcaption>
</figure>

模型产物是第一项检查。确认文件摘要、section 类型、模型要求的后端以及运行时版本。`.litertlm` 可以包含主模型、drafter、tokenizer 和元数据；文件能够打开，不代表目标执行路径需要的 section 都存在。第 7、9 章分别说明容器读取和 MTP 能力查询。

Engine 创建失败时，问题尚未进入 prompt。应保留选定后端、编译日志、缓存路径和第一条底层错误。若 cpu 可以创建而 gpu 失败，两次测试应使用同一模型文件；除 backend 外，其余 Engine 设置保持不变。这样才能把调查范围缩小到后端编译、算子覆盖或设备初始化。

prefill 失败时，先记录 tokenizer 产生的 token 数、所选 signature、输入 shape 和 `max_num_tokens`。第 4 章说明固定 signature 的选择，第 6 章给出了容量过小时的 `dynamic_update_slice` 越界案例。把“长 prompt 失败”缩减成最短仍失败的 token 序列，比只提供自然语言原文更便于复现 shape 边界。

decode 阶段应分别记录停止 token 或停止序列、`max_output_tokens`、`max_num_tokens` 对应的 KV cache 末端，以及 benchmark 模式的固定步数。应用主动取消属于独立控制路径，也要单独记录。开启 MTP 后，一次 executor 调用还可能返回多个 token；第 9 章说明任务层如何逐 token 检查停止条件。约束解码或工具调用失败时，则要继续区分语法约束、参数语义与宿主执行。

最后检查语言边界。若 C++ 示例稳定，而 Kotlin、Swift 或 Web 路径失败，应把输入与 Engine 设置序列化为可比较记录。对象销毁线程、字符串所有权和回调生命周期属于绑定契约，不应归入模型数值问题。第 11 章列出了三类边界的所有权规则。

| 现象 | 首先保留的证据 | 下一组对照 |
|---|---|---|
| 模型文件无法打开 | 文件摘要、大小、下载来源、容器解析错误 | 同一文件换最小能力查询工具 |
| Engine 创建失败 | 后端、编译日志、缓存状态、设备运行时版本 | 同模型切换 cpu；清理后的独立缓存目录 |
| prefill 越界或 shape 错误 | token 数、signature、输入张量 shape、预留宽度 | 缩短 token 序列；固定同一 signature |
| decode 吞吐下降 | TTFT、decode tokens/s、上下文、采样、MTP、温度 | 只切换一个后端或一个开关 |
| 输出提前结束 | 停止 token/序列、`current_step`、输出上限、KV 上限、benchmark 步数、取消状态 | 关闭自定义停止条件后复测 |
| 绑定层崩溃 | 调用线程、对象创建/销毁顺序、回调是否仍在执行 | 使用等价 C++ 最小程序 |

> 表 2　每个对照只改变一个因素。若第一组证据已经位于上游层，就不必先分析下游采样结果。

### 3.1 诊断案例：切换 GPU 后吞吐下降

这个案例只演示诊断方法，不构成本书新增的 GPU 性能实测。假设同一模型从 cpu 切换到 gpu 后变慢，不能直接归结为“GPU 后端较慢”。先用同一模型摘要、prompt、上下文和输出长度重跑 cpu 基线。随后只改 backend，并分别记录 Engine 创建、prefill、首个 token 与后续 decode。Engine 创建时间变化时，检查编译、缓存与设备初始化。decode tokens/s 下降后，再检查采样位置、设备回传与持续负载。

隔离 backend 变量时，应显式固定 sampler backend，并关闭 MTP、约束解码和 repetition penalty 等非必要开关。当前实现在未显式指定 sampler 时，主 backend 的变化还可能改变采样位置；第 8 章说明了这一路径。若 GPU 组还开启设备侧采样或 MTP，就必须拆成额外对照。两项同时变化时，即使最终吞吐下降，也无法判断是采样数据路径还是 drafter/verify 成本造成。

输出内容也要保存。浮点后端可能在接近的 logits 上选择不同 token，后续自回归序列便会分叉。性能对照应固定采样设置，并把首个分叉位置作为结果的一部分。不同后端不保证逐 token 生成相同文本。

## 4. 参与上游开发

参与 LiteRT-LM 上游开发前，应先区分问题是否依赖特定硬件：

- 硬件相关问题包括特定 GPU/NPU 上的崩溃和特定 SoC 上的数值异常，需要对应设备才能复现。第 8、9 章引用的 `LiteRT-LM#2281`[^epilogue-issue-2281]、`LiteRT-LM#2227`[^epilogue-issue-2227] 属于这一类。
- 接口问题包括缺失的 API、误导性错误信息和边界处理。`LiteRT-LM#2589` 请求为 Swift Conversation 增加显式关闭接口；[^epilogue-issue-2589] `LiteRT-LM#2613` 讨论 Engine 在析构线程上的崩溃与显式关闭。[^epilogue-issue-2613] 两者涉及不同对象与失败条件，提交修改前必须分别核对当前版本。

两类问题需要不同的验证路径。硬件相关问题应保留设备、驱动、模型产物和运行时日志；接口问题可先定位公开 API、所有权契约和失败分支。上游的贡献说明当时明确写明仓库尚未开放代码贡献，建议先提交 issue；因此，本书不把准备补丁写成当前可用的上游流程。[^epilogue-contributing]

一个可复现的 issue 至少应包含以下六项：版本、模型摘要、设备与后端、完整命令或最小程序、预期结果、实际日志。性能问题还要提供预热和重复次数。回归问题应尽量给出最后正常版本与首个异常版本；若没有完成二分，应明确写出已测试的版本范围。

日志应覆盖失败点，但不要上传访问令牌、私有 prompt 或不能再分发的模型文件。模型不能公开时，可先尝试用公开模型复现同一接口或 shape 问题。若问题只在私有产物出现，应提供容器 section、张量 shape 和错误位置，并说明哪些内容无法共享。

修复接口问题时，还要补一条在修复前会失败的自动化测试。硬件问题若无法进入持续集成，可以把可独立验证的前置条件拆出来。例如，设备初始化无法在通用 runner 上运行，但模型能力检查、参数校验和错误信息仍可以覆盖。

## 5. 版本升级的双基线验证

本书锁定 v0.17.0。应用升级到后续版本时，不应直接用新版覆盖已验证环境。保留旧版基线，建立一个只改变运行时版本的并行环境。第一组对照保持模型文件和输入不变；若新版要求重新导出模型，再增加第二组“新版运行时 + 新模型”对照。

| 组合 | 运行时 | 模型 | 目的 |
|---|---|---|---|
| A | 已验证版本 | 已验证模型 | 保留当前正确性与性能基线 |
| B | 新版本 | 已验证模型 | 检查运行时兼容性与执行路径变化 |
| C | 新版本 | 新导出模型 | 验证新格式或新能力 |
| D | 已验证版本 | 新导出模型 | 仅在格式声明兼容时使用，用于分离模型变化 |

> 表 3　A 与 B 只改变运行时；B 与 C 只改变模型。D 是否成立取决于格式兼容声明，不能预设。

若 B 无法加载旧模型，应把该结果记录为兼容性变化，不能继续用 B、C 隔离模型差异。C 仍可验证“新运行时 + 新模型”能否工作，但相对 A 同时改变了两个因素。D 只有在旧运行时明确接受新格式时才成立。

升级测试先比较加载、prefill、停止行为和资源释放，再比较性能。缓存文件应按运行时和模型摘要隔离，避免把旧编译产物误当作新版结果。性能组需要重新预热，并同时报告旧、新两套结果的离散程度。

若输出发生变化，先定位首个不同 token，并检查 tokenizer、模板、采样参数和后端是否保持一致。若接口发生变化，则从语言绑定追溯到 C ABI 和 C++ 所有权契约。条件对齐后，再继续检查 kernel、delegate 或编译器、缓冲区和数值路径，不能预先把差异归到其中一项。

## 6. 待验证问题

- 低比特量化：收益取决于模型质量、kernel 支持、解包成本和实际瓶颈，不能只按文件体积判断。
- NPU 部署：当前需要核对算子覆盖、模型包与 SoC 的匹配、签名策略和厂商运行时。第 8 章的两台设备尚未完成 NPU 推理，后续实验需要匹配模型包与可执行环境。
- 工具调用与多步编排：约束解码可以保证输出符合给定语法，但函数权限、参数语义、失败恢复和多步状态仍由应用层处理。

表 4 列出三项实验的最小对照、指标和适用边界。

| 主题 | 最小对照 | 主要指标 | 不能省略的边界 |
|---|---|---|---|
| INT4 / INT8 | 同一基础 checkpoint、任务集和后端；保持其他导出选项不变 | 文件大小、峰值内存、TTFT、decode tokens/s、任务质量 | kernel 是否原生支持低比特；量化参数、校准集与验收阈值 |
| NPU 部署 | 匹配 SoC 与厂商运行时的模型包；用受支持的 cpu 路径做功能对照 | 创建阶段、prefill、decode tokens/s、峰值内存、功耗 | 产物不同时，cpu/NPU 差异不能归因于 backend 一项 |
| 工具调用 | 固定工具 schema 与各类样本数，测试合法、越权、缺参和执行失败输入 | 结构有效率、语义拒绝率、宿主执行成功率、恢复路径 | 模型输出不能直接获得宿主权限；参数仍需应用校验 |

> 表 4　三项待验证实验分别规定最小对照、指标和适用边界；未满足边界时不作性能或能力归因。

低比特实验不能只比较模型文件大小。应从同一基础 checkpoint 导出产物，固定校准集与导出工具选项，并在测试前声明任务质量阈值。实验至少记录理想权重字节数（参数量 × 位宽）、进程峰值内存和 decode 吞吐，并报告重复测量的分布。若后端把低比特权重解包到更高精度执行，磁盘体积下降也不等于相同比例的运行时收益。

NPU 实验先以“完成可重复的 prefill 和 decode”为阶段目标。两台设备分别停在 QNN context 创建之后与 backend/device 创建阶段，尚无可用于比较的 NPU 吞吐。下一轮应先固定匹配的模型包、SoC 代际与厂商运行时，再加入 cpu 功能对照和功耗记录。若 cpu 与 NPU 使用不同产物，cpu 组只能验证输入与任务链路，不能构成 backend 性能 A/B。

工具调用实验需要三层判定。第一层检查输出是否符合约束语法；第二层检查函数名、参数类型与业务规则；第三层才在受控宿主中执行。测试前应声明四类样本各自的数量与预期拒绝位置。多步调用还要保存每次工具结果、重试次数和会话状态。只报告 JSON 可解析率，不能证明工具调用能够安全完成。

## 7. 版本与测量边界

本书对模型文件、会话状态、prefill、decode、采样和后端分派的描述统一锚定 LiteRT-LM v0.17.0。切换运行时、版本、模型或设备后，接口与执行路径可能变化。内存预算、数据流和性能结论都需要按新的源码与测量条件重新核对。

对新的运行时和设备，应先固定模型与输入，验证正确性后再测量性能。每次只改变一个因素，并保存足以复算的原始记录。版本变化后的结论必须重新绑定源码锚点和实验条件。

[^epilogue-cli]: Google AI Edge，[*LiteRT-LM CLI*](https://developers.google.com/edge/litert-lm/cli)，更新日期：2026-07-09；访问日期：2026-07-18。

[^epilogue-android]: Google AI Edge，[*Get Started with LiteRT-LM on Android*](https://developers.google.com/edge/litert-lm/android)，更新日期：2026-05-28；访问日期：2026-07-18。

[^epilogue-swift]: Google AI Edge，[*LiteRT-LM Swift API*](https://developers.google.com/edge/litert-lm/swift)，更新日期：2026-06-01；访问日期：2026-07-18。

[^epilogue-web]: Google AI Edge，[*LiteRT-LM Web API*](https://developers.google.com/edge/litert-lm/js)，更新日期：2026-06-01；访问日期：2026-07-18。

[^epilogue-issue-2281]: 4ntoine，[*Different inference result depending on backend*](https://github.com/google-ai-edge/LiteRT-LM/issues/2281)，LiteRT-LM issue #2281，2026-05-15；访问日期：2026-07-18。

[^epilogue-issue-2227]: Shoolife，[*MTP / speculative decoding regresses decode tok/s on PowerVR GPU (Tensor G6) — even with GPU sampler fully loaded*](https://github.com/google-ai-edge/LiteRT-LM/issues/2227)，LiteRT-LM issue #2227，2026-05-11；访问日期：2026-07-18。

[^epilogue-issue-2589]: google-ai-edge/LiteRT-LM，[*[Swift] Add a public `close()` method to `Conversation` for deterministic session release*](https://github.com/google-ai-edge/LiteRT-LM/issues/2589)，LiteRT-LM issue #2589，2026-06-16；访问日期：2026-07-18。

[^epilogue-issue-2613]: google-ai-edge/LiteRT-LM，[*[Swift] Engine teardown crashes with `litert_lm_engine_delete` running on an arbitrary thread in `deinit` - adding a public `close()` to solve*](https://github.com/google-ai-edge/LiteRT-LM/issues/2613)，LiteRT-LM issue #2613，2026-06-19；访问日期：2026-07-18。

[^epilogue-contributing]: google-ai-edge/LiteRT-LM，[*CONTRIBUTING.md*](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.17.0/CONTRIBUTING.md)，v0.17.0；访问日期：2026-09-13。
