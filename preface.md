# 前言

与大语言模型对话，如今已是常事。常见的体验来自云端聊天产品：提问、贴代码、传文档，回答逐字出现在屏幕上，还可以来回追问几轮。这些回答由数据中心里的加速器算出。

本地模型也开始用于手机、手表和浏览器。设备的可用内存、芯片性能与供电条件各不相同。部署数十亿参数的模型，先要核对内存容量与带宽。持续生成还受功耗和发热限制。系统地回答这些工程问题，是本书的任务。

本书的分析对象是生产级端侧运行时 LiteRT-LM。它是什么、与 LiteRT 在软件栈中如何分工，见第 1 章 1.1 节。在进入实现之前，先交代两件事：端侧模型目前的能力水平，以及 Google 为端侧部署提供了哪些配套。

## 云侧与端侧模型

Gemma 4 的 31B 稠密模型支持 256K 上下文、多模态输入与原生函数调用。2026 年 4 月 2 日的发布文章引用 Arena AI 文本榜单：31B 在开放模型中排第 3 位，26B MoE（Mixture of Experts，混合专家）排第 6 位。[^preface-gemma4-launch] 这些名次描述发布时点的一项评测，不能据此概括所有任务的能力，也不决定模型应部署在云侧还是端侧。

本书关注的是资源受限的设备上能提供哪些功能。端侧部署的价值、边界与代价见第 1 章 1.2 节。下面对照两个时间点的模型功能与部署条件：

- 2023 年 7 月发布的 Llama 2（7B、13B、70B 三档参数）是纯文本模型，上下文长度为 4096 个 token。[^preface-llama2] 同年 9 月 1 日开始训练的 TinyLlama 选择了 1.1B 参数规模，面向计算和内存受限的应用。[^preface-tinyllama]
- 2026 年 4 月 2 日发布的 Gemma 4，让支持文本、图像、视频与音频输入和 128K 上下文的模型能在手机和笔记本的内存限制内运行。官方公布的部署数据称，其中 E2B（2B 有效参数）使用 2-bit/4-bit 权重与按层内存映射的 embedding，可在部分设备上以不足 1.5 GB 的内存运行；decode 吞吐在 CPU 上（树莓派 5）约 7.6 tokens/s，在 NPU 上（Qualcomm Dragonwing IQ8）约 31 tokens/s。[^preface-gemma4-edge]

这些部署资料说明，端侧开放模型已能提供多模态输入与长上下文支持。[^preface-gemma4-edge] 上下文长度、输入模态和吞吐分别描述功能范围与运行性能，不能据此判定不同模型的任务能力相当。比较任务能力，还需要在相同评测集上检验输出质量。

华为的 Mate XT 2 官方资料将 30B MoE 列为端侧模型配置。[^preface-huawei-moe] 稀疏激活让总参数量与每个 token 使用的参数量分开，容量与带宽也需要分别计算。第 10 章以本书实测的 Gemma 4 为例，核算专家工作集，并与生成及内存数据对照。

## Google 的端侧方案

只有模型还不够，还需要把模型部署到设备上的运行时与工具链。Google 同时提供模型、运行时与部署平台。Gemma 4 是 Google DeepMind 在 2026 年推出的开放模型家族，官方将其定位为“可在云端、笔记本电脑和手机上部署的开放模型”，其中 E 系列专门面向边缘设备。[^preface-gemma-family] 发布当日，Google Developers Blog 同步公布了端侧配套方案：AI Edge Gallery 示例应用、Agent Skills 技能库以及 LiteRT-LM 部署路径。[^preface-gemma4-edge] 在 Android 侧，AICore 预览版已将 Gemma 4 定位为下一代 Gemini Nano 的基础模型。[^preface-gemma4-aicore] 同一时期，GDG China 的 Gemma 4 开发者大赛要求用 E2B/E4B 在真实硬件上演示完全离线的端侧部署。[^preface-gemma4-hackathon]

模型、运行时与部署平台出自同一家厂商，使 LiteRT-LM 值得作为本书的分析对象。它需要解决的问题——内存容量与带宽约束、异构后端调度、投机解码的接受率、多模态 embedding 路径、约束解码的信任边界——并非 LiteRT-LM 独有，其他端侧推理系统在实现同类功能时同样要处理。

除了问题本身足够通用，选择它还有两个原因。其一，它有公开的产品部署记录：Google Developers Blog 记载了它在 Chrome、Chromebook Plus 和 Pixel Watch 中的实际应用，[^preface-litertlm-deploy] Google AI Edge Gallery 则通过示例应用展示了端侧模型部署的完整链路。[^preface-edge-gallery] 这些场景要求运行时能适配不同设备，而不只是在单一 benchmark 指标上占优。其二，它的源码覆盖了端侧推理的关键技术：KV cache 管理与双缓冲、GPU 设备侧采样、量化权重的加载与后端分派、投机解码、多模态输入、约束解码、工具调用和 LoRA。每一项实现都能在固定版本的源码中定位到具体位置，书中贴出的代码片段都带出处（体例见“关于代码引用与数字”一节）。

## 这本书讲什么

本书分析生产级端侧运行时的实现与工程权衡。它并非 Google 官方出版物，书中的所有观点及可能存在的错漏，均由作者负责。

本书以 LiteRT-LM v0.17.0 为主要分析对象，回答一个核心问题：大语言模型如何在手机、手表和浏览器等受限设备上运行。主基准模型 Gemma 4 E4B 在 4B 有效参数规模下提供多模态理解与函数调用能力，模型产物以单文件 `.litertlm` 形式分发。[^preface-gemma-e4b]

本书不是使用手册，也不是逐行代码注释，而是关注**实现与权衡**：代码采用了什么设计，每种设计依赖哪些条件，又在哪些场景下受到限制。不同运行时的具体实现会变化，但都可以从内存预算、数据流、后端约束和测量口径这几个方面逐项分析。

模型定义了计算与能力，运行时则负责在具体设备上加载、调度和执行。本书同时分析模型产物与运行时代码：既用 `litertlm_print` 检查 `.litertlm` 的 section 布局，也跟踪 Engine 如何读取这些 section 并创建后端资源。只有同时核对文件格式、执行路径和设备约束，读者才能独立判断一次部署失败出在哪一层。

## 推理流水线

全书按照推理流水线组织：输入文本先转换为 token 序列，经 prefill（并行处理整段输入）建立上下文；随后进入 decode，每个 decode step（一次前向与采样的迭代）产出一个 token，直到满足停止条件；投机解码让一次前向确认多个 token，见第 9 章。

第一部分（第 1、2 章）先把端侧设备的内存、带宽与功耗约束量化成可验算的数字，再引入贯穿全书的指标与分析方法。第二部分（第 3 至 5 章）按调用顺序分析这条流水线：输入编排、prefill、decode 与采样。第三部分（第 6 至 10 章）讨论 KV cache、模型格式、异构后端、投机解码与 MoE。第四部分（第 11、12 章）处理多模态输入、约束解码、工具调用与多语言绑定。

## 写给谁

本书假定读者熟悉 C++，能够读懂模板和智能指针相关代码，并了解 Transformer 的基本原理。正文不再重复这些基础知识；KV cache 双缓冲、投机解码接受率等运行时概念则会逐步展开。

本书面向从事端侧模型产品集成的工程师，也面向想读懂一个生产级推理运行时实现的学生与研究者。读完后，读者应能沿源码追踪一次完整的生成请求，从 `GenerateContentStream` 一直定位到逐 token 输出；能够在给定设备、模型和测量口径下解释性能数据，并判断新增后端、采样策略或语言绑定会对哪些接口产生影响。

如果仅需调用 API，LiteRT-LM 官方文档更为直接。[^preface-litertlm-docs] 而当遇到部署失败或性能没有达到预期时，本书提供了一种按输入编排、执行器、模型资源和硬件后端逐层定位问题的方法。

## 阅读路径

- 顺序阅读时，可按四个部分依次推进，各部分覆盖的内容见“推理流水线”一节；尾声单独列出实践入口与待验证问题。
- 第 2 章列出二十个问题，每个问题均指向后续章节中的对应分析，可作为阅读索引。
- 阅读前无需先完成源码编译。第 2 章会先用一条命令运行模型；涉及实现细节时，关键机制附有源码片段，出处标在片段首行（引用体例见下一节），读者可直接对照冻结版本。

## 关于代码引用与数字

全书对 LiteRT-LM 的代码引用统一锁定在 `release/v0.17.0` 对应的 `v0.17.0` tag，冻结提交为 `e9fd8c53ff968071774206163027dd84bedfe925`。附录 D 的既有实验保留原采集版本与条件，其中 v0.13.1 数据不代表新版性能。为不打断行文，正文中不出现文件路径与行号；引用的代码以代码块呈现，出处标注在代码块首行注释里，不逐处重复版本号。对其他项目的引用在脚注中显式标注版本（如 llama.cpp 的 b9873）。代码之外的来源，在相关断言后以页下注给出。锁定冻结版本，可以保证文件、行号与实现描述保持一致；上游后续变化只收入“版本注记”侧栏。

第 10 章进一步分析 v0.17.0 所锁定的 LiteRT 依赖。LiteRT 代码块使用 `LiteRT/` 前缀，构建时同时核查其冻结提交和引文；具体版本与导出端分析范围见 10.6 节。

书中的理论上限，例如 decode 上限公式，均在正文中逐步推导，读者可据此验算。实测数据按设备与采集批次分别列示：Mac 主基准运行 Gemma 4 E4B，decode 实测每秒可生成数十个 token。2026 年 7 月的 Android 扩展基准使用 P0210 手机和自编译二进制。同年 9 月另在 HONOR MEP-AN00 上补充进程内存、持续吞吐、客户端文本时延与图片输入案例。这两台手机的结果分别记录，不混算。

方法与全部数据见附录 D，所有标注“〔基准 D〕”之处均指向该附录，理论估算与真机实测明确区分。进程内存读数与文本回调时延各有计量范围，不能代替完整的 GPU 内存或屏幕显示时刻。同一基础 checkpoint 的量化质量与性能对照尚未完成。NPU 执行行为等仅有代码分析的部分，书中也就地标明。

[^preface-tinyllama]: TinyLlama 项目，[TinyLlama/TinyLlama-1.1B-Chat-v1.0 模型卡](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)，项目训练启动日期：2023-09-01（非 Chat-v1.0 发布日期）；访问日期：2026-09-05。
[^preface-gemma4-launch]: Google DeepMind，Clement Farabet、Olivier Lacombe，[*Gemma 4: Byte for byte, the most capable open models*](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)，2026-04-02；访问日期：2026-08-04。
[^preface-llama2]: Meta AI，[*Meta and Microsoft Introduce the Next Generation of Llama*](https://ai.meta.com/blog/llama-2/)，2023-07-18；访问日期：2026-08-04。
[^preface-gemma4-edge]: Google Developers Blog，[*Bring state-of-the-art agentic skills to the edge with Gemma 4*](https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/)，2026-04-02；访问日期：2026-09-05。
[^preface-gemma4-aicore]: Android Developers Blog，[*Announcing Gemma 4 in the AICore Developer Preview*](https://android-developers.googleblog.com/2026/04/AI-Core-Developer-Preview.html)，2026-04；访问日期：2026-08-04。
[^preface-gemma4-hackathon]: GDG China，[Gemma 4 开发者大赛｜2026](https://hackathon.googdg.cn/)，报名 2026-04-18 至 2026-05-18，决赛在 2026 Google I/O Connect 中国站（2026-08）举行；访问日期：2026-08-04。
[^preface-litertlm-deploy]: Google Developers Blog，Yu-hui Chen、Ram Iyengar，[*On-device GenAI in Chrome, Chromebook Plus, and Pixel Watch with LiteRT-LM*](https://developers.googleblog.com/on-device-genai-in-chrome-chromebook-plus-and-pixel-watch-with-litert-lm/)，2025-09-24；访问日期：2026-07-18。
[^preface-edge-gallery]: Google AI Edge，[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)，GitHub 仓库；访问日期：2026-07-18。
[^preface-gemma-family]: Google DeepMind，[Gemma](https://deepmind.google/models/gemma/)；访问日期：2026-07-18。
[^preface-gemma-e4b]: Google AI Edge Community，[Gemma 4 E4B LiteRT-LM 模型卡](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm)；访问日期：2026-07-18。
[^preface-litertlm-docs]: Google AI Edge，[LiteRT-LM 官方文档](https://developers.google.com/edge/litert-lm)，更新日期：2026-07-09；访问日期：2026-07-18。
[^preface-huawei-moe]: 华为，[*HUAWEI Mate XT 2 | ULTIMATE DESIGN 卖点*](https://consumer.huawei.com/cn/support/content/zh-cn16114946/)，适用版本 HarmonyOS 7.0，“大屏 AI 再进化”；访问日期：2026-09-13。
