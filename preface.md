# 前言

与大语言模型对话，如今已是常事。多数人的体验来自云端聊天产品：提问、贴代码、传文档，回答逐字出现在屏幕上，还可以来回追问几轮。屏幕背后，是数据中心里成排的加速器。

但同样的能力正在进入手机、手表和浏览器。这些环境中没有机房，只有几 GiB 内存、一颗功耗受限的芯片和一块电池。要让一个数十亿参数的模型在这样的设备上持续生成文字，必须先回答几个工程问题：内存是否放得下，带宽是否够用，功耗和发热是否允许持续推理。这些问题的答案决定了端侧部署是否可行；系统地回答它们，正是本书的任务。

本书的分析对象是生产级端侧运行时 LiteRT‑LM。它是什么、与 LiteRT 在软件栈中如何分工，见第 1 章 1.1 节。在进入实现之前，先交代两件事：端侧模型的能力走到了哪一步，Google 为此准备了什么。

## 云侧与端侧模型

云端依然是模型能力的前沿。OpenAI、Google、Anthropic 的闭源旗舰，DeepSeek、Kimi、GLM 等国产开放模型，以及 Gemma 4 这样的 Google 开放模型，都在上下文长度、多模态理解与原生工具调用上持续演进。以 Gemma 4 旗舰 31B 稠密模型为例，它在 256K 上下文窗口下支持多模态与原生函数调用；发布文章引用 Arena AI 文本榜单的口径，该模型在开放模型中排第 3 位，26B MoE（Mixture of Experts，混合专家）排第 6 位。[^preface-gemma4-launch] 排在开放模型前面的，仍是闭源服务。模型规模仍在持续增长，但本书关注的是另一端：同样一套能力，有多少可以离开数据中心运行？端侧部署的独特价值、边界与代价将在第 1 章 1.2 节展开，这里先看第一件事。两个时间节点足以说明：

- 2023 年 7 月发布的 Llama 2（7B、13B、70B 三档参数）还只是纯文本模型，上下文长度仅 4096 个 token，主要部署在数据中心；同期面向端侧的开放模型，停留在 TinyLlama 这类 1B 参数量级。[^preface-llama2][^preface-tinyllama]
- 2026 年 4 月 2 日发布的 Gemma 4，把支持文本、图像、视频与音频输入和 128K 上下文的模型压进了手机和笔记本的内存预算。官方公布的部署数据称，其中 E2B（2B 有效参数）使用 2-bit/4-bit 权重与按层内存映射的 embedding，可在不足 1.5 GB 的内存中运行；decode 吞吐在 CPU 上（树莓派 5）约 7.6 tokens/s，在 NPU 上（Qualcomm Dragonwing IQ8）约 31 tokens/s。[^preface-gemma4-edge]

从这两组官方数据可以保守推断：今天一部手机上的开放模型，能力组合大致相当于 2023 年云侧开放模型的水平。

端侧的追赶不只靠硬件和推理技术的演进，也靠模型形态本身的变化。Andrej Karpathy 在 2024 年两次公开谈到这个方向。7 月他在 X 上写道，模型规模竞赛的方向反了：模型之所以大，是因为训练需要记住互联网文本、常见数字的散列值和冷门事实，而思考本身并不需要这么多参数；他预计会出现体积非常小、却能够可靠思考的模型，甚至可能回到 GPT‑2 的参数规模。[^preface-karpathy-x] 到了 9 月的 No Priors 播客，他把路径说得更具体：蒸馏极其有效，可以用大模型的大量计算教出一个小模型，而小模型能保留大模型的能力；思考核心也许 1B 参数就够，其余知识通过工具获取。他还设想未来的模型体系如同一家公司，强大的云侧模型担任 CEO，大量廉价的小模型分工执行；他自己运行在本地设备的个人知识助理（exo‑cortex）参数量不到 1B。[^preface-karpathy-nopriors]

如果这一方向成立，数百亿乃至万亿级模型的能力将被蒸馏进 1B 级的小模型，端侧将成为这类模型最自然的部署位置。结合前文的能力现状，我们推测，未来一两年会有更多产品把推理放进设备。

## Google 的端侧布局

光有模型还不够，还需要一整套把它们送上设备的平台。Google 是少数同时掌控模型、运行时与部署平台的厂商。Gemma 4 是 Google DeepMind 在 2026 年推出的主力开放模型家族，官方将其定位为“可在云端、笔记本电脑和手机上部署的开放模型”，其中 E 系列专门面向边缘设备。[^preface-gemma-family] 发布当日，Google Developers Blog 同步公布了端侧配套方案：AI Edge Gallery 示例应用、Agent Skills 技能库以及 LiteRT‑LM 部署路径。[^preface-gemma4-edge] 在 Android 侧，AICore 预览版已将 Gemma 4 定位为下一代 Gemini Nano 的基础模型。[^preface-gemma4-aicore] 同一时期，GDG China 的 Gemma 4 开发者大赛要求用 E2B/E4B 在真实硬件上演示完全离线的端侧部署。[^preface-gemma4-hackathon]

这种全栈布局，使 LiteRT‑LM 值得作为本书的分析对象。它需要解决的问题——内存容量与带宽约束、异构后端调度、投机解码的接受率、多模态 embedding 路径、约束解码的信任边界——并非 LiteRT‑LM 独有，任何想在受限设备上运行大模型的系统都必须面对。

除了问题本身足够通用，选择它还有两个原因。其一，它有公开的产品部署记录：Google Developers Blog 记载了它在 Chrome、Chromebook Plus 和 Pixel Watch 中的实际应用，[^preface-litertlm-deploy] Google AI Edge Gallery 则通过示例应用展示了端侧模型部署的完整链路。[^preface-edge-gallery] 这些场景要求运行时能适配不同设备，而不只是在单一 benchmark 指标上占优。其二，它的源码覆盖了端侧推理的关键技术：KV cache 管理与双缓冲、GPU 设备侧采样、量化权重的加载与 kernel 调度、投机解码、多模态输入、约束解码、工具调用和 LoRA。每一项实现都能在固定版本的源码中定位到对应的 `file:line`。

## 这本书讲什么

截至 2026 年 7 月，据我们检索，还没有一本中文专著系统分析过生产级端侧运行时。本书尝试补上这个空白。它并非 Google 官方出版物，书中的所有观点及可能存在的错漏，均由作者负责。

本书以 LiteRT‑LM v0.13.1 为主要分析对象，回答一个核心问题：大语言模型如何在手机、手表和浏览器等受限设备上运行。主基准模型 Gemma 4 E4B 在 4B 有效参数规模下提供多模态理解与函数调用能力，模型产物以单文件 `.litertlm` 形式分发。[^preface-gemma-e4b]

本书不是使用手册，也不是逐行代码注释，而是关注**实现与权衡**：代码采用了什么设计，每种设计依赖哪些条件，又在哪些场景下受到限制。不同运行时的具体实现会变化，但都可以从内存预算、数据流、后端约束和测量口径几个维度逐项分析。

模型定义了计算与能力，运行时则负责在具体设备上加载、调度和执行。本书同时分析模型产物与运行时代码：既用 `litertlm_print` 检查 `.litertlm` 的 section 布局，也跟踪 Engine 如何读取这些 section 并创建后端资源。只有把文件格式、执行路径和设备约束放在同一条证据链中，读者才能独立判断一次部署失败究竟出在哪一层。

## 推理流水线

全书按照推理流水线组织：输入文本先转换为 token 序列，经 prefill 建立上下文；随后模型在每个 decode step 生成一个 token，直到满足停止条件。

第一部分先把端侧设备的内存、带宽与功耗约束量化成可验算的数字，再引入贯穿全书的指标与分析方法。第二部分按调用顺序分析这条链路；第三部分继续讨论 KV cache、模型格式、异构后端与投机解码；第四部分处理多模态输入、约束解码、工具调用与多语言绑定。

## 写给谁

本书假定读者熟悉 C++，能够读懂模板和智能指针相关代码，并了解 Transformer 的基本原理。正文不再重复这些基础知识；KV cache 双缓冲、投机解码接受率等运行时概念则会逐步展开。

本书面向从事端侧模型产品集成的工程师，也面向想读懂一个生产级推理运行时实现的学生与研究者。读完后，读者应能沿源码追踪一次完整的生成请求，从 `GenerateContentStream` 一直定位到逐 token 输出；能够在给定设备、模型和测量口径下解释性能数据，并判断新增后端、采样策略或语言绑定会对哪些接口产生影响。

如果仅需调用 API，LiteRT‑LM 官方文档更为直接。[^preface-litertlm-docs] 而当遇到部署失败或性能没有达到预期时，本书提供了一种按输入编排、执行器、模型资源和硬件后端逐层定位问题的方法。

## 阅读路径

- 顺序阅读时，可按四个部分依次推进，各部分覆盖的内容见“推理流水线”一节；尾声单独列出实践入口与待验证问题。
- 第 2 章列出二十个问题，每个问题均指向后续章节中的对应分析，可作为阅读索引。
- 阅读前无需先完成源码编译。第 2 章会先用一条命令运行模型；涉及实现细节时，正文会标出关键机制的源码位置（引用体例见下一节），读者可直接对照冻结版本。

## 关于代码引用与数字

全书对 LiteRT‑LM 的代码引用统一锁定在 `v0.13.1` 版本。为不打断行文，正文中不出现文件路径与行号；引用的代码以代码块呈现，出处标注在代码块首行注释里，不逐处重复版本号。对其他项目的引用在脚注中显式标注版本（如 llama.cpp 的 b9873）。代码之外的来源，在相关断言后以页下注给出。锁定冻结版本，可以保证文件、行号与实现描述保持一致；上游后续变化只收入“版本注记”侧栏。

书中的理论上限，例如 decode 上限公式，均在正文中逐步推导，读者可据此验算。实测数据来自两套分别标注的基准：主基准为一台 Mac，运行 Gemma 4 E4B，decode 实测每秒可生成数十个 token；扩展基准为一台搭载 Qualcomm SoC 的手机，使用自编译二进制。方法与全部数据见附录 D；所有标注“〔基准 D〕”之处均出自这套数据，纸面推算与真机实测明确区分，不混用。对于仅有代码分析、尚未经真机验证的部分，如 NPU 的执行行为，书中均就地标明。

[^preface-tinyllama]: Hugging Face，[TinyLlama/TinyLlama-1.1B-Chat-v1.0 模型卡](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)，2023-09 发布；访问日期：2026-08-04。
[^preface-gemma4-launch]: Google DeepMind，Clement Farabet、Olivier Lacombe，[*Gemma 4: Byte for byte, the most capable open models*](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)，2026-04-02；访问日期：2026-08-04。
[^preface-llama2]: Meta AI，[*Meta and Microsoft Introduce the Next Generation of Llama*](https://ai.meta.com/blog/llama-2/)，2023-07-18；访问日期：2026-08-04。
[^preface-karpathy-x]: Andrej Karpathy（@karpathy），X 帖文 [*LLM model size competition is intensifying... backwards!*](https://x.com/karpathy/status/1814038096218083497)，2024-07-19；访问日期：2026-08-04。
[^preface-karpathy-nopriors]: No Priors 播客，[*The Road to Autonomous Intelligence with Andrej Karpathy*](https://podtail.com/pt-PT/podcast/no-priors/the-road-to-autonomous-intelligence-with-andrej-ka/)（Ep. 80，Sarah Guo、Elad Gil 主持），2024-09-05；视频版见 https://www.youtube.com/watch?v=6P2ItWQY_uw；访问日期：2026-08-04。
[^preface-gemma4-edge]: Google Developers Blog，[*Bring state-of-the-art agentic skills to the edge with Gemma 4*](https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/)，2026-04-02；访问日期：2026-08-04。
[^preface-gemma4-aicore]: Android Developers Blog，[*Announcing Gemma 4 in the AICore Developer Preview*](https://android-developers.googleblog.com/2026/04/AI-Core-Developer-Preview.html)，2026-04；访问日期：2026-08-04。
[^preface-gemma4-hackathon]: GDG China，[Gemma 4 开发者大赛｜2026](https://hackathon.googdg.cn/)，报名 2026-04-18 至 2026-05-18，决赛在 2026 Google I/O Connect 中国站（2026-08）举行；访问日期：2026-08-04。
[^preface-litertlm-deploy]: Google Developers Blog，Yu-hui Chen、Ram Iyengar，[*On-device GenAI in Chrome, Chromebook Plus, and Pixel Watch with LiteRT-LM*](https://developers.googleblog.com/on-device-genai-in-chrome-chromebook-plus-and-pixel-watch-with-litert-lm/)，2025-09-24；访问日期：2026-07-18。
[^preface-edge-gallery]: Google AI Edge，[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)，GitHub 仓库；访问日期：2026-07-18。
[^preface-gemma-family]: Google DeepMind，[Gemma](https://deepmind.google/models/gemma/)；访问日期：2026-07-18。
[^preface-gemma-e4b]: Google AI Edge Community，[Gemma 4 E4B LiteRT-LM 模型卡](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm)；访问日期：2026-07-18。
[^preface-litertlm-docs]: Google AI Edge，[LiteRT-LM 官方文档](https://developers.google.com/edge/litert-lm)，更新日期：2026-07-09；访问日期：2026-07-18。