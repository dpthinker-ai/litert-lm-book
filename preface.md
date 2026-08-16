# 前言

端侧大模型正在成为手机、手表和浏览器上的常规能力。随之而来的是一批端侧部署首先要回答的工程问题：内存放不放得下、带宽够不够用、电量和发热允不允许持续推理。

这些问题决定部署能否成立，值得系统学习。本书的学习对象是一个生产级端侧运行时：LiteRT-LM。它是什么、与 LiteRT 在软件栈中的分工，见第 1 章 1.1 节。

## 云侧与端侧模型

到 2026 年，一部中端手机已经能够在本地运行 4B 有效参数的多模态模型：理解文本、图像和语音，调用外部工具，全部在设备端完成。本书主基准是一台 Mac（Gemma 4 E4B，条件与数据见附录 D），decode 实测每秒可生成数十个 token。而三年前，在同等硬件上运行一个 1B 纯文本模型，每一兆内存都需要精打细算。[^preface-tinyllama] 然而，支撑这一变化的整套技术，如模型架构、量化方案、运行时调度等，据我们 2026 年 7 月的检索，仍缺少一本以生产级端侧运行时为对象、可逐行核对的中文系统性专著。

云端依然是模型能力的前沿。从 OpenAI、Google、Anthropic 的闭源旗舰，到 DeepSeek、Kimi、GLM 等国产开放模型，再到以 Gemma 4 为代表的谷歌开放模型，模型在上下文长度、多模态理解、原生工具调用等方面持续突破。Gemma 4 旗舰 31B 稠密模型在 256K 上下文窗口下支持多模态与原生函数调用，在开放模型公开榜单中排名第 3；26B MoE（Mixture of Experts，混合专家）排名第 6。[^preface-gemma4-launch] 闭源服务的上限更高。模型规模仍在持续增长，但本书关注的是另一端：同样一套能力，有多少可以离开数据中心运行？对这个问题的回答，很大程度上取决于端侧部署的独特价值。

端侧部署的独特价值体现在哪些方面、有什么边界与代价，在第 1 章 1.2 节展开。这些需求其实长期存在，只是过去受限于端侧算力而难以实现，如今正逐一得到满足。从两个时间节点，可以清楚看出追赶的速度：

- 2023 年 7 月发布的 Llama 2（7B、13B、70B 三档参数）还只是纯文本模型，上下文长度仅 4096 token，主要部署在数据中心。[^preface-llama2] 
- 2026 年 4 月 2 日发布的 Gemma 4，已经将文本、图像、视频与音频输入，以及 128K 上下文，压缩到手机和笔记本的内存预算之内。根据官方公布的数据，其中 E2B（2B 有效参数）在 2-bit 权重与按层内存映射的 embedding 配置下，仅需约 1.4 GiB 内存即可运行：在 CPU 端（树莓派 5）约为 7.6 tokens/s，在 NPU 端（Qualcomm Dragonwing IQ8）约为 31 tokens/s。[^preface-edge-ai-vision] 

基于这些公开资料可以判断，今天一部手机上的开放模型，其能力组合大致相当于 2023 年的云侧开放模型。

端侧的追赶不只靠硬件和推理技术演进，也靠模型形态本身的变化。Andrej Karpathy 在 2024 年的两次公开表态指向同一方向。

- 7 月，他在 X 上写道：模型规模竞赛的方向反了——模型之所以大，是因为训练需要记住互联网、常见数字的散列值和冷门事实，而思考本身并不需要这么多参数；他下注会出现“非常非常小”却能够可靠思考的模型，甚至可能回到 GPT‑2 的参数规模。[^preface-karpathy-x]
- 9 月，在 No Priors 播客中，他进一步指出，蒸馏极其有效：可以通过大模型的大量计算，教出一个小模型，而小模型保留大模型的能力；思考核心也许 1B 参数就够了，其余知识通过工具获取。他还设想未来的模型体系如同一家公司：强大的云侧模型担任 CEO，大量廉价的小模型则是分工明确的员工；他本人不到 1B 参数的外脑（exo‑cortex）就运行在本地设备上。[^preface-karpathy-nopriors]

如果这一方向成立，数百亿乃至万亿级模型的能力将被蒸馏进 1B 级的小模型，端侧将成为这类模型最自然的部署位置。我们据此推断，未来一两年端侧推理的部署密度将持续增长。当然，这一推断是否成立，最终仍取决于模型形态的实际走向。

## Google 的端侧布局

Gemma 4 是 Google DeepMind 在 2026 年推出的主力开放模型家族。官方将其定位为“可在云端、笔记本电脑和手机上部署的开放模型”，其中 E 系列专门面向边缘设备。[^preface-gemma-family] 发布当日，Google Developers Blog 同步公布了端侧配套方案：AI Edge Gallery 示例应用、Agent Skills 技能库以及 LiteRT‑LM 部署路径。[^preface-gemma4-edge] 在 Android 侧，AICore 预览版已将 Gemma 4 定位为下一代 Gemini Nano 的基础模型。[^preface-gemma4-aicore] 同一时期，GDG China 的 Gemma 4 开发者大赛还设有 Edge AI 赛道，要求用 E2B/E4B 在真实硬件上演示完全离线的端侧部署。[^preface-gemma4-hackathon] 从模型训练、操作系统到开发者生态，Google 是少数同时掌控模型、运行时与部署平台的厂商。

正是这种全栈布局，让 LiteRT‑LM 值得作为研究样本：它需要解决的问题——内存容量与带宽约束、异构后端调度、投机解码的接受率、多模态 embedding 路径、约束解码的信任边界——具有很强的通用性，任何想在受限设备上运行大模型的系统都必须面对。

## 这本书讲什么

本书并非 Google 官方出版物，书中的所有观点及可能存在的错漏，均由作者负责。

本书以 LiteRT‑LM v0.13.1 为主要分析对象，系统性地回答一个核心问题：大语言模型如何在手机、手表和浏览器等受限设备上运行。

主基准模型 Gemma 4 E4B 在 4B 有效参数规模下提供多模态理解与函数调用能力，模型产物以单文件 `.litertlm` 形式分发。[^preface-gemma-e4b]

选择 LiteRT‑LM，主要有两个原因。其一，它有公开的产品部署记录——Google Developers Blog 记载了它在 Chrome、Chromebook Plus 和 Pixel Watch 中的实际应用。[^preface-litertlm-deploy] Google AI Edge Gallery 则通过示例应用展示了端侧模型部署的完整链路。[^preface-edge-gallery] 这些场景要求运行时能够适配不同设备，而不只是追求单一 benchmark 指标。其二，它的源码覆盖了端侧推理的关键技术：KV cache 管理与双缓冲、GPU 设备侧采样、量化权重的加载与 kernel 调度、投机解码、多模态输入、约束解码、工具调用和 LoRA。每一项实现都能在固定版本的源码中定位到对应的 `file:line`。

本书不是使用手册，也不是逐行代码注释，而是关注**实现与权衡**：代码采用了什么设计，每种设计依赖哪些条件，又在哪些场景下受到限制。不同运行时的具体实现会变化，但都可以从内存预算、数据流、后端约束和测量口径几个维度逐项分析。

## 推理流水线

模型定义了计算与能力，运行时则负责在具体设备上加载、调度和执行。本书同时分析模型产物与运行时代码：既用 `litertlm_print` 检查 `.litertlm` 的 section 布局，也跟踪 Engine 如何读取这些 section 并创建后端资源。只有把文件格式、执行路径和设备约束放在同一条证据链中，读者才能独立判断一次部署失败究竟出在哪一层。

全书按照推理流水线组织，即从输入进入模型到 token 逐个输出的完整过程。

第一部分先把端侧设备的内存、带宽与功耗约束量化成可验算的数字，再引入贯穿全书的指标与分析方法。输入文本先被转换为 token 序列，经 prefill 建立上下文；随后模型在每个 decode step 生成 token，直到满足停止条件。第二部分按调用顺序分析这条链路；第三部分继续讨论 KV cache、模型格式、异构后端与投机解码；第四部分处理多模态输入、约束解码、工具调用与多语言绑定。数据与状态的流动路径，为后续章节分析局部设计提供了可核对的上下文。

## 写给谁

本书假定读者熟悉 C++，能够读懂模板和智能指针相关代码，并了解 Transformer 的基本原理。正文不再重复这些基础知识；KV cache 双缓冲、投机解码接受率等运行时概念则会逐步展开。

本书面向从事端侧模型产品集成的工程师，以及研究推理系统的学生或工程师。读完后，读者应能沿源码追踪一次完整的生成请求，从 `generate()` 一直定位到逐 token 输出；能够在给定设备、模型和测量口径下解释性能数据，并判断新增后端、采样策略或语言绑定会对哪些接口产生影响。

如果仅需调用 API，LiteRT‑LM 官方文档更为直接。[^preface-litertlm-docs] 而当遇到部署失败或性能偏离预期时，本书提供了一种按输入编排、执行器、模型资源和硬件后端逐层定位问题的方法。

## 阅读路径

- 顺序阅读时，可按四个部分依次推进，各部分的作用见前文；尾声单独列出实践入口与待验证问题。
- 第 2 章列出二十个问题，每个问题均指向后续章节中的对应分析，可作为阅读索引。
- 阅读前无需先完成源码编译。第 2 章会先用一条命令运行模型；涉及实现细节时，正文会标出关键机制的源码位置（引用体例见下一节），读者可直接对照冻结版本。

## 关于代码引用与数字

全书对 LiteRT‑LM 的代码引用统一锁定在 `v0.13.1` 版本，正文中写作 `runtime/core/tasks.cc:413` 这类格式，不再逐处重复版本号；对其他项目的引用则显式标注版本，例如 llama.cpp 的 `@ b9873`。代码之外的来源，在相关断言后以页下注给出。锁定冻结版本，可以保证文件、行号与实现描述保持一致；上游后续变化只收入「版本注记」侧栏。

书中的理论上限，例如 decode 上限公式，均在正文中逐步推导，读者可据此验算。实测数据来自两套分别标注的基准：主基准为一台 Mac，运行 Gemma 4 E4B；扩展基准为一台 Qualcomm 手机，使用自编译二进制。方法与全部数据见附录 D；所有标注「〔基准 D〕」之处均出自这套数据，纸面推算与真机实测明确区分，不混用。对于仅有代码分析、尚未经真机验证的部分，如 NPU 的执行行为，书中均就地标明。

[^preface-tinyllama]: Hugging Face，[TinyLlama/TinyLlama-1.1B-Chat-v1.0 模型卡](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)，2023-09 发布；访问日期：2026-08-04。
[^preface-gemma4-launch]: Google DeepMind，Clement Farabet、Olivier Lacombe，[*Gemma 4: Byte for byte, the most capable open models*](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)，2026-04-02；访问日期：2026-08-04。
[^preface-llama2]: Meta AI，[*Meta and Microsoft Introduce the Next Generation of Llama*](https://ai.meta.com/blog/llama-2/)，2023-07-18；访问日期：2026-08-04。
[^preface-edge-ai-vision]: Edge AI and Vision Alliance，[*Google Pushes Multimodal AI Further Onto Edge Devices with Gemma 4*](https://www.edge-ai-vision.com/2026/04/google-pushes-multimodal-ai-further-onto-edge-devices-with-gemma-4/)，2026-04-03（转述 Google 官方发布口径）；访问日期：2026-08-04。
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

