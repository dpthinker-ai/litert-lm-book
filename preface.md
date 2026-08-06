# 前言

2026 年，一部中端手机已经能在本地运行 40 亿有效参数的多模态模型：理解文本、图像和语音，调用外部工具，全部在设备上完成。本书的主基准（Mac，Gemma 4 E4B，条件与数据见附录 D）测得 decode 阶段每秒生成数十个 token。三年前，同样硬件上跑一个 1B 的纯文本模型，每一兆内存都需要精打细算。[^preface-tinyllama] 变化之快，超出多数人的预期。然而，支撑这一变化的整套技术——模型架构、量化方案、运行时调度——至今还没有一本系统性的中文书把它讲清楚。

## 云侧与端侧

云端依然是模型能力的前沿。以 Google 开放模型家族 Gemma 4 为例：旗舰 31B 稠密模型在 256K 上下文下提供多模态与原生函数调用，在开放模型公开榜单中排名第 3；26B MoE（Mixture of Experts，混合专家）排名第 6。[^preface-gemma4-launch] 闭源服务的上限更高。模型规模仍在增长，但本书关注的是另一端：同样一套能力，有多少可以离开数据中心运行？

端侧部署的驱动力是工程性的：数据不出设备，隐私边界清晰；无需网络往返，时延更低；计算成本随设备一次性支付，不随调用次数增加；离线也能工作。这四个理由及各自代价，详见 1.1 节。这些需求长期存在，过去受限于端侧算力而难以实现，如今正逐一得到满足。

从两个时间节点可以读出追赶速度。2023 年 7 月发布的 Llama 2 仍是纯文本模型，上下文长度仅 4096，主要运行于数据中心。[^preface-llama2] 而 2026 年 4 月 2 日发布的 Gemma 4，已将文本、图像、视频与音频输入及 128K 上下文，放入手机和笔记本的内存预算。据发布口径，E2B 在 2-bit 权重与按层内存映射的 embedding 下，仅用约 1.4 GiB 内存运行：CPU 端在树莓派 5 上约 7.6 tokens/s，NPU 端在 Qualcomm Dragonwing IQ8 上约 31 tokens/s。[^preface-edge-ai-vision] 据此判断（基于上述公开资料），今天一部手机上的开放模型，其能力组合大致相当于 2023 年的云侧开放模型——差距在缩小，而非扩大。

追赶不只依赖硬件与量化进步，模型形态本身也在转向。Andrej Karpathy 在 2024 年的两次公开表态指向同一方向。7 月他在 X 上写道：模型规模竞赛的方向反了——模型大，是因为训练要记住互联网、常见数字的散列值和冷门事实，思考本身并不需要这么多参数；他下注会出现“非常非常小”却可靠思考的模型，甚至可能回到 GPT‑2 的参数规模。[^preface-karpathy-x] 9 月在 No Priors 播客中，他进一步说蒸馏极其有效，可以用大模型通过大量计算教出一个小模型，而小模型保留大模型的能力；思考核心也许 1B 参数就够了，其余知识通过工具获取。他还设想未来的模型体系像一家公司：强大的云侧模型任 CEO，大量廉价的小模型做分工明确的员工；他本人不到 1B 参数的外脑（exo‑cortex）就放在本地设备运行。[^preface-karpathy-nopriors] 若此方向成立，数百亿乃至万亿级模型的能力将被蒸馏进 1B 级的小模型，而端侧正是这类模型最自然的部署位置。我们推断，未来一两年端侧推理的部署密度将持续增长。这个推断是否成立，取决于模型形态的实际走向。

## Google 的端侧布局

Gemma 4 是 Google DeepMind 2026 年的主力开放模型家族，官方定位是“可在云端、笔记本电脑和手机上部署的开放模型，E 系列专门面向边缘设备”。[^preface-gemma-family] 发布当天，Google Developers Blog 同步介绍了端侧配套：AI Edge Gallery 示例应用、Agent Skills 技能库及 LiteRT‑LM 部署路径。[^preface-gemma4-edge] Android 方面，AICore 预览版将 Gemma 4 定位为下一代 Gemini Nano 的基础模型。[^preface-gemma4-aicore] 同期，GDG China 举办 Gemma 4 开发者大赛，Edge AI 赛道要求用 E2B/E4B 在真实硬件上演示完全离线的端侧部署，总冠军可获得价值 2 万美元的 Google Cloud 额度。[^preface-gemma4-hackathon] 从模型训练到操作系统再到开发者生态，Google 是少数同时掌控模型、运行时与部署平台的厂商。

这种全栈布局，使 LiteRT‑LM 成为研究端侧推理的优质样本。其核心代码公开，可锚定冻结版本逐行核对。它要解决的问题——内存容量与带宽约束、异构后端调度、投机解码的接受率、多模态 embedding 路径、约束解码的信任边界——具有通用性，任何想在受限设备上运行大模型的系统都要面对。

本书并非 Google 官方出版物，书中的所有观点及可能存在的错漏，均由作者负责。

## 这本书讲什么

本书以 LiteRT‑LM v0.13.1 为主要分析对象，系统回答一个核心问题：大语言模型如何在手机、手表和浏览器等**受限设备**上运行。

本书主基准使用的 Gemma 4 E4B，在 4B 有效参数规模下提供多模态理解和函数调用能力，模型产物以单文件 `.litertlm` 分发。[^preface-gemma-e4b]

选择 LiteRT‑LM 有两个原因。其一，它有公开的产品部署记录——Google Developers Blog 记载了它在 Chrome、Chromebook Plus 和 Pixel Watch 中的实际应用。[^preface-litertlm-deploy] Google AI Edge Gallery 以示例应用展示了端侧模型部署的完整链路。[^preface-edge-gallery] 这些场景要求运行时适配不同设备，而不只追求单一 benchmark 指标。其二，它的源码覆盖了端侧推理的关键技术：KV cache 管理与双缓冲、GPU 设备侧采样、量化权重的加载与 kernel 调度、投机解码、多模态输入、约束解码、工具调用和 LoRA。每一项实现都能在固定版本的源码中找到对应的 `file:line`。

这本书不是使用手册，也不是逐行代码注释。它关注**实现与权衡**：代码采用了什么设计，每种设计依赖哪些条件，在哪些场景下受到限制。不同运行时的实现会变化，但都可以按内存预算、数据流、后端约束和测量口径逐项分析。

## 模型与运行时：同一条证据链

模型定义计算与能力，运行时负责在具体设备上加载、调度和执行。本书同时分析模型产物与运行时代码——既用 `litertlm_print` 检查 `.litertlm` 的 section 布局，也跟踪 Engine 如何读取这些 section 并创建后端资源。将文件格式、执行路径和设备约束放在同一条证据链中，读者才能独立判断一次部署失败究竟出在哪一层。

本书不预测具体产品路线。第 7、8、10 章仅描述 v0.13.1 中可由代码与实验确认的模型体积、低比特 kernel、NPU 部署条件和工具调用接口，并说明相应限制。

## 推理流水线

全书按推理流水线组织——即输入进入模型后，直到 token 逐个输出的完整过程。

输入文本先转换为 token 序列，经 prefill 建立上下文；模型随后在每个 decode step 生成 token，直到满足停止条件。第二部分按调用顺序分析这条链路。第三部分继续讨论 KV cache、模型格式、异构后端与投机解码。第四部分处理多模态输入、约束解码、工具调用与多语言绑定。数据与状态的流动路径，为后续章节分析局部设计提供了可核对的上下文。

## 写给谁

本书假定读者会 C++，能读懂模板和智能指针，并了解 Transformer 的基本原理。正文不再解释这些基础知识；KV cache 双缓冲、投机解码接受率等运行时概念则会逐步展开。

本书面向将端侧模型集成到产品中的工程师，以及研究推理系统的学生。读完后，读者应能沿源码追踪一次生成请求，从 `generate()` 定位到逐 token 输出，并在给定设备、模型和测量口径下解释性能数据，判断新增后端、采样策略或语言绑定会影响哪些接口。

如果只需调用 API，LiteRT‑LM 官方文档更为直接。[^preface-litertlm-docs] 而遇到部署失败或性能偏离预期时，本书提供按输入编排、执行器、模型资源和硬件后端逐层定位的方法。

## 阅读路径

- 顺序阅读时，按四个部分依次推进，各部分的角色见上文；尾声单独列出实践入口与待验证问题。
- 第 2 章列出二十个问题，每个问题都指向后续章节中的对应分析，可作为阅读索引。
- 阅读前不必先完成源码编译。第 2 章先用一条命令运行模型；涉及实现时，正文会标出关键机制的源码位置（引用体例见下一节），可直接对照冻结版本。

## 关于代码引用与数字

全书对 LiteRT‑LM 的代码引用统一锁定在版本 `v0.13.1`，正文写成 `runtime/core/tasks.cc:413`，不再逐处重复版本号；对其他项目的引用显式标注版本，如 llama.cpp 的 `@ b9873`。代码之外的来源，在所支撑的断言后以页下注给出。冻结版本使文件、行号与所述实现保持同一口径；上游后续变化仅收入「版本注记」侧栏。

书中的理论上限（例如 decode 上限公式）均在正文中逐步推导，读者可据此验算。实测数据来自两套分别标注的基准：主基准（一台 Mac，Gemma 4 E4B）与扩展基准（一台 Qualcomm 手机，自编译二进制）。方法与全部数据见附录 D；所有标注「〔基准 D〕」处均出自这套数据，纸面推算与真机实测不混淆。仅有代码分析、未经真机验证的部分（如 NPU 的执行行为），书中均就地标明。

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