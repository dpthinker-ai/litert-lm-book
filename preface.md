# 前言

2026 年，一部中端手机可以在本地运行 40 亿参数的多模态模型：能理解文本、图像和语音，可以调用工具，每秒生成几十个 token。三年前，同等硬件上跑一个 1B 纯文本模型还需要仔细核算每一兆内存。这个变化速度超出了多数人的预期，而它背后的技术栈，从模型架构、量化方案到运行时调度，至今没有一本系统性的中文书把它讲清楚。

端侧大模型不是"把云端模型缩小塞进手机"。它是一组独立的技术决策：模型架构要为边缘设备设计，量化方案要匹配硬件的实际数据路径，运行时要在 CPU、GPU、NPU 之间调度计算，内存管理要同时容纳权重、KV cache 和激活张量。每一步从"能跑"到"跑得好"，中间隔着一系列相互关联的实现问题。

Google 在端侧大模型上的布局是少见的全栈策略。模型层有 Gemma 开放家族：Google DeepMind 将其定位为可在云端、笔记本电脑和手机上部署的开放模型，其中 E 系列专门面向边缘设备。[^preface-gemma-family] 本书基准使用的 Gemma 4 E4B 在 4B 参数规模下提供了多模态理解和函数调用能力，模型产物以单文件 `.litertlm` 分发。[^preface-gemma-e4b] 运行时层有 LiteRT-LM——它在 LiteRT（原 TensorFlow Lite）之上构建了完整的 LLM 编排栈，提供 Engine/Session API、多语言绑定和异构后端调度，已经部署在 Chrome、Chromebook Plus 和 Pixel Watch 等产品中。[^preface-litertlm-deploy] 从模型训练到操作系统，Google 是为数不多的同时控制模型、运行时和部署平台的玩家。

这种全栈布局使 LiteRT-LM 成为研究端侧推理的优质样本。它的核心代码公开，可以锚定到冻结版本逐行核对。它要解决的问题，从内存容量与带宽约束、异构后端调度、投机解码的接受率，到多模态 embedding 路径和约束解码的信任边界，不是某一个模型的特殊需求，而是任何想在受限设备上运行大模型的系统都必须面对的通用问题。

本书并非 Google 官方出版物，书中的观点与任何错漏均由作者负责。

## 这本书讲什么

本书以 LiteRT-LM v0.13.1 为主要分析对象，系统回答一个问题：**大语言模型如何在手机、手表和浏览器等受限设备上运行**。

选择 LiteRT-LM 有两个理由。其一，它有公开的产品部署记录——Google Developers Blog 记录了它在 Chrome、Chromebook Plus 和 Pixel Watch 中的使用。[^preface-litertlm-deploy] Google AI Edge Gallery 以示例应用展示了端侧模型部署的完整链路。[^preface-edge-gallery] 这些场景要求运行时适配不同设备，而不只追求单一 benchmark 指标。其二，它的源码覆盖了端侧推理的关键技术：KV cache 管理与双缓冲、GPU 设备侧采样、量化权重的加载与 kernel 调度、投机解码、多模态输入、约束解码、工具调用和 LoRA。每一项实现都能在固定版本的源码中找到对应的 `file:line`。

这本书不是使用手册，也不是逐行代码注释。它关注**实现与权衡**：代码采用了什么设计，每种设计依赖哪些条件，又在哪些场景受到限制。不同运行时的实现会变化，但都可以按内存预算、数据流、后端约束和测量口径逐项分析。

## 模型与运行时：同一条证据链

模型定义计算与能力，运行时负责在具体设备上加载、调度和执行。本书同时分析模型产物与运行时代码——既用 `litertlm_print` 检查 `.litertlm` 的 section 布局，也跟踪 Engine 如何读取这些 section 并创建后端资源。把文件格式、执行路径和设备约束放在同一条证据链中，读者才能独立判断一个部署失败究竟出在哪一层。

本书不预测具体产品路线。第 7、8、10 章只描述 v0.13.1 中可由代码与实验确认的模型体积、低比特 kernel、NPU 部署条件和工具调用接口，并说明相应限制。

## 推理流水线

全书按推理流水线组织——输入进入模型后，直到 token 逐个输出的完整处理过程。

输入文本先转换为 token 序列，经 prefill 建立上下文；模型随后在每个 decode step 生成 token，直到满足停止条件。第二部分按调用顺序分析这条链路。第三部分继续讨论 KV cache、模型格式、异构后端与投机解码。第四部分处理多模态输入、约束解码、工具调用与多语言绑定。数据与状态的流动路径为后续章节分析局部设计提供了可核对的上下文。

## 写给谁

本书假定读者会 C++，能够读懂模板和智能指针，并了解 Transformer 的基本原理。正文不再解释这些基础知识；KV cache 双缓冲、投机解码接受率等运行时概念则会逐步展开。

本书面向把端侧模型集成到产品中的工程师，以及研究推理系统的学生。读完后，读者应能沿源码追踪一次生成请求，从 `generate()` 定位到逐 token 输出。读者还应能在给定设备、模型和测量口径下解释性能数据，并判断新增后端、采样策略或语言绑定会影响哪些接口。

如果只需调用 API，LiteRT-LM 官方文档更直接。[^preface-litertlm-docs] 遇到部署失败或性能偏离预期时，本书提供按输入编排、执行器、模型资源和硬件后端逐层定位的方法。

## 阅读路径

- 顺序阅读时，第一部分建立资源约束、性能指标和系统分层的共同口径。第二部分沿推理流水线分析调用路径。第三部分讨论运行时优化与异构执行。第四部分处理能力扩展和跨语言集成。尾声单列实践入口与待验证问题。
- 第 2 章列出二十个问题。每个问题都指向后续章节中的对应分析，可作为阅读索引。
- 阅读前不必先完成源码编译。第 2 章先用一条命令运行模型；涉及实现时，正文会标出关键机制的源码位置（引用体例见下一节），可直接对照冻结版本。

## 关于代码引用与数字

全书的 LiteRT-LM 代码引用统一锁定在版本 `v0.13.1`，正文写成 `runtime/core/tasks.cc:413`，不逐处重复版本号；对其他项目的引用显式标注版本，如 llama.cpp 的 `@ b9873`。代码之外的来源在所支撑的断言后以页下注给出。冻结版本使文件、行号与所述实现保持同一口径；上游后续变化只收入「版本注记」侧栏。

书中的理论上限（例如 decode 上限公式）均在正文中逐步推导，读者可据此验算。实测数据来自两套分开标注的基准：主基准（一台 Mac，Gemma 4 E4B）与扩展基准（一台 Qualcomm 手机，自编译二进制）。方法与全部数据见附录 D；所有标注「〔基准 D〕」处都出自这套数据，纸面推算与真机实测不混淆。仅有代码分析、未经真机验证的部分（如 NPU 的执行行为），书中均就地标明。

[^preface-litertlm-deploy]: Google Developers Blog，Yu-hui Chen、Ram Iyengar，[*On-device GenAI in Chrome, Chromebook Plus, and Pixel Watch with LiteRT-LM*](https://developers.googleblog.com/on-device-genai-in-chrome-chromebook-plus-and-pixel-watch-with-litert-lm/)，2025-09-24；访问日期：2026-07-18。
[^preface-edge-gallery]: Google AI Edge，[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)，GitHub 仓库；访问日期：2026-07-18。
[^preface-gemma-family]: Google DeepMind，[Gemma](https://deepmind.google/models/gemma/)；访问日期：2026-07-18。
[^preface-gemma-e4b]: Google AI Edge Community，[Gemma 4 E4B LiteRT-LM 模型卡](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm)；访问日期：2026-07-18。
[^preface-litertlm-docs]: Google AI Edge，[LiteRT-LM 官方文档](https://developers.google.com/edge/litert-lm)，更新日期：2026-07-09；访问日期：2026-07-18。
