# 前言

把训练后的模型部署到手机等端侧设备，会直接遇到运行时约束。设备依靠电池供电，内存有限，温度升高还会触发降频。如何在这些条件下满足容量、时延与稳定性要求，是本书讨论的主题。

## 这本书讲什么

本书的分析对象是 **LiteRT-LM**，Google 的端侧大模型运行时。它在 LiteRT（原 TFLite）之上增加 LLM 编排层：上层提供 Engine/Session API 及多种语言边界，中间把请求编排为 prefill 与 decode，底层经编译模型调度到 CPU、GPU 或 NPU，模型则以单文件 `.litertlm` 分发。Python 与 Swift 使用 C ABI；Kotlin 走 JNI，Web 走 Embind，并非所有绑定都经过同一层。

选择 LiteRT-LM 有两个理由。其一，Google Developers Blog 记录了它在 Chrome、Chromebook Plus 和 Pixel Watch 等产品中的部署。[^preface-litertlm-deploy] Google AI Edge Gallery 还以示例应用展示端侧模型部署。[^preface-edge-gallery] 这些使用场景要求运行时适配不同设备，而不只追求单一 benchmark 指标。其二，它覆盖 KV cache、GPU 设备侧采样、量化与 mmap、推测解码、多模态输入、约束解码、工具调用和 LoRA 等关键问题。相应实现都能在固定版本的源码中核对。

这本书不是使用手册，也不是逐行代码注释。它关注实现与权衡：代码采用了什么设计，每种设计依赖哪些条件，又在哪些场景受到限制。不同运行时的实现会变化，但都可以按内存预算、数据流、后端约束和测量口径逐项分析。

本书并非 Google 官方出版物，书中观点与任何错漏均由作者负责。

## 为什么是 Gemma，为什么是端侧

本书主要使用 Google 的 Gemma 开放模型家族。Google DeepMind 将其定位为可部署在云端、笔记本电脑和手机上的开放模型。[^preface-gemma-family] E 系列面向端侧部署；本书基准使用的 E4B 模型体积为数 GiB，并包含多模态与函数调用能力。[^preface-gemma-e4b] 本地执行可以减少请求离开设备的需要，也可在无网络时运行；具体隐私边界、设备成本与模型能力仍取决于应用实现。

模型定义计算与能力，运行时负责在具体设备上加载、调度和执行。LiteRT-LM 在 LiteRT 之上增加 LLM 所需的编排、状态和语言绑定。本书同时分析模型产物与运行时代码，是为了把文件布局、执行路径和设备约束放在同一条证据链中。

本书不预测具体产品路线。第 7、8、10 章只描述 v0.13.1 中可由代码与实验确认的模型体积、低比特 kernel、NPU 部署条件和工具调用接口，并说明相应限制。

## 推理流水线

全书按推理流水线组织，即输入进入模型后，直到 token 逐个输出的完整处理过程。

输入文本先转换成 token 序列，经 prefill 建立上下文；模型随后在每个 decode step 生成 token，直到满足停止条件。第二部分按调用顺序分析这条链路。第三部分继续讨论 KV cache、模型格式、异构后端与推测解码。数据与状态的流动路径为后续章节分析局部设计提供了可核对的上下文。

## 写给谁

本书假定读者会 C++，能够读懂模板和智能指针，并了解 Transformer 的基本原理。因此，正文不再解释这些基础知识；KV cache 双缓冲、推测解码接受率等运行时概念则会逐步展开。

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
