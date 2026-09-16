# 前言

与大语言模型对话，如今已是常事。常见的体验来自云端聊天产品：提问、贴代码、传文档，回答逐字出现在屏幕上，还可以来回追问几轮。这类产品使用的是云侧模型，生成回答的计算在数据中心完成，用户的设备只负责收发。

模型同样可以部署在端侧，即在手机、手表和浏览器上本地运行；这些设备的可用内存、芯片性能与供电条件各不相同。要在其中部署参数量以 B（十亿）计的模型，先要回答三个工程问题：内存容量放不放得下，内存带宽够不够，功耗与发热是否允许持续生成。系统地回答这些问题，是本书的任务。

本书的分析对象是生产级端侧运行时 LiteRT-LM。它是什么、与 LiteRT 在软件栈中如何分工，见第 1 章 1.1 节。在进入实现之前，先交代两件事：端侧模型目前的能力水平，以及 Google 为端侧部署提供了哪些配套。

## 云侧与端侧模型

Gemma 4 的 31B 稠密模型支持 256K 上下文、多模态输入与原生函数调用。发布文章引用 Arena AI 文本榜单：31B 在开放模型中排第 3 位，26B MoE（Mixture of Experts，混合专家）排第 6 位。[^preface-gemma4-launch]

本书关注的是资源受限的设备上能运行什么样的模型。下面三组例子分别取 2023 年的开放模型、2026 年的开放模型和一款手机的官方资料，对照参数规模、功能范围与部署条件这几年的变化：

- 2023 年 7 月发布的 Llama 2（7B、13B、70B 三档参数）是纯文本模型，上下文长度为 4096 个 token。[^preface-llama2] 同年 9 月 1 日开始训练的 TinyLlama 选择了 1.1B 参数规模，面向计算和内存受限的应用。[^preface-tinyllama]
- 2026 年 4 月发布的 Gemma 4，其中 E2B 与 E4B 接受文本、图像与音频输入，输出文本，模型上下文窗口为 128K。[^preface-gemma-e4b-google] Google AI Edge 团队的博客称，借助 LiteRT 对 2-bit/4-bit 权重和按层内存映射 embedding 的支持，E2B（2B 有效参数）在部分设备上可以在不足 1.5 GB 的内存内运行；较小的 Gemma 4 模型在树莓派 5 的 CPU 上达到 133 prefill、7.6 decode tokens/s，在 Qualcomm Dragonwing IQ8 的 NPU 上达到 3,700 prefill、31 decode tokens/s。该博客没有注明这些吞吐对应的模型规格、量化与上下文长度。[^preface-gemma4-edge]
- 2026 年 9 月发布的华为 Mate XT 2，[^preface-huawei-launch] 官方资料称其在端侧部署了 30B MoE 模型。[^preface-huawei-moe] 稀疏激活让总参数量与每个 token 使用的参数量分开：每一步只计算其中一部分专家，设备仍要保存全部参数。

这些资料说明，端侧开放模型已能提供多模态输入与长上下文支持。[^preface-gemma4-edge] 稀疏激活的模型还要把容量与带宽分开计算，第 10 章以本书实测的 Gemma 4 26B-A4B 为例，核算专家工作集，并与生成及内存数据对照。端侧部署的价值、边界与代价见第 1 章 1.2 节。

## Google 的端侧方案

只有模型还不够，还需要把模型部署到设备上的运行时与工具链。Google 同时提供模型、运行时与部署平台。Gemma 4 是 Google DeepMind 推出的开放模型家族，官方将其定位为“可在云端、笔记本电脑和手机上部署的开放模型”。[^preface-gemma-family] 发布当日，Google Developers Blog 同步公布了端侧配套方案：AI Edge Gallery 示例应用、Agent Skills 技能库以及 LiteRT-LM 部署路径。[^preface-gemma4-edge] 在 Android 侧，AICore 预览版已将 Gemma 4 定位为下一代 Gemini Nano 的基础模型。[^preface-gemma4-aicore] 同一时期，GDG China 的 Gemma 4 开发者大赛要求用 E2B/E4B 在真实硬件上演示完全离线的端侧部署。[^preface-gemma4-hackathon]

本书选择 LiteRT-LM 作为分析对象，首先因为它处理的问题足够通用。内存容量与带宽约束、异构后端调度、投机解码的接受率、多模态 embedding 路径、约束解码的信任边界，其他端侧推理系统在实现同类功能时同样要处理。

除此之外还有两个原因。其一，Google 已将它部署在自己的产品中：Google Developers Blog 记载了它在 Chrome、Chromebook Plus 和 Pixel Watch 中的实际应用，[^preface-litertlm-deploy] Google AI Edge Gallery 则通过示例应用展示了端侧模型部署的完整链路。[^preface-edge-gallery] 其二，它的源码覆盖了端侧推理的关键技术：KV cache 管理与双缓冲、GPU 设备侧采样、量化权重的加载与后端分派、投机解码、多模态输入、约束解码、工具调用和 LoRA，每一项实现都能在固定版本的源码中定位到具体位置。

## 这本书讲什么

本书以 LiteRT-LM v0.17.0 为主要分析对象，回答一个核心问题：大语言模型如何在手机、手表和浏览器等受限设备上运行。主基准模型 Gemma 4 E4B 在 4B 有效参数规模下提供多模态理解与函数调用能力，模型产物以单文件 `.litertlm` 形式分发；litert-community 的模型卡称该产物最长支持 32K 上下文，低于模型本身的 128K 窗口。[^preface-gemma-e4b]

本书不是使用手册，也不是逐行代码注释，而是关注**实现与权衡**：代码采用了什么设计，每种设计依赖哪些条件，又在哪些场景下受到限制。不同运行时的具体实现会变化，但都可以从内存预算、数据流、后端约束和测量口径这几个方面逐项分析。

模型定义了计算与能力，运行时则负责在具体设备上加载、调度和执行。本书同时分析模型产物与运行时代码：既用 `litertlm_print` 检查 `.litertlm` 的 section 布局，也跟踪 Engine 如何读取这些 section 并创建后端资源。只有同时核对文件格式、执行路径和设备约束，读者才能独立判断一次部署失败出在哪一层。

本书并非 Google 官方出版物，书中的所有观点及可能存在的错漏，均由作者负责。

## 推理流水线

全书按照推理流水线组织：输入文本先转换为 token 序列，经 prefill（并行处理整段输入）建立上下文；随后进入 decode，每个 decode step（一次前向与采样的迭代）产出一个 token，直到满足停止条件；投机解码让一次前向确认多个 token，见第 9 章。

第一部分（第 1、2 章）先把端侧设备的内存、带宽与功耗约束量化成可验算的数字，再引入贯穿全书的指标与分析方法。第二部分（第 3 至 5 章）按调用顺序分析这条流水线：输入编排、prefill、decode 与采样。第三部分（第 6 至 10 章）讨论 KV cache、模型格式、异构后端、投机解码与 MoE。第四部分（第 11、12 章）处理多模态输入、约束解码、工具调用与多语言绑定。

## 写给谁

本书假定读者熟悉 C++，能够读懂模板和智能指针相关代码，并了解 Transformer 的基本原理。正文不再重复这些基础知识；KV cache 双缓冲、投机解码接受率等运行时概念则会逐步展开。

本书面向从事端侧模型产品集成的工程师，也面向想读懂一个生产级推理运行时实现的学生与研究者。读完后，读者应能沿源码追踪一次完整的生成请求，从 `GenerateContentStream` 一直定位到逐 token 输出；能够在给定设备、模型和测量口径下解释性能数据，并判断新增后端、采样策略或语言绑定会对哪些接口产生影响。

如果仅需调用 API，LiteRT-LM 官方文档更为直接。[^preface-litertlm-docs] 而当遇到部署失败或性能没有达到预期时，本书提供了一种按输入编排、执行器、模型资源和硬件后端逐层定位问题的方法。

## 阅读路径

- 顺序阅读时按四个部分依次推进；尾声单独列出实践入口与待验证问题。
- 第 2 章列出二十个问题，每个问题均指向后续章节中的对应分析，可作为阅读索引。
- 阅读前无需先完成源码编译。第 2 章会先用一条命令运行模型；涉及实现细节时，关键机制附有源码片段，引用体例见下一节。

## 关于代码引用与数字

全书对 LiteRT-LM 的代码引用统一锁定在 `release/v0.17.0` 对应的 `v0.17.0` tag，冻结提交为 `e9fd8c53ff968071774206163027dd84bedfe925`。为不打断行文，正文中不出现文件路径与行号；引用的代码以代码块呈现，出处标注在代码块首行注释里，不逐处重复版本号。对其他项目的引用在脚注中显式标注版本（如 llama.cpp 的 b9873）。代码之外的来源，在相关断言后以页下注给出。上游后续变化只收入“版本注记”侧栏。

第 10 章进一步分析 v0.17.0 所锁定的 LiteRT 依赖，其代码块以 `LiteRT/` 前缀标出，同样锁定冻结提交；具体提交号与导出端分析范围见 10.6 节。

书中的理论上限，例如 decode 上限公式，均在正文中逐步推导，读者可据此验算。实测数据按设备与采集批次分别列示，既有实验保留原采集版本与条件，其中 v0.13.1 的数据不代表 v0.17.0 的性能：Mac 主基准运行 Gemma 4 E4B；2026 年 7 月的 Android 扩展基准使用 P0210 手机和自编译二进制；同年 9 月另在 HONOR MEP-AN00 上补充进程内存、持续吞吐、客户端文本时延与图片输入案例。

方法与全部数据见附录 D，所有标注“〔基准 D〕”之处均指向该附录。进程内存读数不等于完整的 GPU 内存占用，文本回调时刻也早于屏幕显示时刻。同一基础 checkpoint 的量化质量与性能对照尚未完成。NPU 执行行为等仅有代码分析的部分，书中也就地标明。

[^preface-tinyllama]: TinyLlama 项目，[TinyLlama/TinyLlama-1.1B-Chat-v1.0 模型卡](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)，项目训练启动日期：2023-09-01（非 Chat-v1.0 发布日期）；访问日期：2026-09-05。
[^preface-gemma4-launch]: Google DeepMind，Clement Farabet、Olivier Lacombe，[*Gemma 4: Byte for byte, the most capable open models*](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)，2026-04-02；访问日期：2026-08-04。
[^preface-llama2]: Meta AI，[*Meta and Microsoft Introduce the Next Generation of Llama*](https://ai.meta.com/blog/llama-2/)，2023-07-18；访问日期：2026-08-04。
[^preface-gemma4-edge]: Google Developers Blog，[*Bring state-of-the-art agentic skills to the edge with Gemma 4*](https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/)，2026-04-02；访问日期：2026-09-05。
[^preface-gemma4-aicore]: Android Developers Blog，[*Announcing Gemma 4 in the AICore Developer Preview*](https://android-developers.googleblog.com/2026/04/AI-Core-Developer-Preview.html)，2026-04；访问日期：2026-08-04。
[^preface-gemma4-hackathon]: GDG China，[Gemma 4 开发者大赛｜2026](https://hackathon.googdg.cn/)，报名 2026-04-18 至 2026-05-18，决赛在 2026 Google I/O Connect 中国站（2026-08）举行；访问日期：2026-08-04。
[^preface-litertlm-deploy]: Google Developers Blog，Yu-hui Chen、Ram Iyengar，[*On-device GenAI in Chrome, Chromebook Plus, and Pixel Watch with LiteRT-LM*](https://developers.googleblog.com/on-device-genai-in-chrome-chromebook-plus-and-pixel-watch-with-litert-lm/)，2025-09-24；访问日期：2026-07-18。
[^preface-edge-gallery]: Google AI Edge，[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)，GitHub 仓库；访问日期：2026-07-18。
[^preface-gemma-family]: Google DeepMind，[Gemma](https://deepmind.google/models/gemma/)；访问日期：2026-07-18。
[^preface-gemma-e4b]: Google AI Edge Community，[Gemma 4 E4B LiteRT-LM 模型卡](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm)；访问日期：2026-09-14。
[^preface-gemma-e4b-google]: Google，[Gemma 4 E4B-it 模型卡](https://huggingface.co/google/gemma-4-E4B-it)，Hugging Face；访问日期：2026-09-14。
[^preface-litertlm-docs]: Google AI Edge，[LiteRT-LM 官方文档](https://developers.google.com/edge/litert-lm)，更新日期：2026-07-09；访问日期：2026-07-18。
[^preface-huawei-moe]: 华为，[*HUAWEI Mate XT 2 | ULTIMATE DESIGN 卖点*](https://consumer.huawei.com/cn/support/content/zh-cn16114946/)，适用版本 HarmonyOS 7.0，“大屏 AI 再进化”；访问日期：2026-09-13。
[^preface-huawei-launch]: 华为，[*HUAWEI Mate XT 2 | ULTIMATE DESIGN 产品发布介绍*](https://consumer.huawei.com/cn/support/content/zh-cn16112373/)，页面载明发布会为 2026 年 9 月 7 日 14:30 的“HarmonyOS 7 | HUAWEI Mate XT 2 及全场景新品发布会”，适用版本 HarmonyOS 7.0；访问日期：2026-09-16。
