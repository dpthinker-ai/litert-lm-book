# 《端侧大模型推理：原理与 LiteRT-LM 实现》全书档案

*On-Device LLM Inference: Principles and Practice with LiteRT-LM*

> 全稿已成：前言、12 章、尾声、附录 A-E，40 张 SVG。已发布书稿基线为 Tag v0.2.0（源码分析为 v0.13.1）；当前源码分析基线为 release/v0.17.0 对应的冻结 tag，第 10 章 MoE 接续 MTP，作为第三部分末章。开放项见下文“当前实测与开放项”。
> 修订记录以 git 提交历史为准；本文件只保留全书定位、目录与开放项。
> 初稿期的详细规划（章卡、路线图、决策记录）见 git 历史与 `archive/`。

## 定位

以 Google 的端侧 LLM 运行时 **LiteRT-LM**（`google-ai-edge/LiteRT-LM` @ **v0.17.0**，已冻结）为主要分析对象，系统回答：大语言模型如何在手机、手表和浏览器等受限设备上运行。视角是**实现与权衡**，不是 API 教程，也不是优化技术综述；非官方出版物。

**对读者的承诺**：读完本书，你能
1. 完整走通一次生成请求：从 `GenerateContentStream` 到输出 token 的每一步发生了什么；
2. 用系统工程师的方式度量与解释端侧推理性能，包括数字口径与资源约束；
3. 看懂并有能力修改/扩展一个生产级推理运行时（加后端、加采样策略、修真实 issue）。

**已定决策**：体例为问题驱动 + 「对照视野」侧栏（简评 llama.cpp / MLC-LLM / ExecuTorch）；读者为中阶工程师（会 C++、懂 Transformer 基础）；简体中文；私有仓库内部打磨，暂不开源；正文 + 附录 250-300 页；工具链 mdBook + WebKit PDF；主基准模型 Gemma 4 E4B（`litert-community/gemma-4-E4B-it-litert-lm`），主基准设备为作者的 Mac（Apple Silicon）。

## 目录

**第一部分 · 约束、指标与系统概览**
- 第 1 章　端侧 LLM 与 LiteRT-LM：三类物理约束与同类运行时对照——LiteRT-LM 与 LiteRT 的分工；内存容量、内存带宽、功耗与异构约束的量化分析
- 第 2 章　从运行到架构：benchmark、Roofline 与五层视图——跑通基准并正确解读数字；「20 个问题」清单；五层架构图

**第二部分 · 推理流水线（主线）**
- 第 3 章　输入侧：从 Engine API 到 token 序列——Engine/Session 抽象、Conversation、模板增量渲染、双 tokenizer
- 第 4 章　Prefill：并行处理提示词——静态/动态形状两条路径、异步底座、取消时序
- 第 5 章　Decode：单步解码循环——DecodeOneStep、采样、停止序列、流式回调（全书深度标尺）

**第三部分 · 资源管理与推理优化**
- 第 6 章　KV cache：容量、带宽与会话生命周期——内存账、双缓冲、Clone/Checkpoint/Rewind
- 第 7 章　模型文件与权重：量化、容器格式与 LoRA——int4/int8 收益账、.litertlm 分段、weight cache、发布验收
- 第 8 章　异构算力：CPU、GPU 与 NPU——后端分派、设备侧采样、buffer 交接、NPU 部分基于代码分析
- 第 9 章　一次前向，多个 token：投机解码与 MTP——drafter/verifier、聚合接受比例与收益边界

- 第 10 章　端侧 MoE：稀疏激活、专家执行与内存管理——Gemma 4 参数与访问成本核算、专家分派、CPU/GPU 源码、数值对照与完整模型部署

**第四部分 · 多模态、工具调用与跨语言集成**
- 第 11 章　多模态与工具调用：视觉/音频编码、约束解码与函数调用
- 第 12 章　多语言绑定：C ABI、JNI 与 Embind

**尾声 · 实践入口与待验证问题**；**附录** A 术语表 / B 代码地图 / C 环境搭建与实验复现 / D 基准数据集 / E 练习提示与参考答案

## 当前实测与开放项（与附录 D 对账，口径以附录 D 为准）

Mac 主基准与 2026 年 7 月 Android 扩展记录保留原有口径。2026-09-05 新增 HONOR MEP-AN00 的进程内存、持续吞吐、客户端时延及单图视觉案例，分别见附录 D 第十四、十五节。仍未取得完整 GPU 分配量、可靠功率、频率分解或屏幕显示事件；进程 RSS/PSS 与文本回调不能替代这些指标。

2026-09-13 在隔离环境使用 v0.17.0 预编译包，完成同一 E4B 模型的 CPU/GPU 基础文本生成检查。模型哈希、环境、参数及成功与失败日志保存在 `experiments/data/2026-09-13/upgrade-v0.17.0/manifest.json`。基础生成检查之后，另完成 v0.17.0 的 Mac 性能重测，归档于 `experiments/data/2026-09-13/benchmark-v0.17.0/`，结果见附录 D 第十六节。共 76 次运行，包含 19 次预热；扣除环形缓冲参数未生效的诊断组，性能表覆盖 18 个条件、54 次正式测量。主矩阵、prefill 与 KV 容量扫描及 MTP 显式开关均已完成。GPU 实际走 WebGPU/Metal，sampler 回退与未生效配置逐项记录。本轮未连接 Android 设备，也未重测峰值内存和功耗；旧版实测矩阵保持原样。执行使用预编译包，仍不是本地 C++ 构建验证。

**待重跑：Android 数据尚未更新到 v0.17.0。** 范围包括 P0210 主基准、MTP 与线程扫描，以及 HONOR MEP-AN00 的内存、持续吞吐、文本回调和图片输入实验。现有手机数据保留 v0.13.1 的原始版本与采集条件。

第 10 章已形成从数值验证到完整生成的实验链路。CPU 专家算子的 FP32／INT8／INT4 数值案例通过；真实导出的激活语义及 GPU 计算精度另设对照。FP32 单层规模与路由实验完成 16 组配置、CPU/GPU 各 3 次进程重复，数据见附录 D 第十七至二十节。

完整 Gemma 4 26B-A4B GPU 产物在 M5 Pro、24 GiB、v0.17.0 预编译包上使用 Artisan／Metal。已完成中英文短文本、容量 512/1024/2048/4096、128/256 输出计数及三个六轮会话，共 60 次正式请求。实际单轮 prefill 测至 3981 token，多轮状态更新 18 次全部正确；较长输出到达上限后截断，不代表文章自然完成。启动、首次请求和引擎复用分别统计，见附录 D 第二十一至二十四节。

分阶段内存采集固定容量 4096，区分进程 RSS、physical footprint 与 vmmap 摘要，见附录 D 第二十五节。其结果不能替代 GPU 分配或模型独占内存。真实专家访问、内存分配细目、热稳态、长时间稳定性与一般任务质量仍未验证。INT8 自定义 GPU 路径尚无通过配置；完整 Artisan 成功不代表该路径已修复。Web、Android/NPU 和本地源码构建也不据此计为通过。

尚未完成的实验，正文均已按"来源于代码分析/无实测数据"如实标注：
- MoE 完整模型导出与本地构建、自定义量化 GPU 路径与 NPU 支持、系统性质量、真实路由访问、GPU 分配和权重驻留分解、热稳态与长时间性能测量
- 编译缓存开/关的外部墙钟受控对照
- 同一基础 checkpoint 的可比量化质量与性能对照（缺少两档可比产物及可说明的量化、校准设置）
- `parallel_file_section_loading` 开/关对照
- 同一 prompt 的 Python/C++ 行为一致性对照
- 音频、多图与通用视觉任务质量测量（单图输入、两档预算与两类输入错误已完成；有效视觉位置由运行计数和 tokenizer 诊断间接复算）
- 同一输入的双 tokenizer 对照
- 使用匹配模型产物、SoC 代际和运行时的 NPU prefill/decode（两台设备仅完成部署探测；本书没有 NPU 吞吐、时延或功耗数据）

## 构建与检查

```
bash scripts/build_book.sh          # mdBook 构建
bash scripts/make_pdf.sh            # PDF 成书
bash scripts/lint_prose.sh <文件>   # 禁词/填充词机检
python3 scripts/check_book_consistency.py
python3 scripts/check_citations.py
python3 scripts/check_code_references.py
python3 scripts/check_book_links.py
python3 scripts/check_pdf_outline.py dist/book.pdf
```

写作规范见 `CLAUDE.md`；修订工作流见其第七节。
