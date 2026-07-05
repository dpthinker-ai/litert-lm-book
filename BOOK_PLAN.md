# 《端侧大模型推理：原理与 LiteRT-LM 实现》规划书

> 状态：**v4.0 — 完整初稿**（2026-07-05）：**前言 + 11 章 + 尾声 + 附录 A-D 全部成稿**（正文约 10 万字、12 张 SVG 图、12 份 review.md，全书 lint 0 命中，代码引用逐条对 v0.13.1 核验）。
> - 全书 12 篇正文：两个 pass 全部完成（事实核查 + 除 AI 味含**独立审校 12/12**）。
> - P5 已完成：前言、附录 A 术语表、B 代码地图、C 复现指南、D 基准数据集（表格骨架，数据待采集）；统稿（图表编号理顺、SUMMARY 链接校验、全书 lint）。
> **唯一未完项**：实测数字采集——待 HF 登录后跑 `experiments/bench_baseline.sh`，回填全书「〔基准 D〕」与附录 D 表格。
> 书名（已定）：《端侧大模型推理：原理与 LiteRT-LM 实现》
> *On-Device LLM Inference: Principles and Practice with LiteRT-LM*
> 素材基础：`/Users/dpthinker/workspace/litert-lm-guide/data.json`（12 模块深度分析，约 20 万字）
> 代码锚点：`google-ai-edge/LiteRT-LM` **@ v0.13.1（已冻结，2026-07-05）**；核对代码在独立 worktree checkout 该 tag，不扰动 main 工作区。CLAUDE.md 示例引用已按 v0.13.1 核验回填
> 写作规范：`CLAUDE.md`（已经三方审校定稿；本规划与其冲突时，文风层面以 CLAUDE.md 为准）

---

## 〇、一页立意书（全书的宪法）

**这本书讲什么**：以 Google 投产的端侧 LLM 运行时 LiteRT-LM 为解剖标本，
系统回答一个问题——**如何让大语言模型在手机、手表、浏览器这样的受限设备上高效地跑起来**。

**对读者的承诺**：读完本书，你能
1. 在脑中完整走通一次生成请求：从 `generate()` 到逐字吐出的每一步发生了什么；
2. 用系统工程师的方式度量与解释端侧推理性能（为什么是这个数字、瓶颈在哪堵墙）；
3. 看懂并有能力修改/扩展一个生产级推理运行时（加后端、加采样策略、修真实 issue）。

**不做的事**：
- 不写 API 使用教程（官方文档已覆盖；本书视角是"实现与权衡"）；
- 不搞论文综述式的优化技术罗列（只讲被这套代码验证过的、以及它为什么没选别的）；
- 不假装官方（书名与前言明确"非官方深度解析"）。

**定位判断**（内部表述，属【推测】级）：据我们 2026-07 的检索，尚未见到以生产级
端侧 LLM 运行时为标本的系统性中文书；llama.cpp 的公开解读多为碎片文章，
论文综述又不落地到工程实现。本书押注这个空档。

**体量约束（作者拍板）**：正文 + 附录控制在 **250-300 页**；11 章 + 尾声，
单章 16-28 页——章少而厚，每章仍必须是"一个问题"的完整解答。

---

## 一、四项奠基决策

### D1. 体例 【✅ 已拍板：混合路线 C】
每章由领域问题驱动，LiteRT-LM 作为"被深度研究的标准实现"，另设「对照视野」
侧栏简评 llama.cpp / MLC-LLM / ExecuTorch 的不同选择（DDIA 模式；
侧栏写法约束见 CLAUDE.md 第二节）。

### D2. 读者起点 【✅ 已拍板：中阶工程师】
会 C++（读得懂模板与智能指针）、懂 Transformer 基本原理；不假设读过推理框架源码。

### D3. 发布形态 【✅ 已拍板：内部打磨先行，暂不开源】
私有仓库内部迭代，mdBook/Quarto 仅做本地/私有构建；先自己学习和打磨。
核心版（第 1-9 章）打磨满意后，再评估开源与出版路径。
现有学习网站降级为本书的"代码地图"附录与伴生参考站。

### D4. 语言 【✅ 已拍板：简体中文先行】
成稿后再评估翻译。

> ✅ 拍板记录：D3=内部打磨先行（2026-07-04）；D1=混合路线 C、D2=中阶工程师、
> D4=简体中文先行、书名=候选 A《端侧大模型推理：原理与 LiteRT-LM 实现》（2026-07-05）。
> 全部奠基决策拍板完毕；"一个 token 的一生"意象留给前言与第二部部题。

---

## 二、结构设计的六条原则

1. **问题驱动，不是目录驱动**。每章标题必须能改写成一个读者真实会问的问题；
   写作前每章过"一句话使命"测试。
2. **脊柱先行**。先叙事（token 的一生），后分析（逐系统深潜）。
3. **螺旋深化**。核心概念出现三次、一次比一次深（KV cache：第 2 章直觉 →
   第 4/5 章使用 → 第 6 章深潜）。
4. **每章可验证**。每章至少一个可复现实验；全书共用一套"基准数据集"
   （同一模型、同一设备，P1 一次性采集，见附录 D）；多个实验直接解释上游真实 issue。
5. **素材与正文分离**。每章 `notes.md`（素材）+ `chapter.md`（正文）+
   `review.md`（两个 pass 的留痕）；正文只贴 ≤30 行关键代码，必带 `file:line @ <tag>`。
6. **图表到位**（作者拍板）。流程/结构/数据流必配 SVG 手绘图（规范见 CLAUDE.md
   第六节），对比信息用表；每章图表在章卡里预先规划，不为凑数配图。

**压缩装配原则**："底座型"话题不独立成章，就近并入它发挥作用的地方——
线程/异步 → Prefill 章与后端章；会话状态(Clone/Checkpoint) → KV cache 章；
Roofline → 跑起来章 + KV cache 章开篇；LoRA → 模型形态章。
好处：话题出现在读者正需要它的上下文里；代价：内部版的评审粒度从"章"细化为"节"。

**版本锚定策略**：全书代码引用锁定一个 release tag；「版本注记」侧栏消化上游演进，不改正文。

---

## 三、目录 v3（四部 · 11 章 + 尾声 · 4 附录，约 285 页）

> 章卡字段：使命 / 核心问题 / 内容 / 图表 / 实验 / 素材 / 缺口

### 第一部 · 起点（2 章，约 40 页）

**第 1 章 端侧 LLM：三堵墙与一张版图**（约 16 页）
- 使命：读者能亲手算出"4B 模型在手机上的理论 decode 上限"这笔账。
- 内容：隐私/延迟/成本/离线的收益；内存墙、带宽墙、功耗/异构墙的量化分析；
  端侧运行时版图（llama.cpp / MLC / ExecuTorch / MediaPipe → LiteRT-LM 的位置与投产实绩）。
- 图表：图 1-1 三堵墙与模型需求对照；表 1-1 端侧运行时版图对比（定位/后端/格式/投产情况）。
- 实验：纸面算账（公式与代入过程完整给出，第 6 章实测对账）。
- 缺口：收集 2-3 款典型 SoC 的带宽/算力参数（须注官方来源，按【文档】级）。

**第 2 章 跑起来与鸟瞰：从 benchmark 数字到五层架构**（约 24 页）
- 使命：跑通 + 会读性能数字 + 获得全书地图。
- 内容：CLI 跑 Gemma → `--benchmark` 逐项解读（Roofline 初讲：prefill=算力瓶颈、
  decode=带宽瓶颈）→ 「本书要回答的 20 个问题」清单（每问标注解答章节）→
  五层架构图 + 三条设计原则（接口隔离/可插拔后端/状态即对象）→
  `litertlm_print` 初探模型文件（埋线第 7 章）。
- 图表：图 2-1 五层架构总图（学习网站 SVG 重制为书版）；图 2-2 Roofline 草图；
  表 2-1 benchmark 输出字段解读；表 2-2 二十个问题 → 章节对照。
- 实验：基准数据集第一次采集（全书数字之源，方法与原始数据入附录 D）。
- 素材：架构 SVG（已有）、synth.layers、engine.h 注释。

### 第二部 · 一个 token 的一生（3 章，约 70 页，全书脊柱）

**第 3 章 输入之路：从 Engine API 到 token 序列**（约 25 页）
- 使命：走通输入侧全程——API 设计、对话组装、模板渲染、分词。
- 核心问题：一个好的推理 API 长什么样？多轮对话如何不重复 prefill 历史（模板 diff 增量渲染）？
- 内容：Engine/Session 两级抽象与三层配置、EngineFactory 静态自注册；
  Conversation 的 Message(JSON)/Preface/历史管理；模板增量渲染 diff（本章高潮）；
  SentencePiece 与 HuggingFace 两种分词；model_data_processor 工厂（gemma3/4、qwen3 格式差异）。
- 图表：图 3-1 输入侧数据流（Message → 模板 diff → token ids）；
  图 3-2 Engine/Session/Conversation 关系；表 3-1 各模型 data processor 格式差异。
- 实验：20 行最小 C++ 调用；`renderMessageIntoString` 观察模板输出；双 tokenizer 对比。
- 素材：runtime/engine/、runtime/conversation/、components 的 tokenizer。
- 缺口：conversation.cc 的 diff 实现细节需补读。

**第 4 章 Prefill：吞下提示词**（约 22 页）
- 使命：理解 prefill 为什么快、静态/动态形状两条路径、以及支撑它的异步底座。
- 核心问题：为什么固定长度 signature 是移动端的合理权衡？长 prefill 如何做到及时取消？
- 内容：Tasks::Prefill 流程；SortedPrefillSignatureMap 选择 / prefill_chunk_size 分块；
  异步执行底座（ExecutionManager/任务队列/任务依赖链）；
  cancel 原子量 + max_prefill_sequence_length 限长保取消。
- 图表：图 4-1 prefill 时序（含异步任务队列）；图 4-2 静态 signature 选择与动态分块对照；
  表 4-1 静态/动态两路径权衡。
- 实验：prompt 长度 100→4000 扫描画 prefill 耗时曲线；async 开/关对比。
- 素材：runtime/core/tasks.cc、session_advanced.cc、runtime/framework/execution_queue、
  executor 的 PrefillInternal。

**第 5 章 Decode：逐 token 的心跳**（约 23 页）⭐ 样章
- 使命：走通 DecodeOneStep 完整循环——全书最核心一章。
- 核心问题：一个 token 的产生经过哪些步骤？内部/外部采样两条路径为何并存？
- 内容：自回归循环与 ShouldStop；采样(greedy/top-k/top-p/温度)；logits 处理挂载点；
  停止符检测（部分匹配回吐）；BPE 半截 token 的 MergeTokenIds；流式回调链到用户。
- 图表：图 5-1 DecodeOneStep 循环流程（内部/外部采样双路径）；
  图 5-2 停止词部分匹配回吐的状态变迁；表 5-1 采样策略对照。
- 实验：温度 0 与 1.0 对比；构造"停止词部分匹配"用例观察回吐。
- 素材：tasks.cc（Tasks::Decode、DecodeOneStep）、top_p_cpu_sampler、stop_token_detector。

### 第三部 · 快的代价（4 章，约 90 页，攻坚部分）

**第 6 章 KV cache 与会话状态**（约 25 页）
- 使命：从内存账到双缓冲实现，再到"状态即对象"的全部收益。
- 核心问题：KV cache 有多大（算给读者看）？GPU 为什么要双缓冲？
  克隆一段对话为什么可以不重算？
- 内容：开篇 Roofline 深化（与第 1/2 章的账对上）；KV cache 结构与内存公式；
  kv_cache_buffers_1_/2_ 指针互换免拷贝；LlmContext 可迁移状态；
  Clone / SaveCheckpoint / RewindToStep；thinking channel 内容从 KV 过滤（rewind 的应用）。
- 图表：图 6-1 KV cache 结构与随上下文增长示意；图 6-2 双缓冲指针互换；
  图 6-3 Clone/Rewind 的状态分叉；表 6-1 KV cache 内存账（模型 × 上下文长度）。
- 实验：`--max-num-tokens` 扫描看内存与速度（解释 `LiteRT-LM#2568`）；
  Clone 后分叉对话验证独立性；get_token_count 观察多轮增长。
- 素材：kv_cache_interface.h、executor 双缓冲成员、session_advanced 的 checkpoint。

**第 7 章 模型的形态：量化、.litertlm 格式与 LoRA**（约 22 页）
- 使命：理解"模型如何被压小、装箱、变体"的完整链路。
- 核心问题：int4 省的是什么（体积?带宽?算力?）？为什么要自造一个容器格式？
- 内容：int4/int8 量化收益账 + FP16 激活；.litertlm 分段设计
  （FlatBuffer header + sections + mmap 加载）；分段并行加载与冷启动；
  LoRA 适配器（挂载小增量权重，热加载）；weight cache。
- 图表：图 7-1 .litertlm 文件分段结构；图 7-2 mmap 与分段并行加载示意；
  表 7-1 量化精度收益账（体积/带宽/质量三角）。
- 实验：litertlm_print 解剖文件；int4 与 int8 对比（如社区有对应产物）；
  parallel_file_section_loading 开/关的冷启动差异。
- 素材：schema/core/、ActivationDataType、lora.cc/lora_manager.cc。
- 缺口：litertlm_read.cc 的 mmap 细节需补读。

**第 8 章 异构算力：CPU、GPU 与 NPU**（约 24 页）
- 使命：三类后端的本质差异、工厂分派、以及 CPU 侧的线程功课。
- 核心问题：同一模型换后端为什么行为/速度不同（`LiteRT-LM#2281`）？
  GPU 片上采样省掉的那次拷贝值多少？
- 内容：Backend 枚举与工厂分派；CPU(XNNPack) + 线程池/CPU 亲和性；
  GPU(OpenCL/Metal/WebGPU) 与片上采样；NPU(QNN) 与 embedder 缓冲共享
  （无真机，全程标注"基于代码分析"）。
- 图表：图 8-1 后端工厂分派；图 8-2 片上采样与回传采样的数据路径对比；
  表 8-1 CPU/GPU/NPU 特性权衡。
- 实验：cpu 与 gpu 同机对比；cpu_thread_count 扫描（`LiteRT-LM#2505` 加入的 flag）。
- 素材：executor factory、NPU executor、threadpool.cc、cpu_affinity_utils。

**第 9 章 一次前向，多个 token：推测解码与 MTP**（约 19 页）⭐ 压轴
- 使命：讲透 drafter/verifier 机制与接受率经济学——Gemma 4 的"快 3 倍"
  （官方博客口径，本章实验实测核对）从哪来。
- 核心问题：为什么"猜"能更快？接受率多低时反而更慢（解释 `LiteRT-LM#2227` 的 GPU 回退）？
- 内容：推测解码原理谱系（多模型与 MTP 单模型变体）；RunDraftingLoop/RunVerification；
  num_drafted/num_verified 的接受率统计；能力如何在 .litertlm 里声明。
- 图表：图 9-1 drafter/verifier 时序；图 9-2 接受率-收益曲线（用本章实测数据绘制）；
  表 9-1 MTP 开/关实测对照。
- 实验：`--enable-speculative-decoding` 开/关；代码与散文两种文体的接受率差异。
  注：`LiteRT-LM#2227` 的 PowerVR 回退现象无真机可复现，按【文档】级引用 issue 数据。
- 素材：llm_litert_mtp_drafter、schema/capabilities/speculative_decoding。
- 缺口：llm_litert_mtp_drafter.cc 仅读过头文件，实现需补读（压轴章硬依赖，入 P1 补读清单）。

### 第四部 · 能力与工程（2 章 + 尾声，约 55 页）

**第 10 章 不止聊天：多模态、约束解码与 Tool Use**（约 28 页）
- 使命：模型如何"看见/听见"，输出如何被约束成可执行的结构。
- 内容：视觉/音频编码器 → embedding 注入序列（special token 对齐约定、patchify）；
  约束解码（llguidance，逐步屏蔽非法 token）；Tool Use 全链路
  （Preface.tools 声明 → 约束保证结构 → function_gemma 解析回填）。
- 图表：图 10-1 多模态 embedding 注入序列；图 10-2 约束解码逐步屏蔽示意；
  表 10-1 Tool Use 全链路各环节职责。
- 实验：图片输入端到端 + 数 visual token 验证 patchify 公式；
  开/关约束解码对比工具调用成功率。
- 素材：vision/audio executor、preprocessor、logits_processor/constrained_decoding、tool_use/。
- 缺口：vision/audio executor 的 .cc 实现需补读（此前主要读头文件）；多模态模型体积数 GiB
  且 Gemma 系列为 HuggingFace 受限发布（需接受许可条款），下载与磁盘预算提前安排。
- 注：本章双主题（感知输入 + 受控输出），小节要切干净。

**第 11 章 一套核心，六种语言：C ABI、绑定与工程纪律**（约 20 页）
- 使命：跨语言桥的设计模式 + 生产级代码的工程实践。
- 内容：C ABI（不透明句柄/所有权/流式回调跨 FFI）；Python ctypes、Kotlin JNI、
  Swift actor、Web WASM 各自的封装术；`LiteRT-LM#2589`/`#2613`（deinit 时机）作缺陷案例；
  FakeLlmExecutor 可测试性设计；Bazel/CMake 双轨与 Copybara 工作流（简写）。
- 图表：图 11-1 C ABI 桥与各语言绑定结构；表 11-1 各语言 FFI 机制对照。
- 实验：同一 prompt 走 Python 与 C++ 验证行为一致；给 FakeLlmExecutor 写一个新用例。
- 素材：c/engine.h、python/_ffi.py、kotlin JNI、swift actor、fake_llm_executor。

**尾声 · 出发**（约 7 页）
- 三个集成路径速写（Android/iOS/Web）；如何找 issue 提 PR（引用真实 issue 全景）；
  端侧推理的下一步（更激进量化、NPU 普及、agent 化）。
- 图表：无（速写性质，不配图）。

### 附录（约 35 页）
- A. 术语表（工作文件 `appendix/glossary.md` 随章累积，成书时定稿；学习网站 90 条为底本）
- B. 代码地图（模块 → 文件 → 章节对照；学习网站内容的降级归宿）
- C. 环境搭建与全部实验复现命令（`experiments/` 脚本的汇总说明）
- D. 基准数据集说明（设备、模型、采集方法、原始数据）

### 页数预算汇总

| 部 | 章 | 页数 |
|---|---|---|
| 一 · 起点 | 1-2 | ~40 |
| 二 · token 的一生 | 3-5 | ~70 |
| 三 · 快的代价 | 6-9 | ~90 |
| 四 · 能力与工程 | 10-11 + 尾声 | ~55 |
| 附录 | A-D | ~35 |
| **合计** | **11 章 + 尾声** | **约 285（区间 250-300）✓** |

### v2→v3 压缩装配对照（内容去哪了）

| 原章（19 章版） | 去向 |
|---|---|
| 原 8 Roofline | → 第 2 章初讲 + 第 6 章开篇深化 |
| 原 13 线程/内存/调度 | → 异步底座入第 4 章；线程池/亲和性入第 8 章；并行加载入第 7 章 |
| 原 16 状态管理 | → Clone/Checkpoint/channel 入第 6 章；LoRA 入第 7 章；历史入第 3 章 |
| 原 17+18 绑定/工程 | → 合并为第 11 章 |
| 原 19 展望 | → 缩为尾声 |
| 原 4+5 / 原 14+15 | → 各自合并为第 3 章 / 第 10 章 |

**被压缩掉的深度**（诚实记录）：构建系统细节（Bazel/CMake 只保留一节）；
NPU 独立展开（并入第 8 章一节）；框架层线程实现细节（保留设计思想，略去代码级展开）。
若日后出"完整版/v2"，这些是首选扩展点。

### 「20 个问题」清单初稿（第 2 章的牵引装置，兼作目录覆盖度自检）

> 状态：初稿。样章完成后随第 2 章定稿；每问必须在标注章节得到正面回答。

| # | 读者的问题 | 解答章节 |
|---|---|---|
| 1 | 同样的模型，为什么云端流畅、手机上就吃力？ | 第 1 章 |
| 2 | 4B 参数的模型，8 GiB 内存的手机放得下吗？ | 第 1、7 章 |
| 3 | 为什么 prefill 每秒几千 token，decode 只有几十？ | 第 2 章初讲，第 6 章深化 |
| 4 | time-to-first-token 由哪几段时间构成？ | 第 2 章 |
| 5 | Engine 和 Session 为什么要分成两层？ | 第 3 章 |
| 6 | 多轮对话的历史，每一轮都要重新计算吗？ | 第 3 章（模板 diff）、第 6 章（KV cache） |
| 7 | 聊天模板是谁、在什么时候套上去的？ | 第 3 章 |
| 8 | 模型文件里为什么有好几个固定长度的 prefill 入口？ | 第 4 章 |
| 9 | 生成中途取消，为什么能立刻停下？ | 第 4 章 |
| 10 | 温度、top-k、top-p 各自改变了什么？ | 第 5 章 |
| 11 | 流式输出为什么偶尔"吐半个字"？ | 第 5 章 |
| 12 | 停止词只出现了一半时，吐不吐字？ | 第 5 章 |
| 13 | KV cache 占多少内存？`--max-num-tokens` 为什么影响速度？ | 第 6 章（`LiteRT-LM#2568`） |
| 14 | 克隆对话做分叉，需要重算公共前缀吗？ | 第 6 章 |
| 15 | int4 量化省的是体积、带宽还是算力？ | 第 7 章 |
| 16 | .litertlm 单文件里都装了什么？ | 第 7 章 |
| 17 | 换个后端，速度甚至输出为什么都会变？ | 第 8 章（`LiteRT-LM#2281`） |
| 18 | GPU 片上采样比拷回 CPU 采样快在哪？ | 第 8 章 |
| 19 | 推测解码靠"猜"，为什么反而更快？什么时候更慢？ | 第 9 章（`LiteRT-LM#2227`） |
| 20 | 模型怎么"看见"图片？输出怎么保证是合法 JSON？ | 第 10 章 |

覆盖度自检结论：11 章中除第 11 章外均被清单覆盖；第 11 章的问题（"一套 C++
怎么变出六种语言的 SDK？"）偏集成向，刻意不入面向推理读者的主清单，
在第 2 章正文以"工程读者另见第 11 章"一句带过。

---

## 四、路线图（六阶段，含验收门）

| 阶段 | 内容 | 时长 | 验收门 |
|---|---|---|---|
| **P0 定调** | 拍板 D1/D2/D4；冻结目录 v3；书名候选 | 1 周 | 每章通过"一句话使命"测试 |
| **P1 立骨** | 冻结 tag（建议 v0.13.1）并回填规范示例；选工具链（mdBook 或 Quarto）建私有仓；**规范基建**：`assets/book.css`（从学习网站复制变量表）、`scripts/lint_prose.sh`（禁词/填充词机检）、`appendix/glossary.md` 初始化、`review.md` 模板；data.json 素材迁入各章 notes.md；补读清单（tasks.cc、llm_litert_compiled_model_executor.cc、conversation.cc、litertlm_read.cc、llm_litert_mtp_drafter.cc、vision/audio executor 实现）；按下方规格 v1 采集基准数据集 | 2-3 周 | ⭐ 第 5 章样章通过下方"样章验收判据"全部四条 |
| **P2 长脊柱** | 第 3-5 章 | 3-4 周 | 内部版 v0.1（私有 tag）；自读复盘，可邀 1-2 位信任同行试读 |
| **P3 攻坚** | 第 6-9 章（实验最多） | 5-6 周 | 实验脚本全部入库可复现 |
| **P4 展翼** | 第 10-11 章 + 尾声 | 3 周 | 内部版 v0.3 |
| **P5 收官** | 第 1-2 章（最后写开头）；附录；统稿（含术语首现按成书顺序的专项校正 pass、图表编号与交叉引用核对） | 2-3 周 | 完整通读修订 → 内部版 v1.0 |
| **P6 开源/出版评估** | 核心版打磨满意后启动（非并行前置） | 待定 | 由作者决策是否开源、是否接触出版社 |

**总计**：约 4-5.5 个月出完整初稿（每周 0.7-1 章的可持续节奏）。

**核心版里程碑**：第一~三部（9 章，约 200 页）本身即完整的书——
全书核心承诺全部兑现；第四部与附录是完整版增量。这是防止烂尾的安全着陆点。

**写作顺序**：样章(5) → 脊柱(3-4) → 攻坚(6-9) → 展翼(10-11) → **最后写第 1、2 章**。
先写开头是烂尾书的常见病：等身体长出来，才知道该怎么开头。

**基准数据集规格 v1**（P1 采集；脚本入 `experiments/bench_baseline.sh`，原始数据入 `experiments/data/`，说明入附录 D）：
- 主基准设备：作者的 Mac（Apple Silicon；具体型号/内存/系统版本随采集记录）
- 主基准模型：**Gemma 4 E4B**（`litert-community/gemma-4-E4B-it-litert-lm`，公开非受限、支持 MTP；主模型即覆盖第 9 章 MTP 实测，无需另找）
- 条件矩阵：backend ∈ {cpu, gpu} × 上下文 ∈ {256, 1024, 4096} × MTP ∈ {关, 开（仅支持的模型）}；每条件跑 3 次取中位数
- 指标：prefill tokens/s、decode tokens/s、TTFT (ms)、峰值内存（`--report_peak_memory_footprint`）、冷启动加载时间
- 纪律：全书正文只引用本数据集与注明出处的官方数据（CLAUDE.md 第二节）；扩展基准（如 Android 真机）单独标注，不与主基准混算
- 可得性：litert-community 的 Gemma 4 版**非受限**，直接可下（约 3.4 GB），无需 HF 登录；google/ 官方版才受限

**样章（第 5 章）验收判据**（四条全过才算通过 P1 门）：
1. `review.md` 中断言归级完成率 100%，代码引用逐条 Read 核验通过；
2. `lint_prose.sh` 零命中；除 AI 味清单逐项打钩，第 14 条由独立会话执行；
3. 章卡的使命、图表清单（2 图 1 表）、实验脚本全部兑现且可一键复现；
4. 作者通读认可口吻与深度——通过后该章即全书"深度标尺"，后续章节向它对齐。

---

## 五、风险登记册

| 风险 | 概率 | 对策 |
|---|---|---|
| 上游演进导致内容过时 | 高 | 锁 tag +「版本注记」侧栏；讲设计与权衡而非易变细节 |
| 素材深度不足 | 中 | P1 补读清单硬性完成；章卡"缺口"栏跟踪 |
| 烂尾（内部先行后缺少公开连载的外部约束） | 中高 | 核心版里程碑 + 每章完成即打私有 tag + 固定双周自查复盘 + 验收看 review.md 留痕 + 样章先行 |
| 厚章写作失焦 | 中 | 每章先立 4-6 个小节的节级大纲再动笔；单节 ≤6 页 |
| 规范执行漂移（多会话乱序写作） | 中 | CLAUDE.md 逐会话加载；lint_prose.sh 机检；除 AI 味第 14 条由独立会话执行 |
| 与官方文档重叠 | 低 | 坚守"实现与权衡"视角 |
| NPU 等无真机部分失实 | 中 | 全程标注"基于代码分析"；不写无证据结论 |
| 单人精力波动 | 高 | notes.md 与正文分离，读码与写作分时段；内部评审以"节"为粒度 |

---

## 六、下一步行动

**P0 收尾**：
- [x] 拍板 D1（体例）、D2（读者）、D4（语言）——2026-07-05 作者确认
- [x] 逐章过"一句话使命"测试：11 章全部通过（第 2/10 章为刻意的复合使命章，
      约束已写入章卡），目录 v3 冻结——2026-07-05
- [x] 书名候选 3 个（含英文名），待作者选定：
      - **候选 A（描述型，推荐）**：《端侧大模型推理：原理与 LiteRT-LM 实现》
        *On-Device LLM Inference: Principles and Practice with LiteRT-LM*
        —— 检索友好、定位精准、出版社友好；缺点是平实。
      - **候选 B（叙事型）**：《一个 token 的一生：深入端侧大模型推理》
        *The Life of a Token: Inside On-Device LLM Inference*
        —— 记忆点强，直接呼应全书脊柱；缺点是书架上看不出主题词"LiteRT-LM"。
      - **候选 C（问题型）**：《大模型如何跑进手机：LiteRT-LM 深度解析》
        *How LLMs Run on Your Phone: A Deep Dive into LiteRT-LM*
        —— 用读者的问题当书名；"跑进手机"具体不夸大，但口语化程度最高。
      - 推荐组合：A 为主书名；B 的意象留给前言与第二部部题。
- [x] 作者选定书名：**候选 A**《端侧大模型推理：原理与 LiteRT-LM 实现》——2026-07-05。
      P0 全部完成 ✅

**P1 开工项**（依赖 P0 完成）：
- [x] 冻结 tag = v0.13.1，回填 CLAUDE.md 示例（两处示例行号已按 v0.13.1 核验：h:329、engine.h:238）——2026-07-05
- [x] 选定工具链：**mdBook**（内部打磨轻快优先；正文纯 markdown，日后可平迁 Quarto 出 PDF）；仓库 `git init` 本地私有——2026-07-05
- [x] 建规范基建：`assets/book.css`、`scripts/lint_prose.sh`、`appendix/glossary.md`(种子)、`chapters/_shared/review-template.md`——2026-07-05
- [x] 章节骨架 + data.json 素材迁移：`chapters/_shared/module-*.md`(12) + `synth-architecture.md` 为单一事实源，各章 `notes.md` 做策展层，`chapter.md` 骨架就位（`scripts/gen_scaffold.py` 可幂等重生成）——2026-07-05
- [ ] 补读 6 个文件（tasks.cc、llm_litert_compiled_model_executor.cc、conversation.cc、litertlm_read.cc、llm_litert_mtp_drafter.cc、vision/audio executor 实现）
- [进行中] 基准数据集采集（规格 v1，Gemma 4 E4B 已下载，矩阵采集中）
- [ ] 第 5 章样章（过样章验收判据四条）
