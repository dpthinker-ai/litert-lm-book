#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 litert-lm-guide/data.json 生成书稿脚手架：
   _shared/module-*.md（模块素材单一事实源）+ 各章 notes.md（策展层）+ chapter.md（骨架）。
   幂等：重复运行覆盖 _shared/* 与 chapter.md 骨架；notes.md 若已被人工编辑则跳过（见 --force）。
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "..", "litert-lm-guide", "data.json")
FORCE = "--force" in sys.argv

d = json.load(open(DATA, encoding="utf-8"))
mods = {m["id"]: m for m in d["modules"]}
synth = d["synth"]

def w(path, content, skip_if_edited=False):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if skip_if_edited and os.path.exists(full) and not FORCE:
        # 若文件已存在且包含人工编辑标记，跳过
        if "<!-- edited -->" in open(full, encoding="utf-8").read():
            print(f"跳过（已人工编辑）：{path}"); return
    open(full, "w", encoding="utf-8").write(content)
    print(f"写入：{path}")

# ---------- 1. 每模块素材 → _shared/module-<id>.md ----------
def render_module(m):
    b = [f"# 模块素材：{m['title']}  `{m['id']}`\n",
         f"> 来源：litert-lm-guide/data.json（多智能体源码分析）。**这是素材，不是正文**。",
         f"> 引用进正文前，须按 CLAUDE.md 第二节逐条 Read 源码核验、补 `@ v0.13.1`。\n",
         f"**一句话**：{m['summary']}\n",
         f"**在架构中的位置**：{m['role_in_arch']}\n"]
    if m.get("key_files"):
        b.append("## 关键文件")
        for f in m["key_files"]:
            b.append(f"- `{f['path']}` — {f['desc']}")
        b.append("")
    if m.get("key_abstractions"):
        b.append("## 核心抽象")
        for a in m["key_abstractions"]:
            fileref = f" 〔`{a.get('file','')}`〕" if a.get("file") else ""
            b.append(f"- **{a['name']}** ({a.get('kind','')}){fileref}：{a['desc']}")
        b.append("")
    if m.get("data_flow"):
        b.append("## 数据流")
        for i, s in enumerate(m["data_flow"], 1):
            b.append(f"{i}. {s}")
        b.append("")
    if m.get("concepts"):
        b.append("## 概念")
        for c in m["concepts"]:
            b.append(f"- **{c['term']}**：{c['explanation']}")
        b.append("")
    if m.get("optimizations"):
        b.append("## 优化")
        for o in m["optimizations"]:
            b.append(f"- **{o['name']}**：{o['desc']}")
        b.append("")
    if m.get("code_snippets"):
        b.append("## 关键代码片段（待核验 @ v0.13.1）")
        for sn in m["code_snippets"]:
            b.append(f"**{sn['caption']}**" + (f" — 待核验：`{sn.get('ref','')}`" if sn.get("ref") else ""))
            b.append(f"```{sn.get('lang','')}\n{sn['code']}\n```")
        b.append("")
    if m.get("how_to_start"):
        b.append("## 入手顺序")
        for s in m["how_to_start"]:
            b.append(f"- {s}")
        b.append("")
    return "\n".join(b)

for mid, m in mods.items():
    w(f"chapters/_shared/module-{mid}.md", render_module(m))

# ---------- 2. 架构综述 → _shared/synth-architecture.md ----------
def render_synth(s):
    b = ["# 架构综述素材（synth）\n", "> 来源同上；素材，非正文。\n", "## 总览"]
    for p in s["overview"].split("\n"):
        if p.strip(): b.append(p)
    b.append("\n## 分层")
    for L in s["layers"]:
        b.append(f"- **{L['name']}**：{L['desc']}  〔模块：{', '.join(L['modules'])}〕")
    b.append("\n## 端到端流程")
    for st in s["end_to_end_flow"]:
        files = f"  〔{st['files']}〕" if st.get("files") else ""
        b.append(f"- **{st['step']}**：{st['detail']}{files}")
    b.append("\n## 设计亮点")
    for h in s["design_highlights"]:
        b.append(f"- **{h['title']}**：{h['desc']}")
    b.append("\n## 全局术语（90 条底本节选，写作时并入 appendix/glossary.md）")
    for g in s["glossary"]:
        b.append(f"- **{g['term']}**：{g['def']}")
    b.append("\n## 架构图 Mermaid 源（第 2 章将重制为书版 SVG）")
    b.append(f"```mermaid\n{s['mermaid']}\n```")
    return "\n".join(b)

w("chapters/_shared/synth-architecture.md", render_synth(synth))

# ---------- 3. 章节策展 notes.md + 骨架 chapter.md ----------
CH = [
 ("ch01-three-walls", 1, "端侧 LLM：三堵墙与一张版图",
  "读者能亲手算出「4B 模型在手机上的理论 decode 上限」这笔账。",
  ["concepts"],
  "以背景与算账为主，代码素材少。用 concepts 模块的量化/后端/带宽相关概念；SoC 参数须另查官方来源（【文档】级）。",
  ["图 1-1 三堵墙与模型需求对照", "表 1-1 端侧运行时版图对比"],
  ["纸面算账（公式与代入过程完整给出，第 6 章实测对账）"],
  ["收集 2-3 款典型 SoC 的带宽/算力参数（官方来源）"]),
 ("ch02-run-and-overview", 2, "跑起来与鸟瞰：从 benchmark 数字到五层架构",
  "跑通 + 会读性能数字 + 获得全书地图。",
  ["engine-api"],
  "架构总图与五层用 synth-architecture.md；benchmark 字段解读靠实机采集（附录 D）；「20 个问题」清单见 BOOK_PLAN 第三部。",
  ["图 2-1 五层架构总图（学习网站 SVG 重制为书版）", "图 2-2 Roofline 草图",
   "表 2-1 benchmark 输出字段解读", "表 2-2 二十个问题 → 章节对照"],
  ["基准数据集第一次采集（全书数字之源，方法与原始数据入附录 D）"],
  []),
 ("ch03-input-path", 3, "输入之路：从 Engine API 到 token 序列",
  "走通输入侧全程——API 设计、对话组装、模板渲染、分词。",
  ["engine-api", "conversation", "components-text"],
  "重点取 conversation 的模板 diff 增量渲染（本章高潮）、engine-api 的两级抽象、components-text 的两种 tokenizer；components-text 的采样/停止部分留给第 5 章。",
  ["图 3-1 输入侧数据流（Message → 模板 diff → token ids）",
   "图 3-2 Engine/Session/Conversation 关系", "表 3-1 各模型 data processor 格式差异"],
  ["20 行最小 C++ 调用", "renderMessageIntoString 观察模板输出", "双 tokenizer 对比"],
  ["conversation.cc 的 diff 实现细节需补读"]),
 ("ch04-prefill", 4, "Prefill：吞下提示词",
  "理解 prefill 为什么快、静态/动态形状两条路径、以及支撑它的异步底座。",
  ["core-pipeline", "executor-llm", "framework"],
  "取 executor-llm 的 prefill 路径（signature/分块）、core 的 Tasks::Prefill、framework 的异步任务队列；executor 的 decode/KV 部分留给第 5、6 章。",
  ["图 4-1 prefill 时序（含异步任务队列）", "图 4-2 静态 signature 选择与动态分块对照",
   "表 4-1 静态/动态两路径权衡"],
  ["prompt 长度 100→4000 扫描画 prefill 耗时曲线", "async 开/关对比"],
  []),
 ("ch05-decode", 5, "Decode：逐 token 的心跳",
  "走通 DecodeOneStep 完整循环——全书最核心一章（样章）。",
  ["core-pipeline", "components-text"],
  "取 core 的 Tasks::Decode/DecodeOneStep/ShouldStop、components-text 的采样与停止符检测（部分匹配回吐）；这是样章，深度标尺按此定。",
  ["图 5-1 DecodeOneStep 循环流程（内部/外部采样双路径）",
   "图 5-2 停止词部分匹配回吐的状态变迁", "表 5-1 采样策略对照"],
  ["温度 0 与 1.0 对比", "构造「停止词部分匹配」用例观察回吐"],
  []),
 ("ch06-kv-cache", 6, "KV cache 与会话状态",
  "从内存账到双缓冲实现，再到「状态即对象」的全部收益。",
  ["executor-llm", "core-pipeline"],
  "取 executor-llm 的 KV cache 双缓冲、core 的 Clone/SaveCheckpoint/RewindToStep；开篇做 Roofline 深化（与第 1/2 章的账对上）。",
  ["图 6-1 KV cache 结构与随上下文增长示意", "图 6-2 双缓冲指针互换",
   "图 6-3 Clone/Rewind 的状态分叉", "表 6-1 KV cache 内存账（模型 × 上下文长度）"],
  ["--max-num-tokens 扫描看内存与速度（解释 LiteRT-LM#2568）",
   "Clone 后分叉对话验证独立性", "get_token_count 观察多轮增长"],
  []),
 ("ch07-model-shape", 7, "模型的形态：量化、.litertlm 格式与 LoRA",
  "理解「模型如何被压小、装箱、变体」的完整链路。",
  ["schema-format", "components-resources"],
  "取 schema-format 的 .litertlm 分段/mmap、components-resources 的 LoRA；量化收益账结合第 1 章带宽墙。",
  ["图 7-1 .litertlm 文件分段结构", "图 7-2 mmap 与分段并行加载示意",
   "表 7-1 量化精度收益账（体积/带宽/质量三角）"],
  ["litertlm_print 解剖文件", "int4 与 int8 对比（如社区有对应产物）",
   "parallel_file_section_loading 开/关的冷启动差异"],
  ["litertlm_read.cc 的 mmap 细节需补读"]),
 ("ch08-heterogeneous", 8, "异构算力：CPU、GPU 与 NPU",
  "三类后端的本质差异、工厂分派、以及 CPU 侧的线程功课。",
  ["executor-llm", "framework"],
  "取 executor-llm 的 Backend 分派/片上采样/NPU、framework 的线程池与 CPU 亲和性；NPU 无真机，全程标注「基于代码分析」。",
  ["图 8-1 后端工厂分派", "图 8-2 片上采样与回传采样的数据路径对比",
   "表 8-1 CPU/GPU/NPU 特性权衡"],
  ["cpu 与 gpu 同机对比", "cpu_thread_count 扫描（LiteRT-LM#2505）"],
  []),
 ("ch09-speculative", 9, "一次前向，多个 token：推测解码与 MTP",
  "讲透 drafter/verifier 机制与接受率经济学——Gemma 4「快 3 倍」（官方口径，实测核对）从哪来。",
  ["executor-llm", "schema-format"],
  "取 executor-llm 的 MTP drafter、schema-format 的 speculative_decoding 能力声明；#2227 的 PowerVR 回退无真机，按【文档】级引 issue。",
  ["图 9-1 drafter/verifier 时序", "图 9-2 接受率-收益曲线（用本章实测数据绘制）",
   "表 9-1 MTP 开/关实测对照"],
  ["--enable-speculative-decoding 开/关", "代码与散文两种文体的接受率差异"],
  ["llm_litert_mtp_drafter.cc 仅读过头文件，实现需补读（压轴章硬依赖）"]),
 ("ch10-multimodal", 10, "不止聊天：多模态、约束解码与 Tool Use",
  "模型如何「看见/听见」，输出如何被约束成可执行的结构。",
  ["executor-multimodal", "components-resources", "components-text"],
  "取 executor-multimodal 的 embedding 注入、components 的约束解码(llguidance)与 tool_use；双主题，小节切干净。",
  ["图 10-1 多模态 embedding 注入序列", "图 10-2 约束解码逐步屏蔽示意",
   "表 10-1 Tool Use 全链路各环节职责"],
  ["图片输入端到端 + 数 visual token 验证 patchify 公式", "开/关约束解码对比工具调用成功率"],
  ["vision/audio executor 的 .cc 实现需补读", "多模态模型数 GiB + Gemma 为 HF 受限发布，下载/磁盘预算提前安排"]),
 ("ch11-bindings", 11, "一套核心，六种语言：C ABI、绑定与工程纪律",
  "跨语言桥的设计模式 + 生产级代码的工程实践。",
  ["bindings", "build-docs"],
  "取 bindings 的 C ABI/各语言封装、build-docs 的测试与构建；#2589/#2613（deinit 时机）作缺陷案例。",
  ["图 11-1 C ABI 桥与各语言绑定结构", "表 11-1 各语言 FFI 机制对照"],
  ["同一 prompt 走 Python 与 C++ 验证行为一致", "给 FakeLlmExecutor 写一个新用例"],
  []),
]

for slug, num, title, mission, modules, focus, figs, exps, gaps in CH:
    # notes.md（策展层）
    nb = [f"<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->",
          f"# 第 {num} 章 · {title} — notes（策展）\n",
          f"**一句话使命**：{mission}\n",
          f"## 素材取用（单一事实源在 `chapters/_shared/`）"]
    for mid in modules:
        title_m = mods[mid]["title"] if mid in mods else mid
        nb.append(f"- [`module-{mid}.md`](../_shared/module-{mid}.md) — {title_m}")
    if num == 2:
        nb.append(f"- [`synth-architecture.md`](../_shared/synth-architecture.md) — 五层架构与端到端流程")
    nb.append(f"\n**聚焦**：{focus}\n")
    nb.append("## 本章图表（先规划后动笔，CLAUDE.md 第六节）")
    for f in figs: nb.append(f"- [ ] {f}")
    nb.append("\n## 本章实验（脚本入 `experiments/`）")
    for e in exps: nb.append(f"- [ ] {e}")
    nb.append("\n## 补读 / 缺口（写作前须清零）")
    if gaps:
        for g in gaps: nb.append(f"- [ ] {g}")
    else:
        nb.append("- （无额外补读；仍须按四级制核验每条引用）")
    nb.append("\n## 待核实清单 / 随手记")
    nb.append("- ")
    w(f"chapters/{slug}/notes.md", "\n".join(nb), skip_if_edited=True)

    # chapter.md（骨架）
    star = " ⭐样章" if num == 5 else (" ⭐压轴" if num == 9 else "")
    cb = [f"# 第 {num} 章 {title}{star}\n",
          f"> 状态：骨架（未开写）｜使命：{mission}",
          f"> 素材见 `notes.md`｜两个 pass 留痕见 `review.md`（模板在 `chapters/_shared/review-template.md`）\n",
          "<!-- 正文从这里开始。节级大纲：4-6 节，单节 ≤6 页；先写节标题再填充。 -->\n",
          "## （节 1 标题）\n",
          "## （节 2 标题）\n"]
    w(f"chapters/{slug}/chapter.md", "\n".join(cb))

# ---------- 4. 尾声、前言、附录 stub ----------
w("chapters/epilogue/chapter.md",
  "# 尾声 · 出发\n\n> 状态：骨架｜三个集成路径速写（Android/iOS/Web）；如何找 issue 提 PR；端侧推理的下一步。\n")
w("preface.md",
  "# 前言\n\n> 状态：骨架（P5 最后写）。\n> 「一个 token 的一生」意象在此点题；说明本书定位（非官方深度解析）、读者画像、代码锚点 v0.13.1、如何使用本书。\n")
for ap, name in [("code-map","B 代码地图"),("reproduce","C 环境搭建与实验复现"),("benchmark-dataset","D 基准数据集")]:
    w(f"appendix/{ap}.md", f"# 附录 {name}\n\n> 状态：骨架（P5 定稿）。\n", skip_if_edited=True)

print("\n完成。_shared 模块素材 + 12 章 notes/chapter 骨架 + 附录 stub 已生成。")
