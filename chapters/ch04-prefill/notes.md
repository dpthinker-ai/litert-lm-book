<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->
# 第 4 章 · Prefill：吞下提示词 — notes（策展）

**一句话使命**：理解 prefill 为什么快、静态/动态形状两条路径、以及支撑它的异步底座。

## 素材取用（单一事实源在 `chapters/_shared/`）
- [`module-core-pipeline.md`](../_shared/module-core-pipeline.md) — 核心调度 (Core Pipeline)
- [`module-executor-llm.md`](../_shared/module-executor-llm.md) — LLM 执行器 (Executor)
- [`module-framework.md`](../_shared/module-framework.md) — 并发框架 (Framework)

**聚焦**：取 executor-llm 的 prefill 路径（signature/分块）、core 的 Tasks::Prefill、framework 的异步任务队列；executor 的 decode/KV 部分留给第 5、6 章。

## 本章图表（先规划后动笔，CLAUDE.md 第六节）
- [ ] 图 4-1 prefill 时序（含异步任务队列）
- [ ] 图 4-2 静态 signature 选择与动态分块对照
- [ ] 表 4-1 静态/动态两路径权衡

## 本章实验（脚本入 `experiments/`）
- [ ] prompt 长度 100→4000 扫描画 prefill 耗时曲线
- [ ] async 开/关对比

## 补读 / 缺口（写作前须清零）
- （无额外补读；仍须按四级制核验每条引用）

## 待核实清单 / 随手记
- 