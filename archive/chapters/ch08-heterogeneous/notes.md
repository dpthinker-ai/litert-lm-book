<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->
# 第 8 章 · 异构算力：CPU、GPU 与 NPU — notes（策展）

**一句话使命**：三类后端的本质差异、工厂分派、以及 CPU 侧的线程功课。

## 素材取用（单一事实源在 `chapters/_shared/`）
- [`module-executor-llm.md`](../_shared/module-executor-llm.md) — LLM 执行器 (Executor)
- [`module-framework.md`](../_shared/module-framework.md) — 并发框架 (Framework)

**聚焦**：取 executor-llm 的 Backend 分派/片上采样/NPU、framework 的线程池与 CPU 亲和性；NPU 无真机，全程标注「基于代码分析」。

## 本章图表（先规划后动笔，CLAUDE.md 第六节）
- [ ] 图 8-1 后端工厂分派
- [ ] 图 8-2 片上采样与回传采样的数据路径对比
- [ ] 表 8-1 CPU/GPU/NPU 特性权衡

## 本章实验（脚本入 `experiments/`）
- [ ] cpu 与 gpu 同机对比
- [ ] cpu_thread_count 扫描（LiteRT-LM#2505）

## 补读 / 缺口（写作前须清零）
- （无额外补读；仍须按四级制核验每条引用）

## 待核实清单 / 随手记
- 