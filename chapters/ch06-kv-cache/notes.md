<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->
# 第 6 章 · KV cache 与会话状态 — notes（策展）

**一句话使命**：从内存账到双缓冲实现，再到「状态即对象」的全部收益。

## 素材取用（单一事实源在 `chapters/_shared/`）
- [`module-executor-llm.md`](../_shared/module-executor-llm.md) — LLM 执行器 (Executor)
- [`module-core-pipeline.md`](../_shared/module-core-pipeline.md) — 核心调度 (Core Pipeline)

**聚焦**：取 executor-llm 的 KV cache 双缓冲、core 的 Clone/SaveCheckpoint/RewindToStep；开篇做 Roofline 深化（与第 1/2 章的账对上）。

## 本章图表（先规划后动笔，CLAUDE.md 第六节）
- [ ] 图 6-1 KV cache 结构与随上下文增长示意
- [ ] 图 6-2 双缓冲指针互换
- [ ] 图 6-3 Clone/Rewind 的状态分叉
- [ ] 表 6-1 KV cache 内存账（模型 × 上下文长度）

## 本章实验（脚本入 `experiments/`）
- [ ] --max-num-tokens 扫描看内存与速度（解释 LiteRT-LM#2568）
- [ ] Clone 后分叉对话验证独立性
- [ ] get_token_count 观察多轮增长

## 补读 / 缺口（写作前须清零）
- （无额外补读；仍须按四级制核验每条引用）

## 待核实清单 / 随手记
- 