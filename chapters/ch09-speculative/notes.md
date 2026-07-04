<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->
# 第 9 章 · 一次前向，多个 token：推测解码与 MTP — notes（策展）

**一句话使命**：讲透 drafter/verifier 机制与接受率经济学——Gemma 4「快 3 倍」（官方口径，实测核对）从哪来。

## 素材取用（单一事实源在 `chapters/_shared/`）
- [`module-executor-llm.md`](../_shared/module-executor-llm.md) — LLM 执行器 (Executor)
- [`module-schema-format.md`](../_shared/module-schema-format.md) — 模型格式与 Schema (Model Format & Schema)

**聚焦**：取 executor-llm 的 MTP drafter、schema-format 的 speculative_decoding 能力声明；#2227 的 PowerVR 回退无真机，按【文档】级引 issue。

## 本章图表（先规划后动笔，CLAUDE.md 第六节）
- [ ] 图 9-1 drafter/verifier 时序
- [ ] 图 9-2 接受率-收益曲线（用本章实测数据绘制）
- [ ] 表 9-1 MTP 开/关实测对照

## 本章实验（脚本入 `experiments/`）
- [ ] --enable-speculative-decoding 开/关
- [ ] 代码与散文两种文体的接受率差异

## 补读 / 缺口（写作前须清零）
- [ ] llm_litert_mtp_drafter.cc 仅读过头文件，实现需补读（压轴章硬依赖）

## 待核实清单 / 随手记
- 