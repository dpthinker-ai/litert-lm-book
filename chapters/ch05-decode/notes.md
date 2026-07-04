<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->
# 第 5 章 · Decode：逐 token 的心跳 — notes（策展）

**一句话使命**：走通 DecodeOneStep 完整循环——全书最核心一章（样章）。

## 素材取用（单一事实源在 `chapters/_shared/`）
- [`module-core-pipeline.md`](../_shared/module-core-pipeline.md) — 核心调度 (Core Pipeline)
- [`module-components-text.md`](../_shared/module-components-text.md) — 文本组件: 分词与采样 (Text Components: Tokenization & Sampling)

**聚焦**：取 core 的 Tasks::Decode/DecodeOneStep/ShouldStop、components-text 的采样与停止符检测（部分匹配回吐）；这是样章，深度标尺按此定。

## 本章图表（先规划后动笔，CLAUDE.md 第六节）
- [ ] 图 5-1 DecodeOneStep 循环流程（内部/外部采样双路径）
- [ ] 图 5-2 停止词部分匹配回吐的状态变迁
- [ ] 表 5-1 采样策略对照

## 本章实验（脚本入 `experiments/`）
- [ ] 温度 0 与 1.0 对比
- [ ] 构造「停止词部分匹配」用例观察回吐

## 补读 / 缺口（写作前须清零）
- （无额外补读；仍须按四级制核验每条引用）

## 待核实清单 / 随手记
- 