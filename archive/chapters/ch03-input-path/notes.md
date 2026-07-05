<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->
# 第 3 章 · 输入之路：从 Engine API 到 token 序列 — notes（策展）

**一句话使命**：走通输入侧全程——API 设计、对话组装、模板渲染、分词。

## 素材取用（单一事实源在 `chapters/_shared/`）
- [`module-engine-api.md`](../_shared/module-engine-api.md) — Engine 公共 API 层
- [`module-conversation.md`](../_shared/module-conversation.md) — 对话层 (Conversation)
- [`module-components-text.md`](../_shared/module-components-text.md) — 文本组件: 分词与采样 (Text Components: Tokenization & Sampling)

**聚焦**：重点取 conversation 的模板 diff 增量渲染（本章高潮）、engine-api 的两级抽象、components-text 的两种 tokenizer；components-text 的采样/停止部分留给第 5 章。

## 本章图表（先规划后动笔，CLAUDE.md 第六节）
- [ ] 图 3-1 输入侧数据流（Message → 模板 diff → token ids）
- [ ] 图 3-2 Engine/Session/Conversation 关系
- [ ] 表 3-1 各模型 data processor 格式差异

## 本章实验（脚本入 `experiments/`）
- [ ] 20 行最小 C++ 调用
- [ ] renderMessageIntoString 观察模板输出
- [ ] 双 tokenizer 对比

## 补读 / 缺口（写作前须清零）
- [ ] conversation.cc 的 diff 实现细节需补读

## 待核实清单 / 随手记
- 