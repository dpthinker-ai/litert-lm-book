<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->
# 第 10 章 · 不止聊天：多模态、约束解码与 Tool Use — notes（策展）

**一句话使命**：模型如何「看见/听见」，输出如何被约束成可执行的结构。

## 素材取用（单一事实源在 `chapters/_shared/`）
- [`module-executor-multimodal.md`](../_shared/module-executor-multimodal.md) — 多模态执行器 (Multimodal Executor: Vision/Audio)
- [`module-components-resources.md`](../_shared/module-components-resources.md) — 模型资源与扩展组件 (Model Resources & Extension Components)
- [`module-components-text.md`](../_shared/module-components-text.md) — 文本组件: 分词与采样 (Text Components: Tokenization & Sampling)

**聚焦**：取 executor-multimodal 的 embedding 注入、components 的约束解码(llguidance)与 tool_use；双主题，小节切干净。

## 本章图表（先规划后动笔，CLAUDE.md 第六节）
- [ ] 图 10-1 多模态 embedding 注入序列
- [ ] 图 10-2 约束解码逐步屏蔽示意
- [ ] 表 10-1 Tool Use 全链路各环节职责

## 本章实验（脚本入 `experiments/`）
- [ ] 图片输入端到端 + 数 visual token 验证 patchify 公式
- [ ] 开/关约束解码对比工具调用成功率

## 补读 / 缺口（写作前须清零）
- [ ] vision/audio executor 的 .cc 实现需补读
- [ ] 多模态模型数 GiB + Gemma 为 HF 受限发布，下载/磁盘预算提前安排

## 待核实清单 / 随手记
- 