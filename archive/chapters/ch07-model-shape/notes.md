<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->
# 第 7 章 · 模型的形态：量化、.litertlm 格式与 LoRA — notes（策展）

**一句话使命**：理解「模型如何被压小、装箱、变体」的完整链路。

## 素材取用（单一事实源在 `chapters/_shared/`）
- [`module-schema-format.md`](../_shared/module-schema-format.md) — 模型格式与 Schema (Model Format & Schema)
- [`module-components-resources.md`](../_shared/module-components-resources.md) — 模型资源与扩展组件 (Model Resources & Extension Components)

**聚焦**：取 schema-format 的 .litertlm 分段/mmap、components-resources 的 LoRA；量化收益账结合第 1 章带宽墙。

## 本章图表（先规划后动笔，CLAUDE.md 第六节）
- [ ] 图 7-1 .litertlm 文件分段结构
- [ ] 图 7-2 mmap 与分段并行加载示意
- [ ] 表 7-1 量化精度收益账（体积/带宽/质量三角）

## 本章实验（脚本入 `experiments/`）
- [ ] litertlm_print 解剖文件
- [ ] int4 与 int8 对比（如社区有对应产物）
- [ ] parallel_file_section_loading 开/关的冷启动差异

## 补读 / 缺口（写作前须清零）
- [ ] litertlm_read.cc 的 mmap 细节需补读

## 待核实清单 / 随手记
- 