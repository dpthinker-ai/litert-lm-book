# 第 2 章 review.md 

> 首稿验收留痕。状态：**初稿完成，Pass 1 部分完成，Pass 2 lint 已过、独立审校（§14）待做**。

## Pass 1 · 事实核查（断言四级制 + 引用逐条核验）

引用核验（本会话用 `git show v0.13.1:<path>` 逐条比对，标 ✅ 者已确认支撑断言）：
- litert_lm_main.cc:52（backend 默认 gpu）✅ git show 核验
- litert_lm_main.cc:54（model_path）✅
- c/engine.h:583/591/634/643（BenchmarkInfo 四字段）✅
- schema/core/litertlm_print.cc 存在 ✅

- 【推测】级断言：均已显式标注（"我们推测/据此推断"或"待补/待核验"）✅
- 「对照视野」侧栏：仅陈述取舍不裁决优劣，其他框架断言标【文档】级、出处待 P5 回填 ✅
- 数字：性能上限为纸面算账（可复算），实测数字统一标注待基准 D 回填 ✅

## Pass 2 · 除 AI 味

- [x] `scripts/lint_prose.sh chapters/ch02-run-and-overview/chapter.md` —— **0 命中**（2026-07-05）
- [x] 除 AI 味清单第 1-13 条：撰写时自查（空洞总结句/填充词/排比/轻动词/同义堆叠/翻译腔/句式/列表节制/段落节奏/破折号）
- [ ] 第 14 条 独立审校：**待独立会话或 prose-review skill 执行**（禁本章撰写会话自评，故此项 P1 内单列）

## 待清零（补读/数据）
- [ ] 基准数据集实测数字（TTFT/prefill/decode tok-s）回填「〔基准 D〕」
- [ ] 表 2-2 完整 20 问

## 验收自问
- [x] 本章"一句话使命"兑现
- [x] 图表清单已落地并编号（见章内 figure 与 notes.md）
- [ ] 实验可一键复现（脚本待随基准数据集入 `experiments/`）
- [x] 与 BOOK_PLAN 章卡一致；交叉引用用"见第 N 章"格式
