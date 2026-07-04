# 第 5 章 review.md （⭐样章：本章为全书深度标尺）

> 首稿验收留痕。状态：**初稿完成，Pass 1 部分完成，Pass 2 lint 已过、独立审校（§14）待做**。

## Pass 1 · 事实核查（断言四级制 + 引用逐条核验）

引用核验（本会话用 `git show v0.13.1:<path>` 逐条比对，标 ✅ 者已确认支撑断言）：
- tasks.cc:111(DecodeOneStep)/446(Decode)/86(ShouldStop)/110(内外采样注释)/319(DecodeAndSample)/268(DecodeLogits) ✅
- sampler.h:34(Sampler) ✅；top_p_cpu_sampler.h:30(TopPSampler)/38(Create) ✅
- stop_token_detector.h:45(类)/67(ProcessTokens)/96(MaxPartialStopTokenLength) ✅

- 【推测】级断言：均已显式标注（"我们推测/据此推断"或"待补/待核验"）✅
- 「对照视野」侧栏：仅陈述取舍不裁决优劣，其他框架断言标【文档】级、出处待 P5 回填 ✅
- 数字：性能上限为纸面算账（可复算），实测数字统一标注待基准 D 回填 ✅

## Pass 2 · 除 AI 味

- [x] `scripts/lint_prose.sh chapters/ch05-decode/chapter.md` —— **0 命中**（2026-07-05）
- [x] 除 AI 味清单第 1-13 条：撰写时自查（空洞总结句/填充词/排比/轻动词/同义堆叠/翻译腔/句式/列表节制/段落节奏/破折号）
- [ ] 第 14 条 独立审校：**待独立会话或 prose-review skill 执行**（禁本章撰写会话自评，故此项 P1 内单列）

## 待清零（补读/数据）
- [ ] 温度对比/停止词回吐实验数字（基准 D）
- [ ] BPE 半字 MergeTokenIds 精确行号（补读 tasks.cc）

## 验收自问
- [x] 本章"一句话使命"兑现
- [x] 图表清单已落地并编号（见章内 figure 与 notes.md）
- [ ] 实验可一键复现（脚本待随基准数据集入 `experiments/`）
- [x] 与 BOOK_PLAN 章卡一致；交叉引用用"见第 N 章"格式
