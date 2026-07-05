# 第 9 章 review.md （⭐压轴）

> 首稿验收留痕。状态：**初稿完成，Pass 1 完成（含实现补读），Pass 2 lint 已过、独立审校（§14）待做**。

## Pass 1 · 事实核查（断言四级制 + 引用逐条核验）

**本章代码引用来自实现补读**（`llm_litert_mtp_drafter.cc @ v0.13.1` 逐行核验，见 notes.md 补读结论）：
- Draft:453；RunDraftingLoop:328/336；RunVerification:437(RunAsync verify)/449(RET_CHECK size G+1) ✅
- 接受循环:473-483（首个不匹配取 bonus / 全对白赚第 G+1）；输出:490-492；统计:493-494；析构接受率:165-171 ✅
- num_draft_steps 固定:256(verify input_pos 维度-1)；verify signature:63/228 ✅
- HasSpeculativeDecodingSupport（speculative_decoding.h:33/44）✅
- 贴出的接受循环代码片段与 v0.13.1 逐字一致，仅加中文行末注释说明（未改源码逻辑）；已核验 ✅

- 「约 3 倍」：明标 Gemma 4 官方博客【文档】级，实测待基准 D（需支持 MTP 的 Gemma 4 类模型，非 3n E2B）✅
- `#2227`（某 GPU 上 MTP 更慢）：现象【文档】级引用；成因为本章接受率经济账的推断，明标"无真机、基于推断" ✅
- 接受率经济学、bonus 保底"token 数不亏"：均由实读逻辑直接推出，非臆测 ✅

## Pass 2 · 除 AI 味

- [x] `scripts/lint_prose.sh chapters/ch09-speculative/chapter.md` —— 0 命中（2026-07-05）
- [x] 除 AI 味清单第 1-13 条：撰写时自查（压轴章，收尾避免自夸/元评论）
- [ ] 第 14 条 独立审校：**待独立会话执行**（与 ch6/7/8 一起攒批审）

## 待清零（补读/数据）
- [x] MTP drafter 实现补读 —— 已完成（commit bf832b0）
- [ ] 实测（MTP 开/关、文体接受率）待基准 D（需 Gemma 4 类 MTP 模型）
- [ ] 图 9-2（接受率-收益曲线）、表 9-1（开/关实测）需实测数据，待基准 D（本轮出签名图 9-1 时序）

## 验收自问
- [x] 使命兑现（讲透 drafter/verifier 机制 + 接受率经济学 + 为何有时更慢；结清第 19 问）
- [x] 图 9-1 已落地并编号
- [ ] 实验可复现（脚本随基准 D）
- [x] 与 BOOK_PLAN 章卡一致；承接第 1 章带宽墙第三招、第 4 章固定形状、第 7 章能力声明；收束第三部
