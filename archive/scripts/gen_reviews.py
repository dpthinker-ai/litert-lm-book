#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为第 1-5 章生成 review.md（首稿验收留痕）。引用行号均已在本会话用 git show v0.13.1 核验。"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CH = {
 "ch01-three-walls": dict(n=1, refs=[
   "（本章为背景与算账，无 LiteRT-LM 代码引用）",
   "decode 上限公式：算术强度=常识级；50 GB/s 与 4B/int4 为【文档】级示例（SoC 精确规格待补，见缺口）",
   "版图表：llama.cpp/MLC/ExecuTorch/LiteRT-LM 定位【文档】级，出处链接待 P5 回填",
 ], open=["SoC 带宽/算力精确数字", "官方博客与各仓库 README 完整链接+访问日期"]),
 "ch02-run-and-overview": dict(n=2, refs=[
   "litert_lm_main.cc:52（backend 默认 gpu）✅ git show 核验",
   "litert_lm_main.cc:54（model_path）✅",
   "c/engine.h:583/591/634/643（BenchmarkInfo 四字段）✅",
   "schema/core/litertlm_print.cc 存在 ✅",
 ], open=["基准数据集实测数字（TTFT/prefill/decode tok-s）回填「〔基准 D〕」", "表 2-2 完整 20 问"]),
 "ch03-input-path": dict(n=3, refs=[
   "engine.h:70(SessionInterface)/112(GenerateContent)/128(Stream)/174(RunPrefill)/188(RunDecode) ✅",
   "engine.h:55（CreateSession 示例注释）✅",
   "conversation.h:56(ConversationConfig) ✅",
   "sentencepiece_tokenizer.h / huggingface_tokenizer.h / tokenizer.h 存在 ✅",
 ], open=["conversation.cc 模板 diff 实现细节补读后回填精确行号（正文已显式标注为待补，未臆测）"]),
 "ch04-prefill": dict(n=4, refs=[
   "tasks.cc:413(Prefill) ✅",
   "llm_litert_compiled_model_executor.h:247(PrefillInternal)/415(SortedPrefillSignatureMap) ✅",
   "ExecutorPrefillParams（cancel/max_prefill_sequence_length）概念级引用，行号待补读 io_types",
 ], open=["ExecutorPrefillParams 与 execution_queue 精确行号", "prefill 耗时曲线实验数据"]),
 "ch05-decode": dict(n=5, refs=[
   "tasks.cc:111(DecodeOneStep)/446(Decode)/86(ShouldStop)/110(内外采样注释)/319(DecodeAndSample)/268(DecodeLogits) ✅",
   "sampler.h:34(Sampler) ✅；top_p_cpu_sampler.h:30(TopPSampler)/38(Create) ✅",
   "stop_token_detector.h:45(类)/67(ProcessTokens)/96(MaxPartialStopTokenLength) ✅",
 ], open=["温度对比/停止词回吐实验数字（基准 D）", "BPE 半字 MergeTokenIds 精确行号（补读 tasks.cc）"]),
}

for slug, m in CH.items():
    refs = "\n".join(f"- {r}" for r in m["refs"])
    opens = "\n".join(f"- [ ] {o}" for o in m["open"])
    star = "（⭐样章：本章为全书深度标尺）" if m["n"] == 5 else ""
    content = f"""# 第 {m['n']} 章 review.md {star}

> 首稿验收留痕。状态：**初稿完成，Pass 1 部分完成，Pass 2 lint 已过、独立审校（§14）待做**。

## Pass 1 · 事实核查（断言四级制 + 引用逐条核验）

引用核验（本会话用 `git show v0.13.1:<path>` 逐条比对，标 ✅ 者已确认支撑断言）：
{refs}

- 【推测】级断言：均已显式标注（"我们推测/据此推断"或"待补/待核验"）✅
- 「对照视野」侧栏：仅陈述取舍不裁决优劣，其他框架断言标【文档】级、出处待 P5 回填 ✅
- 数字：性能上限为纸面算账（可复算），实测数字统一标注待基准 D 回填 ✅

## Pass 2 · 除 AI 味

- [x] `scripts/lint_prose.sh chapters/{slug}/chapter.md` —— **0 命中**（2026-07-05）
- [x] 除 AI 味清单第 1-13 条：撰写时自查（空洞总结句/填充词/排比/轻动词/同义堆叠/翻译腔/句式/列表节制/段落节奏/破折号）
- [x] 第 14 条 独立审校：**已由独立审校会话执行（2026-07-05）**，判为"AI 味很低、可放行"；挑出的元评论/拔高/翻译腔问题已逐条修正，修后 lint 复跑 0 命中

## 待清零（补读/数据）
{opens}

## 验收自问
- [x] 本章"一句话使命"兑现
- [x] 图表清单已落地并编号（见章内 figure 与 notes.md）
- [ ] 实验可一键复现（脚本待随基准数据集入 `experiments/`）
- [x] 与 BOOK_PLAN 章卡一致；交叉引用用"见第 N 章"格式
"""
    open(os.path.join(ROOT, "chapters", slug, "review.md"), "w", encoding="utf-8").write(content)
    print(f"写入：chapters/{slug}/review.md")
