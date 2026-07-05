# 第 11 章 review.md

> 首稿验收留痕。状态：**初稿完成，Pass 1 部分完成，Pass 2 lint 已过、独立审校（§14）待做**。

## Pass 1 · 事实核查（断言四级制 + 引用逐条核验）

引用核验（本会话用 `git show v0.13.1:<path>` 逐条比对，✅ 者已确认支撑断言）：
- c/engine.h:41(LiteRtLmEngine)/44(LiteRtLmSession)/380(engine_create)/386(engine_delete)/396(create_session)/403(session_delete)/421(run_prefill) ✅
- python/litert_lm/_ffi.py:24(c_string_p UTF-8)/36(LiteRtLmSamplerParams ctypes.Structure) ✅
- kotlin/.../LiteRtLmJni.kt:19(object)/52(external fun nativeCreateEngine) ✅
- swift/Engine.swift:17(import CLiteRTLM)/28(public actor Engine)/57(initialize) ✅
- fake_llm_executor.h:37(FakeLlmExecutor : LlmExecutor，脚本化 token) ✅

- `#2589`/`#2613`（Swift close 生命周期）：**open issue**，作缺陷案例研究、【文档】级引用，正文明标"不宣称已修复" ✅
- WASM/双构建系统：概述级，不展开、指向附录 C ✅
- Python 与 C++ 行为一致：作为设计承诺陈述，实测标注待环境 ✅

## Pass 2 · 除 AI 味

- [x] `scripts/lint_prose.sh chapters/ch11-bindings/chapter.md` —— 0 命中（2026-07-05）
- [x] 除 AI 味清单第 1-13 条：撰写时自查（收尾避免自夸；deinit 案例落到具体机制而非泛谈）
- [x] 第 14 条 独立审校：**已由独立审校会话执行（2026-07-05）**，判为"AI 味很淡、可放行"；升华式收尾格言/自我总结拔高/长句等问题已逐条修正，修后 lint 复跑 0 命中

## 待清零（补读/数据）
- [ ] 实测（Python/C++ 行为一致、给 FakeLlmExecutor 写新用例）待环境/基准 D
- [ ] 表 11-1（各语言 FFI 机制对照）待补（本轮出签名图 11-1）

## 验收自问
- [x] 使命兑现（C ABI 通用桥 + 各语言 FFI + 生命周期坑 + 可测/可构建）
- [x] 图 11-1 已落地并编号
- [ ] 实验可复现
- [x] 与 BOOK_PLAN 章卡一致；承接第 3 章 Engine/Session、第 5 章 LlmExecutor 接口、第 2 章接口隔离；收束第四部
