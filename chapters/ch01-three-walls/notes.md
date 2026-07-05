<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->
# 第 1 章 · 端侧 LLM：三堵墙与一张版图 — notes（策展）

**一句话使命**：读者能亲手算出「4B 模型在手机上的理论 decode 上限」这笔账。

## 素材取用（单一事实源在 `chapters/_shared/`）
- [`module-concepts.md`](../_shared/module-concepts.md) — 核心概念与性能优化 (Core Concepts & Optimizations)

**聚焦**：以背景与算账为主，代码素材少。用 concepts 模块的量化/后端/带宽相关概念；SoC 参数须另查官方来源（【文档】级）。

## 本章图表（先规划后动笔，CLAUDE.md 第六节）
- [ ] 图 1-1 三堵墙与模型需求对照
- [ ] 表 1-1 端侧运行时版图对比

## 本章实验（脚本入 `experiments/`）
- [x] 纸面算账（公式与代入过程完整给出，第 6 章实测对账）
  - 四项内存预算（权重/KV cache/激活/系统）代入 8 GiB 手机，给出"放得下/顶穿"判定
  - KV cache 字节公式 2·L·n_kv·d_head·T·b，代 L32/nkv8/dhead128/FP16 → 128 KiB/token，4096→0.5 GiB
  - 带宽墙含 KV cache 的每 token 字节账：25 → 19.8 tok/s，对照附录 D（256→4096，24.8→20.7）
  - 功耗墙能耗账：~20 pJ/byte × 2 GiB ≈ 0.04 J/token，15 Wh 电池上限 ~1.35e6 token（理想值）

## 补读 / 缺口（写作前须清零）
- [ ] 收集 2-3 款典型 SoC 的带宽/算力参数（官方来源）

## 待核实清单 / 随手记
- 