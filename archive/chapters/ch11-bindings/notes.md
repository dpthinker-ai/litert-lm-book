<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->
# 第 11 章 · 一套核心，六种语言：C ABI、绑定与工程纪律 — notes（策展）

**一句话使命**：跨语言桥的设计模式 + 生产级代码的工程实践。

## 素材取用（单一事实源在 `chapters/_shared/`）
- [`module-bindings.md`](../_shared/module-bindings.md) — 多语言绑定 (Language Bindings)
- [`module-build-docs.md`](../_shared/module-build-docs.md) — 构建系统、文档与示例

**聚焦**：取 bindings 的 C ABI/各语言封装、build-docs 的测试与构建；#2589/#2613（deinit 时机）作缺陷案例。

## 本章图表（先规划后动笔，CLAUDE.md 第六节）
- [ ] 图 11-1 C ABI 桥与各语言绑定结构
- [ ] 表 11-1 各语言 FFI 机制对照

## 本章实验（脚本入 `experiments/`）
- [ ] 同一 prompt 走 Python 与 C++ 验证行为一致
- [ ] 给 FakeLlmExecutor 写一个新用例

## 补读 / 缺口（写作前须清零）
- （无额外补读；仍须按四级制核验每条引用）

## 待核实清单 / 随手记
- 