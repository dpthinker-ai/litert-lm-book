<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->
# 第 8 章 · 异构算力：CPU、GPU 与 NPU — notes（策展）

**一句话使命**：三类后端的实现差异、运行时分派、buffer 交接条件与可复现诊断方法。

## 素材取用（单一事实源在 `chapters/_shared/`）
- [`module-executor-llm.md`](../_shared/module-executor-llm.md) — LLM 执行器 (Executor)
- [`module-framework.md`](../_shared/module-framework.md) — 并发框架 (Framework)

**聚焦**：取 executor-llm 的 Backend 分派、设备侧采样、buffer/事件、NPU 多子图，以及 framework 的线程池与 CPU 亲和性；NPU 尚无成功推理，全程标明源码分析与真机探测的边界。

## 本章图表（先规划后动笔，CLAUDE.md 第六节）
- [x] 图 8-1 后端工厂分派
- [x] 图 8-2 设备侧采样与 host 采样的数据路径对比
- [x] 图 8-3 Backend 配置到编译选项与实际 buffer
- [x] 图 8-4 buffer 直接交接的条件与退化路径
- [x] 表 8-1 buffer 交接条件与退化路径
- [x] 表 8-2 CPU/GPU/NPU 特性权衡

## 本章实验（脚本入 `experiments/`）
- [x] CPU 与 GPU 同机对比
- [x] CPU thread count 扫描（LiteRT-LM#2505）
- [ ] GPU sampler / CPU sampler 受控 A/B 与同步、复制分项
- [ ] CPU/GPU 首个输出分叉 step 的全量 logits 与逐层对齐

## 补读 / 缺口（写作前须清零）
- （无额外补读；仍须按四级制核验每条引用）

## 待核实清单 / 随手记

- `Duplicate()` 只支持“当前调用点无显式复制”的结论；端到端 zero-copy 仍需 delegate/driver profile。
- NPU 两台设备均未完成 prefill/decode，`LatencyStats` 结构不能替代真机数据。
