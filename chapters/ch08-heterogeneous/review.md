# 第 8 章 review.md

> 首稿验收留痕。状态：**初稿完成，Pass 1 部分完成，Pass 2 lint 已过、独立审校（§14）待做**。

## Pass 1 · 事实核查（断言四级制 + 引用逐条核验）

引用核验（本会话用 `git show v0.13.1:<path>` 逐条比对，✅ 者已确认支撑断言）：
- executor_settings_base.h:34(Backend)/45(CPU)/48(GPU)/54(NPU)/39-51(ARTISAN 各项) ✅
- llm_litert_compiled_model_executor_factory.cc:165(工厂)/168(GetBackend)/170-172(CPU/GPU)/174-175(NPU) ✅
- llm_litert_compiled_model_executor.h:354(gpu_sampler_max_top_k_)/160(InitializeSampler) ✅
- threadpool.h:51(ThreadPool)/57(ctor max_num_threads) ✅；cpu_affinity_utils.h:27(性能核)/31(设亲和性) ✅
- llm_litert_npu_compiled_model_executor.h:51(类)/65-77(embedder 延迟统计) ✅

- 贴出的 switch 代码片段：与 v0.13.1 结构一致，用 `...` 省略，已删去无对应讲解的编号标记 ✅
- **NPU 无真机**：全程明写"基于代码分析"，只陈述代码可佐证的结构差异（独立 embedder 子模型），不对 NPU 性能/取舍下无证据结论 ✅
- `#2281`（换后端输出变）：现象按【文档】级引用；成因解释基于浮点不可逐比特一致 + 自回归放大（常识级推理），明标"未臆测 issue 内部" ✅
- 片上采样省多少、线程数扫描、cpu/gpu 对比 标注待基准 D ✅

## Pass 2 · 除 AI 味

- [x] `scripts/lint_prose.sh chapters/ch08-heterogeneous/chapter.md` —— 0 命中（2026-07-05）
- [x] 除 AI 味清单第 1-13 条：撰写时自查
- [x] 第 14 条 独立审校：**已由独立审校会话执行（2026-07-05）**，判为"AI 味很淡、可放行"；升华式收尾/自夸形容词/括号内同义重述等问题已逐条修正，修后 lint 复跑 0 命中

## 待清零（补读/数据）
- [ ] 实测（cpu vs gpu、cpu_thread_count 扫描、片上采样省的拷贝量）待基准 D
- [ ] 图 8-2（片上/回传数据路径）、表 8-1（CPU/GPU/NPU 权衡）待补（本轮出签名图 8-1）

## 验收自问
- [x] 使命兑现（三后端脾气、工厂分派、线程功课；结清第 17、18 问）
- [x] 图 8-1 已落地并编号
- [ ] 实验可复现（脚本随基准 D）
- [x] 与 BOOK_PLAN 章卡一致；交叉引用规范；承接第 1 章异构墙、第 5 章片上采样，预告第 9 章
