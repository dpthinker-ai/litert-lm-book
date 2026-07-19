# 第 9 章审校记录

> 当前记录：2026-07-18，v4.9 独立扩章审校。适用对象为当前第 9 章正文与图 9-1 至图 9-3，源码基线为 LiteRT-LM v0.13.1。

## 审校范围与当前结论

- [x] 语言：统一 MTP、drafter、verify、接受比例与成本口径，删除无条件加速结论。
- [x] `humanizer-cn` 复核：高风险套话、否定式排比、加粗列表骨架、破折号与超 50 汉字单句均为 0 命中。
- [x] 禁词与术语：检查禁词、退役叙事标签、概率术语和 SVG 文本。
- [x] 严谨度：核对草拟、验证、接受循环、首轮与稳态、采样分布、CLI 设置链路及实验条件。
- [x] 叙事姿态：删除戏剧化收益、文体定性和把单次观测写成稳定上限的表述。
- [x] 源码锚点语义：检查引用行是否支撑 token 数、状态推进、signature 形状和自动启用行为。

本文件不记录整书 build 或 PDF 已通过；两项由主会话最终验收。

## v4.9 独立扩章审校

本轮由未参与第 9 章扩写的独立会话执行，范围为“多 token 返回后的停止检测与回退”一节、图 9-3，以及受其影响的图表编号。

- [x] 多 token 返回：核对 `DecodeOneStep::Run` 的等长检查、逐位置循环、停止检测与 BPE 合并顺序（`runtime/core/tasks.cc:147-179`）。
- [x] 停止与回退：核对 `sequence_length - step` 的回退量，并补明末位命中不调用 `SetCurrentStep`（`runtime/core/tasks.cc:223-233`）。
- [x] 执行器状态：核对 compiled executor 只更新逻辑 `current_step`，不清除 KV cache 缓冲（`runtime/executor/llm_litert_compiled_model_executor.cc:1452-1479`）。
- [x] 流式回调：核对一次 `Run` 后至多一次 `kProcessing` 更新，且 `any_updates=false` 时不回调（`runtime/core/tasks.cc:523-566`）。
- [x] 数值上限：明确 `num_decode_steps` 是 `current_step` 增量；稳态最多越界 G、首轮最多越界 G+1 的结论标为带前提的任务层推断（`runtime/core/tasks.cc:86-105`、`:569-572`）。
- [x] 测试边界：确认现有 `FakeLlmExecutor` 每候选每次只返回 1 个 token，正文改为“新增测试 executor”，不声称现有 fake 已覆盖该场景。
- [x] 图表顺序：正文出现顺序为图 9-1、图 9-2、图 9-3，以及表 9-1、表 9-2、表 9-3；include、文件名与图注一致。
- [x] 图 9-3：补充末位不回退说明，并将图中 `AllDone` 放回 `MergeTokenIds` 之后，符合源码执行顺序。

验证结果：

- [x] `bash scripts/lint_prose.sh chapters/ch09-speculative/chapter.md`：0 命中。
- [x] `python3 scripts/check_code_references.py`：623 个唯一锚点通过，源码提交为 `a0afb5a56acd`。
- [x] `xmllint --noout`：图 9-1 至图 9-3 均通过。
- [x] 中文单句长度复查：未发现超过 50 个汉字的句子。
- [x] `mdbook build`：HTML 构建成功，0 WARN。

## 本轮已纠正的关键事实

- 当前 MTP 路径对 drafter 与 verifier 都采用贪心 token 比较，不含保持随机采样分布不变的概率接受；temperature 或 top-p 下不能声称与普通路径分布一致。
- 稳态每轮接受 K 个草稿时返回 K+1 个 token；prefill 后第一次 `Decode()` 还包含一次普通 decode，因此返回 K+2 个 token，并执行两次基础模型前向。
- 日志的聚合比例 r 是 `ΣK/(RG)`；理论曲线中的 p 是逐位条件匹配概率。两者不能直接互换，旧盈亏平衡计算据此修正。
- v0.13.1 的 CLI `auto` 映射为 `None`，随后保留 C++ 默认关闭；模型能力查询没有自动接入启用链路。
- 合成负载、自然文本接受计数和自然代码吞吐来自不同运行。单次 2.28 倍或 2.01 倍观测不能证明达到理论上限，也不能由 drafter 文件大小反推单步成本。

## 当前仍存在的实验缺口

- 需要在同一次运行中同时记录接受比例、drafter/verify 分项成本与端到端吞吐，并对自然代码条件重复采样。
- 主 benchmark 的 pad 合成负载没有接受计数；两个自然文本样本不足以建立内容与接受比例的一般关系。
- 当前日志只提供 drafter 生命周期聚合值，缺少逐轮与分阶段遥测。

## 历史记录

- `2026-07-05 · 初稿验收`：旧文件记录了 MTP 实现补读、接受率与实验计划。
- 旧记录中的图表待办和未复现判断已被 v4.8 的实测边界取代；当前缺口以上一节为准。
