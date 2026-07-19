<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->
# 第 6 章 · KV cache 与会话状态 — notes（策展）

**一句话使命**：计算 KV cache 的容量与逻辑扫描量，并说明双缓冲、会话克隆和检查点回退的实现边界。

## 素材取用（单一事实源在 `chapters/_shared/`）
- [`module-executor-llm.md`](../_shared/module-executor-llm.md) — LLM 执行器 (Executor)
- [`module-core-pipeline.md`](../_shared/module-core-pipeline.md) — 核心调度 (Core Pipeline)

**聚焦**：核对 executor 的 KV 缓冲路径，以及 core 的 Clone、SaveCheckpoint、RewindToCheckpoint 调用链；容量与吞吐估算沿用第 1、2 章的口径。

## 本章图表
- [x] 图 6-1 GPU out-of-place KV 双缓冲与指针交换
- [x] 图 6-2 条件化双缓冲与写时分离容量案例
- [x] 图 6-3 Clone 写时分离与 channel 过滤时序
- [x] 表 6-1 KV 逻辑扫描量
- [x] 表 6-2 `--max-num-tokens` 扫描
- [x] 表 6-3 公共前缀分支容量变化
- [x] 表 6-4 Clone、后端快照与 KV 接口边界
- [x] 表 6-5 checkpoint map 语义

## 本章实验（脚本入 `experiments/`）
- [x] `--max-num-tokens` 扫描 decode 吞吐与失败边界（`experiments/data/max_tokens_sweep.csv`）
- [ ] Clone 后分叉对话验证独立性
- [ ] get_token_count 观察多轮增长

## 补读 / 缺口
- GPU in-place `RestoreContext` 跳过活动 map 替换；尚缺受控实验验证多分支恢复语义。
- NPU K/V/C cache 的快照与恢复只有源码核对，没有真机数据。
- 双缓冲容量案例是显式条件化计算，不是 Gemma 4 E4B 实际 GPU 路径的内存实测。

## 待核实清单 / 随手记
- 后续若补 Clone 基准，需拆分 Clone API、首次写时分离和独立上下文切换三个采样点。
