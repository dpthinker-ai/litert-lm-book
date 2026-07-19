# 第 6 章审校记录

> 当前记录：2026-07-18，独立扩章审校。审校会话未参与本轮扩写，适用对象为当前第 6 章正文与图 6-1 至图 6-3；源码基线为 LiteRT-LM v0.13.1。

## 审校范围与当前结论

- [x] 语言：逐段检查长句、术语和叙事姿态；章节 lint 为 0 命中。
- [x] `humanizer-cn` 复核：高风险套话、否定式排比、加粗列表骨架、破折号与超 50 汉字单句均为 0 命中；技术条件和证据边界保持不变。
- [x] 数字：复算 28 KiB/token、112/224/448 MiB、0.37/2.94/5.87/22.9 GB/s 与 19% 吞吐降幅。
- [x] 严谨度：把“KV 带宽上界”改为“KV 逻辑扫描量”，并说明它不是 DRAM 实测值或严格总线流量上界。
- [x] 双缓冲：区分 CPU `Duplicate()`、GPU in-place 与 GPU out-of-place；容量案例改为显式条件化计算。
- [x] checkpoint：核对同名 label 覆盖、回退时删除较晚检查点，以及回退不复制或清零 KV 的语义；补充它不保存随机数生成器状态。
- [x] 源码锚点：从 `SessionAdvanced::Clone` 追到资源管理器与 executor，并逐条检查本章引用。
- [x] 图表编号：正文、文件名、include 与 caption 按出现顺序统一为图 6-1、图 6-2、图 6-3；表 6-1 至表 6-5 顺序连续。

本文件不记录整书 build 或 PDF 已通过；两项由主会话最终验收。

## 本轮已纠正的关键事实

- 表 6-1 计算的是按 KV 张量形状得到的逻辑扫描量。缓存、重复加载、布局转换和 out-of-place 输出都会改变实际总线流量。
- `.litertlm` 整文件约 3.4 GiB，包含 10 个模型段；主 decode 模型段 payload 为 2.26 GB。整文件大小不能作为每个 decode step 的权重读取量。
- Gemma 4 E4B 的签名含 `param_tensor[1,1,1,7]`，实际 GPU 路径不能按两块独立 KV 缓冲估算。图 6-2 的 224 MiB 常驻容量只适用于明确写出的 GPU out-of-place 条件。
- `SessionAdvanced::Clone` 注册任务后，由资源管理器建立新 handler。新旧 handler 起初共享同一个 `SharedProcessedContext`；`RuntimeConfig` 与 `RuntimeState` 按值复制，但其中 `rand_gen` 的 `shared_ptr` 仍指向同一随机数生成器。此时 LLM KV 搬运量为 0，音频上下文另行克隆。
- 从已处理序列末尾继续 prefill 不触发复制。较短分支仍有不匹配输入需要写入，且当前步不是共享链最长步数时，资源管理器才调用 executor `CloneContext` 保存原上下文并完成写时分离；decode 截断已处理 token 前采用同类判断。
- compiled executor 只复制活动的 `input_kv_cache_buffers_`。底层按每块缓冲的 `PackedSize()` 完整复制，不按 `current_step` 裁剪有效前缀，也不是把双缓冲的两套 map 全部复制。
- 112 MiB 只适用于活动输入 map 含一套基准模型 K/V、上下文宽度为 4096 且实际进入快照路径的条件；它不是 `Session::Clone` 的固定成本。GPU in-place 与动态形状路径不能直接套用。
- NPU executor 复制 prefill 输入中的 K、V、C cache，并在恢复时再次按完整缓冲复制；缓冲集合与恢复方式不同，不能沿用 112 MiB。
- `--max-num-tokens` 扫描中，只有相同 prompt 的 4096 与 8192 两档可直接比较：decode 从 26.4 降至 21.5 tokens/s，约下降 19%。1024 档的短 prompt 样本和越界失败属于不同输入条件。

## 当前仍存在的实验缺口

- 尚未单独测量 `Session::Clone`、首次写时分离及后续独立上下文切换的时延和峰值内存。
- 没有 NPU 真机推理数据可验证其 K/V/C cache 快照与恢复成本。
- 单缓冲 `RestoreContext` 会跳过活动 map 替换；当前只确认源码分支，尚无受控实验说明其性能与完整运行语义。

## 验证命令

- `bash scripts/lint_prose.sh chapters/ch06-kv-cache/chapter.md`
- `python3 scripts/check_code_references.py --source ../LiteRT-LM-v0.13.1`
- `xmllint --noout chapters/ch06-kv-cache/figs/fig-6-1.svg chapters/ch06-kv-cache/figs/fig-6-2.svg chapters/ch06-kv-cache/figs/fig-6-3.svg`

## 历史记录

- `2026-07-05 · 初稿验收`：旧文件记录了 KV 公式、双缓冲、Clone、Rewind 与图表待办。
- 旧记录把 Clone 与 KV 深拷贝合并为一次操作；v4.8 已按完整调用链取代该结论，当前缺口以上一节为准。
