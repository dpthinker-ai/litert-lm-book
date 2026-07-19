# MTP 自然文本对照（真机 P0210，2026-07-18）

背景：benchmark 模式（--benchmark_prefill_tokens=N）的负载是「真实 prompt 分词后 ids.resize(N)」
——用 pad（token 0）填满到目标长度（runtime/core/session_utils.cc:68-73）。
这不是自然文本。本次端到端结果显示 MTP 在该负载下为负收益；运行时没有同时记录聚合接受比例，不能把结果只归因于接受行为。

用 benchmark_prefill_tokens=0（正常文本负载）+ --benchmark_decode_tokens 对比：

## gpu，input_prompt = "Write a Python function that computes the fibonacci sequence, then explain it."，decode 128
    mtp=false   Decode Speed: 16.00 tokens/sec.
    mtp=true    Decode Speed: 32.01 tokens/sec.    ← 2.0×

## cpu，同上，decode 64
    mtp=false   Decode Speed: 11.57 tokens/sec.
    mtp=true    Decode Speed: 12.63 tokens/sec.    ← +9%

## 对照：pad 填充负载（--benchmark_prefill_tokens=1024）
    cpu: 10.0 → 2.8（降至约 28%）；gpu: 18.0 → 12.6（降低约 30%）

## 结论
在这台设备上，pad 合成负载为负收益；自然代码文本的单次观测中，gpu 吞吐约为关闭模式的 2 倍，cpu 高约 9%。自然文本每个条件只运行 1 次，且 cpu 与 gpu 的 decode 长度不同。该对照只说明结果会随输入与后端变化；本实验没有分项测量 drafter、verify 成本，也没有记录两种负载的聚合接受比例，不能进一步确定各因素的贡献。官方最高约 3 倍的数据来自不同条件，不能由本记录验证或否定。
