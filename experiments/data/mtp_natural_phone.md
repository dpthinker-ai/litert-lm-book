# MTP 自然文本对照（真机 P0210，2026-07-18）

背景：benchmark 模式（--benchmark_prefill_tokens=N）的负载是「真实 prompt 分词后 ids.resize(N)」
——用 pad（token 0）填满到目标长度（runtime/core/session_utils.cc:68-73）。
这不是自然文本，drafter 接受率天然塌掉，MTP 在此负载下必亏。

用 benchmark_prefill_tokens=0（正常文本负载）+ --benchmark_decode_tokens 对比：

## gpu，input_prompt = "Write a Python function that computes the fibonacci sequence, then explain it."，decode 128
    mtp=false   Decode Speed: 16.00 tokens/sec.
    mtp=true    Decode Speed: 32.01 tokens/sec.    ← 2.0×

## cpu，同上，decode 64
    mtp=false   Decode Speed: 11.57 tokens/sec.
    mtp=true    Decode Speed: 12.63 tokens/sec.    ← +9%

## 对照：pad 填充负载（--benchmark_prefill_tokens=1024）
    cpu: 10.0 → 2.8（慢 3.6 倍）；gpu: 18.0 → 12.6（慢 30%）

## 结论
MTP 的收益 = f(负载可预测性 α, 硬件 drafter 相对成本)。
pad 负载最坏（双端负收益）；自然代码文本 gpu 翻倍、cpu 微赚（手机 CPU 上 drafter 相对更贵，吃掉大部分 α 收益）。
模型与 Google 宣称均无矛盾：3 倍口径在高 α 文体 + 官方硬件上成立。
