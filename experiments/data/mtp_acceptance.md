# MTP 接受率实测实录（gpu, gemma-4-e4b, v0.13.1, 2026-07-17）

方法：Python SDK + `set_min_log_severity(VERBOSE)`，drafter 析构时打印计数器
（`llm_litert_mtp_drafter.cc:166-169`）。`enable_speculative_decoding=True`，
run_prefill + run_decode_async 流式生成至自然结束。无需改码重编。

脚本要点：
    engine = litert_lm.Engine(model_path=..., backend=litert_lm.Backend.GPU(),
                              enable_speculative_decoding=True)
    session.run_prefill([prompt]); list(session.run_decode_async())

## 创造性文本（"Write a 100-word story about a robot learning to paint."）
    Num drafted tokens: 213
    Num verified tokens: 56
    Success rate: 0.262911        ← 聚合接受比例 r ≈ 26%

## 套路性文本（"Write a Python function that computes the fibonacci sequence, then explain how it works."）
    Num drafted tokens: 3069
    Num verified tokens: 3054
    Success rate: 0.995112        ← 聚合接受比例 r ≈ 99.5%

## 解读
- 日志的 `Success rate` 是 `verified / drafted`。设每轮草拟 G 个 token、接受前缀长度为 K，则该值是聚合比例 \\(r=E[K]/G\\)，不是各位置相互独立时的逐位命中概率。
- 本模型 G=3，因此两组样本对应的期望产出分别为 \\(1+3r\\)：创造性文本约 1.79 token/轮，代码文本约 3.99 token/轮。端到端加速还取决于 drafter 与 verify 的实际成本，不能只由 r 推出。
- 两个提示词记录到不同的聚合接受比例，但样本不足以估计输入内容与该比例的一般关系。benchmark 使用另一种合成输入，本实验没有测到该输入的 r，不能据此直接解释 benchmark 的开关结果。
