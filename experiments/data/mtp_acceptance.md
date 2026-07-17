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
    Success rate: 0.262911        ← α ≈ 26%

## 套路性文本（"Write a Python function that computes the fibonacci sequence, then explain how it works."）
    Num drafted tokens: 3069
    Num verified tokens: 3054
    Success rate: 0.995112        ← α ≈ 99.5%

## 解读
- 代码类高可预测文本 α≈0.995 → 代入 speedup ≈ (1+α+α²+α³)/1.45 ≈ 2.7×，与官方「约 3 倍」口径同区。
- 创造性文本 α≈0.26 → speedup ≈ 0.93，恰好压在盈亏平衡（α*≈0.25）上方边缘，解释本书 benchmark 合成负载「开关无差异」的实测。
- 接受率不是模型常数，是文体函数。
