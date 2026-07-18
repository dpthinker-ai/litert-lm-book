# MTP 加速比天花板实测（2026-07-18）

## G 核实（verify signature，主模型 subgraph 3，tensorflow flatbuffer 解析）
    embeddings [1, 4, 2560]，input_pos [4]，per_layer_embeddings [1, 4, 42, 256]
    → G + 1 = 4，G = 3（导出定死，运行时不可调）

## 最优手法（温度 0、纯代码长生成、自然文本）下的实测
    Mac M5 Pro GPU（SDK Conversation，temperature=0，纯代码模块 prompt）：
        off 58.7 tok/s → on 133.9 tok/s  = 2.28×
    Phone qcom GPU（advanced main，benchmark_prefill_tokens=0，decode 192）：
        off 18.62 → on 37.38 tok/s       = 2.01×
    Phone qcom CPU（同法，fibonacci）：
        off 11.57 → on 12.63 tok/s       = +9%

## 天花板分析（speedup = E[产出] / (1 + G·c)，G=3，E≤4）
    Mac GPU：2.28× → 反解 c_draft/c_base ≈ 0.24 → 天花板 4/1.73 ≈ 2.31×（已达）
    Phone GPU：2.01× → 反解 c ≈ 0.32 → 天花板 4/1.97 ≈ 2.03×（已达）
    Google「约 3 倍」→ 需 c ≈ 0.11（推测为 Pixel Tensor ARTISAN 手写路径或服务级 GPU）

    drafter 仅 45 MB（主模型 2.26 GB 的 2%），但 c 是 0.24-0.32 而非 0.02：
    小模型的每步固定开销（kernel 启动、KV 管理）在这些后端上主导成本。
