# MTP 自然代码文本吞吐记录与有效成本反算（2026-07-18）

## G 核实（verify signature，主模型 subgraph 3，tensorflow flatbuffer 解析）
    embeddings [1, 4, 2560]，input_pos [4]，per_layer_embeddings [1, 4, 42, 256]
    → G + 1 = 4，G = 3（导出定死，运行时不可调）

## 温度 0、自然代码文本条件下的单次观测
    Mac M5 Pro GPU（SDK Conversation，temperature=0，纯代码模块 prompt）：
        off 58.7 tok/s → on 133.9 tok/s  = 2.28×
    Phone qcom GPU（advanced main，benchmark_prefill_tokens=0，decode 192）：
        off 18.62 → on 37.38 tok/s       = 2.01×
    Phone qcom CPU（同法，fibonacci）：
        off 11.57 → on 12.63 tok/s       = +9%

## 有效成本参数（speedup = E[产出] / (1 + G·c)，G=3，E≤4）

若额外假设每轮都达到最大产出 E=4，可由端到端吞吐比反推出一个有效成本参数：

    Mac GPU：2.28× → c_eff ≈ 0.25
    Phone GPU：2.01× → c_eff ≈ 0.33

这些结果均为单次观测，不提供运行间方差。该参数也不是 drafter 单步成本的独立测量。用同一组端到端数据反解 c，再代回公式，不能证明设备已经达到理论上限。模型段大小不能直接换算执行成本；kernel 启动、KV 管理等分项需要单独 profile 才能归因。

## 记录完整性（2026-09-05 核对）

本文件保存的是 2026-07-18 单次观测的摘要，各条件没有重复测量。它与 `mtp.csv` 中 Mac 的 `false` / `auto` 主基准是不同批次；主基准两组均为关闭行为，本文件另记了自然代码负载的开关观测。

摘要未保存完整 prompt、完整调用参数和原始计时日志；Mac 组也未记录输出 token 数。现有内容可复算吞吐比，不能据此重算原始生成耗时、核对计时起止位置或估计运行间波动。上面的原始摘要数字保持不变，没有补填缺失条件。
