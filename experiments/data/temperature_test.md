# 温度/种子实验实录（cpu, gemma-4-e4b, 2026-07-05）
# 命令: litert-lm run gemma-4-e4b --backend cpu --prompt "Write one sentence about the ocean." \
#        --temperature T --seed S [--top-k 64 --top-p 0.95] --cache disk
T=0 seed=42 (run1): The vast, mysterious ocean covers over seventy percent of the Earth's surface, teeming with diverse life and holding immense power.
T=0 seed=42 (run2): 与 run1 逐字一致（确定性复现）
T=1.0 默认k/p seed=1: The ocean is a vast, mysterious expanse covering more than seventy percent of the Earth's surface.
T=1.0 默认k/p seed=2: 与 seed=1 逐字一致（分布尖锐，不同种子命中同一高概率序列）
T=1.0 k=64 p=0.95 seed=7: The vast, restless ocean remains the Earth's mysterious cradle of life, shifting between tranquil serenity and powerful turbulence.
# 结论：温度 0 确定可复现；温度 1.0 输出随种子可变，但分布尖锐时多个种子产出相同序列——
# "开采样"不等于"每次必不同"。
