# 约束解码开/关工具调用成功率实测（gpu, gemma-4-e4b, v0.13.1, 2026-07-17）

方法：Python SDK `Engine.create_conversation(tools=[...], enable_constrained_decoding=True/False,
automatic_tool_calling=False, sampler_config=SamplerConfig(temperature=T))`，
声明真实 Python 函数为工具，考察返回的 tool_calls 是否结构合法。

## 场景一：单工具单参数（get_weather(city)），温度 0，各 2 次
- constraint=True：2/2 合法（`get_weather{city: Paris}`）
- constraint=False：2/2 合法

## 场景二：单工具三参数带类型（get_forecast(city, days:int, unit)），各 2 次
- 温度 0：True 2/2、False 2/2，且四次输出完全一致（`{city: Paris, days: 5.0, unit: celsius}`）
- 温度 1.0：True 2/2、False 2/2

## 结论
在 Gemma 4 E4B 上，简单到中等难度的工具调用下，开/关约束解码 8/8 全部结构合法：
模型的函数调用训练已足够强，约束解码在这类场景是保险而非必需。其价值场景在多工具混淆、
嵌套自由 JSON、更弱的模型，或必须保证下游可执行的自动化链路——这些场景本书未覆盖。
注意 MaskLogits 每步对整词表跑一遍位图（第 10 章），简单场景付的是用不上的保单开销。
