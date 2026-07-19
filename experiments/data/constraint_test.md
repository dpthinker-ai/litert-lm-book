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
本记录共 12 次生成，开启与关闭约束解码各 6 次。在所测两个工具模式与两个温度条件下，未观察到结构非法的工具调用。样本量和场景覆盖不足以比较两种设置的失败率，也不能据此判断模型在更复杂工具调用中的可靠性。多工具混淆、嵌套 JSON、弱模型和实际执行结果均未覆盖。
