# 第四部分 · 多模态、工具调用与跨语言集成

第三部分覆盖了纯文本推理的核心运行时——从 KV cache 到投机解码。本部分把视野扩展到文本之外：模型如何理解图像和音频，如何保证输出符合结构化语法，以及 C++ 运行时如何被 Python、Kotlin、Swift 和 Web 调用。

第 10 章处理多模态输入与结构化输出。前半部分追踪图像和音频从 `Message` 对象到主干模型 embedding 序列的完整路径：patchify 切块、变分辨率视觉 signature 选择、视觉 token budget 的作用、log-mel 频谱的分块编码流程，以及多图在 token 维拼接（不是 batch）的语义差异。每层的数据表示、shape 约束和所有权变化，都在一张六层边界图中逐级标注。后半部分分析约束解码：llguidance 如何按文法状态生成 token 位图、`MaskLogits` 如何在每个 decode step 屏蔽非法候选，以及 tools-derived 文法究竟能保证什么——它能编码函数名和参数结构，但不负责权限校验和外部执行。一个完整的 `set_device_mode` 端到端案例展示了从工具声明到宿主执行的五层信任边界。

第 11 章讨论多语言绑定。LiteRT-LM 的核心是 C++ 运行时，但它同时向 Python、Kotlin、Swift 和 Web 暴露 API。这三类原生边界——C ABI（Python/Swift）、JNI（Kotlin）和 Embind（Web）——在句柄形态、字符串所有权、流式回调生命周期和资源释放时机上各有不同的契约。Swift 的两个 issue（Conversation 与 Engine 的显式释放问题）在这里被拆开分析，说明看似相似的缺陷实际上涉及不同的对象、故障条件和修复路径。Swift actor 与 Kotlin `synchronized` 对并发隔离的不同处理，以及脱离真实模型的 `FakeLlmExecutor` 测试替身，也为工程实践提供了可直接对照的参考。

