# 第三部分 · 运行时优化与异构执行

> 本部分讨论影响内存占用、初始化时间和生成吞吐的运行时设计。四章分别分析会话状态、模型资源、硬件后端与推测解码，并明确各项优化成立的条件。

- [第 6 章　KV cache 与会话状态](../chapters/ch06-kv-cache/chapter.md)：缓存容量、双缓冲、会话克隆、检查点与回退。
- [第 7 章　模型的形态：量化、.litertlm 格式与 LoRA](../chapters/ch07-model-shape/chapter.md)：模型容器、低比特表示、文件映射、编译缓存与适配器。
- [第 8 章　异构算力：CPU、GPU 与 NPU](../chapters/ch08-heterogeneous/chapter.md)：后端选择、线程配置、设备侧采样与缓冲共享。
- [第 9 章　一次前向，多个 token：推测解码与 MTP](../chapters/ch09-speculative/chapter.md)：草拟、验证、接受比例与端到端收益条件。
