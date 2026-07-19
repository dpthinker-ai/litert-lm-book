# 第二部分 · 推理流水线

> 本部分按实际调用顺序分析一次生成请求。第 3 章处理输入与会话，第 4 章分析 prefill，第 5 章分析逐 token 的 decode step。

- [第 3 章　输入侧：从 Engine API 到 token 序列](../chapters/ch03-input-path/chapter.md)：消息、模板、tokenizer、embedding 与会话状态。
- [第 4 章　Prefill：并行处理提示词](../chapters/ch04-prefill/chapter.md)：固定 signature、动态分块、异步提交与取消边界。
- [第 5 章　Decode：单步解码循环](../chapters/ch05-decode/chapter.md)：logits、采样、停止检测、回调与逐步计时。
