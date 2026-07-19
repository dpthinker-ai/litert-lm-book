# Summary

[封面](cover.md)

[前言](preface.md)

---

- [第一部分 · 约束、指标与系统概览](parts/part01-foundations.md)
  - [第 1 章 端侧 LLM：约束与总览](chapters/ch01-three-walls/chapter.md)
  - [第 2 章 从运行到架构：benchmark 指标解读与五层概览](chapters/ch02-run-and-overview/chapter.md)

- [第二部分 · 推理流水线](parts/part02-inference-pipeline.md)
  - [第 3 章 输入侧：从 Engine API 到 token 序列](chapters/ch03-input-path/chapter.md)
  - [第 4 章 Prefill：并行处理提示词](chapters/ch04-prefill/chapter.md)
  - [第 5 章 Decode：单步解码循环](chapters/ch05-decode/chapter.md)

- [第三部分 · 运行时优化与异构执行](parts/part03-runtime-optimization.md)
  - [第 6 章 KV cache 与会话状态](chapters/ch06-kv-cache/chapter.md)
  - [第 7 章 模型的形态：量化、.litertlm 格式与 LoRA](chapters/ch07-model-shape/chapter.md)
  - [第 8 章 异构算力：CPU、GPU 与 NPU](chapters/ch08-heterogeneous/chapter.md)
  - [第 9 章 一次前向，多个 token：推测解码与 MTP](chapters/ch09-speculative/chapter.md)

- [第四部分 · 能力扩展与工程集成](parts/part04-capabilities-integration.md)
  - [第 10 章 多模态输入、约束解码与工具调用](chapters/ch10-multimodal/chapter.md)
  - [第 11 章 多语言绑定：C ABI、JNI 与 Embind](chapters/ch11-bindings/chapter.md)

- [尾声 · 实践入口与待验证问题](chapters/epilogue/chapter.md)

---

# 附录

- [附录 A 术语表](appendix/glossary.md)
- [附录 B 代码地图](appendix/code-map.md)
- [附录 C 环境搭建与实验复现](appendix/reproduce.md)
- [附录 D 基准数据集](appendix/benchmark-dataset.md)
- [附录 E 练习提示与参考答案](appendix/exercises-answers.md)
