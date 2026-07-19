# Summary

[封面](cover.md)

[前言](preface.md)

---

- [第一部分 · 约束、指标与系统概览](parts/part01-foundations.md)
  - [第 1 章 端侧 LLM：三类物理约束与运行时版图](chapters/ch01-three-walls/chapter.md)
  - [第 2 章 从运行到架构：benchmark、Roofline 与五层视图](chapters/ch02-run-and-overview/chapter.md)

- [第二部分 · 推理流水线](parts/part02-inference-pipeline.md)
  - [第 3 章 输入侧：从 Engine API 到 token 序列](chapters/ch03-input-path/chapter.md)
  - [第 4 章 Prefill：并行处理提示词](chapters/ch04-prefill/chapter.md)
  - [第 5 章 Decode：单步解码循环](chapters/ch05-decode/chapter.md)

- [第三部分 · 模型加载、异构执行与投机解码](parts/part03-runtime-optimization.md)
  - [第 6 章 KV cache：容量、带宽与会话生命周期](chapters/ch06-kv-cache/chapter.md)
  - [第 7 章 模型文件与权重：量化、容器格式与 LoRA](chapters/ch07-model-shape/chapter.md)
  - [第 8 章 异构算力：CPU、GPU 与 NPU](chapters/ch08-heterogeneous/chapter.md)
  - [第 9 章 一次前向，多个 token：投机解码与 MTP](chapters/ch09-speculative/chapter.md)

- [第四部分 · 多模态、工具调用与跨语言集成](parts/part04-capabilities-integration.md)
  - [第 10 章 多模态与工具调用：视觉/音频编码、约束解码与函数调用](chapters/ch10-multimodal/chapter.md)
  - [第 11 章 多语言绑定：C ABI、JNI 与 Embind](chapters/ch11-bindings/chapter.md)

- [尾声 · 实践入口与待验证问题](chapters/epilogue/chapter.md)

---

# 附录

- [附录 A 术语表](appendix/glossary.md)
- [附录 B 代码地图](appendix/code-map.md)
- [附录 C 环境搭建与实验复现](appendix/reproduce.md)
- [附录 D 基准数据集](appendix/benchmark-dataset.md)
- [附录 E 练习提示与参考答案](appendix/exercises-answers.md)
