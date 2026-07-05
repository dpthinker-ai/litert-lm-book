# 第 8 章 异构算力：CPU、GPU 与 NPU

> 使命：讲清一部手机上三类计算单元各自的脾气，LiteRT-LM 如何用一个工厂把它们藏在同一套接口后面，以及一个让很多人困惑的现象——为什么换个后端，连模型的输出都会变。

第 1 章说过第三堵墙有两半：功耗和异构。这一章处理异构这半。手机上不止一个能算的地方，CPU、GPU、NPU 各有各的强项和麻烦。一个端侧运行时既要用得上它们，又要对上层藏起它们的差异。先从一个让人挠头的现象说起。

## 同一个模型，三种脾气

第 2 章"二十个问题"的第 17 问，来自一个真实的上游困惑（`LiteRT-LM#2281`）：同一个 `.litertlm` 文件，`--backend=cpu` 和 `--backend=gpu` 跑出来，不只是速度不同，连**输出的文字都可能不一样**。这反直觉——不是同一个模型、同一个提示词吗？

先接受这个事实，本章最后一节解释它的成因。现在只需记住：后端不是"同一个计算的不同速度"，它们是**不同的计算路径**。理解了这一点，"可插拔后端"这个设计的分量才显出来——它不是锦上添花，是端侧必须面对的现实。

## 一个工厂，按 Backend 分派

LiteRT-LM 用一个枚举把后端列全（`Backend`，`runtime/executor/executor_settings_base.h:34 @ v0.13.1`）：除了 `CPU`、`GPU`、`NPU`（`:45`、`:48`、`:54`），还有几个带 `ARTISAN` 后缀的（`CPU_ARTISAN`、`GPU_ARTISAN`、`GOOGLE_TENSOR_ARTISAN`）。这个后缀值得解释一句：`ARTISAN` 指手写算子的路径，不带后缀的走 LiteRT 编译路径——同一类硬件，可以有两种实现路子。

选哪个实现，交给一个工厂函数（`CreateLlmLiteRtCompiledModelExecutor`，`runtime/executor/llm_litert_compiled_model_executor_factory.cc:165 @ v0.13.1`）。它的逻辑一眼能看完：读出 `GetBackend()`（`:168`），CPU 和 GPU 走同一条创建路径（`:170`–`:172`），NPU 走另一条（`:174`–`:175`）。

```cpp
Backend backend = executor_settings.GetBackend();
switch (backend) {
  case Backend::CPU:
  case Backend::GPU:
    return CreateCpuOrGpuLlmLiteRtCompiledModelExecutor(...);
  case Backend::NPU:
    return CreateNpuLlmLiteRtCompiledModelExecutor(...);
  // ...
}
```

上层拿到的都是同一个 `LlmExecutor` 抽象（第 5 章那个接口），完全不知道底下是 CPU 还是 NPU。这就是"可插拔后端"落到代码里的样子：一个枚举、一个工厂、一个共同接口。加一个新后端，就是加一个 case，上层一行不改。

<figure>
{{#include figs/fig-8-1.svg}}
<figcaption>图 8-1　后端工厂分派：Backend 枚举经工厂函数分成 CPU/GPU 一路、NPU 一路，都产出同一个 LlmExecutor 抽象。上层对后端差异无感知。</figcaption>
</figure>

## CPU：随时可用，但要做线程功课

CPU 的好处是随时都在、什么算子都能跑。它的挑战是第三堵墙的另一半——功耗与调度。

手机 CPU 是大小核混合的：几个高性能核，几个高能效核。推理这种重活，跑在性能核上才快；但如果不管不顾，系统调度器可能把线程挪到能效核上，速度就掉了。LiteRT-LM 为此提供了 CPU 亲和性工具（`runtime/engine/cpu_affinity_utils.h @ v0.13.1`），能查出性能核的编号（`:27`），把推理线程绑上去（`:31`），告诉调度器"尽量让这些线程待在这些核上"。

线程本身由一个线程池管（`ThreadPool`，`runtime/framework/threadpool.h:51 @ v0.13.1`），创建时指定最多用多少线程（`:57`）。用几个线程是个权衡：多了未必快（内存带宽是瓶颈，第 1 章），还更耗电、更烫。所以 CLI 开放了 `cpu_thread_count` 这个旋钮让人手调（对应上游需求 `LiteRT-LM#2505`）。这些都是"可持续速度"的功课——不是把资源开满，而是让线程数匹配内存带宽这个真正的上限。

## GPU：并行强，还能就地采样

GPU 的强项是大规模并行，适合矩阵运算这种活。它在端侧的一个精巧优化，第 5 章已经埋过引子——**片上采样**。

回忆第 5 章的两条采样路径。内部采样之所以快，是因为采样这一步可以直接在 GPU 上做，不用把 logits 搬回 CPU。这个"省一次搬运"值多少？decode 每一步都会产出一整组 logits——词表有多大，这组数就有多大，动辄几十万个。如果采样在 CPU 做，这几十万个数每步都要从 GPU 拷回 CPU；在 GPU 上就地采样，这次拷贝就省了。代码里能看到它就地做 top-k 的痕迹（`gpu_sampler_max_top_k_`，`runtime/executor/llm_litert_compiled_model_executor.h:354 @ v0.13.1`，由 `InitializeSampler` 配置，`:160`）。

这正是第 18 问的答案。decode 是整个生成里最频繁的操作，每一步省掉这一次数据搬运，累积起来不小。（这一项很难从整机数字中单独剥离，本书未单测；两个后端的整体差距见附录 D——本书基准上 gpu 的 decode 约为 cpu 的 2 倍、prefill 约 3.9 倍〔基准 D〕。）

## NPU：能效最高，接口最封闭

NPU 是三者里能效比最高的——同样的活，它最省电、最不发热，这在端侧是硬通货。代价是它最封闭：接口是厂商的（比如高通的 QNN），能跑的算子有限，灵活性最低。

本节的内容基于对代码的阅读，没有真机验证（作者手上是一台 Mac，跑不了手机 NPU），所以只讲代码能佐证的部分。NPU 走的是工厂里那条独立路径（`CreateNpuLlmLiteRtCompiledModelExecutor`），产出一个专门的执行器（`LlmLiteRtNpuCompiledModelExecutor`，`runtime/executor/llm_litert_npu_compiled_model_executor.h:51 @ v0.13.1`）。从它的延迟统计字段能看出一个结构上的差异：NPU 路径里有一个独立的 **embedder 子模型**——代码专门记录了 prefill 和 decode 阶段 embedder 的推理延迟（`:65`–`:77`）。这说明 NPU 上"把 token 变成 embedding"这一步是单独一个模型在跑，而不像 CPU/GPU 那样融在主模型里。为什么这么切，涉及 NPU 算子约束下的工程取舍，没有真机不宜妄下结论，此处只把代码能确证的结构差异记录在案。

## 为什么换后端连输出都变

回到开头的困惑（第 17 问，`LiteRT-LM#2281`）。现在有了前面的铺垫，可以给一个合理的解释。

三个后端是三条不同的计算路径：算子的实现不同，数值精度的处理不同（第 7 章那条激活精度谱系在不同后端上落点也不同），量化权重的反量化方式也可能有细微差别。这些差别单看每一步都极小，小到肉眼看不见。但它们会累积进 logits——那组决定"下一个字是什么"的分数。

关键在最后一步：采样。当两个候选 token 的分数本就接近时，后端间那点数值微差，足以让排序翻个个儿，选出不同的 token。一旦某一步选了不同的字，后面整段就顺着岔开了。所以换后端输出会变，不是谁算错了，而是浮点计算在不同硬件路径上本就不可能逐比特一致，而自回归又把这点微差沿着序列放大了。（这是基于代码与浮点常识的解释；`LiteRT-LM#2281` 的具体现象按上游报告，属【文档】级。）

这个现象有个实际含义：端侧的"可复现"要带上后端这个条件。本书附录 D 的基准数据集因此严格固定后端——换了后端，就是另一组数据，不能混着比（这也是第 2 章数字纪律的由来）。

## 小结

三类后端，各有强项也各有麻烦：CPU 随时可用但要做线程与亲和性的功课，GPU 并行强还能片上采样省搬运，NPU 最省电但最封闭。LiteRT-LM 用一个枚举加一个工厂把它们藏在同一个 `LlmExecutor` 接口后面，上层无感。而"换后端连输出都变"这个反直觉现象，根子是浮点在不同硬件路径上算不出逐比特一致的结果，又被自回归放大。它也提醒我们：端侧的性能与正确性讨论，都要带上"哪个后端"这个前提。

下一章是第三部的压轴：推测解码与 MTP——一次模型前向，怎么吐出不止一个字。

---

## 参考

- 后端枚举与工厂：`runtime/executor/executor_settings_base.h:34 @ v0.13.1`（`Backend`，CPU/GPU/NPU 于 `:45`/`:48`/`:54`）；`runtime/executor/llm_litert_compiled_model_executor_factory.cc @ v0.13.1`（`CreateLlmLiteRtCompiledModelExecutor`:165；`GetBackend`:168；分派:170/174）。
- 片上采样：`runtime/executor/llm_litert_compiled_model_executor.h @ v0.13.1`（`gpu_sampler_max_top_k_`:354；`InitializeSampler`:160）。
- 线程与亲和性：`runtime/framework/threadpool.h:51 @ v0.13.1`；`runtime/engine/cpu_affinity_utils.h @ v0.13.1`（性能核:27；设亲和性:31）。
- NPU：`runtime/executor/llm_litert_npu_compiled_model_executor.h @ v0.13.1`（类:51；embedder 延迟统计:65-77）。

<!-- NPU 无真机，全程标注"基于代码分析"。片上采样省多少、cpu 线程数扫描、cpu vs gpu 对比 待基准 D 回填〔基准 D〕。#2281 现象按【文档】级引用，成因为基于浮点常识的解释、未臆测 issue 内部。图 8-2(数据路径) 表 8-1(权衡) 规格见 notes.md，本轮出签名图 8-1。 -->
