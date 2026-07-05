# 第 10 章 不止聊天：多模态、约束解码与 Tool Use

> 使命：走出纯文本，看两个方向的扩展——模型怎么"看见"图片、"听见"声音（感知的输入端），以及模型的输出怎么被约束成合法 JSON、变成能执行的函数调用（受控的输出端）。

前三部把纯文本的生成讲透了：一句话进去，一串字出来。但真实应用要的往往不止聊天——它要看图、听音，要模型输出的不是散文而是一个能直接跑的函数调用。这一章讲这两件事。它们方向相反（一个管输入、一个管输出），但各自都建立在前面的地基上。

## 模型怎么"看见"一张图

先说输入端。模型的骨架是处理 token 序列的（第 3 章），它不认识像素。那图片怎么进去？

答案朴素得意外：**把图片也变成 embedding，混进 token 序列里。** 第 3 章说过，文本要先经 embedding 才进模型；图片走的是同一条路，只是"变 embedding"的方式不同。

分三步。第一步，切块。一张图先被切成许多小块（patch）——这一步叫 patchify，由图像预处理器完成，它会把图缩放到合适尺寸，让 patch 数量不超过上限，且每个 patch 是正方形（`runtime/components/preprocessor/image_preprocessor_utils.h:28 @ v0.13.1`）。第二步，编码。这些 patch 交给一个专门的视觉执行器编码成 embedding（`VisionLiteRtCompiledModelExecutor::Encode`，`runtime/executor/vision_litert_compiled_model_executor.h:57 @ v0.13.1`，返回一个装着 embedding 的 `ExecutorVisionData`）。第三步，注入。把这些视觉 embedding 填进 token 序列里预留好的位置。

第三步的"预留位置"是整件事最巧的地方。序列里先放一串占位的特殊 token（代码里 `kSpecialToken` 值为 -1，`runtime/executor/llm_executor_io_types.h:220 @ v0.13.1`）。头文件里有个直观的例子（`:202`）：`token_ids = [2, kSpecialToken, kSpecialToken, kSpecialToken, 106, 77, ...]`——那几个 `kSpecialToken` 就是给视觉 embedding 占的坑。真正 prefill 前，`FillVisionEmbeddings`（`runtime/executor/llm_executor_base.h:178 @ v0.13.1`）把视觉 embedding 填进这些坑。填完，序列里一部分槽装文本 embedding、一部分装视觉 embedding，对模型来说都是一样的 embedding，一视同仁地往下算。

<figure>
{{#include figs/fig-10-1.svg}}
<figcaption>图 10-1　模型怎么"看见"：图片先 patchify 切块、经视觉执行器编码成 embedding，再填进 token 序列里由特殊 token（kSpecialToken）占好的坑。对模型而言，视觉 embedding 和文本 embedding 无差别。</figcaption>
</figure>

音频走的是同一条路（`ExecutorAudioData`，用同样的特殊 token 对齐约定，`llm_executor_io_types.h:263 @ v0.13.1`）。这就是多模态的统一技巧：**万物皆 embedding**。不同模态各有各的编码器把自己变成 embedding，一旦变成了 embedding，后面的 prefill、decode（第 4、5 章）一个字都不用改——它们本就是在 embedding 上工作，不在乎这些 embedding 当初是文本、图片还是声音。这也是为什么第 2 章那五层架构能保持干净：多模态是"在输入端多接一个编码器"，而不是"把整条流水线改一遍"。

## 让输出守规矩：约束解码

现在转到输出端。聊天场景下，模型爱怎么说怎么说。但如果你要它输出一段 JSON、或者一个格式严格的函数调用，"爱怎么说"就成了问题——它可能漏个引号、多个逗号，让下游解析崩掉。

约束解码的思路是：**在每一步采样前，把不合法的 token 全部掐掉。** 回忆第 5 章的 decode 循环，采样是从 logits（每个 token 的分数）里挑一个。约束解码在采样前插一手，把当前语法不允许的 token 的分数全设成负无穷——它们的概率就成了零，永远不会被选中。

代码把这套流程写得很清楚（`ConstrainedDecoder`，`runtime/components/constrained_decoding/constrained_decoder.h:48 @ v0.13.1`，用法注释在 `:43`–`:46`）：

```cpp
TensorBuffer logits = Decode(...);
decoder.MaskLogits(logits);              // 非法 token 的 logit 设为 -inf
TensorBuffer next = sampler.Sample(logits);
decoder.UpdateConstraintState(next);     // 按选中的 token 推进语法状态
```

三步咬合成一个循环：解码出 logits → `MaskLogits` 把非法 token 打成 -inf → 采样（此时只可能选到合法 token）→ `UpdateConstraintState` 根据选中的 token 推进语法状态，决定下一步哪些 token 合法。"哪些合法"由一个词表位图表示（`Bitmap`，`runtime/components/constrained_decoding/bitmap.h:21 @ v0.13.1`），底层的语法引擎是 llguidance（一个通过 cxx bridge 接进来的 Rust 库，见仓库的 `PATCH.llguidance*` 与 `docs/api/cpp/constrained-decoding.md`）。

约束解码的意义是把"结构合法"从"祈祷模型别出错"变成"从机制上不可能出错"——只要语法写对了，输出就一定合法。这对下一节的工具调用是刚需。

## Tool Use：让模型调用函数

把感知输入和受控输出接起来，就是 Tool Use（工具调用/函数调用）——让模型不只是回话，而是能调外部函数：查天气、算数、搜数据库。

一次工具调用走一条完整的链路：

1. **声明工具**。你在对话的开场白里告诉模型有哪些工具可用——就是第 3 章 `Preface` 里的 `tools` 字段（`runtime/conversation/io_types.h:36 @ v0.13.1`）。
2. **格式化进 prompt**。这些工具描述被按模型认得的格式写进提示词（`runtime/components/tool_use/fc_tool_format_utils.h @ v0.13.1`，`fc` 即 function call）。
3. **生成调用**。模型决定要用某个工具时，输出一段结构化的函数调用文本。这里约束解码派上用场：开着它，模型吐出的调用就一定是结构合法的（第 3 章 `ConversationConfig` 那个开关的用途之一）。
4. **解析回填**。运行时把这段文本解析成结构化的函数名和参数。解析用的是 ANTLR 语法（`runtime/components/tool_use/antlr/` 下的 `AntlrFcParser`、`AntlrJson`、`AntlrPython` 等 `.g4` 文法，配合 `fc_parser_utils.h`）——用正儿八经的文法解析器，而不是拿正则去凑。解析出的调用交给你的函数执行，结果再作为一条消息喂回模型，继续对话。

这条链路把前面几章的零件串了起来：Preface（第 3 章）声明工具，约束解码（本章上一节）保证输出合法，ANTLR 文法把文本解析回结构。四步下来，一个只会输出文本的模型，就有了调用真实函数的能力。（各环节职责与完整时序，另见 `docs/api/cpp/tool-use.md`。）

> 版本注记
> 工具调用的文法与解析细节（ANTLR 那几个 `.g4`、不同模型的函数调用格式差异）在版本间有演进，本节只讲稳定的四步骨架。某些模型在嵌套 JSON 参数上的解析边界曾有过问题（如上游报告的 Gemma 4 相关 issue），属实现细节，不在本节的骨架之列。

## 小结

这一章讲了两个方向的扩展，它们共享同一套地基。输入端，多模态靠"万物皆 embedding"——图片经 patchify、视觉执行器编码、再填进特殊 token 占的坑，之后的流水线一字不改。输出端，约束解码靠"每步掐掉非法 token"把结构合法从祈祷变成保证，Tool Use 再把它和 Preface、ANTLR 文法串成完整的函数调用链路。两个方向都印证了第 2 章那句"接口隔离"：新能力是在输入端多接一个编码器、在输出端多插一道约束长出来的，核心流水线一字未动。

下一章是最后一块工程拼图：这套 C++ 核心，怎么变成 Python、Kotlin、Swift、Web 六种语言都能用的 SDK。

---

## 参考

- 多模态：`runtime/executor/vision_litert_compiled_model_executor.h:57 @ v0.13.1`（`Encode`）；`runtime/executor/llm_executor_io_types.h @ v0.13.1`（`ExecutorVisionData`:216；`kSpecialToken`:220；对齐示例:202；`ExecutorAudioData`:263）；`runtime/executor/llm_executor_base.h:178 @ v0.13.1`（`FillVisionEmbeddings`）；`runtime/components/preprocessor/image_preprocessor_utils.h:28 @ v0.13.1`（patchify）。
- 约束解码：`runtime/components/constrained_decoding/constrained_decoder.h:48 @ v0.13.1`（`MaskLogits`/`UpdateConstraintState`，用法:43-46）；`bitmap.h:21`；llguidance（`PATCH.llguidance*`）；`docs/api/cpp/constrained-decoding.md`。
- Tool Use：`runtime/conversation/io_types.h:36 @ v0.13.1`（`Preface.tools`）；`runtime/components/tool_use/`（`fc_tool_format_utils.h`、`fc_parser_utils.h`、`antlr/*.g4`）；`docs/api/cpp/tool-use.md`。

<!-- 补读：vision/audio executor 读了 .h（Encode 接口 + kSpecialToken 对齐约定，机制层面足够）；full .cc 走查可留待深化。实测（图片端到端、visual token 计数、约束解码开/关工具调用成功率）待基准 D 回填〔基准 D〕。双主题章，两半已切干净。图 10-2(约束解码逐步屏蔽) 表 10-1(Tool Use 各环节) 规格见 notes.md，本轮出签名图 10-1。 -->
