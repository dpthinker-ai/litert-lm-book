# 第 10 章 不止聊天：多模态、约束解码与 Tool Use

> 使命：走出纯文本，看两个方向的扩展——模型怎么"看见"图片、"听见"声音（感知的输入端），以及模型的输出怎么被约束成合法 JSON、变成能执行的函数调用（受控的输出端）。

前三部把纯文本的生成讲透了：一句话进去，一串字出来。但真实应用要的往往不止聊天——它要看图、听音，要模型输出的不是散文而是一个能直接跑的函数调用。这一章讲这两件事。它们方向相反（一个管输入、一个管输出），但各自都建立在前面的地基上。

## 模型怎么"看见"一张图

先说输入端。模型的骨架是处理 token 序列的（第 3 章），它不认识像素。那图片怎么进去？

答案朴素得意外：**把图片也变成 embedding，混进 token 序列里。** 第 3 章说过，文本要先经 embedding 才进模型；图片走的是同一条路，只是"变 embedding"的方式不同。

分三步：切块、编码、注入。

第一步，切块。一张图先被切成许多小块（patch），这一步叫 patchify。切之前要先决定切成多大——`GetAspectRatioPreservingSize`（`runtime/components/preprocessor/image_preprocessor_utils.cc:26 @ v0.13.1`）算的就是这个尺寸。这段算法值得摊开看，它把"patch 数不能超上限"这条约束落成了几行浮点运算：

```cpp
float total_px = width * height;
float target_px =
    patchify_config.max_num_patches *
    (patchify_config.patch_width * patchify_config.patch_height);  // (1)
float factor = std::sqrt(target_px / total_px);                    // (2)
float ideal_height = factor * height;
float ideal_width = factor * width;
int side_mult =
    patchify_config.pooling_kernel_size * patchify_config.patch_width;  // (3)
int target_height =
    static_cast<int>(std::floor(ideal_height / side_mult)) * side_mult;  // (4)
int target_width =
    static_cast<int>(std::floor(ideal_width / side_mult)) * side_mult;
```

(1) 先算允许的总像素上限：`max_num_patches` 个 patch，每个 patch 占 `patch_width × patch_height` 像素。(2) 拿这个上限除以原图像素数、开方，得到一个各向同性的缩放系数 `factor`——乘在宽高上，缩放后的总面积恰好压到上限，同时不改变长宽比。(3)(4) 是这段最容易被忽略的一步：缩放后的宽高不能是任意整数，必须是 `side_mult` 的整数倍，`side_mult` 等于池化核尺寸乘 patch 边长。所以代码先除以 `side_mult`、向下取整、再乘回去——把宽高对齐到网格。为什么要对齐？因为下游要按 `patch_width` 均匀切块、再按 `pooling_kernel_size` 做池化，宽高不是这个乘积的整数倍就切不齐。函数开头还有一句硬性检查：`patch_width != patch_height` 直接返回错误——patch 必须是正方形。向下取整可能把某一边压成 0（细长图），代码专门兜了这个情况：把 0 的那边设成一个 `side_mult`、另一边按原始长宽比放大但不超过 `max_side_length`。这笔账读者可以自己验算：给定 `max_num_patches`、`patch_width`、原图尺寸，`target_height × target_width / (patch_width × patch_height)` 就是这张图最终占多少个 visual token——也就是要在序列里挖多少个坑。

第二步，编码。切好的图交给视觉执行器编码成 embedding。执行器的 `Encode`（`runtime/executor/vision_litert_compiled_model_executor.h:57 @ v0.13.1`）返回一个装着 embedding 的 `ExecutorVisionData`，它的实现是两级串联：

```cpp
absl::StatusOr<ExecutorVisionData> VisionLiteRtCompiledModelExecutor::Encode(
    const litert::TensorBuffer& input_image_tensor) {
  // ...
  LITERT_ASSIGN_OR_RETURN(auto input_image_data,
                          ReferTensorBufferAsSpan<float>(input_image_tensor));
  LITERT_RETURN_IF_ERROR(
      vision_encoder_->GetMutableInputBuffers()[0].Write<float>(
          input_image_data));                                        // (1)
  // ...
  LITERT_RETURN_IF_ERROR(vision_encoder_->GetCompiledModel().Run(
      /*input_buffers=*/vision_encoder_->GetInputBuffers(),
      /*output_buffers=*/encoder_outputs));                          // (2)
  LITERT_RETURN_IF_ERROR(vision_adapter_->GetCompiledModel().Run(
      /*input_buffers=*/encoder_outputs,
      /*output_buffers=*/output_tensor_buffers));                    // (3)
  return ExecutorVisionData(std::move(output_tensor_buffers[0]),
                            /*per_layer_embeddings=*/std::nullopt);   // (4)
}
```

(1) 把预处理好的图像张量写进编码器的输入 buffer。(2) 跑视觉编码器（vision encoder）——这是一个独立的 LiteRT 编译模型，产出的是编码器自己空间里的特征。(3) 编码器的输出直接喂给视觉适配器（vision adapter）再跑一次：适配器负责把编码器特征投影到 LLM 的 embedding 空间，维度对齐成 `model_dimension`，模型骨架才认。两个模型都是编译好的 `CompiledModel`，一个负责"看懂图"、一个负责"翻译成 LLM 的语言"，职责切得很干净。(4) 返回时 `per_layer_embeddings` 传 `std::nullopt`——这条路只产一份普通 embedding，不产逐层 embedding（后者是某些模型的额外通道，这里用不上）。中间那段被 `// ...` 省掉的代码在处理一个后端相关的坑：WebGPU 和 Metal 内存下，输出 buffer 不能复用，第二次调 `Encode` 会报 lock 失败，得每次新建（注释里挂着内部 bug 号 b/457483190）。

第三步，注入——整件事最巧的地方在"预留位置"。序列里先放一串占位的特殊 token，`kSpecialToken` 值为 -1（`runtime/executor/llm_executor_io_types.h:220 @ v0.13.1`）。头文件里给了直观的例子（`:202`）：

```cpp
// token_ids = [2, kSpecialToken, kSpecialToken, kSpecialToken, 106, 77, ...]
// (contains 3 vision tokens)
// Then, the vision embeddings should have shape [3, model_dimension]:
// [[0.1, ...],  // Embedding for the 1st kVisionSpecialToken
//  [0.5, ...],  // Embedding for the 2nd kVisionSpecialToken
//  [0.9, ...]]  // Embedding for the 3rd kVisionSpecialToken
```

三个 `kSpecialToken` 就是给视觉 embedding 占的三个坑，`ExecutorVisionData` 里 embedding 的行数必须严格等于序列中 `kSpecialToken` 的个数——上一步 patchify 算出的 visual token 数，在这里必须对得上。真正 prefill 前，`FillVisionEmbeddings`（`runtime/executor/llm_executor_base.h:178 @ v0.13.1`）把这些 embedding 按行填进对应的坑。它的签名里带一个 `image_index` 参数：一次对话可以塞多张图，每张图占一段连续的 `kSpecialToken`，`image_index` 指定这批 embedding 覆盖哪一张。填完，序列里一部分槽装文本 embedding、一部分装视觉 embedding，对模型来说都是一样的 embedding，一视同仁地往下算。

<figure>
{{#include figs/fig-10-1.svg}}
<figcaption>图 10-1　模型怎么"看见"：图片先 patchify 切块、经视觉执行器编码成 embedding，再填进 token 序列里由特殊 token（kSpecialToken）占好的坑。对模型而言，视觉 embedding 和文本 embedding 无差别。</figcaption>
</figure>

音频走的是同一条路，只是特殊 token 换成 -2（`ExecutorAudioData::kSpecialToken`，`llm_executor_io_types.h:281 @ v0.13.1`；对齐约定在 `:263` 的注释里，和视觉逐字对应）。音频执行器 `AudioLiteRtCompiledModelExecutor::Encode`（`runtime/executor/audio_litert_compiled_model_executor.cc:941 @ v0.13.1`）的结构和视觉那段几乎一样：编码器 `Run` 一次、适配器 `Run` 一次、包成 `ExecutorAudioData` 返回。三个模态各用一个 `kSpecialToken` 值（文本无、视觉 -1、音频 -2）把坑区分开，填的时候各填各的。这就是多模态的统一技巧：**万物皆 embedding**。不同模态各有各的编码器把自己变成 embedding，一旦变成了 embedding，后面的 prefill、decode（第 4、5 章）一个字都不用改——它们本就是在 embedding 上工作，不在乎这些 embedding 当初是文本、图片还是声音。这也是为什么第 2 章那五层架构能保持干净：多模态是"在输入端多接一个编码器"，而不是"把整条流水线改一遍"。

## 让输出守规矩：约束解码

现在转到输出端。聊天场景下，模型爱怎么说怎么说。但如果你要它输出一段 JSON、或者一个格式严格的函数调用，"爱怎么说"就成了问题——它可能漏个引号、多个逗号，让下游解析崩掉。

约束解码的思路是：**在每一步采样前，把不合法的 token 全部掐掉。** 回忆第 5 章的 decode 循环，采样是从 logits（每个 token 的分数）里挑一个。约束解码在采样前插一手，把当前语法不允许的 token 的分数全设成负无穷——它们的概率就成了零，永远不会被选中。

头文件把这套流程写在类注释里（`ConstrainedDecoder`，`runtime/components/constrained_decoding/constrained_decoder.h:48 @ v0.13.1`），是逐字从源码抄下来的循环：

```cpp
//   ConstrainedDecoder decoder(constraint, batch_size);
//   while (!done) {
//     TensorBuffer logits = Decode(...);
//     RETURN_IF_ERROR(decoder.MaskLogits(logits));            // (1)
//     TensorBuffer next_tokens = sampler.Sample(logits);      // (2)
//     RETURN_IF_ERROR(decoder.UpdateConstraintState(next_tokens));  // (3)
//   }
```

(1) `MaskLogits` 把非法 token 打成 -inf。(2) 采样此时只可能选到合法 token。(3) `UpdateConstraintState` 根据选中的 token 推进语法状态，决定下一步哪些合法。三步咬合成一个循环，插在第 5 章 decode 循环的采样前后。注意这里的 `decoder` 是有状态的——它在构造时给 batch 里每条序列各调一次 `constraint_->Start()` 存下初始语法状态，后面每步都在这份状态上推进。

`MaskLogits` 的实现（`runtime/components/constrained_decoding/constrained_decoder.cc:73 @ v0.13.1`）核心就是一个双重循环：

```cpp
for (int b = 0; b < batch_size; ++b) {
  auto& constraint_state = constraint_states_[b];
  ASSIGN_OR_RETURN(auto bitmap,
                   constraint_->ComputeBitmap(*constraint_state));   // (1)
  for (int i = 0; i < vocab_size; ++i) {
    if (!bitmap->Get(i)) {                                           // (2)
      logits.data()[b * vocab_size + i] =
          std::numeric_limits<float>::lowest();                      // (3)
    }
  }
}
```

(1) 对每条序列，问当前语法状态："此刻哪些 token 合法？"答案是一张按词表大小铺开的位图（`Bitmap`，`runtime/components/constrained_decoding/bitmap.h:21 @ v0.13.1`）。(2)(3) 遍历整个词表，位图里为 0 的位置——不合法的 token——把它的 logit 直接压到 `float` 的最小值。这就是"负无穷"的工程写法：不是数学上的 $-\infty$，而是 `std::numeric_limits<float>::lowest()`，经过 softmax 后概率无限趋近 0，采样再也选不到它。这段前面还有一串 `RET_CHECK`：要求 logits 形状是 `[batch_size, 1, vocab_size]`（序列长度必须是 1，一次只掩一步），并且模型词表不能大于约束词表——约束词表可以更大，多出来的位当未用 token 处理。代价也在这段里明摆着：每步要对整个词表跑一遍，`vocab_size` 量级在几万到十几万，这是约束解码相比裸采样多付的每步开销。

"哪些合法"最终由谁算？底层的语法引擎是 llguidance，一个 Rust 库，通过 C bridge 接进来（见仓库的 `PATCH.llguidance*` 与 `docs/api/cpp/constrained-decoding.md`）。C++ 侧的 `LlgConstraint`（`runtime/components/constrained_decoding/llg_constraint.cc @ v0.13.1`）只是把三个动作转成三次 FFI 调用：`Start` 调 `llg_clone_constraint` 克隆一份初始状态，`ComputeNext` 调 `llg_commit_token` 推进，`ComputeBitmap` 调 `llg_compute_mask` 取掩码。取回的掩码是 llguidance 打包的 32 位字数组，C++ 侧再解包成布尔位图：

```cpp
mask_vector.push_back(sample_mask[i / 32] & (1 << (i % 32)));  // (1)
```

(1) 第 `i` 个 token 的允许位，藏在第 `i / 32` 个 32 位字的第 `i % 32` 位——用位运算把它取出来。语法怎么写、GBNF 或 JSON Schema 怎么编译成状态机，都在 Rust 那边；C++ 这层只管"每步取一张位图、按位掩 logits"。分工清楚：约束的表达力归 llguidance，掩码的执行归这三十行 C++。

约束解码的意义是把"结构合法"从"祈祷模型别出错"变成"从机制上不可能出错"——只要语法写对了，输出就一定合法。这对下一节的工具调用是刚需。

## Tool Use：让模型调用函数

把感知输入和受控输出接起来，就是 Tool Use（工具调用/函数调用）——让模型不只是回话，而是能调外部函数：查天气、算数、搜数据库。

一次工具调用走一条完整的链路：

1. **声明工具**。你在对话的开场白里告诉模型有哪些工具可用——就是第 3 章 `Preface` 里的 `tools` 字段。`Preface` 是个 `variant`，实际装的是 `JsonPreface`（`runtime/conversation/io_types.h:30 @ v0.13.1`），里面三个 `nlohmann::ordered_json` 字段并排：`messages` 是对话历史，`tools` 是可用工具列表（`:36`），`extra_context` 留给模型特定的模板渲染。用 `ordered_json` 而非普通 `json` 是有意的——工具和参数的书写顺序要保住，格式化进 prompt 时不能被容器重排。
2. **格式化进 prompt**。这些工具描述被按模型认得的格式写进提示词。`FormatValueAsFc`（`runtime/components/tool_use/fc_tool_format_utils.h @ v0.13.1`，`fc` 即 function call）把标准 JSON 转成一种更省 token 的 FC 格式，头文件里的例子把差异讲明白了：键不加引号（`"string_value"` 变 `string_value`），字符串用 `<escape>` 标签包起来而不是双引号（`"foo"` 变 `<escape>foo<escape>`）。去掉成对的引号，是为了让同样的工具声明少占 token——上下文窗口寸土寸金，工具声明又常年占在 prompt 开头。
3. **生成调用**。模型决定要用某个工具时，输出一段结构化的函数调用文本，长这样：`call:tool_name{param_1:7,param_2:<escape>foo<escape>}`。这里约束解码派上用场：开着它，模型吐出的调用就一定是结构合法的（第 3 章 `ConversationConfig` 那个开关的用途之一）。
4. **解析回填**。运行时把这段文本解析回结构化的函数名和参数。解析用的是 ANTLR 语法，而不是拿正则去凑。FC 格式的整个文法只有六条规则（`runtime/components/tool_use/antlr/AntlrFcParser.g4 @ v0.13.1`）：

```antlr
start : functionCall EOF;
functionCall: CALL COLON ID object?;            // (1)
object : OPEN_BRACE ( pair (COMMA pair)* )? CLOSE_BRACE;
pair : ID COLON value;
value
    : ESCAPED_STRING | NUMBER | BOOLEAN | NULL_LITERAL | object | array;  // (2)
array: OPEN_BRACKET ( value (COMMA value)* )? CLOSE_BRACKET;
```

(1) 一次调用就是 `call` 关键字、冒号、函数名（`ID`）、后面跟一个可选的参数 `object`——`call:tool_name{...}` 逐字对上。(2) `value` 是递归定义的：一个值可以是字符串、数字、布尔、null，也可以再嵌一个 `object` 或 `array`。这条递归是正则做不到的——正则识别不了任意深度的嵌套括号，而文法解析器天生能。`ParseFcExpression`（`runtime/components/tool_use/fc_parser_utils.h:41 @ v0.13.1`）跑完这套文法，把 `call:tool_name{param_1:7,param_2:<escape>foo<escape>}` 还原成 `{"name":"tool_name","arguments":{"param_1":7,"param_2":"foo"}}`——`<escape>` 标签脱掉、`7` 还原成数字而非字符串。解析出的调用交给你的函数执行，结果再作为一条消息喂回模型，继续对话。同目录下另有 `AntlrJson`、`AntlrPython` 两套文法，对应不同模型偏好的调用格式（有的吐 JSON、有的吐 Python 风格的函数调用）。

这条链路把前面几章的零件串了起来：Preface（第 3 章）声明工具，约束解码（本章上一节）保证输出合法，ANTLR 文法把文本解析回结构。四步下来，一个只会输出文本的模型，就有了调用真实函数的能力。（各环节职责与完整时序，另见 `docs/api/cpp/tool-use.md`。）

> 版本注记
> 工具调用的文法与解析细节（ANTLR 那几个 `.g4`、不同模型的函数调用格式差异）在版本间有演进，本节只讲稳定的四步骨架。某些模型在嵌套 JSON 参数上的解析边界曾有过问题（上游 `LiteRT-LM#2418`），属实现细节，不在本节的骨架之列。

## 小结

这一章讲了两个方向的扩展，它们共享同一套地基。输入端，多模态靠"万物皆 embedding"——图片经 patchify、视觉执行器编码、再填进特殊 token 占的坑，之后的流水线一字不改。输出端，约束解码靠"每步掐掉非法 token"把结构合法从祈祷变成保证，Tool Use 再把它和 Preface、ANTLR 文法串成完整的函数调用链路。两个方向都印证了第 2 章那句"接口隔离"：新能力是在输入端多接一个编码器、在输出端多插一道约束长出来的，核心流水线一字未动。

下一章是最后一块工程拼图：这套 C++ 核心，怎么服务从 Python 到 Web 的六种语言（这笔账，下一章开头数清）。

---

## 参考

- 多模态：`runtime/components/preprocessor/image_preprocessor_utils.cc:26 @ v0.13.1`（`GetAspectRatioPreservingSize` 缩放/切块，头文件声明 `image_preprocessor_utils.h:28`）；`runtime/executor/vision_litert_compiled_model_executor.cc:454 @ v0.13.1`（`Encode` 两级串联，头文件声明 `.h:57`）；`runtime/executor/llm_executor_io_types.h @ v0.13.1`（`ExecutorVisionData`:216；`kSpecialToken`:220；对齐示例:202；`ExecutorAudioData`:277，注释:263，`kSpecialToken`:281）；`runtime/executor/llm_executor_base.h:178 @ v0.13.1`（`FillVisionEmbeddings`，带 `image_index`）；`runtime/executor/audio_litert_compiled_model_executor.cc:941 @ v0.13.1`（音频 `Encode`）。
- 约束解码：`runtime/components/constrained_decoding/constrained_decoder.h:48 @ v0.13.1`（循环用法注释:39-47）；`constrained_decoder.cc:73 @ v0.13.1`（`MaskLogits` 双重循环 + `float` 最小值）；`bitmap.h:21`（`Bitmap::Get`）；`llg_constraint.cc @ v0.13.1`（`Start`/`ComputeNext`/`ComputeBitmap` 三次 FFI、掩码解包:57）；llguidance（`PATCH.llguidance*`）；`docs/api/cpp/constrained-decoding.md`。
- Tool Use：`runtime/conversation/io_types.h:30 @ v0.13.1`（`JsonPreface`，`tools`:36，`Preface` variant:59）；`runtime/components/tool_use/fc_tool_format_utils.h @ v0.13.1`（`FormatValueAsFc`，FC 格式）；`fc_parser_utils.h:41`（`ParseFcExpression`）；`antlr/AntlrFcParser.g4 @ v0.13.1`（六条规则文法，另有 `AntlrJson`/`AntlrPython`）；`docs/api/cpp/tool-use.md`。

<!-- 补读：vision/audio executor 已贴 .cc（Encode 两级串联 encoder→adapter），patchify 贴 .cc（GetAspectRatioPreservingSize 缩放对齐公式），约束解码贴 MaskLogits 双重循环 + llg_constraint FFI 三调用 + 掩码位解包，tool_use 贴 AntlrFcParser.g4 六条文法 + FC 格式差异 + ParseFcExpression。实测（图片端到端、visual token 计数、约束解码开/关工具调用成功率）待基准 D 回填〔基准 D〕。双主题章，两半已切干净。图 10-2(约束解码逐步屏蔽) 表 10-1(Tool Use 各环节) 规格见 notes.md，本轮出签名图 10-1。 -->
