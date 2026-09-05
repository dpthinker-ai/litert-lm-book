# M2 文字精简样稿对照

日期：2026-09-05。原文基线：`3817ec5`（M1 完成后）。

**状态：未采用。** 作者反馈两处改稿过于精简、阅读效果不如原文。正文已恢复至 M1 完成版；以下仅保留此次对照，不作为后续修改范例。技术审校通过不代表风格认可，阶段状态见 [Roadmap](ROADMAP.md)。

## S01：第 2 章小结

### 原文

本章完成了三件事。第一，建立了运行起点：Python 与 C++ 两个入口都能运行，对流式片段与 token 的关系、取消的生效时机、后端的可选性都有了直接观察。第二，明确了测量方法：四项指标按 `BenchmarkInfo` 的源码定义来读，TTFT 是计算值，Init 各阶段可能重叠，单点 tokens/s 无法判断是算力约束还是带宽约束。第三，给出了三份索引：二十个问题标注了对应章节，五层视图用于定位代码职责，`.litertlm` 的 section 布局说明了模型文件的组成。接下来第 3 至 5 章沿调用链继续推进，依次讨论 Session 状态、prefill/decode 编排与输出处理。

### 未采用改稿

解读 benchmark 结果，先核对模型、设备、后端、输入与输出长度及缓存状态。`BenchmarkInfo` 的 TTFT 由首轮 prefill 的完整耗时与首轮 decode 的平均单 token 耗时相加得到。Init 则累加可能重叠的阶段耗时；这两项都不能代替应用侧的独立计时。

吞吐对照用于确定下一步测什么。改变后端或上下文长度可以观察性能对条件的敏感度，但单个 tokens/s 数值不能判定计算或带宽瓶颈。定位实现时，可用五层职责视图查找相关组件，再按 `.litertlm` 的实际 section 布局核对模型组成。

### 原修改意图（未获认可）

删除“第一、第二、第三”的章节内容回顾与后续章节预告。保留 TTFT、Init 的统计口径和单点吞吐不能确定瓶颈的限制，以测量与定位方法收尾。运行入口、取消行为和二十个问题索引仍在章内原处。

## S02：第 11 章 11.6，输入缓冲的复制与所有权

本样稿从 C 输入结构体代码块后的说明开始，直到本节末段。结构体定义及其中的类型、指针、字节数注释保留不动。

### 代码前原文

每个元素由 `type` 和 `(data, size)` 组成。代码行 `(1)` 的 `const void*` 指向缓冲，`(2)` 给出字节数；文本数据采用 UTF-8，图像与音频使用相应的编码字节，`type` 决定 C++ 侧如何解释缓冲。转换函数把这个数组变成 C++ 的 `InputData`：

### 未采用改稿（代码前）

`type` 决定 `(data, size)` 的解释方式：文本采用 UTF-8，图像与音频使用相应的编码字节。C++ 侧复制这些内容，使原生输入不再依赖调用方缓冲的生命周期：

### 两版共用的代码（保持原样）

```cpp
// c/engine.cc:127-154
std::vector<litert::lm::InputData> ToEngineInputData(
    const LiteRtLmInputData* inputs, size_t num_inputs) {
  std::vector<litert::lm::InputData> engine_inputs;
  engine_inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    switch (inputs[i].type) {
      case kLiteRtLmInputDataTypeText:
        engine_inputs.emplace_back(litert::lm::InputText(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));  // (1)
        break;
      case kLiteRtLmInputDataTypeImage:
        engine_inputs.emplace_back(litert::lm::InputImage(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));  // (2)
        break;
      case kLiteRtLmInputDataTypeImageEnd:
        engine_inputs.emplace_back(litert::lm::InputImageEnd());          // (3)
        break;
      case kLiteRtLmInputDataTypeAudio:
        engine_inputs.emplace_back(litert::lm::InputAudio(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));
        break;
      case kLiteRtLmInputDataTypeAudioEnd:
        engine_inputs.emplace_back(litert::lm::InputAudioEnd());
        break;
    }
  }
  return engine_inputs;
}
```

### 代码后原文

`switch` 根据标签构造对应类型：代码行 `(1)` 生成 `InputText`，`(2)` 生成 `InputImage`，`(3)` 生成不含缓冲的 `InputImageEnd`，音频分支采用相同结构。`(1)`、`(2)` 中的 `std::string(ptr, size)` 会把 C 缓冲复制到新字符串，复制量等于输入的字节数。

接口因此不接管调用方缓冲的所有权，也不要求缓冲在函数返回后继续存活：实现复制了数据，`InputData` 独立拥有内容。若要避免复制，C ABI 需要增加可验证的生命周期契约，例如所有权转移或释放回调，当前没有实现这些方案。本书也没有测量这次复制在端到端时延中的占比。

### 未采用改稿（代码后）

复制发生在 `std::string(ptr, size)`：每个文本、图像或音频数据项复制 `size` 字节。图像结束标记 `(3)` 与音频结束标记不携带缓冲。

复制后，`InputData` 独立拥有内容，原缓冲仍归调用方所有，可在 C API 返回后释放。省去复制需要另行约定缓冲的生命周期，例如转移所有权或提供释放回调；当前接口没有这些安排。本书未测量这次复制在端到端时延中的占比。

### 原修改意图（未获认可）

删除逐个复述 switch 分支和类型名称的解释，把复制目的放在代码前，复制量与释放条件放在代码后。保留 type/data/size、结束标记、调用方所有权、零复制所需的生命周期约定，以及尚无时延占比测量的限制。代码和出处注释保持原样。
