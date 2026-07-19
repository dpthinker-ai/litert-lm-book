# 第 10 章审校记录

> 当前记录：2026-07-18，v4.8 独立审校与技术扩充。适用对象为当前第 10 章正文、图 10-1 至图 10-4、表 10-1 至表 10-5，源码基线为 LiteRT-LM v0.13.1。

## 审校范围与当前结论

- [x] 独立交叉审校：扩写由独立会话完成，主会话复核了输入所有权、visual token 与 KV 容量计算、音频分块、约束状态时序、多模态诊断和宿主执行边界。
- [x] 独立文风审校：按 `humanizer-cn` 与本仓规范删去排查演练的编辑提示语，并把抽象的执行入口改为具体宿主动作。
- [x] 语言：清理长句、口语标题、类比残留和结尾预告。
- [x] 禁词与术语：检查禁词、退役叙事标签、visual token、音频特征位置和文法术语。
- [x] 严谨度：核对图像与音频 embedding、KV 公式、signature、约束采样、parser 与应用执行边界。
- [x] 叙事姿态：删除拟人化输入、保证扩大和函数调用自动可执行等表述。
- [x] 源码锚点语义：逐项核验完整路径与行号是否支撑相邻断言；图 10-1 至图 10-4 同步核对。
- [x] 输入契约：核对 `Message`、`LoadItemData`、`InputData`、移动语义和 `TensorBuffer::Duplicate`。
- [x] shape 与生命周期：核对 Gemma 4 标记配对、patchify、visual token budget、多图组合、音频分块与 reset。
- [x] 约束边界：区分 tools-derived llguidance 文法与通用 ANTLR FC parser，不把结构约束扩大为完整 schema 或授权保证。
- [x] Tool Use：核对原始文本、parser JSON、宿主校验、执行与结果回填的责任边界。

本文件不记录整书 build 或 PDF 已通过；两项由主会话最终验收。

## 本轮已纠正的关键事实

- 视觉占位符在 prefill 时由 `EmbeddingLookupMultiModal` 依次写入，不再引用并不存在于当前链路中的 `FillVisionEmbeddings`。
- 多个 signature 是多个命名入口，数量本身不能证明 encoder 权重被复制；是否共享需继续检查 subgraph 与常量张量。
- visual token 的 KV 增量使用各层 `H_kv × D`，不能代入 `model_dimension`。按附录 D 的混合 KV 形状，256 个 visual token 对应 7 MiB 活动 KV 数据。
- 音频张量 `[1, 204, 1536]` 中 204 是编码器特征序列位置数，不是原始波形或 log-mel 帧数。
- v0.13.1 的约束解码同时覆盖外部采样和内部采样；给定文法只保证其中编码的结构，不保证函数存在、参数语义、权限或可执行性。
- 约束实验为开启与关闭各 6 次、共 12 次；全部样本结构可解析，但样本量不足以估计失败率。
- `InputImage` 的 `string_view` 分支不拥有源字节；Gemma 4 的 path/blob 路径在预处理前构造 owned `std::string`。
- 多图 embedding 沿 token 维拼接，不等同于图像 batch；多个输入会分配组合 host buffer 并复制 packed bytes。
- 音频无输出 mask 时按块对有效长度向上取整，总 token 数应累加每块结果，不能在所有配置下直接写成整段长度除以缩减因子。
- tools-derived FC 文法限制声明过的函数名、顶层参数、基本类型与部分枚举，但当前生成代码没有读取数值范围、字符串 pattern 或嵌套 schema。

## 当前仍存在的实验缺口

- 图像输入、patch 数与 visual token 数的多模态端到端验证尚未完成。
- 音频端到端、分块时延及预处理占比尚未测量。
- 12 次约束实验没有覆盖多工具混淆、嵌套 JSON、参数语义与真实函数执行，也不足以比较稳定失败率。
- `set_device_mode` 是应用侧设计案例，不是已执行的 LiteRT-LM 样例；授权、幂等、超时与结果裁剪未做本书实验。

## 历史记录

- `2026-07-05 · 初稿验收`：旧文件记录了多模态与约束解码的早期事实核查和图表计划。
- 旧文件中的单图、机制与实验待办已被 v4.8 重新核定；当前缺口以上一节为准。
