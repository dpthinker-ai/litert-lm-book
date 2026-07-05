# 第 10 章 review.md

> 首稿验收留痕。状态：**初稿完成，Pass 1 部分完成，Pass 2 lint 已过、独立审校（§14）待做**。

## Pass 1 · 事实核查（断言四级制 + 引用逐条核验）

引用核验（本会话用 `git show v0.13.1:<path>` 逐条比对，✅ 者已确认支撑断言）：
- vision_litert_compiled_model_executor.h:43(类)/48(Create)/57(Encode) ✅
- llm_executor_io_types.h:216(ExecutorVisionData)/220(kSpecialToken=-1)/202(对齐示例)/263(ExecutorAudioData) ✅
- llm_executor_base.h:178(FillVisionEmbeddings) ✅
- image_preprocessor_utils.h:28(patchify 缩放/正方 patch) ✅
- constrained_decoding/constrained_decoder.h:48(类)/34-46(MaskLogits→Sample→UpdateConstraintState 用法) ✅；bitmap.h:21(Bitmap) ✅
- 约束解码代码片段（Decode/MaskLogits/Sample/UpdateConstraintState）来自 :43-46 用法注释，逐字对齐 ✅
- tool_use/：fc_tool_format_utils.h、fc_parser_utils.h、antlr/*.g4（AntlrFcParser/Json/Python）路径已核对存在 ✅；conversation/io_types.h:36(Preface.tools) ✅
- 约束解码路径修正：v0.13.1 在 `components/constrained_decoding/`（非 logits_processor/），已核对 ✅

- llguidance 为 Rust 库经 cxx bridge（PATCH.llguidance*）——【文档】级，未展开其内部 ✅
- 嵌套 JSON 参数解析边界问题：以「版本注记」【文档】级提及，不臆测细节 ✅
- 实测（图片端到端、visual token 计数、约束开/关成功率）标注待基准 D ✅

## Pass 2 · 除 AI 味

- [x] `scripts/lint_prose.sh chapters/ch10-multimodal/chapter.md` —— 0 命中（2026-07-05）
- [x] 除 AI 味清单第 1-13 条：撰写时自查（双主题章，两半切干净，各自独立小结前不硬拔高）
- [ ] 第 14 条 独立审校：**待独立会话执行**

## 待清零（补读/数据）
- [x] vision/audio executor：读 .h（Encode 接口 + kSpecialToken 对齐约定），机制层面足够；full .cc 走查可留待深化
- [ ] 实测待基准 D
- [ ] 图 10-2（约束解码逐步屏蔽）、表 10-1（Tool Use 各环节职责）待补（本轮出签名图 10-1）

## 验收自问
- [x] 使命兑现（输入端多模态"万物皆 embedding"、输出端约束解码+Tool Use；两方向都印证接口隔离）
- [x] 图 10-1 已落地并编号
- [ ] 实验可复现（脚本随基准 D）
- [x] 与 BOOK_PLAN 章卡一致；承接第 3 章 Preface/embedding、第 5 章采样、第 2 章接口隔离；预告第 11 章
