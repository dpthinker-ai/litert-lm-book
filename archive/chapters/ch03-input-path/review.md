# 第 3 章审校记录

> 当前记录：2026-07-18，v4.8 独立审校。适用对象为当前第 3 章正文与图 3-1、图 3-2，源码基线为 LiteRT-LM v0.13.1。

## 审校范围与当前结论

- [x] 语言：清理长句、重复定义、口语判断与无测量支撑的性能形容。
- [x] 禁词与术语：检查禁词、退役叙事标签、所有权术语及 SVG 文本。
- [x] 严谨度：核对 Engine、Session、Conversation 的所有权，模板渲染分支和 embedding 输入路径。
- [x] 叙事姿态：删除拟人化、作者预告和把结构性结论写成性能收益的表述。
- [x] 源码锚点语义：检查每个引用位置是否支撑调用关系、状态复制与缓冲偏移语义。

本文件不记录整书 build 或 PDF 已通过；两项由主会话最终验收。

## 本轮已纠正的关键事实

- `SessionAdvanced::Clone` 负责排队与等待；新旧 handler 起初共享 `SharedProcessedContext`，并按值复制运行配置与状态。只有后续写时分离才进入 executor 的 `CloneContext` 并复制 KV 缓冲。
- 正常逐消息路径使用 `GetSingleTurnText`；不支持单轮渲染时，才对全历史的新旧文本做前缀校验。`GetPrefillTextForMessages` 属于 rewind/refill 的 channel 过滤路径。
- `new_context = old_context` 只保持同一对渲染输入中的 now、tools 与 extra_context 一致，代码仍需显式验证新文本以前缀包含旧文本。
- `byte_offset` 只是当前输出 tensor 的字节写入起点，由 token offset、float 大小与每 token 元素数计算；它不是跨轮缓存或复用标志。
- 删除字符串处理的纳秒级与数量级收益断言；全历史重复提交在固定每轮增长下可呈二次增长，但设备上的实际成本尚未测量。

## 当前仍存在的实验缺口

- 基准模型只覆盖 SentencePiece；HF tokenizer 与 SentencePiece 的双 tokenizer 对照尚未完成。
- 没有 Conversation 增量渲染与全历史回退路径的设备级时延、分配量和 TTFT 对照。
- 写时分离触发 KV 复制的延迟与瞬时内存尚未单独测量。

## 历史记录

- `2026-07-05 · 初稿验收`：旧文件记录了早期模板路径待补和文风审校状态。
- 旧记录把模板实现细节列为当前待办；v4.8 已按冻结源码重写，当前缺口以上一节为准。
