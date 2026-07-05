# 第 6 章 review.md

> 首稿验收留痕。状态：**初稿完成，Pass 1 部分完成，Pass 2 lint 已过、独立审校（§14）待做**。

## Pass 1 · 事实核查（断言四级制 + 引用逐条核验）

引用核验（本会话用 `git show v0.13.1:<path>` 逐条比对，✅ 者已确认支撑断言）：
- kv_cache_interface.h:28(类)/39(Serialize)/42(Load)/50(SelectAndCopyFrom)/58(BroadcastAndCopyFrom)/61(DeepCopy) ✅
- llm_litert_compiled_model_executor.h:327(注释)/329-330(双缓冲)/331-333(读写指针) ✅
- engine.h:245(Clone)/263(CloneAsync)/270(SaveCheckpoint)/238(Clone 示例) ✅
- llm_executor_io_types.h:78(RuntimeState)/92(LlmContext)/80(current_step) ✅

- 【推测】/示例级：KV cache 公式的 L/H_kv/D 代入值为**示例量级**，正文已明说"具体模型数字可从元数据读出"，未冒充某模型真实参数 ✅
- 数字：480 MiB / 120 KiB/token 为公式示例，非实测；实测（decode 随上下文变慢、--max-num-tokens 影响）标注待基准 D ✅
- `LiteRT-LM#2568` 按【文档】级引用现象，不臆测其内部实现 ✅

## Pass 2 · 除 AI 味

- [x] `scripts/lint_prose.sh chapters/ch06-kv-cache/chapter.md` —— 0 命中（2026-07-05）
- [x] 除 AI 味清单第 1-13 条：撰写时自查
- [x] 第 14 条 独立审校：**已由独立审校会话执行（2026-07-05）**，判为"AI 味很淡、可放行"；升华式收尾/自夸形容词/括号内同义重述等问题已逐条修正，修后 lint 复跑 0 命中

## 待清零（补读/数据）
- [ ] 实验数字（--max-num-tokens 扫描 #2568、Clone 分叉、get_token_count 增长）待基准 D 回填
- [ ] KV cache 公式的精确 L/H_kv/D（待第 7 章 litertlm_print 读出某模型真实值后可补）
- [ ] 图 6-2（KV cache 增长）、图 6-3（Clone/Rewind 状态分叉）、表 6-1（内存账）待补（本轮出了签名图 6-1(双缓冲)）

## 验收自问
- [x] 使命兑现（算清 KV cache 大小、补上第 5 章欠账、讲清双缓冲与状态即对象）
- [x] 图 6-1（双缓冲）已落地并编号；其余图表规格在 notes.md
- [ ] 实验可复现（脚本随基准 D）
- [x] 与 BOOK_PLAN 章卡一致；交叉引用用"见第 N 章"格式；回收了第 2 章第 13/14 问
