# 第 7 章 review.md

> 首稿验收留痕。状态：**初稿完成，Pass 1 部分完成，Pass 2 lint 已过、独立审校（§14）待做**。

## Pass 1 · 事实核查（断言四级制 + 引用逐条核验）

引用核验（本会话用 `git show v0.13.1:<path>` 逐条比对，✅ 者已确认支撑断言）：
- executor_settings_base.h:62(ActivationDataType)/64-73(FLOAT32/16、INT16/8) ✅
- litertlm_header_schema.fbs:56(KeyValuePair) ✅
- litertlm_section.h:98(FileBacked)/189(ProtoBuf)/252(Zlib) ✅
- litertlm_read.h:116(ReadHeaderFromLiteRTLM)/151(按段读，注释含 mmapped buffer) ✅
- c/engine.h:295(parallel_file_section_loading) ✅
- lora.h:40(LoRA)；lora_manager.h:39(LoraManager)/57(LoadLoRA) ✅

- 【常识】int4/带宽/内存账为教科书级+第 1 章已建立的推论，直接陈述 ✅
- 未展开量化内部（分组/scale/校准）——正文明说"未臆测未核验的细节"，只到"权重压 4bit + 激活精度谱系"层面 ✅
- 冷启动收益（mmap/并行/编译缓存）标注待基准 D，未虚构数字 ✅

## Pass 2 · 除 AI 味

- [x] `scripts/lint_prose.sh chapters/ch07-model-shape/chapter.md` —— 0 命中（2026-07-05）
- [x] 除 AI 味清单第 1-13 条：撰写时自查
- [x] 第 14 条 独立审校：**已由独立审校会话执行（2026-07-05）**，判为"AI 味很淡、可放行"；升华式收尾/自夸形容词/括号内同义重述等问题已逐条修正，修后 lint 复跑 0 命中

## 待清零（补读/数据）
- [ ] 实测（int4 vs int8 三角、并行加载 on/off 冷启动、litertlm_print 实剖）待基准 D
- [ ] 图 7-2（mmap/并行加载）、表 7-1（量化三角）待补（本轮出签名图 7-1）

## 验收自问
- [x] 使命兑现（压小/装箱/变体三条链路讲清；结清第 15、16 问）
- [x] 图 7-1 已落地并编号
- [ ] 实验可复现（脚本随基准 D）
- [x] 与 BOOK_PLAN 章卡一致；交叉引用规范；承接第 6 章、预告第 8 章
