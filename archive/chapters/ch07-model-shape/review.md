# 第 7 章审校记录

> 当前记录：2026-07-18，v4.8 独立审校；2026-07-18 完成扩写事实核查与部署核验段独立审校。适用对象为当前第 7 章正文与图 7-1 至图 7-8，源码基线为 LiteRT-LM v0.13.1。

## 审校范围与当前结论

- [x] 独立交叉审校：扩写由独立会话完成，主会话复核了新增技术链、容量案例、图表顺序和练习体例；部署核验段再由未参与撰写的会话复核，图号按正文出现顺序排列为 7-1 至 7-7。

- [x] 语言：压缩重复叙述，删除未测量收益和含混的内存口径。
- [x] 禁词与术语：检查禁词、退役叙事标签、mmap、LoRA 与 cache 术语。
- [x] 严谨度：核对量化条件、文件 schema、当前 loader、平台映射、缓存标识与 LoRA 生命周期。
- [x] 叙事姿态：删除量化必然加速、加载必然提速和资源始终只占一份等绝对表述。
- [x] 源码锚点语义：检查引用位置是否真正属于当前 Engine 主路径，避免把辅助函数当成默认路径。
- [x] 扩写事实核查：核对量化支持链、builder 两阶段写入、模型段约束、cache 接入与 LoRA signature 行为。
- [x] 案例复算：2B 参数题设的 payload、scale、容器填充、KV cache 与运行预算均可由正文公式复算。
- [x] 扩写后的独立文风审校：未参与本轮撰写的会话按 `humanizer-cn` 与本仓规范复核，删除编辑提示语并拆分 LoRA 长句，未改动技术断言或锚点。

本文件不记录整书 build 或 PDF 已通过；两项由主会话最终验收。

## 本轮已纠正的关键事实

- fp16 到 int4 的 4 倍只是理想权重 payload 比例；scale、zero point、padding、文件头与 tokenizer 均未包含，吞吐和质量也不能按位宽比例推出。
- 当前 Engine 主加载路径经 `BuildLiteRtCompiledModelResources`、`BuildModelResourcesFromLitertLmFormat` 与 `LitertLmLoader`；`litertlm_read.cc` 的 TFLite 辅助重载不是该主路径。
- schema 的 required 字段不等于读取器已经执行完整 verifier；整文件映射与分段映射的行为不同，`madvise` 也只是平台相关建议。
- `parallel_file_section_loading` 重叠的是 tokenizer 创建与后续模型加载，不是所有 section 并行读盘。
- weight cache 标识来自秒级 mtime 与文件大小，并按 path 在进程内缓存，不是内容哈希。
- LoRA ID 首次使用后保留在 `loras_` 中，公开接口没有逐 ID 卸载；多个已用 ID 的数据与后端 buffer 可能累积到 manager 析构。

## 当前仍存在的实验缺口

- 缺少同一模型 int4/int8 的速度、内存与质量对照。
- 缺少 cache 开关加外部墙钟的冷/热受控实验，也未隔离测量并行加载开关。
- mmap 与平台页建议的实际页调入效果、多个 LoRA ID 的长期内存曲线尚未测量。
- 未取得各目标 GPU 的低比特 kernel 清单，也没有 delegate 节点落点与权重转换后缓冲大小的受控记录。

## 2026-07-18 扩写核查记录

- 量化支持链：`runtime/executor/llm_executor_settings.h:197-238`、`runtime/executor/llm_executor_settings_utils.cc:164-255`。
- 容器写入与属性：`python/litert_lm_builder/litertlm_core.py:24-31`、`python/litert_lm_builder/litertlm_builder.py:434-505`、`python/litert_lm_builder/litertlm_builder.py:671-730`。
- loader 与设置校验：`runtime/util/litert_lm_loader.cc:71-117`、`runtime/util/litert_lm_loader.cc:196-301`、`runtime/engine/engine_settings.cc:73-136`。
- 外挂权重与编译：`runtime/executor/llm_litert_compiled_model_executor.cc:1624-1652`。
- cache 接入与失效：`runtime/executor/litert_compiled_model_executor_utils.cc:460-527`、`runtime/util/file_util.cc:97-239`。
- LoRA 格式与兼容性：`runtime/util/lora_data.cc:41-247`、`runtime/util/lora_util.cc:33-80`、`runtime/components/lora.cc:64-123`、`runtime/components/lora_manager.cc:46-79`。

## 历史记录

- `2026-07-05 · 初稿验收`：旧文件记录了量化、加载和 LoRA 的早期事实核查与实验待办。
- 旧记录中的收益判断和图表待办已由 v4.8 改写；当前实验缺口以上一节为准。

## 2026-07-18 部署核验段独立审校

- 容器边界：确认主 loader 只显式拒绝 `begin_offset > end_offset`，未统一检查零长度、文件末尾、头部重叠、section 重叠、重复 `BufferKey` 与 16 KiB 对齐。正文已把 FlatBuffer verifier 和全局范围检查写成发布前的应用要求，没有写成 v0.13.1 现有能力。
- 容器案例：3.66 GB Gemma 4 E4B 数据来自 16 KiB 边界扫描与 TFLite FlatBuffer 静态解析。正文已改称 10 个 TFLite payload，并说明它们不是 `litertlm_print` 导出的 section 目录。
- cache 路径：核对路径模式、scoped-file 模式和 `:nocache` 的优先级。CPU 接收 XNNPACK path 或 fd；GPU 路径模式接收 serialization dir 与 model cache key，实际 weight cache 文件由下层决定。正文没有把 GPU 候选路径写成已确定的下层文件名。
- 发布建议：不可变文件名、独立 cache 目录、新进程、路由切换与回滚窗口均为应用层方案。正文已删除“LiteRT-LM 天然提供原子发布或 cache 兼容性保证”的含义，并保留下层可能另行校验 cache 的不确定性。
- LoRA 静态解析：`experiments/data/ch07_lora_capacity.md` 记录了 flatc 解析方法与统计口径。基座的 280 个输入按 shape 合计 28.4375 MiB；适配器的 220 个 tensor 合计 23.75 MiB；缺少的 60 个 key/value 输入合计 4.6875 MiB；8 个同形状适配器合计 227.5 MiB。以上均为逻辑 payload，不是 RSS、GPU 内存或任意后端的 `PackedSize()` 实测。
- LoRA 接入：文本 `ScopedLoraFile` 在当前 `ResourceManager` 路径返回不支持；可完整追踪的登记、切换和执行接入位于音频编码器。`UseLoRA(std::nullopt)` 不清除输入 map，也不是逐 ID 卸载。
- 图表：图 7-5 明确 verifier 属于完整校验流程；图 7-6 将跨进程切换标为应用实现；图 7-7 将物化对象限定为 LoRA 输入 buffer，并说明只有切换到已经物化的 ID 才只是更新当前选择。
- 文风：按 `book-review` 与 `humanizer-cn` 检查长句、否定式排比、加粗列表骨架、破折号和模糊结尾。拆分部署节开头长句，修正两处行内公式转义，未改变公式或源码结论。
- 检查结果：`bash scripts/lint_prose.sh chapters/ch07-model-shape/chapter.md` 0 命中；图 7-5 至图 7-7 通过 `xmllint --noout`；限定范围的 `git diff --check` 通过。

仍存实验缺口：没有 N/P/W cache 外部墙钟对照，也没有 8 个 LoRA 依次物化后的 `PackedSize()`、RSS、private footprint 或设备内存曲线；文本生成热切换在 v0.13.1 当前资源管理路径中不可用。

补充复核：发布清单示例已标明应用层属性；双版本预算将 7.32 GB 存储与 Engine 内存分开，`M_peak` 按时间窗口取最大值并单列初始化临时量，图 7-8 延长双 Engine 轨道至 P41 排空，顺序重启仅降低 Engine 上限而不降低两代产物的存储峰值；同进程销毁 Engine 不会清除静态路径标识 map。限定 lint、图 7-8 `xmllint` 与 `git diff --check` 均通过。

证据包复核：目录树与表 7-14 均明确属于应用层归档模板；表头改为“可识别或排除的问题”，避免把配置快照等记录夸大为自动排除故障的 LiteRT-LM 能力。
表 7-15 已明确为应用发布系统的治理模板，而非 LiteRT-LM 能力；`BLOCK` 覆盖静态检查，`WAIVED` 不等同于 `PASS`，示例与本章已记录的证据缺口一致。
