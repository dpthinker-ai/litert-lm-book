# 附录 B · 代码地图

> 本附录给出 LiteRT-LM 源码（`v0.17.0`）与本书章节的对应关系，可按源码目录定位相关章节，也可从章节定位实现文件。
> 核对代码建议在独立 worktree 检出该 tag：`git worktree add ../LiteRT-LM-v0.17.0 v0.17.0`。

## 按目录

| 源码目录 / 关键文件 | 作用 | 主要章节 |
|---|---|---|
| `runtime/engine/engine.h` | Engine / Session 对外接口 | 第 3 章 |
| `runtime/engine/engine_settings.*`、`engine_factory.h` | 引擎配置、后端自注册工厂 | 第 3、8、10 章 |
| `runtime/engine/litert_lm_main.cc`、`litert_lm_lib.cc` | CLI 演示程序、引擎装配（后端配置分支） | 第 2、8 章 |
| `runtime/engine/cpu_affinity_utils.*` | CPU 亲和性（线程绑定到性能核） | 第 8 章 |
| `runtime/conversation/` | 多轮对话、聊天模板、单轮渲染与全历史后缀提取、model_data_processor | 第 3 章 |
| `runtime/conversation/io_types.h` | Message(JSON)、Preface、Channel | 第 3、6、11 章 |
| `runtime/core/tasks.cc` | Prefill / Decode / DecodeOneStep / ShouldStop 算法 | 第 4、5 章 |
| `runtime/core/session_advanced.*` | 会话状态机、Clone / Checkpoint / Rewind | 第 3、6 章 |
| `runtime/executor/llm_executor_base.h` | Executor 抽象（Prefill/Decode/DecodeLogits） | 第 4、5 章 |
| `runtime/executor/llm_litert_compiled_model_executor.*` | 核心执行器、KV cache 双缓冲、设备侧采样 | 第 4、6、8 章 |
| `runtime/executor/llm_litert_compiled_model_executor_factory.cc` | 按 Backend 分派创建 | 第 8 章 |
| `runtime/executor/state_interface.h`、`runtime/executor/litert/state.*` | 执行器状态接口、KV/线性状态、原地与双缓冲、快照 | 第 6 章 |
| `runtime/executor/llm_litert_mtp_drafter.*` | MTP 投机解码 drafter / verify / 接受循环 | 第 9 章 |
| `runtime/executor/llm_litert_npu_compiled_model_executor.*` | NPU 执行器（QNN、embedder 子模型、KV 快照与恢复） | 第 6、8 章 |
| `runtime/executor/vision_litert_compiled_model_executor.*`、`audio_litert_compiled_model_executor.*` | 视觉/音频编码器 → embedding | 第 11 章 |
| `runtime/executor/npu/llm_litert_npu_embedder.*` | NPU 专用路径的 CPU embedding 执行 | 第 8 章 |
| `support/util/io_types.h`、`runtime/conversation/model_data_processor/multimodal_processor_helper.cc` | 输入数据表示与多模态预处理组装 | 第 3、11、12 章 |
| `runtime/executor/executor_settings_base.h` | Backend、ActivationDataType 枚举 | 第 7、8 章 |
| `support/tokenizer/` | SentencePiece/HuggingFace 分词与缓冲式流式反分词 | 第 3 章 |
| `runtime/components/sampler.h`、`top_p_cpu_sampler.*`、`sampling_cpu_util.*` | 采样抽象、top-p 包装与 CPU top-k/top-p/温度实现 | 第 5 章 |
| `runtime/components/stop_token_detector.*` | 停止序列检测（前缀匹配期间暂存，失配后释放） | 第 5 章 |
| `runtime/components/constrained_decoding/` | 约束解码（ProcessLogits / llguidance / LogitMask） | 第 11 章 |
| `runtime/components/tool_use/`（含 `antlr/*.g4`） | 工具调用格式化与解析 | 第 11 章 |
| `runtime/components/lora.*`、`lora_manager.*` | LoRA 适配器 | 第 7 章 |
| `support/preprocessor/` | 图像预处理、patchify | 第 11 章 |
| `runtime/framework/threadpool.*`、`execution_queue.*` | 线程池、异步任务队列 | 第 4、8 章 |
| `runtime/framework/resource_management/threaded_execution_manager.*` | 异步执行管理器（任务依赖链、会话克隆任务） | 第 4、6 章 |
| `runtime/framework/resource_management/resource_manager.*`、`context_handler/` | 会话上下文共享、写时分离与执行器状态切换 | 第 6 章 |
| `support/util/memory_mapped_file.*`、`runtime/util/litert_lm_loader.*`、`runtime/util/lora_data.h` | 跨平台 mmap、段加载器、LoRA 数据视图 | 第 7 章 |
| `runtime/executor/llm_executor_settings.h`、`llm_executor_io_types.h`、`magic_number_configs_helper.*` | 执行器配置（线程数/KV 增量/取消开关）、IO 类型与 magic number 占位宽度替换 | 第 4、6、8 章 |
| `schema/core/`（`litertlm_header_schema.fbs`、`litertlm_read.*`、`litertlm_print.*`） | `.litertlm` 文件格式与读取 | 第 2、7 章 |
| `schema/capabilities/capabilities.*` | 根据模型分段类型判断是否支持投机解码 | 第 9 章 |
| `c/engine.{h,cc}`、`c/engine_internal.h`、`c/conversation.{h,cc}`、`c/conversation_internal.h` | C ABI、不透明句柄、结构化流式回调 | 第 12 章 |
| `python/`、`kotlin/`、`swift/`、`js/` | 各语言绑定 | 第 12 章 |
| `runtime/executor/fake_llm_executor.h` | 测试用假执行器 | 第 12 章 |
| `WORKSPACE`、`BUILD`、`CMakeLists.txt`、`cmake/` | Bazel / CMake 双构建 | 附录 C |

## 按主题查找入口

下表按任务组织读码入口。文件与符号均已在 v0.17.0 中核实。

| 主题 | 文件与符号 |
|---|---|
| 一次生成的完整链路 | `runtime/engine/engine.h` → `runtime/core/session_advanced.cc` → `runtime/core/tasks.cc` → `runtime/executor/llm_litert_compiled_model_executor.cc`；依次查看 Session、prefill、decode 与执行器调用 |
| KV cache 双缓冲与异步 prefill | `runtime/executor/litert/state.cc`：`PrepareForModelInvocation`、`DeepCopy`；再查 `runtime/executor/llm_litert_compiled_model_executor.cc` 的 `prefill_chunk_size_`、`RunAsync` |
| 投机解码 | `runtime/executor/llm_litert_mtp_drafter.cc`：`RunDraftingLoop`、`RunVerification`、`num_drafted_tokens_`、`num_verified_tokens_` |
| 对话文本增量生成 | `runtime/conversation/conversation.cc`：`Conversation::GetSingleTurnText`、`old_string`、`new_string` 及相邻的前缀检查与后缀提取 |
| 采样策略 | `runtime/components/sampler.h`、`runtime/components/sampling_cpu_util.cc`、`runtime/components/top_p_cpu_sampler.cc`；内外采样分支见 `runtime/core/tasks.cc` 的 `DecodeAndSample` |
| 约束解码 | 从 `runtime/core/tasks.cc` 的 `ProcessLogits` 进入 `runtime/components/constrained_decoding/`，再查看 `llg_constraint` FFI |
| 停止序列的部分匹配 | `runtime/components/stop_token_detector.h`、`runtime/components/stop_token_detector.cc`；`runtime/core/tasks.cc` 中的 `StopTokenStreamFilter`，再看 `support/tokenizer/buffered_streaming_detokenizer.cc` |
| 多模态 embedding 注入 | `runtime/components/embedding_lookup/embedding_lookup_manager.cc` → `runtime/components/embedding_lookup/embedding_lookup_multi_modal.cc`；再对照 vision/audio executor 的 `Encode` |
| 冷启动与 mmap | `support/util/memory_mapped_file_posix.cc`、`runtime/util/litert_lm_loader.cc`：段缓存、双检锁、对齐补偿 |
| 扩展采样策略、后端或绑定语言 | `runtime/components/sampler.h`、`runtime/executor/llm_executor_base.h`、`runtime/executor/llm_litert_compiled_model_executor_factory.cc`、`c/engine.h` |

## MoE 依赖代码入口

第 10 章的运行时实现位于 LiteRT 依赖，冻结提交与导出端范围见 10.6 节。以下 `LiteRT/` 路径相对于独立 LiteRT checkout，不相对于 LiteRT-LM。

| 入口 | 用途 |
|---|---|
| `LiteRT/tflite/delegates/xnnpack/moe_delegate_kernel.cc` | 专家支持检查、分派、当前专家权重展开与输出累加 |
| `LiteRT/tflite/delegates/xnnpack/xnnpack_delegate.cc` | 将符合条件的 MoE 节点交给专用 kernel |
| `LiteRT/litert/runtime/compiled_model.cc` | CPU custom op 注册与占位实现 |
| `LiteRT/ml_drift_delegate/delegate/composite/moe_experts_parser.cc` | GPU 输入、布局及量化约束 |
| `LiteRT/ml_drift_delegate/delegate/composite/moe_experts_kernel.cc` | 按形状选择专家计算路径及加权合并 |
| `LiteRT/ml_drift_delegate/delegate/composite/experts_remap_builder.cc` | 专家重排缓冲及矩阵实现选择 |

公开 Artisan 产物的引擎选择发生在 LiteRT-LM 层。`runtime/engine/engine_settings.cc:181` 检查文本模型类型，`runtime/engine/engine_factory.h:202` 列出 GPU_ARTISAN 的两个 Legacy 候选。工厂在 `runtime/engine/engine_factory.h:88` 按注册情况选择引擎；无可用实现时，错误会列出候选和实际注册类型。候选类型出现在映射表中，不等于当前发行包已经注册该实现。实际完整产物检查见附录 D 第二十一节。
