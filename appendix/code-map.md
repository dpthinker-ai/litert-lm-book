# 附录 B · 代码地图

> 本附录给出 LiteRT-LM 源码（`v0.13.1`）与本书章节的对应关系，可按源码目录定位相关章节，也可从章节定位实现文件。
> 核对代码建议在独立 worktree 检出该 tag：`git worktree add ../LiteRT-LM-v0.13.1 v0.13.1`。

## 按目录

| 源码目录 / 关键文件 | 作用 | 主要章节 |
|---|---|---|
| `runtime/engine/engine.h` | Engine / Session 对外接口 | 第 3 章 |
| `runtime/engine/engine_settings.*`、`engine_factory.h` | 引擎配置、后端自注册工厂 | 第 3、8 章 |
| `runtime/engine/litert_lm_main.cc`、`litert_lm_lib.cc` | CLI 演示程序、引擎装配（后端配置分支） | 第 2、8 章 |
| `runtime/engine/cpu_affinity_utils.*` | CPU 亲和性（线程绑定到性能核） | 第 8 章 |
| `runtime/conversation/` | 多轮对话、聊天模板、单轮渲染与全历史后缀提取、model_data_processor | 第 3 章 |
| `runtime/conversation/io_types.h` | Message(JSON)、Preface、Channel | 第 3、6、10 章 |
| `runtime/core/tasks.cc` | Prefill / Decode / DecodeOneStep / ShouldStop 算法 | 第 4、5 章 |
| `runtime/core/session_advanced.*` | 会话状态机、Clone / Checkpoint / Rewind | 第 3、6 章 |
| `runtime/executor/llm_executor_base.h` | Executor 抽象（Prefill/Decode/DecodeLogits） | 第 4、5 章 |
| `runtime/executor/llm_litert_compiled_model_executor.*` | 核心执行器、KV cache 双缓冲、设备侧采样 | 第 4、6、8 章 |
| `runtime/executor/llm_litert_compiled_model_executor_factory.cc` | 按 Backend 分派创建 | 第 8 章 |
| `runtime/executor/kv_cache_interface.h`、`litert/kv_cache.*` | KV cache 接口抽象及其唯一实现（v0.13.1 无执行器调用；`Serialize`/`Load` 未实现） | 第 6 章 |
| `runtime/executor/llm_litert_mtp_drafter.*` | MTP 投机解码 drafter / verify / 接受循环 | 第 9 章 |
| `runtime/executor/llm_litert_npu_compiled_model_executor.*` | NPU 执行器（QNN、embedder 子模型、KV 快照与恢复） | 第 6、8 章 |
| `runtime/executor/vision_litert_compiled_model_executor.*`、`audio_litert_compiled_model_executor.*` | 视觉/音频编码器 → embedding | 第 10 章 |
| `runtime/executor/executor_settings_base.h` | Backend、ActivationDataType 枚举 | 第 7、8 章 |
| `runtime/components/tokenizer.h`、`sentencepiece_/huggingface_tokenizer.*` | 分词 | 第 3 章 |
| `runtime/components/sampler.h`、`top_p_cpu_sampler.*`、`sampling_cpu_util.*` | 采样抽象、top-p 包装与 CPU top-k/top-p/温度实现 | 第 5 章 |
| `runtime/components/stop_token_detector.*` | 停止序列检测（前缀匹配期间暂存，失配后释放） | 第 5 章 |
| `runtime/components/constrained_decoding/` | 约束解码（MaskLogits / llguidance / Bitmap） | 第 10 章 |
| `runtime/components/tool_use/`（含 `antlr/*.g4`） | 工具调用格式化与解析 | 第 10 章 |
| `runtime/components/lora.*`、`lora_manager.*` | LoRA 适配器 | 第 7 章 |
| `runtime/components/preprocessor/` | 图像预处理、patchify | 第 10 章 |
| `runtime/framework/threadpool.*`、`execution_queue.*` | 线程池、异步任务队列 | 第 4、8 章 |
| `runtime/framework/resource_management/threaded_execution_manager.*` | 异步执行管理器（任务依赖链、会话克隆任务） | 第 4、6 章 |
| `runtime/framework/resource_management/resource_manager.*`、`context_handler/` | 会话上下文共享、写时分离与执行器状态切换 | 第 6 章 |
| `runtime/util/memory_mapped_file.*`、`litert_lm_loader.*`、`lora_data.h` | 跨平台 mmap、段加载器、LoRA 数据视图 | 第 7 章 |
| `runtime/executor/llm_executor_settings.h`、`llm_executor_io_types.h`、`magic_number_configs_helper.*` | 执行器配置（线程数/KV 增量/取消开关）、IO 类型与 magic number 占位宽度替换 | 第 4、6、8 章 |
| `schema/core/`（`litertlm_header_schema.fbs`、`litertlm_read.*`、`litertlm_print.*`） | `.litertlm` 文件格式与读取 | 第 2、7 章 |
| `schema/capabilities/speculative_decoding.*` | 根据模型分段类型判断是否支持投机解码 | 第 9 章 |
| `c/engine.h`、`c/engine.cc` | C ABI（不透明句柄 + C 函数） | 第 11 章 |
| `python/`、`kotlin/`、`swift/`、`js/` | 各语言绑定 | 第 11 章 |
| `runtime/executor/fake_llm_executor.h` | 测试用假执行器 | 第 11 章 |
| `WORKSPACE`、`BUILD`、`CMakeLists.txt`、`cmake/` | Bazel / CMake 双构建 | 附录 C |

## 按主题查找入口

下表按任务组织读码入口。文件与符号均已在 v0.13.1 中核实。

| 主题 | 文件与符号 |
|---|---|
| 一次生成的完整链路 | `runtime/engine/engine.h` → `runtime/core/session_advanced.cc` → `runtime/core/tasks.cc` → `runtime/executor/llm_litert_compiled_model_executor.cc`；依次查看 Session、prefill、decode 与执行器调用 |
| KV cache 双缓冲与异步 prefill | `runtime/executor/llm_litert_compiled_model_executor.cc`：`input_kv_cache_buffers_`、`std::swap`、`prefill_chunk_size_`、`RunAsync` |
| 投机解码 | `runtime/executor/llm_litert_mtp_drafter.cc`：`RunDraftingLoop`、`RunVerification`、`num_drafted_tokens_`、`num_verified_tokens_` |
| 对话文本增量生成 | `runtime/conversation/conversation.cc`：`Conversation::GetSingleTurnText`、`old_string`、`new_string` 及相邻的前缀检查与后缀提取 |
| 采样策略 | `runtime/components/sampler.h`、`runtime/components/sampling_cpu_util.cc`、`runtime/components/top_p_cpu_sampler.cc`；内外采样分支见 `runtime/core/tasks.cc` 的 `DecodeAndSample` |
| 约束解码 | 从 `runtime/core/tasks.cc` 的 `MaskLogits` 进入 `runtime/components/constrained_decoding/`，再查看 `llg_constraint` FFI |
| 停止序列的部分匹配 | `runtime/components/stop_token_detector.h`、`runtime/components/stop_token_detector.cc`；`runtime/core/tasks.cc` 中的 `bpe_partial_token_ids_`、`pending_stop_tokens_` |
| 多模态 embedding 注入 | `runtime/components/embedding_lookup/embedding_lookup_manager.cc` → `runtime/components/embedding_lookup/embedding_lookup_multi_modal.cc`；再对照 vision/audio executor 的 `Encode` |
| 冷启动与 mmap | `runtime/util/memory_mapped_file_posix.cc`、`runtime/util/litert_lm_loader.cc`：段缓存、双检锁、对齐补偿 |
| 扩展采样策略、后端或绑定语言 | `runtime/components/sampler.h`、`runtime/executor/llm_executor_base.h`、`runtime/executor/llm_litert_compiled_model_executor_factory.cc`、`c/engine.h` |
