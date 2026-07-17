# 附录 B · 代码地图

> LiteRT-LM 源码（`v0.13.1`）与本书章节的对照，方便你从"想读哪块代码"反查"书里哪章讲它"，或反过来。
> 核对代码建议在独立 worktree 检出该 tag：`git worktree add ../LiteRT-LM-v0.13.1 v0.13.1`。

## 按目录

| 源码目录 / 关键文件 | 作用 | 主要章节 |
|---|---|---|
| `runtime/engine/engine.h` | Engine / Session 对外接口 | 第 3 章 |
| `runtime/engine/engine_settings.*`、`engine_factory.h` | 引擎配置、后端自注册工厂 | 第 3、8 章 |
| `runtime/engine/litert_lm_main.cc`、`litert_lm_lib.cc` | CLI 演示程序、引擎装配（后端配置分支） | 第 2、8 章 |
| `runtime/engine/cpu_affinity_utils.*` | CPU 亲和性（绑性能核） | 第 8 章 |
| `runtime/conversation/` | 多轮对话、聊天模板、模板 diff、model_data_processor | 第 3 章 |
| `runtime/conversation/io_types.h` | Message(JSON)、Preface、Channel | 第 3、6、10 章 |
| `runtime/core/tasks.cc` | Prefill / Decode / DecodeOneStep / ShouldStop 算法 | 第 4、5 章 |
| `runtime/core/session_advanced.*` | 会话状态机、Clone / Checkpoint / Rewind | 第 3、6 章 |
| `runtime/executor/llm_executor_base.h` | Executor 抽象（Prefill/Decode/DecodeLogits） | 第 4、5 章 |
| `runtime/executor/llm_litert_compiled_model_executor.*` | 核心执行器、KV cache 双缓冲、片上采样 | 第 4、6、8 章 |
| `runtime/executor/llm_litert_compiled_model_executor_factory.cc` | 按 Backend 分派创建 | 第 8 章 |
| `runtime/executor/kv_cache_interface.h` | KV cache 搬运接口（Serialize/DeepCopy/…） | 第 6 章 |
| `runtime/executor/llm_litert_mtp_drafter.*` | MTP 推测解码 drafter / verify / 接受循环 | 第 9 章 |
| `runtime/executor/llm_litert_npu_compiled_model_executor.h` | NPU 执行器（QNN、embedder 子模型） | 第 8 章 |
| `runtime/executor/vision_/audio_litert_compiled_model_executor.*` | 视觉/音频编码器 → embedding | 第 10 章 |
| `runtime/executor/executor_settings_base.h` | Backend、ActivationDataType 枚举 | 第 7、8 章 |
| `runtime/components/tokenizer.h`、`sentencepiece_/huggingface_tokenizer.*` | 分词 | 第 3 章 |
| `runtime/components/sampler.h`、`top_p_cpu_sampler.*` | 采样策略 | 第 5 章 |
| `runtime/components/stop_token_detector.*` | 停止符检测（部分匹配回吐） | 第 5 章 |
| `runtime/components/constrained_decoding/` | 约束解码（MaskLogits / llguidance / Bitmap） | 第 10 章 |
| `runtime/components/tool_use/`（含 `antlr/*.g4`） | 工具调用格式化与解析 | 第 10 章 |
| `runtime/components/lora.*`、`lora_manager.*` | LoRA 适配器 | 第 7 章 |
| `runtime/components/preprocessor/` | 图像预处理、patchify | 第 10 章 |
| `runtime/framework/threadpool.*`、`execution_queue.*` | 线程池、异步任务队列 | 第 4、8 章 |
| `runtime/framework/threaded_execution_manager.*` | 异步执行管理器（任务依赖链） | 第 4 章 |
| `runtime/util/memory_mapped_file.*`、`litert_lm_loader.*`、`lora_data.h` | 跨平台 mmap、段加载器、LoRA 数据视图 | 第 7 章 |
| `runtime/executor/llm_executor_settings.h`、`llm_executor_io_types.h` | 执行器配置（线程数/KV 增量/取消开关）与 IO 类型 | 第 4、6、8 章 |
| `runtime/components/sampling_cpu_util.*` | CPU 采样实现（top-k/top-p/温度） | 第 5 章 |
| `schema/core/`（`litertlm_header_schema.fbs`、`litertlm_read.*`、`litertlm_print.*`） | `.litertlm` 文件格式与读取 | 第 2、7 章 |
| `schema/capabilities/speculative_decoding.*` | 推测解码能力声明 | 第 9 章 |
| `c/engine.h`、`c/engine.cc` | C ABI（不透明句柄 + C 函数） | 第 11 章 |
| `python/`、`kotlin/`、`swift/`、`js/` | 各语言绑定 | 第 11 章 |
| `runtime/executor/fake_llm_executor.h` | 测试用假执行器 | 第 11 章 |
| `WORKSPACE`、`BUILD`、`CMakeLists.txt`、`cmake/` | Bazel / CMake 双构建 | 附录 C |

## 按"想理解 X 从哪读起"

- **一次生成的完整链路**：`engine.h` → `session_advanced.cc` → `tasks.cc`（Prefill 然后 Decode）→ `llm_litert_compiled_model_executor.cc`。
- **性能为什么是这样**：`kv_cache_interface.h` 与执行器的双缓冲成员、`llm_litert_mtp_drafter.cc`、`executor_settings_base.h` 的 Backend。
- **想加一个后端**：看 `..._factory.cc` 的 `switch(GetBackend())` 与 `LlmExecutor` 抽象。

下面是几条 grep 式读码路径（符号均在 v0.13.1 核实），照着搜就能落到机制现场：

- **KV cache 双缓冲与异步 prefill**：在 `llm_litert_compiled_model_executor.cc` 搜 `input_kv_cache_buffers_`（成员声明与注释）、`std::swap`（prefill/decode 两处交换）、`prefill_chunk_size_`、`RunAsync`。
- **推测解码**：`llm_litert_mtp_drafter.cc` 的 `RunDraftingLoop` / `RunVerification`，配接受率计数器 `num_drafted_tokens_` / `num_verified_tokens_`。
- **模板 diff 增量渲染**：`conversation.cc` 搜 `old_string` / `new_string`，相减逻辑在两者相邻处。
- **采样策略**：`sampling_cpu_util.cc`（top-k/top-p/温度的 CPU 实现）与 `top_p_cpu_sampler.cc`；内外采样分岔在 `tasks.cc` 的 `DecodeAndSample`。
- **约束解码**：`tasks.cc` 的 `MaskLogits` 调用点 → `components/constrained_decoding/` 的掩码实现与 llg_constraint FFI。
- **停止词部分匹配**：`stop_token_detector.h/.cc`，配 `tasks.cc` 里 `bpe_partial_token_ids_` / `pending_stop_tokens_` 两个队列。
- **多模态 embedding 注入**：`llm_executor_base.h` 的 `FillVisionEmbeddings` 接口 → vision/audio executor 的 `Encode` 实现。
- **冷启动与 mmap**：`runtime/util/memory_mapped_file_posix.cc`（Linux/macOS 实现）与 `litert_lm_loader.cc` 的段缓存、双检锁、对齐补偿。
- **想加一个采样策略/后端/绑定语言**：分别看 `sampler.h` 抽象、`..._factory.cc` 的 `switch(GetBackend())`、`c/engine.h` 的句柄配对。

