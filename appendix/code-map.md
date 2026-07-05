# 附录 B · 代码地图

> LiteRT-LM 源码（`@ v0.13.1`）与本书章节的对照，方便你从"想读哪块代码"反查"书里哪章讲它"，或反过来。
> 核对代码建议在独立 worktree 检出该 tag：`git worktree add ../LiteRT-LM-v0.13.1 v0.13.1`。

## 按目录

| 源码目录 / 关键文件 | 作用 | 主要章节 |
|---|---|---|
| `runtime/engine/engine.h` | Engine / Session 对外接口 | 第 3 章 |
| `runtime/engine/engine_settings.*`、`engine_factory.h` | 引擎配置、后端自注册工厂 | 第 3、8 章 |
| `runtime/engine/litert_lm_main.cc` | CLI 演示程序 `litert_lm_main` | 第 2 章 |
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

> 一份更细、可点击的模块导览，见本书伴生的离线学习站点（`litert-lm-guide/`）。
