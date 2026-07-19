# 第 2 章 从运行到架构：benchmark、Roofline 与五层视图

> 本章目标：运行 LiteRT-LM，按源码定义解释 benchmark 的四项指标，并建立后续章节使用的五层架构视图。

第 1 章在明确的假设下推导了 decode 的带宽侧上限。本章把同一分析方法用于实测数据。第 1 章的 25 tokens/s 是示意点值，不能直接用来验证另一台设备。

## 运行命令行工具

使用官方 Python 包就无需本地编译 C++。v0.13.1 的 README 给出以下安装与运行方式。[^ch02-litertlm-readme]

```bash
uv tool install litert-lm
litert-lm run \
  --from-huggingface-repo=litert-community/gemma-4-E4B-it-litert-lm \
  gemma-4-E4B-it.litertlm \
  --prompt="What is the capital of France?"
```

`litert-lm` 的 CLI 入口使用 click，并注册八个子命令模块（`python/litert_lm_cli/main.py:52`）。本章使用 `run` 和 `benchmark`；同一入口还注册 `convert`、`list`、`import`、`delete`、`rename` 与 `serve`。

```python
_serve_module.register(cli)      // (1)
_convert_module.register(cli)
_list_module.register(cli)
_import_module.register(cli)     // (2)
_delete_module.register(cli)
_rename_module.register(cli)
_benchmark_module.register(cli)
_run_module.register(cli)
```

每个子命令由独立模块实现，再通过 `register(cli)` 注册到同一个 click group。`import` 是 Python 关键字，因此对应模块通过 `importlib.import_module` 加载（`python/litert_lm_cli/main.py:33`）。入口只负责注册，命令逻辑由各模块实现。

`run` 在解析模型与后端参数后创建 `Engine`，再用上下文管理器限定其生命周期（`python/litert_lm_cli/commands/run.py:240`）：

```python
      engine_cm = litert_lm.Engine(       // (1)
          model_obj.model_path,
          backend=backend_val,
          enable_speculative_decoding=enable_speculative_decoding,
          max_num_tokens=max_num_tokens,
          vision_backend=vision_backend_val,
          audio_backend=audio_backend_val,
          cache_dir=cache_dir_val,
      )

    with engine_cm as engine:             // (2)
```

(1) 把模型路径、后端与生成配置传入 `Engine`。(2) 进入上下文后，`run` 再调用 `engine.create_session`（`python/litert_lm_cli/commands/run.py:252`）。`SessionInterface` 的注释说明，Session 保存每次独立交互的内部状态（`runtime/engine/engine.h:65`）。Python `Engine.__exit__` 调用 `close`（`python/litert_lm/engine.py:137`），后者删除原生 Engine 句柄（`python/litert_lm/engine.py:129`）。各后端何时释放设备资源仍由其实现决定。

指定 `--from-huggingface-repo` 时，`run` 调用 `common.download_from_huggingface`（`python/litert_lm_cli/commands/run.py:571`）。本书基准模型文件为 3.66 GB，下载前应检查磁盘空间；耗时取决于网络与缓存状态。模型就绪后，CLI 流式打印增量文本。

`run` 迭代 `send_message_async` 返回的 stream，并打印每个响应字典中的文本项（`python/litert_lm_cli/commands/run.py:101`）：

```python
  stream = conversation.send_message_async(prompt)
  try:
    for chunk in stream:                          // (1)
      content_list = chunk.get("content", [])
      for item in content_list:
        if item.get("type") == "text":
          if state.active_channel is not None:
            click.echo()
            state.active_channel = None
          click.echo(click.style(item.get("text", ""), fg="yellow"), nl=False)
```

(1) 每次迭代取得的是 Python 层生成的响应字典，不对应一个 token 或一次 decode step。`Conversation.send_message_async` 先把原生回调字符串解析为字典（`python/litert_lm/conversation.py:202`），随后按工具调用配置选择相应的 `yield` 分支（`python/litert_lm/conversation.py:218`、`python/litert_lm/conversation.py:220`）。当前 step 没有可输出文本时，C++ decode 循环也会跳过回调（`runtime/core/tasks.cc:533`）。因此，不能用 stream 迭代次数计算 token 数或吞吐；`nl=False` 只负责连续打印文本片段。

按下 Ctrl-C 不会同步终止后台生成。`run` 捕获 `KeyboardInterrupt` 后调用 `cancel_process()`，随后继续消费 stream，直至后台处理结束（`python/litert_lm_cli/commands/run.py:123`）：

```python
  except KeyboardInterrupt:
    conversation.cancel_process()
    for _ in stream:
      pass
```

运行时只在预设检查点读取取消标志。decode 在下一次循环迭代开始前读取该标志；若请求发生在当前 decode step 内，必须先等该 step 返回。v0.13.1 的 `Prefill` 函数没有接收取消标志（`runtime/core/tasks.cc:413`），正在执行的 prefill 不会轮询该请求。详细调用链见第 4 章。

从源码构建后，还可以运行 C++ 示例程序 `litert_lm_main`。完整构建命令见第 11 章与附录 C；这里先看两个基本参数：

- `--backend`：选执行后端，默认是 `gpu`（`runtime/engine/litert_lm_main.cc:52`）。想用 CPU 就传 `--backend=cpu`。
- `--model_path`：指向一个 `.litertlm` 模型文件（`runtime/engine/litert_lm_main.cc:54`）。

以下命令选择 CPU 后端：

```bash
litert_lm_main --backend=cpu --model_path=<你的模型>.litertlm
```

`MainHelper` 依次创建模型资源、引擎设置、Engine 和 Conversation，再异步提交消息（`runtime/engine/litert_lm_main.cc:113`）。下面只保留与调用链有关的语句，省略 Conversation 的装配细节：

```cpp
  ASSIGN_OR_RETURN(ModelAssets model_assets,  // NOLINT
                   ModelAssets::Create(model_path));
  auto backend_str = absl::GetFlag(FLAGS_backend);
  ASSIGN_OR_RETURN(Backend backend,
                   litert::lm::GetBackendFromString(backend_str));  // (1)
  ASSIGN_OR_RETURN(
      EngineSettings engine_settings,
      EngineSettings::CreateDefault(std::move(model_assets), backend));
  // Enable benchmark by default.
  engine_settings.GetMutableBenchmarkParams() =
      litert::lm::proto::BenchmarkParams();                         // (2)
  // ...
  ASSIGN_OR_RETURN(auto engine, litert::lm::EngineFactory::CreateDefault(
                                    std::move(engine_settings)));
  // ...
  RETURN_IF_ERROR(conversation->SendMessageAsync(                   // (3)
      json::object({{"role", "user"}, {"content", content_list}}),
      CreateMessageCallback()));
  RETURN_IF_ERROR(engine->WaitUntilDone(absl::Minutes(10)));
```

(1) 把 `--backend` 字符串解析为 `Backend` 枚举，再写入 `EngineSettings`；`EngineFactory` 接收这份设置。具体后端装配见第 8 章。(2) 为示例程序启用 benchmark 记录。(3) `SendMessageAsync` 提交异步生成，随后由 `WaitUntilDone` 等待任务结束。`CreateMessageCallback` 接收增量消息；回调收到空消息时，示例程序只输出换行（`runtime/engine/litert_lm_main.cc:76`）。

按主生成路径，输入先进入对话与 Session，再经过 prefill、decode，最后由回调返回结果。第 3 至 5 章分别分析状态管理、两阶段执行和输出处理。

## benchmark 输出的四项指标

`litert-lm benchmark` 以固定输入触发一次生成，并通过 benchmark 参数指定 prefill 与 decode 的 token 数。Python 包装层把输入文本设为 `benchmark`，再调用 C API 的同步生成函数（`python/litert_lm/benchmark.py:74`）。命令完成后输出四项指标（`python/litert_lm_cli/commands/benchmark.py:102`）：

```python
    result = benchmark_obj.run()                            // (1)

    click.echo("----- Results -----")
    click.echo(
        f"Prefill speed:        {result.last_prefill_tokens_per_second:.2f}"
        " tokens/s"
    )
    click.echo(
        f"Decode speed:         {result.last_decode_tokens_per_second:.2f}"
        " tokens/s"
    )
    click.echo(f"Init time:            {result.init_time_in_second:.4f} s")     // (2)
    click.echo(
        f"Time to first token:  {result.time_to_first_token_in_second:.4f} s"  // (3)
    )
```

(1) 返回 Python 的 `BenchmarkInfo`。其中字段名虽然带 `last_`，当前包装层实际读取 C API 的第 0 个 turn；本命令只触发一次生成，因此二者指向同一条记录（`python/litert_lm/benchmark.py:94`）。(2) 初始化时间单列，单位为秒。(3) TTFT 也以秒输出，而且不含初始化时间。下面是附录 D 主矩阵的一组中位数，条件为 Apple M5 Pro、Gemma 4 E4B、cpu 后端、prefill 256 token、decode 128 token。

```text
Backend                    : cpu
Number of tokens in prefill: 256
Number of tokens in decode : 128
----- Results -----
Prefill speed:        65.60 tokens/s
Decode speed:         24.80 tokens/s
Init time:            0.5400 s
Time to first token:  3.9400 s
```

四项数据都由 C++ `BenchmarkInfo` 提供，C API 分别暴露读取函数：

| 指标 | v0.13.1 中的定义 | 解释时需要检查的条件 | C API |
|---|---|---|---|
| TTFT（s） | 首轮 prefill 完整耗时，加首轮 decode 的平均单 token 耗时；不含初始化 | prompt 长度、prefill 形状、首轮 decode turn 的 token 数 | `litert_lm_benchmark_info_get_time_to_first_token`（`c/engine.h:583`） |
| prefill 吞吐（tokens/s） | 本 turn 的输入 token 数除以耗时 | 序列长度、固定形状填充率、后端与 kernel | `litert_lm_benchmark_info_get_prefill_tokens_per_sec_at`（`c/engine.h:634`） |
| decode 吞吐（tokens/s） | 本 turn 的生成 token 数除以耗时 | 上下文长度、KV cache、采样、后端与 kernel | `litert_lm_benchmark_info_get_decode_tokens_per_sec_at`（`c/engine.h:643`） |
| 初始化时间（s） | C API 将 `GetInitPhases()` 中各条 duration 相加；阶段可能重叠 | 文件缓存、模型映射、delegate 初始化与缓存；只作同口径对照 | `litert_lm_benchmark_info_get_total_init_time_in_second`（`c/engine.h:591`） |

> 表 2-1　benchmark 的四项指标及其源码定义。资源瓶颈不能只由指标名称预先指定，需要结合工作负载和后端验证。

初始化时间必须按实现口径解释。C API 遍历 `GetInitPhases()`，把每条 duration 换算为毫秒后求和，最后除以 1000 返回秒（`c/engine.cc:821`）。`EngineAdvancedImpl::Create` 先开始记录 `kTotal`，紧接着开始 `kModelAssets` 子阶段（`runtime/core/engine_advanced_impl.cc:180`）。后续还记录 `kLlmMetadata` 子阶段（`runtime/core/engine_advanced_impl.cc:193`）。因此，这些 duration 并非互斥区间，CLI 的 `Init time` 不能直接视为一条无重叠的端到端墙钟计时。附录 D 保留 API 原始口径；分析单个初始化步骤时，应读取各 phase 或另设外部墙钟计时。

v0.13.1 的 TTFT 是计算值，并非从请求发起直接计时至首个流式回调。cpu/256 档按同一批 turn 数据复算为 `256 ÷ 65.6 + 1 ÷ 24.8 ≈ 3.94 s`。这与表中 3.94 s 一致，是对指标定义和数据记录的内部一致性检查，不是一次独立测量。cpu/4096 档同理：`4096 ÷ 226.5 + 1 ÷ 20.7 ≈ 18.13 s`〔基准 D〕。

prefill 与 decode 的吞吐对应不同阶段，不能合并为单一吞吐值。同模型、同设备、同后端时，两者仍会受序列长度、固定 prefill signature 的填充率和 kernel 实现影响。附录 D 的主矩阵来自 Apple M5 Pro；Android 真机数据作为扩展实验单列。引用基准数据时，正文会同时给出设备、模型、后端与上下文等条件。

## 计时器与测量语义

`BenchmarkInfo` 保存各阶段的计时记录（`runtime/engine/io_types.h:420`）。源码把一次 `RunPrefill` 或 `RunDecode` 调用定义为一个 turn；每条 `BenchmarkTurnData` 包含持续时间和 token 数（`runtime/engine/io_types.h:409`）。prefill 的结束计时如下（`runtime/engine/io_types.cc:306`）：

```cpp
absl::Status BenchmarkInfo::TimePrefillTurnEnd(uint64_t num_prefill_tokens) {
  const std::string phase_name = absl::StrCat("prefill:", prefill_turn_index_);
  // ...
  prefill_turns_.emplace_back(num_prefill_tokens,
                              absl::Now() - start_time_map_[phase_name]);  // (1)
  prefill_turn_index_++;
  return absl::OkStatus();
}
```

(1) 将 `num_prefill_tokens` 与开始、结束时间之差写入 `prefill_turns_`。decode 在 `TimeDecodeTurnEnd` 中记录相同结构（`runtime/engine/io_types.cc:339`）。prefill 和 decode 吞吐都按 `num_tokens / duration` 计算，分别见 `runtime/engine/io_types.cc:395` 与 `runtime/engine/io_types.cc:434`。

TTFT 的实现位于 `runtime/engine/io_types.cc:455`：

```cpp
double first_decode_token_seconds = absl::ToDoubleSeconds(
    decode_turns_[0].duration / decode_turns_[0].num_tokens);   // (1)
double first_prefill_token_seconds =
    absl::ToDoubleSeconds(prefill_turns_[0].duration);          // (2)
return first_decode_token_seconds + first_prefill_token_seconds;
```

(1) 是首轮 decode turn 的平均单 token 耗时，(2) 是首轮 prefill 的完整耗时。函数返回二者之和。这里没有记录“首个流式回调发生时刻”；当首轮 decode turn 包含多个 token，且各 step 耗时不同时，这个平均值不等于首个回调的直接计时。C API 也明确以秒返回该值，并说明它不含初始化（`c/engine.h:574`）。

benchmark 模式还改变了两处执行条件。第一处要求 prefill 等待完成（`runtime/core/tasks.cc:435`）：

```cpp
// Wait for prefill to complete if benchmark mode is enabled.
params.SetWaitForCompletion(wait_for_completion | benchmark_info.has_value());
```

`SetWaitForCompletion` 使 benchmark 请求同步完成 prefill，避免只记录异步提交所需的 host 时间。具体后端如何落实等待，见第 4 章。第二处位于 `ShouldStop`：只要 benchmark 指定的 decode 步数大于 0，命中停止序列不会提前结束，循环在达到指定步数后停止（`runtime/core/tasks.cc:85`）。这样，各次吞吐记录使用相同的 decode step 数。

因此，benchmark 数字反映的是启用等待和固定 decode 步数后的测量路径，不能直接等同于真实对话的端到端时延。报告结果时，需要同时记录参数与执行模式。

解读一组结果时，可按以下顺序检查：

1. 对 Init，先区分冷缓存与热缓存，并核对模型文件、delegate 初始化和编译缓存；第 7 章讨论这些阶段。
2. 按源码公式把 TTFT 拆成首轮 prefill 耗时和首轮 decode 的平均单 token 耗时。只有前者占主要比例时，才把后续分析集中到 prefill。
3. 对 prefill，对照序列长度、signature 形状和后端，以区分计算效率与填充效率；不能仅凭 tokens/s 判定算力瓶颈。
4. 对 decode，先记录上下文长度、采样配置和后端，再与带宽侧上限对照。上下文扫描只能显示相关成本随长度变化；若要区分 KV 流量与计算，应增加硬件计数器或更小范围的探针。

## Roofline 分析框架

Roofline 用算术强度连接计算吞吐与内存带宽。沿用第 1 章的记号：每个 token 的计算量为 \\(F\\)，数据搬运量为 \\(D\\)，工作负载可用的有效计算吞吐和内存带宽分别为 \\(P_{\mathrm{eff}}\\) 与 \\(B_{\mathrm{eff}}\\)。token 率满足

$$ R \leq \min\left(\frac{P_{\mathrm{eff}}}{F},\frac{B_{\mathrm{eff}}}{D}\right) $$

较低的一项决定这个工作点的 Roofline 上限。若带宽项更低，称为 memory-bound；若计算项更低，称为 compute-bound。端到端测得一个 tokens/s 点值，还不足以判定哪一项更低。

prefill 一次处理一段 token，同一份权重可在序列维度复用，因此算术强度通常高于 decode。较长的 prefill 在 kernel 利用率足够时，可能进入计算约束区。batch=1 的稠密 decode 每次只生成一个新 token。若每步激活全部权重，且权重工作集超过可用片上缓存，它通常更接近带宽约束区。这个判断还要求量化解包、采样与同步没有先成为瓶颈。每步只激活部分参数、稀疏计算、权重常驻缓存、分层卸载或 kernel 利用率不足都会改变这个判断。

附录 D 的 Apple M5 Pro 主矩阵提供一组后端敏感度数据。Gemma 4 E4B、context 1024、decode 128 token 时，prefill 从 cpu 的 259.2 tokens/s 变为 gpu 的 999.1 tokens/s，约为 3.9 倍；decode 从 24.7 变为 50.6 tokens/s，约为 2.0 倍。切换后端同时改变了有效计算吞吐、有效带宽、delegate 和 kernel。这组比例表明两个阶段的后端敏感度不同，不能单独证明 prefill 已受算力约束、decode 已受带宽约束。

第 1 章的 25 tokens/s 来自假想手机的题设，不能直接套到这台 Mac。对本书基准模型，`.litertlm` 整文件为 3.66 GB，并包含多个模型段；附录 D 识别出的主 decode 段 payload 为 2.26 GB。若额外假设每个 decode step 恰好读取这 2.26 GB 一次，并忽略其他流量，那么 gpu/256 档的等效主干 payload 速率为 `50.6 × 2.26 ≈ 114 GB/s`，cpu/256 档为 `24.8 × 2.26 ≈ 56 GB/s`。这两个数是吞吐与假设 payload 的乘积，不是硬件计数器测得的 DRAM 带宽，也不能用来核验设备标称带宽。

<figure>
{{#include figs/fig-2-1.svg}}
<figcaption>图 2-1　Roofline 给出计算侧和带宽侧两条上限；prefill 与 decode 的工作点位置取决于序列、模型、缓存和 kernel 条件。</figcaption>
</figure>

### 用吞吐差异估算上下文相关开销

附录 D 中，cpu 后端的 decode 吞吐从 context 256 时的 24.8 tokens/s 降到 context 4096 时的 20.7 tokens/s；gpu 后端从 50.6 降到 45.6 tokens/s。模型、设备与后端在各自对照中保持不变，输入上下文长度发生变化〔基准 D〕。

这里仅估算等效字节数。假设两档使用相同的 \\(B_{\mathrm{eff}}\\)，全部时间都能表示为字节数除以该带宽，并忽略 256-token 档的上下文相关流量。取主 decode 段 \\(D_w=2.26\\) GB。若长上下文为 \\(L\\)，短、长两档吞吐分别为 \\(R_s\\) 和 \\(R_l\\)，则每个上下文 token 的等效附加项为

$$ d_{\mathrm{eq}}=\frac{D_w}{L}\left(\frac{R_s}{R_l}-1\right) $$

代入 cpu 数据得到约 107 KiB/token；代入 gpu 数据得到约 59 KiB/token。第 6 章按模型张量形状计算的逻辑 KV 容量是 28 KiB/token。这三个数不应相等：`d_{\mathrm{eq}}` 把注意力计算、带宽利用率变化、缓存行为和其他随上下文变化的成本都折算成字节。它不是实际 DRAM 流量的测量值，也不能单独证明降速全部来自 KV cache。这个对照只说明上下文相关成本不能从权重 payload 一项解释；第 6 章再按 KV 张量形状和访问路径核算。

## 二十个问题

表 2-2 把后续章节的核心问题与解答位置对应起来。表中只给出索引，各章仍会说明结论的适用条件。

| # | 问题 | 解答章 |
|---|---|---|
| 1 | 同样的模型，为什么云端可以流畅运行，手机端通常更受限制？ | 1 |
| 2 | 8 GiB 内存的手机能否容纳 4B 参数模型？ | 1、6、7 |
| 3 | prefill 每秒几千 token、decode 只有几十，为什么？ | 2 初讲、6 深化 |
| 4 | time-to-first-token 由哪几段时间构成？ | 2 |
| 5 | Engine 和 Session 为什么要分成两层？ | 3 |
| 6 | 多轮对话的历史，每一轮都要重新计算吗？ | 3（单轮渲染或前缀后缀提取）、6（KV cache） |
| 7 | 聊天模板由哪一层在何时应用？ | 3 |
| 8 | 模型文件中为什么包含多个固定长度的 prefill 入口？ | 4 |
| 9 | 取消请求在哪些检查点被读取，响应延迟由什么决定？ | 4、5 |
| 10 | 温度、top-k、top-p 各自改变了什么？ | 5 |
| 11 | 流式输出如何处理跨 token 的 UTF-8 字节序列？ | 5 |
| 12 | 停止序列只有前缀匹配时，哪些 token 需要暂缓输出？ | 5 |
| 13 | KV cache 占多少内存？`--max-num-tokens` 为什么影响速度？ | 6（`LiteRT-LM#2568`[^ch02-issue-2568]） |
| 14 | 克隆对话并创建分支时，需要重算公共前缀吗？ | 3、6 |
| 15 | int4 量化减少的是体积、带宽需求还是计算量？ | 7 |
| 16 | `.litertlm` 单文件包含哪些分段？ | 7 |
| 17 | 切换后端后，速度和输出为什么可能变化？ | 8（`LiteRT-LM#2281`[^ch02-issue-2281]） |
| 18 | GPU 设备侧采样（device-side sampling）减少了哪些 host/device 数据传输？ | 8 |
| 19 | speculative decoding 在什么条件下提高吞吐，又会增加哪些开销？ | 9（`LiteRT-LM#2227`[^ch02-issue-2227]） |
| 20 | 图像怎样编码并进入语言模型？约束解码怎样限制结构化输出？ | 10 |

> 表 2-2　二十个推理与运行时问题。不同语言绑定如何复用核心 runtime，同时采用不同的原生边界，见第 11 章。

## 五层职责视图

本书按主要职责把生成路径整理为五层，供后续章节定位实现位置。这是分析视图，不是仓库声明的强制依赖规则。主调用路径大体自上而下；工厂、元数据、日志与工具代码仍可能跨越相邻层。

1. 对外接口层。包括 Engine、Session、CLI、C API 与各语言绑定，应用通常从这里进入（第 3、11 章）。
2. 对话与编排层。把多轮消息转换为模型输入，并组织 prefill、decode、取消和回调（第 3 至 5 章）。
3. 推理执行层。通过 executor 和 LiteRT 模型执行操作管理推理状态，并适配不同后端（第 6、8、9 章）。
4. 组件层。包括 tokenizer、采样器和约束解码等可复用组件（第 5、10 章）。
5. 格式与基础设施层。包括 `.litertlm` 文件格式、模型资源和通用运行设施（第 7 章）。

以下用前三层的接口和调用点说明这套划分。

对外接口层的主要入口之一是 `SessionInterface`（`runtime/engine/engine.h:70`）。类注释说明，Session 保存一次独立交互的内部状态，并负责生成、prefill 与 decode（`runtime/engine/engine.h:65`）：

```cpp
class SessionInterface {
 public:
  // ...
  virtual absl::StatusOr<Responses> GenerateContent(   // (1)
      const std::vector<InputData>& contents) = 0;
  // ...
  virtual absl::Status RunPrefill(const std::vector<InputData>& contents) = 0; // (2)
  // ...
  virtual absl::StatusOr<Responses> RunDecode() = 0;   // (3)
};
```

(1) `GenerateContent` 在一次调用中处理 prefill 与 decode（`runtime/engine/engine.h:107`）。(2)(3) 允许调用方分别执行两个阶段，见 `runtime/engine/engine.h:168` 与 `runtime/engine/engine.h:184`。`= 0` 表示这些方法由具体 Session 实现提供；接口本身不固定后端。

对话与编排层的核心循环位于 `runtime/core/tasks.cc`。`Prefill` 先读取 executor 给出的最大 token 数，再校验输入 token 数是否严格小于该值（`runtime/core/tasks.cc:413`）：

```cpp
  auto num_tokens = token_id_tensor_type.Layout().Dimensions().back();
  if (num_tokens >= max_num_tokens) {                        // (1)
    return absl::InvalidArgumentError(absl::StrCat(
        "Input token ids are too long. Exceeding the maximum number of tokens "
        "allowed: ",
        num_tokens, " >= ", max_num_tokens));
  }
  // ...
  RETURN_IF_ERROR(executor.Prefill(inputs, params));         // (2)
```

(1) 在调用 executor 之前拒绝 `num_tokens >= max_num_tokens`；第 6 章解释该上限与 KV cache 容量的关系。(2) 校验通过后调用参数化的 `executor.Prefill`。

decode 循环中各步骤的顺序决定了取消与输出的语义，核心实现位于 `runtime/core/tasks.cc`。每轮开始先读取 `cancelled`（:487），随后执行一个 `DecodeOneStep`（:518）。如果本轮得到可输出文本，流式回调发生在 :563。本轮末尾，`ShouldStop` 检查停止序列、benchmark 步数、`max_num_tokens` 与 `max_output_tokens`（:571，定义见 :85）。取消请求不会中断已经进入 executor 的当前 step，只能在下一轮检查点生效。第 4、5 章分别展开取消、停止序列与 UTF-8 输出处理。

推理执行层的公共边界是 `LlmExecutorBase`（`runtime/executor/llm_executor_base.h:40`）：

```cpp
class LlmExecutorBase {
 public:
  // ...
  virtual absl::Status Prefill(const ExecutorInputs& inputs) = 0;        // (1)
  // ...
  virtual absl::StatusOr<std::vector<std::vector<int>>> Decode() = 0;    // (2)
  // ...
  virtual absl::string_view ExecutorBackendName() const = 0;             // (3)
};
```

(1) 定义基本 prefill 接口。`runtime/core/tasks.cc` 使用的参数化重载位于 `runtime/executor/llm_executor_base.h:51`；基类默认返回未实现，支持该路径的具体 executor 需要覆盖它。(2) 定义基本 decode 接口。(3) 返回 executor 的后端名称。工厂选择的 executor 与 delegate 配置可因 CPU、GPU、NPU 路径而异，但上层编排仍通过这些公共方法调用。组件层主要位于 `runtime/components/`；文件格式实现位于 `schema/`。

<figure>
{{#include figs/fig-2-2.svg}}
<figcaption>图 2-2　本书用于定位职责的五层视图；主生成路径自接口向执行与模型资源传递，跨层辅助依赖未在图中展开。</figcaption>
</figure>

图 2-3 沿调用方向展开一次生成请求的主数据流：

<figure>
{{#include figs/fig-2-3.svg}}
<figcaption>图 2-3　一次生成请求的主数据流；侧框列出各阶段使用的输入、组件和后端资源。</figcaption>
</figure>

`SessionInterface` 与 `LlmExecutorBase` 提供抽象边界，但不保证所有模块只依赖相邻层。工厂和设置对象把后端选择传递给 executor 与 delegate；上层编排可以复用，具体能力仍要逐后端检查。会话相关状态由 Session 及其 executor 上下文持有。Clone、checkpoint 与 rewind 分别复制或调整哪些状态，需按第 6 章的具体实现判断。

## `.litertlm` 文件的组成

`.litertlm` 是 LiteRT-LM 定义的容器格式。一个文件可以包含 TFLite 模型、tokenizer、LLM 元数据和通用二进制数据等 section；具体 section 组合由模型文件决定。`litertlm_print` 读取文件头（`schema/core/litertlm_print.cc:110`），输出系统元数据（`schema/core/litertlm_print.cc:133`），再遍历 section（`schema/core/litertlm_print.cc:159`）：

```cpp
    for (size_t i = 0; i < section_objects->size(); ++i) {
      auto sec_obj = section_objects->Get(i);
      // ...
      output_stream << std::string(INDENT_SPACES, ' ')
                    << "Begin Offset: " << sec_obj->begin_offset() << "\n"; // (1)
      output_stream << std::string(INDENT_SPACES, ' ')
                    << "End Offset:   " << sec_obj->end_offset() << "\n";
      output_stream << std::string(INDENT_SPACES, ' ')
                    << "Data Type:    "
                    << AnySectionDataTypeToString(sec_obj->data_type())    // (2)
                    << "\n";
    }
```

(1) `begin_offset` 与 `end_offset` 定义半开区间 `[begin, end)`；schema 注释直接给出这一约定（`schema/core/litertlm_header_schema.fbs:85`），读取函数也用 `end_offset - begin_offset` 计算大小（`schema/core/litertlm_read.cc:200`）。(2) `data_type` 区分 section 内容。常见值包括 `TFLiteModel`、`SP_Tokenizer`、`HF_Tokenizer_Zlib`、`LlmMetadataProto` 与 `GenericBinaryData`（`schema/core/litertlm_header_schema.fbs:72`）。打印函数用 `AnySectionDataTypeToString` 转换名称（`schema/core/litertlm_utils.cc:25`），遇到 LLM 元数据时还会解析 proto 并输出文本（`schema/core/litertlm_print.cc:181`）。

下面是一份三段示例文件的节选，并非本书基准模型。附录 D 对 Gemma 4 E4B 文件识别出 10 个 TFLite 模型段，完整清单见附录 D 第六节。

```text
LiteRT-LM Version: 1.5.0

+----------------------+
|   System Metadata    |
+----------------------+
  Key: Authors, Value (String): Google
  ...
+----------------------+
|     Sections (3)     |
+----------------------+
Section 0:
  Begin Offset: 8192
  End Offset:   3579204608
  Data Type:    AnySectionDataType_TFLiteModel     <- 权重与计算图
Section 1:
  Begin Offset: 3579204608
  End Offset:   3583074304
  Data Type:    AnySectionDataType_SP_Tokenizer    <- SentencePiece 分词器
Section 2:
  Begin Offset: 3583074304
  End Offset:   3583078400
  Data Type:    AnySectionDataType_LlmMetadataProto
    <<<<<<<< start of LlmMetadata
      start_token { token_ids: 2 }                 <- BOS
      stop_tokens { token_str: "<end_of_turn>" }   <- 停止序列配置，第 12 问
      prompt_templates { ... }                     <- 聊天模板，第 7 问
    >>>>>>>> end of LlmMetadata
```

这个示例把 TFLite 模型、SentencePiece tokenizer 与 LLM 元数据放在三个 section 中；其他模型可以采用不同布局。构建 Session 配置时，源码会从 LLM 元数据读取 `start_token`（`runtime/engine/engine_settings.cc:594`）、`stop_tokens`（`runtime/engine/engine_settings.cc:603`）和 `prompt_templates`（`runtime/engine/engine_settings.cc:623`）。它们如何影响输入构造与停止判断，见第 3、5 章。

TFLite section 的读取路径先用 `end_offset - begin_offset` 计算模型大小（`schema/core/litertlm_read.cc:223`），再创建 `tflite::MMAPAllocation`（`schema/core/litertlm_read.cc:226`）。这说明 v0.13.1 的该路径使用文件映射分配对象；页错误、物理驻留与初始化耗时仍需结合操作系统和访问行为测量。第 7 章分析 section 布局、映射和冷启动。

## 小结

benchmark 指标必须按 `BenchmarkInfo` 的源码定义解释，Roofline 结论必须附带工作负载假设，五层视图只用于定位主要职责。第 3 至 5 章继续分析 Session、prefill/decode 编排和输出处理。

## 练习与自查

1. TTFT 验算。用附录 D 的 gpu/1024 档数据（prefill 999.1 tokens/s、decode 50.6 tokens/s）按 `GetTimeToFirstToken` 的公式复算 TTFT，并说明为什么这不是独立计时。
2. 等效字节估算。用 gpu 的 context 256 与 4096 两档 decode 吞吐（50.6 与 45.6 tokens/s）和 2.26 GB 主段 payload 计算 `d_{\mathrm{eq}}`，再说明它为何不能当作 KV cache 的实测 DRAM 流量。
3. 条件检查。某设备报告 Init 1.8 s、TTFT 4.4 s、decode 45 tokens/s，上下文加倍后 decode 下降 8%。列出判断异常前仍需补充的模型、输入、后端和测量条件。
4. 代码定位。找到 benchmark 模式要求 prefill 等待完成的语句，并说明不等待时计时可能覆盖什么范围。
5. 架构归位。约束解码的 `MaskLogits` 调用属于五层视图中的哪一层？它接收的 logits 来自哪一层？

[^ch02-litertlm-readme]: Google AI Edge，[LiteRT-LM README](https://github.com/google-ai-edge/LiteRT-LM/tree/v0.13.1)，版本 v0.13.1；访问日期：2026-07-18。
[^ch02-issue-2568]: Yegorsh，[*`--max-num-tokens` unreasonably affects decoding speed*](https://github.com/google-ai-edge/LiteRT-LM/issues/2568)，LiteRT-LM issue #2568，2026-06-13；访问日期：2026-07-18。
[^ch02-issue-2281]: 4ntoine，[*Different inference result depending on backend*](https://github.com/google-ai-edge/LiteRT-LM/issues/2281)，LiteRT-LM issue #2281，2026-05-15；访问日期：2026-07-18。
[^ch02-issue-2227]: Shoolife，[*MTP / speculative decoding regresses decode tok/s on PowerVR GPU (Tensor G6) — even with GPU sampler fully loaded*](https://github.com/google-ai-edge/LiteRT-LM/issues/2227)，LiteRT-LM issue #2227，2026-05-11；访问日期：2026-07-18。
