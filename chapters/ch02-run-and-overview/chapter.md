# 第 2 章 从运行到架构：benchmark 指标解读与五层概览

> 本章目标：在本机把模型跑起来，读懂它输出的第一批性能指标，并建立一张贯穿全书的架构地图。

第 1 章的推导停留在纸面。本章把它落到终端里真实的输出，并用实测数据核对第 1 章那条「约 25 tokens/s 的 decode 上限」。

## 让它先跑起来

最快的路径不需要编译。LiteRT-LM 提供了一个命令行工具，一条命令装好：

```bash
uv tool install litert-lm
litert-lm run \
  --from-huggingface-repo=litert-community/gemma-4-E4B-it-litert-lm \
  gemma-4-E4B-it.litertlm \
  --prompt="What is the capital of France?"
```

`litert-lm` 是一层轻量的 Python 封装：它通过 FFI 绑定 C++ 运行时，再用 click 注册八个子命令（`python/litert_lm_cli/main.py:52`）。日常常用五个：`run` 交互对话、`benchmark` 采集性能指标、`import` 把模型收进本地目录、`list` 查看已有模型、`serve` 起一个 OpenAI 兼容的本地服务。

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

每个子命令是一个独立模块，各自调用 `register(cli)` 挂进同一个 click group。新增子命令不牵动其他模块，这是全书第一条设计原则「接口隔离」在 CLI 层的一次缩影。(2) 处的 `import` 子命令要靠 `importlib.import_module` 动态加载，因为 `import` 是 Python 关键字，不能直接写成 `from ... import import`（`main.py:33`）。

`run` 子命令的核心，是拿到模型路径后如何把一次对话跑起来。略去参数解析，核心逻辑如下：

```python
      engine_cm = litert_lm.Engine(       // (1)
          model_obj.model_path,
          backend=backend_val,
          enable_speculative_decoding=enable_speculative_decoding,
          max_num_tokens=max_num_tokens,
          # ...
      )

    with engine_cm as engine:             // (2)
      # ...
      runner_cm = engine.create_session(...)
```

(1) `Engine` 只接受一个模型路径和几个后端开关。它是第 3 章要拆解的两层结构里的外层，负责加载权重、装配后端。(2) `create_session` 才返回真正承载对话的 `session`。一个 Engine 可以派生多个 Session，这是第 5 问「Engine 和 Session 为什么分两层」的入口：权重加载一次，会话状态各自独立。`run` 用 `with` 托管 Engine 的生命周期，退出时释放显存与 KV cache。

第一次运行会从 Hugging Face 拉取模型：`from_huggingface_repo` 触发 `common.download_from_huggingface`（`run.py:571`）。litert-community 的 Gemma 4 版可直接下载，google/ 官方版是受限发布，需先接受许可条款。模型文件通常数 GiB，需预留磁盘空间，首次下载耗时较长。跑通之后，答案会逐 token 输出：每个 decode step 产出一个增量 token，边生成边输出，这就是第 1 章的内存带宽约束（后文有时称带宽墙）在终端里的直接表现。

逐 token 的流式输出，在 `run` 的输出循环里看得最清楚：它对 `send_message_async` 返回的 stream 逐块迭代，每块是一小段文本，边收边打印，不等整段生成完（`run.py:100`）：

```python
  stream = conversation.send_message_async(prompt)
  try:
    for chunk in stream:                          // (1)
      content_list = chunk.get("content", [])
      for item in content_list:
        if item.get("type") == "text":
          # ...
          click.echo(click.style(item.get("text", ""), fg="yellow"), nl=False)
      # ...
  except KeyboardInterrupt:                        // (2)
    conversation.cancel_process()
```

(1) 每次循环体对应一个 decode step 产出的增量 token，`nl=False` 让它们首尾相接连成一条流。(2) 按下 Ctrl-C，`KeyboardInterrupt` 直接调用 `cancel_process()`。这是第 9 问「生成中途取消为什么能立即停止」在最外层的入口：取消信号沿调用链传到 decode 循环，第 4、5 章会顺着这条链讲清楚。本章稍后走读 decode 循环时，会看到这个取消信号在 `tasks.cc` 里被检查的确切位置。

如果要读源码、改代码，就得从源码编译 C++ 演示程序 `litert_lm_main`（完整构建见第 11 章与附录 C，这里先直接用它）。它最核心的两个开关：

- `--backend`：选执行后端，默认是 `gpu`（`runtime/engine/litert_lm_main.cc:52`）。想用 CPU 就传 `--backend=cpu`。
- `--model_path`：指向一个 `.litertlm` 模型文件（`litert_lm_main.cc:54`）。

一个纯 CPU 的最小跑法：

```bash
litert_lm_main --backend=cpu --model_path=<你的模型>.litertlm
```

这个 C++ 程序把「一次对话」压缩成十来行，正好当作五层架构的第一张概览图（`runtime/engine/litert_lm_main.cc:113`）。下面这段是 `MainHelper` 的主干（省略了错误处理与 conversation 的装配）：

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
  ASSIGN_OR_RETURN(auto engine, litert::lm::EngineFactory::CreateDefault(
                                    std::move(engine_settings)));
  // ...
  RETURN_IF_ERROR(conversation->SendMessageAsync(                   // (3)
      json::object({{"role", "user"}, {"content", content_list}}),
      CreateMessageCallback()));
  RETURN_IF_ERROR(engine->WaitUntilDone(absl::Minutes(10)));
```

三步完成一次推理：(1) 把 `--backend` 字符串解析成 `Backend` 枚举，再交给工厂。CPU/GPU/NPU 的执行器实现从这里分岔，对应第二条设计原则「可插拔后端」（第 8 章）；本章末尾会走读这条分发路径的具体代码。(2) 这个演示程序默认打开 benchmark，因此每跑一次都会输出一份性能指标，附录 D 的数据即由此采集。(3) `SendMessageAsync` 是非阻塞调用，prefill 与 decode 在后台线程执行，主线程靠 `WaitUntilDone` 等待；文本通过 `CreateMessageCallback` 分段回调，`message->is_null()` 时输出一个换行表示结束。本书的主线是推理流水线——一段输入经 prefill 处理后逐 token 生成的完整链路，起点就是这一句 `SendMessageAsync`，第二篇会沿着这条链路深入展开。

## 读懂第一批数字

`litert-lm benchmark` 专门输出性能指标（本书附录 D 的基准数据集即由此采集）。它跑一个纯性能循环，不做真实对话，只按指定的 token 数各跑一轮 prefill 和 decode，再输出四个数字（`python/litert_lm_cli/commands/benchmark.py:102`）：

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

四行输出，分别对应四个受不同资源约束的指标。(1) `benchmark_obj.run()` 底层是 C++ 的 `BenchmarkInfo`：prefill 和 decode 分别计时，`last_*_per_second` 取最后一轮的吞吐。(2) Init 是把模型加载到就绪的时间，单列一项，不计入 (3) 的 TTFT。这一点是下面验算 TTFT 的关键。一次真实采集大致如下（M5 Pro、Gemma 4 E4B、cpu、prefill 256 token / decode 128 token〔基准 D〕）：

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

这四个数字背后是一个叫 `BenchmarkInfo` 的结构，C API 把每一项都单独暴露了出来。四个最值得关注的：

| 指标 | 含义 | 受何种资源约束 | C API（`c/engine.h`） |
|---|---|---|---|
| **TTFT**（首 token 时延，ms） | 从发起到首个 token 生成的时间 | prefill（算力） | `..._get_time_to_first_token`（:583） |
| **prefill 吞吐**（tokens/s） | 提示词被处理的速度 | prefill（算力） | `..._get_prefill_tokens_per_sec_at`（:634） |
| **decode 吞吐**（tokens/s） | 逐 token 生成的速度 | decode（带宽） | `..._get_decode_tokens_per_sec_at`（:643） |
| **初始化时间**（s） | 加载模型到就绪 | 冷启动（第 7 章） | `..._get_total_init_time_in_second`（:591） |

> 表 2-1　benchmark 报告的四个核心指标，以及它们各自受何种资源约束。第 6、7 章会分别深挖 decode 吞吐和初始化时间。

其中 TTFT 可以当场拆开（这就是第 4 问的答案）：它 ≈ 整段提示词的 prefill 耗时 + 第一步 decode，**不含**模型加载（Init 单列）。可以用基准数据验算：cpu、上下文 256 时，256 ÷ 65.6 + 1 ÷ 24.8 ≈ 3.94 s，与实测 TTFT 3.94 s 精确吻合；上下文拉到 4096，TTFT 涨到 18.1 s，几乎全部来自 prefill 的耗时〔基准 D〕。所以想让第一个字更快，要么缩短提示词，要么加快 prefill——换个算力更强的后端效果直接可见。

注意这张表最左和第三列的对应关系：**prefill 吞吐和 decode 吞吐不是一个东西，也不该被平均成"一个速度"。** 一个模型的 prefill 吞吐常常比 decode 高出一到两个数量级（本书基准机上是 10-20 倍〔基准 D〕；NPU 或更强的 GPU 上差距更大）。原因第 1 章已经埋下：prefill 受计算能力约束（可并行运算），decode 受内存带宽约束（每步搬运全部权重）。把它俩混在一起谈"这模型多少 tok/s"，是端侧性能讨论里一个常见的混淆。

> 本书所有实测数据来自附录 D 的基准数据集（同一台 Mac、Gemma 4 E4B、公开可复现的采集脚本）。凡标注「〔基准 D〕」处，即由这套数据回填。

## 这些数字怎么来的：计时器与测量语义

读懂这些指标之后，还要知道数字是怎么得到的，否则无从判断它可不可信。四个指标全部出自 `BenchmarkInfo`（`runtime/engine/io_types.h:420`），它的计时模型很简单：每一轮 prefill 或 decode 记为一个 turn，起点打一个时间戳，终点用当前时间减起点，连同这一轮处理的 token 数存进列表（`runtime/engine/io_types.cc:307`）：

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

(1) 一个 turn 就是一条 `(token 数, 耗时)` 记录。吞吐的定义随之而来：某一轮的 tokens/s 就是这条记录的 `num_tokens / duration`（`GetDecodeTokensPerSec`，`io_types.cc:434`，函数里对零时长与越界各有防护）。前面表 2-1 里的 `last_*_per_second`，取的是最后一轮的这个商。

TTFT 的口径也写在代码里（`GetTimeToFirstToken`，`io_types.cc:455`）：

```cpp
double first_decode_token_seconds = absl::ToDoubleSeconds(
    decode_turns_[0].duration / decode_turns_[0].num_tokens);   // (1)
double first_prefill_token_seconds =
    absl::ToDoubleSeconds(prefill_turns_[0].duration);          // (2)
return first_decode_token_seconds + first_prefill_token_seconds;
```

(2) 是首轮 prefill 的完整耗时，(1) 是首轮 decode 平摊到单 token 的时长，两者相加。上一节那笔 256 ÷ 65.6 + 1 ÷ 24.8 ≈ 3.94 s 的验算不是本书发明的口径，就是这个函数的算式本身。也因此 Init 不在里面：模型加载在任何 turn 开始之前就结束了。

数字可信还依赖两处刻意改变执行语义的设计。其一，benchmark 模式强制 prefill 同步完成（`runtime/core/tasks.cc:435`）：

```cpp
// Wait for prefill to complete if benchmark mode is enabled.
params.SetWaitForCompletion(wait_for_completion | benchmark_info.has_value());
```

正常路径上 prefill 可以异步提交、不等硬件实际执行完成就返回（第 4 章）；计时若落在这样的路径上，测到的只是「提交耗时」。这一行把 benchmark 模式下的计时终点钉在硬件完成之后。其二，`ShouldStop` 对 benchmark 有专门分支（`tasks.cc:86`，第 5 章贴过全文）：指定了 decode 步数就跑满 N 步、忽略停止词。原因也是测量语义：若模型第 20 步恰好生成了停止词，一次 128 步的吞吐测量就会缩减为 20 步的样本，抖动大且不可比。

这两处合起来是一条测量原则：**基准模式可以改变执行语义，但改变的方向必须让数字更接近想测的那件事**（硬件真实耗时、固定长度的稳态吞吐），并且写在代码里可以被核对。

有了定义与语义，瓶颈判定可以给出一个操作化流程，全书各章会反复用到：

1. **Init 偏大**：与推理无关，是加载问题，查第 7 章（mmap、分段、并行加载）。
2. **TTFT 偏大而 decode 正常**：几乎总是 prefill 的账（TTFT 算式里 prefill 项占大头），受算力约束，换更强的后端收益直接（本节上文 3.9 倍）。
3. **decode 吞吐低**：先用第 1 章的公式算纯权重上限（带宽 ÷ 权重字节），实测贴近上限说明已被内存带宽约束住，加算力无用；离上限还远则查采样、约束解码等每步的额外开销（第 5、10 章）。
4. **decode 随上下文变长而变慢**：KV cache 的带宽占用在增长，见下一节的反解练习与第 6 章的正式对账。

## 一个分析框架：Roofline

有了 prefill/decode 两类指标，就可以引入贯穿全书的分析框架：Roofline（屋顶线）模型。它一句话讲完：**一段计算的速度，要么受算力限制，要么受带宽限制，取决于它每读一字节数据能摊上多少次计算**（这个比值叫算术强度，是体系结构教科书的常识内容）。

- **prefill** 一次处理许多 token，同一批权重被许多 token 共用，算术强度高，落在 Roofline 的"算力受限"区——所以它受 TOPS 约束。
- **decode** 一次只处理一个 token，把全部权重读进来只为算这一个字，算术强度极低，落在"带宽受限"区——所以它受 GB/s 约束，与算力无关。这正是第 1 章那条 25 tok/s 上限公式的来历。

顺着这个视角，本书基准数据里得到直接体现〔基准 D〕：换到 GPU（都在 1024 上下文档测），prefill 从 259 tok/s 跳到 999（约 3.9 倍，算力受限，增加算力即可提速）；decode 却只从 24.7 到 50.6 tok/s（约 2 倍，被内存带宽约束，算力再高也无济于事）。同一台机器、同一个模型，两类操作对"更强的硬件"的反应截然不同。

这里也对一下第 1 章的账。那条 25 tok/s 是为一部假想手机（50 GB/s、1.86 GiB 权重）算的；本书基准机是另一套参数，得按同一条公式重算。这里有一个容易算错的分母：3.66 GB 是整个 `.litertlm` 文件，其中还打包着视觉/音频编码器、词嵌入表等段；decode 每步真正要读一遍的是主干模型那一段，实剖为 2.26 GB（附录 D 的段表）。于是 gpu decode 50.6 tok/s × 2.26 GB ≈ 114 GB/s 的有效搬运速率，cpu 24.8 tok/s × 2.26 GB ≈ 56 GB/s，量级都落在桌面级统一内存芯片的合理区间。至于 cpu 实测贴着"25"，只是巧合——分子分母都不是同一套。**公式可以迁移，数字不能照搬**，这正是第 1 章说"这是把尺子"而不是"这是个答案"的原因。

<figure>
{{#include figs/fig-2-2.svg}}
<figcaption>图 2-2　Roofline 模型：prefill 落在算力受限区，decode 落在带宽受限区。两者受完全不同的资源约束，这是全书性能分析的基准框架。</figcaption>
</figure>

### 差的那一段：从两个实测数据反解 KV cache

Roofline 框架立刻能做一次有内容的练习。附录 D 里，cpu 后端 decode 吞吐在 256 上下文时是 24.8 tok/s，4096 上下文时掉到 20.7〔基准 D〕。权重没变，后端没变，变的只有上下文长度。这部分性能下降是什么引起的？

按带宽受限的模型推一遍（以下是推算，假设两档的有效带宽相同、每步读全量 KV cache；每步读取的权重取主干模型段的 2.26 GB，见上文）。256 上下文时 KV cache 只有几 MB，相对权重可以忽略，于是有效带宽约为 24.8 × 2.26 ≈ 56 GB/s。4096 上下文时每步搬运总量变为 56 ÷ 20.7 ≈ 2.71 GB，比权重多出约 0.45 GB。这多出的部分除以 4096 个 token，约 110 KiB/token。也就是说，**仅凭两档吞吐实测加一条带宽等式，就能反解出「每 token 有一笔随上下文增长的额外搬运」**——这就是 KV cache 的带宽签名。

同样的练习放到 gpu 上（50.6 → 45.6 tok/s）反解出约 60 KiB/token，与 cpu 侧的 110 KiB 并不一致；而从模型文件正着算的真实值是 28 KiB/token（24 层 int8 KV，实剖与算式见第 6 章与附录 D）。三个数字不一致，这个偏差本身有信息量：恒定带宽是近似假设，注意力计算随上下文增长的算力开销、两个后端不同的带宽利用率，全都被这条等式折进了"字节"里。反解正确探测到了额外搬运的存在与量级（每 token 数十 KiB 级），但把所有随上下文增长的成本都记在了它头上，高估了 2-4 倍。精确的账要正着算，第 6 章做这件事，第 13 问（`--max-num-tokens` 为什么影响速度）也在那里得到解答。

## 二十个问题

一本书最怕读者读着读着忘了自己为什么在读。所以把全书要回答的问题先摆出来，读到答案时回头勾掉。这份清单也是给作者的自检——每一问都必须在标注的章节得到正面回答。

| # | 你会问 | 解答章 |
|---|---|---|
| 1 | 同样的模型，为什么云端流畅、手机上就吃力？ | 1 |
| 2 | 4B 参数的模型，8 GiB 内存的手机放得下吗？ | 1、6、7 |
| 3 | prefill 每秒几千 token、decode 只有几十，为什么？ | 2 初讲、6 深化 |
| 4 | time-to-first-token 由哪几段时间构成？ | 2 |
| 5 | Engine 和 Session 为什么要分成两层？ | 3 |
| 6 | 多轮对话的历史，每一轮都要重新计算吗？ | 3（模板 diff）、6（KV cache） |
| 7 | 聊天模板是谁、在什么时候套上去的？ | 3 |
| 8 | 模型文件里为什么有好几个固定长度的 prefill 入口？ | 4 |
| 9 | 生成中途取消，为什么能立刻停下？ | 4、5 |
| 10 | 温度、top-k、top-p 各自改变了什么？ | 5 |
| 11 | 流式输出为什么偶尔出现不完整的 token？ | 5 |
| 12 | 停止词只出现了一半时，要不要输出？ | 5 |
| 13 | KV cache 占多少内存？`--max-num-tokens` 为什么影响速度？ | 6（`LiteRT-LM#2568`） |
| 14 | 克隆对话做分叉，需要重算公共前缀吗？ | 6 |
| 15 | int4 量化省的是体积、带宽还是算力？ | 7 |
| 16 | `.litertlm` 单文件里都装了什么？ | 7 |
| 17 | 换个后端，速度甚至输出为什么都会变？ | 8（`LiteRT-LM#2281`） |
| 18 | GPU 片上采样比拷回 CPU 采样快在哪？ | 8 |
| 19 | 推测解码靠"猜"，为什么反而更快？什么时候更慢？ | 9（`LiteRT-LM#2227`） |
| 20 | 模型怎么"看见"图片？输出怎么保证是合法 JSON？ | 10 |

> 表 2-2　二十个问题。工程集成向的问题（"一套 C++ 怎么变出六种语言 SDK"）另见第 11 章，不入这份面向推理的主清单。

## 五层架构与三条原则

下面来看这张架构地图。LiteRT-LM 从上到下分五层，每一层只跟相邻层打交道：

1. **对外接口层**——Engine / Session API、CLI、C API 与各语言绑定。使用者只碰这一层（第 3 章、第 11 章）。
2. **对话与编排层**——把多轮对话转成模型输入，把一次请求编排成 prefill→decode 的有状态流程（第 3、4、5 章）。
3. **推理执行层**——真正用 LiteRT 跑模型、管 KV cache、屏蔽 CPU/GPU/NPU 差异（第 6、8、9 章）。
4. **组件层**——tokenizer、采样器、约束解码等可复用零件（第 5、10 章）。
5. **格式与基础设施层**——`.litertlm` 文件格式、线程池等底座（第 7 章）。

这五层不是抽象的概念划分，每一层都落在具体文件上。往下看三层，越贴近硬件抽象越薄。

第 1 层的门面是 `SessionInterface`（`runtime/engine/engine.h:70`）。它把「跑一次生成」暴露成一个纯虚接口，使用者只面对方法名，看不见后端：

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
  // ...
  virtual absl::StatusOr<BenchmarkInfo> GetBenchmarkInfo() = 0;
};
```

(1) `GenerateContent` 是高层的一步到位；(2)(3) 把它拆成 `RunPrefill` 和 `RunDecode` 两个可分别调用的原语。推理流水线在接口层就已经分成两段：先 prefill 并行处理提示词，再 decode 逐 token 生成。第 3 章讲 Engine/Session 分层，就从这个 `= 0` 的纯虚签名开始。

第 2 层的编排落在 `runtime/core/tasks.cc`。`RunPrefill`/`RunDecode` 往下调，就到这里的 `Prefill` 和 `Decode` 两个自由函数。`Prefill` 的开头先检查上下文长度上限（KV cache 容量，第 1 章内存容量约束的一个具体面）（`runtime/core/tasks.cc:413`）：

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

(1) 提示词的 token 数一旦顶到 `max_num_tokens`（KV cache 的容量上限，即第 13 问里的 `--max-num-tokens`），直接报错——这堵上下文长度上限在调用执行器之前就拦截。(2) 校验过了才把 `inputs` 交给下一层的 `executor.Prefill`。decode 侧则是一个 `while (true)` 循环，每转一圈调一次 `DecodeOneStep`，再调用 `ShouldStop` 判断是否需要终止（`runtime/core/tasks.cc:571` 调 `ShouldStop`）：遇到停止词、达到 benchmark 指定步数、达到 `max_num_tokens`、或超过 `max_output_tokens`，四者任一为真就跳出（`ShouldStop` 定义在 `tasks.cc:86`）。第 4、5 章顺着这个循环展开取消、停止词判断和输出不完整字符的处理。

第 3 层是执行器，一个纯虚基类把「用什么硬件跑」这件事完全封装起来（`runtime/executor/llm_executor_base.h:40`）：

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

(1)(2) 上一层调的 `executor.Prefill` / `Decode` 就是这两个纯虚方法；CPU、GPU、NPU 各有一个子类实现它们，同一套 `tasks.cc` 编排代码因此一字不改就能换后端。(3) `ExecutorBackendName` 让上层能问「我现在跑在哪个后端」，第 8 章讲换后端为什么连输出都会变，就从这里的多态分发切进去。第 4、5 层是可复用的 tokenizer / 采样器组件（`runtime/components/`）和 `.litertlm` 文件格式——留到第 5、7、10 章各自展开。

<figure>
{{#include figs/fig-2-1.svg}}
<figcaption>图 2-1　LiteRT-LM 的五层架构。使用者只与最上层打交道；越往下越贴近硬件。括号里是本书对应的章节。</figcaption>
</figure>

支撑这五层的是三条设计原则，它们会在后续每一章反复出现，先记住名字：

- **接口隔离**：每层用抽象接口挡住下层实现，换后端、换 tokenizer 都不震动别处；
- **可插拔后端**：CPU/GPU/NPU 由一个工厂按需装配，一套代码覆盖多硬件（第 8 章）；
- **状态即对象**：把一次会话的全部状态（KV cache、步数、配置）打包成可搬运的对象，多轮、克隆、回退都因此成立（第 6 章）。

## 一个 .litertlm 里装了什么

地图的最底层是模型文件本身。LiteRT-LM 用一个自定义的单文件格式 `.litertlm`，把权重、tokenizer、元数据、能力声明全打包进去。仓库附带一个打印工具（`schema/core/litertlm_print.cc`），能把一个模型文件的分段结构打印出来。它的主循环很直白：先读文件头里的版本号和系统元数据，再遍历每个 section，把偏移和数据类型逐段打出来（`schema/core/litertlm_print.cc:156`）：

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

(1) 每个 section 只是一对字节偏移 `[begin, end)`——文件本身是连续排布的一块，section 表就是一张目录。这正是第 7 章说 mmap 能帮上冷启动的物理前提：权重那一段可以直接映射进地址空间，不必先拷进堆。(2) `data_type` 是个枚举，取值只有那么几种：`AnySectionDataType_TFLiteModel`（权重与图）、`AnySectionDataType_SP_Tokenizer` 或 `HF_Tokenizer_Zlib`（两种 tokenizer）、`AnySectionDataType_LlmMetadataProto`（元数据）（`schema/core/litertlm_utils.cc:31`）。碰到 `LlmMetadataProto` 那一段，工具还会把 proto 展开成文本打进来。一份真实 dump 的骨架长这样：

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
      stop_tokens { token_str: "<end_of_turn>" }   <- 停止词，第 12 问
      prompt_templates { ... }                     <- 聊天模板，第 7 问
    >>>>>>>> end of LlmMetadata
```

一个文件里三件东西各占一段：几个 GiB 的权重、几 MiB 的 tokenizer、几 KiB 的元数据。元数据段里的 `start_token`、`stop_tokens`、`prompt_templates` 不是可有可无的——它们是第 7 问「聊天模板谁在什么时候套上去」、第 12 问「停止词只出现一半是否输出」的答案所在，模型文件自带一份它该怎么被对话包裹的说明书。现在先记住它分段的、每段一对偏移、可被打印工具拆开查看这个事实。第 7 章会深入分析，解释为什么要自造一个格式、以及 mmap 加载怎么帮上冷启动的忙。

## 小结

至此你掌握了：一个能在本机运行的模型、一个解读性能数据的 Roofline 分析框架、一张五层架构地图。接下来第二篇进入架构的第二层和第三层，跟随一个 token 经历从输入到输出的完整推理流水线。

---

<div class="aside-version">

本书写作期间上游发布了 v0.14.0（2026-07，共 144 个提交，见 GitHub Releases）。抽样核对：全书锚定的核心机制叙述（取消标志的消费位置、`BenchmarkInfo` 的定义位置等）在 v0.14.0 中未发生结构性变化；变动集中在依赖更新、NPU 质量修复与 C/Python API 扩充。全书引用维持锚定 `v0.13.1` 不变，理由见前言的锁版本约定；整体迁移留待下一次修订版统一进行。

</div>

## 练习与自查

1. **TTFT 验算。** 用附录 D 的 gpu/1024 档数据（prefill 999.1 tok/s、decode 50.6 tok/s）按 `GetTimeToFirstToken` 的算式验算 TTFT，并与实测 1.04 s 对照。
2. **反解练习重走。** 用 gpu 的 256 与 4096 两档 decode 实测（50.6 与 45.6 tok/s）、主干权重 2.26 GB，自己走一遍本章的反解，得出每 token 额外搬运量，并解释它为什么高于实剖真值 28 KiB。
3. **判定流程应用。** 某设备实测：Init 1.8 s、TTFT 4.4 s、decode 45 tok/s、上下文加倍后 decode 降 8%。按本章四步判定流程，逐项判断各环节是否异常。
4. **代码定位。** benchmark 模式为什么必须强制 prefill 同步完成？找出实现这一行为的那一行代码。
5. **架构归位。** 约束解码的 `MaskLogits` 调用发生在五层架构的哪一层？它修改的 logits 来自哪一层？


<!-- 基准数字已回填（附录 D）；表 2-2 已定稿为完整 20 问。 -->
