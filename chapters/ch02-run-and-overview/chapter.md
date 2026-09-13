# 第 2 章 从运行到架构：benchmark、Roofline 与五层视图

> 进入实现细节之前，读者需要三项内容：一个可复现的运行起点（2.1 节）、一套按源码口径解读测量数字的方法（2.2 至 2.4 节）、一份说明各问题在哪章解答、代码归哪层、模型文件里有什么的索引（2.5 至 2.7 节）。本章依次给出这三样。

第 1 章在明确的假设下推导了 decode 的带宽侧上限，那还只是估算。本章先运行 LiteRT-LM，再按源码定义逐项解读 benchmark 输出的四个数字，检验第 1 章的判断在实测工作点上是否成立；随后建立的五层职责视图与二十个问题索引，是后续各章共用的索引。

## 2.1　运行命令行工具

这一节回答三个问题：怎么运行 LiteRT-LM，运行后能看到什么，哪些行为和直觉不同。使用官方 Python 包就无需本地编译 C++。以下沿用官方 README 的命令形式，模型换成本书所用的 Gemma 4 E4B，并把安装版本固定为 v0.17.0。[^ch02-litertlm-readme]

```bash
uv tool install 'litert-lm==0.17.0'
litert-lm run \
  --from-huggingface-repo=litert-community/gemma-4-E4B-it-litert-lm \
  gemma-4-E4B-it.litertlm \
  --prompt="What is the capital of France?"
```

经 SOCKS 代理联网时，下载会报缺少 `socksio`（httpx 的可选依赖），任选一种解法：

```bash
# 方案一（临时）：本次改走 HTTP 代理，7890 换成自己代理的 HTTP 端口
ALL_PROXY=http://127.0.0.1:7890 litert-lm run \
  --from-huggingface-repo=litert-community/gemma-4-E4B-it-litert-lm \
  gemma-4-E4B-it.litertlm \
  --prompt="What is the capital of France?"

# 方案二（一次性）：重装补上 socks 依赖，此后按原命令运行
uv tool install --force --with 'httpx[socks]' 'litert-lm==0.17.0'
```

指定 `--from-huggingface-repo` 时，`run` 调用下载模块。本书基准模型文件为 3.66 GB，下载前应检查磁盘空间；耗时取决于网络与缓存状态。模型就绪后，回答逐段出现在终端里。上面的 README 示例用默认采样，输出不保证每次相同；要展示一条可逐字核对的输出，得换用温度 0 的 v0.13.1 归档运行（采样确定性实验，实录见附录 D 第八节）。下面的 `gemma-4-e4b` 是本地注册名称，须先按附录 C 第一节导入模型。前面的直接下载运行不会建立这个名称。命令与输出如下：

```bash
litert-lm run gemma-4-e4b --backend cpu \
  --prompt "Write one sentence about the ocean." \
  --temperature 0 --seed 42 --cache disk
```

```text
The vast, mysterious ocean covers over seventy percent of the Earth's surface,
teeming with diverse life and holding immense power.
```

归档实验中这条命令连续运行两次，输出逐字一致；但确定性只在同一环境内成立：2026 年 8 月用重装的新版运行时复测，同一命令在温度 0 下给出了另一句（实录见附录 D 第八节）。可复现的是“同一环境内重复运行结果一致”，不是“任何机器都得到这一句”；采样参数如何影响输出，见第 5 章。

命令行工具本身没有多少需要展开的：它用 click（Python 的命令行框架）把 `run`、`benchmark` 等子命令注册在同一个入口下，一个子命令一个模块。`run` 的职责只有三步：解析参数，创建 `Engine` 并开启会话，随 Python 上下文管理器退出时关闭并释放原生句柄；至于各后端何时真正释放设备资源，由其实现自行决定。命令行只是一层很薄的封装，需要细看的是它之下三个和直觉不同的行为。

第一个行为就在刚才的输出方式里：文字逐段出现，但一次迭代拿到的既不是一个 token，也不是一次 decode step。`run` 迭代 `send_message_async` 返回的 stream，打印每个响应字典中的文本项：

```python
# python/litert_lm_cli/commands/run.py:105
    stream = conversation.send_message_async(prompt)

  try:
    for chunk in stream:  # (1)
      content_list = chunk.get("content", [])
      for item in content_list:
        if item.get("type") == "text":
          state.close_channel()
          click.echo(click.style(item.get("text", ""), fg="yellow"), nl=False)
```

代码行 `(1)`：每次迭代取得的是 Python 层生成的响应字典。`Conversation.send_message_async` 先把原生回调字符串解析为字典，随后按工具调用配置选择相应的 `yield` 分支；当前 step 没有可输出文本时，C++ decode 循环还会跳过回调。所以不能用 stream 迭代次数计算 token 数或吞吐；回调与 token 的完整对应关系见第 5 章。

第二个行为：按下 Ctrl-C，生成不会立即停。`run` 捕获 `KeyboardInterrupt` 后调用 `cancel_process()`，随后继续消费 stream，直至后台处理结束：

```python
# python/litert_lm_cli/commands/run.py:126
  except KeyboardInterrupt:
    conversation.cancel_process()
    for _ in stream:
      pass
```

取消标志只在预设检查点被读取，decode 侧的检查点位置见 2.6 节，完整调用链见第 4 章。当前的 `Prefill` 函数不接收取消标志，正在执行的 prefill 不会轮询取消请求。

第三个行为要看数据才能发现：换一个后端，速度甚至输出都可能变化。定量证据在 2.4 节的后端敏感度数据与第 8 章；这里先说明一点：后端是命令行可选的参数，下面的 C++ 入口也有这个参数。

从源码构建的读者可以用另一个入口：C++ 命令行入口程序 `litert_lm_main`（构建命令见第 11 章与附录 C）。用法和 Python 入口对应，指定模型文件与后端即可，后端默认是 `gpu`：

```bash
litert_lm_main --backend=cpu --model_path=<你的模型>.litertlm
```

对本书而言，这个入口程序有两点需要说明。第一，它默认开启 benchmark 记录，2.2 节解读的四项指标就产自这条路径。第二，它的主流程很短：读入模型资产，解析后端选择，生成引擎设置，交给工厂创建 Engine，最后异步提交消息、等待生成结束；工厂怎样按后端选择创建具体的执行器，是第 8 章的主题。

至此 Python 与 C++ 两个入口都能运行。输入进入对话与 Session，经过 prefill、decode，由回调返回结果。第 3 至 5 章沿这条链分别展开状态管理、两阶段执行和输出处理。

## 2.2　benchmark 输出的四项指标

上一节验证的是功能，本节起转向性能。第 1 章的推算要靠实测的性能数据来检验，数据来自 `litert-lm benchmark`：它用参数指定 prefill 与 decode 的 token 数，并打印四项指标，即 prefill 吞吐、decode 吞吐、初始化时间与 TTFT。下面仍使用 v0.13.1 的本书主基准矩阵，不表示 v0.17.0 性能。主矩阵（完整数据集与采集方法见附录 D）的一组中位数如下，条件为 Apple M5 Pro、Gemma 4 E4B、cpu 后端、prefill 256 token、decode 128 token。

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

v0.17.0 的 CLI 默认先执行一次不计入结果的 warmup，再运行 `--runs` 指定的测量次数（默认 1）。每次 `Benchmark.run()` 都创建并销毁自己的 Engine 与 Session，不能把它理解为同一引擎连续多轮对话。多次测量时，CLI 对 TTFT 和两项吞吐取算术平均，Init 取首个测量结果；本书主矩阵仍按归档脚本取中位数。

四项数据由 C++ `BenchmarkInfo` 提供，C API 分别暴露读取函数：

| 指标 | 源码定义 | 对比时须固定的条件 | C API |
|---|---|---|---|
| TTFT（s） | 首轮 prefill 完整耗时，加首轮 decode 的平均单 token 耗时；不含初始化 | prompt 长度、prefill 形状、首轮 decode turn 的 token 数 | `litert_lm_benchmark_info_get_time_to_first_token` |
| prefill 吞吐（tokens/s） | 本 turn 的输入 token 数除以耗时 | 序列长度、固定形状填充率、后端与 kernel | `litert_lm_benchmark_info_get_prefill_tokens_per_sec_at` |
| decode 吞吐（tokens/s） | 本 turn 的生成 token 数除以耗时 | 上下文长度、KV cache、采样、后端与 kernel | `litert_lm_benchmark_info_get_decode_tokens_per_sec_at` |
| 初始化时间（s） | C API 将 `GetInitPhases()` 中各条 duration 相加；阶段可能重叠 | 文件缓存、模型映射、delegate 初始化与缓存 | `litert_lm_benchmark_info_get_total_init_time_in_second` |

> 表 2-1　benchmark 四项指标的源码定义、对比时须固定的条件与对应的 C API 读取函数。

这份输出里有三个地方容易被误读：Init time 的统计口径、TTFT 的计时方式，以及两个吞吐指标之间的关系。下面分别说明。

1. **Init time** 不是一段没有重叠的启动耗时，而是多个初始化阶段各自耗时的简单相加。引擎创建时，既记录了一个覆盖全程的总计时阶段，也记录了读模型、读元数据等子阶段。这些时间区间互相重叠，直接相加后的总和并不对应任何一段真实经过的时间。这个数字只适合在相同口径下做对比；如果要分析某个初始化步骤本身，需要查看各阶段的原始记录，或者自己额外计时；本书基准数据保留的就是这个 API 口径。

2. **TTFT** 不是从请求发出后一直计时到首个流式回调的实测值，而是根据其他数据计算出来的。例如 cpu/256 档，用同一批 turn 数据复算：\\(256 \div 65.6 + 1 \div 24.8 \approx 3.94\\) s。这个结果与上面输出的 3.94 s 一致，说明指标定义和数据记录内部是自洽的，并不是一次独立测量。cpu/4096 档同理：\\(4096 \div 226.5 + 1 \div 20.7 \approx 18.13\\) s。

3. **两个吞吐指标** 分别来自 prefill（预填充）和 decode（解码）阶段，不能混成一个数值来看。混合后的数字既不代表 prefill，也不代表 decode。即使在模型、设备、后端都相同的情况下，它们仍会分别受到序列长度、固定 prefill signature 的填充率以及 kernel 实现方式的影响。因此，本书引用基准数据时，会同时注明设备、模型、后端和上下文等条件。

## 2.3　计时器与测量语义

这些数字的可信程度，取决于计时器实际测量的是什么。源码里的计时单位叫 turn：每执行一次 prefill 或 decode 调用，就记下这次处理的 token 数和耗时；两项吞吐就是各自 turn 的 token 数除以耗时，没有其他成分。2.2 节提到 TTFT 是计算出来的，其依据就在源码实现里：

```cpp
// runtime/engine/io_types.cc:352
  double first_decode_token_seconds = absl::ToDoubleSeconds(
      decode_turns_[0].duration / decode_turns_[0].num_tokens);  // (1)
  double first_prefill_token_seconds =
      absl::ToDoubleSeconds(prefill_turns_[0].duration);  // (2)
  return first_decode_token_seconds + first_prefill_token_seconds;
```

代码行 `(1)` 取首轮 decode turn 的平均单 token 耗时，`(2)` 取首轮 prefill 的完整耗时，函数返回二者之和。整个过程没有测量“首个流式回调发生的时刻”。因此，当首轮 decode turn 包含多个 token 且各 token 耗时不同时，这个和不等于用户真实等到首字的时间。C API 也明确规定该值以秒为单位返回，并且不包含初始化时间。

测量本身还改变了被测路径。v0.17.0 的 Python benchmark 在 GPU 路径要求等待权重上传完成，CLI 还默认启用可用的局部注意力环形缓冲。除此之外，任务层要求 benchmark 的 prefill 等待完成：

```cpp
// runtime/core/tasks.cc:562
  // Wait for prefill to complete if benchmark mode is enabled.
  params.SetWaitForCompletion(wait_for_completion | benchmark_info.has_value());
```

`SetWaitForCompletion` 会让 benchmark 请求同步完成 prefill，避免只记录异步提交本身消耗的 host 时间。具体后端如何实现等待，见第 4 章。停止判断也有测量专用条件：只要 benchmark 指定的 decode 步数大于 0，即使命中停止序列也不会提前结束，循环会在达到指定步数后停止。这样，每次吞吐记录都使用相同的 decode step 数。

因此，benchmark 的数字反映的是启用等待、固定 decode 步数后的测量结果，不能直接等同于真实对话的端到端时延。报告结果时，需要同时给出参数与执行模式。

解读一组结果时，可以按以下顺序检查：

1. Init 先看缓存冷热：磁盘缓存是冷是热、delegate 初始化与编译缓存是否命中，状态不同的 Init 不能直接比较；这些阶段将在第 7 章展开。
2. TTFT 先拆开：按源码公式分成首轮 prefill 耗时与首轮 decode 的平均单 token 耗时，看哪段占比大，再决定分析方向。
3. prefill 吞吐先对照序列长度、signature 形状与后端：吞吐低可能是计算慢，也可能是填充多，单看 tokens/s 分不出来。
4. decode 吞吐先记下上下文长度、采样配置与后端，再和带宽侧上限对照；上下文扫描只能显示成本随长度增长的趋势，要分清 KV 流量与计算各占多少，需要硬件计数器或更细粒度的测量。

### 2.3.1　从请求开始测到实际文本回调

测量首段文本的等待时间，需要在客户端记录请求开始与首个非空回调的时刻，而不是用平均 decode 耗时替代。附录 D 第十四节给出一组 v0.13.1 手机数据：HONOR MEP-AN00 使用 Gemma 4 E4B、GPU OpenCL 和设备侧采样，输入为 77 token，上下文上限 4096，该组连续执行 5 个独立会话，每次 decode 计数为 201 token。

在这组关闭内存采集的对照中，引擎加载完成后的首次请求，从客户端请求开始到首段文本为 1663 ms，同一引擎上的后续 4 次为 489–498 ms；若把首次加载也纳入等待，从客户端启动计时点到首段文本约为 37791 ms。三个数字分别回答“首次请求要等多久”“同进程后续请求要等多久”和“还没有引擎时要等多久”，不能合并成一个 TTFT，也不能用这 5 次运行估计 P99。

这些时延来自本书的 C API 测试客户端，计时包括文件读取、模板渲染、会话创建，以及首段文本生成前的推理与调度；客户端保留 benchmark 对 prefill 的同步等待，但不强制固定生成步数。记录终点是首个非空文本回调的入口，尚未包含界面排版、绘制和屏幕刷新；含加载时延的启动计时点位于 `main` 内，没有覆盖此前的动态库加载。集成到应用时，应把应用入口和首次显示事件另行计时，这组分段指标则用于解释差异。

## 2.4　Roofline 分析框架

清单的最后两条问的是同一件事：限制一个吞吐数字的是算力还是带宽。要回答这个问题，可以借助第 1 章介绍的 Roofline 模型：token 生成速率由算力上限和带宽上限中更低的那条决定（公式见 1.4 节）。如果带宽上限更低，就称为受带宽约束（memory-bound）；如果算力上限更低，则称为受算力约束（compute-bound）。问题在于，端到端测量只能得到一个 tokens/s 的点值，单凭这一个数字无法判断哪条上限在起作用。

第 1 章根据算术强度给出了一个分阶段判断：prefill 通常受算力约束，batch=1 的稠密 decode 通常受带宽约束。第 1 章计算出的 25 tokens/s 是基于假想手机的设定，不能直接套用，但分析方法本身可以检验。下面用两组数据检验这个判断在本书所用的这台 Mac 上是否成立。

第一组数据看后端敏感度。在主矩阵中，对于 Gemma 4 E4B、context 1024、decode 128 token 的配置，从 cpu 切换到 gpu 后，prefill 吞吐从 259.2 提升到 999.1 tokens/s（约 3.9 倍），decode 从 24.7 提升到 50.6 tokens/s（约 2.0 倍）。prefill 对后端的敏感度明显更高，这与分阶段判断的方向一致。不过，切换后端同时改变了有效算力、有效带宽、delegate 和 kernel 实现，因此这组倍率只能作为旁证，不能单独证明 prefill 已经受算力约束、decode 已经受带宽约束。

第二组数据把 decode 吞吐换算成等效读取速率，再与这台 Mac 的标称带宽进行对照。Apple 公布的 M5 Pro 统一内存带宽最高为 307 GB/s。[^ch02-m5pro-bandwidth] 本书基准模型的 `.litertlm` 文件总大小为 3.66 GB，包含多个模型段；对模型文件做静态分析，识别出主 decode 段 payload 为 2.26 GB。假设每个 decode step 恰好将这 2.26 GB 完整读取一遍，并忽略其他流量，那么 gpu/256 档的等效读取速率为 \\(50.6 \times 2.26 \approx 114\\) GB/s，cpu/256 档为 \\(24.8 \times 2.26 \approx 56\\) GB/s。两者都低于标称带宽，量级上自洽。但需要明确边界：这两个数字是吞吐量与假设 payload 的乘积，并非硬件计数器实测的 DRAM 流量；它们与 307 GB/s 之间的差距同时包含了计算耗时、实际带宽利用率和假设误差，因此既不能解读为带宽利用率，也不能用来反推或核验标称值。

<figure>
{{#include figs/fig-2-1.svg}}
<figcaption>图 2-1　Roofline 给出计算侧和带宽侧两条上限；prefill 与 decode 的工作点位置取决于序列、模型、缓存和 kernel 条件。</figcaption>
</figure>

### 2.4.1　用吞吐差异估算上下文相关开销

主矩阵里还有一个问题。上下文从 256 加长到 4096 后，cpu 的 decode 吞吐从 24.8 降到 20.7 tokens/s，gpu 从 50.6 降到 45.6 tokens/s（模型、设备、后端都没变）。减速的一个可能原因是"上下文长了，每步要多读 KV cache"——第 6 章会按张量形状算出，本书基准模型的 KV cache 每个上下文 token 占 28 KiB。本节做一次核对：把实测的减速也折算成"每个上下文 token 多读了多少字节"，看它是否接近 28 KiB。接近，减速就能用多读 KV 解释；相差很大，说明还有别的成本在增长。

折算沿用第 1 章的思路：decode 受带宽约束时，每 token 耗时约等于每 token 读取的字节数除以有效带宽 \\(B_{\mathrm{eff}}\\)（见 1.4 节）。据此列两个式子。短上下文档假定只读取权重，每 token 耗时 \\(1/R_s = D_w/B_{\mathrm{eff}}\\)，其中 \\(D_w=2.26\\) GB 是主 decode 段大小、\\(R_s\\) 是短档吞吐；长上下文档每 token 多读取 \\(S \times d_{\mathrm{eq}}\\) 字节，\\(S\\) 是上下文长度，\\(d_{\mathrm{eq}}\\) 就是要求的"每上下文 token 附加字节"，于是 \\(1/R_l = (D_w + S\,d_{\mathrm{eq}})/B_{\mathrm{eff}}\\)。两式相减并消去 \\(B_{\mathrm{eff}}\\)，得

$$ d_{\mathrm{eq}}=\frac{D_w}{S}\left(\frac{R_s}{R_l}-1\right) $$

这两个式子有三个假设：两档共用同一个 \\(B_{\mathrm{eff}}\\)，全部时间都能写成字节除以带宽，短上下文档自身的上下文相关流量忽略不计。取 \\(S=4096\\) 代入：cpu 数据给出约 107 KiB/token，gpu 数据给出约 59 KiB/token。

核对结果：107 和 59 分别是 28 的约 3.8 倍和 2.1 倍，减速不能只用"多读 KV"解释。\\(d_{\mathrm{eq}}\\) 把注意力计算量的增长、带宽利用率的变化、缓存行为等一切随上下文增长的成本全都换算成了字节，KV 读取只是其中一部分；它不是实测的 DRAM 流量，也不能单独证明降速全部来自 KV cache。这个对照只确定一件事：上下文相关的成本不能只用权重读取来解释。第 6 章再按 KV 张量的形状和访问路径细算。

## 2.5　二十个问题

在能够运行和测量之后，接下来的需求是定位，本章用三份索引回答：问题在哪章解答（本节），一段代码归哪层管（2.6 节），当前这个模型文件里有什么（2.7 节）。第一份索引是表 2-2，它把二十个问题与解答章节关联起来，其中有几个已经在本章出现过：问题 3 对应 2.4 节数据中 prefill 近千、decode 只有几十 tokens/s 的速度差异；问题 4 则是 2.2 节拆解过的 TTFT 构成。表中只列出索引，各章会进一步说明结论的适用条件。

| # | 问题 | 解答章 |
|---|---|---|
| 1 | 同样的模型，为什么云端可以流畅运行，手机端通常更受限制？ | 1 |
| 2 | 8 GiB 内存的手机能否容纳 4B 参数模型？ | 1、6、7 |
| 3 | prefill 每秒近千 token、decode 只有几十，为什么？ | 1 初讲、2 检验、6 深化 |
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
| 15 | INT4 量化减少的是文件大小、带宽需求还是计算量？ | 7 |
| 16 | `.litertlm` 单文件包含哪些分段？ | 7 |
| 17 | 切换后端后，速度和输出为什么可能变化？ | 8（`LiteRT-LM#2281`[^ch02-issue-2281]） |
| 18 | GPU 设备侧采样（device-side sampling）减少了哪些 host/device 数据传输？ | 8 |
| 19 | speculative decoding 在什么条件下提高吞吐，又会增加哪些开销？ | 9（`LiteRT-LM#2227`[^ch02-issue-2227]） |
| 20 | 图像怎样编码并进入语言模型？约束解码怎样限制结构化输出？ | 10 |

> 表 2-2　二十个推理与运行时问题；许多问题由多个章节共同解答，表中同时列出了各部分的对应位置。

表中没有列出与语言绑定相关的问题；关于各语言绑定如何复用核心 runtime、以及原生边界如何划分，请参见第 11 章。

## 2.6　五层职责视图

第二份索引回答“代码归哪层管”。第 1 章 1.1 节的分层把 LiteRT-LM 放在应用与 LiteRT 之间；本节换个角度，按主要职责把 LiteRT-LM 的生成路径分成五层。这样，读到任何一段代码时，就能回答“这段代码在哪层、归哪章管”。这是分析视图，不是仓库声明的强制依赖规则。主调用路径大致自上而下，但工厂、元数据、日志和工具代码仍可能跨相邻层。

1. 对外接口层。包括 Engine、Session、CLI、C API 与各语言绑定，应用通常从这里进入（第 3、11 章）。
2. 对话与编排层。把多轮消息转换为模型输入，并组织 prefill、decode、取消和回调（第 3 至 5 章）。
3. 推理执行层。由 executor 管理推理状态、调用 LiteRT 执行模型，并适配不同后端（第 6、8、9 章）。
4. 组件层。包括 tokenizer、采样器和约束解码等可复用组件（第 5、10 章）。
5. 格式与基础设施层。包括 `.litertlm` 文件格式、模型资源和通用运行设施（第 7 章）。

<figure>
{{#include figs/fig-2-2.svg}}
<figcaption>图 2-2　本书用于定位职责的五层视图；主生成路径自接口向执行与模型资源传递，跨层辅助依赖未在图中展开。</figcaption>
</figure>

前三层的边界需要分别说明；第四、五层偏工具性质，代码位置最后说明。

对外接口层的主要入口之一是 `SessionInterface`。类注释说明，Session 保存一次独立交互的内部状态，并负责生成、prefill 与 decode：

```cpp
// runtime/engine/engine.h:119
  ABSL_DEPRECATED(
      "Prefer Conversation API for chat/context management, or RunPrefill and "
      "RunDecode for fine-grained execution control.")
  virtual absl::StatusOr<Responses> GenerateContent(  // (1)
      const std::vector<InputData>& contents) = 0;
// ...
  virtual absl::Status RunPrefill(const std::vector<InputData>& contents) = 0;  // (2)
// ...
  virtual absl::StatusOr<Responses> RunDecode() = 0;  // (3)
```

代码行 `(1)` 的 `GenerateContent` 在一次调用中处理 prefill 和 decode，但已被标记为弃用。聊天与上下文管理应使用 Conversation；需要单独控制两个阶段时，使用 `(2)`、`(3)` 的接口。`= 0` 表示这些方法由具体 Session 实现提供，接口本身不绑定后端。

对话与编排层的核心是 prefill 和 decode 的调度循环。`Prefill` 在调用 executor 之前会校验输入 token 数是否严格小于 executor 给出的上限，超限直接拒绝；这个上限与 KV cache 容量的关系见第 6 章。decode 循环里各步骤的顺序决定了取消和输出的语义：每轮先读取消标志，再执行一步 decode，有可输出的文本就发一次流式回调，轮末统一检查停止条件，包括停止序列、benchmark 步数、`max_num_tokens` 和 `max_output_tokens`。这个顺序决定了取消的行为：取消不会中断已经进入 executor 的当前 step，任务层在下一轮检查点观察到取消；内部采样还会把标志传给 executor，是否提前响应取决于具体实现。第 4、5 章分别展开取消、停止序列与 UTF-8 输出处理。

推理执行层的公共边界是 `LlmExecutorBase`：

```cpp
// runtime/executor/llm_executor_base.h:48
  virtual absl::Status Prefill(const ExecutorInputs& inputs) = 0;  // (1)
// ...
  virtual absl::StatusOr<std::vector<std::vector<int>>> Decode() = 0;  // (2)
// ...
  virtual absl::string_view ExecutorBackendName() const = 0;  // (3)
```

代码行 `(1)` 定义了基本的 prefill 接口，编排层实际使用的是它的参数化重载；基类默认返回未实现，支持该路径的具体 executor 需要覆盖它。`(2)` 定义基本 decode 接口；`(3)` 返回 executor 的后端名称。工厂选择的 executor 与 delegate 配置可能因 CPU、GPU、NPU 路径而异，但上层编排仍通过这些公共方法调用。

组件层包括运行时组件与共享支持库中的 tokenizer、采样器等；文件格式与模型资源分别管理分段描述和内容加载。

图 2-3 沿调用方向展开一次生成请求的主数据流：

<figure>
{{#include figs/fig-2-3.svg}}
<figcaption>图 2-3　一次生成请求的主数据流；侧框列出各阶段使用的输入、组件和后端资源。</figcaption>
</figure>

`SessionInterface` 和 `LlmExecutorBase` 划出的是抽象边界，不隔离后端差异：工厂和设置对象会把后端选择传给 executor 和 delegate，因此上层编排代码可以跨后端复用，但具体能力仍需逐后端检查。会话状态由 Session 及其 executor 上下文持有；Clone、checkpoint 与 rewind 各自复制或调整哪些状态，需要看第 6 章的具体实现。

## 2.7　`.litertlm` 文件的组成

最后一份索引是模型文件本身。从 2.1 节起，一个 3.66 GB 的 `.litertlm` 文件一直在使用：`run` 命令下载了它，`--model_path` 指向它，2.4 节还用到了它内部主 decode 段的大小。这一节说明它包含什么。`.litertlm` 是 LiteRT-LM 定义的容器格式，一个文件可以包含 TFLite 模型、tokenizer、LLM 元数据和通用二进制数据等多种 section，具体组合由模型文件决定。要查看文件内容，可以使用 `litertlm_print`：它先读取文件头并输出系统元数据，然后逐个列出每个 section 的字节范围和类型；遇到 LLM 元数据时，还会解析 proto 并输出文本。下面展示的是一份三段示例文件的节选，并非本书基准模型；基准模型实际含 10 个 TFLite 模型段，完整清单见附录 D 第六节。

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
  Data Type:    AnySectionDataType_TFLiteModel     # (1)
Section 1:
  Begin Offset: 3579204608
  End Offset:   3583074304
  Data Type:    AnySectionDataType_SP_Tokenizer    # (2)
Section 2:
  Begin Offset: 3583074304
  End Offset:   3583078400
  Data Type:    AnySectionDataType_LlmMetadataProto
    <<<<<<<< start of LlmMetadata
      start_token { token_ids: 2 }                 # (3)
      stop_tokens { token_str: "<end_of_turn>" }   # (4)
      prompt_templates { ... }                     # (5)
    >>>>>>>> end of LlmMetadata
```

`(1)` 是存放权重和计算图的 TFLite 模型段，`(2)` 是 SentencePiece 分词器；这个示例把它们和 LLM 元数据分别放在三个 section 中，其他模型可以采用不同布局。在元数据段内，`(3)` 的 `start_token` 是 BOS，`(4)` 的 `stop_tokens` 对应停止序列配置（第 12 问），`(5)` 的 `prompt_templates` 是聊天模板（第 7 问）。构建 Session 配置时，源码会依次读取这三项；它们如何影响输入构造和停止判断，见第 3、5 章。

section 的定位规则很简单：`begin_offset` 和 `end_offset` 定义半开区间 `[begin, end)`，schema 注释直接给出了这一约定，读取函数也使用 `end_offset - begin_offset` 计算大小；`data_type` 用于区分 section 的内容，常见取值包括 `TFLiteModel`、`SP_Tokenizer`、`HF_Tokenizer_Zlib`、`LlmMetadataProto` 和 `GenericBinaryData`。

TFLite section 的读取路径先用 `end_offset - begin_offset` 算出模型大小，然后并不将权重整块读入内存，而是通过 mmap 把模型段映射到地址空间。映射之后，页错误、物理驻留和初始化耗时都取决于操作系统和实际访问行为，必须实测；第 7 章会结合 section 布局与冷启动进一步展开。

## 小结

本章完成了三件事。第一，建立了运行起点：Python 与 C++ 两个入口都能运行，对流式片段与 token 的关系、取消的生效时机、后端的可选性都有了直接观察。第二，明确了测量方法：四项指标按 `BenchmarkInfo` 的源码定义来读，TTFT 是计算值，Init 各阶段可能重叠，单点 tokens/s 无法判断是算力约束还是带宽约束。第三，给出了三份索引：二十个问题标注了对应章节，五层视图用于定位代码职责，`.litertlm` 的 section 布局说明了模型文件的组成。接下来第 3 至 5 章沿调用链继续推进，依次讨论 Session 状态、prefill/decode 编排与输出处理。

## 练习与自查

1. **TTFT 验算**：使用主矩阵的 gpu/1024 档数据（prefill 999.1 tokens/s、decode 50.6 tokens/s），按 `GetTimeToFirstToken` 的公式复算 TTFT，并说明为什么这不是一次独立计时。

2. **等效字节估算**：使用 gpu 在 context 256 与 4096 两档下的 decode 吞吐（50.6 与 45.6 tokens/s）和 2.26 GB 主段 payload，计算 \\(d_{\mathrm{eq}}\\)，并说明它为什么不能当作 KV cache 的实测 DRAM 流量。

3. **条件检查**：某设备报告 Init 1.8 s、TTFT 4.4 s、decode 45 tokens/s，上下文加倍后 decode 下降 8%。在判断是否存在异常之前，列出还需要补充的模型、输入、后端和测量条件。

4. **代码定位**：找到 benchmark 模式要求 prefill 等待完成的语句，并说明如果不等待，计时可能覆盖什么范围。

5. **架构归位**：约束解码中的 `ProcessLogits` 调用属于五层视图中的哪一层？它接收的 logits 来自哪一层？

[^ch02-litertlm-readme]: Google AI Edge，[LiteRT-LM README](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.17.0/README.md#L88-L98)，版本 v0.17.0；访问日期：2026-09-13。
[^ch02-m5pro-bandwidth]: Apple，[*Apple debuts M5 Pro and M5 Max to supercharge the most demanding pro workflows*](https://www.apple.com/au/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)，2026-03-04；访问日期：2026-08-30。
[^ch02-issue-2568]: Yegorsh，[*`--max-num-tokens` unreasonably affects decoding speed*](https://github.com/google-ai-edge/LiteRT-LM/issues/2568)，LiteRT-LM issue #2568，2026-06-13；访问日期：2026-07-18。
[^ch02-issue-2281]: 4ntoine，[*Different inference result depending on backend*](https://github.com/google-ai-edge/LiteRT-LM/issues/2281)，LiteRT-LM issue #2281，2026-05-15；访问日期：2026-07-18。
[^ch02-issue-2227]: Shoolife，[*MTP / speculative decoding regresses decode tok/s on PowerVR GPU (Tensor G6) — even with GPU sampler fully loaded*](https://github.com/google-ai-edge/LiteRT-LM/issues/2227)，LiteRT-LM issue #2227，2026-05-11；访问日期：2026-07-18。
