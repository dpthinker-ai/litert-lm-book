# 附录 D · 基准数据集

> 书中所有标注“〔基准 D〕”的实测数据都来自本附录定义的数据集。采集条件、原始记录与结果分开列出，避免把实测与理论估算混为一谈。

## 采集状态

> 主矩阵采集于 2026-07-05，扩展实验采集至 2026-07-18。Mac 记录共 30 次：18 次 backend × context 主矩阵，以及 12 次 `false`/`auto` 再采样；Android、自然文本 MTP、线程数和约束解码结果另列，不与主矩阵混算。原始记录位于 `experiments/data/`。

## 一、采集环境（固定，不可混）

| 项 | 值 |
|---|---|
| 芯片 / 内存 | Apple M5 Pro / 24 GiB |
| 系统 | macOS 26.5 |
| 主基准模型 | Gemma 4 E4B（`litert-community/gemma-4-E4B-it-litert-lm`，公开、支持 MTP，3.66 GB ≈ 3.4 GiB；SHA-256 `0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0`） |
| 后端 | cpu、gpu（Metal） |

主模型支持 MTP，因此第 9 章的投机解码实测与主基准使用同一个模型。

## 二、方法

- 矩阵：backend ∈ {cpu, gpu} × prefill/上下文 ∈ {256, 1024, 4096}，decode 固定 128 token。
- 重复：每个条件跑 3 次，取中位数（抵消抖动）。
- 指标：prefill tokens/s、decode tokens/s、Init API 聚合值（s）、TTFT（s）。（`litert-lm benchmark` 不报峰值内存，故本表不含该列。）C API 会把 `GetInitPhases()` 中可能重叠的阶段 duration 相加，因此 Init 列不是无重叠的端到端墙钟时间。
- 证据范围：本书的性能数字只引用本数据集与注明出处的官方数据。更换后端、模型或机器后，结果作为另一组数据单独标注，不参与混合计算（第 8 章）。
- 原始 CSV 存 `experiments/data/baseline.csv` 与 `mtp.csv`，元信息存 `experiments/data/_meta.txt`。

## 三、结果

主基准使用 Gemma 4 E4B，decode 长度为 128 token；表中数据为各条件的中位数：

| backend | 上下文 | prefill tokens/s | decode tokens/s | Init API 聚合值（s） | TTFT（s） |
|---|---|---|---|---|---|
| cpu | 256 | 65.6 | 24.8 | 0.54 | 3.94 |
| cpu | 1024 | 259.2 | 24.7 | 0.54 | 3.99 |
| cpu | 4096 | 226.5 | 20.7 | 0.57 | 18.13 |
| gpu | 256 | 259.8 | 50.6 | 1.76 | 1.01 |
| gpu | 1024 | 999.1 | 50.6 | 1.77 | 1.04 |
| gpu | 4096 | 925.2 | 45.6 | 1.78 | 4.45 |

投机解码使用同一模型，context 为 1024；表中数据为 decode 吞吐中位数：

| MTP | cpu | gpu | 说明 |
|---|---|---|---|
| 关（false） | 22.8 | 50.0 | 基线 |
| auto | 24.9 | 50.2 | `auto` 沿用默认的关闭行为（第 9 章），此行与基线是同行为的再采样 |

> 注：MTP 表与主表来自两个独立采集批次，故 cpu“关”的 22.8 与主表同条件的 24.7 不必相等。该批三次运行为 20.1 / 22.8 / 24.9（极差 4.8）；gpu“关”批首跑 44.1，后两次为 50.0 / 50.1。`false` 与 `auto` 在 v0.13.1 中都是关闭行为，表中差异只反映再采样波动。当前没有可核查的 Mac 强制开启记录；有记录的强制开启对照见第十三节 Android 扩展实验。

## 四、结果使用边界

各表只支持所列设备、模型、后端、输入和采集方法下的观察。端到端吞吐不能单独分离计算、访存、同步与调度成本；相应机制与结果解读见正文各章。

## 五、实验索引

各章实验的完成状态、复现入口与未形成结果的项目见附录 C，本附录不重复列出。

## 六、模型文件分析（gemma-4-e4b model.litertlm）

本次分析不依赖 `litertlm_print`：先按 16 KiB 对齐边界查找 TFLite 魔数 `TFL3`，再用 TFLite schema 的 FlatBuffers 绑定读取每段的 signature 与张量形状。文件 3.66 GB，共 10 个 TFLite 段；完整记录见 `experiments/data/model_anatomy.md`：

| 段起点（字节） | 大小 | 签名 | 关键张量 |
|---:|---:|---|---|
| 4,734,976 | 171 MB | `embedder` | `token_ids[1,1]` |
| 175,669,248 | 837 MB | `per_layer_embedder` | `token_ids[1,1]` |
| 1,012,449,280 | 94 MB | `serving_default`（音频编码器） | `mask[1,1,816]` |
| 1,106,509,824 | 16 MB | `audio_adapter` | `features[1,204,1536]` |
| 1,122,254,848 | 16 KB | `eoa` | 不适用 |
| 1,122,271,232 | 224 MB | `vision_70/140/280` | `images[1,1260,768]` |
| 1,346,420,736 | 8 MB | `vision_adapter_70/140/280` | `soft_tokens[1,140,768]` |
| 1,354,317,824 | 16 KB | `eoi` | 不适用 |
| 1,354,334,208 | 2,260 MB | `decode` / `prefill_1024` / `prefill_128` / `verify` | `embeddings[1,1,2560]` |
| 3,614,392,320 | 45 MB | `mtp_drafter` | `activations[1,1,5120]` |

主干模型 `decode` signature 的关键事实：

- KV cache 输入 48 个张量 = 24 层 × (K+V)，dtype 全部为 INT8；20 层 `[1,2,32003,256]`、4 层 `[1,2,32003,512]`（V 侧维度转置存放）。
- KV 每 token = 2 × 2 × (20×256 + 4×512) × 1 B = 28,672 B = 28 KiB；4096 上下文 = 112 MiB；文件中的宽度 32003 是 magic number 占位值（见 6.2.2 节），实际静态宽度由 `--max-num-tokens` 决定，默认上限 32000 对应 875 MiB。
- `embeddings[1,1,2560]` → model_dimension = 2560；`per_layer_embeddings[1,1,42,256]`；logits `[1,1,262144]` → 词表 262,144 = 2^18；`param_tensor[1,1,1,7]`（单缓冲 KV 路径的位置参数，见第 6 章）。
- prefill 入口集为 {1024, 128}（第 4 章分块示例所用的实际入口）；`verify` 与 `mtp_drafter` 段互相配套（第 9 章）；vision 三档签名与三档 adapter 配套（第 10 章）。
- 元数据中的聊天模板以 `<turn|>` 作为轮次结束标记。模型文件中不存在 `end_of_turn` 字样；这是第 5 章讨论停止符的依据。

## 七、prefill 长度扫描（cpu，-d 32，disk 缓存热，单次）

| prefill tokens | prefill tokens/s | decode tokens/s | TTFT (s) |
|---:|---:|---:|---:|
| 100 | 48.3 | 12.8 | 2.15 |
| 250 | 64.0 | 25.7 | 3.95 |
| 500 | 128.4 | 26.3 | 3.93 |
| 1000 | 257.0 | 25.9 | 3.93 |
| 2000 | 260.9 | 26.1 | 7.71 |
| 3000 | 261.3 | 26.0 | 11.52 |
| 4000 | 229.3 | 20.7 | 17.49 |

基准模型只有 `prefill_128` 与 `prefill_1024` 两个静态入口。250、500、1000 token 都使用 1024-token signature，对应墙钟时间约为 3.91、3.89、3.89 s；吞吐差异主要反映有效 token 占比，不能单独证明算术强度变化。2000 token 以上需要多个分块，现有数据不能分离注意力与调度成本。本表 decode 只运行 32 步，主结果仍以第二节的 128 步矩阵为准。原始数据见 `experiments/data/prefill_sweep.csv`。

## 八、采样确定性实验（cpu）

同一提示词（"Write one sentence about the ocean."）在温度 0、相同种子下运行两次，输出逐字一致。温度 1.0 时，seed 7 与 seed 1 生成了不同句子，seed 1 与 seed 2 则生成了相同序列；启用随机采样不保证每次输出都不同。复现时须固定运行时版本、模型文件、后端、提示词、温度、top-k、top-p 与缓存模式；2026-08-31 以重装的新版运行时复测同一命令，温度 0 得到另一句，确定性不跨环境成立。实录见 `experiments/data/temperature_test.md`。

## 九、`--max-num-tokens` 预留宽度扫描（cpu，2026-07-17 补采）

观察第 6 章所述固定形状路径中预留宽度与 decode 吞吐的关系。脚本 `experiments/max_tokens_sweep.sh`，原始数据 `experiments/data/max_tokens_sweep.csv`（`-d 128`，磁盘缓存热，各 2 次）。实验同时改变了所选 signature 或张量宽度所带来的多项执行成本，未用性能计数器把 KV 访存单独分离出来。

| max_num_tokens | prompt | decode tokens/s | 备注 |
|---:|---:|---:|---|
| 1024 | 100 | 33.9 | 分块 [128] |
| 2048 | 256 | 23.5 / 29.5 | 两次散布大，取区间 |
| 4096 | 256 | 26.4 | 与默认值推导一致（(256+1023)/4096+1）×4096 |
| 8192 | 256 | 21.5 | 同一 prompt，仅放宽预留 |
| 1024 | 256 | 运行失败 | 分块 [1024] 恰好占满预留宽度，prefill 报 `dynamic_update_slice` 维度越界 |

同为 256-token prompt 时，4096 档约 26.4 tokens/s，8192 档约 21.5 tokens/s，后者低约 19%。1024/100 的 33.9 tokens/s 使用不同 prompt 与不同 signature，只能作为不同条件下的复现记录，不能与 8192/256 计算速度比例。1024/256 直接失败，说明预留宽度过小可能触发维度越界。2048 档两次结果散布较大，不据此下点值结论。

## 十、MTP 聚合接受比例实测

Python SDK 将日志级别设为 VERBOSE 后，drafter 析构时会打印 drafted 与 verified token 计数。完整记录见 `experiments/data/mtp_acceptance.md`。

| 输入 | drafted | verified | 聚合比例 \\(r=verified/drafted\\) | \\(G=3\\) 时每轮期望产出 \\(1+3r\\) |
|---|---:|---:|---:|---:|
| 机器人学习绘画的 100 词故事 | 213 | 56 | 0.2629 | 1.79 token |
| Fibonacci 函数及解释 | 3069 | 3054 | 0.9951 | 3.99 token |

这里的 \\(r\\) 是每轮接受前缀长度的聚合比例，不是“各位置具有相同独立命中概率”的 \\(p\\)。端到端加速还取决于 drafter、verify 和固定调度开销，不能只由 \\(r\\) 推出。两个提示词记录到的比例不同，但样本不足以建立一般性的内容规律。benchmark 的 pad 合成输入没有记录 \\(r\\)，因此本表不作为主基准开关结果的直接归因。

## 十一、约束解码开/关工具调用观察

Python SDK 在 Gemma 4 E4B、GPU 后端上比较 `enable_constrained_decoding=True/False`。原始记录见 `experiments/data/constraint_test.md`。

| 场景 | 温度 | 开启约束 | 关闭约束 |
|---|---:|---:|---:|
| `get_weather(city)` | 0 | 2/2 结构合法 | 2/2 结构合法 |
| `get_forecast(city, days, unit)` | 0 | 2/2 结构合法 | 2/2 结构合法 |
| `get_forecast(city, days, unit)` | 1.0 | 2/2 结构合法 | 2/2 结构合法 |

共 12 次生成，开启与关闭各 6 次；所测样本均未出现结构错误。该样本量不足以估计失败率，也未覆盖多工具混淆、嵌套 JSON、参数语义和函数执行结果。结论仅限于“在这些样本中未观察到差异”。

## 十二、未形成指标的扩展记录

- Android 峰值内存：无可报告数据；原始 CSV 位于 `experiments/data/android_*.csv`，采集脚本为 `experiments/android_bench.sh`。
- NPU 端到端推理：无可报告数据；加载与失败阶段记录于 `experiments/data/npu_enablement.md`。

## 十三、扩展基准（Android 真机，2026-07-18 采集）

> 本节数据与主基准（Mac M5 Pro）分开标注，不参与混合计算。设备：P0210（qcom，Android 16）；二进制：自编译 `litert_lm_advanced_main`（arm64，v0.13.1）；模型同主基准。真机 CLI 的 `max_num_tokens` 默认取 prompt 与 decode 长度之和，本次显式对齐为 256/1024 → 4096、4096 → 8192。decode 长度为 128 token，各条件运行 3 次取中位数。脚本为 `experiments/android_bench.sh`，原始数据位于 `experiments/data/android_*.csv`。

### Android 主基准

表中数据为真机各条件的吞吐中位数：

| backend | 上下文 | prefill tokens/s | decode tokens/s | TTFT (s) |
|---|---|---|---|---|
| cpu | 256 | 19.9 | 9.9 | 13.0 |
| cpu | 1024 | 79.1 | 10.0 | 13.0 |
| cpu | 4096 | 62.6 | 7.0 | 65.6 |
| gpu | 256 | 258.3 | 18.8 | 1.04 |
| gpu | 1024 | 957.2 | 18.8 | 1.12 |
| gpu | 4096 | 858.6 | 17.4 | 4.83 |

### MTP 开关

以下结果来自真机，context 为 1024，指标为 decode tokens/s：

| MTP | cpu | gpu |
|---|---|---|
| 关 | 10.0 | 18.0 |
| 强制开 | 2.8（降至关闭模式的约 28%） | 12.6（比关闭模式低约 30%） |

benchmark 的负载是“prompt + pad 填充”（`ids.resize`，`runtime/core/session_utils.cc:68-73`），不是自然文本。在该负载下，强制开启 MTP 的端到端吞吐低于关闭模式；本次运行没有记录聚合接受比例，不能把负收益只归因于接受比例。换用自然代码文本（`benchmark_prefill_tokens=0`）后，同一台手机的单次观测高于关闭模式：

| 负载 | cpu | gpu |
|---|---|---|
| 自然代码文本，关 | 11.6 | 16.0 |
| 自然代码文本，强制开 | 12.6（约 +9%） | 32.0（约 2.0 倍） |

实录见 `experiments/data/mtp_natural_phone.md`。这组自然文本结果每个条件只运行 1 次，prompt、decode 长度和采集入口也不完全相同。它只能说明观测结果随负载与后端变化，不能估计稳定加速比。

### 自然代码文本补测

本次补测采集于 2026-07-18，实录见 `experiments/data/mtp_ceiling.md`。verify signature 的 `input_pos` 形状为 [4]，因此 G=3。温度 0、长代码生成时，Mac GPU 从 58.7 增至 133.9 tokens/s（2.28 倍），手机 GPU 从 18.6 增至 37.4 tokens/s（2.01 倍）。若额外假设每轮都产出最大值 4，可由端到端结果反推出有效成本参数约 0.25 和 0.33。该参数不是 drafter 单步成本的独立测量，也不能证明设备已经达到理论上限。

### CPU 线程数扫描

以下结果来自真机，context 为 1024，指标为 tokens/s：

| 线程数 | prefill | decode |
|---:|---:|---:|
| 1 | 22.8 | 4.4 |
| 2 | 45.3 | 7.4 |
| 4 | 77.9 | 10.1 |
| 8 | 131.6 | 13.4 |
