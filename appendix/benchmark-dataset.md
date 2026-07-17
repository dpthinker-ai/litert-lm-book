# 附录 D · 基准数据集

> 书中所有标注「〔基准 D〕」的实测数据，都来自这里定义的这一套数据集。目的只有一个：让实测可复现、可对照，不与纸面推算混淆。

## 采集状态

> ✅ **已采集（2026-07-05）。** litert-lm 0.13.1，完整矩阵 30 次运行 + 1 次强制 MTP 验证；原始 CSV 在 `experiments/data/`。下表为各条件 3 次的中位数。

## 一、采集环境（固定，不可混）

| 项 | 值 |
|---|---|
| 芯片 / 内存 | Apple M5 Pro / 24 GiB |
| 系统 | macOS 26.5 |
| 主基准模型 | **Gemma 4 E4B**（`litert-community/gemma-4-E4B-it-litert-lm`，公开、支持 MTP，3.66 GB ≈ 3.4 GiB） |
| 后端 | cpu、gpu（Metal） |

主模型本身支持 MTP，因此第 9 章的推测解码实测与主基准**共用同一个模型**，无需另找。

## 二、方法

- **矩阵**：backend ∈ {cpu, gpu} × prefill/上下文 ∈ {256, 1024, 4096}，decode 固定 128 token。
- **重复**：每个条件跑 3 次，取中位数（抵消抖动）。
- **指标**：prefill tok/s、decode tok/s、Init（加载）时间(s)、TTFT(s)。（`litert-lm benchmark` 不报峰值内存，故本表不含该列。）
- **纪律**：正文只引用本数据集与注明出处的官方数据；换后端/模型/机器即另一组数据，单独标注，不混算（第 8 章）。
- 原始 CSV 存 `experiments/data/baseline.csv` 与 `mtp.csv`，元信息存 `experiments/data/_meta.txt`。

## 三、结果

**主基准（Gemma 4 E4B，decode 128 token，各条件中位数）：**

| backend | 上下文 | prefill tok/s | decode tok/s | Init (s) | TTFT (s) |
|---|---|---|---|---|---|
| cpu | 256 | 65.6 | 24.8 | 0.54 | 3.94 |
| cpu | 1024 | 259.2 | 24.7 | 0.54 | 3.99 |
| cpu | 4096 | 226.5 | 20.7 | 0.57 | 18.13 |
| gpu | 256 | 259.8 | 50.6 | 1.76 | 1.01 |
| gpu | 1024 | 999.1 | 50.6 | 1.77 | 1.04 |
| gpu | 4096 | 925.2 | 45.6 | 1.78 | 4.45 |

**推测解码（同一模型，context 1024，decode 中位数 tok/s）：**

| MTP | cpu | gpu | 说明 |
|---|---|---|---|
| 关（false） | 22.8 | 50.0 | 基线 |
| auto | 24.9 | 50.2 | v0.13.1 中 auto 不触碰默认关（第 9 章），此行与基线本质是同行为的再采样 |
| 强制开（true，gpu 单次验证） | — | 49.0 | 不报错、正常运行 → 模型与平台确实支持 MTP；单次临时验证，未入 CSV |

> 注：MTP 表与主表为两个独立采集批次，故 cpu「关」的 22.8 与主表同条件的 24.7 不必相等——该批三次运行为 20.1 / 22.8 / 24.9（极差 4.8），gpu「关」批首跑 44.1、后两次 50.0 / 50.1。因此 MTP 的结论以「差异落在批内抖动幅度内」表述，不取精确点值。

## 四、结果解读（与正文对账）

1. **Roofline 眼镜的实证（第 2 章）**：prefill 对算力/后端极敏感——cpu 从 65.6（短上下文吃不满算力）升到 259.2，gpu 到 999.1，gpu 约为 cpu 的 3.9 倍；decode 却几乎纹丝不动（cpu ≈24.7、gpu ≈50.6），gpu 仅为 cpu 的 2 倍。两类操作被不同资源顶住，实测清晰可见。（两个后端的 prefill 在 4096 档都略有回落，注意力开销随上下文增长，属预期。）
2. **KV cache 占带宽的实证（第 6 章）**：上下文从 256 → 4096，decode 从 24.8 降到 20.7（cpu，-17%）、50.6 降到 45.6（gpu，-10%）——对话越长逐字越慢，正是 KV cache 读写分走带宽的直接证据。另一处自洽：cpu/4096 的 TTFT 18.1 s ≈ 4096 ÷ 226.5（prefill 耗时），公式与实测对上。
3. **编译产物缓存的实证（第 7 章）**：GPU 首次运行（缓存未热）Init 5.29 s，其后稳定 ≈1.77 s——`--cache disk` 的冷启动收益直接可见（原始数据 `baseline.csv` gpu/256 run1）。
4. **MTP 的诚实结果（第 9 章）**：在本基准（benchmark 模式的合成负载）下，auto 与关的 decode 差异在运行间抖动范围内（cpu 22.8→24.9，gpu 50.0→50.2；按第 9 章的代码链路，v0.13.1 中 auto 实为关，唯一真正开启的是强制 `true` 的一组，gpu 49.0，与关同样在抖动内），**远未复现官方"约 3 倍"口径**。这与第 9 章的接受率经济学一致：收益全看 drafter 猜得准不准，而 benchmark 的合成负载接近"最难猜的文体"；官方口径来自其特定的模型/硬件/负载条件。CLI 未输出接受率，无法进一步归因——如实记录。

> 关于第 1 章那条"25 tok/s"：本机 cpu decode 恰为 24.7-24.8，与第 1 章示例数字接近纯属**巧合**——第 1 章算的是一部假想手机（50 GB/s、1.86 GiB 权重），本机是另一套参数。正确的用法是公式本身：按第 1 章公式反推，且分母取 decode 每步真正读取的主干模型段 2.26 GB（见第六节实剖，而非 3.66 GB 整文件），gpu decode 50.6 tok/s × 2.26 GB ≈ 114 GB/s、cpu 24.8 × 2.26 ≈ 56 GB/s 的有效搬运速率，量级落在桌面级统一内存芯片的合理区间（具体带宽规格未查证，不作断言）。对账细节见第 2 章。

## 五、对账清单（哪些结了、哪些还开着）

- ✅ **decode 上限与 KV cache 差额**（第 1、6 章）：结清，见上"结果解读"第 2 条——上下文变长，decode 实测下降 10-17%。
- ✅ **Roofline 分野**（第 2 章）：结清，见"结果解读"第 1 条。
- ✅ **编译缓存的冷启动收益**（第 7 章）：结清，见"结果解读"第 3 条。
- ✅ **MTP 与官方口径对照**（第 9 章）：结清（结果为"未复现"，如实报告），见"结果解读"第 4 条。
- ✅ **KV cache 公式代入**（第 6 章）：结清，见下「六、模型实剖」——真实参数 24 层 / H_kv=2 / D=256（20 层）与 512（4 层）/ **int8**，每 token 28 KiB，4096 上下文 112 MiB。
- ✅ **`--max-num-tokens` 预留宽度与速度**（第 6 章，`LiteRT-LM#2568`）：结清，见下「九、预留宽度扫描」——8192 比 1024 慢约 37%；过小预留直接报错。
- ⬜ **int4 vs int8 三角**（第 7 章）：需社区有同模型两种量化产物，未做。
- ⬜ **分段并行加载开关对比**（第 7 章）：开关仅 C API 暴露，未单测。
- ⬜ **CPU 线程数扫描**（第 8 章）：flag 仅 C++ `litert_lm_main` 暴露（`shared_flags.cc:74`），Python CLI 无此项，未做。
- ⬜ **Python/C++ 行为一致性**（第 11 章）：需 Bazel 构建 `litert_lm_main`，未做（正文已降为推断级）。
- ⬜ **接受率插桩实测**（第 9 章）：需改码重编译，方案见第 9 章，未做。

## 六、模型实剖（gemma-4-e4b model.litertlm）

不依赖 `litertlm_print` 的构建，用两步直接解剖模型文件（脚本思路：段按 16 KiB 对齐，扫对齐边界找 TFLite 魔数 `TFL3` 定段起点；再用 TFLite schema 的 flatbuffers 绑定读每段的 signature 与张量形状）。文件 3.66 GB，共 10 个 TFLite 段：

| 段起点（字节） | 大小 | 签名 | 关键张量 |
|---:|---:|---|---|
| 4,734,976 | 171 MB | `embedder` | `token_ids[1,1]` |
| 175,669,248 | 837 MB | `per_layer_embedder` | `token_ids[1,1]` |
| 1,012,449,280 | 94 MB | `serving_default`（音频编码器） | `mask[1,1,816]` |
| 1,106,509,824 | 16 MB | `audio_adapter` | `features[1,204,1536]` |
| 1,122,254,848 | 16 KB | `eoa` | — |
| 1,122,271,232 | 224 MB | `vision_70/140/280` | `images[1,1260,768]` |
| 1,346,420,736 | 8 MB | `vision_adapter_70/140/280` | `soft_tokens[1,140,768]` |
| 1,354,317,824 | 16 KB | `eoi` | — |
| 1,354,334,208 | **2,260 MB** | `decode` / `prefill_1024` / `prefill_128` / `verify` | `embeddings[1,1,2560]` |
| 3,614,392,320 | 45 MB | `mtp_drafter` | `activations[1,1,5120]` |

主干模型 `decode` signature 的关键事实：

- KV cache 输入 48 个张量 = 24 层 × (K+V)，**dtype 全部 INT8**；20 层 `[1,2,32003,256]`、4 层 `[1,2,32003,512]`（V 侧维度转置存放）。
- **KV 每 token = 2 × 2 × (20×256 + 4×512) × 1 B = 28,672 B = 28 KiB**；4096 上下文 = 112 MiB；静态槽位 32003 全预留 ≈ 875 MiB。
- `embeddings[1,1,2560]` → model_dimension = 2560；`per_layer_embeddings[1,1,42,256]`；logits `[1,1,262144]` → 词表 262,144 = 2^18；`param_tensor[1,1,1,7]`（单缓冲 KV 路径的位置参数，见第 6 章）。
- prefill 入口集恰为 {1024, 128}（第 4 章工单示例的真实版本）；`verify` 与 `mtp_drafter` 段互相配套（第 9 章）；vision 三档签名与三档 adapter 配套（第 10 章）。
- 元数据侧：聊天模板以 `<turn|>` 作轮次收尾标记（解剖可见；`end_of_turn` 字样在本文件中不存在）——第 5 章停止符讨论的依据。

## 七、prefill 长度扫描（cpu，-d 32，disk 缓存热，单次）

| prefill tokens | prefill tok/s | decode tok/s | TTFT (s) |
|---:|---:|---:|---:|
| 100 | 48.3 | 12.8 | 2.15 |
| 250 | 64.0 | 25.7 | 3.95 |
| 500 | 128.4 | 26.3 | 3.93 |
| 1000 | 257.0 | 25.9 | 3.93 |
| 2000 | 260.9 | 26.1 | 7.71 |
| 3000 | 261.3 | 26.0 | 11.52 |
| 4000 | 229.3 | 20.7 | 17.49 |

曲线三段式：短提示词吞吐低（向量单元喂不满、算术强度不足），约 1000 token 后进入 257-261 tok/s 的平台，4000 token 回落到 229（注意力二次项占比上升）。与第 4 章的解读一致。注意本表 decode 列在 -d 32 的短测量窗下抖动较大（p=100 档的 12.8 属预热效应），decode 结论以第二节的 -d 128 矩阵为准。原始数据 `experiments/data/prefill_sweep.csv`。

## 八、采样确定性实验（cpu）

同一提示词（"Write one sentence about the ocean."）：温度 0、同种子跑两次，输出逐字一致；温度 1.0 时输出随种子可变（seed 7 与 seed 1 产出不同句子），但默认参数下 seed 1 与 seed 2 产出了相同序列——分布尖锐时，多个种子会命中同一条高概率路径。**开采样不等于每次必不同**。实录 `experiments/data/temperature_test.md`。

## 九、`--max-num-tokens` 预留宽度扫描（cpu，2026-07-17 补采）

验证第 6 章的因果：预留宽度决定每步 decode 的 KV 访存宽度，进而影响速度（`LiteRT-LM#2568`）。脚本 `experiments/max_tokens_sweep.sh`，原始数据 `experiments/data/max_tokens_sweep.csv`（-d 128，disk 缓存热，各 2 次）。

| max_num_tokens | prompt | decode tok/s | 备注 |
|---:|---:|---:|---|
| 1024 | 100 | 33.9 | 工单 [128] |
| 2048 | 256 | 23.5 / 29.5 | 两次散布大，取区间 |
| 4096 | 256 | 26.4 | 与默认值推导一致（(256+1023)/4096+1）×4096 |
| 8192 | 256 | 21.5 | 同一 prompt，仅放宽预留 |
| 1024 | 256 | **运行失败** | 工单 [1024] 恰好打满预留宽度，prefill 报 `dynamic_update_slice` 维度越界 |

两条结论。其一，方向与量级都符合第 6 章的分析：预留越宽每步越慢，8192 比 1024 慢约 37%；2048 档的散布提醒短扫描同样有抖动。其二，过小预留的后果不是变慢而是失败：工单恰好打满预留宽度时，越界在 prefill 阶段以编译模型错误抛出（`DYNAMIC_UPDATE_SLICE`），不是静默截断——手动调小该参数时需要注意这个边界。

## 十、原始记录存档

以下两份为采集现场的原始记录，未加修饰，备查。

### 模型实剖原始记录（`experiments/data/model_anatomy.md`）

```text
# gemma-4-e4b model.litertlm 实剖（自研 16KiB 对齐扫描 + TFLite flatbuffer 解析）
# 文件 3.66 GB @ litert-community/gemma-4-E4B-it-litert-lm，运行时 v0.13.1
# 方法：扫 16KiB 边界找 TFL3 魔数定段；tflite python 绑定读 SignatureDefs 与张量形状

offset,size_mb,signatures,key_tensor
4734976,170.9,embedder,token_ids[1;1]
175669248,836.8,per_layer_embedder,token_ids[1;1]
1012449280,94.1,serving_default(audio_encoder),mask[1;1;816]
1106509824,15.7,audio_adapter,features[1;204;1536]
1122254848,0.016,eoa,-
1122271232,224.1,vision_70|vision_140|vision_280,images[1;1260;768]
1346420736,7.9,vision_adapter_70|140|280,soft_tokens[1;140;768]
1354317824,0.016,eoi,-
1354334208,2260.1,decode|prefill_1024|prefill_128|verify,embeddings[1;1;2560]
3614392320,45.1,mtp_drafter,activations[1;1;5120]

# 主模型 decode signature 关键事实：
# - KV cache 输入 48 个张量 = 24 层 × (K+V)，dtype 全部 INT8
#   - 20 层形状 K:[1,2,32003,256] / V:[1,2,256,32003]（H_kv=2, D=256）
#   -  4 层形状 K:[1,2,32003,512] / V:[1,2,512,32003]（H_kv=2, D=512）
# - KV 每 token 字节 = 2(KV) × 2(H) × (20×256 + 4×512) × 1B = 28,672 B = 28 KiB/token
# - 4096 上下文 KV 合计 = 112 MiB；静态槽位 32003 全预留 ≈ 875 MiB
# - embeddings 输入 [1,1,2560] → model_dimension=2560
# - per_layer_embeddings [1,1,42,256]（42 层 PLE × 256）
# - logits [1,1,262144] → 词表 262,144（=2^18）
# - param_tensor [1,1,1,7] INT32（单缓冲 KV 路径的位置参数，见 ch06）
# - mask [1,1,1,32003] BOOL
```

### 采样确定性实验原始记录（`experiments/data/temperature_test.md`）

```text
# 温度/种子实验实录（cpu, gemma-4-e4b, 2026-07-05）
# 命令: litert-lm run gemma-4-e4b --backend cpu --prompt "Write one sentence about the ocean." \
#        --temperature T --seed S [--top-k 64 --top-p 0.95] --cache disk
T=0 seed=42 (run1): The vast, mysterious ocean covers over seventy percent of the Earth's surface, teeming with diverse life and holding immense power.
T=0 seed=42 (run2): 与 run1 逐字一致（确定性复现）
T=1.0 默认k/p seed=1: The ocean is a vast, mysterious expanse covering more than seventy percent of the Earth's surface.
T=1.0 默认k/p seed=2: 与 seed=1 逐字一致（分布尖锐，不同种子命中同一高概率序列）
T=1.0 k=64 p=0.95 seed=7: The vast, restless ocean remains the Earth's mysterious cradle of life, shifting between tranquil serenity and powerful turbulence.
# 结论：温度 0 确定可复现；温度 1.0 输出随种子可变，但分布尖锐时多个种子产出相同序列——
# "开采样"不等于"每次必不同"。
```
