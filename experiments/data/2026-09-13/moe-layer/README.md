# CPU MoE 单算子正确性验证

2026-09-13，通过 LiteRT-LM 0.17.0 的 PyPI 预编译动态库执行 26 个合成单算子模型。23 个数值用例均实际完成 `LiteRtRunCompiledModel`，与独立 NumPy float64 逐 token、逐 route 公式的比较通过；3 个非法配置均在 Invoke 阶段返回错误。没有未预期失败。

最大绝对误差为 `6.877335978483501e-09`，预设容差为 `atol=2e-6, rtol=2e-5`。这里只验证小张量数值与错误行为，不测语言模型、路由器、自然输入质量、性能、峰值内存、GPU 或 NPU。

| 用例 | 数量 | 结果 |
|---|---:|---|
| FP32 / INT8 / INT4 × token 数 1 / 2 / 8 × GELU / GELU-tanh | 18 | Invoke 成功，数值通过 |
| 重复专家索引、合法负索引、零路由权重、route 顺序交换 | 4 | Invoke 成功，数值通过 |
| INT4 数据使用 INT8 容器保存 packed bytes | 1 | Invoke 成功，数值通过 |
| 专家索引 E、专家索引 -E-1、不支持的 silu 激活 | 3 | Invoke 返回非零状态；原生错误日志保留 |

所有模型使用 `E=3,K=2,D=4,H=6`。图包含单个 custom `moe`；前三个动态图输入为 FP32 `src[T,D]`、FP32 `top_weights[T,K]`、INT32 `top_indices[T,K]`。gate/up 权重布局是 `[H,E,D]`，down 权重布局是 `[D,E,H]`，三组权重和专家 scale 写入模型常量。INT8 使用每行一个 scale，INT4 使用每行两个等长组，覆盖有符号负值；没有验证任意分组方式或奇数维度。

参考公式直接遍历每条 route：`y[t] += route_weight[t,r] * expert_scale[e] * down_e @ (GELU(gate_e @ x[t]) * (up_e @ x[t]))`。参考端使用原始有符号整数与 scale 还原量化权重，不使用原生 kernel 的 packed-row 寻址代码。`fixture-readback.json` 另行核对序列化模型结构、FlexBuffer 属性、输入字节和常量字节，INT4 用 NumPy 位掩码展开后与原始有符号值比较。

执行仅选择 CPU accelerator，使用默认 CPU 编译选项，没有注册替代 MoE kernel，也没有设置专用 MoE 开关。原生日志确认创建 XNNPACK CPU delegate。环境创建时也注册了 GPU accelerator，并报告 NPU 不可用，这些日志不表示本实验运行了 GPU/NPU。不支持 silu 的用例先在委托选择阶段报告错误，随后在 Invoke 落到 stub 并报错；仅创建模型成功不足以证明算子受支持。

`report.json` 包含软件版本、动态库与脚本 SHA-256、逐案例结果和全部案例文件的 SHA-256。`provenance.json` 记录 PyPI 包元数据及 ABI 参考头文件哈希。源码分析提交为 `9fe5be45564c868408e6514c8aabb83e211a0911`；调用预编译包成功不证明该二进制由本地锁定源码构建。`pilot/` 保存首个成功用例的初次执行，不计入 26 例结果。

在装有 `numpy==2.4.3`、`flatbuffers==25.12.19`、`tflite==2.18.0` 的 Python 3.12 环境运行：

```sh
python experiments/moe_layer_check.py \
  --library /path/to/litert_lm/liblitert-lm.dylib \
  --output /path/to/new-result-directory
```

输出目录必须不存在，避免覆盖原始记录。每个用例由新子进程执行；`model.tflite`、`inputs-and-weights.npz`、三个输入二进制、`expected.npy`、`actual.npy`、API 调用状态及 stdout/stderr 均保存在对应案例目录。预期拒绝案例不保存预期数值输出。脚本的 `--quick` 只运行第一个 FP32 用例，不替代完整验证。
