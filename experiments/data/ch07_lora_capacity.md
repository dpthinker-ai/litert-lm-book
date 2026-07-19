# 第 7 章 LoRA 测试资产容量分析

分析对象来自 LiteRT-LM v0.13.1：

- 基座：`runtime/testdata/litert_dummy_lora32_f16_model.tflite`
- 适配器：`runtime/testdata/test_lora_rank32_f16_all_ones.tflite`

本记录只分析随仓测试资产。结果不能替代产品模型的逐 tensor 检查，也不能当作后端 RSS 或 GPU 内存实测值。

## 复现方法

使用 Bazel 缓存中的 `flatc` 和 TensorFlow Lite schema，把两个二进制文件转为 JSON。缓存根目录会随机器变化，先定位工具：

```bash
find /private/var/tmp/_bazel_dpthinker -type f -name flatc -perm -111
find /private/var/tmp/_bazel_dpthinker -path \
  '*/external/org_tensorflow/tensorflow/compiler/mlir/lite/schema/schema.fbs'
```

设 `FLATC` 与 `SCHEMA` 分别指向上面找到的文件，在 LiteRT-LM v0.13.1 worktree 中执行：

```bash
OUT=$(mktemp -d /private/tmp/lora-flatbuffer.XXXXXX)
"$FLATC" -t --raw-binary --strict-json -o "$OUT" "$SCHEMA" -- \
  runtime/testdata/litert_dummy_lora32_f16_model.tflite
"$FLATC" -t --raw-binary --strict-json -o "$OUT" "$SCHEMA" -- \
  runtime/testdata/test_lora_rank32_f16_all_ones.tflite
```

基座统计取 `decode` signature 中匹配 `^(query|key|value|post)_w_prime_(left|right)_[0-9]+$` 的输入，并依据 tensor shape 与 FLOAT16 的 2 字节宽度计算逻辑 payload。FlatBuffers 会省略值为 0 的默认字段，因此缺少 `tensor_index` 时按 0 处理。

适配器统计同名 tensor 对应的 `buffers[].size`。这与 `LoraData` 按 buffer offset 和 size 建立数据视图的口径一致（`runtime/util/lora_data.cc:74-82`、`runtime/util/lora_data.cc:125-140`）。

## 基座 signature

| 投影 | side | 数量 | shape | 类型 | 单个字节数 |
|---|---|---:|---|---|---:|
| query | left | 35 | `[32, 2048]` | FLOAT16 | 131,072 |
| query | right | 35 | `[32, 2048]` | FLOAT16 | 131,072 |
| post | left | 35 | `[32, 2048]` | FLOAT16 | 131,072 |
| post | right | 35 | `[32, 2048]` | FLOAT16 | 131,072 |
| key | left | 35 | `[32, 2048]` | FLOAT16 | 131,072 |
| key | right | 35 | `[32, 512]` | FLOAT16 | 32,768 |
| value | left | 35 | `[32, 2048]` | FLOAT16 | 131,072 |
| value | right | 35 | `[32, 512]` | FLOAT16 | 32,768 |

合计 280 个输入、29,818,880 B，即 28.4375 MiB。`runtime/components/lora_test.cc:139-152` 另行断言物化后返回 280 个 buffer；`runtime/util/lora_data_test.cc:94-118` 验证 rank 为 32，并检查一个 `32 × 2048 × 2 = 131072` B 的 query tensor。

## 适配器与缺项

适配器共 220 个匹配 tensor。query 与 post 各有 35 层的左右矩阵；key 与 value 各有 20 层的左右矩阵。其 tensor payload 合计 24,903,680 B，即 23.75 MiB。

基座输入名减去适配器 tensor 名后，缺少层 20—34 的 key left、key right、value left、value right，各 15 个。60 个缺项的逻辑 payload 为：

```text
15 × (128 + 32 + 128 + 32) KiB = 4.6875 MiB
```

因此，`23.75 + 4.6875 = 28.4375 MiB`。`LoRA::Init` 仍为缺项创建输入 buffer 并清零（`runtime/components/lora.cc:70-101`）；`runtime/components/lora_test.cc:111-130` 检查了缺失的 `value_w_prime_left_20`，确认返回内容全为零。

28.4375 MiB 是测试基座按 shape 计算的逻辑 payload。后端实际分配应以 `TensorBuffer::PackedSize()` 为准，还可能包含 allocator 对齐、元数据与临时对象。
