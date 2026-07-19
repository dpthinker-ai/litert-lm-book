# 附录 C · 环境搭建与实验复现

> 本附录汇总运行模型与复现各章实验所需的命令。书中实测数据均来自这些流程，方法与原始结果见附录 D。复现时须对齐设备、模型、后端与运行参数。

## 一、使用预编译 Python 包

不需要编译源码。用 Python CLI（第 2 章）：

```bash
uv tool install 'litert-lm==0.13.1'
litert-lm import \
  --from-huggingface-repo litert-community/gemma-4-E4B-it-litert-lm \
  gemma-4-E4B-it.litertlm gemma-4-e4b
litert-lm run gemma-4-e4b \
  --prompt="What is the capital of France?"
```

`import` 命令把模型导入本地注册表并命名为 `gemma-4-e4b`，后续脚本和校验路径都使用这个名称。本书基准使用 litert-community 发布的 LiteRT-LM 模型包。对应的 Google Gemma 基础模型页面要求先接受许可；[^appc-google-model] 下载受限仓库前，还要按 Hugging Face CLI 文档运行 `hf auth login`。[^appc-hf-auth] 模型文件为数 GiB，下载前需确认磁盘空间充足。

本书实际使用的 `model.litertlm` SHA-256 为 `0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0`。下载后先校验文件；仓库 revision 未在首次采集时保存，因此以文件哈希作为模型产物的最终标识：

```bash
shasum -a 256 ~/.litert-lm/models/gemma-4-e4b/model.litertlm
```

## 二、从源码编译

需要 Bazel（经 Bazelisk 自动取 7.6.1）。冻结版的官方构建指南列出了对应平台的前置条件与命令。[^appc-build-guide] 编译 CLI 演示程序：

```bash
git clone https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM && git checkout v0.13.1
bazel build //runtime/engine:litert_lm_main
bazel-bin/runtime/engine/litert_lm_main --backend=cpu --model_path=<你的模型>.litertlm
```

- GPU：加 `--define=litert_link_capi_so=true --define=resolve_symbols_in_exec=false`，并把 `prebuilt/<平台>/` 下的动态库放到二进制同目录。
- Android 扩展基准使用 advanced CLI，以便取得 benchmark、线程数和峰值内存参数：`bazel build --config=android_arm64 --define=litert_link_capi_so=true --define=resolve_symbols_in_exec=false //runtime/engine:litert_lm_advanced_main`。再把 `bazel-bin/runtime/engine/litert_lm_advanced_main`、模型和 `prebuilt/android_arm64/*.so` 推送到 `/data/local/tmp/litertlm/`；其中 constraint provider 是进程启动依赖，GPU 后端还需要 accelerator 与 sampler 动态库。
- 嵌入式/无 Bazel：改用 CMake 超级构建，见 `docs/getting-started/cmake.md`。

## 三、复现书中实验

采集脚本位于本仓库的 `experiments/`。基准数据集可用以下命令一次性采集：

```bash
# 前置：litert-lm 已装、模型已 import（litert-lm list 可见；litert-community 公开模型无需登录）
experiments/bench_baseline.sh          # backend×context 矩阵，每条件 3 次，输出存 experiments/data/
```

各章实验、状态与证据如下。“未做”表示当前没有可报告的测量结果：

| 章 | 实验 | 状态 | 命令或证据 |
|---|---|---|---|
| 2 | benchmark 主矩阵 | 已做 | `experiments/bench_baseline.sh`；`experiments/data/baseline.csv` |
| 2、7 | 分析 `.litertlm` 分段 | 已做 | `experiments/data/model_anatomy.md` |
| 4 | prefill 长度扫描 | 已做 | `experiments/prefill_sweep.sh`；附录 D「prefill 长度扫描」 |
| 4 | 异步开/关 | 未做 | C++ `litert_lm_main --async=true/false`；Python benchmark 无此参数 |
| 5 | 温度与种子 | 已做 | `experiments/data/temperature_test.md` |
| 6 | `max_num_tokens` 扫描 | 已做 | `experiments/max_tokens_sweep.sh`；附录 D「预留宽度扫描」 |
| 7 | 编译缓存冷启动 | 部分记录 | `experiments/data/baseline.csv` 记录了 Init API 聚合值的批内变化；缺少 cache 开/关与外部墙钟对照 |
| 7 | 分段并行加载 | 未做 | 开关仅 C API 暴露 |
| 8 | CPU/GPU 与线程扫描 | 已做 | `experiments/android_bench.sh`；`experiments/data/android_threads.csv` |
| 8 | NPU 端到端推理 | 未完成 | 加载与失败阶段见 `experiments/data/npu_enablement.md` |
| 9 | MTP 开关、自然文本与聚合比例 | 已做 | `mtp.csv`、`android_mtp.csv`、`mtp_natural_phone.md`、`mtp_acceptance.md` |
| 10 | 约束解码开/关 | 已做，小样本 | `experiments/data/constraint_test.md` |
| 10 | 多模态端到端 | 未做 | 无结果数据 |
| 11 | Python/C++ 一致性 | 未做 | 无结果数据 |

> Android 扩展基准首次采集未保存 build fingerprint、RAM、温度与功耗模式，现有结果只能按附录 D 所列条件解释。已保存的元信息见 `experiments/data/_meta_android.txt`。

[^appc-google-model]: Google，*gemma-4-E4B-it* 模型卡，<https://huggingface.co/google/gemma-4-E4B-it>（访问 2026-07-18）。

[^appc-hf-auth]: Hugging Face，*Command Line Interface (CLI)*，`hf auth login`，<https://huggingface.co/docs/huggingface_hub/en/guides/cli>（访问 2026-07-18）。

[^appc-build-guide]: google-ai-edge/LiteRT-LM，*Build and Run LiteRT-LM*，v0.13.1，<https://github.com/google-ai-edge/LiteRT-LM/blob/v0.13.1/docs/getting-started/build-and-run.md>（访问 2026-07-18）。
