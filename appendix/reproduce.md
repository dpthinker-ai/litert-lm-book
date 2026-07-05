# 附录 C · 环境搭建与实验复现

> 本附录汇总跑起来、以及复现书中各章实验所需的命令。书中所有实测数字来自这套流程（方法与原始数据见附录 D）。

## 一、最快：免编译体验

不需要编译源码。用 Python CLI（第 2 章）：

```bash
uv tool install litert-lm
litert-lm run \
  --from-huggingface-repo=google/gemma-3n-E2B-it-litert-lm \
  gemma-3n-E2B-it-int4.litertlm \
  --prompt="What is the capital of France?"
```

> Gemma 是受限模型：首次下载前需在 Hugging Face 网页接受许可，并本机登录（`huggingface-cli login` 或 `python3 -m huggingface_hub.commands.huggingface_cli login`，粘贴一个 Read 权限的 token）。模型文件数 GiB，留足磁盘。

CLI 常用子命令：`run`（交互/单条）、`benchmark`（性能）、`import`（下载/导入）、`list`、`serve`（OpenAI 兼容服务）。

## 二、从源码编译（读代码/改代码时）

需要 Bazel（经 Bazelisk 自动取 7.6.1）。编译 CLI 演示程序：

```bash
git clone https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM && git checkout v0.13.1
bazel build //runtime/engine:litert_lm_main
bazel-bin/runtime/engine/litert_lm_main --backend=cpu --model_path=<你的模型>.litertlm
```

- GPU：加 `--define=litert_link_capi_so=true --define=resolve_symbols_in_exec=false`，并把 `prebuilt/<平台>/` 下的动态库放到二进制同目录。
- Android：`bazel build --config=android_arm64 //runtime/engine:litert_lm_main`，再 `adb push` 二进制、模型（GPU 还需 `.so`）到 `/data/local/tmp`。
- 嵌入式/无 Bazel：改用 CMake 超级构建，见 `docs/getting-started/cmake.md`。

## 三、复现书中实验

采集脚本在书稿仓的 `experiments/`。基准数据集一次性采集：

```bash
# 前置：litert-lm 已装、模型已 import（litert-lm list 可见）、已 hf 登录
experiments/bench_baseline.sh          # backend×context 矩阵，每条件 3 次，输出存 experiments/data/
```

各章实验与对应命令（数字回填后见正文与附录 D）：

| 章 | 实验 | 命令要点 |
|---|---|---|
| 2 | 读 benchmark 数字 | `litert-lm benchmark <model> --backend cpu -p 256 -d 128` |
| 2、7 | 解剖 .litertlm 分段 | `litertlm_print`（源码工具） |
| 4 | prefill 耗时随长度 | benchmark 扫 `-p 100…4000`；`--async` 开/关 |
| 5 | 采样对比 | `run` 时改温度 0 vs 1.0 |
| 6 | KV cache 内存/速度 | benchmark 扫 `--max-num-tokens`；`get_token_count` 观察多轮 |
| 7 | 冷启动 | `--cache disk/no` 对比；分段并行加载开/关 |
| 8 | 后端对比 | `--backend cpu` vs `gpu`；扫 CPU 线程数 |
| 9 | 推测解码 | `--enable-speculative-decoding true/false`（需支持 MTP 的模型，如 Gemma 4） |
| 10 | 多模态 / 约束 | 图片输入端到端；约束解码开/关看工具调用成功率 |
| 11 | 多语言一致性 | 同一 prompt 走 Python 与 C++，对比输出 |

## 四、本书基准机器

- 芯片 / 内存：Apple M5 Pro / 24 GiB
- 系统：macOS 26.5
- 主基准模型：Gemma 3n E2B int4（`google/gemma-3n-E2B-it-litert-lm`）

> 换机器、换模型、换后端都会得到不同的数字——这正是第 8 章的教训。复现时请对齐这三项，否则不能与书中数字直接比。
