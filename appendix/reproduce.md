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
| 1、2、5–8 | HONOR 同机阶段内存、持续吞吐与文本回调 | 已做，计量范围见附录 D 第十四节 | 本附录第四节；`experiments/data/2026-09-05/M3_RUNS.json` |
| 2、7 | 分析 `.litertlm` 分段 | 已做 | `experiments/data/model_anatomy.md` |
| 4 | prefill 长度扫描 | 已做 | `experiments/prefill_sweep.sh`；附录 D“prefill 长度扫描” |
| 4 | 异步开/关 | 未做 | C++ `litert_lm_main --async=true/false`；Python benchmark 无此参数 |
| 5 | 温度与种子 | 已做 | `experiments/data/temperature_test.md` |
| 6 | `max_num_tokens` 扫描 | 已做 | `experiments/max_tokens_sweep.sh`；附录 D“预留宽度扫描” |
| 7 | 编译缓存冷启动 | 部分记录 | `experiments/data/baseline.csv` 记录了 Init API 聚合值的批内变化；缺少 cache 开/关与外部墙钟对照 |
| 7 | 分段并行加载 | 未做 | 开关仅 C API 暴露 |
| 8 | CPU/GPU 与线程扫描 | 已做 | `experiments/android_bench.sh`；`experiments/data/android_threads.csv` |
| 8 | NPU 端到端推理 | 未完成 | 加载与失败阶段见 `experiments/data/npu_enablement.md` |
| 9 | MTP 主基准、Android 开关、自然文本与聚合比例 | 已做；Mac 主基准只有 `false`/`auto`，均为关闭 | `mtp.csv`、`android_mtp.csv`、`mtp_natural_phone.md`、`mtp_acceptance.md` |
| 9 | Mac / 手机自然代码长生成补测 | 单次摘要，缺完整输入与原始计时日志 | `experiments/data/mtp_ceiling.md`；证据范围见附录 D |
| 10 | 约束解码开/关 | 已做，小样本 | `experiments/data/constraint_test.md` |
| 10 | 多模态端到端 | 未做 | 无结果数据 |
| 11 | Python/C++ 一致性 | 未做 | 无结果数据 |

> Android 扩展基准首次采集未保存 build fingerprint、RAM、温度与功耗模式，现有结果只能按附录 D 所列条件解释。已保存的元信息见 `experiments/data/_meta_android.txt`。

## 四、HONOR MEP-AN00 分阶段与持续测量

2026-09-05 的手机案例采用本仓库的 C API 测试客户端。它在设备内记录单调时间、文本回调和进程内存，再由 Mac 收集日志。完整条件、结果与计量范围见附录 D 第十四节；本节命令用于复现同一采集方法。换手机或运行环境后，应建立新的一组结果。

构建依赖 Android SDK、NDK r28c 和 macOS 的命令行开发工具。`build_m3_android.sh` 先检查源码版本和工作区。检查通过后，编译官方 C API 动态库，再编译测试客户端。输出目录包含同批运行库、GPU accelerator 和 sampler。启动器预加载 `libLiteRt.so`，供 sampler 解析全局符号。只复制 GPU sampler 文件而不满足符号依赖时，实际采样可能回退到 CPU，须检查原始日志。

在书稿仓库执行，源码路径和设备序列号替换为本机的值：

```bash
export LITERT_LM_SOURCE=/你的路径/LiteRT-LM-v0.13.1
export ANDROID_HOME="$HOME/Library/Android/sdk"
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/28.2.13676358"
bash experiments/build_m3_android.sh
adb devices -l
adb -s <设备序列号> shell mkdir -p /data/local/tmp/litertlm
adb -s <设备序列号> push ~/.litert-lm/models/gemma-4-e4b/model.litertlm \
  /data/local/tmp/litertlm/model.litertlm
python3 experiments/m3_preflight.py --serial <设备序列号> \
  --source "$LITERT_LM_SOURCE"
```

预检通过只表示设备清单与模型身份已核对。先运行 3 请求试采，确认每轮有完整文本、正常结束回调和有效内存读数；再检查是否实际加载了 OpenCL sampler。固定 prompt 为 `experiments/m3_prompt.txt`，采样配置为 TOP_P、top-k=1、top-p=1、temperature=1、seed=42。上下文上限 4096，输出上限 512，关闭 MTP；不强制生成固定步数，遇到模型停止序列时结束。

```bash
python3 experiments/m3_run.py --serial <设备序列号> --backend gpu \
  --seconds 180 --requests 3 --label pilot
python3 experiments/m3_run.py --serial <设备序列号> --backend gpu \
  --seconds 180 --requests 5 --sample-ms 0 --label control-before
python3 experiments/m3_run.py --serial <设备序列号> --backend gpu \
  --seconds 900 --requests 1000 --sample-ms 500 --label continuous
python3 experiments/m3_run.py --serial <设备序列号> --backend gpu \
  --seconds 180 --requests 5 --sample-ms 0 --label control-after
```

每个命令只运行一组实验，结束后再执行下一组。连续组按引擎加载后的时间计满 900 s，允许当前请求完成；达到严重热状态、推理错误或采集失联时提前结束。进程退出码和停止原因须一起检查，不能只看是否生成了日志文件。测试期间保留真实电池与热状态，不模拟读数；供电、手机壳、散热、性能模式和室温另存为本次环境记录。

运行器打印新建的日期子目录，保存完整事件、资源读数、输入、构建产物哈希和运行日志。汇总器先核对原始文件的哈希，再重算数值；缺失字段保持缺失，不填 0。用实际输出目录替换下面的占位符：

```bash
python3 experiments/m3_summarize.py experiments/data/<日期>/<运行目录>
```

汇总文件给出阶段边界、离散采样最大值及每请求统计。`requests.csv` 用于逐轮比较。500 ms 是目标采样间隔，实际时刻与读取耗时均有记录。关闭内存采集的对照仍保留回调日志和系统热状态查询，所以两组差值不能解释为所有测量工具的总开销。

[^appc-google-model]: Google，[*gemma-4-E4B-it* 模型卡](https://huggingface.co/google/gemma-4-E4B-it)；访问日期：2026-07-18。

[^appc-hf-auth]: Hugging Face，[*Command Line Interface (CLI)*](https://huggingface.co/docs/huggingface_hub/en/guides/cli)，`hf auth login`；访问日期：2026-07-18。

[^appc-build-guide]: google-ai-edge/LiteRT-LM，[*Build and Run LiteRT-LM*](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.13.1/docs/getting-started/build-and-run.md)，v0.13.1；访问日期：2026-07-18。
