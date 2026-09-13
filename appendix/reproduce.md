# 附录 C · 环境搭建与实验复现

> 本附录汇总运行模型与复现各章实验所需的命令。书中实测数据均来自这些流程，方法与原始结果见附录 D。复现时须对齐设备、模型、后端与运行参数。

## 一、使用预编译 Python 包

本节运行当前 v0.17.0。第三至第五节的历史实验仍使用各自记录的 v0.13.1 环境，不应以新版重装结果覆盖归档数据。第六节提供新版性能采集入口。用 Python CLI（第 2 章）：

```bash
uv tool install 'litert-lm==0.17.0'
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

`run` 与 `benchmark` 还接受 `--cache`，它决定后端编译产物的缓存方式（`python/litert_lm_cli/common.py:95-116`）：`disk`（默认）把编译产物持久化到模型旁的缓存文件；`memory` 请求内存缓存，实际可用性取决于后端与构建是否启用；`no` 关闭缓存，每次运行重新编译。本书用这两条命令采集的归档运行都使用 `disk`，其中 gpu/256 条件的首次 Init 聚合值高于后续调用（记录见附录 D 第二节）。归档没有独立核对运行前的缓存内容与后续命中情况，不能把时间差全部归因于缓存复用。切换取值会改变初始化路径，比较 Init 时间时须固定该取值。

## 二、从源码编译

需要 Bazel（经 Bazelisk 自动取 7.6.1）。冻结版的官方构建指南列出了对应平台的前置条件与命令。[^appc-build-guide] 编译 CLI 演示程序：

```bash
git clone https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM
git checkout --detach v0.17.0
git lfs pull
bazel build //runtime/engine:litert_lm_main
bazel-bin/runtime/engine/litert_lm_main --backend=cpu --model_path=<你的模型>.litertlm
```

- GPU：加 `--define=litert_runtime_link_mode=dynamic`，并把 `prebuilt/<平台>/` 下的动态库放到二进制同目录。
- Android 扩展基准使用 advanced CLI，以便取得 benchmark、线程数和峰值内存参数：`bazel build --config=android_arm64 --define=litert_runtime_link_mode=dynamic //runtime/engine:litert_lm_advanced_main`。再把 `bazel-bin/runtime/engine/litert_lm_advanced_main`、模型和 `prebuilt/android_arm64/*.so` 推送到 `/data/local/tmp/litertlm/`；其中 constraint provider 是进程启动依赖，GPU 后端还需要 accelerator 与 sampler 动态库。
- 嵌入式/无 Bazel：改用 CMake 超级构建，见 `docs/getting-started/cmake.md`。

## 三、复现书中实验

采集脚本位于本仓库的 `experiments/`。以下脚本用于复现原采集流程。先在独立环境安装 `litert-lm==0.13.1`；手机客户端及其构建脚本也锁定旧版 C ABI。升级后的 CLI 参数、分块与回调行为已有变化，不能直接把这些脚本的输出作为同条件新版对照。主基准矩阵的命令为：

```bash
# 前置：litert-lm 已装、模型已 import（litert-lm list 可见；litert-community 公开模型无需登录）
experiments/bench_baseline.sh          # backend×context 矩阵，每条件 3 次，输出存 experiments/data/
```

Android 数据尚未用 v0.17.0 重跑。下表 Android 项目的“已做”指 v0.13.1 的历史采集；本轮新版性能重测仅覆盖 Mac。

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
| 7 | 同一 checkpoint 的量化质量与性能对照 | 未完成，缺少可比产物 | `experiments/M4_PROTOCOL.md`；附录 D 第十五节 |
| 7 | 分段并行加载 | 未做 | 开关仅 C API 暴露 |
| 8 | CPU/GPU 与线程扫描 | 已做 | `experiments/android_bench.sh`；`experiments/data/android_threads.csv` |
| 8 | NPU 端到端推理 | 未完成 | 加载与失败阶段见 `experiments/data/npu_enablement.md` |
| 9 | MTP 主基准、Android 开关、自然文本与聚合比例 | 已做；Mac 主基准只有 `false`/`auto`，均为关闭 | `mtp.csv`、`android_mtp.csv`、`mtp_natural_phone.md`、`mtp_acceptance.md` |
| 9 | Mac / 手机自然代码长生成补测 | 单次摘要，缺完整输入与原始计时日志 | `experiments/data/mtp_ceiling.md`；证据范围见附录 D |
| 10 | 专家算子数值、导出与 GPU 精度对照 | 已做，限所列小型产物 | 本附录第七至九节；附录 D 第十七至十九节 |
| 10 | 单层规模与固定路由 | 已做，合成层 | 本附录第十节；附录 D 第二十节 |
| 10 | 完整模型生成、上下文、长输出与多轮 | 已做，Artisan／Metal | 本附录第十一至十四节；附录 D 第二十一至二十四节 |
| 10 | 分阶段进程内存 | 已做，未分解 GPU 分配与权重驻留 | 本附录第十五节；附录 D 第二十五节 |
| 11 | 约束解码开/关 | 已做，小样本 | `experiments/data/constraint_test.md` |
| 11 | 图片输入、视觉预算与输入错误 | 已做，单图小样本 | 本附录第五节；`experiments/data/2026-09-05/M4_RUNS.json` |
| 11 | 音频端到端 | 未做 | 无结果数据 |
| 12 | Python/C++ 一致性 | 未做 | 无结果数据 |

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

预检通过只表示设备清单与模型身份已核对。先运行 3 个请求的试采，确认每轮有完整文本、正常结束回调和有效内存读数；再检查是否实际加载了 OpenCL sampler。固定 prompt 为 `experiments/m3_prompt.txt`，采样配置为 TOP_P、top-k=1、top-p=1、temperature=1、seed=42。上下文上限 4096，输出上限 512，关闭 MTP；不强制生成固定步数，遇到模型停止序列时结束。

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

## 五、图片输入、视觉预算与错误处理

2026-09-05 的图片案例仍使用第四节的手机、模型文件与冻结运行库。固定输入为本仓库生成的 640 × 480 RGB PNG，比较 70 和 280 两档视觉预算；损坏图片和零预算用于检查错误处理。实际结果、完整输出及测量边界见附录 D 第十五节。两档使用同一份权重，因此不属于量化对照；同一基础 checkpoint 的可比量化产物尚未备齐。

先按第四节设置源码与 Android 工具路径，传输模型并完成预检。随后构建图片测试客户端与独立 tokenizer 诊断程序。运行器会上传输入、客户端和同批动态库，并核对模型哈希。测试保留 500 ms 内存采样；本轮没有无插桩对照，观察开销也包含在结果中。

```bash
bash experiments/build_m4_android.sh
uv run --with pillow==11.3.0 python experiments/m4_make_fixture.py
python3 experiments/m4_run.py --serial <设备序列号> --backend gpu \
  --vision-backend gpu --pilot --label vision-pilot --timeout 300 \
  --environment-report <本轮现场条件.json>
```

现场条件文件是 JSON，记录是否带壳、主动散热、性能模式、供电方式和室温；不清楚的项目写明未记录，不沿用别次测量的读数。试采成功后检查完整回答、实际后端和正常结束状态。主干与视觉编码器使用 GPU OpenCL，视觉适配器实际使用 CPU/XNNPACK。上下文上限 4096，输出上限 256，MTP 关闭；采样参数与第四节相同。

```bash
python3 experiments/m4_run.py --serial <设备序列号> --backend gpu \
  --vision-backend gpu --label vision-series --timeout 600 \
  --environment-report <本轮现场条件.json>
python3 experiments/m4_summarize.py <本轮结果目录>
python3 experiments/m4_audit_tokens.py <本轮结果目录> --serial <设备序列号>
uv run --with tflite==2.18.0 --with numpy python experiments/m4_inspect_vision.py \
  ~/.litert-lm/models/gemma-4-e4b/model.litertlm <本轮模型结构.json>
```

正式序列依次为 70、280、280、70、70、280、损坏图片、零预算、70 预算恢复请求。所有请求复用同一 Engine，各自新建 Conversation。首次请求含额外视觉初始化，必须单列。两条预期输入错误不计入成功时延统计；恢复请求使用新会话，不能据此判断原会话已经恢复。`--timeout` 限定整轮主机等待时间，包含引擎加载。

tokenizer 诊断沿用指定结果目录的模型与动态库，只渲染和分词，不生成答案。它将文本片段计数与首轮 BOS、图像结束位置相加，得到非视觉位置数，再供报告从实际 prefill 计数中扣除。模型结构检查另存 signature 的输入、输出和 mask 形状。这种有效视觉位置数是间接复算，没有直接读取运行中的 mask。

只复核本书已归档结果时，不需要连接手机或重新运行推理：

```bash
python3 experiments/m4_report.py experiments/data/2026-09-05/M4_RUNS.json
```

汇总器核对原始记录哈希并重算两档预算的结果。索引分别列出试采、正式序列、采用的 tokenizer 诊断，以及因遗漏 BOS 而排除的首次诊断。复现实验产生新目录后，应另建相应索引，不替换本书的原始记录。输入生成规则、预定检查条件与采集细节保存在 `experiments/M4_PROTOCOL.md`。

## 六、v0.17.0 的 Mac 性能重测

新版采集使用独立 Python 环境。下列命令在书稿仓库根目录运行，结果目录必须尚不存在：

```bash
uv venv --python 3.12 tmp/bench-v0.17.0-venv
uv pip install --python tmp/bench-v0.17.0-venv/bin/python 'litert-lm==0.17.0'
mkdir -p tmp/bench-v0.17.0-model
ln ~/.litert-lm/models/gemma-4-e4b/model.litertlm tmp/bench-v0.17.0-model/model.litertlm
tmp/bench-v0.17.0-venv/bin/python experiments/bench_release.py \
  --model tmp/bench-v0.17.0-model/model.litertlm \
  --out experiments/data/<本轮日期>/benchmark-v0.17.0
```

模型硬链接使编译缓存保存在独立目录；跨文件系统时可改用文件副本。脚本核对包版本，记录模型、动态库与采集脚本的 SHA-256。它直接调用与新版 CLI 相同的 Python `Benchmark` 接口，保留未四舍五入的指标和实际 token 计数，不经过模型注册表的默认配置。

主矩阵固定 CPU 8 线程、KV 容量 8192，关闭 MTP。随后请求切换 GPU 环形缓冲、开启 MTP，并扫描 CPU prefill 长度和 KV 容量。每个条件先预热 1 次，再测 3 次；所有运行串行，各自创建新进程和引擎。脚本还检查退出码、原生错误日志、指标有效性和 token 计数。失败条件保留日志，不生成性能中位数。

`manifest.json` 保存条件、每次调用和原始指标，`summary.csv` 保存各条件的中位数及最小、最大值。Init 是 API 聚合值，TTFT 等于首次 prefill 耗时加 decode 平均每 token 耗时。进程墙钟时间另存，但不等于客户端首段文本时延。本轮不采集峰值内存、功耗或输出质量。正式归档位置与结果见附录 D 第十六节。

采集后复算并检查配置是否生效：

```bash
python3 experiments/bench_release_report.py experiments/data/<本轮日期>/benchmark-v0.17.0
```

复算器校验原始日志哈希、token 计数与 CSV 中位数，并列出后端回退及未生效的配置。本次 GPU 日志显示，环形缓冲参数请求被忽略，主干实际使用 WebGPU/Metal。切换该参数的记录因此标为 `unsupported_control`，不能作为环形缓冲开关对照。

[^appc-google-model]: Google，[*gemma-4-E4B-it* 模型卡](https://huggingface.co/google/gemma-4-E4B-it)；访问日期：2026-07-18。

[^appc-hf-auth]: Hugging Face，[*Command Line Interface (CLI)*](https://huggingface.co/docs/huggingface_hub/en/guides/cli)，`hf auth login`；访问日期：2026-07-18。

[^appc-build-guide]: google-ai-edge/LiteRT-LM，[*Build and Run LiteRT-LM*](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.17.0/docs/getting-started/build-and-run.md)，v0.17.0；访问日期：2026-09-13。

## 七、CPU MoE 单算子验证

第 10 章的最小实验直接构造专家算子，使用公共 C ABI 执行。它绕过完整模型导出，只验证 CPU 小张量的数值与拒绝行为。原始记录和完整边界见附录 D 第十七节。

在独立 Python 3.12 环境安装实验依赖，并使用已取得的 LiteRT-LM 0.17.0 动态库。下面的示例适用于本书的 macOS 环境，`--library` 应指向该版本安装目录里的实际动态库，输出目录必须不存在：

```bash
uv venv tmp/moe-venv --python 3.12
uv pip install --python tmp/moe-venv/bin/python \
  numpy==2.4.3 flatbuffers==25.12.19 tflite==2.18.0
tmp/moe-venv/bin/python experiments/moe_layer_check.py \
  --library /path/to/litert_lm/liblitert-lm.dylib \
  --output tmp/moe-layer-recheck
```

脚本为每个案例启动新进程，保存序列化模型、输入、期望输出、实际输出和原生日志。预期数值案例必须完成 `LiteRtRunCompiledModel` 并满足容差；预期拒绝案例必须得到可定位的 API 错误，进程崩溃或 harness 失败不能算作通过。`--quick` 只运行第一个案例，不能替代完整 26 例验证。

要检验完整模型导出，应另行固定转换器、原始模型和导出配置，并执行真实产物。将 CPU 单算子的通过结果迁移到 GPU 或完整 `.litertlm` 模型之前，还要核对 10.6 节的布局、量化元数据、激活函数与后端覆盖范围。

## 八、MoE 真实导出与容器核查

本节复现附录 D 第十八节的小型专家模块实验，适用于 macOS arm64。导出器从冻结源码安装，CPU 执行复用第七节的环境和公共 C ABI 脚本。先准备干净的源码目录和独立依赖环境：

```bash
git clone https://github.com/google-ai-edge/litert-torch.git tmp/litert-torch-moe
git -C tmp/litert-torch-moe checkout --detach d592a2f09da4839ea34daaef92e53e638b57090a
uv venv --python 3.12.13 tmp/moe-export-recheck-venv
uv pip install --python tmp/moe-export-recheck-venv/bin/python \
  -r experiments/data/2026-09-13/moe-export/requirements-macos.txt
uv pip install --python tmp/moe-export-recheck-venv/bin/python \
  --no-deps tmp/litert-torch-moe
tmp/moe-export-recheck-venv/bin/python experiments/moe_export_check.py \
  --source tmp/litert-torch-moe --worker-python tmp/moe-venv/bin/python \
  --library /path/to/litert_lm/liblitert-lm.dylib \
  --output tmp/moe-export-recheck
```

输出目录必须不存在。脚本导出 FP32／INT8、1／2 token 的四份原始产物，再分别建立只改变激活属性的诊断副本。所有 CPU 调用在独立进程执行，结果分别与 PyTorch、精确 GELU、tanh-GELU 比较。脚本退出 0 只表示八次 Invoke 均完成；是否数值一致须读取 `report.json` 中各项 `matches`，不能把退出码当作导出兼容性通过。

依赖清单固定了本次安装的 nightly 包；版本号与环境完整保存在报告中。本次 `uv pip check` 对 backports-strenum 的 Python 版本声明报错，具体边界见附录 D 第十八节。若对应发行文件已不可取得，或安装工具拒绝该组合，应记录新的依赖组合并重新核查，不将新的产物重标为本次实验。该脚本不导出完整语言模型，也不运行 GPU 或性能测试。

容器核查复用导出环境中的 litert-lm-builder 0.17.0：

```bash
tmp/moe-export-recheck-venv/bin/python experiments/moe_model_header.py \
  --output tmp/moe-model-header-recheck
```

脚本固定模型仓库提交，只读取两份产物各自的前 32768 字节，并检查 HTTP Range 响应。报告区分本地头部哈希与发布方提供的完整文件哈希。读取元数据不创建推理引擎，也不验证完整权重、输出或运行内存。

## 九、MoE GPU 覆盖与精度对照

本节使用第七、八节准备的两套 Python 环境，复用已归档的四份原始导出产物和一份激活诊断副本。采集器校验来源文件哈希后复制到新目录；每次调用均由第七节的执行环境加载指定的 v0.17.0 动态库。

```bash
tmp/moe-export-recheck-venv/bin/python experiments/moe_gpu_check.py \
  --exports experiments/data/2026-09-13/moe-export \
  --worker-python tmp/moe-venv/bin/python \
  --library /path/to/litert_lm/liblitert-lm.dylib \
  --output tmp/moe-gpu-recheck
```

输出目录必须不存在。13 项包含 CPU 对照、WebGPU 默认与显式精度、自动 GPU、预期不受支持的产物，以及普通 ADD 对照。报告同时保存实际 Invoke、非 CPU 加速覆盖接口和 Metal 设备日志。只完成环境注册或模型解析，不能计为 GPU 执行成功。

`MATCH` 表示按指定容差通过比较；`NUMERICAL_MISMATCH` 表示执行完成但未通过；`API_REJECTED` 表示原生 API 返回非零状态。进程崩溃、超时或缺少调用记录均作为采集失败处理，不混入 API 拒绝。退出 0 仅表示未发现采集失败或未确认的 GPU 覆盖，必须逐项读取结果。实际结果和证明范围见附录 D 第十九节。

## 十、MoE 单层规模与路由对照

沿用第七节的执行环境，准备相同版本的预编译动态库。采集入口按现有工作区约定，从 `tmp/upgrade-v0.17.0-venv/lib/python3.12/site-packages/litert_lm/liblitert-lm.dylib` 加载库；换位置时应先为入口配置正确路径，并记录脚本差异。运行：

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  tmp/moe-venv/bin/python experiments/moe_scale_check.py \
  --output tmp/moe-scale-recheck
```

输出目录必须不存在。脚本固定随机种子，重建 16 组 FP32 合成模型，再以随机排列的顺序运行 CPU、GPU 各 3 个进程。每个进程预热 3 次、计时 12 次，每次回读都检查数值。`--pilot` 只运行两个形状用于核对环境，不可替代完整矩阵。模型文件没有加入版本库，重建后应与归档 `fixture.json` 中的 SHA-256 对照。当前采集入口实时读取硬件；实际原始采集脚本另存为 `collector-at-run.py`，其硬件字段与本次机器实查一致。

报告中的 `valid` 同时要求调用完成、数值通过、无后端验证错误，GPU 还要求 Metal 日志和加速覆盖接口确认。计时范围、三次进程重复的统计及解释边界见附录 D 第二十节。进程 RSS 不是 GPU 内存；唯一选中专家权重字节也不是实测访存量。

## 十一、完整 MoE 的重复生成与流式计数

从 10.6.4 节所引固定模型仓库取得完整 GPU 文件，核对大小 15786524672 字节和下列 SHA-256。在独立 Python 3.12.13 环境安装依赖，使用 `moe_generation_check.py`：

```bash
uv venv --python 3.12.13 tmp/moe-full-recheck-venv
uv pip install --python tmp/moe-full-recheck-venv/bin/python \
  -r experiments/data/2026-09-13/moe-full-model/requirements-macos.txt
```

该脚本显式开启 benchmark，直接记录冻结 C API 的流事件，再读取会话的运行时计数。英文配置为：

```bash
tmp/moe-full-recheck-venv/bin/python experiments/moe_generation_check.py \
  --model /path/to/gemma-4-26B-A4B-it-gpu.litertlm \
  --sha256 94bbde2453dd9b67c61c16017af331e5841cbbd9edf83bd2f84bc73e2a7cbdb1 \
  --output tmp/moe-stream-english-r1 --cache tmp/moe-stream-english-cache-r1 \
  --context 128 --output-tokens 32 --repeats 3 --timeout 180 \
  --prompt 'In one short sentence, explain what RAM stores.'
```

同一命令使用不同输出和缓存目录运行三次，得到三个独立进程、每进程三个新会话。中文组只运行一个进程，把容量改为 256、输出上限改为 96，提示词改为“请用两句话解释 RAM 和 SSD 的区别。”。每次都使用新目录；不并行启动完整模型，以免模型之间争用内存。

保护条件为 critical 内存压力立即停止、进程 RSS 上限 18 GiB、总时限 180 秒。采集器保存初始化前的内存状态，以及生成期间的逐次流事件；事件时间戳位于消费队列之前。只有无错误的 final 事件、非空文本、至少两个 runtime decode 计数、会话及引擎关闭、进程正常退出共同满足，才计为完整多 token 生成。后端验证错误另行归类，不能只看运行 API 返回值。

采集完成后，用只读汇总器从 `report.json`、`worker.json`、流事件和日志复算，不以已有 `summary.json` 为输入。它只依赖 Python 标准库；传入三个单次运行目录即可合并同条件结果：

```bash
python3 experiments/moe_generation_report.py \
  tmp/moe-stream-english-r1 tmp/moe-stream-english-r2 \
  tmp/moe-stream-english-r3 --output tmp/moe-stream-summary.json
```

输出文件必须尚不存在，且位于输入目录之外。复算本书归档时，将上述输入换为下列七个实验目录：

```bash
python3 experiments/moe_generation_report.py \
  experiments/data/2026-09-13/moe-full-model-retry \
  experiments/data/2026-09-13/moe-context \
  experiments/data/2026-09-13/moe-context-2048 \
  experiments/data/2026-09-13/moe-context-4096 \
  experiments/data/2026-09-13/moe-long-output \
  experiments/data/2026-09-13/moe-multiturn \
  experiments/data/2026-09-13/moe-memory \
  --output tmp/moe-archive-recomputed.json
```

汇总按模型与库哈希、设备、运行配置、提示词以及首次／后续请求分组。进程数与请求数分别列出，同一进程的多次请求不充当独立进程重复。内存采集单列为 `memory_groups`，不混入性能统计。吞吐沿用运行时 benchmark 记录，不从文本回调数推算。

归档清单存在时，先核验其中的文件哈希。新采集目录没有清单时标为 `not_present`，仍须通过运行记录中的文件哈希检查。缺失、损坏或未完成的运行列入 `errors` 并排除，命令以状态码 2 退出；核验哈希不证明产物来源或源码与二进制等价。

查看结果时，将每个引擎的首次请求与后两个会话分别统计。`first_text_callback_seconds` 是客户端可见文本到达时间。benchmark 中的 TTFT 与初始化字段有不同定义，详见附录 D 第二十一节。每次回调不保证对应一个 token，也不能从完整模型文件大小或进程 RSS 推算 GPU 独占内存。

## 十二、完整 MoE 的上下文容量与实际输入

沿用第十一节的环境与完整 GPU 文件。先用 `moe_generation_check.py` 保持英文短提示词和输出上限 32，将容量分别设为 512、1024、2048、4096。每项仍运行三个新会话，并分别指定新的输出和缓存目录；这一步只改变容量，实际输入长度须读取运行时 prefill 计数。

材料生成器把识别码放在开头，用带编号的说明逐条填充原始 token 预算，最后添加问题。原文和分词 ID 分别保存为 `prompt.txt`、`prompt-tokenization.json`，构造与分词均不进入生成计时。预算不含对话格式；脚本要求原始预算、输出上限与预留的 32 token 之和不超过容量。这个预留值不是固定格式开销的定义，运行后还要核对实际 prefill 与容量。本次四组原始计数为 384、896、1920、3968，运行时分别为 397、909、1933、3981。

材料输入的采集命令如下，原始预算 384、输出上限 64：

```bash
tmp/moe-full-recheck-venv/bin/python experiments/moe_context_check.py \
  --model /path/to/gemma-4-26B-A4B-it-gpu.litertlm \
  --sha256 94bbde2453dd9b67c61c16017af331e5841cbbd9edf83bd2f84bc73e2a7cbdb1 \
  --output tmp/moe-context-notes-512 --cache tmp/moe-context-notes-cache-512 \
  --context 512 --input-tokens 384 --output-tokens 64 --repeats 3 --timeout 180
```

其余材料测试按 `--context / --input-tokens` 配置为 1024/896、2048/1920、4096/3968。每项均使用新目录，输出上限保持 64。按容量 512、1024、2048、4096 的顺序，每档先测短输入、再测材料，八项串行执行。保护阈值与第十一节相同。

生成完成条件沿用第十一节；识别码检查是独立字段，不决定 `GENERATION_COMPLETED`。查看 `answer_contains_marker` 后，还要核对完整回答是否与材料相符。这个固定材料任务不代替系统性质量测试。逐轮计时、首次与复用请求以及压力边界见附录 D 第二十二节。

## 十三、完整 MoE 的较长输出

复用第十一节的环境、完整 GPU 文件与 `moe_generation_check.py`。固定容量 4096，提示词明确要求较长教程：

第二组只将输出上限改为 256，另用新的输出和缓存目录，两组串行执行。先核对每次实际 decode 计数是否达到目标，再检查完整原文是否自然结束。本次均达到上限并截在句中；无错误的 final 不能作为文章完整性的判据。首次请求和后两个新会话分别统计，完整条件与结果见附录 D 第二十三节。

```bash
tmp/moe-full-recheck-venv/bin/python experiments/moe_generation_check.py \
  --model /path/to/gemma-4-26B-A4B-it-gpu.litertlm \
  --sha256 94bbde2453dd9b67c61c16017af331e5841cbbd9edf83bd2f84bc73e2a7cbdb1 \
  --output tmp/moe-long-128 --cache tmp/moe-long-cache-128 \
  --context 4096 --output-tokens 128 --repeats 3 --timeout 180 \
  --prompt 'Write a technical tutorial of at least 600 words explaining how RAM and SSD differ. Discuss volatility, capacity, access latency, bandwidth, virtual memory, and why both are needed. Use complete paragraphs and concrete examples.'
```

## 十四、完整 MoE 的多轮状态更新

沿用相同环境与模型，使用 `moe_multiturn_check.py`。脚本固定三个独立会话、每个会话六轮，同一会话的六轮之间保留会话对象。

每轮发送前检查当前会话计数、该轮原始 token 数、输出预算及 32-token 格式预留之和。运行后核对前后计数与增量指标，不能把预留值当作实际格式开销。各轮的状态字段要求见附录 D 第二十四节；`state_matches` 是完整 JSON 字典比较，与生成及清理完成状态分开记录。

保护条件沿用第十一节，输出和缓存目录必须为新目录。逐轮首文本时间读取 `first_text_callback_seconds`；运行时 TTFT 字段对应会话首轮，后续轮次不能直接使用它。核对下一轮是否接续上一轮的会话计数，并在三个会话之间确认计数重新从 0 开始。

采集命令如下：

```bash
tmp/moe-full-recheck-venv/bin/python experiments/moe_multiturn_check.py \
  --model /path/to/gemma-4-26B-A4B-it-gpu.litertlm \
  --sha256 94bbde2453dd9b67c61c16017af331e5841cbbd9edf83bd2f84bc73e2a7cbdb1 \
  --output tmp/moe-multiturn --cache tmp/moe-multiturn-cache \
  --context 4096 --output-tokens 64 --repeats 3 --timeout 180
```

## 十五、完整 MoE 的分阶段进程内存

沿用第十一节的 Python 环境、完整 GPU 文件和保护条件，使用 macOS 自带的 libproc 与 vmmap。脚本固定容量 4096、材料原始预算 3968、输出上限 64，以及两个新会话：先短输入，再材料输入。阶段表通过 `proc_pid_rusage` 读取字节数，结构定义对应本机 macOS SDK 的 `sys/resource.h` 中 `rusage_info_v0`，调用声明在 `libproc.h`。

```bash
tmp/moe-full-recheck-venv/bin/python experiments/moe_memory_check.py \
  --model /path/to/gemma-4-26B-A4B-it-gpu.litertlm \
  --sha256 94bbde2453dd9b67c61c16017af331e5841cbbd9edf83bd2f84bc73e2a7cbdb1 \
  --output tmp/moe-memory-r1 --cache tmp/moe-memory-cache-r1
```

使用新目录串行执行三次，容量与输入不变。核对 `memory_stages` 的九个阶段、实际计数、完整响应和清理状态；四份 vmmap 记录各须返回 0。采集错误单独归类。

RSS 与 footprint 分别取三个进程同阶段的中位数。libproc 先于 vmmap，二者异时且口径重叠。采集会扰动执行，生成阶段包含 prefill 与 decode，时延不纳入性能对照。结果及版本见附录 D 第二十五节。

三次新采集结果也可交给第十一节的汇总器：

```bash
python3 experiments/moe_generation_report.py \
  tmp/moe-memory-r1 tmp/moe-memory-r2 tmp/moe-memory-r3 \
  --output tmp/moe-memory-recomputed.json
```
