# 模块素材：构建系统、文档与示例  `build-docs`

> 来源：litert-lm-guide/data.json（多智能体源码分析）。**这是素材，不是正文**。
> 引用进正文前，须按 CLAUDE.md 第二节逐条 Read 源码核验、补 `@ v0.13.1`。

**一句话**：把源码变成可运行产物的所有“脚手架”：主用 Bazel(7.6.1) 构建 C++ 运行时，另有一套 CMake 超级构建(super-build) 供嵌入式/自定义编译；通过 rust_cxx_bridge 接入 Rust 依赖；prebuilt/ 放各平台 GPU 运行所需的预编译动态库。docs/ 提供上手与各语言 API 指南，samples/ 与 agents/skills/ 给出端到端示例。

**在架构中的位置**：它不参与运行时数据流，但决定“怎么编出来、怎么跑起来、怎么学”。Bazel/CMake 把 runtime/ + schema/ + 第三方依赖编成 litert_lm_main(CLI) 与各语言库；prebuilt/ 提供 GPU 后端 .so/.dylib/.dll；docs/ 与 samples/ 是新人的入口。

## 关键文件
- `WORKSPACE` — Bazel 工作区：声明全部外部依赖——litert、org_tensorflow、sentencepiece、tokenizers_cpp、minja(jinja模板)、miniaudio、stb(图像)、rules_rust+crate_index、rules_kotlin、apple/swift rules 等。
- `.bazelrc / .bazelversion / .bazeliskrc` — Bazel 配置与版本锁定(7.6.1)；定义 --config=android_arm64 / windows 等平台 config 与编译开关。
- `runtime/engine/BUILD` — 定义关键构建目标 //runtime/engine:litert_lm_main（CLI 可执行）。
- `CMakeLists.txt / CMakePresets.json` — CMake 超级构建入口：先用 prebuild 阶段构建 protoc/flatc 等 host 工具，再驱动子构建；presets 提供 make / android-arm64。
- `cmake/ (modules/packages/toolchains/patches/scripts)` — CMake 的依赖查找、工具链(交叉编译)、第三方补丁与脚本，是 cmake.md 指南的实现。
- `rust_cxx_bridge.bzl / Cargo.toml / cxxbridge_cmd/` — 用 cxx bridge 把 Rust crate（如 llguidance 约束解码）暴露给 C++ 调用。
- `BUILD.* / PATCH.*` — 第三方依赖的自定义 BUILD 与补丁：sentencepiece/skia/llguidance/tensorflow/tokenizers_cpp/minja/miniaudio/stb/antlr4/minizip 等。
- `prebuilt/<platform>/` — 各平台(android_arm64/ios/linux/macos/windows...)预编译动态库，GPU 后端运行时需与 litert_lm_main 放同目录。
- `docs/getting-started/build-and-run.md` — 从源码构建与运行的权威指南（Linux/macOS/Windows/Android 四平台 + GPU/NPU 注意事项 + CLI flag 表）。
- `docs/api/cpp/*.md, docs/api/kotlin/*.md` — C++/Kotlin API 指南：conversation、constrained-decoding、tool-use(含 ANTLR)、getting_started。
- `samples/ios_and_mac/, agents/skills/create-litert-lm-android-demo-app` — 示例：iOS/macOS SwiftUI ContentView；以及用 Agent skill 一键生成带后端选择+多模态的 Android demo app。

## 核心抽象
- **//runtime/engine:litert_lm_main** (build target) 〔`runtime/engine/BUILD`〕：最重要的构建目标：CLI 可执行文件，演示加载 .litertlm、选后端、prefill/decode、benchmark。新人“跑起来”的第一站。
- **Bazel + Bazelisk(7.6.1)** (toolchain) 〔`.bazelversion`〕：主构建系统。Bazelisk 读 .bazelversion 自动拉取正确 Bazel；--config 切平台。一条 bazel build //runtime/engine:litert_lm_main 即可出 CLI。
- **CMake super-build** (toolchain) 〔`CMakeLists.txt`〕：面向嵌入式/无 Bazel 环境的备选构建。LANGUAGES NONE + ExternalProject 风格：先 prebuild 出 protoc/flatc，再分阶段构建。
- **rust_cxx_bridge** (build rule) 〔`rust_cxx_bridge.bzl`〕：把 Rust crate 经 cxx 生成 C++ 绑定接入 Bazel，典型用于 llguidance（约束解码语法引擎）。
- **prebuilt 动态库** (artifact) 〔`prebuilt/`〕：GPU 后端依赖的预编译 .so/.dylib/.dll。运行 GPU 时需 cp 到 CLI 同目录并设 LD_LIBRARY_PATH。

## 数据流
1. 装环境：Git LFS（拉 prebuilt 二进制）+ Bazelisk（自动用 7.6.1）。需要时装 Android NDK r28b+ 并设 ANDROID_NDK_HOME。
2. 下载模型：从 HuggingFace/官方下载 .litertlm，export MODEL_PATH 指向它。
3. 构建：bazel build //runtime/engine:litert_lm_main（Android 加 --config=android_arm64；Windows 加 --config=windows；GPU 加 --define=litert_link_capi_so=true 等）。
4. 运行：bazel-bin/runtime/engine/litert_lm_main --backend=cpu --model_path=$MODEL_PATH；GPU 需把 prebuilt/<os>/*.so 放同目录。
5. 上设备：adb push 二进制+模型(+GPU 的 .so) 到 /data/local/tmp 后 adb shell 运行。
6. 学习/集成：读 docs/ 指南；跑 samples/；或用 Python CLI(uv tool install litert-lm) 与各语言 SDK 直接集成，无需自行编译。

## 概念
- **Bazel vs CMake 两条路**：Bazel 是主路径（贡献者/默认）；CMake super-build 是给嵌入式或不便用 Bazel 的场景的备选。两者都产出同一套运行时。
- **Bazelisk 版本锁定**：.bazelversion 写死 7.6.1，Bazelisk 自动下载对应版本，避免“在我机器上能编”的版本漂移。
- **prebuilt 动态库与 GPU**：GPU 后端的算子实现放在预编译 .so 里（体积大、平台相关），运行时动态加载——所以 GPU 跑前要把对应平台库放到二进制旁。
- **.litertlm 模型分发**：运行只需一个 .litertlm 文件（权重+tokenizer+元数据，见 schema 模块）；App 开发者用预编译 SDK，无需从源码构建。
- **跨平台 config**：用 bazel --config=android_arm64/windows 等切换工具链与编译选项，一套源码多平台产出。
- **Agent skill 生成 Demo**：agents/skills 里有一个技能，可据提示一键生成带后端选择与多模态的 Android demo app——展示如何把运行时落到真实 App。

## 优化
- **预编译二进制(prebuilt + Git LFS)**：GPU 重型库预编译分发，省去用户本地编译 GPU 栈的时间与复杂度。
- **Host 工具预构建(CMake prebuild)**：先把 protoc/flatc 等代码生成工具编好再进主构建，隔离交叉编译复杂度。
- **App 开发者免编译**：Kotlin/Swift/Python 提供预built SDK，绝大多数用户根本不需要碰 Bazel/CMake。

## 关键代码片段（待核验 @ v0.13.1）
**最小构建 + 运行（Linux/macOS）** — 待核验：`docs/getting-started/build-and-run.md`
```bash
export MODEL_PATH=<path to your .litertlm>
bazel build //runtime/engine:litert_lm_main
bazel-bin/runtime/engine/litert_lm_main \
    --backend=cpu \
    --model_path=$MODEL_PATH
```
**免编译快速体验（Python CLI）** — 待核验：`README.md`
```bash
uv tool install litert-lm
litert-lm run \
  --from-huggingface-repo=google/gemma-3n-E2B-it-litert-lm \
  gemma-3n-E2B-it-int4 \
  --prompt="What is the capital of France?"
```
**Android 构建 + 推到设备(GPU)** — 待核验：`docs/getting-started/build-and-run.md:316`
```bash
bazel build --config=android_arm64 //runtime/engine:litert_lm_main
adb push prebuilt/android_arm64/*.so $DEVICE_FOLDER
adb push bazel-bin/runtime/engine/litert_lm_main $DEVICE_FOLDER
adb shell LD_LIBRARY_PATH=$DEVICE_FOLDER \
  $DEVICE_FOLDER/litert_lm_main --backend=gpu \
  --model_path=$DEVICE_FOLDER/model.litertlm
```
**性能 benchmark** — 待核验：`docs/getting-started/build-and-run.md:359`
```bash
litert_lm_main --backend=cpu --model_path=$MODEL_PATH \
  --benchmark --benchmark_prefill_tokens=1024 \
  --benchmark_decode_tokens=256 --async=false
```

## 入手顺序
- 先读 docs/getting-started/build-and-run.md —— 它是“跑起来”的权威步骤，按你的 OS 看对应折叠块。
- 装 Bazelisk + Git LFS，下载一个 .litertlm 模型，跑 bazel build //runtime/engine:litert_lm_main 再 --backend=cpu 运行。
- 想免编译快速体验：uv tool install litert-lm，直接 litert-lm run（见 README）。
- 要交叉编译/嵌入式：看 docs/getting-started/cmake.md 与 cmake/ 目录。
- 要看真实集成：samples/ios_and_mac 与 agents/skills/create-litert-lm-android-demo-app。
