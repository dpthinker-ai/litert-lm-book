# M4：质量与视觉案例的采集约定

本轮以冻结的 LiteRT-LM v0.13.1、M3 的同一 E4B 文件和 HONOR MEP-AN00 为基础。E04 与 E05 分开验收；视觉预算对照不构成量化对照。

## E04：可比量化产物检查

2026-09-05，本机模型注册表只有一份 Gemma 4 E4B 产物。其混合存储类型已由 M3 检查确认，不能把文件内不同张量的位宽当作两个可比模型。作者确认没有另备的量化模型或 Linux 转换环境，并要求按实际条件保留缺口。

本轮检查了发布方的 E4B 和 Gemma 3 270M 文件清单、模型说明与 LiteRT Torch 转换要求。检查范围与访问结果保存在 `data/2026-09-05/m4-quantization-availability/inspection.json`，原始响应同目录归档。E4B 的通用、GPU、Web 文件名称与大小差异没有提供本实验需要的两份 checkpoint revision、对应量化配方与校准记录。270M 清单中的 q4/q8 Web 产物不能直接当成冻结 Android 执行路径下的可比产物。LiteRT Torch 官方仓库列出 Linux 环境要求；本机为 macOS，未检测到 Docker 命令。本轮没有建立额外的转换环境。

因此 E04 未完成，也未生成量化质量、速度或内存差异表。后续须提供同一基础 checkpoint 的两档可运行产物，记录 tokenizer、模型分段、量化方法、校准数据或不需校准的理由。先固定带参考答案的任务集与评分规则，再执行全部输入；同时保存文件大小、进程内存、客户端首段文本时延与吞吐。变更后端或其他模型配置时，应另列差异，不能把全部变化归因于位宽。

## E05：输入与运行条件

输入是本仓生成的 640 × 480 RGB PNG，白底，从左到右依次为红色正方形、蓝色圆形、绿色三角形。`fixtures/m4/fixture.json` 记录生成方式、几何内容和文件哈希；SVG 只用于书内展示，推理实际读取 PNG。无照片、个人数据或外部图像授权依赖。`m4_make_fixture.py` 使用 Pillow 11.3.0 重建文件。

固定提示词要求按从左到右的顺序给出三种图形的颜色和名称，用一个短句回答。判读标准在运行前写入 fixture：三个颜色与形状的对应关系、左右顺序。汇总器的短语检查只是对此图的辅助检查，完整文本仍须人工核对；它不是通用视觉质量指标。

手机、系统、模型 SHA-256 与运行库沿用 M3；带壳、无主动散热、未开性能模式、USB 供电和估计室温 22–25 °C 的说明也沿用作者本次现场报告。逐次采集系统与热状态。主干与视觉编码器配置 GPU；视觉适配器实际使用 CPU/XNNPACK，按日志记录，不把整条视觉路径称为 GPU 执行。MTP 关闭，上下文上限 4096 token，每请求输出上限 256 token。采样配置为 TOP_P、top-k=1、top-p=1、temperature=1、seed=42。

每个进程复用 Engine，每请求创建新的 Conversation，先验证单个 70 预算请求，再运行预定序列：70、280、280、70、70、280、损坏图片、零预算、70 恢复请求。两档各有 3 次正式观察，首次请求单列；不清理系统文件缓存，复用 `m4-cache` 目录，不能把加载命名为磁盘冷启动。错误请求之后的正常请求使用新 Conversation，不能据此推断出错的原 Conversation 已恢复。

## 计时与内存

`m4_observe.cc` 在设备上以 CLOCK_MONOTONIC 记录请求开始、消息发送、每个回调入口及完成时刻。C API 的 Conversation 回调包含 JSON；汇总器只取 `content` 中非空文本，不能把角色或工具字段当成首段文本。请求开始在文件读取与 Conversation 创建之前；发送开始在创建完成之后。两种首段文本时延分别报告，引擎创建时间另外列出。

运行时 benchmark 保留自然输入与停止条件，prefill/decode 强制计数设为 0；prefill 会等待完成。C API 的 benchmark TTFT 是聚合估值，本轮不使用它替代回调实测。报告的 prefill 时间由该阶段 token 数及速率复算，只覆盖主干 prefill，不包括此前的图片读取、预处理与视觉编码；不从两个时间差直接推导纯编码器耗时。

进程每 500 ms 读取自身 status，在加载、请求和释放边界读取 smaps_rollup。请求前的边界读取发生在开始事件之前，请求后的边界读取发生在结束事件之后；两者不计入请求时延。周期采集、JSON 序列化与日志仍有开销，本轮没有无插桩对照。阶段 RSS 是离散观测最大值，PSS 是指定时刻快照。完整 GPU 内存不可得，不能把这两类读数当作设备总内存，或相减得到视觉分配量。

每请求等待 180 s，取消等待最多 60 s；主机保存约 5 s 一次的热状态，严重热限制、连接错误或运行超时会停止本轮。输入错误保留原始返回值和日志，然后测试下一条独立会话。正式序列的两种输入错误属于预定案例，不纳入成功时延统计。

## visual token 核验

保存输入尺寸、日志中的缩放尺寸与 patch 数、实际选中的编码器/适配器 signature，以及模型文件内的输入/输出 shape。该模型输出含 mask，运行时按 mask 的真值数量裁剪有效 embedding；本轮不转储 mask，不能把 signature 的 70/280 容量直接称为有效 token 数。

另外执行不生成文本的 tokenizer 核验：用相同模型模板渲染原消息，按 Gemma4DataProcessor 的切分规则逐段 tokenize，并加上首轮 Session 的 BOS 和 ImageEnd 各 1 个位置。组合序列的非视觉位置数从这个诊断复算，再从实际 prefill 计数扣除，得到有效视觉位置数。该值是依赖冻结代码的间接复算，不是直接读取 mask；同时与尺寸、patch 池化和 signature 选择相互核对。首次诊断遗漏了 Session 单独插入的 BOS，保留其记录并排除，修正后另存诊断结果。

## 复现入口

在书稿仓库执行：

```bash
bash experiments/build_m4_android.sh
uv run --with pillow==11.3.0 python experiments/m4_make_fixture.py
python3 experiments/m4_run.py --serial <设备序列号> --backend gpu \
  --vision-backend gpu --label vision-series --timeout 600 \
  --environment-report <本轮现场条件.json>
python3 experiments/m4_summarize.py <本轮结果目录>
python3 experiments/m4_audit_tokens.py <本轮结果目录> --serial <设备序列号>
uv run --with tflite==2.18.0 --with numpy python experiments/m4_inspect_vision.py \
  ~/.litert-lm/models/gemma-4-e4b/model.litertlm <本轮模型结构.json>
```

模型传输、源版本与设备核验见附录 C。所有结果另建目录；不改写 M3 或历史基准。本轮采用与排除的记录见 `data/2026-09-05/M4_RUNS.json`，最终状态以 Roadmap 为准。
