# 附录 F · 外部来源与延伸阅读

> 全书的 LiteRT-LM 代码引用统一锚定 `v0.13.1`（体例见前言），正文只写 `file:line`。本附录收录代码之外的来源：官方文档、对照项目、学术文献、硬件资料。正文以短括注引用（如"官方博客"），完整链接与访问日期在此。

## 官方文档与博客

- LiteRT-LM 投产于 Chrome / Chromebook Plus / Pixel Watch：Google 开发者博客 *On-device GenAI in Chrome, Chromebook Plus and Pixel Watch*，https://developers.googleblog.com/on-device-genai-in-chrome-chromebook-plus-and-pixel-watch-with-litert-lm/ （经 LiteRT-LM 仓库 README 索引，访问 2026-07-05）。
- Gemma 开放模型家族的使命与目标：DeepMind 官方页面 *Gemma*（"Our most advanced open models help developers create AI applications that run wherever users need them — from cloud servers to laptops and even phones"），https://deepmind.google/models/gemma/ （访问 2026-07-18）。E4B 模型卡（"ready for deployment on Android, iOS, Desktop, IoT and Web"）见 huggingface.co/litert-community/gemma-4-E4B-it-litert-lm。
- 更完整的投产口径：LiteRT-LM 仓库 README（main 分支）——"LiteRT-LM powers on-device GenAI experiences in Chrome, Chromebook Plus, Pixel Watch, **and more**"，并附官方示例应用 Google AI Edge Gallery（Android/iOS）的安装指引，https://github.com/google-ai-edge/LiteRT-LM （访问 2026-07-18）。
- Android 侧的互补路径：Android 开发者 ML Kit GenAI 材料——ML Kit GenAI 经 AI Core 提供开箱即用的 Gemini Nano，LiteRT-LM 承载自定义模型部署，developers.google.com/ml-kit 与 Android Developers 官方频道（访问 2026-07-18）。
- 推测解码「约 3 倍」（第 9 章）：Google 官方博客 *Accelerating Gemma 4: faster inference with multi-token prediction drafters*，https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/ （访问 2026-07-05）。本书基准实测未复现，见附录 D。

## 对照项目（第 3–10 章「对照视野」侧栏）

- llama.cpp：github.com/ggml-org/llama.cpp。侧栏中的行号锚定其提交 `b9873`。
- MLC-LLM：github.com/mlc-ai/mlc-llm（文档 mlc.ai）；第 8 章提到其经 TVM 提前编译的做法为文档级引用。
- ExecuTorch：github.com/pytorch/executorch。

以上访问于 2026-07-05。

## 学术文献

- Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via Speculative Decoding*. ICML 2023.
- C. Chen, S. Borgeaud, G. Irving, 等. *Accelerating Large Language Model Decoding with Speculative Sampling*. 2023.

第 9 章用这两篇对照经典推测采样的「概率接受、保持分布无损」与 LiteRT-LM 的 MTP 所走的「贪心接受」路径的差别。

## 硬件资料

- 移动内存带宽 = 数据率 × 总线宽度（体系结构常识）。Dimensity 9400 支持 LPDDR5X-10667：联发科官方博客 *Top 11 Features of the Dimensity 9400*（mediatek.com，访问 2026-07）。Snapdragon 8 Elite 的 LPDDR5X 档位：高通产品简介（qualcomm.com Product Brief）。各 SoC 峰值带宽按数据率 × 64 bit 推导得出，非厂商实测值；NPU TOPS 因口径不一未收录。
- DRAM 每字节访存能耗约为片上乘加的一到两个数量级（计算机体系结构公认量级）。第 1 章的功耗账只用其数量级方向，不引具体工艺数字。

## 上游 issue

书中作缺陷或现象案例引用的 GitHub issue（`LiteRT-LM#2568`、`#2281`、`#2227`、`#2589`、`#2613`、`#2418`、`#2505`）见项目 issue 跟踪器 github.com/google-ai-edge/LiteRT-LM/issues，正文在讨论处标注编号。issue 反映的是写作时的上游状态，可能已被后续版本修复。
