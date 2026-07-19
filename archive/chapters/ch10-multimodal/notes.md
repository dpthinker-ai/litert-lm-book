<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->
# 第 10 章 · 多模态输入、约束解码与工具调用 — notes

**一句话使命**：说明图像与音频如何从消息 API 变为主干 embedding，以及结构约束与宿主执行权限的边界。

## 素材取用（单一事实源在 `chapters/_shared/`）
- [`module-executor-multimodal.md`](../_shared/module-executor-multimodal.md) — 多模态执行器 (Multimodal Executor: Vision/Audio)
- [`module-components-resources.md`](../_shared/module-components-resources.md) — 模型资源与扩展组件 (Model Resources & Extension Components)
- [`module-components-text.md`](../_shared/module-components-text.md) — 文本组件: 分词与采样 (Text Components: Tokenization & Sampling)

**聚焦**：取 executor-multimodal 的 embedding 注入、components 的约束解码(llguidance)与 tool_use；双主题，小节切干净。

## 本章图表
- [x] 图 10-1 图像 embedding 注入序列
- [x] 图 10-2 约束解码逐步屏蔽示意
- [x] 图 10-3 多模态数据边界与诊断检查点
- [x] 图 10-4 工具调用信任边界
- [x] 表 10-1 输入表示与所有权
- [x] 表 10-2 多模态 shape 账本
- [x] 表 10-3 tools-derived 约束边界
- [x] 表 10-4 Tool Use 全链路各环节职责
- [x] 表 10-5 宿主信任检查

## 本章实验（脚本入 `experiments/`）
- [ ] 图片输入端到端 + 数 visual token 验证 patchify 公式
- [x] 开/关约束解码小样本：每种设置 6 次，共 12 次；仅观察结构可解析性

## 已核对源码
- [x] `Message` → `InputData`、path/blob 加载与模态标记配对
- [x] `InputImage` / `InputAudio` 所有权、移动与 `TensorBuffer::Duplicate`
- [x] vision/audio preprocessor、executor、组合与 embedding lookup
- [x] `ConstrainedDecoder` 的 mask、sample、状态推进和终态 reset
- [x] tools-derived FC 文法、通用 FC parser 与 `ToMessageImpl`

## 当前实验缺口
- [ ] 图片输入端到端、visual token 计数与不同 budget 的质量/时延对照
- [ ] 音频端到端与分块时延 profile
- [ ] 工具调用的宿主执行、授权失败、幂等与多轮上限测试
