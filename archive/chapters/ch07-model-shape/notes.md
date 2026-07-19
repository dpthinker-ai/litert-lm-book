<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->
# 第 7 章 · 模型的形态：量化、.litertlm 格式与 LoRA — notes（策展）

**一句话使命**：理解「模型如何被压小、装箱、变体」的完整链路。

## 素材取用（单一事实源在 `chapters/_shared/`）
- [`module-schema-format.md`](../_shared/module-schema-format.md) — 模型格式与 Schema (Model Format & Schema)
- [`module-components-resources.md`](../_shared/module-components-resources.md) — 模型资源与扩展组件 (Model Resources & Extension Components)

**聚焦**：取 schema-format 的 .litertlm 分段/mmap、components-resources 的 LoRA；量化收益账结合第 1 章带宽墙。

## 本章图表（先规划后动笔，CLAUDE.md 第六节）
- [x] 图 7-1 低比特表示到 kernel 的支持链
- [x] 图 7-2 .litertlm 文件分段结构
- [x] 图 7-3 mmap 与分段并行加载示意
- [x] 图 7-4 weight cache 路径派生与失效边界
- [x] 图 7-5 容器分层校验
- [x] 图 7-6 模型与 cache 分代发布和回滚
- [x] 图 7-7 LoRA ID 生命周期与资源保留
- [x] 图 7-8 分代发布的存储与 Engine 工作集峰值
- [x] 表 7-1 量化精度收益账（体积/带宽/质量三角）
- [x] 表 7-2 量化部署逐层取证
- [x] 表 7-3 cache 失效矩阵
- [x] 表 7-4 LoRA signature 兼容性
- [x] 表 7-5 初始化故障分层诊断
- [x] 表 7-6 Gemma 4 E4B 的 10 个 TFLite 段
- [x] 表 7-7 容器校验层与失败位置
- [x] 表 7-8 cache 路径与文件描述符入口
- [x] 表 7-9 分代发布关口
- [x] 表 7-10 N/P/W/S 冷热启动矩阵
- [x] 表 7-11 rank-32 测试资产的 LoRA buffer payload
- [x] 表 7-12 多适配器累计资源
- [x] 表 7-13 双版本、顺序重启与覆盖发布的资源权衡
- [x] 表 7-14 发布验收证据包
- [x] 表 7-15 发布证据状态语义

## 本章实验（脚本入 `experiments/`）
- [ ] litertlm_print 解剖文件
- [ ] int4 与 int8 对比（如社区有对应产物）
- [ ] parallel_file_section_loading 开/关的冷启动差异
- [x] 随仓 LoRA 测试资产静态解析与容量复算（`experiments/data/ch07_lora_capacity.md`）

## 补读 / 缺口（写作前须清零）
- [x] `litertlm_read.cc` 与 `litert_lm_loader.cc` 的 mmap 细节
- [x] Python builder 的固定前缀、偏移回填与 16 KiB 对齐
- [x] CPU/GPU 编译选项、weight cache 接入与清理条件
- [x] LoRA 文件读取、signature 匹配、资源保留与主文本接入边界
- [x] 主 loader 的前缀、FlatBuffer、section 范围与重复键校验边界
- [x] 路径/scoped cache 入口、分代发布与受控冷热启动矩阵
- [x] 35 层 rank-32 LoRA 测试资产的 280 个基座输入与 220 个适配器 tensor 容量账
- [x] 双版本发布的磁盘与运行内存峰值，以及单 Engine 顺序重启边界
- [x] 制品、容器、cache、性能、内存、LoRA 与回滚记录的归档边界

## 待核实清单 / 随手记
- 量化位宽只确定文件表示；需依次核对图、delegate、kernel 与受控实测。
- `TFLiteWeights` 外挂权重段在当前主 executor 只接受 GPU 后端。
- cache 标识是秒级 mtime + size，并按 path 在进程内保存，不是内容哈希。
- `LoadLoRA` 对 ID 的重复检查只查 `lora_data_`；已物化 ID 不应复用。
- 主文本 executor 会识别 LoRA 输入名，但 v0.13.1 可直接追踪的 Load/Use 调用点在音频编码器。
- 真实 3.66 GB 容器含 10 个 TFLite 段，主文本段约 2.26 GB；整文件体积不是一次 decode 的权重读取量。
- 当前主 loader 未统一检查 section 越界、重叠、零长度、重复 `BufferKey` 与 FlatBuffer verifier；外部模型须在应用侧补完整性校验。
- LoRA 测试基座的 280 个输入 buffer 为 28.4375 MiB 逻辑 payload；稀疏适配器缺项会创建并清零相应 buffer。
- 3.66 GB 模型保留两个不可变版本时，仅模型文件就需要约 7.32 GB；双 Engine 与顺序重启分别换取回滚速度或更低运行内存峰值。
