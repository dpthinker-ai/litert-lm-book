# 附录 A · 术语表

> 全书术语的统一写法与速查。英文技术术语在正文首次出现时括注中文，此后统一用英文写法（如全书用 prefill，不用"预填充"）。
> "首现章"指正文按成书顺序首次定义该术语的章节，不含前瞻性提及。

| 术语 | 对照 | 首现章 | 一句话 |
|---|---|---|---|
| prefill | 预填充 | 1 | 并行处理提示词并填充 KV cache 的阶段；其瓶颈取决于 signature、后端和序列长度 |
| decode | 解码 | 1 | 逐 token 自回归生成的阶段；典型 batch=1 大模型常受内存带宽约束 |
| token | — | 1 | 模型处理的基本单位；一段文本经 tokenizer 切成一串 token id |
| batch | 批大小 | 1 | 一次前向计算并行处理的序列数；端侧单用户场景通常为 1，此时每步读入的权重只被一个新 token 使用 |
| 三类物理约束 | — | 1 | 端侧 LLM 的三类约束：内存容量、内存带宽、功耗与异构 |
| 内存容量约束 | — | 1 | 权重、KV cache、激活与系统占用之和不得超过可用物理内存 |
| 内存带宽约束 | — | 1 | batch=1 的稠密模型 decode 往往需要每步读取主干权重；当算力与其他开销不先到顶时，带宽限制吞吐 |
| quantization | 量化 | 1 | 用较低比特数表示权重；存储位宽下降可减少体积与权重读取量，实际收益还取决于 kernel 和硬件支持 |
| scale | 缩放系数 | 1 | 量化整数每差 1 对应的实数步长；反量化公式 x ≈ s × (q − z)，可按整层、通道或分组共享 |
| zero point | 零点 | 1 | 实数 0 对应的量化整数；对称量化中恒为 0 可省略，非对称量化需与 scale 一起保存 |
| 稠密模型 | dense model | 1 | 所有参数每步前向都参与计算的模型；与 MoE 等稀疏激活架构相对 |
| MoE | 混合专家 | 1 | Mixture of Experts，稀疏激活的架构；每个 token 只经过部分专家参数，以较少的有效计算换取更大的总参数量 |
| Roofline | 屋顶线 | 1 | 根据算术强度与硬件上限判断计算或带宽约束的分析框架 |
| 算术强度 | arithmetic intensity | 1 | 每单位数据访问量对应的运算量；Roofline 用它判断工作点靠近算力还是带宽约束 |
| memory-bound / compute-bound | 带宽受限 / 算力受限 | 2 | 工作点的上限由带宽项或计算项决定：带宽项更低称 memory-bound，计算项更低称 compute-bound |
| TTFT | 首 token 时延 | 1 | time-to-first-token，从发起请求到首个输出 token 可用的时间 |
| Engine | — | 3 | 持有模型、tokenizer 与执行器等可共享资源，并创建 Session |
| Session | 会话 | 3 | 保存一次交互的独立状态，包括 KV cache、当前位置与生成配置 |
| Conversation | 对话层 | 3 | 面向使用者的多轮对话 API，维护历史并应用聊天模板 |
| Preface | 前置上下文 | 3 | 对话开始前提供的系统指令、few-shot 示例和工具声明；可在创建对话时预先 prefill |
| tokenizer | 分词器 | 3 | 文本↔token id 的双向转换（SentencePiece / HuggingFace 两种） |
| embedding | 嵌入 | 3 | 主干模型接收的稠密向量表示；token、图像和音频可由不同的查找或编码路径转换成该表示 |
| 模板增量渲染 | — | 3 | 支持单轮渲染时直接生成本轮文本；否则比较新旧完整渲染串，仅在前缀关系成立时提交新增后缀 |
| signature | 签名 | 1 | 模型导出的具名调用入口，输入输出张量形状（含长度）在导出时固定；如 prefill_128、decode、verify |
| 静态/动态形状 | — | 4 | 预编译固定长度入口（静态）vs 序列可变、分块 prefill（动态） |
| logits | — | 5 | 模型每步为词表中各 token 输出的分数 |
| sampler | 采样器 | 5 | 根据 logits 选择下一个 token 的策略：greedy / temperature / top-k / top-p |
| 内部/外部采样 | — | 5 | 执行器内部直接返回 token id，或把 logits 交给上层处理后再采样；前者可接设备侧采样实现，后者便于组合重复惩罚与约束解码 |
| 重复惩罚 | repetition penalty | 5 | 降低近期已出现 token 的 logits 以减少重复输出；该操作需要修改 logits，只能走外部采样路径 |
| KV cache | 键值缓存 | 1 | 缓存历史 token 的注意力 Key/Value，以额外内存避免重复计算；理论大小为 2×L×n_kv×d_head×S×b，实际分配另含预留与多缓冲 |
| 预留宽度 | — | 1 | KV cache 预先分配的 token 槽位数（由 `--max-num-tokens` 等参数决定）；固定形状路径下决定张量静态宽度，与已缓存 token 数 S 是两个量 |
| GQA | 分组查询注意力 | 1 | grouped-query attention，多个查询头共享较少的 KV 头；相对每个查询头各有一组 K/V，可按头数比例减少 KV cache |
| 双缓冲 | — | 1 | 维护两套 KV 缓冲，读旧写新后交换指针，适配不允许就地更新的后端 |
| 写时复制 | copy-on-write，COW | 3 | 多个会话先共享已处理上下文；较短分支需要截断或改写共享历史时，才复制执行器上下文并分离所有权 |
| channel | 通道 | 6 | 将思考等内容与最终回复分开；启用相应配置时，可在下一轮输入前回退并重建需保留的 KV 状态 |
| magic number | 占位维度值 | 6 | LiteRT 模型用大于 10 的素数（如 32003）作为张量维度的占位值；运行时按 `--max-num-tokens` 等设置算出目标值，编译时替换，固定形状路径的预留宽度由此进入张量形状 |
| .litertlm | — | 2 | 单文件模型容器：FlatBuffer 头描述一组具名分段，分段可承载 TFLite 模型、tokenizer、`LlmMetadataProto` 等数据 |
| FlatBuffer | — | 7 | 可直接读取缓冲中类型化字段的二进制序列化格式；`.litertlm` 头与 TFLite 模型都使用它 |
| backend constraint | 后端约束 | 7 | `.litertlm` 模型段声明的允许后端集合；Engine 会在编译 executor 前检查请求后端是否包含在集合中 |
| weight cache | 权重缓存 | 7 | 后端编译阶段使用的派生缓存；当前以模型 mtime 与文件大小参与命名，不等同于模型权重段或内容哈希 |
| LoRA | — | 7 | 在基座模型之外加载增量权重；其管理器按已使用的 adapter id 保留资源，未提供卸载接口 |
| mmap | — | 7 | 把文件区域映射到进程地址空间；页面何时读入由访问模式、操作系统与 `madvise` 等条件决定 |
| 后端 | backend | 1 | 设备内某类处理器加上驱动它的软件实现；同一设备可有 CPU/GPU/NPU 多个后端，`Backend` 枚举（含 ARTISAN 路径）见第 8 章 |
| zero-copy | 零拷贝 | 8 | 生产者与消费者复用同一底层存储；还需满足 buffer 类型、布局与完成事件相容，不能由一次 `Duplicate()` 单独证明 |
| 设备侧采样 | device-side sampling | 5 | 由设备后端消费完整 logits 并选出 token；可避免把整个 logits 张量传回 host，仍会回传少量 token 结果 |
| CPU 亲和性 | — | 8 | 限制线程允许运行的 CPU 集合；LiteRT-LM 的 Pixel 路径使用预置核编号 |
| 投机解码 | speculative decoding | 9 | 由计算成本较低的 drafter 生成多个候选 token，再由 base 模型一次前向验证 |
| MTP | 多 token 预测 | 9 | Multi-Token Prediction，投机解码的一种形态，Gemma 4 所用 |
| drafter / verify | 草稿模型 / 验证 | 9 | drafter 草拟候选 token，base 模型的 verify signature 一次验一串 |
| 聚合接受比例 | aggregate acceptance ratio | 9 | 日志中的 verified/drafted；等于每轮接受前缀长度的期望除以草拟步数 G，不等同于逐位独立命中概率 |
| bonus token | — | 9 | 首个不匹配处或草稿全部匹配时，base 模型给出的一个额外 token；它使每轮至少返回 1 个 token，不保证性能收益 |
| patchify | — | 10 | 把图像划分为正方形 patch，供视觉执行器编码为 embedding |
| 约束解码 | constrained decoding | 10 | 每步采样前把当前语法状态下不合法的 token logit 设为 `-inf`；只保证符合给定语法，不保证参数语义或函数可执行 |
| Tool Use | 工具调用 | 10 | 模型输出结构化函数调用，运行时可解析为调用对象；权限检查、实际执行与结果回填由应用层负责 |
| 信任边界 | trust boundary | 10 | 数据跨越该边界后才获得执行权限；模型文本与 parser 输出仍须由宿主完成 allowlist、schema 和授权检查 |
| llguidance | — | 10 | 约束解码的语法引擎（Rust 库，经 C bridge 即 llguidance.h 的纯 C 接口接入），逐步给出合法 token 位图 |
| ANTLR | — | 10 | 文法解析器生成器；tool_use 用它的 .g4 文法把函数调用文本解析回结构 |
| C ABI | — | 11 | 用不透明句柄与 C 函数提供稳定原生边界；Python 与 Swift 使用该层，Kotlin 和 Web 另有 JNI/Embind 路径 |
| 不透明句柄 | opaque handle | 11 | 跨语言边界传递指针而不暴露 C++ 类型；create/delete 成对管理生命周期 |
| FFI | — | 11 | 外部函数接口；本书涉及 ctypes、Swift C 互操作、JNI 与 Embind 等不同原生边界机制 |
