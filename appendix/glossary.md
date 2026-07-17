# 附录 A · 术语表

> 全书术语的统一写法与速查。英文技术术语在正文首次出现时括注中文，此后统一用英文写法（如全书用 prefill，不用"预填充"）。
> "首现章"指该术语被**正式引入**（给定义，而非前瞻性提及）的章，按成书顺序。

| 术语 | 对照 | 首现章 | 一句话 |
|---|---|---|---|
| prefill | 预填充 | 1 | 把整段提示词一次性并行喂进模型、批量填 KV cache 的阶段；算力受限 |
| decode | 解码 | 1 | 逐个 token 自回归生成的阶段，每步读一遍全部权重；带宽受限 |
| token | — | 1 | 模型处理的基本单位；一段文本经 tokenizer 切成一串 token id |
| 三类物理约束 | — | 1 | 端侧 LLM 的三重约束：内存容量、内存带宽、功耗与异构（正文偶以"三堵墙"代称） |
| 内存容量约束 | — | 1 | 权重 + KV cache + 激活 + 系统占用必须放得进物理内存 |
| 内存带宽约束 | — | 1 | decode 每 token 要读取一遍全部权重，吞吐由内存带宽决定 |
| quantization | 量化 | 1 | 把权重降到低比特（如 int4=0.5 字节/参数）以压体积、提带宽利用率 |
| Roofline | 屋顶线 | 2 | 判断一段计算被算力还是带宽顶住的分析框架 |
| TTFT | 首 token 时延 | 2 | time-to-first-token，从发起到第一个字出来的时间 |
| tokens/s | — | 2 | 吞吐单位；prefill 常比 decode 高数倍到一个数量级，两者不可混谈 |
| Engine | — | 3 | 重量级、持有模型权重、可被多会话共享的资源持有者 |
| Session | 会话 | 3 | 轻量、有状态的一次对话，持 KV cache 与采样配置 |
| Conversation | 对话层 | 3 | 面向使用者的多轮对话 API，维护历史、套模板 |
| tokenizer | 分词器 | 3 | 文本↔token id 的双向转换（SentencePiece / HuggingFace 两种） |
| embedding | 嵌入 | 3 | token id 查表得到的高维向量；模型真正计算的对象（图像/音频也各自编码成它） |
| 模板 diff 增量渲染 | — | 3 | 只 prefill 新旧渲染串的差值，多轮对话不重算历史 |
| signature | 签名 | 4 | LiteRT CompiledModel 的具名入口（不同长度的 prefill、decode、verify） |
| 静态/动态形状 | — | 4 | 预编译固定长度入口（静态）vs 序列可变、分块 prefill（动态） |
| logits | — | 5 | 模型每步输出的、词表里每个 token 的分数 |
| sampler | 采样器 | 5 | 从 logits 挑下一个 token 的策略：greedy / temperature / top-k / top-p |
| ShouldStop | — | 5 | 集中判定 decode 何时停（停止 token、超长、取消）的纯函数 |
| KV cache | 键值缓存 | 6 | 缓存历史 token 的注意力 Key/Value，用内存换掉重复计算 |
| GQA | 分组查询注意力 | 6 | grouped-query attention，多个查询头共享少量 KV 头，成倍缩小 KV cache |
| 双缓冲 | — | 6 | 备两套 KV 缓冲、读旧写新再交换指针，避开 GPU 同缓冲读写限制 |
| 状态即对象 | — | 6 | 把 KV cache + step + 配置打包成可搬运对象，支撑克隆/检查点/回退 |
| LlmContext | — | 6 | 承载会话可迁移状态（KV cache、current_step、配置）的容器 |
| .litertlm | — | 7 | 单文件模型容器：FlatBuffer 头 + 分段，打包权重/tokenizer/元数据/能力声明 |
| ActivationDataType | 激活精度 | 7 | 激活值的数据类型（FP32/FP16/INT16/INT8），独立于权重量化 |
| LoRA | — | 7 | 不动基座、挂一小份增量权重实现领域适配，可热加载 |
| mmap | — | 7 | 把模型文件按需分页映射进地址空间，用哪段读哪段，缩短冷启动 |
| Backend | 后端 | 8 | 执行后端：CPU / GPU / NPU（及 ARTISAN 手写算子路径） |
| 片上采样 | — | 8 | 采样直接在 GPU 上做，省掉 decode 每步一次 logits 回传 CPU |
| CPU 亲和性 | — | 8 | 把推理线程绑到性能核，避免被调度到能效核而掉速 |
| 推测解码 | speculative decoding | 9 | 用便宜的 drafter 一次猜多 token、base 模型一次前向验证 |
| MTP | 多 token 预测 | 9 | Multi-Token Prediction，推测解码的一种形态，Gemma 4 所用 |
| drafter / verify | 草稿模型 / 验证 | 9 | drafter 草拟候选 token，base 模型的 verify signature 一次验一串 |
| 接受率 | — | 9 | 草拟 token 被接受的比例，决定推测解码到底快多少 |
| bonus token | — | 9 | 首个不匹配处或全对时 base 给的额外正确 token，保证 token 数不亏 |
| patchify | — | 10 | 把图像切成正方 patch，供视觉执行器编码成 embedding |
| 约束解码 | constrained decoding | 10 | 每步采样前把不合语法的 token 的 logit 设为 -inf，保证输出结构合法 |
| Tool Use | 工具调用 | 10 | 让模型输出结构化函数调用，经 ANTLR 文法解析后执行、回填 |
| Preface | 开场白 | 10 | 对话的初始背景：系统消息 + 可用工具声明，Tool Use 链路的第一步 |
| llguidance | — | 10 | 约束解码的语法引擎（Rust 库，经 C bridge 即 llguidance.h 的纯 C 接口接入），逐步给出合法 token 位图 |
| ANTLR | — | 10 | 文法解析器生成器；tool_use 用它的 .g4 文法把函数调用文本解析回结构 |
| C ABI | — | 11 | 收敛成纯 C 的接口层，用不透明句柄 + C 函数当所有语言绑定的公约数 |
| 不透明句柄 | opaque handle | 11 | 跨语言只传指针、不暴露 C++ 类型；create/delete 成对管理生命周期 |
| FFI | — | 11 | 外部函数接口；各语言接 C ABI 的机制（ctypes / JNI / C 互操作 / WASM） |
| FakeLlmExecutor | — | 11 | 脚本化返回 token 的假执行器，让上层逻辑脱离真模型/硬件单测 |
