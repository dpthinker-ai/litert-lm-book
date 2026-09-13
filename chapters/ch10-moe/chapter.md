# 第 10 章　端侧 MoE：稀疏激活、专家执行与内存管理

一个模型有 30B 总参数，每个 token 激活 2B 参数，并不意味着设备只需容纳 2B 权重。未参与本次计算的参数仍属于模型；它们是否驻留内存、下一步会不会被访问，取决于路由结果与执行方式。

前几章分别计算了 KV cache、权重和异构执行的成本。第 9 章讨论一次验证多个 token，以分摊权重读取开销。本章转向另一种结构：每个 token 只使用部分专家。两种方法都会改变权重访问，但改变的是不同的量；把它们组合起来时，需要重新计算专家工作集。

华为的 Mate XT 2 官方资料已将 30B MoE 列为端侧模型配置。[^moe-huawei] 这说明稀疏模型已进入具体产品，但该资料不足以确定激活参数量、量化位宽或专家驻留方式。本章的 30B／2B 组合仅用于演算，实际运行案例另用可取得的 Gemma 4 产物。

## 10.1　总参数与激活参数

混合专家模型让不同的输入使用不同的参数子集。MoE 中的专家（expert）通常是一组前馈网络参数；对某个 token，可以只执行选中的几个专家。这里的“专家”是网络结构中的参数分组，不保证对应人能命名的某个知识领域。

本章讨论用稀疏专家前馈层替换部分稠密前馈层的 Transformer。注意力层及其他共享计算仍然存在。设模型共有 \\(L_m\\) 个这样的层，每层有 \\(E\\) 个等大的路由专家，每个专家包含 \\(P_e\\) 个参数，每个 token 选择 \\(K\\) 个不同专家。将始终参与前向的参数合计为 \\(P_s\\)，其中包含路由器和可能存在的共享专家参数，则简化模型中的参数量为：

$$
P_{\mathrm{total}}=P_s+L_m E P_e,\qquad
P_{\mathrm{active}}=P_s+L_m K P_e.
$$

这里把各层专家数和大小设为相同，只为便于验算。实际模型可以逐层求和；embedding 的统计口径也要单独核对。总参数量（total parameters）描述模型包含多少参数；激活参数量（active parameters）描述一次前向所选择的参数范围。两个数字都没有包含权重表示、访问次数和缓存状态。

以 30B／2B 为假设算例，B 按 \\(10^9\\) 个参数计。若所有参数都用理想 INT4 保存，每参数占 0.5 字节，则：

$$
\begin{aligned}
W_{\mathrm{total}}&=30\times10^9\times0.5/2^{30}
 \approx13.97\ \mathrm{GiB},\\
W_{\mathrm{active}}&=2\times10^9\times0.5/2^{30}
 \approx0.93\ \mathrm{GiB}.
\end{aligned}
$$

13.97 GiB 是全部权重的理想表示大小。0.93 GiB 是所选参数按同一位宽换算出的字节数，既不是进程内存，也不是每步实测读取量。这两个数是教学演算，不代表某款手机或模型的实际部署配置；量化 scale、对齐填充和高精度保留层尚未计入。

若后端准备了全部专家的设备缓冲，即使每个 token 只用其中一小部分，缓冲也可能覆盖全量权重。反过来，mmap 可以建立全量地址映射而暂不让每个页面驻留，后续触页仍会发生读入。第 7 章区分的文件大小、映射范围和物理驻留量，在 MoE 中仍须分别记录。

因此，30／2 的比值只给出两个参数统计值之比。它不能直接转化为 15 倍加速：共享计算不会随路由专家数同比缩小，路由和分派还会增加操作。低比特权重也可能先展开再计算。完整部署还要为第 6 章的 KV cache、激活、工作区和后端重排预留内存。

## 10.2　一次 MoE 前向计算

给定一个 token 的隐藏向量，模型先判断使用哪些专家，再合并这些专家的输出。路由器（router）产生每个专家的分数；Top-k 路由（top-k routing）从中选出 \\(K\\) 个专家及对应系数。这里的 Top-k 选择对象是专家，和第 5 章从词表中采样 token 的 top-k 不同。

设输入行向量为 \\(x_t\\)，长度为模型维度 \\(D\\)。路由器可通过一个线性投影得到 \\(E\\) 个分数，再得到所选专家集合 \\(S_t\\) 和系数 \\(\alpha_{t,e}\\)。是否先做 softmax、是否对选中系数重新归一化，属于模型定义；不能仅凭“Top-2”确定。Mixtral 的公开结构采用每层 8 个专家、每个 token 选择其中 2 个，其公式给出了具体的路由与加权方式。[^moe-mixtral]

本章用带门控的前馈专家说明计算。每个专家有 gate、up、down 三组矩阵，隐藏宽度为 \\(H\\)。gate 和 up 将 \\(D\\) 维输入映射到 \\(H\\) 维，down 再投影回 \\(D\\) 维。若矩阵分别记作 \\(W_{g,e}\\)、\\(W_{u,e}\\)、\\(W_{d,e}\\)，则：

$$
f_e(x_t)=\left[\phi(x_tW_{g,e}^{\mathsf T})
 \odot(x_tW_{u,e}^{\mathsf T})\right]W_{d,e}^{\mathsf T},
\qquad
y_t=\sum_{e\in S_t}\alpha_{t,e}f_e(x_t).
$$

\\(\phi\\) 是模型规定的激活函数，\\(\odot\\) 表示逐元素乘法。本章冻结的 CPU 专家实现接受 GELU 及其 tanh 近似，具体约束见 10.6 节。路由权重进入输出求和；若实现另外提供每专家缩放系数，也要纳入乘法，不能遗漏。

<figure>
<img src="figs/fig-10-1-routing.svg" alt="图 10-1　路由先选择专家，执行器按专家聚合输入，再恢复 token 顺序并加权合并"/>
<figcaption>图 10-1　路由先选择专家，执行器按专家聚合输入，再恢复 token 顺序并加权合并</figcaption>
</figure>

图 10-1 中只有 3 个专家，每个 token 选择 2 个。token 0 选择专家 0 和 2，token 1 选择专家 2 和 1。按输入顺序逐条调用也能得到结果，但专家 2 会被调用两次。执行器可以把选中同一专家的输入行汇集起来，让专家 2 一次处理两行。

这种 token 分派（token dispatch）改变张量排列，不改变路由决定。分派记录需保留专家输出与原 token、路由系数的对应关系。本章 CPU 实现为此保存 token 位置和所选路由的位置。专家计算完成后，执行器按记录和路由系数，将结果加权累加到对应输出行。若只保存专家编号，两个 token 的输入仍然可以计算，输出却可能被合并到错误的位置。

并非所有专家都必须参与 Top-k 选择。模型也可以设置每次都执行的共享专家；此时输出还包含共享分支，其参数计入 \\(P_s\\)。共享分支的计算成本也须单独统计。共享专家和细粒度路由专家在公开的 DeepSeekMoE 结构中有明确区分。[^moe-deepseek] 本章后续的 \\(E\\)、\\(K\\) 均只统计需要路由的专家。

## 10.3　容量、计算与访问成本

同一层里，专家数量影响三种不同的成本。增加 \\(E\\) 会增加模型保存的参数；增加 \\(K\\) 会增加单个 token 的专家计算。实际访问量还取决于一批 token 触及多少专家，以及这些专家的权重当前在哪里。

设一次调用处理 \\(T\\) 个 token，专家 \\(e\\) 收到 \\(n_e\\) 行输入。每个 token 选择 \\(K\\) 个专家时，分派总数为 \\(\sum_e n_e=TK\\)。单个门控专家有约 \\(3DH\\) 个矩阵参数，忽略偏置。按一次乘加计 2 FLOPs，专家矩阵计算合计为：

$$
F_{\mathrm{experts}}\approx 6TKDH\quad\mathrm{FLOPs}.
$$

这个表达式没有计入路由投影、激活函数、加权合并和注意力。线性路由投影本身约需 \\(2TDE\\) FLOPs，还要读取路由权重。因而固定 \\(K\\) 增加 \\(E\\)，只能说专家矩阵乘的理想计算量不变，不能说整个模型的计算量完全不变。

将本次调用涉及的不同专家数记为 \\(U=|\bigcup_t S_t|\\)。若每个专家权重在本次调用中只读一次，每参数存储字节数为 \\(b_w\\)，专家权重的理想读取量为 \\(3UDHb_w\\) 字节。相应的权重项算术强度为：

$$
I_{\mathrm{weights}}\approx
\frac{6TKDH}{3UDHb_w}=\frac{2TK}{Ub_w}
\quad\mathrm{FLOPs/byte}.
$$

这只是权重项的模型。它假定按专家聚合了输入，并且没有重复解包或重排。真实算术强度的分母还要加上输入输出、分派记录及中间张量的读写。若权重从存储设备换入，存储到主存和主存到计算单元是两段不同的传输，不能混用同一个带宽值。

**表 10-1　参数规模、工作集与运行指标之间不能相互替代**

| 观察量 | 简化模型中的主要决定因素 | 尚不能据此确定的量 |
|---|---|---|
| 全部权重的表示大小 | 共享参数、各层全部专家、位宽及量化元数据 | 进程 RSS、GPU 分配量 |
| 单 token 专家计算 | 每层选中专家数 \\(K\\)、专家矩阵大小 | 单步时延、能耗 |
| 一次调用的专家权重集 | 不同专家数 \\(U\\)、权重布局和表示 | 实际主存读取量、缺页量 |
| 分派与中间激活 | \\(TK\\) 条分派及各专家的 \\(n_e\\) | 后端工作区的具体峰值 |
| KV cache | 注意力结构、上下文长度与预留形状 | 不随 \\(K/E\\) 自动缩小 |
| 持续生成体验 | 计算、传输、同步、路由序列和设备状态 | 不能由激活参数量单独预测 |

用一个假设层比较这些成本：\\(E=8\\)、\\(K=2\\)、\\(D=2048\\)、\\(H=1024\\)，权重为理想 INT4。单专家包含 6,291,456 个参数，大小为 0.00293 GiB；全部专家约 0.02344 GiB。一个 token 选中的两份专家权重约 0.00586 GiB，而专家矩阵计算约为 25.17 MFLOPs。这里的 M 按 \\(10^6\\) 计。

若一次处理 8 个 token，且其选择覆盖全部 8 个专家，专家矩阵计算增至约 201.33 MFLOPs。理想权重读取量只增至约 0.02344 GiB，权重项算术强度从 4 增至 8 FLOPs/byte。在这组条件下，增加 token 数使权重项算术强度提高两倍；不能沿用“同一批权重供 8 个位置共用”的稠密模型估算。

运行内存需要另列预算：源权重的实际驻留、后端权重副本、KV cache、激活和临时工作区各占多少。若其中两项共享同一底层分配，就只计一次。10.6 节会给出一个具体例子：源权重保存低比特表示，当前专家却被展开到 FP32 临时缓冲，压缩权重与展开后的临时权重同时存在。

## 10.4　Prefill 与 decode

prefill 的输入位置可以并行计算，但它们不一定选择相同的专家。稠密层把多行输入交给同一组矩阵，MoE 层则先将这些行分散到不同专家。第 4 章中“扩大输入块可增加权重复用”的判断，在这里需要加上专家分布这一条件。

每个专家最终收到 \\(n_e\\) 行，其矩阵乘形状分别近似为 \\([n_e,D]\times[D,H]\\) 和 \\([n_e,H]\times[H,D]\\)。即使总输入 \\(T\\) 很大，一些专家也可能只收到一两行。GPU 是否能得到足够大的矩阵任务，取决于这些实际形状，而不只取决于 prompt 长度。

<figure>
<img src="figs/fig-10-2-working-set.svg" alt="图 10-2　输入位置增多时，专家并集与每专家行数共同决定权重复用"/>
<figcaption>图 10-2　输入位置增多时，专家并集与每专家行数共同决定权重复用</figcaption>
</figure>

假设每个 token 独立、均匀地从 \\(E\\) 个专家中选择 \\(K\\) 个不同专家。某个专家未被一个 token 选中的概率为 \\(1-K/E\\)，未被全部 \\(T\\) 个 token 选中的概率为 \\((1-K/E)^T\\)。因此，专家并集大小的期望为：

$$
\mathbb E[U]=E\left[1-\left(1-\frac KE\right)^T\right].
$$

对于 \\(E=8,K=2\\)，单 token 的并集就是 2；处理 8 个 token 时，期望约为 7.20。它是多次随机选择的平均值，单次调用仍只会出现整数个专家。真实路由会受内容、层位置和训练影响，均匀独立假设只是一个可计算的参照，不能当作模型的路由统计。

当多行集中到少数专家，权重复用较多，其他专家没有工作。若实现逐专家串行执行，这主要改变各次矩阵乘的大小。若后端并行处理多个专家，负载不均衡（load imbalance）还可能让最忙的专家决定完成时刻。训练中的负载均衡约束不保证某条端侧请求在每一步都均匀分派。

decode 通常一次只加入一个新 token，本层只选择 \\(K\\) 个专家。但相邻 step 可以选择不同集合，持续生成所涉及的专家会累积。一次短输出中没有访问某个专家，不代表长对话也不会访问它。据短窗口推算长期内存或缓存命中率，需要额外的路由记录。

第 9 章的批量验证也会扩大专家并集。候选 token 即使最终被拒绝，也可能已经在验证前向中触发专家计算和权重读取。因此，MoE 与 MTP 的组合收益必须同时记录接受比例、验证调用的专家并集和草拟成本。仅把激活参数量代入稠密模型的验证耗时，无法解释额外的专家访问。

prefill chunk size 也会改变专家访问量和临时内存。块太小，每个专家的输入行数少，下一块可能再次读取同一专家。块太大，分派记录和中间张量占用增加，还可能触及更多专家。选择块大小时应同时观察首 token 时延和内存峰值，不能仅按 decode 的单步工作集配置。

## 10.5　专家权重的驻留与搬运

稀疏计算只有避开不需要的数据传输，才可能减轻带宽压力。若每次调用都把全部专家复制到设备，即使 kernel 只执行其中两个，仍需传输全量权重。本节讨论几种可选的存储设计；冻结版 LiteRT 已实现到哪一步，在 10.6 节单独核对。

全量驻留把所有专家保留在可供后端访问的内存中。它需要较大的容量，但不必在路由确定后等待专家从存储设备读入。主存已有权重仍不等于片上缓存已有权重；decode 的主存访问与 10.3 节的专家工作集依然相关。

按需加载把不常用专家留在较慢的存储层，等选择结果确定后才准备它们。若从文件换入，还会涉及文件页面、访问粒度和读入顺序。假设本步需要的专家共 0.00586 GiB，按理想连续读入估算时，先换算成字节，再除以实测存储带宽。小块随机读、缺页处理和后端布局转换都会增加耗时，不能把存储厂商公布的顺序读取带宽直接代入。

保留已经准备过的专家，就形成专家缓存（expert cache）。它与第 7 章的编译后 weight cache 不是同一概念：这里按运行时专家访问管理驻留集合。容量必须同时说明保存多少专家、使用哪种表示，以及共享层和临时区占用了多少内存。只报告“能缓存 4 个专家”，不足以复算字节预算。

**表 10-2　专家驻留方案改变的是容量需求和等待位置，收益均依赖访问模式**

| 方案 | 可以减少的成本 | 额外条件与代价 |
|---|---|---|
| 全量驻留 | 路由后的存储换入等待 | 容量覆盖全量表示及后端副本 |
| 按需加载 | 尚未访问专家的驻留量 | 未命中时发生读入、布局转换及同步 |
| 有限专家缓存 | 重复访问专家的换入 | 需要命中统计、淘汰策略和明确容量上限 |
| 预取 | 预测正确且能重叠时的等待 | 预测错误浪费带宽和容量，还可能挤出即将需要的专家 |

预取（prefetch）必须在专家真正使用之前发起。本层的路由依赖本层输入，下一层的输入又依赖当前层输出。提前加载可以使用预测或已经确定的部分信息，但预测依据、提前量和失败成本都需要验证。没有足够可重叠的计算，异步接口本身不会消除等待。

<figure>
<img src="figs/fig-10-3-residency.svg" alt="图 10-3　缓存未命中增加等待，预取只有提前完成且预测正确时才能覆盖换入成本"/>
<figcaption>图 10-3　缓存未命中增加等待，预取只有提前完成且预测正确时才能覆盖换入成本</figcaption>
</figure>

可以先建立一个近似时延模型。设计算与固定分派耗时为 \\(t_c\\)，本步仍需换入的字节数为 \\(B_{\mathrm{miss}}\\)，有效传输带宽为 \\(\beta\\) 字节每秒。若最多可覆盖 \\(t_o\\) 秒传输时间，则额外等待可近似为：

$$
t_{\mathrm{wait}}\approx
\max\left(0,\frac{B_{\mathrm{miss}}}{\beta}-t_o\right),
\qquad t_{\mathrm{step}}\approx t_c+t_{\mathrm{wait}}.
$$

这不是完整的硬件模型。计算和传输可能竞争同一内存带宽，多个专家也可能分批到达；此时不能独立相加。它的用途是明确需要测量什么：本步缺多少字节、读入用了多久、究竟有多少传输发生在可重叠区间。

平均命中率相同的两条请求，停顿分布也可能不同。未命中若分散在多个 step，额外等待也可能分散出现。若在切换话题时集中未命中，就可能形成一段较长停顿。持续生成应保存逐 step 时延及路由序列，并分别报告中位数、高分位和最大值。平均 tokens/s 会掩盖这些位置上的差异。

## 10.6　LiteRT 的实现

LiteRT-LM 编排会话、输入和生成循环，专家层的具体执行位于它依赖的 LiteRT。本章沿用全书的 LiteRT-LM v0.17.0，并锁定其依赖提交 `9fe5be45564c868408e6514c8aabb83e211a0911`。下述 CPU/GPU 结论来自这个版本的代码；出现执行路径，不表示任意 MoE 模型都已能经 LiteRT-LM 完成端到端生成。[^moe-dependency]

转换器与运行时需要就同一份算子契约达成一致。冻结版运行时识别名为 `moe` 的自定义算子（custom op）。输入包括激活、已经选好的专家索引与路由权重，以及专家矩阵。路由分数计算和 Top-k 选择在该算子之外；不能把它理解成一个包含全部路由过程的独立 MoE 层。[^moe-cpu]

### 10.6.1　算子契约与集成边界

FP32 权重模式有 7 个输入：激活、路由权重、专家索引、gate 权重、up 权重、down 权重、每专家缩放系数。量化模式在三组权重之后分别插入 scale，合计 10 个输入。CPU 要求激活、路由权重和输出为 FP32，索引为 INT32；权重及缩放张量为只读常量。支持的权重模式为 FP32、INT8、INT4，激活函数为 GELU 或 GELU 的 tanh 近似。[^moe-cpu]

输入顺序还不足以描述布局。gate/up 的逻辑权重排列是 \\([H,E,D]\\)，down 是 \\([D,E,H]\\)，专家轴位于输出行与输入维度之间。按专家访问时，需要从这些行中取出对应片段；不能假定每个专家天然占据一个连续的大块。量化分组、scale 排列和 INT4 半字节编码也必须匹配消费方。[^moe-cpu]

自定义算子被解析后，还需要由可执行的 kernel 接管。LiteRT 的 CPU 编译路径注册了 `moe` 占位算子，其 Prepare 返回成功，实际 Invoke 则报错；需要由支持该节点的 delegate 接管执行。XNNPACK 检查节点名和支持条件，将符合条件的 MoE 节点交给专用 kernel。该分派代码没有以专用 MoE 开关为前提。[^moe-registration]

部署验证至少包含三项检查。转换产物须符合输入契约，目标后端须接受该算子，实际输出须与参考结果一致。模型加载成功只能覆盖其中的一部分。10.7.1 节分别验证直接构造的算子与真实导出的小型专家模块；两组实验都没有包含完整语言模型。

另取 litert-torch 提交 `d592a2f09da4839ea34daaef92e53e638b57090a` 核查导出端。该版本封装的 FP32／INT8 输入顺序及权重布局与上述契约对应，但不提供同一路径的 INT4 包装分支。[^moe-export] 本书用该版本导出的小型 INT8 模块保留了独立 scale 输入。其权重张量没有量化 scale 与 zero point 元数据，不满足下述 GPU parser 的仿射量化要求。CPU 执行成功不能证明 GPU 支持。导出环境与产物检查见附录 D 第十八节。

导出配置也可能改变计算方式。该版本的 `split_cache` 配置会将专家实现改为顺序封装，逐个专家做稠密运算后按 mask 累加。因而分析产物时必须确认使用了哪条实现，不能只根据原模型属于 MoE，就认定导出后会跳过未选专家。[^moe-export-config]

### 10.6.2　CPU：按专家分派与临时权重展开

CPU kernel 先按专家编号统计分派数量，再求前缀和，将 token 与 route 位置填进连续记录。相同专家的记录落在同一段中，未收到输入的专家可以跳过。实际执行顺序在这段循环里可直接看到：[^moe-cpu]

```cpp
// LiteRT/tflite/delegates/xnnpack/moe_delegate_kernel.cc:420-434
    for (int expert = 0; expert < attr_.num_experts; ++expert) {
      const int begin = expert_offsets_[expert];
      const int end = expert_offsets_[expert + 1];
      const int routed_tokens = end - begin;
      if (routed_tokens == 0) {
        continue;
      }
      if (!RunExpert(context, expert, assignments_.data() + begin,
                     routed_tokens, src, top_weights, gate_weight, gate_scale,
                     gate_scale_elements, ff1_weight, ff1_scale,
                     ff1_scale_elements, linear_weight, linear_scale,
                     linear_scale_elements, per_expert_scale, output)) {
        return kTfLiteError;
      }
    }
```

循环按专家编号依次调用 `RunExpert`。矩阵乘可使用线程池，但这一循环不是把多个专家同时派给独立线程执行。对每个有输入的专家，kernel 先汇集输入行，执行 gate/up 投影、激活与逐元素乘法，再执行 down 投影。随后按路由权重和每专家 scale，将结果累加回原 token 的输出行。

低比特权重进入矩阵乘之前的准备过程，也会产生时间和内存开销。实现把当前专家的 gate/up 权重复制或反量化到一个 float 缓冲，然后调用动态 FP32 全连接算子：[^moe-cpu]

```cpp
// LiteRT/tflite/delegates/xnnpack/moe_delegate_kernel.cc:816-823
    CopyGateUpExpertWeight(gate_weight, gate_scale, gate_scale_elements,
                           ff1_weight, ff1_scale, ff1_scale_elements, expert);
    if (!RunDynamicFullyConnected(context, gate_up_fc_.get(), routed_tokens,
                                  attr_.model_dim, 2 * attr_.hidden_dim,
                                  routed_src_.data(), kernel_buffer_.data(),
                                  gate_up_.data())) {
      return false;
    }
```

down 投影随后复用这个权重缓冲。三个投影不是直接在源 INT4/INT8 数据上完成低比特矩阵乘；源文件的压缩表示与当前专家的 FP32 表示会同时存在。对于 gate/up，临时权重元素数至少为 \\(2DH\\)，仅该缓冲对应 \\(8DH\\) 字节，XNNPACK 自身的打包和工作区另计。

代入 10.3 节的 \\(D=2048,H=1024\\)，gate/up 的 FP32 临时权重就约为 0.01563 GiB。它大于单专家三组理想 INT4 权重的 0.00293 GiB。两者覆盖的投影范围不同，比较的目的只是说明：即使只处理当前专家，展开表示也能成为内存预算中的独立大项。

输入汇集区、gate/up 输出、down 输出及工作区会按需要扩容并复用。成员 vector 的增长逻辑没有在本次输入减少时缩回去，因此一次较大的 prefill 可能留下后续 decode 不再需要的容量。这些临时缓冲保留此前扩容得到的容量，并未按专家编号缓存已展开权重。[^moe-cpu]

每次处理选中的专家仍会执行复制或反量化。此 kernel 中没有专家权重缓存的命中表、淘汰规则或预测预取，也没有根据路由从磁盘换入专家的调度。源张量仍然包含所有专家；其页面是否驻留，要结合外层加载与操作系统行为判断。不能据“跳过无输入专家”推导出“只加载 K 个专家”。

### 10.6.3　GPU：重排、专家计算与输出合并

GPU 的专家计算图同样接收外部已经选择好的索引和系数。构图逻辑把 token 行与专家选择关联起来，再建立专家全连接、门控激活和输出合并。构图时以分派数是否超过专家数为界：当 \\(TK>E\\) 时使用按专家分组的 remap 路径，否则先汇集输入行，再用专家索引选择权重。这个判断使用张量形状，不区分 prefill/decode 名称，也不比较两条路径的实测速度。[^moe-gpu]

专家计算后的结果可按 \\([1,T,K,D]\\) 理解，路由权重对应 \\([1,T,1,K]\\)。最后一次批量矩阵乘沿 \\(K\\) 这一维完成加权求和，输出恢复为每个 token 一行。这段形状设置说明了专家输出如何与路由系数对应：[^moe-gpu]

```cpp
// LiteRT/ml_drift_delegate/delegate/composite/moe_experts_kernel.cc:336-345
  auto reshaped_top_weights = model_builder->Reshape(
      top_weights, ::ml_drift::BHWC(1, sequence_size, 1, num_active_experts));
  auto reshaped_expert_outputs = model_builder->Reshape(
      expert_outputs,
      ::ml_drift::BHWC(1, sequence_size, num_active_experts, model_dim));
  ABSL_ASSIGN_OR_RETURN(auto combined,
                        model_builder->BatchedMatMul(reshaped_top_weights,
                                                     reshaped_expert_outputs));
  combined = model_builder->Reshape(combined, output_shape);
  return model_builder->UpdateOutputTensor(combined, output_id);
```

分组路径需要映射表、计数、偏移，以及重排后的输入输出等临时张量。其映射表的一个形状为 \\([1,E,T,2]\\)，所以减少未选专家的计算并不意味着所有工作区都只随 \\(K\\) 增长。内层采用全连接还是特定矩阵实现，还受每专家任务规模和设备能力影响。[^moe-remap]

激活函数还存在一个必须单独核对的数值边界。导出参考使用 GELU 的 tanh 近似，却把算子属性写成 `gelu`；GPU 对应实现也固定使用 tanh 近似。CPU 则将 `gelu` 与 `gelu_tanh` 分别处理。[^moe-export] [^moe-cpu] [^moe-gpu] 10.7.1 节的小型导出实验观察到了这种差异。在该实验的诊断副本中，仅将激活属性改为 `gelu_tanh`，CPU 结果便通过了参考比较。但 GPU parser 不接受该属性值，因此这项修改不能通用于两个后端。

GPU parser 对量化权重要求仿射量化元数据及全零 zero point，独立 scale 对应每专家、每输出通道；这与 CPU 可接收的 INT4 分组配置不是同一范围。它接受的激活属性也只有 `gelu`。这些支持检查应当在性能测量之前完成。[^moe-gpu-parser]

GPU 代码为量化专家权重建立描述与转换步骤，不能由此套用 CPU 的“先展开为 FP32”结论。反过来，构图代码也不足以证明某设备最终选用了原生 INT4 指令，或全模型都留在同一 GPU 路径上。需要结合生成的 kernel、实际后端与分配记录验证。[^moe-gpu]

### 10.6.4　实现路径与验证范围

模型所属的架构、容器声明和实际后端须分别核对。LiteRT Community 发布了 Gemma 4 26B-A4B GPU 产物。[^moe-gemma-artifact] 本书读取固定版本的容器头，确认文本模型标为 `tf_lite_artisan_text_decoder`，后端约束为 `gpu_artisan`；核查记录见附录 D 第十八节。 v0.17.0 的 `EngineSettings::CreateDefault` 检测到这种模型时，会把请求的 GPU 改为 `GPU_ARTISAN`。[^moe-artisan-selection] 本书的完整模型实际使用 Artisan／Metal；它与 10.6.3 节分析的自定义专家构图路径分别验证。

**表 10-3　同一设备上的执行证据只适用于对应产物与后端**

| 分析或实验对象 | 已验证范围 | 适用边界 |
|---|---|---|
| CPU 自定义专家算子 | FP32、INT8、INT4 数值案例，XNNPACK 执行 | 小型专家层，未包含完整生成 |
| FP32 自定义专家算子 | WebGPU/Metal 显式 FP32 的输出与参考一致 | 默认精度不能直接沿用相同容差 |
| INT8 自定义专家算子 | 已核对输入契约及 GPU parser 条件 | GPU 执行尚未形成通过验证的配置 |
| 完整 Gemma 4 GPU 产物 | Artisan／Metal 文本生成、上下文与多轮验证 | 预编译包，不是本地源码构建 |
| Web 与手机 NPU | 本章没有完整模型通过验证的数据 | 原生 Metal 结果不能外推 |

前三项的数值与支持条件对照见附录 D 第十七至十九节。完整模型的条件与数据见第二十一至二十五节。浏览器运行时、Android 驱动和 NPU 编译产物都有各自的依赖关系；不能只凭文件名含有 GPU 或设备具备某类加速器，就判定部署路径可用。

## 10.7　如何验证部署收益

部署验证需要同时回答三个问题：计算是否正确、资源成本发生在哪里，以及完整请求能完成什么任务。单算子实验让输入和参考输出可控，完整模型实验覆盖加载、会话和生成。两者的结论范围不同，应保留各自的条件与测量口径。

### 10.7.1　数值与执行精度

最小正确性实验先固定路由，使算子错误与路由选择错误可以分别定位。给定输入、专家索引、路由系数和三组权重，逐 token、逐 route 计算 10.2 节的公式，作为参考结果；再让后端执行同一组数。先验证输出，随后才测量时延和内存。

本书在 Apple M5 Pro、24 GiB 内存、macOS 26.5 上，使用 LiteRT-LM 0.17.0 预编译动态库完成 CPU 单算子验证。固定 \\(E=3,K=2,D=4,H=6\\)，改变 token 数、权重类型和 GELU 形式。23 个数值案例均完成 Invoke，最大绝对误差约为 \\(6.88\times10^{-9}\\)；另有 3 个非法输入案例按预期返回错误。输入、模型与输出见附录 D 第十七节，复现见附录 C 第七节。

真实导出还要与导出前的模块比较。冻结 litert-torch 导出的 FP32／INT8、1／2 token 四份小型产物，在 CPU 上分别呈现约 \\(9.87\times10^{-4}\\)、\\(9.81\times10^{-4}\\) 的最大绝对误差。用序列化权重独立复算，可以确认 CPU 对应精确 GELU，PyTorch 参考对应 tanh 近似。只将激活属性改为 `gelu_tanh` 后，四份副本均通过比较，最大绝对误差降至约 \\(9.54\times10^{-7}\\)。这是激活语义的数值对照；副本不代表导出器的原始输出。产物与容差见附录 D 第十八节。

接口类型和计算精度也要分开记录。上述两份 FP32 原始产物在 WebGPU/Metal 显式 FP32 配置下均通过与 PyTorch 的比较，最大绝对误差不超过 \\(9.54\times10^{-7}\\)。默认配置与显式 FP16 的输出逐元素相同，但未满足同一容差。GPU 实际 Invoke、非 CPU 节点覆盖及普通 ADD 对照见附录 D 第十九节。仅写“FP32 模型”，不足以复现这组差异。

正确性检查还应覆盖 token 顺序改变、同一专家接收多行、未选专家、非单位路由系数及量化边界。量化执行应使用量化后再反量化的权重计算参考值，避免把量化误差与 kernel 错误混在一起。报告绝对和相对容差，并说明零附近的比较方式；后端日志与实际输出须同时核对。

### 10.7.2　专家分布与单层耗时

固定每个 token 的激活专家数，仍可能得到不同的执行时间。投影维度固定时，10.3 节的专家矩阵计算量取决于 \\(TK\\)，权重并集和矩阵形状还取决于路由分布。为验证这一区别，本书在同一 M5 Pro 和 v0.17.0 动态库上构造 FP32 专家层，令 \\(E=8,K=2,T=16,D=512,H=1024\\)。

**表 10-4　相同专家投影计算量下，路由分布改变单层同步耗时**

| 路由配置 | 专家并集 U | 唯一选中权重（GiB） | CPU（ms） | GPU（ms） |
|---|---:|---:|---:|---:|
| 全部 token 选择同两个专家 | 2 | 0.01171875 | 2.846 | 0.899 |
| 循环分派，覆盖八个专家 | 8 | 0.046875 | 7.755 | 1.073 |

GPU 使用 WebGPU/Metal 显式 FP32。两种后端采用相同的 tanh-GELU 数学参考，数字取三个新进程各自中位数的中位数。每进程预热 3 次、测量 12 次；计时包含输出等待与读回，权重字节数是表示大小演算，不是实测内存流量。完整 16 组配置、96 次进程运行及逐次正确性检查见附录 D 第二十节。

这组数据说明，激活参数量相同并不足以预测耗时。它尚未分解权重读取、矩阵形状和调度开销各自的贡献；较小矩阵的 GPU 对照也并非总是集中路由更快。将结论用于真实模型时，还须采集逐层路由和逐 step 的专家并集，不能把人为固定路由当作模型的访问统计。

### 10.7.3　完整模型、启动与会话

完整模型采用公开的 Gemma 4 26B-A4B GPU 量化产物。模型名称中的数字是规模标签，表 10-5 使用 Google 模型卡给出的参数口径。该模型有 30 层，专家层同时包含路由分支和共享分支。[^moe-gemma-card] 这些结构信息说明了 10.1 节公式中共享参数与路由专家的区别，不能直接确定转换后产物的驻留量。

**表 10-5　模型结构、发布产物与本书运行配置采用不同口径**

| 项目 | 本章案例 |
|---|---|
| 模型结构 | 总参数 25.2B，激活 3.8B；每个 MoE 层 128 个路由专家，选 8 个，另有 1 个共享专家 |
| GPU 文件 | 15786524672 字节，约 14.70 GiB；固定版本及完整 SHA-256 见附录 D 第二十一节 |
| 权重表示 | 发布方提供的量化产物；本书未重新量化，未完成逐张量位宽统计 |
| 运行环境 | M5 Pro、24 GiB、macOS 26.5；v0.17.0 预编译包 |
| 实际后端 | Artisan／Metal，计算配置 F16；不等于全部权重均为 FP16 |
| 生成设置 | benchmark 开启，thinking 和投机解码关闭；top-k=1、temperature=0、seed=42 |
| 本书验证范围 | 容量至 4096，实际单轮 prefill 至 3981；输出计数至 256；同会话六轮 |

启动和请求耗时应分开记录。调用生成接口前，引擎可能已经完成数十秒的准备；同一引擎的后续会话还会复用准备状态。以下取容量 4096、输出上限 256 的同一进程，三次新会话均使用 58-token prefill，实际 decode 均为 256。模型、后端和采样条件沿用表 10-5。

引擎创建与首次请求是顺序发生的两个阶段。但不能把它们相加称为已测得的完整启动耗时：会话创建等间隔尚未包含。操作系统文件页缓存未清除，因而新进程也不等于存储冷启动。运行时名为 TTFT 的字段与客户端首文本时间有不同定义，不能互换；完整口径见附录 D 第二十一、二十三节。

**表 10-6　引擎创建、首次请求和后续请求分别计时**

| 阶段 | 主机侧耗时或首文本等待（ms） | 计时范围 |
|---|---:|---|
| 创建引擎 | 24638 | Engine 创建调用；一个进程的一次记录 |
| 引擎首次请求 | 3913.8 | 生成调用至首个非空文本回调，不含引擎创建 |
| 后续两个新会话 | 171.1（165.0–177.3） | 同口径首文本等待，中位数及范围 |

生成范围由实际计数和任务结果共同描述。容量 512、1024、2048、4096 的材料输入，运行时 prefill 分别为 397、909、1933、3981，每项三个新会话均找回材料开头的识别码。短输入对照和首次／复用计时见附录 D 第二十二节。固定识别码与重复材料只验证简单信息提取，不是模型最大上下文或通用检索质量评测。

在容量 4096 下，输出上限 128、256 各执行三个新会话，实际 decode 计数均达到上限。复用引擎后的吞吐中位数分别为 52.25、51.70 tokens/s。六次流正常结束，但正文都截在句中，不能称为教程自然完成。每次请求的原文与时间见附录 D 第二十三节；相邻文本回调的间隔也不能当作逐 token GPU 时延。

持续使用同一会话还要检查历史状态。三个独立会话各连续六轮，在第 3、5 轮分别更新识别码与城市，要求保留其余字段。18 次 JSON 回答均符合预期，最终会话计数为 2509–2516，第 2–6 轮首文本等待为 564–619 ms。会话计数包含输入、格式和生成推进，不能当作单轮输入长度；逐轮字段与计数关系见附录 D 第二十四节。

### 10.7.4　内存证据与尚未覆盖的指标

模型文件大小、进程 RSS 和 GPU 分配量属于不同统计范围。分阶段采样应对齐引擎创建、生成完成和资源释放等事件，并明确采集工具是否会打断执行。本书在表 10-5 的环境下固定容量 4096，对短输入和 3981-token 材料输入补充阶段采样；结果与复现见附录 D 第二十五节、附录 C 第十五节。

三个独立进程均完成两次生成。按相同阶段取中位数，引擎创建返回时 RSS 为 5.012 GiB、physical footprint（macOS 的进程内存记账指标）为 2.156 GiB；材料输入生成结束后，分别为 0.229、3.184 GiB。两个统计量走势不同，不能用 RSS 的下降推导专家权重已释放。vmmap 摘要还列有图形相关映射与换出列，但未将它们对应到具体权重、KV cache 或工作区。这里测到的是带采集干扰的进程状态，不是“完整模型只需 3.184 GiB”的部署结论。

进程统计仍不足以分解源权重驻留、GPU 权重副本、KV cache 与临时工作区。若不同对象共享同一底层分配，需要先识别共享关系，再列内存预算，不能把多个工具的合计值直接相加。当前数据也没有给出真实模型的专家访问序列、实际传输字节数或缓存命中率，因此尚不能将某次停顿归因于专家换入。

短时生成不代表热稳态或长时间稳定性。本章未测功率和每 token 能耗，也未做系统性任务质量评估，也没有同等质量条件下的稠密模型对照。这些指标保持未验证状态。部署时应按应用负载采集，而不能从激活参数量或一次成功运行推导出来。

## 小结

MoE 将总参数量与单 token 的参数选择范围分开，也使资源分析必须跟随实际专家访问。总参数决定权重保存规模，路由决定本次计算范围，后端布局与驻留策略决定数据如何到达计算单元。本章冻结的 CPU 实现提供了具体例子：它跳过未分派专家，却仍需为当前专家准备 FP32 临时权重。因此，30B／2B 这样的参数描述还需结合权重位宽、实际驻留和后端临时缓冲，才能换算为部署预算。

## 练习

1. 假设模型有 16 个 MoE 层，每层 8 个路由专家、每次选择 2 个。每专家 50M 参数，共享参数合计 400M。计算总参数量、激活参数量及理想 INT4 权重大小，并说明这些数为什么不足以确定运行内存。
2. 一层的 4 个 token 分别选择 `{0,1}`、`{1,2}`、`{2,3}`、`{0,3}`。求分派总数、专家并集与各专家收到的行数。若全部改为 `{0,1}`，哪些量改变？
3. 在 10.4 节均匀独立假设下，令 \\(E=16,K=2,T=8\\)，计算专家并集的期望。为什么不能直接用它预测真实模型的专家缓存命中率？
4. 一个实现保存全部 INT4 源权重，每次把当前专家展开为 FP32，执行后复用临时缓冲。它是否实现了“只驻留激活参数”？应采集哪些数据才能回答？
5. 两条请求的平均 decode 吞吐和缓存命中率相同，用户却感到其中一条停顿更多。设计记录方法，区分集中未命中、后端同步与设备降频。

[^moe-mixtral]: Albert Q. Jiang 等，[*Mixtral of Experts*](https://arxiv.org/abs/2401.04088)，2024-01-08，第 2 节；访问日期：2026-09-13。
[^moe-deepseek]: Damai Dai 等，[*DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*](https://arxiv.org/abs/2401.06066)，2024-01-11，第 2 节；访问日期：2026-09-13。
[^moe-dependency]: Google AI Edge，LiteRT-LM，[WORKSPACE 中的 LiteRT 依赖声明](https://github.com/google-ai-edge/LiteRT-LM/blob/e9fd8c53ff968071774206163027dd84bedfe925/WORKSPACE#L6-L8)，v0.17.0；访问日期：2026-09-13。
[^moe-cpu]: Google AI Edge，LiteRT，[CPU MoE 专家 kernel](https://github.com/google-ai-edge/LiteRT/blob/9fe5be45564c868408e6514c8aabb83e211a0911/tflite/delegates/xnnpack/moe_delegate_kernel.cc#L106-L889)，`tflite/delegates/xnnpack/moe_delegate_kernel.cc`，106–245、420–434、495–560、583–858、879–889 行；提交 `9fe5be45564c868408e6514c8aabb83e211a0911`；访问日期：2026-09-13。
[^moe-registration]: Google AI Edge，LiteRT，[CPU custom op 注册](https://github.com/google-ai-edge/LiteRT/blob/9fe5be45564c868408e6514c8aabb83e211a0911/litert/runtime/compiled_model.cc#L162-L172)，`litert/runtime/compiled_model.cc`，162–172、391–397 行；[XNNPACK 专家节点委托](https://github.com/google-ai-edge/LiteRT/blob/9fe5be45564c868408e6514c8aabb83e211a0911/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7289-L7297)，7289–7297、7754–7788 行；提交 `9fe5be45564c868408e6514c8aabb83e211a0911`；访问日期：2026-09-13。
[^moe-gpu]: Google AI Edge，LiteRT，[GPU MoE 专家构图](https://github.com/google-ai-edge/LiteRT/blob/9fe5be45564c868408e6514c8aabb83e211a0911/ml_drift_delegate/delegate/composite/moe_experts_kernel.cc#L111-L348)，`ml_drift_delegate/delegate/composite/moe_experts_kernel.cc`，111–173、244–348 行；提交 `9fe5be45564c868408e6514c8aabb83e211a0911`；访问日期：2026-09-13。

[^moe-export]: Google AI Edge，litert-torch，[MoE 导出封装](https://github.com/google-ai-edge/litert-torch/blob/d592a2f09da4839ea34daaef92e53e638b57090a/litert_torch/generative/layers/moe.py#L125-L185)，`litert_torch/generative/layers/moe.py`，125–185、261–277、368–462 行；提交 `d592a2f09da4839ea34daaef92e53e638b57090a`（2026-09-11）；访问日期：2026-09-13。
[^moe-export-config]: Google AI Edge，litert-torch，[专家导出实现选择](https://github.com/google-ai-edge/litert-torch/blob/d592a2f09da4839ea34daaef92e53e638b57090a/litert_torch/generative/export_hf/core/exportable_module_config.py#L192-L197)，192–197 行；[顺序专家实现](https://github.com/google-ai-edge/litert-torch/blob/d592a2f09da4839ea34daaef92e53e638b57090a/litert_torch/generative/layers/moe.py#L480-L528)，480–528 行；提交 `d592a2f09da4839ea34daaef92e53e638b57090a`；访问日期：2026-09-13。
[^moe-remap]: Google AI Edge，LiteRT，[专家重排缓冲与矩阵实现选择](https://github.com/google-ai-edge/LiteRT/blob/9fe5be45564c868408e6514c8aabb83e211a0911/ml_drift_delegate/delegate/composite/experts_remap_builder.cc#L43-L119)，`ml_drift_delegate/delegate/composite/experts_remap_builder.cc`，43–119、173–221 行；提交 `9fe5be45564c868408e6514c8aabb83e211a0911`；访问日期：2026-09-13。
[^moe-gpu-parser]: Google AI Edge，LiteRT，[GPU 专家算子支持检查](https://github.com/google-ai-edge/LiteRT/blob/9fe5be45564c868408e6514c8aabb83e211a0911/ml_drift_delegate/delegate/composite/moe_experts_parser.cc#L75-L198)，`ml_drift_delegate/delegate/composite/moe_experts_parser.cc`，75–198、345–482 行；提交 `9fe5be45564c868408e6514c8aabb83e211a0911`；访问日期：2026-09-13。

[^moe-gemma-artifact]: LiteRT Community，[*Gemma 4 26B-A4B LiteRT-LM 模型卡*](https://huggingface.co/litert-community/gemma-4-26B-A4B-it-litert-lm/blob/7228819fa9580751b57b41a93ee54d5c08c4e001/README.md)，仓库提交 `7228819fa9580751b57b41a93ee54d5c08c4e001`；访问日期：2026-09-13。
[^moe-artisan-selection]: Google AI Edge，LiteRT-LM，[Artisan 模型的后端选择](https://github.com/google-ai-edge/LiteRT-LM/blob/e9fd8c53ff968071774206163027dd84bedfe925/runtime/engine/engine_settings.cc#L145-L199)，`runtime/engine/engine_settings.cc`，145–199 行，v0.17.0；访问日期：2026-09-13。


[^moe-gemma-card]: Google，[*Gemma 4 model card*](https://ai.google.dev/gemma/docs/core/model_card_4)，26B A4B MoE 结构表，Gemma 4；访问日期：2026-09-13。
[^moe-huawei]: 华为，[*HUAWEI Mate XT 2 | ULTIMATE DESIGN 卖点*](https://consumer.huawei.com/cn/support/content/zh-cn16114946/)，适用版本 HarmonyOS 7.0，“大屏 AI 再进化”；访问日期：2026-09-13。
