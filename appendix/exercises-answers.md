# 附录 E · 练习提示与参考答案

> 本附录逐章对应正文末尾的“练习与自查”。复算题给出完整算式，机制题列出要点并指向对应章节，动手题给出预期观察结果。

## 第 1 章

1. INT4 7B 权重 7 × 10⁹ × 0.5 B ≈ 3.26 GiB；8K 上下文 KV（示例参数 128 KiB/token）≈ 1 GiB，两项合计约 4.26 GiB。能否在可用 7 GiB 内运行，还取决于后端工作区、激活峰值和其他进程占用，不能只由这两项保证。32K 上下文 KV 约 4 GiB，仅权重与 KV 已约 7.26 GiB，超过题设可用内存。
2. 带宽 9600 × 10⁶ × 8 B = 76.8 GB/s；INT8 4B 权重每 token 读 4 × 10⁹ B；带宽侧算术上限为 76.8 ÷ 4 ≈ 19 tokens/s。该上限假定接口峰值可持续供给模型，而且每步只有一次权重搬运。实际还受有效带宽、KV 流量、同步、kernel 与温度约束，不能当作持续性能承诺。
3. 若题设明确假定 20 pJ/byte，则每 token 读 1 GB 对应约 0.02 J；15 Wh = 54000 J，理想上限为 2.7 × 10⁶ token。20 pJ/byte 不是跨工艺通用常数；结果只用于展示计算方法，不代表具体设备续航。
4. 在典型 batch=1 稠密模型中，prefill 的多个位置可以复用同一批权重，算术强度通常较高。decode 每步只处理一个新 token，却仍需访问大量权重，算术强度较低。短 prompt、较大 batch、稀疏或 MoE 模型、量化解包与固定调度开销，都可能使这种简化判断失效；是否落入哪一侧仍要结合实测工作点。
5. 若该工作点已确认受带宽约束，算力翻倍不会改变带宽上限；提高带宽或减少每 token 访存量才可能提升上限。若 kernel、量化解包或同步仍受计算与固定开销影响，则算力变化仍可能改变实测吞吐。
6. `Run` 是 LiteRT `CompiledModel` 的执行入口，驱动错误落在平台后端层，不在 LiteRT-LM 的编排逻辑内。先核对 LiteRT 的编译选项、delegate 与 buffer 交接，再逐层向上验证 LiteRT-LM 的输入组装。

## 第 2 章

1. 1024 ÷ 999.1 + 1 ÷ 50.6 ≈ 1.025 + 0.020 = 1.045 s，按表中精度显示为约 1.04 s。它由同一轮的 prefill、decode 指标计算得到，不是另一只计时器记录的独立 TTFT。
2. 在两档 \\(B_{\mathrm{eff}}\\) 相同、全部耗时都可折算为字节、256-token 档上下文成本忽略不计的假设下，\\(d_{\mathrm{eq}} = 2.26\ \text{GB} \div 4096 \times (50.6 \div 45.6 - 1) \approx 59\ \text{KiB/token}\\)。它高于 28 KiB 的 KV cache 理论大小，因为这项等效估算还吸收了注意力计算、缓存行为、带宽利用率变化和其他随上下文增长的成本。它不是 DRAM 流量测量值。
3. 仅凭这四个数字不能判断异常。模型条件至少要补全模型产物与哈希、量化和运行时版本。输入条件要有 prompt token 数、prefill/decode 长度、batch 与 `max_num_tokens`。后端条件要有 CPU/GPU/NPU、delegate、线程数、缓存与 MTP/采样配置。测量条件要说明冷启动或热启动、重复次数与统计量、Init 字段定义、外部墙钟、温度和功耗模式。缺少这些条件时，不应把结果与本书某一档数据直接比较。
4. 否则计时终点落在异步提交返回处，测得的是提交耗时而非硬件完成耗时。实现是 `params.SetWaitForCompletion(wait_for_completion | benchmark_info.has_value())`（`runtime/core/tasks.cc:435`）。
5. `MaskLogits` 在第 4 层（组件层）实现、被第 2 层（编排层）的 `DecodeAndSample` 外部路径调用；它修改的 logits 来自第 3 层执行器的 `DecodeLogits`。

## 第 3 章

1. 可让模板根据 `loop.last` 改写既有消息，例如把最后一条渲染为 `<last>…</last>`、其他条渲染为 `<old>…</old>`。加入新消息后，原来的最后一条会从 `<last>` 变成 `<old>`，所以新串不再保留旧串前缀。`runtime/conversation/conversation.cc:241-245` 的显式前缀检查会返回 `InternalError`，而不是把不正确的差值交给 prefill。
2. 同步 `SessionAdvanced::Clone` 先把克隆任务排到源 Session 的既有任务之后；执行管理器随后调用 `ResourceManager::CloneContextHandler`。这一步让两个 handler 共享同一个 `SharedProcessedContext`，只按值复制 `RuntimeConfig` 与 `RuntimeState`，不复制 KV。后续分支 prefill 需要写时分离时，`SaveProcessedContextAndSeparateLoadedHandler` 才调用 executor 的 `CloneContext()`；compiled executor 的 `CloneKVCacheBuffers()` 或 NPU 对应分支再逐块调用 `CopyTensorBuffer`。因此，深拷贝属于后续写时分离，不是初次 Clone 的固定成本。
3. 单轮路径要求模板声明 `supports_single_turn`，并由模型数据处理器实现 `RenderSingleTurnTemplate`；该调用失败时错误直接返回。模板不支持单轮渲染时，Conversation 才进入全历史回退。处理器须能把历史消息转换为模板输入。模板须能分别渲染旧历史和加入新消息后的完整历史，而且新串必须以旧串开头。转换、渲染或前缀检查任一失败都会返回错误，不会再退到第三条路径。
4. `byte_offset` 为 \\(3 \times 4\ \text{B} \times 2048 = 24576\ \text{B} = 24\ \text{KiB}\\)。它只把本次写指针定位到当前输出张量的第 3 个 embedding 槽位之后。跨轮复用由 Session 上下文、模板后缀提取与 KV 状态承担；`byte_offset` 不保存历史，也不会跨调用自动累积。
5. 例如 `{% if messages[0].content.startswith ("x") %}` 在方法名与左括号之间留了空格。Python 接受这种写法，但当前 RE2 规则只匹配紧邻的 `.startswith(`，所以不会把它改成 MiniJinja 的 `is startingwith` 形式。`PromptTemplate` 构造时会保存 MiniJinja 的模板创建错误；若错误直到求值才出现，也沿同一路径返回。C++ 侧第一次调用 `PromptTemplate::Apply` 时把错误转换为 `InternalError("Failed to apply template: …")`，不是在 `EditTemplateForMinijinja` 中主动拒绝。

## 第 4 章

1. 700 < 1024，铺满循环不执行；收尾阶段从大到小查找入口。由于 128 < 700，下一较小入口无法容纳剩余 token，故选择 1024。分块计划为 [1024]，填充 324，填充率约 32%。若允许组合 6 个 128 入口（768 ≥ 700），填充可降至 68（约 9%）；源码 TODO 记录了这一备选方向。
2. 循环依次取 512、512、512、512 个 token，因此能确认会调用 4 次 `PrefillInternal`。代码没有给出分块前后的峰值激活生命周期、分配器复用、各块执行时间或后端同步成本。激活峰值和 TTFT 的方向与幅度都要在目标后端实测，不能由块数直接推出。
3. `do_prefill_sync_` 为 false 时，最后一组之前的工作组异步提交；最后一组是否异步还取决于 `wait_for_completion`。输入缓冲中只要发现 Metal memory，代码就把 `do_prefill_sync_` 设为 true，所有工作组改为同步执行；源码旁的 TODO 明确把 Metal 异步 prefill 留作后续工作。
4. 基准模型只有 128 与 1024 两档静态 prefill signature。250、500、1000 token 都使用 1024 档，对应墙钟时间约 3.9 s；吞吐上升主要来自同一固定分块中的有效 token 比例提高。仅凭这条曲线不能判断算术强度或计算单元利用率。
5. v0.13.1 不会在正在执行的 prefill 内检查取消标志，动态 chunk 之间也没有该检查。取消只能阻止尚未开始的任务，或在整段 prefill 返回后使结果被丢弃。decode 在每次迭代开始时检查，若当前 step 已开始，需等它返回。

## 第 5 章

1. 262144 × 2 B = 512 KiB；数据量约为 1.86 GiB 的 0.26‰。这个比例不能直接证明回传耗时可忽略，因为设备同步、映射和跨地址域传输与本地权重读取的成本模型不同。是否可忽略需要用分项计时验证。
2. 不停。`ShouldStop` 第一分支要求 `hit_stop_tokens && benchmark_decode_token_count == 0`，benchmark 模式下该条件为假，循环继续执行至 128 步。此时测量语义优先于生成语义。
3. 产出 A：暂存，无输出；产出 B：继续暂存，无输出；产出 C：匹配失败。若 C 也不是任何停止序列的首 token，当前最大部分匹配长度归零，队列释放 A、B，再追加 C，本次回调输出 ABC。若 C 同时开启另一条停止序列，则还要保留与新前缀对应的后缀，不能仅由题干推出 ABC 全部发送。
4. v0.13.1 的 SentencePiece 与 Hugging Face 实现把解码结果以 Unicode 替换字符 U+FFFD 结尾视为序列尚不完整，并返回 `DataLossError`。`DecodeOneStep` 因此保存这批 token id；下一步由 `MergeTokenIds` 把旧 id 放到新 id 前面，再整体重试转换。若只转换最新 id，先前构成该文本片段所需的 token 就会丢失。这里描述的是实现采用的完整性判据，并不假定每个 token 都对应一个完整字符。
5. 种类数应按实际 5 次输出去重，不能预先指定。本书已有记录只覆盖部分 seed：seed 1 与 2 得到同一序列，seed 7 得到另一序列。温度 1.0 使随机采样生效，但高概率 token 仍可能被不同随机序列同时选中；因此“已启用采样”不蕴含“每次输出必不同”。

## 第 6 章

1. 28 KiB × 8192 = 224 MiB。32768 不小于占位值 32003，`GetTargetNumber` 回落到占位值以下最大的 256 的倍数 32000 并打印警告；28672 B × 32000 = 917,504,000 B = 875 MiB。
2. `Session::Clone` 本身不搬运 LLM KV，搬运量为 0。新旧 handler 共享同一个 `SharedProcessedContext`，各自持有按值复制的 `RuntimeConfig` 与 `RuntimeState`；其中 `RuntimeState::rand_gen` 是 `shared_ptr`，复制状态时不会复制底层随机数生成器。较短分支后续需要截断或改写共享历史时才触发写时分离。若此时使用 compiled executor，且上下文宽度为 4096，一组活动 KV 输入缓冲约为 \\(4096 \times 28\ \text{KiB} = 112\ \text{MiB}\\)；`CopyTensorBuffer` 按 `PackedSize()` 复制完整容量，不按有效前缀裁剪。
3. 缓冲内容不复制，只交换输入和输出缓冲指针。读旧写新之后调用 `std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_)`；prefill 路径在 `runtime/executor/llm_litert_compiled_model_executor.cc:738`，decode 路径在 `runtime/executor/llm_litert_compiled_model_executor.cc:947`。
4. 按 28 KiB/token 估算，预留容量从约 112 MiB 增至约 224 MiB。本书同 prompt 实验中，decode 从 26.4 降至 21.5 tokens/s，约下降 19%。实验没有用性能计数器分离注意力、KV 访存和其他执行成本，不能把全部降幅归到单一原因。
5. compiled executor 的 `CloneContext` 只遍历活动的 `input_kv_cache_buffers_`，每块都按 `PackedSize()` 完整复制；非单缓冲恢复路径通过移动保存的 map 接管所有权，不再复制内容。NPU executor 则从 prefill 输入中选择名称以 K、V 或 C cache 前缀开头的缓冲，仍按各自 `PackedSize()` 完整复制；恢复时先核对源、目标大小，再把保存内容 `memcpy` 回固定输入缓冲。两条 executor 路径都不调用 `LitertKVCache::DeepCopy`；后者是 `KVCacheInterface` 的独立实现，会复制 bank 1 及可选的 bank 2。
6. assistant 完成含 channel 字段的消息时只设置待过滤标记，KV 尚未改变。下一条非追加式 user 消息到达后，会话回到上次检查点，按历史重建并 prefill 不含 channel 的内容，保存新检查点，再 prefill 当前 user 消息并进入 decode。回退不会清零旧槽位；refill 会覆盖相应内容。

## 第 7 章

1. 文件大小会受到分组 scale、zero point、对齐填充、文件头、tokenizer 和其他模型段影响。decode 吞吐还取决于后端是否保持低比特表示、是否有对应 kernel，以及运行是否主要受权重带宽限制。任一条件不成立，都不能从位宽比推出 4 倍吞吐。
2. `alignment_gap = 49152 mod 65536 = 49152`，`aligned_begin_offset = 0`。映射从文件头开始，返回给上层的指针在映射基址上前移 49152 字节，指向 section 数据。
3. 两个文件的标识都是相同的 `<mtime_seconds>_<size>`，无法区分内容差异。同一进程首次按路径计算后还会缓存该标识；即使随后替换文件，继续查询同一路径也可能复用旧结果。
4. 两次 `LoadLoRA` 后，ID 0 与 1 都位于 `lora_data_`。`UseLoRA(0)` 把 ID 0 的数据移入新建对象并写入 `loras_`，当前 ID 变为 0；`UseLoRA(1)` 对 ID 1 做同样处理。再次选择 ID 0 时直接复用已有对象。最终两项都留在 `loras_`，`current_lora_id_` 为 0，公开接口没有逐项卸载操作。
5. 对 `LlmMetadataProto`，工具会额外调用 `ReadLlmMetadataFromSection` 并输出 protobuf 的 `DebugString()`。`PrintKeyValuePair` 没有处理合法的 UInt8 union 值，因此会输出 `Unknown Type`。
6. 分组数减半为 28,125,000，每组 2 字节，scale 共 56,250,000 B。文件在填充前约为 \\(900{,}000{,}000 + 200{,}000{,}000 + 56{,}250{,}000 + 80 \times 2^{20} = 1{,}240{,}136{,}080\\) B，约 1.15 GiB。加 224 MiB KV cache、0.40 GiB 运行开销和 0.25 GiB 余量后约为 2.02 GiB；若再保留一份同规模转换权重，约为 3.18 GiB。分组变大降低了 scale 开销，但“无完整副本时低于 3.0 GiB、有完整副本时超过 3.0 GiB”的结论没有变化。
7. 失败发生在 executor 接入外挂 `TFLiteWeights` 的产物布局层。当前主 executor 只允许 GPU 使用该段的 offset map，CPU 会在 `CompiledModel::Create` 之前返回错误。XNNPACK cache 尚未参与这一结构约束，删除它不会改变结果。
8. 主 loader 建索引时只显式拒绝 `begin_offset > end_offset`，不会统一拒绝两个合法局部范围之间的重叠。发布前校验器应确认每段非空、位于 `[header_end_offset, file_size]` 内，按起点排序后前一段结束位置不大于后一段起点，并检查对齐规则与 `BufferKey` 唯一性。
9. N、P、W 都使用新进程。N 设为 `:nocache`；P 使用空的版本目录并允许生成 cache；W 使用 P 产物的受控副本。外部墙钟从调用 Engine 创建之前开始，到创建返回后结束；首次 prefill 和 decode 另计。文件存在只说明后端曾写过产物，不能证明本次兼容并读取了它；还要检查后端日志、文件是否重写，并比较受控时间差。
10. 三个适配器物化后，输入 buffer 的理论大小为 \\(3 \times 28.4375 = 85.3125\\) MiB。切回 A 只改变 `current_lora_id_`，数值不减少。8 个适配器都物化后为 \\(8 \times 28.4375 = 227.5\\) MiB，尚未计入源数据视图、allocator 对齐和后端元数据。
11. 存储峰值为 \\(3.66 \times 2 + 0.28 + 0.44 + 0.60 + 0.50 = 9.14\\) GB，比 9.0 GB 配额多 0.14 GB。排空旧 Engine 后再创建新 Engine 不会删除旧模型、旧 cache、压缩包或余量，所以存储峰值不变；它避免新旧 Engine 工作集同时存在，运行内存峰值由“双 Engine 重叠”改为“两个 Engine 依次出现”。但 Engine 数量不是内存比例：页驻留、初始化临时量、会话和系统内存随时间变化，不能据此断言峰值减半。

## 第 8 章

1. 设备侧采样避免把完整 logits 回传 CPU，并可把选中 token 写入下一步设备输入。少量 token id 仍返回 host，供停止检测、detokenize 和回调使用；见第 8 章“GPU：并行执行与设备侧采样”。
2. `sched_setaffinity(0, ...)` 限制调用线程可运行的 CPU 集合，具体 CPU 和运行时机仍由调度器决定。系统调用成功后约束由内核执行，不是软性提示；失败时当前引擎创建方记录 warning 后继续。
3. 数据只说明 8 线程是 1、2、4、8 四个测试点中的最高值。未测试更多线程，不能定位饱和点；未测功耗、温度和持续性能，也不能据此选择部署配置。
4. 算子实现、归约顺序与舍入可能改变 logits，首个 token 分叉又会改变后续上下文。应先固定模型、prompt、后端配置、采样参数、随机种子和线程数，再比较首个分叉 step 的 logits 或中间张量；见第 8 章“后端差异为何可能改变输出”。
5. 需要扩展 `Backend` 及其字符串解析、executor、模型资源构造、配置项和执行器工厂分支；专属 cache、sampler 或 delegate 也应进入配置。`LlmExecutor`、Session 与 Conversation 接口可以保持不变。
6. `Duplicate()` 只证明 LiteRT-LM 复用了句柄且没有显式复制。还要核对底层 storage、buffer 类型、shape、元素类型、layout、stride 与对齐是否相容，确认生产者完成事件可传给消费者，并用 delegate/驱动 trace 排除隐藏的分配、格式转换和复制。代码证据的边界是“共享句柄”；端到端零拷贝需要运行时证据，见第 8 章“缓冲驻留：共享条件与退化路径”。
7. 每步差值为 \\(128\ \text{ms} \div 256 = 0.5\ \text{ms/step}\\)。若按 1 MiB FP32 logits 折算，等效速率为 \\(1\ \text{MiB} \div 0.5\ \text{ms} \approx 2.10\ \text{GB/s}\\)。分母还包含两种 sampler 的计算、同步与等待，因此它不是互连带宽测量值；见第 8 章“如何观察同步与复制成本”。

## 第 9 章

1. 一轮归一化成本为 1.3，期望产出为 \\(1+p+p^2+p^3\\)。令二者相等，数值解约为 \\(p=0.233\\)；例如 \\(0.233+0.233^2+0.233^3\approx 0.300\\)。这只是 \\(G=3\\)、\\(c=0.1\\) 且条件概率恒定时的边界，不能外推为固定接受阈值。
2. verify 同时给出各候选位置的基础模型输出。首个草稿不匹配时，该位置的基础模型输出成为 bonus token，所以 `Draft()` 仍返回 1 个 token。drafter 前向、verify 形状和缓冲操作仍会增加轮次成本；即使至少返回 1 个 token，端到端成本仍可能上升。
3. verify signature 的 `input_pos` 维度是编译期固定的形状，\\(G\\) 等于该维度减 1（`runtime/executor/llm_litert_mtp_drafter.cc:256`）。改 \\(G\\) 意味着重新导出模型。
4. \\(\widehat r=\sum K_j/(RG)=0.4\\)，所以每轮平均接受 \\(G\widehat r=1.2\\) 个草稿，连同 bonus token 平均返回 2.2 个 token。理论曲线中的 \\(p\\) 是在此前位置都匹配的条件下继续匹配的概率；聚合比例 \\(\widehat r\\) 是接受前缀长度的样本平均，二者定义不同。
5. 前半是当前 token 的词嵌入（2560 维，来自 embedder 查表），后半是主干上一步的隐藏态 activation（2560 维，来自 decode/verify 的输出缓冲）。

## 第 10 章

1. `target_px` 为 \\(256 \times 16 \times 16 = 65536\\)；`factor` 为 \\(\sqrt{65536 \div (768 \times 512)} \approx 0.408\\)。理想尺寸约为 313.5 × 209.0，按 `side_mult`（\\(1 \times 16 = 16\\)）向下对齐后得到 304 × 208；patch 数为 19 × 13 = 247。无输出 mask 且 `patch_num_shrink_factor = 4` 时，visual token 数为 \\(\lceil 247 \div 4 \rceil = 62\\)。
2. 视觉、音频占位符分别为 −1、−2。执行管理器按模态 embedding 行数或有效 token 数插入等量占位符，`EmbeddingLookupMultiModal` 再逐行替换；见第 10 章图像、音频 embedding 替换两节。
3. 每个 token 的混合 KV 数据量为 \\(2 \times 2 \times (20 \times 256 + 4 \times 512) \times 1\ \text{B} = 28\ \text{KiB}\\)。280 个 visual token 因而对应 \\(280 \times 28\ \text{KiB} = 7.65625\ \text{MiB}\\)，约 7.66 MiB。`model_dimension = 2560` 是主干 embedding 宽度；KV 容量由各层的 KV 头数、头维度和数据类型决定，二者不能互换。
4. `features[1, 204, 1536]` 表示 batch 为 1、编码器输出有 204 个特征位置、每个位置宽 1536。原始 log-mel 帧此前还经过分块和编码器缩减，不能把 204 解释为原始频谱帧数。
5. FC 文法保证 `call:函数名{参数}` 的结构、参数值词法与 EOF，不保证函数在白名单中、参数满足完整 schema、调用者有权限或目标可执行。应用仍须完成这些检查并处理执行错误与结果回填；见第 10 章“Tool Use：结构解析与应用执行”。
6. 实现时序为：首步 `MaskLogits(s0) → Sample(y0)`；下一步先 `UpdateConstraintState(y0)`，再 `MaskLogits(s1) → Sample(y1)`；第三步先提交 `y1`，再屏蔽并采样 `y2`。`y2` 只在下一步开始时提交。首步传入执行器的是 prefill 留下的最后一个输入 token，并非约束采样生成的 token，因此不能用它推进约束状态；见第 10 章“状态推进发生在下一次采样之前”。
7. 最先返回错误的是 Gemma 4 模型数据处理器：它消费唯一的图像标记后发现图像队列仍有一项，返回 `Provided more images than expected in the prompt.`。错误发生在消息与模板标记配对阶段，视觉执行器尚未运行；见第 10 章“按数据边界定位多模态输入失败”。
8. 宿主先规范化目标路径，只允许写入授权根目录，并拒绝越界、符号链接跳转和超限内容；再按当前用户、资源和操作重新授权。幂等键必须来自可信请求上下文，写入采用可恢复的原子提交；外部适配器设置超时、取消与重试上限。结果只回填最小结构化状态、receipt 或错误码，并裁剪、转义和脱敏。parser 输出仍是不可信输入，LiteRT-LM 不负责文件系统授权或副作用一致性；见第 10 章“Tool Use：结构解析与应用执行”。

## 第 11 章

1. 好处是 C++ 对象布局不进入 ABI；只要 C 函数签名、所有权与行为契约保持兼容，内部类布局可以调整。代价是调用方只能通过句柄和 C 函数操作对象，不能直接访问成员，也需要为创建、错误与生命周期设计显式接口。
2. 两个案例的外部来源、版本条件和完整分析见 11.8 节。题目前一个案例讨论 Swift `Conversation` 的确定释放，后一个案例讨论 Swift actor `Engine` 的销毁线程。两者涉及不同对象与故障表现：前一个案例不能证明 v0.13.1 核心 Engine 普遍只支持一个 Session，后一个案例也不能替代 Conversation 的生命周期证据。
3. 本章两个 `const char*` 示例都由所属句柄持有，调用方不得单独 `free`。response 文本在 `LiteRtLmResponses` 删除前有效；渲染结果还会在下一次渲染时被覆盖。绑定层若需更长生命周期，必须在相应失效点之前复制；继续访问失效指针会形成悬空引用。
4. 停止词、暂存和采样编排都以 token id 序列为输入，与 id 的产生方式无关；脚本化的假执行器给出确定的 id 流即可覆盖这些逻辑。此类测试不能覆盖量化误差、后端数值差异或真实性能。
5. 创建：`litert_lm_engine_settings_create`、必要的 settings setter、`litert_lm_engine_create`，再用 `litert_lm_engine_create_session(engine, NULL)` 采用默认会话配置。使用：构造 `LiteRtLmInputData`，调用 `litert_lm_session_run_prefill` 与 `litert_lm_session_run_decode`，再用 responses getter 复制所需文本。销毁：每个返回的 responses 调 `litert_lm_responses_delete`，随后依次调用 `litert_lm_session_delete`、`litert_lm_engine_delete` 与 `litert_lm_engine_settings_delete`；若绑定层另建 session config，也要包装对应的 create/delete。
