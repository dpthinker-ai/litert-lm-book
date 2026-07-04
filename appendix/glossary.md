# 附录 A · 术语表（工作文件）

> 用途：全书术语的**统一写法**单一事实源（CLAUDE.md 第四节）。写作时新术语随章追加；
> 成书时按"首次出现（成书顺序）"列排序定稿。
> 完整 90 条底本在 `litert-lm-guide/data.json` 的 `synth.glossary`，写到相关章节时取用、核对后并入本表。
>
> 列含义：**术语**（全书统一写法）｜**中文对照**（首次出现时括注）｜**首现章**（成书顺序）｜**备注**（写法约定/易错点）

| 术语 | 中文对照 | 首现章 | 备注 |
|---|---|---|---|
| prefill | 预填充 | 2 | 全书用 prefill，不用"预填充"；首现括注即可 |
| decode | 解码 | 2 | 与 prefill 并列，全书用 decode |
| KV cache | 键值缓存 | 2 | 统一小写 cache；不写 "KV Cache" / "KV-Cache" |
| token | — | 2 | 不译；"逐 token"这类混排照 CJK-ASCII 空格规则 |
| signature | 签名 | 4 | 指 LiteRT CompiledModel 的具名入口；首现需解释，不假设读者已知 |
| Roofline | — | 2 | 屋顶线模型；保留英文，首现括注"屋顶线" |
| speculative decoding | 推测解码 | 9 | 全书用"推测解码" |
| MTP | 多 token 预测 | 9 | Multi-Token Prediction；首现给全称 |
| Engine | — | 3 | 指 LiteRT-LM 的 Engine 类，保留原名不译 |
| Session | 会话 | 3 | 指 Session 类；与口语"会话"混用时以代码体 `Session` 区分 |
| Conversation | 对话（层） | 3 | 指 Conversation 类 |
| quantization | 量化 | 1 | int4 / int8 写法统一小写 |
| tokens/s | — | 2 | 吞吐单位；TTFT 用 ms（CLAUDE.md 第二节单位纪律） |
| TTFT | 首 token 时延 | 2 | time-to-first-token；首现给全称 |
| backend | 后端 | 8 | CPU/GPU/NPU；`Backend` 指代码枚举时用代码体 |
