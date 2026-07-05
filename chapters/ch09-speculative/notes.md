<!-- edited -->  <!-- 删掉本标记会被 gen_scaffold 覆盖；保留则视为人工维护 -->
# 第 9 章 · 一次前向，多个 token：推测解码与 MTP — notes（策展）

**一句话使命**：讲透 drafter/verifier 机制与接受率经济学——Gemma 4「快 3 倍」（官方口径，实测核对）从哪来。

## 素材取用（单一事实源在 `chapters/_shared/`）
- [`module-executor-llm.md`](../_shared/module-executor-llm.md) — LLM 执行器 (Executor)
- [`module-schema-format.md`](../_shared/module-schema-format.md) — 模型格式与 Schema (Model Format & Schema)

**聚焦**：取 executor-llm 的 MTP drafter、schema-format 的 speculative_decoding 能力声明；#2227 的 PowerVR 回退无真机，按【文档】级引 issue。

## 本章图表（先规划后动笔，CLAUDE.md 第六节）
- [ ] 图 9-1 drafter/verifier 时序
- [ ] 图 9-2 接受率-收益曲线（用本章实测数据绘制）
- [ ] 表 9-1 MTP 开/关实测对照

## 本章实验（脚本入 `experiments/`）
- [ ] --enable-speculative-decoding 开/关
- [ ] 代码与散文两种文体的接受率差异

## 补读 / 缺口（写作前须清零）
- [x] llm_litert_mtp_drafter.cc 实现已补读（2026-07-05，v0.13.1 实读核验，见下）

## 补读结论：MTP `Draft()` 的真实机制（`llm_litert_mtp_drafter.cc @ v0.13.1`，全部逐行核验）

**一次 Draft 调用 = 一次 base 模型前向，吐出 1..num_draft_steps+1 个 token。** 这是提速的根：把带宽墙上"读一遍全部权重"的那一次昂贵前向（第 1 章），摊到多个 token 上。

流程（`Draft()`:453）：
1. `RunDraftingLoop`（:328，循环 :336）：drafter **小模型**逐个草拟 `num_draft_steps_` 个 token（串行，但便宜）。
2. `PrepareVerifierInputBuffers`（:374）：把 [上一个真 token + 草拟的 num_draft_steps 个] 拼成 verify 输入。
3. `RunVerification`（:437）：`base_model_.RunAsync(verify_signature)` **一次前向**验证所有草拟位置，verifier_sampler 采样 → `id_vector`，大小 `num_draft_steps_ + 1`（:449 RET_CHECK_EQ）。
4. 接受循环（:473-483）：`for i in [0, num_draft_steps_)`：若 `verifier_id_vector[i] != drafted_tokens[i]` → 首个不匹配，`bonus_token = verifier_id_vector[i]`（base 的正确 token），break；否则 `num_correct_tokens++`。
5. 若全对（bonus==-1，:481-483）：`bonus_token = verifier_id_vector[num_draft_steps_]` —— 多拿一个"白送"的 bonus token（verify 产 num_draft_steps+1 个，最后一个是免费的）。
6. 输出（:490-492，注释"first token from decode output always correct"）：`drafted_tokens.resize(num_correct)` + `push_back(bonus)`。即 **接受的前缀 + 1 个 bonus**。
7. 统计（:493-494）：`num_drafted_tokens_ += num_draft_steps_`；`num_verified_tokens_ += num_correct_tokens`。
8. 析构（:165-171）：打印**接受率** `num_verified_tokens_ / num_drafted_tokens_`。

关键结论（写正文用）：
- **接受率经济学**：一次 base 前向的成本≈一次普通 decode，但产出 1..G+1 个 token（G=num_draft_steps）。接受率越高，摊得越薄→越快。
- **bonus token 保底**：即使第一个草拟就错，也能拿到 base 给的 1 个正确 token（bonus）——token 数上永不比普通 decode 差；但**多付了 drafter+verify 的开销**。
- **为何某些 GPU 反而更慢（#2227，无真机→【文档】级）**：接受率低或 drafter/verify 在该 GPU 上开销相对高时，草拟白做、净负收益。所以 MTP 是"接受率够高才划算"的赌注。
- `num_draft_steps` 固定（:256 = verify signature `input_pos` 维度 -1），由模型的 verify signature 形状决定，不是运行时可调。
- 能力声明：`HasSpeculativeDecodingSupport(...)`（`schema/capabilities/speculative_decoding.h:33`/`:44` @ v0.13.1）读模型元数据判断是否支持 → 对应 CLI `--enable-speculative-decoding=auto`。
- drafter 是**独立小模型**（`mtp_drafter_model_`）；verify 是 base 模型上的 `"verify"` signature（:63/:228）。

## 待核实清单 / 随手记
- 「快 3 倍」是官方博客口径（Gemma 4），本章实验实测核对；实测数字待基准 D（需 Gemma 4 类支持 MTP 的模型，非 3n E2B）。
- 图 9-1 时序 = drafter 逐个草拟 G 个 → base 一次 verify G+1 → 接受前缀+bonus。