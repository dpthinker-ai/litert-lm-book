# gemma-4-e4b model.litertlm 文件分析（16 KiB 对齐扫描 + TFLite FlatBuffer 解析）
# 文件 3.66 GB @ litert-community/gemma-4-E4B-it-litert-lm，运行时 v0.13.1
# 方法：扫 16KiB 边界找 TFL3 魔数定段；tflite python 绑定读 SignatureDefs 与张量形状

offset,size_mb,signatures,key_tensor
4734976,170.9,embedder,token_ids[1;1]
175669248,836.8,per_layer_embedder,token_ids[1;1]
1012449280,94.1,serving_default(audio_encoder),mask[1;1;816]
1106509824,15.7,audio_adapter,features[1;204;1536]
1122254848,0.016,eoa,-
1122271232,224.1,vision_70|vision_140|vision_280,images[1;1260;768]
1346420736,7.9,vision_adapter_70|140|280,soft_tokens[1;140;768]
1354317824,0.016,eoi,-
1354334208,2260.1,decode|prefill_1024|prefill_128|verify,embeddings[1;1;2560]
3614392320,45.1,mtp_drafter,activations[1;1;5120]

# 主模型 decode signature 关键事实：
# - KV cache 输入 48 个张量 = 24 层 × (K+V)，dtype 全部 INT8
#   - 20 层形状 K:[1,2,32003,256] / V:[1,2,256,32003]（H_kv=2, D=256）
#   -  4 层形状 K:[1,2,32003,512] / V:[1,2,512,32003]（H_kv=2, D=512）
# - KV 每 token 字节 = 2(KV) × 2(H) × (20×256 + 4×512) × 1B = 28,672 B = 28 KiB/token
# - 4096 上下文 KV 合计 = 112 MiB；静态槽位 32003 全预留 ≈ 875 MiB
# - embeddings 输入 [1,1,2560] → model_dimension=2560
# - per_layer_embeddings [1,1,42,256]（42 层 PLE × 256）
# - logits [1,1,262144] → 词表 262,144（=2^18）
# - param_tensor [1,1,1,7] INT32（单缓冲 KV 路径的位置参数，见 ch06）
# - mask [1,1,1,32003] BOOL
# 2026-09-14 复核：decode 段的 `decode` 与 `verify` signature 的 logits 输出 dtype 均为 FLOAT32（tflite 2.18.0 解析 signature 输出张量；形状分别为 [1,1,262144] 与 [1,4,262144]）
