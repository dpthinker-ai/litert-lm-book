#!/usr/bin/env bash
# --max-num-tokens 扫描：验证 KV cache 预留宽度对 decode 速度的影响（解释 LiteRT-LM#2568）。
# 解析方式与 bench_baseline.sh 一致；输出 experiments/data/max_tokens_sweep.csv。
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

MODEL="${MODEL:-gemma-4-e4b}"
DECODE=128
CONTEXT=256
SIZES=(1024 2048 4096 8192)
REPEATS=2
OUT=data
mkdir -p "$OUT"

field() { grep -i "$1" | head -1 | grep -oE '[0-9]+\.?[0-9]*' | head -1; }

run_one() { # max_num_tokens context -> "prefill,decode,init,ttft,status"
  local mt="$1"
  local context="$2"
  local o status=ok
  if ! o="$(litert-lm benchmark "$MODEL" --backend cpu -p "$context" \
        --decode-tokens "$DECODE" --cache disk --max-num-tokens "$mt" \
        --enable-speculative-decoding false 2>&1)"; then
    status=failed
  fi
  local p d i t
  p="$(printf '%s\n' "$o" | field 'Prefill speed')"
  d="$(printf '%s\n' "$o" | field 'Decode speed')"
  i="$(printf '%s\n' "$o" | field 'Init time')"
  t="$(printf '%s\n' "$o" | field 'Time to first token')"
  echo "${p:-NA},${d:-NA},${i:-NA},${t:-NA},$status"
}

echo "backend,max_num_tokens,prompt_tokens,repeat,prefill_tok_s,decode_tok_s,init_s,ttft_s,status" > "$OUT/max_tokens_sweep.csv"
for mt in "${SIZES[@]}"; do
  for i in $(seq 1 "$REPEATS"); do
    echo ">>> sweep max_num_tokens=$mt run=$i"
    echo "cpu,$mt,$CONTEXT,$i,$(run_one "$mt" "$CONTEXT")" >> "$OUT/max_tokens_sweep.csv"
  done
done

# Issue #2568 used a shorter prompt. Keep this diagnostic as a separate,
# explicitly labelled condition; do not mix it into the controlled 256-token
# speed comparison.
for i in $(seq 1 "$REPEATS"); do
  echo ">>> diagnostic max_num_tokens=1024 prompt=100 run=$i"
  echo "cpu,1024,100,$i,$(run_one 1024 100)" >> "$OUT/max_tokens_sweep.csv"
done

echo "=== 完成 ==="; cat "$OUT/max_tokens_sweep.csv"
