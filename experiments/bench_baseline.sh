#!/usr/bin/env bash
# 基准数据集 v1 采集（BOOK_PLAN 规格；主模型 Gemma 4 E4B，非受限）。
# 直接产出 CSV：experiments/data/baseline.csv 与 experiments/data/mtp.csv。
# 指标来自 litert-lm benchmark 输出：Prefill speed / Decode speed / Init time / TTFT
# （该 CLI 不报峰值内存，故不含该列）。
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

MODEL="${MODEL:-gemma-4-e4b}"
DECODE=128
CONTEXTS=(256 1024 4096)
BACKENDS=(cpu gpu)
REPEATS=3
OUT=data
mkdir -p "$OUT"

# 从 benchmark 输出提取四个数（单位后缀去掉）
field() { grep -i "$1" | head -1 | grep -oE '[0-9]+\.?[0-9]*' | head -1; }

run_one() { # backend context extra_flags -> "prefill,decode,init,ttft"
  local b="$1" c="$2"; shift 2
  local o; o="$(litert-lm benchmark "$MODEL" --backend "$b" --prefill-tokens "$c" \
        --decode-tokens "$DECODE" --cache disk "$@" 2>&1)"
  local p d i t
  p="$(printf '%s\n' "$o" | field 'Prefill speed')"
  d="$(printf '%s\n' "$o" | field 'Decode speed')"
  i="$(printf '%s\n' "$o" | field 'Init time')"
  t="$(printf '%s\n' "$o" | field 'Time to first token')"
  echo "${p:-NA},${d:-NA},${i:-NA},${t:-NA}"
}

record_meta() {
  {
    echo "# 基准数据集 v1"
    echo "date=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    echo "ram=$(sysctl -n hw.memsize 2>/dev/null | awk '{print $1/1073741824" GiB"}')"
    echo "os=$(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "litert_lm=$(litert-lm --version 2>&1 | head -1)"
    echo "model=$MODEL (litert-community/gemma-4-E4B-it-litert-lm)"
    echo "decode_tokens=$DECODE repeats=$REPEATS contexts=${CONTEXTS[*]} backends=${BACKENDS[*]}"
  } > "$OUT/_meta.txt"; cat "$OUT/_meta.txt"
}

echo "backend,context,repeat,prefill_tok_s,decode_tok_s,init_s,ttft_s" > "$OUT/baseline.csv"
record_meta
for b in "${BACKENDS[@]}"; do
  for c in "${CONTEXTS[@]}"; do
    for i in $(seq 1 "$REPEATS"); do
      echo ">>> baseline backend=$b context=$c run=$i"
      echo "$b,$c,$i,$(run_one "$b" "$c" --enable-speculative-decoding false)" >> "$OUT/baseline.csv"
    done
  done
done

# MTP 开/关（Gemma 4 E4B 支持 MTP）；固定 context=1024
echo "backend,mtp,repeat,prefill_tok_s,decode_tok_s,init_s,ttft_s" > "$OUT/mtp.csv"
for b in "${BACKENDS[@]}"; do
  for mode in false auto; do
    for i in $(seq 1 "$REPEATS"); do
      echo ">>> mtp backend=$b mode=$mode run=$i"
      echo "$b,$mode,$i,$(run_one "$b" 1024 --enable-speculative-decoding "$mode")" >> "$OUT/mtp.csv"
    done
  done
done

echo "=== 采集完成 ==="
echo "--- baseline.csv ---"; cat "$OUT/baseline.csv"
echo "--- mtp.csv ---"; cat "$OUT/mtp.csv"
