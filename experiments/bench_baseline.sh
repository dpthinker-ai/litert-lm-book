#!/usr/bin/env bash
# 基准数据集 v1 采集（BOOK_PLAN「基准数据集规格 v1」）。
# 前置：litert-lm 已装；模型已 import（默认 ID 见下 MODEL）；已 hf 登录（Gemma 受限）。
# 矩阵：backend {cpu,gpu} × prefill/context {256,1024,4096}，decode 固定 128，每条件跑 3 次。
# 产出：每次运行原始输出存 experiments/data/<backend>_p<ctx>_run<i>.txt，供后续解析取中位数。
# 设备/模型/日期等元信息由 record_meta 写入 experiments/data/_meta.txt。
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

MODEL="${MODEL:-gemma-3n-e2b-int4}"     # litert-lm list 里的 ID；或传本地 .litertlm 路径
DECODE=128
CONTEXTS=(256 1024 4096)
BACKENDS=(cpu gpu)
REPEATS=3
OUT=data
mkdir -p "$OUT"

record_meta() {
  {
    echo "# 基准数据集 v1 元信息"
    echo "采集日期: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "芯片: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    echo "内存: $(sysctl -n hw.memsize 2>/dev/null | awk '{print $1/1073741824" GiB"}')"
    echo "系统: $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "litert-lm: $(litert-lm --version 2>&1 | head -1)"
    echo "模型: $MODEL"
    echo "矩阵: backend={${BACKENDS[*]}} × prefill={${CONTEXTS[*]}} × decode=$DECODE，每条件 $REPEATS 次"
  } > "$OUT/_meta.txt"
  cat "$OUT/_meta.txt"
}

run_matrix() {
  for b in "${BACKENDS[@]}"; do
    for c in "${CONTEXTS[@]}"; do
      for i in $(seq 1 "$REPEATS"); do
        f="$OUT/${b}_p${c}_run${i}.txt"
        echo ">>> backend=$b prefill=$c decode=$DECODE run=$i → $f"
        litert-lm benchmark "$MODEL" \
          --backend "$b" \
          --prefill-tokens "$c" \
          --decode-tokens "$DECODE" \
          --enable-speculative-decoding false \
          --cache disk \
          > "$f" 2>&1 || echo "  （本次运行返回非零，输出已存，见 $f）"
      done
    done
  done
}

record_meta
run_matrix
echo "采集完成。原始输出在 $OUT/，用 parse_bench.py 解析取中位数。"
