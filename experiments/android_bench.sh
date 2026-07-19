#!/usr/bin/env bash
# Android 真机扩展基准采集（BOOK_PLAN：扩展基准单独标注，不与 Mac 主基准混算）。
# 前置：手机已开开发者模式+USB 调试；litert_lm_advanced_main（android_arm64）与模型已 push 到设备。
# 用法：experiments/android_bench.sh；输出 experiments/data/android_baseline.csv
set -uo pipefail
cd "$(dirname "$0")"

DEVICE_DIR=/data/local/tmp/litertlm
BIN="$DEVICE_DIR/litert_lm_advanced_main"
MODEL="$DEVICE_DIR/model.litertlm"
DECODE=128
CONTEXTS=(256 1024 4096)
BACKENDS=(cpu gpu)
REPEATS=3
OUT=data
mkdir -p "$OUT"

die() { echo "ERROR: $*" >&2; exit 1; }
adb get-state >/dev/null 2>&1 || die "无 adb 设备（检查 USB 调试与授权）"
adb shell "test -x $BIN" 2>/dev/null || die "设备上没有 $BIN（先 push 二进制与模型，见注释块）"
adb shell "test -f $MODEL" 2>/dev/null || die "设备上没有 $MODEL"

# push 参考（手动执行一次）：
#   adb shell mkdir -p /data/local/tmp/litertlm
#   adb push bazel-bin/runtime/engine/litert_lm_advanced_main /data/local/tmp/litertlm/
#   adb push ../LiteRT-LM-v0.13.1/prebuilt/android_arm64/*.so /data/local/tmp/litertlm/
#   # constraint provider 是进程启动依赖；gpu 后端还需要 accelerator 与 sampler 动态库
#   adb push ~/.litert-lm/models/gemma-4-e4b/model.litertlm /data/local/tmp/litertlm/

field() { grep -i "$1" | head -1 | grep -oE '[0-9]+\.?[0-9]*' | head -1; }

peak_private_mib() {
  grep -i 'Peak private footprint' | head -1 |
    grep -oE '[0-9]+\.?[0-9]*[[:space:]]*Mi?B' | head -1 |
    awk '{
      text = $0
      value = text
      sub(/[[:space:]]*Mi?B$/, "", value)
      unit = text
      sub(/^[0-9.]+[[:space:]]*/, "", unit)
      if (unit == "MB") value = value * 1000000 / 1048576
      printf "%.3f\n", value
    }'
}

run_one() { # backend context extra_flags -> "prefill,decode,init,ttft,peak_mb"
  local b="$1" c="$2"; shift 2
  local mt=4096; [ "$c" -ge 4096 ] && mt=8192
  local o; o="$(adb shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR ./litert_lm_advanced_main --backend=$b --model_path=$MODEL \
        --benchmark=true --benchmark_prefill_tokens=$c --benchmark_decode_tokens=$DECODE \
        --max_num_tokens=$mt --report_peak_memory_footprint=true \
        --input_prompt='Write a short story about the ocean.' $* " 2>&1)"
  local p d i t m
  p="$(printf '%s\n' "$o" | field 'Prefill speed')"
  d="$(printf '%s\n' "$o" | field 'Decode speed')"
  i="$(printf '%s\n' "$o" | field 'Init Total' | awk '{printf "%.3f", $1/1000}')"
  t="$(printf '%s\n' "$o" | field 'Time to first token')"
  m="$(printf '%s\n' "$o" | peak_private_mib)"
  echo "${p:-NA},${d:-NA},${i:-NA},${t:-NA},${m:-NA}"
}

{
  echo "# Android 真机扩展基准（与 Mac 主基准分开，不混算）"
  echo "date=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  adb shell getprop ro.product.model
  adb shell getprop ro.hardware
  adb shell getprop ro.build.version.release
} | tee "$OUT/_meta_android.txt"

echo "backend,context,repeat,prefill_tok_s,decode_tok_s,init_total_s,ttft_s,peak_private_mib" > "$OUT/android_baseline.csv"
for b in "${BACKENDS[@]}"; do
  for c in "${CONTEXTS[@]}"; do
    for i in $(seq 1 "$REPEATS"); do
      echo ">>> android backend=$b context=$c run=$i"
      echo "$b,$c,$i,$(run_one "$b" "$c" --enable_speculative_decoding=false)" >> "$OUT/android_baseline.csv"
    done
  done
done

# MTP（context=1024）
echo "backend,mtp,repeat,prefill_tok_s,decode_tok_s,init_total_s,ttft_s,peak_private_mib" > "$OUT/android_mtp.csv"
for b in "${BACKENDS[@]}"; do
  for mode in false true; do
    for i in $(seq 1 "$REPEATS"); do
      echo ">>> android mtp backend=$b mode=$mode run=$i"
      echo "$b,$mode,$i,$(run_one "$b" 1024 --enable_speculative_decoding=$mode)" >> "$OUT/android_mtp.csv"
    done
  done
done

# CPU 线程数扫描（context=1024，解释「多线程未必更快」，第 8 章）
echo "threads,repeat,prefill_tok_s,decode_tok_s,init_total_s,ttft_s,peak_private_mib" > "$OUT/android_threads.csv"
for n in 1 2 4 8; do
  for i in $(seq 1 "$REPEATS"); do
    echo ">>> android threads=$n run=$i"
    echo "$n,$i,$(run_one cpu 1024 --num_cpu_threads=$n)" >> "$OUT/android_threads.csv"
  done
done

# NPU（Qualcomm 机型；不可用则记录错误，不中断）
echo "backend,context,repeat,result" > "$OUT/android_npu.csv"
if adb shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR ./litert_lm_advanced_main --backend=npu --model_path=$MODEL \
     --benchmark=true --benchmark_prefill_tokens=16 --benchmark_decode_tokens=8 --max_num_tokens=1024" >/tmp/npu_probe.log 2>&1; then
  echo ">>> android npu probe OK，采矩阵"
  for c in "${CONTEXTS[@]}"; do
    for i in $(seq 1 "$REPEATS"); do
      echo "npu,$c,$i,$(run_one npu "$c" --enable_speculative_decoding=false)" >> "$OUT/android_baseline.csv"
    done
  done
else
  echo "npu,probe,1,FAILED（见 /tmp/npu_probe.log）" >> "$OUT/android_npu.csv"
  echo "NPU 探测失败，日志已存 /tmp/npu_probe.log"
fi

echo "=== 完成 ==="; tail -n +1 "$OUT/android_baseline.csv" "$OUT/android_mtp.csv" "$OUT/android_threads.csv" "$OUT/android_npu.csv"
