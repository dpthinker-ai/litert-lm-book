#!/usr/bin/env bash
# prefill 长度扫描（附录 C 实验第 4 项）：cpu，-p 100..4000，-d 32，disk 缓存热
set -e
OUT="experiments/data/prefill_sweep.csv"
echo "backend,prefill_tokens,decode_tokens,prefill_tps,decode_tps,init_s,ttft_s" > "$OUT"
for P in 100 250 500 1000 2000 3000 4000; do
  R=$(litert-lm benchmark gemma-4-e4b --backend cpu -p $P -d 32 --cache disk 2>/dev/null | \
      awk '/Prefill speed/{pf=$3}/Decode speed/{dc=$3}/Init time/{it=$3}/Time to first token/{tt=$5}END{print pf","dc","it","tt}')
  echo "cpu,$P,32,$R" >> "$OUT"
  echo "done p=$P: $R"
done
echo "ALL DONE"
