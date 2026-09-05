#!/usr/bin/env bash
set -euo pipefail
book_dir=$(cd "$(dirname "$0")/.." && pwd)
source_dir=${LITERT_LM_SOURCE:-/Users/dpthinker/workspace/LiteRT-LM-v0.13.1}
output_dir=${M4_BUILD_DIR:-$book_dir/tmp/m4-build}
# Reuse the frozen runtime build recipe; measurement clients remain separate.
bash "$book_dir/experiments/build_m3_android.sh"
runtime_dir=${M3_BUILD_DIR:-$book_dir/tmp/m3-build}
mkdir -p "$output_dir/bundle"
cp -f "$runtime_dir/bundle/"*.so "$output_dir/bundle/"
ndk_dir=${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/28.2.13676358}
compiler="$ndk_dir/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android31-clang++"
for name in m4_observe m4_token_audit; do
  "$compiler" -std=c++17 -O2 -static-libstdc++ -I"$source_dir" \
    "$book_dir/experiments/$name.cc" -L"$output_dir/bundle" -llitert-lm \
    '-Wl,-rpath,$ORIGIN' -o "$output_dir/bundle/$name"
done
cp "$runtime_dir/compiler.txt" "$runtime_dir/ndk.txt" "$runtime_dir/source-commit.txt" "$runtime_dir/source-status.txt" "$output_dir/"
shasum -a 256 "$output_dir/bundle/"* > "$output_dir/bundle-sha256.txt"
