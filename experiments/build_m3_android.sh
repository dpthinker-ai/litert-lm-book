#!/usr/bin/env bash
set -euo pipefail
book_dir=$(cd "$(dirname "$0")/.." && pwd)
source_dir=${LITERT_LM_SOURCE:-/Users/dpthinker/workspace/LiteRT-LM-v0.13.1}
export ANDROID_HOME=${ANDROID_HOME:-$HOME/Library/Android/sdk}
export ANDROID_NDK_HOME=${ANDROID_NDK_HOME:-$ANDROID_HOME/ndk/28.2.13676358}
output_dir=${M3_BUILD_DIR:-$book_dir/tmp/m3-build}
expected=a0afb5a56acd106b23a2b2385b8469834dc268c0
[[ $(git -C "$source_dir" rev-parse HEAD) == "$expected" ]]
[[ -z $(git -C "$source_dir" status --porcelain) ]]
mkdir -p "$output_dir/bundle"
cd "$source_dir"
# macOS host tools (including protoc) need the installed SDK version.
sdk_version=$(xcrun --show-sdk-version)
bazel build --config=android_arm64 --macos_sdk_version="$sdk_version" \
  --define=litert_link_capi_so=true --define=resolve_symbols_in_exec=false \
  //python/litert_lm:litert-lm >"$output_dir/runtime-build.log" 2>&1
cp -f bazel-bin/python/litert_lm/liblitert-lm.so "$output_dir/bundle/"
cp -f bazel-bin/external/litert/litert/c/libLiteRt.so "$output_dir/bundle/"
cp -f prebuilt/android_arm64/*.so "$output_dir/bundle/"
compiler="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android31-clang++"
"$compiler" -std=c++17 -O2 -static-libstdc++ -I"$source_dir" \
  "$book_dir/experiments/m3_observe.cc" -L"$output_dir/bundle" -llitert-lm \
  '-Wl,-rpath,$ORIGIN' -o "$output_dir/bundle/m3_observe"
"$compiler" --version > "$output_dir/compiler.txt"
cp "$ANDROID_NDK_HOME/source.properties" "$output_dir/ndk.txt"
printf '%s\n' "$expected" > "$output_dir/source-commit.txt"
shasum -a 256 "$output_dir/bundle/"* > "$output_dir/bundle-sha256.txt"
git status --porcelain > "$output_dir/source-status.txt"
