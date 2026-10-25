#!/usr/bin/env bash
set -euo pipefail
native_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd -- "$native_dir/../../.." && pwd)"
headers="${LITERT_HEADERS:-/private/tmp/LiteRT-26895c9fbcc25c43faa8c1a98cd1fd28951602c3}"
json_headers="${JSON_HEADERS:-/private/var/tmp/_bazel_dsha0062/7e530b019b8f3b749ec659c9f32acda2/external/nlohmann_json/include}"
ndk="${ANDROID_NDK_HOME:-/path/to/user/Library/Android/sdk/ndk/30.0.14904198}"
output="$workspace/annoymous/app/src/androidTest/jniLibs/arm64-v8a/libkvreconstruction.so"
mkdir -p "$(dirname -- "$output")"
"$ndk/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android31-clang++" \
 -shared -fPIC -std=c++17 -O2 -static-libstdc++ -I "$headers" -I "$json_headers" -I "${json_headers%/nlohmann_json/include}/stb" \
 "$native_dir/worker.cc" -L "$workspace/annoymous/app/src/main/jniLibs/arm64-v8a" \
 -llitert-lm -llog -ldl -Wl,-z,max-page-size=16384 -Wl,--no-undefined -o "$output"
printf '%s\n' "Built $output"
