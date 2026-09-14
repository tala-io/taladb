#!/usr/bin/env bash
#
# Link the Android JSI glue and assert its LOAD segments are 16 KB aligned.
#
# A device with 16 KB pages cannot load a library aligned to 4 KB, and Google
# Play requires 16 KB support of every app targeting Android 15 or later. This
# shipped broken in 0.11.5: every translation unit compiled cleanly, the CI
# syntax check passed, and the library still could not load — because alignment
# is decided by the linker, and nothing in CI ever reached the link step.
#
# So this drives the real CMakeLists with the real NDK. Gradle normally supplies
# two things CMake needs; both are stubbed here, because neither participates in
# how the output is aligned:
#
#   - the ReactAndroid prefab package, replaced by an interface target carrying
#     only the JSI include path;
#   - libtaladb_ffi.so, replaced by an empty library, with undefined symbols
#     ignored at link time.
#
# What is NOT stubbed is the toolchain, the link options, or CMakeLists.txt.
#
# Usage:  ANDROID_NDK_HOME=... ./scripts/check-android-alignment.sh [abi ...]
#         RN_JSI_DIR=...      overrides JSI header discovery.
set -euo pipefail

ABIS=("$@")
[ ${#ABIS[@]} -eq 0 ] && ABIS=(arm64-v8a armeabi-v7a x86_64)

RN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_LATEST_HOME:-${ANDROID_NDK_ROOT:-}}}"
[ -n "$NDK" ] || { echo "error: set ANDROID_NDK_HOME"; exit 1; }
TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"
[ -f "$TOOLCHAIN" ] || { echo "error: no toolchain at $TOOLCHAIN"; exit 1; }

JSI="${RN_JSI_DIR:-$(cd "$RN_DIR" && node -e "const p=require('path'),r=require.resolve('react-native/package.json');console.log(p.join(p.dirname(r),'ReactCommon','jsi'))")}"
[ -f "$JSI/jsi/jsi.h" ] || { echo "error: no jsi.h under $JSI"; exit 1; }

# Globbed rather than `ls | head -1`: every early-exiting pipe in this script
# used to be a race against SIGPIPE (see the alignment read below).
readelfs=("$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-readelf)
clangs=("$NDK"/toolchains/llvm/prebuilt/*/bin/clang)
READELF="${readelfs[0]}"
CLANG="${clangs[0]}"
[ -x "$READELF" ] && [ -x "$CLANG" ] || { echo "error: no llvm toolchain under $NDK"; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Stand in for the prefab package `find_package(ReactAndroid REQUIRED CONFIG)`
# would otherwise resolve out of the Gradle build.
mkdir -p "$WORK/stub/lib/cmake/ReactAndroid"
cat > "$WORK/stub/lib/cmake/ReactAndroid/ReactAndroidConfig.cmake" <<STUB
add_library(ReactAndroid::jsi INTERFACE IMPORTED)
set_target_properties(ReactAndroid::jsi PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "$JSI")
STUB

status=0
for abi in "${ABIS[@]}"; do
  # CMakeLists adds src/main/jniLibs/<abi> to the link path, but the prebuilt
  # Rust library is produced by release.yml and is not in the tree. Supply an
  # empty stand-in on a second -L outside the repository — writing one into
  # jniLibs instead would leave an artifact a stray `git add` can commit.
  stub_lib="$WORK/libs/$abi"
  mkdir -p "$stub_lib"
  triple="$(case $abi in
    arm64-v8a)   echo aarch64-linux-android24 ;;
    armeabi-v7a) echo armv7a-linux-androideabi24 ;;
    x86_64)      echo x86_64-linux-android24 ;;
  esac)"
  echo 'void taladb_ffi_placeholder(void) {}' > "$WORK/stub.c"
  "$CLANG" --target="$triple" -shared -Wl,-soname,libtaladb_ffi.so \
    -o "$stub_lib/libtaladb_ffi.so" "$WORK/stub.c"

  cmake -S "$RN_DIR/android" -B "$WORK/build-$abi" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DANDROID_ABI="$abi" -DANDROID_PLATFORM=android-24 -DANDROID_STL=c++_shared \
    -DReactAndroid_DIR="$WORK/stub/lib/cmake/ReactAndroid" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$stub_lib -Wl,--unresolved-symbols=ignore-all" \
    -DCMAKE_BUILD_TYPE=Release

  # Output is not captured to a file on purpose: $WORK sits on whatever TMPDIR
  # points at, and a full or read-only one made a failure here print nothing at
  # all. A few lines of cmake chatter is a fair price for always seeing why.
  cmake --build "$WORK/build-$abi" --target taladb_jsi -j"$(nproc)"

  so="$(find "$WORK/build-$abi" -name libtaladb_jsi.so -print -quit)"
  [ -n "$so" ] || { echo "error: $abi linked but produced no libtaladb_jsi.so"; exit 1; }

  # awk reads to EOF instead of exiting at the first LOAD. Quitting early sends
  # SIGPIPE to llvm-readelf, which exits 74 (EX_IOERR) and fails the whole run
  # under `set -o pipefail` — intermittently, since it is a race against how
  # much readelf has managed to write.
  align="$("$READELF" -lW "$so" | awk '$1=="LOAD" && !seen { a=$NF; seen=1 } END { print a }')"

  case "$align" in
    0x4000|0x10000) printf '  %-14s %-8s OK\n' "$abi" "$align" ;;
    *)
      printf '  %-14s %-8s NOT 16 KB ALIGNED\n' "$abi" "$align"
      echo "::error file=packages/bindings/react-native/android/CMakeLists.txt::libtaladb_jsi.so is aligned to $align for $abi. It must link with -Wl,-z,max-page-size=16384 or Android 15+ devices cannot load it."
      status=1
      ;;
  esac
done

exit $status
