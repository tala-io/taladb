#!/usr/bin/env bash
# Generate the C header for the taladb-ffi crate.
#
#   include/taladb.h                          — canonical; what the Swift and
#                                               Kotlin packages and the release
#                                               archives ship
#   ../react-native/cpp/taladb.h              — byte-identical copy, because the
#                                               npm package cannot reach outside
#                                               its own directory
#
# Usage:
#   scripts/gen-header.sh           regenerate both files
#   scripts/gen-header.sh --check   fail if either file is stale (used by CI)
#
# Requires cbindgen (`cargo install cbindgen --locked`).
set -euo pipefail

FFI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANONICAL="$FFI_DIR/include/taladb.h"
RN_COPY="$FFI_DIR/../react-native/cpp/taladb.h"

GENERATED="$(mktemp)"
trap 'rm -f "$GENERATED"' EXIT

(cd "$FFI_DIR" && cbindgen --config cbindgen.toml --crate taladb-ffi --output "$GENERATED")

if [[ "${1:-}" == "--check" ]]; then
    status=0
    for target in "$CANONICAL" "$RN_COPY"; do
        # A stale header links fine and then corrupts the stack at run time,
        # because the caller compiles against a signature the library does not
        # export.
        if ! diff -u "$target" "$GENERATED"; then
            echo "::error file=${target#"$FFI_DIR/../../../"}::Header is stale. Run: packages/bindings/ffi/scripts/gen-header.sh"
            status=1
        fi
    done
    exit "$status"
fi

mkdir -p "$(dirname "$CANONICAL")"
cp "$GENERATED" "$CANONICAL"
cp "$GENERATED" "$RN_COPY"
echo "Wrote $CANONICAL"
echo "Wrote $RN_COPY"
