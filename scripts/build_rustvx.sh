#!/usr/bin/env bash
################################################################################
#
# build_rustvx.sh — clone, build, and stage rustVX (a CTS-conformant Rust
# OpenVX 1.3.1 implementation) for use as an openvx-mark backend.
#
# Why this script exists
# ----------------------
# AMD MIVisionX is feature-rich on the Vision profile but does not
# implement most of the Enhanced Vision Feature Set (BilateralFilter,
# Copy, HOG, Tensor*, Select, ScalarOperation, NonMaxSuppression, …).
# rustVX is CTS-conformant for *both* profiles (5923/5923 vision +
# 1235/1235 enhanced vision), so using it as a second OpenVX backend
# lets the openvx-mark suite produce real measured numbers for all 19
# enhanced_vision kernels in addition to the 42 vision kernels.
#
# What it does
# ------------
#   1. Clones (or updates) rustVX into ../rustVX (sibling of this repo)
#      unless --src is overridden.
#   2. Runs `cargo build --release` with the SSE2/AVX2/parallel features
#      that match the rustVX CI config (the headline performance config).
#   3. Creates two backward-compat symlinks in the build artifact dir:
#        libopenvx     -> libopenvx_ffi
#        libvxu        -> libopenvx_ffi
#      so build systems that hard-code `find_library(NAMES openvx vxu)`
#      pick rustVX up without modification.
#
# Usage
# -----
#   scripts/build_rustvx.sh                         # clone to ../rustVX, build
#   scripts/build_rustvx.sh --src /path/to/rustVX   # use an existing checkout
#   scripts/build_rustvx.sh --no-update             # don't `git pull` an
#                                                   #   already-cloned tree
#   scripts/build_rustvx.sh --debug                 # debug build (slower)
#
# Output
# ------
# Prints the absolute paths of OPENVX_INCLUDES and OPENVX_LIB_DIR that
# should be passed to openvx-mark's `cmake` step. Example:
#
#   $ cmake -DOPENVX_INCLUDES=/abs/rustVX/include \
#           -DOPENVX_LIB_DIR=/abs/rustVX/target/release ..
#
################################################################################

set -euo pipefail

# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------
RUSTVX_SRC=""
DO_UPDATE=1
BUILD_PROFILE="release"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)        RUSTVX_SRC="$2"; shift 2 ;;
        --no-update)  DO_UPDATE=0; shift ;;
        --debug)      BUILD_PROFILE="debug"; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# ----------------------------------------------------------------------------
# Resolve paths
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$RUSTVX_SRC" ]]; then
    RUSTVX_SRC="$(cd "$REPO_ROOT/.." && pwd)/rustVX"
fi

# ----------------------------------------------------------------------------
# Clone or update
# ----------------------------------------------------------------------------
if [[ -d "$RUSTVX_SRC/.git" ]]; then
    echo "==> rustVX already cloned at: $RUSTVX_SRC"
    if [[ $DO_UPDATE -eq 1 ]]; then
        echo "==> Pulling latest (use --no-update to skip)"
        git -C "$RUSTVX_SRC" pull --ff-only
        git -C "$RUSTVX_SRC" submodule update --init --recursive
    fi
elif [[ -e "$RUSTVX_SRC" ]]; then
    echo "ERROR: $RUSTVX_SRC exists but is not a git repo." >&2
    echo "       Either remove it or pass --src /elsewhere." >&2
    exit 1
else
    echo "==> Cloning rustVX into: $RUSTVX_SRC"
    git clone --recurse-submodules https://github.com/kiritigowda/rustVX.git "$RUSTVX_SRC"
fi

# ----------------------------------------------------------------------------
# Toolchain check
# ----------------------------------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: 'cargo' not found in PATH. Install Rust toolchain via https://rustup.rs/" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
echo "==> Building rustVX ($BUILD_PROFILE) with SIMD + parallel features"
echo "    SRC: $RUSTVX_SRC"

# Match the rustVX README's headline performance build config.
# `target-cpu=x86-64-v3` is the same baseline openvx-mark uses for the
# Khronos sample in its own Release build, so the two are apples-to-
# apples on AMD Zen 2+ / Intel Haswell+ hardware.
HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
    x86_64|amd64)
        export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=x86-64-v3"
        SIMD_FEATURES="openvx-core/sse2 openvx-core/avx2 openvx-vision/sse2 openvx-vision/avx2 openvx-vision/parallel"
        ;;
    arm64|aarch64)
        SIMD_FEATURES="openvx-core/neon openvx-vision/neon openvx-vision/parallel"
        ;;
    *)
        echo "==> Unknown host arch '$HOST_ARCH' — building scalar (no SIMD)"
        SIMD_FEATURES="openvx-vision/parallel"
        ;;
esac

CARGO_FLAGS=()
if [[ "$BUILD_PROFILE" == "release" ]]; then
    CARGO_FLAGS+=(--release)
fi

(
    cd "$RUSTVX_SRC"
    cargo build "${CARGO_FLAGS[@]}" -p openvx-ffi --features "$SIMD_FEATURES"
)

# ----------------------------------------------------------------------------
# Compute artifact paths + create backward-compat symlinks
#
# `CARGO_TARGET_DIR` (set globally — e.g. by IDEs that redirect cargo
# output to a shared cache) overrides the default `<src>/target` path.
# Honour it here so we look for the .so/.dylib in the place cargo
# actually wrote it.
# ----------------------------------------------------------------------------
case "$(uname -s)" in
    Darwin)   LIB_EXT="dylib" ;;
    Linux|*)  LIB_EXT="so"    ;;
esac

if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
    RUSTVX_LIB_DIR="$CARGO_TARGET_DIR/$BUILD_PROFILE"
else
    RUSTVX_LIB_DIR="$RUSTVX_SRC/target/$BUILD_PROFILE"
fi

LIB_FFI="$RUSTVX_LIB_DIR/libopenvx_ffi.$LIB_EXT"
if [[ ! -f "$LIB_FFI" ]]; then
    echo "ERROR: build succeeded but $LIB_FFI is missing." >&2
    echo "       CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-<unset>}" >&2
    exit 1
fi

# CMakeLists.txt is already taught to recognize `libopenvx_ffi`, so the
# symlinks below are strictly belt-and-suspenders for downstream tools
# (e.g. the Khronos CTS build, third-party scripts) that hard-code the
# legacy `libopenvx` / `libvxu` names.
for legacy in libopenvx libvxu; do
    LEGACY_PATH="$RUSTVX_LIB_DIR/${legacy}.${LIB_EXT}"
    if [[ ! -e "$LEGACY_PATH" ]]; then
        ln -s "libopenvx_ffi.${LIB_EXT}" "$LEGACY_PATH"
        echo "==> Symlinked: $LEGACY_PATH -> libopenvx_ffi.$LIB_EXT"
    fi
done

# ----------------------------------------------------------------------------
# Done — print the cmake invocation
# ----------------------------------------------------------------------------
cat <<EOF

✓ rustVX built successfully.

  Library:  $LIB_FFI
  Headers:  $RUSTVX_SRC/include

To build openvx-mark against rustVX:

  cd $REPO_ROOT/build
  cmake -DOPENVX_INCLUDES=$RUSTVX_SRC/include \\
        -DOPENVX_LIB_DIR=$RUSTVX_LIB_DIR ..
  cmake --build . -j

To run a 3-way comparison (AMD MIVisionX + rustVX + OpenCV):

  scripts/compare_three_way.sh --rustvx-src $RUSTVX_SRC

EOF

# Export the paths in a machine-readable form so callers can source this
# script with eval (e.g. the compare_three_way.sh wrapper). Only emit
# the eval block when stdout is a pipe.
if [[ ! -t 1 ]]; then
    echo "RUSTVX_OPENVX_INCLUDES='$RUSTVX_SRC/include'"
    echo "RUSTVX_OPENVX_LIB_DIR='$RUSTVX_LIB_DIR'"
fi
