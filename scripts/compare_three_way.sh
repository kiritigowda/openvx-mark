#!/usr/bin/env bash
################################################################################
#
# compare_three_way.sh — run AMD MIVisionX, rustVX, and OpenCV through
# their respective benchmark binaries, then produce a single combined
# comparison report joining all three on (kernel name, mode, resolution).
#
# The combined matrix gives:
#
#   * **Vision** (42 kernels): AMD vs rustVX vs OpenCV. All three should
#     have measured values; if AMD shows N/A for a specific variant,
#     that's a real impl gap surfaced by the harness.
#
#   * **Enhanced Vision** (19 kernels): AMD typically shows N/A (its
#     runtime doesn't export most of these APIs), but rustVX and OpenCV
#     produce real numbers. The rustVX-vs-OpenCV column on these rows
#     is the headline number — it answers "what does a CTS-conformant
#     OpenVX impl give me on the enhanced-vision kernels OpenCV
#     already does well?"
#
# Usage
# -----
#   scripts/compare_three_way.sh                           # auto-locate rustVX at ../rustVX
#   scripts/compare_three_way.sh --rustvx-src /path/to/rustVX
#   scripts/compare_three_way.sh --resolution VGA,HD,FHD
#   scripts/compare_three_way.sh --iterations 50 --warmup 5
#   scripts/compare_three_way.sh --skip-build              # use existing builds
#   scripts/compare_three_way.sh --output-dir /tmp/cmp     # custom output dir
#
# Output layout
# -------------
#   $OUTPUT_DIR/
#     amd-mivisionx.json        # openvx-mark linked against AMD MIVisionX
#     rustvx.json               # openvx-mark linked against rustVX
#     opencv.json               # opencv-mark
#     comparison.md             # combined 3-way markdown comparison
#     comparison.csv            # combined 3-way CSV
#
# Dependencies
# ------------
#   * cargo (for rustVX build) — install from https://rustup.rs/
#   * AMD MIVisionX at /opt/rocm (or whatever ROCM_PATH points to)
#   * OpenCV 4 (auto-detected by opencv-mark)
#   * cmake + a C++17 toolchain
#
################################################################################

set -euo pipefail

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUSTVX_SRC=""
RESOLUTION="VGA,FHD"
ITERATIONS="100"
WARMUP="10"
THREADS="1"
SKIP_BUILD=0
SKIP_RUSTVX_BUILD=0
SKIP_AMD=0
OUTPUT_DIR=""

# ----------------------------------------------------------------------------
# Arg parsing
# ----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rustvx-src)         RUSTVX_SRC="$2"; shift 2 ;;
        --resolution)         RESOLUTION="$2"; shift 2 ;;
        --iterations)         ITERATIONS="$2"; shift 2 ;;
        --warmup)             WARMUP="$2"; shift 2 ;;
        --threads)            THREADS="$2"; shift 2 ;;
        --skip-build)         SKIP_BUILD=1; shift ;;
        --skip-rustvx-build)  SKIP_RUSTVX_BUILD=1; shift ;;
        --skip-amd)           SKIP_AMD=1; shift ;;
        --output-dir)         OUTPUT_DIR="$2"; shift 2 ;;
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

if [[ -z "$RUSTVX_SRC" ]]; then
    RUSTVX_SRC="$(cd "$REPO_ROOT/.." && pwd)/rustVX"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/build/comparison-three-way"
fi
mkdir -p "$OUTPUT_DIR"

case "$(uname -s)" in
    Darwin)   LIB_EXT="dylib" ;;
    Linux|*)  LIB_EXT="so"    ;;
esac

echo "========================================================================"
echo "  3-way comparison: AMD MIVisionX vs rustVX vs OpenCV"
echo "------------------------------------------------------------------------"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Resolution:  $RESOLUTION"
echo "  Iterations:  $ITERATIONS  (warmup $WARMUP)"
echo "  Threads:     $THREADS"
echo "  rustVX src:  $RUSTVX_SRC"
echo "========================================================================"

# ----------------------------------------------------------------------------
# Build rustVX
# ----------------------------------------------------------------------------
if [[ $SKIP_RUSTVX_BUILD -eq 0 ]]; then
    echo ""
    echo "==> [1/4] Building rustVX..."
    "$SCRIPT_DIR/build_rustvx.sh" --src "$RUSTVX_SRC"
fi

RUSTVX_INCLUDE="$RUSTVX_SRC/include"

# Honour CARGO_TARGET_DIR the same way build_rustvx.sh does — IDEs and
# CI caches frequently redirect cargo output to a shared tree, in which
# case "<src>/target/release" doesn't exist at all and the script
# would fail with a misleading "rustVX library not found" message.
# Mirror the exact resolution logic so the two scripts stay in lockstep.
if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
    RUSTVX_LIB_DIR="$CARGO_TARGET_DIR/release"
else
    RUSTVX_LIB_DIR="$RUSTVX_SRC/target/release"
fi

if [[ ! -f "$RUSTVX_LIB_DIR/libopenvx_ffi.$LIB_EXT" ]]; then
    echo "ERROR: rustVX library not found at $RUSTVX_LIB_DIR/libopenvx_ffi.$LIB_EXT" >&2
    echo "       Tried CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-<unset>}" >&2
    echo "       Run scripts/build_rustvx.sh first, or pass --rustvx-src." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Build openvx-mark x2 (one binary per OpenVX backend) and opencv-mark x1
#
# We use two separate build directories so each binary embeds the right
# `target_link_libraries` pointing at its OpenVX runtime. The opencv-mark
# binary is identical between the two builds; we just take it from one.
# ----------------------------------------------------------------------------
BUILD_AMD="$REPO_ROOT/build"            # default; AMD MIVisionX backend
BUILD_RUSTVX="$REPO_ROOT/build-rustvx"  # alternate; rustVX backend

if [[ $SKIP_BUILD -eq 0 ]]; then
    if [[ $SKIP_AMD -eq 0 ]]; then
        echo ""
        echo "==> [2/4] Building openvx-mark against AMD MIVisionX (+ opencv-mark)..."
        mkdir -p "$BUILD_AMD"
        ( cd "$BUILD_AMD" && cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null )
        cmake --build "$BUILD_AMD" --target openvx-mark opencv-mark -j 4
    fi

    echo ""
    echo "==> [3/4] Building openvx-mark against rustVX..."
    mkdir -p "$BUILD_RUSTVX"
    # When --skip-amd is set the AMD build is skipped entirely, but the
    # 3-way comparison still needs an opencv-mark binary. Build it inside
    # the rustVX tree in that case so we have a runnable opencv-mark
    # without needing the AMD MIVisionX runtime to be present at all.
    # When the AMD build *did* run we leave opencv-mark OFF in the rustVX
    # tree (it's already in $BUILD_AMD) to avoid building cv:: twice.
    if [[ $SKIP_AMD -eq 0 ]]; then
        BUILD_OPENCV_IN_RUSTVX="OFF"
    else
        BUILD_OPENCV_IN_RUSTVX="ON"
        echo "    (also building opencv-mark here — AMD build was skipped)"
    fi
    ( cd "$BUILD_RUSTVX" && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DOPENVX_INCLUDES="$RUSTVX_INCLUDE" \
        -DOPENVX_LIB_DIR="$RUSTVX_LIB_DIR" \
        -DOPENVX_MARK_BUILD_OPENCV="$BUILD_OPENCV_IN_RUSTVX" > /dev/null )
    if [[ "$BUILD_OPENCV_IN_RUSTVX" == "ON" ]]; then
        cmake --build "$BUILD_RUSTVX" --target openvx-mark opencv-mark -j 4
    else
        cmake --build "$BUILD_RUSTVX" --target openvx-mark -j 4
    fi
fi

# Resolve the directory the opencv-mark binary actually lives in for
# the run step below. When the AMD build was skipped, opencv-mark is in
# the rustVX build tree instead. Compute the path once here so the
# downstream run step doesn't have to branch.
if [[ $SKIP_AMD -eq 0 ]]; then
    OPENCV_MARK_BUILD="$BUILD_AMD"
else
    OPENCV_MARK_BUILD="$BUILD_RUSTVX"
fi

# ----------------------------------------------------------------------------
# Run benchmarks
# ----------------------------------------------------------------------------
COMMON_FLAGS=(
    --resolution "$RESOLUTION"
    --iterations "$ITERATIONS"
    --warmup     "$WARMUP"
    --threads    "$THREADS"
    --feature-set all
    --output-dir "$OUTPUT_DIR/_tmp"
    --format     json
    --quiet
)

# ---- AMD MIVisionX ----
if [[ $SKIP_AMD -eq 0 ]]; then
    echo ""
    echo "==> [4/4a] Running openvx-mark @ AMD MIVisionX..."
    rm -rf "$OUTPUT_DIR/_tmp"
    "$BUILD_AMD/openvx-mark" "${COMMON_FLAGS[@]}" 2>&1 | tail -5
    cp "$OUTPUT_DIR/_tmp/benchmark_results.json" "$OUTPUT_DIR/amd-mivisionx.json"
fi

# ---- rustVX ----
echo ""
echo "==> [4/4b] Running openvx-mark @ rustVX..."
rm -rf "$OUTPUT_DIR/_tmp"
# Ensure the rustVX shared library is picked up at runtime even if it's
# in a non-standard directory (DYLD_LIBRARY_PATH on macOS, LD_LIBRARY_PATH
# elsewhere). The rpath embedded by CMake's default config covers most
# cases but explicit env-var override is the most portable.
case "$(uname -s)" in
    Darwin)  export DYLD_LIBRARY_PATH="$RUSTVX_LIB_DIR:${DYLD_LIBRARY_PATH:-}" ;;
    Linux|*) export LD_LIBRARY_PATH="$RUSTVX_LIB_DIR:${LD_LIBRARY_PATH:-}" ;;
esac
"$BUILD_RUSTVX/openvx-mark" "${COMMON_FLAGS[@]}" 2>&1 | tail -5
cp "$OUTPUT_DIR/_tmp/benchmark_results.json" "$OUTPUT_DIR/rustvx.json"

# ---- OpenCV ----
# Pick up opencv-mark from whichever build dir actually has it (see
# OPENCV_MARK_BUILD resolution above — defaults to $BUILD_AMD, falls
# back to $BUILD_RUSTVX when --skip-amd was passed).
echo ""
echo "==> [4/4c] Running opencv-mark..."
rm -rf "$OUTPUT_DIR/_tmp"
"$OPENCV_MARK_BUILD/opencv-mark/opencv-mark" "${COMMON_FLAGS[@]}" 2>&1 | tail -5
cp "$OUTPUT_DIR/_tmp/benchmark_results.json" "$OUTPUT_DIR/opencv.json"

rm -rf "$OUTPUT_DIR/_tmp"

# ----------------------------------------------------------------------------
# Build the combined N-way summary (one row per kernel, one column per impl)
# ----------------------------------------------------------------------------
echo ""
echo "==> Building 3-way summary..."

SUMMARY_INPUTS=()
[[ $SKIP_AMD -eq 0 ]] && SUMMARY_INPUTS+=(--label "AMD MIVisionX" "$OUTPUT_DIR/amd-mivisionx.json")
SUMMARY_INPUTS+=(--label "rustVX"        "$OUTPUT_DIR/rustvx.json")
SUMMARY_INPUTS+=(--label "OpenCV"        "$OUTPUT_DIR/opencv.json")

python3 "$SCRIPT_DIR/three_way_summary.py" \
    "${SUMMARY_INPUTS[@]}" \
    --output "$OUTPUT_DIR/comparison-three-way"

# ----------------------------------------------------------------------------
# Also emit pairwise 2-way comparisons via the existing compare_reports.py
# (which carries the richer "Conformance & Scores", per-category geomean,
# win/loss count, etc. — useful to drill into any 2 specific impls).
# ----------------------------------------------------------------------------
echo ""
echo "==> Building pairwise 2-way comparisons..."

if [[ $SKIP_AMD -eq 0 ]]; then
    python3 "$SCRIPT_DIR/compare_reports.py" \
        "$OUTPUT_DIR/amd-mivisionx.json" \
        "$OUTPUT_DIR/rustvx.json" \
        --output "$OUTPUT_DIR/comparison-amd-vs-rustvx" 2>/dev/null || true
    python3 "$SCRIPT_DIR/compare_reports.py" \
        "$OUTPUT_DIR/amd-mivisionx.json" \
        "$OUTPUT_DIR/opencv.json" \
        --output "$OUTPUT_DIR/comparison-amd-vs-opencv" 2>/dev/null || true
fi
python3 "$SCRIPT_DIR/compare_reports.py" \
    "$OUTPUT_DIR/rustvx.json" \
    "$OUTPUT_DIR/opencv.json" \
    --output "$OUTPUT_DIR/comparison-rustvx-vs-opencv" 2>/dev/null || true

echo ""
echo "========================================================================"
echo "  ✓ Done. Reports under: $OUTPUT_DIR"
echo "------------------------------------------------------------------------"
[[ $SKIP_AMD -eq 0 ]] && echo "    amd-mivisionx.json                  — raw run @ AMD MIVisionX"
echo "    rustvx.json                         — raw run @ rustVX"
echo "    opencv.json                         — raw run @ OpenCV"
echo "    comparison-three-way.{md,csv}       — combined 3-way table"
[[ $SKIP_AMD -eq 0 ]] && echo "    comparison-amd-vs-rustvx.{md,csv}   — 2-way: AMD vs rustVX"
[[ $SKIP_AMD -eq 0 ]] && echo "    comparison-amd-vs-opencv.{md,csv}   — 2-way: AMD vs OpenCV"
echo "    comparison-rustvx-vs-opencv.{md,csv} — 2-way: rustVX vs OpenCV"
echo "========================================================================"
