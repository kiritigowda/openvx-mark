[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/kiritigowda/openvx-mark/actions/workflows/ci.yml/badge.svg)](https://github.com/kiritigowda/openvx-mark/actions/workflows/ci.yml)

# openvx-mark

**openvx-mark** is a vendor-agnostic benchmark suite for [OpenVX](https://www.khronos.org/openvx/) implementations (1.0 through 1.3.1). It measures individual vision kernels, multi-node pipelines, immediate-mode operations, **and the OpenVX graph framework itself** across configurable resolutions, producing composite scores, conformance reports, per-kernel detail, and cross-implementation comparisons.

It is designed to answer three complementary questions:

1. **How fast are this implementation's kernels?** — per-kernel MP/s + composite Vision Score
2. **How much value does the implementation's graph framework add over a kernel-only baseline?** — Framework Score from graph-vs-immediate dividend, virtual-image fusion, parallel scheduling, async dispatch, etc.
3. **How does any OpenVX implementation compare against OpenCV doing the equivalent work?** — apples-to-apples per-kernel speedup tables across both feature sets

## What's in v1.1

| Area | Change |
|:---|:---|
| Vision Conformance | **42 / 42** — registered the previously-missing `LaplacianReconstruct` kernel, plus 19 separate-input variants so every spec-required input/output combination per kernel is exercised |
| Enhanced Vision | **19 / 19** benchmarks now wired on both `openvx-mark` and `opencv-mark` (was 13 / 19 with no OpenCV counterparts) |
| `opencv-mark` parity | 1:1 kernel-name match with `openvx-mark` for both feature sets — `compare_reports.py` joins enhanced kernels too |
| `rustVX` as 3rd backend | CMake natively accepts `libopenvx_ffi`; `scripts/build_rustvx.sh` + `scripts/compare_three_way.sh` wire it in as a CTS-conformant alternative to AMD MIVisionX |
| `dlsym` shim | `include/openvx_optional_apis.h` gracefully skips Enhanced Vision APIs that an impl declares-but-doesn't-export (instead of a hard `ld: symbol not found`) |
| CI policy | Per-impl feature-set policy: MIVisionX → vision-only; Khronos sample / rustVX / OpenCV → vision + enhanced_vision |

See [`CHANGELOG.md`](CHANGELOG.md) for the full release notes.

## Three companion binaries

openvx-mark ships **three binaries** that share the same JSON schema and CLI so reports drop straight into the same comparison tooling:

| Binary | Built from | Measures |
|:---|:---|:---|
| `openvx-mark` | Always | The OpenVX runtime it's linked against (MIVisionX, Khronos sample, rustVX, or any conformant impl) |
| `opencv-mark` | When OpenCV 4 is present (auto-detected) | OpenCV doing the equivalent per-kernel work — the "does adopting OpenVX pay off vs cv:: code I already have?" baseline |
| Disable opencv-mark with `-DOPENVX_MARK_BUILD_OPENCV=OFF` | | |

Both binaries link the same `bench_core` static library (stats, timer, reporter, system info) so the JSON schema, percentile math, and comparison logic are guaranteed identical across implementations.

## Prerequisites

| Component | Versions / notes |
|:---|:---|
| C++ compiler | C++17 |
| CMake | 3.10+ |
| OpenVX runtime | Any conformant impl exporting the `vx*` / `vxu*` C API. Recognised library names: `openvx + vxu` (MIVisionX, Khronos sample, …) or `openvx_ffi` (rustVX). |
| OpenCV 4 *(optional)* | Auto-detected — enables `opencv-mark`. Components used: `core`, `imgproc`, `features2d`, `video`, `objdetect`. |
| Rust toolchain *(optional)* | Only required to build rustVX from source via `scripts/build_rustvx.sh`. |

> **Important:** It is recommended that the OpenVX implementation passes the [Khronos OpenVX Conformance Test Suite](https://github.com/KhronosGroup/OpenVX-cts) before benchmarking. Non-conformant impls may produce incorrect output, which openvx-mark's output verification flags and excludes from composite scores.

## Building

### Auto-detect OpenVX (recommended)

If your OpenVX implementation is installed in a standard location (`/opt/rocm`, `/usr/local`, `/usr`), CMake will find it automatically:

```bash
mkdir build && cd build
cmake ..
cmake --build . -j
```

### Point at a specific OpenVX implementation

```bash
mkdir build && cd build
cmake -DOPENVX_INCLUDES=/path/to/openvx/include \
      -DOPENVX_LIB_DIR=/path/to/openvx/lib ..
cmake --build . -j
```

This works for **AMD MIVisionX** (`-DROCM_PATH=/opt/rocm` as a shortcut), the **Khronos sample implementation** (`-DOPENVX_LIB_DIR=/path/to/OpenVX-sample-impl/build/lib`), and **rustVX** (point at the cargo `target/release` directory — CMake recognises `libopenvx_ffi` natively, no symlinks needed).

### rustVX — full Vision + Enhanced Vision via a Rust runtime

[**rustVX**](https://github.com/kiritigowda/rustVX) is a memory-safe Rust OpenVX 1.3.1 that passes the full Khronos CTS for **both** the Vision (5923/5923) and Enhanced Vision (1235/1235) profiles. Use it when you want real measured numbers on every one of the 19 enhanced_vision kernels — AMD MIVisionX, for instance, doesn't currently ship that profile.

```bash
# One-shot helper: clone rustVX next to openvx-mark, build with SIMD +
# parallel features, print the cmake invocation you need. Honours
# CARGO_TARGET_DIR (handy in sandboxed IDE setups).
scripts/build_rustvx.sh

# Then build openvx-mark against it:
mkdir build-rustvx && cd build-rustvx
cmake -DOPENVX_INCLUDES=/abs/path/to/rustVX/include \
      -DOPENVX_LIB_DIR=/abs/path/to/rustVX/target/release ..
cmake --build . -j
```

### 3-way comparison: AMD MIVisionX vs rustVX vs OpenCV

The headline workflow — side-by-side numbers for every kernel across an OpenVX vision-only impl, a CTS-conformant full-profile impl, and the OpenCV baseline. Run in one command:

```bash
scripts/compare_three_way.sh --resolution VGA,FHD --iterations 100
```

Produces under `build/comparison-three-way/`:

- `amd-mivisionx.json`, `rustvx.json`, `opencv.json` — raw per-impl reports
- `comparison-three-way.{md,csv}` — combined N-way table (one row per `(kernel, mode, resolution)`, one column-pair per impl)
- `comparison-{amd-vs-rustvx, amd-vs-opencv, rustvx-vs-opencv}.{md,csv}` — drill-down 2-way reports with conformance, scores, per-category geomean, and win/loss counts

The 3-way table makes the Vision-vs-Enhanced-Vision asymmetry obvious: AMD entries show `N/A` on enhanced_vision rows (impl gap), while rustVX and OpenCV both have measured numbers. On vision rows all three columns are populated.

## Usage

```bash
./openvx-mark [OPTIONS]
```

### Quick start

```bash
# Default: graph mode, vision kernels at VGA + FHD + 4K, 100 iterations
./openvx-mark

# Quick smoke
./openvx-mark --resolution VGA --iterations 10 --warmup 3

# All vision + enhanced_vision, both modes
./openvx-mark --feature-set all --mode both

# Vision + framework benchmarks (the "graph dividend" suite)
./openvx-mark --feature-set everything --resolution FHD

# Compare two existing reports
./openvx-mark --compare run-a/benchmark_results.json,run-b/benchmark_results.json
```

### CLI options

#### Benchmark selection

| Option | Description | Default |
|:---|:---|:---|
| `--feature-set SET[,SET,...]` | `vision`, `enhanced_vision`, `framework`, `all` (= vision + enhanced), `everything` (= + framework) | `vision` |
| `--all` | Shorthand for `--feature-set all` | |
| `--category CAT[,CAT,...]` | Filter by category (`pixelwise`, `filters`, `color`, `geometric`, `statistical`, `multiscale`, `feature`, `extraction`, `tensor`, `misc`, `immediate`, `pipeline_vision`, `pipeline_feature`) | all |
| `--kernel NAME[,NAME,...]` | Filter by kernel name | all |
| `--mode graph\|immediate\|both` | Execution mode | `graph` |
| `--skip-pipelines` | Skip multi-node pipeline benchmarks | |

#### Resolution

| Option | Description | Default |
|:---|:---|:---|
| `--resolution RES[,RES,...]` | Presets: `VGA`, `HD`, `FHD`, `4K`, `8K` | `VGA,FHD,4K` |
| `--width W --height H` | Custom resolution | |

#### Timing & stability

| Option | Description | Default |
|:---|:---|:---|
| `--iterations N` | Measured iterations per benchmark | `100` |
| `--warmup N` | Warm-up iterations (excluded from stats) | `10` |
| `--seed N` | PRNG seed for reproducible test data | `42` |
| `--stability-threshold N` | CV% threshold for stability warning | `15` |
| `--max-retries N` | Max retries for unstable benchmarks (doubles iterations each retry) | `0` |
| `--framework-chain-depths N,N,...` | Chain depths for `VerifyChain_Box3x3` | `1,4,16,64` |

#### Threading & accuracy

| Option | Description | Default |
|:---|:---|:---|
| `--threads N` | Threads for OpenMP-using impls (0 = leave impl default) | `1` |
| `--validate-timing` | Run timer self-test (sleep 1/10/100 ms, report clock resolution + error %) and exit | |
| `--dump-outputs DIR` | Dump sentinel kernel outputs for cross-impl numerical verification (see `scripts/cross_verify_outputs.py`) | |

#### Output

| Option | Description | Default |
|:---|:---|:---|
| `--output-dir DIR` | Output directory | `./benchmark_results` |
| `--format json,csv,markdown` | Output formats (comma-separated) | all three |
| `--verbose` / `--quiet` | Output verbosity | |
| `--compare F1,F2[,...]` | Compare existing JSON reports | |

## Benchmarked kernels

### Vision Feature Set — 42 kernels, **42 / 42** Conformance

| Category | Kernels |
|:---|:---|
| **Pixelwise** | And, Or, Xor, Not, AbsDiff, Add, Subtract, Multiply |
| **Filters** | Box3x3, Gaussian3x3, Median3x3, Erode3x3, Dilate3x3, Sobel3x3, CustomConvolution, NonLinearFilter |
| **Color** | ColorConvert (RGB↔IYUV/NV12/YUV4), ChannelExtract, ChannelCombine, ConvertDepth |
| **Geometric** | ScaleImage (Half / Double / Nearest / Area), WarpAffine, WarpPerspective, Remap |
| **Statistical** | Histogram, EqualizeHist, MeanStdDev, MinMaxLoc, IntegralImage |
| **Multiscale** | GaussianPyramid (Half / ORB), LaplacianPyramid, **LaplacianReconstruct**, HalfScaleGaussian (1x1 / 3x3 / 5x5) |
| **Feature** | CannyEdgeDetector, HarrisCorners, FastCorners, OpticalFlowPyrLK |
| **Misc** | Magnitude, Phase, TableLookup, Threshold (Binary / Range), WeightedAverage |

**19 separate-input variants** are also wired so every spec-required input/output combination per kernel is measured independently (e.g. `Add_U8_U8_S16`, `Multiply_S16_S16_S16` with scale=1/255 + ROUND_NEAREST_EVEN, `ConvertDepth_S16toU8`, `WarpAffine_Nearest`, `HalfScaleGaussian_5x5`, …). The conformance matcher recognises `Kernel_Suffix` as covering `Kernel`, so total kernel coverage stays 42/42 while every spec-required *feature* per kernel has its own measurement.

### Enhanced Vision Feature Set — 19 kernels

| Category | Kernels |
|:---|:---|
| **Pixelwise** | Min, Max, Copy |
| **Extraction** | MatchTemplate, LBP, NonMaxSuppression, HOGCells, HOGFeatures, HoughLinesP |
| **Tensor** | TensorAdd, TensorSub, TensorMul, TensorTranspose, TensorConvertDepth, TensorMatMul, TensorTableLookup |
| **Misc** | BilateralFilter, Select, ScalarOperation |

`opencv-mark` has a 1:1 native counterpart for every one of these 19 kernels. On the OpenVX side, runtime support varies — see the per-impl feature-set policy in CI for which impls actually export which APIs.

### Multi-node pipelines

| Pipeline | Nodes |
|:---|:---|
| EdgeDetection | ColorConvert → ChannelExtract → Gaussian3x3 → CannyEdgeDetector |
| SobelMagnitudePhase | Sobel3x3 → (Magnitude + Phase) |
| MorphologyOpen | Erode3x3 → Dilate3x3 |
| MorphologyClose | Dilate3x3 → Erode3x3 |
| DualFilter | Box3x3 → Median3x3 |
| HistogramEqualize | ColorConvert → ChannelExtract → EqualizeHist |
| HarrisTracker | ColorConvert → ChannelExtract → HarrisCorners |
| ThresholdedEdge | Sobel3x3 → Magnitude → ConvertDepth → Threshold |

## Framework benchmarks (opt-in)

Kernel benchmarks measure how fast a single OpenVX node executes; **framework benchmarks** measure what only the OpenVX *graph runtime* can do — verifying a DAG, fusing virtual intermediates, scheduling independent branches, async dispatch overhead, per-node `VX_NODE_PERFORMANCE` attribution. They are the metrics that distinguish an OpenVX implementation from a kernel library.

Framework benchmarks are **opt-in** and do not contribute to the `OpenVX Vision Score`. Enable with `--feature-set framework` (framework only) or `--feature-set everything` (kernels + framework).

| Benchmark | What it measures |
|:---|:---|
| `GraphDividend_Box3x3_x4` | Pure framework overhead — same kernel × 4, isolates orchestration cost |
| `GraphDividend_MixedFilters` | Realistic 4-stage filter pipeline (Gaussian → Box → Median → Erode) |
| `VerifyChain_Box3x3` | Graph build / verify cost vs chain depth (default sweep: 1, 4, 16, 64); reports regression slope as `verify_per_node_ms` |
| `ParallelBranches_Box3x3` | Whether the runtime exploits scheduling parallelism on K=4 independent branches |
| `Async_Single_Box3x3_x4` | `vxScheduleGraph + vxWaitGraph` vs `vxProcessGraph` — cost of the async dispatch API |
| `Async_Concurrent_Box3x3_x2` | Whether the runtime overlaps independent graphs scheduled concurrently |

The headline output is the **OpenVX Framework Score** — equal-weight geometric mean of `graph_speedup`, `virtual_dividend`, `parallelism_efficiency`, and `concurrency_speedup`. **> 1.0** means the OpenVX graph framework adds aggregate value over a kernel-only baseline.

See [`docs/framework-mark-plan.md`](docs/framework-mark-plan.md) for the full design rationale, per-metric definitions, and the v2 backlog (vxMapImagePatch round-trip cost, user-kernel dispatch tax, lifecycle stress, …).

## Cross-vendor comparison

### Built-in (`--compare`)

Run the benchmark on two implementations, then diff the JSON reports:

```bash
./openvx-mark --output-dir results_vendor_a
./openvx-mark --output-dir results_vendor_b   # different machine / impl
./openvx-mark --compare results_vendor_a/benchmark_results.json,results_vendor_b/benchmark_results.json
```

Generates `comparison.md` with side-by-side median latency, throughput, % change, conformance, and (when present) a direction-aware Framework Metrics Comparison.

### Python (2-way drill-down)

```bash
python3 scripts/compare_reports.py run-a.json run-b.json --output cmp
```

Same output shape as `--compare`, plus per-category geomean, win/loss counts, and a benchmarks-only-in-one-report section.

### Python (N-way summary)

For 3+ implementations, use the N-way summary script:

```bash
python3 scripts/three_way_summary.py \
    --label "AMD MIVisionX" amd.json \
    --label "rustVX"        rustvx.json \
    --label "OpenCV"        opencv.json \
    --output cmp_3way
```

One row per `(kernel, mode, resolution)`, one column-pair per impl (`<impl> ms` + `<impl> MP/s`). Rows where an impl produced no result show `N/A` — which is itself useful signal for incomplete impl coverage.

## Output

### Terminal summary

```
=============================================================
  Summary: 102 total | 98 passed | 4 skipped | 0 failed
  OpenVX Vision Score:   1890.92 MP/s (79 benchmarks)
  Enhanced Vision Score:  421.34 MP/s (15 benchmarks)
  OpenVX Framework Score: 4.872x (geomean of 18 framework metrics)
  vision Conformance: PASS (42/42)
  enhanced_vision Conformance: PASS (15/19 — missing: BilateralFilter, HOGCells, HOGFeatures, TensorMatMul)
  vision Top-5 Fastest:
    1. Not                           26835.8 MP/s (graph, FHD)
    2. Threshold_Binary              25550.0 MP/s (graph, VGA)
    ...
  vision Top-5 Slowest:
    1. LaplacianPyramid              727.501 ms (graph, 4K)
    2. NonLinearFilter               580.589 ms (graph, 4K)
    ...
=============================================================
```

### Report files

| File | Description |
|:---|:---|
| `benchmark_results.json` | Full machine-readable results: scores, conformance, scaling analysis, per-result timing stats, build/threading provenance |
| `benchmark_results.csv` | Tabular data for spreadsheet analysis |
| `benchmark_results.md` | Human-readable report with top-N lists, per-category breakdown, and a glossary |

### Composite scores

- **OpenVX Vision Score** — geomean of MP/s across all passing graph-mode vision benchmarks
- **Enhanced Vision Score** — geomean when enhanced_vision benchmarks are included
- **Category Sub-Scores** — per-category geomean (pixelwise, filters, color, …)
- **OpenVX Framework Score** — geomean of `graph_speedup`, `virtual_dividend`, `parallelism_efficiency`, `concurrency_speedup`. **> 1.0** = the OpenVX graph framework adds aggregate value over a kernel-only baseline. Only emitted when framework benchmarks are run.

## Glossary

| Term | Description |
|:---|:---|
| **Median (ms)** | Median wall-clock time across iterations (50th percentile). Primary latency metric. |
| **CV%** | Coefficient of Variation = `(stddev / mean) × 100`. Lower = more stable. Default warning threshold: 15%. |
| **MP/s** | Megapixels per second = `(width × height) / median_time_s / 1e6`. Primary throughput metric. |
| **Peak / Sustained** | `min_ns` (best-case) vs `median_ns` (typical). `Sustained Ratio = min / median`; values near 1.0 indicate consistent performance. |
| **Scaling Efficiency** | `(MP/s at high res) / (MP/s at low res)`. 1.0 = perfect scaling; < 1.0 indicates memory/bandwidth bottleneck. |
| **Vision Score** | Geomean of MP/s across passing graph-mode vision benchmarks. |
| **Framework Score** | Equal-weight geomean of `graph_speedup`, `virtual_dividend`, `parallelism_efficiency`, `concurrency_speedup`. > 1.0 = the OpenVX graph framework adds value. |
| **Conformance** | Whether every kernel registered in a feature set produced a valid graph-mode result. PASS = all benchmarked successfully. |
| **Stability Warning** | CV% exceeded the threshold — increase iterations or reduce system load. |

## Project structure

```
openvx-mark/
├── CMakeLists.txt                  # Build system (recognises libopenvx/libvxu and libopenvx_ffi)
├── cmake/FindOpenVX.cmake          # Vendor-agnostic OpenVX discovery
├── docs/
│   ├── framework-mark-plan.md      # Framework benchmark v1 rationale + v2 backlog
│   └── features-to-add.md          # Verification + coverage TODOs
├── include/
│   ├── bench_runtime.h             # Shared runtime hooks (threading policy, timer self-test)
│   ├── benchmark_catalog.h         # POD snapshot of the kernel catalog (impl-agnostic)
│   ├── benchmark_{config,context,report,runner,stats,timer}.h
│   ├── kernel_registry.h           # OpenVX kernel enumeration + availability probing
│   ├── openvx_optional_apis.h      # dlsym shim for declared-but-not-exported Enhanced Vision APIs
│   ├── openvx_perf_query.h         # vx_perf_t helpers for framework benchmarks
│   ├── openvx_version.h            # OPENVX_HAS_1_{1,2,3} version gates
│   ├── resource_tracker.h          # RAII resource management
│   ├── system_info.h               # Cross-platform CPU/OS/RAM detection
│   ├── test_data_generator.h       # Deterministic seed-driven test data
│   └── verify_utils.h              # Per-kernel output verification helpers
├── src/
│   ├── main.cpp                    # CLI entry point
│   ├── bench_runtime.cpp           # In bench_core (shared with opencv-mark)
│   ├── benchmark_{stats,timer,report}.cpp   # In bench_core
│   ├── system_info.cpp             # In bench_core
│   ├── benchmark_{context,runner}.cpp        # openvx-mark only
│   ├── kernel_registry.cpp         # 61 standard kernel definitions (42 vision + 19 enhanced)
│   ├── openvx_perf_query.cpp       # Per-node performance attribution helpers
│   ├── openvx_output_dumper.cpp    # --dump-outputs sentinel suite writer
│   ├── test_data_generator.cpp
│   ├── verify_utils.cpp
│   └── benchmarks/
│       ├── node_pixelwise.cpp      # And/Or/Xor/Not/AbsDiff/Add/Sub/Mul + S16 variants + Min/Max/Copy
│       ├── node_filters.cpp        # Box/Gaussian/Median/Erode/Dilate/Sobel/CustomConvolution/NonLinearFilter + variants
│       ├── node_color.cpp          # ColorConvert/ChannelExtract/ChannelCombine/ConvertDepth + format variants
│       ├── node_geometric.cpp      # ScaleImage/WarpAffine/WarpPerspective/Remap + Nearest/Area variants
│       ├── node_statistical.cpp    # Histogram/EqualizeHist/MeanStdDev/MinMaxLoc/IntegralImage
│       ├── node_multiscale.cpp     # GaussianPyramid (Half/ORB), LaplacianPyramid (U8/S16), LaplacianReconstruct, HalfScaleGaussian (1/3/5)
│       ├── node_feature.cpp        # Canny/Harris/Fast/OpticalFlowPyrLK
│       ├── node_extraction.cpp     # MatchTemplate/LBP/NonMaxSuppression/HOGCells/HOGFeatures/HoughLinesP
│       ├── node_tensor.cpp         # TensorAdd/Sub/Mul/Transpose/ConvertDepth/MatMul/TableLookup
│       ├── node_misc.cpp           # Magnitude/Phase/TableLookup/Threshold/WeightedAverage/BilateralFilter/Select/ScalarOperation
│       ├── immediate_benchmarks.cpp        # vxu* immediate-mode variants
│       ├── pipeline_{vision,feature}.cpp   # Multi-node pipelines
│       └── framework_benchmarks.cpp        # Graph dividend / verify chain / parallel branches / async
├── opencv-mark/                    # OpenCV companion binary (built when OpenCV 4 is present)
│   ├── CMakeLists.txt
│   ├── include/                    # opencv_{context,runner,test_data,verify}.h
│   └── src/
│       ├── main.cpp                # Mirrors openvx-mark CLI
│       ├── opencv_{context,runner,test_data,verify}.cpp
│       ├── cv_output_dumper.cpp
│       └── benchmarks/
│           ├── cv_{pixelwise,filters,color,geometric,statistical,multiscale,misc,feature}.cpp
│           ├── cv_extraction.cpp                # MatchTemplate/LBP/HOG*/HoughLinesP/NonMaxSuppression
│           ├── cv_tensor.cpp                    # Tensor* ops via cv::add/multiply/gemm/LUT/transpose
│           └── cv_pipeline_{vision,feature}.cpp # Multi-node pipelines mirroring openvx-mark
└── scripts/
    ├── build_rustvx.sh             # Clone + build rustVX, print openvx-mark cmake invocation
    ├── compare_three_way.sh        # End-to-end: run AMD + rustVX + OpenCV, emit N-way + pairwise reports
    ├── compare_reports.py          # 2-way pairwise comparison (rich: scores, win/loss, per-category)
    ├── three_way_summary.py        # N-way joined table (one column-pair per impl)
    ├── ci_pairwise_summary.py      # CI GitHub Step Summary renderer
    └── cross_verify_outputs.py     # Sentinel-suite numerical verification (PSNR + max-abs-diff)
```

## Continuous integration

CI runs on every push and PR. The workflow has two phases:

**Phase 1 — four parallel build + smoke jobs:**
1. **MIVisionX** (AMD OpenVX, CPU backend) — built from source with `-march=x86-64-v3`. Smoke: `vision,framework`.
2. **Khronos sample** — built from source, same compile baseline. Smoke: `vision,enhanced_vision,framework`.
3. **rustVX** — built from source via cargo, AVX2 + parallel features. Smoke: `vision,enhanced_vision,framework`.
4. **opencv-mark** — apt-installed `libopencv-dev`. Smoke: `vision,enhanced_vision`.

**Phase 2 — pairwise comparison.** Downloads all three OpenVX impl artifacts, apt-installs OpenCV, builds openvx-mark × 3 (one per OpenVX impl) plus opencv-mark, runs the full bench at FHD × 20 iterations × `--threads 1`, and emits six pairwise reports posted to the GitHub Actions job summary:

- **OpenVX-vs-OpenCV** (the "does adopting OpenVX pay off?" trio): MIVisionX vs OpenCV, Khronos sample vs OpenCV, rustVX vs OpenCV
- **OpenVX-vs-OpenVX** (cross-implementation): MIVisionX vs Khronos sample, MIVisionX vs rustVX, rustVX vs Khronos sample

Cross-impl numerical verification (`scripts/cross_verify_outputs.py`) runs a sentinel kernel set on every impl, computes max-abs-diff + PSNR + exact-% per kernel, and gates on a per-kernel tolerance table — so the timing comparison is gated on the impls actually agreeing on what the kernel should output.

## License

MIT — see [LICENSE](LICENSE).

The OpenVX logo is a trademark of The Khronos Group Inc.
