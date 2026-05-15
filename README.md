[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# openvx-mark

**openvx-mark** is a vendor-agnostic benchmark suite for [OpenVX](https://www.khronos.org/openvx/) implementations (1.0 through 1.3+). It measures the performance of individual vision kernels, multi-node pipelines, and immediate-mode operations across configurable resolutions, producing composite scores, conformance reports, and detailed analytics.

openvx-mark works with any conformant OpenVX implementation — AMD OpenVX (MIVisionX), Intel OpenVX, Khronos Sample Implementation, or any other vendor's runtime.

## Features

- **60 standard OpenVX kernels** across vision and enhanced vision feature sets
- **Graph mode and immediate mode** benchmarking
- **Multi-resolution testing** — VGA, HD, FHD, 4K, 8K, or custom
- **Composite scoring** — geometric mean of megapixels/sec (OpenVX Vision Score)
- **Conformance checking** — verifies all available kernels produce valid results
- **Stability gating** — CV% threshold with automatic retries for unstable results
- **Multi-resolution scaling analysis** — measures throughput scaling efficiency across resolutions
- **Peak vs sustained performance** — compares best-case to typical latency
- **Baseline comparison** — compare JSON reports across runs or vendors
- **Reports** — JSON, CSV, and Markdown output with glossary

## Important

It is recommended that the OpenVX implementation first passes the [Khronos OpenVX Conformance Test Suite](https://github.com/KhronosGroup/OpenVX-cts) before running openvx-mark. Benchmarking results are only meaningful when the underlying implementation is conformant — non-conformant implementations may produce incorrect outputs, which will be flagged by openvx-mark's output verification and excluded from composite scores.

## Prerequisites

- C++17 compiler
- CMake 3.10+
- An OpenVX implementation with `libopenvx` and `libvxu` libraries

## Building

### Auto-detect OpenVX (recommended)

If your OpenVX implementation is installed in a standard location (`/opt/rocm`, `/usr/local`, `/usr`), CMake will find it automatically:

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### AMD OpenVX (MIVisionX)

```bash
mkdir build && cd build
cmake -DROCM_PATH=/opt/rocm ..
cmake --build .
```

### Custom OpenVX installation

Point CMake to your OpenVX headers and libraries:

```bash
mkdir build && cd build
cmake -DOPENVX_INCLUDES=/path/to/openvx/include \
      -DOPENVX_LIB_DIR=/path/to/openvx/lib ..
cmake --build .
```

### Khronos Sample Implementation

```bash
mkdir build && cd build
cmake -DOPENVX_INCLUDES=/path/to/OpenVX-sample-impl/include \
      -DOPENVX_LIB_DIR=/path/to/OpenVX-sample-impl/build/lib ..
cmake --build .
```

## Usage

```bash
./openvx-mark [OPTIONS]
```

### Quick start

```bash
# Default run: graph mode, VGA+FHD+4K, 100 iterations
./openvx-mark

# Quick test run
./openvx-mark --resolution VGA --iterations 10 --warmup 3

# Full benchmark with all feature sets
./openvx-mark --all --iterations 200

# Include immediate-mode benchmarks
./openvx-mark --mode both --resolution FHD
```

### CLI Options

#### Benchmark Selection

| Option | Description | Default |
|:---|:---|:---|
| `--all` | Run all benchmarks (vision + enhanced_vision) | |
| `--feature-set SET[,SET,...]` | Feature sets: `vision`, `enhanced_vision`, `all` | `vision` |
| `--category CAT[,CAT,...]` | Filter by category | all |
| `--kernel NAME[,NAME,...]` | Filter by kernel name | all |
| `--mode graph\|immediate\|both` | Execution mode | `graph` |
| `--skip-pipelines` | Skip multi-node pipeline benchmarks | |

#### Resolution

| Option | Description | Default |
|:---|:---|:---|
| `--resolution RES[,RES,...]` | Preset: `VGA`, `HD`, `FHD`, `4K`, `8K` | `VGA,FHD,4K` |
| `--width W --height H` | Custom resolution | |

#### Timing

| Option | Description | Default |
|:---|:---|:---|
| `--iterations N` | Measurement iterations per benchmark | `100` |
| `--warmup N` | Warm-up iterations (not measured) | `10` |
| `--seed N` | PRNG seed for reproducible test data | `42` |
| `--stability-threshold N` | CV% threshold for stability warnings | `15` |
| `--max-retries N` | Max retries for unstable benchmarks (2x iterations each retry) | `0` |
| `--framework-chain-depths N,N,...` | Chain depths swept by `VerifyChain_Box3x3` | `1,4,16,64` |

#### Output

| Option | Description | Default |
|:---|:---|:---|
| `--output-dir DIR` | Output directory for reports | `./benchmark_results` |
| `--format json,csv,markdown` | Output formats (comma-separated) | all three |
| `--verbose` | Verbose output with per-benchmark warnings | |
| `--quiet` | Minimal output (suppress per-benchmark lines) | |
| `--compare file1.json,file2.json` | Compare two or more JSON reports | |

## Benchmarked Kernels

### Vision Feature Set (41 kernels)

| Category | Kernels |
|:---|:---|
| **Pixelwise** | And, Or, Xor, Not, AbsDiff, Add, Subtract, Multiply |
| **Filters** | Box3x3, Gaussian3x3, Median3x3, Erode3x3, Dilate3x3, Sobel3x3, CustomConvolution, NonLinearFilter |
| **Color** | ColorConvert (RGB2IYUV, RGB2NV12), ChannelExtract, ChannelCombine, ConvertDepth |
| **Geometric** | ScaleImage (Half, Double), WarpAffine, WarpPerspective, Remap |
| **Statistical** | Histogram, EqualizeHist, MeanStdDev, MinMaxLoc, IntegralImage |
| **Multi-scale** | GaussianPyramid, LaplacianPyramid, HalfScaleGaussian |
| **Feature Detection** | CannyEdgeDetector, HarrisCorners, FastCorners, OpticalFlowPyrLK |
| **Misc** | Magnitude, Phase, TableLookup, Threshold (Binary, Range), WeightedAverage |

### Enhanced Vision Feature Set (19 kernels)

| Category | Kernels |
|:---|:---|
| **Pixelwise** | Min, Max, Copy |
| **Extraction** | MatchTemplate, LBP, NonMaxSuppression, HOGCells, HOGFeatures, HoughLinesP |
| **Tensor** | TensorAdd, TensorSub, TensorMul, TensorTranspose, TensorConvertDepth, TensorMatMul, TensorTableLookup |
| **Misc** | BilateralFilter, Select, ScalarOperation |

### Multi-Node Pipelines

| Pipeline | Nodes |
|:---|:---|
| EdgeDetection | ColorConvert + ChannelExtract + Gaussian3x3 + CannyEdgeDetector |
| SobelMagnitudePhase | Sobel3x3 + Magnitude + Phase |
| MorphologyOpen | Erode3x3 + Dilate3x3 |
| MorphologyClose | Dilate3x3 + Erode3x3 |
| DualFilter | Box3x3 + Median3x3 |
| HistogramEqualize | ColorConvert + ChannelExtract + EqualizeHist |
| HarrisTracker | ColorConvert + ChannelExtract + HarrisCorners |
| ThresholdedEdge | Sobel3x3 + Magnitude + ConvertDepth + Threshold |

### Framework Benchmarks (opt-in)

Kernel benchmarks measure how fast a single OpenVX node executes; **framework benchmarks** measure what only the OpenVX *graph runtime* can do — verifying a DAG, managing virtual intermediates, fusing/aliasing buffers, scheduling work across targets. They are the metrics that distinguish an OpenVX implementation from a kernel library.

Framework benchmarks are **opt-in** — they are not in the default run and do not contribute to the `OpenVX Vision Score`. Enable them with `--feature-set framework` (only framework benchmarks) or `--feature-set everything` (kernels + framework). See [`docs/framework-mark-plan.md`](docs/framework-mark-plan.md) for the roadmap.

| Benchmark | Chain | What it measures |
|:---|:---|:---|
| `GraphDividend_Box3x3_x4` | Box3x3 × 4 | Pure framework overhead (same kernel, isolates orchestration cost) |
| `GraphDividend_MixedFilters` | Gaussian3x3 → Box3x3 → Median3x3 → Erode3x3 | Realistic 4-stage filter pipeline |
| `VerifyChain_Box3x3` | Box3x3 × N (sweeps `--framework-chain-depths`, default 1, 4, 16, 64) | Graph build / verify cost vs N nodes; first-process lazy-alloc tax |

Each `GraphDividend_*` case times the same chain three ways and emits five metrics:

| Metric | Unit | Meaning |
|:---|:---|:---|
| `sum_immediate_ms` | ms | Sum of N back-to-back `vxu*` immediate-mode calls per chain pass |
| `graph_real_ms` | ms | One verified graph; intermediates are real (host-visible) buffers |
| `graph_virtual_ms` | ms | One verified graph; intermediates are `vxCreateVirtualImage` (runtime is free to fuse / alias / tile) |
| `graph_speedup` | × | `sum_immediate_ms / graph_virtual_ms`. **>1 means the graph form beats summed immediate calls** — the headline framework dividend |
| `virtual_dividend` | × | `graph_real_ms / graph_virtual_ms`. **>1 means virtual intermediates help** (runtime did something useful with the freedom) |

`VerifyChain_Box3x3` rebuilds a chain of N Box3x3 nodes for each requested depth and reports per-N timings plus three aggregate metrics:

| Metric | Unit | Meaning |
|:---|:---|:---|
| `n{N}_create_ms` | ms | `vxCreateGraph` + N node creations at depth N |
| `n{N}_verify_ms` | ms | `vxVerifyGraph` cost at depth N |
| `n{N}_first_process_ms` | ms | First `vxProcessGraph` call (often pays a one-shot lazy-allocation / kernel-init tax) |
| `n{N}_steady_process_ms` | ms | Median `vxProcessGraph` cost after warmup |
| `verify_per_node_ms` | ms/node | Linear-regression slope of verify cost over N — the per-node verify tax |
| `verify_intercept_ms` | ms | Linear-regression intercept — fixed verify cost independent of chain length |
| `first_process_overhead_ms` | ms | `first_process_ms - steady_process_ms` at the deepest chain — the cost of the first execution beyond steady state |

Use `--framework-chain-depths 1,4,16,64,256` to sweep custom depths (defaults to `1,4,16,64`).

## Output

### Terminal Summary

```
=============================================================
  Summary: 156 total | 156 passed | 0 skipped | 0 failed
  OpenVX Vision Score: 1586.05 MP/s (156 benchmarks)
  vision Conformance: PASS (41/41)
  vision Top-5 Fastest:
    1. Not                           26835.8 MP/s (graph, FHD)
    2. Threshold_Binary              25550.0 MP/s (graph, VGA)
    3. Threshold_Binary              25037.7 MP/s (graph, FHD)
    4. Threshold_Range               21545.9 MP/s (graph, FHD)
    5. Not                           21533.7 MP/s (graph, VGA)
  vision Top-5 Slowest:
    1. LaplacianPyramid              727.501 ms (graph, 4K)
    2. NonLinearFilter               580.589 ms (graph, 4K)
    3. LaplacianPyramid              225.209 ms (graph, FHD)
    4. FastCorners                   191.288 ms (graph, 4K)
    5. HarrisTracker                 160.251 ms (graph, 4K)
=============================================================
```

### Report Files

| File | Description |
|:---|:---|
| `benchmark_results.json` | Full results with scores, conformance, scaling analysis, per-result timing stats |
| `benchmark_results.csv` | Tabular data for spreadsheet analysis |
| `benchmark_results.md` | Human-readable report with tables, top-10 lists, glossary |

### Composite Scores

- **OpenVX Vision Score** — Geometric mean of MP/s across all passing graph-mode vision benchmarks
- **Enhanced Vision Score** — Geometric mean when enhanced_vision benchmarks are included
- **Category Sub-Scores** — Per-category geometric mean (pixelwise, filters, color, etc.)

### Conformance Summary

Checks whether all available kernels in each feature set produced valid graph-mode results. Reports PASS/FAIL with a list of missing kernels.

## Cross-Vendor Comparison

### C++ (built-in)

Run the benchmark on two different implementations, then compare the JSON reports:

```bash
# Run on Vendor A
./openvx-mark --output-dir results_vendor_a

# Run on Vendor B (different machine/implementation)
./openvx-mark --output-dir results_vendor_b

# Compare
./openvx-mark --compare results_vendor_a/benchmark_results.json,results_vendor_b/benchmark_results.json
```

This generates a `comparison.md` with a side-by-side table showing median latency, throughput, and % change for each benchmark.

### Python

A Python comparison script is also provided for more flexibility:

```bash
python3 scripts/compare_reports.py results_vendor_a/benchmark_results.json \
                                    results_vendor_b/benchmark_results.json \
                                    --output comparison
```

## Glossary

| Metric | Description |
|:---|:---|
| **Median (ms)** | Median wall-clock execution time across all iterations (50th percentile). More stable than mean for benchmarking. |
| **Mean (ms)** | Arithmetic mean of wall-clock execution times. |
| **Min (ms)** | Fastest observed execution time (best case). |
| **Max (ms)** | Slowest observed execution time (worst case). |
| **StdDev (ms)** | Standard deviation of execution times after IQR outlier removal. |
| **P5/P95/P99 (ms)** | 5th, 95th, and 99th percentile execution times from the raw (pre-outlier-removal) samples. |
| **CV%** | Coefficient of Variation — `(stddev / mean) * 100`. Lower values indicate more stable/repeatable results. |
| **MP/s** | Megapixels per second — `(width * height) / median_time / 1e6`. Primary throughput metric. |
| **Samples** | Number of timing samples after IQR outlier removal. |
| **Outliers** | Number of samples removed by the IQR (Interquartile Range) method. |
| **Peak (ms)** | Best-case execution time (`min_ns`). Represents peak achievable performance. |
| **Sustained (ms)** | Typical execution time (`median_ns`). Represents sustained real-world performance. |
| **Sustained Ratio** | `min_ns / median_ns`. Values near 1.0 indicate consistent performance; lower values suggest variance from caching, scheduling, or thermal effects. |
| **Scaling Efficiency** | `(MP/s at high res) / (MP/s at low res)`. 1.0 = perfect scaling; values below 1.0 indicate memory or bandwidth bottlenecks at higher resolutions. |
| **Vision Score** | Geometric mean of MP/s across all passing graph-mode vision benchmarks. Single-number summary for cross-vendor comparison. |
| **Stability Warning** | Flagged when CV% exceeds the stability threshold (default: 15%). Indicates the result may not be reliable — increase iterations or reduce system load. |
| **Conformance** | Whether all available kernels in a feature set produced valid graph-mode results. PASS = all kernels benchmarked successfully. |

## Project Structure

```
openvx-mark/
├── CMakeLists.txt              # Build system
├── cmake/
│   └── FindOpenVX.cmake        # Vendor-agnostic OpenVX discovery
├── include/
│   ├── benchmark_config.h      # Configuration and defaults
│   ├── benchmark_context.h     # OpenVX context wrapper
│   ├── benchmark_report.h      # Report generation + analytics
│   ├── benchmark_runner.h      # Benchmark execution engine
│   ├── benchmark_stats.h       # Statistical computation
│   ├── benchmark_timer.h       # High-resolution timing
│   ├── kernel_registry.h       # OpenVX kernel catalog + availability probing
│   ├── resource_tracker.h      # RAII resource management
│   ├── system_info.h           # Host system information
│   └── test_data_generator.h   # Deterministic test data generation
├── scripts/
│   └── compare_reports.py      # Python cross-vendor comparison tool
└── src/
    ├── main.cpp                # CLI entry point
    ├── benchmark_context.cpp
    ├── benchmark_runner.cpp     # Graph/immediate mode execution + stability gating
    ├── benchmark_report.cpp     # JSON/CSV/Markdown generation + analytics
    ├── benchmark_stats.cpp      # Percentiles, IQR outlier removal
    ├── benchmark_timer.cpp
    ├── kernel_registry.cpp      # 60 standard kernel definitions
    ├── system_info.cpp          # Cross-platform system info collection
    ├── test_data_generator.cpp  # Random image/tensor/auxiliary object creation
    └── benchmarks/
        ├── node_pixelwise.cpp   # And, Or, Xor, Not, AbsDiff, Add, Subtract, Multiply, Min, Max, Copy
        ├── node_filters.cpp     # Box3x3, Gaussian3x3, Median3x3, Erode3x3, Dilate3x3, Sobel3x3, CustomConvolution, NonLinearFilter
        ├── node_color.cpp       # ColorConvert, ChannelExtract, ChannelCombine, ConvertDepth
        ├── node_geometric.cpp   # ScaleImage, WarpAffine, WarpPerspective, Remap
        ├── node_statistical.cpp # Histogram, EqualizeHist, MeanStdDev, MinMaxLoc, IntegralImage
        ├── node_multiscale.cpp  # GaussianPyramid, LaplacianPyramid, HalfScaleGaussian
        ├── node_feature.cpp     # CannyEdgeDetector, HarrisCorners, FastCorners, OpticalFlowPyrLK
        ├── node_extraction.cpp  # MatchTemplate, LBP, NonMaxSuppression
        ├── node_tensor.cpp      # TensorAdd, TensorSub, TensorMul, TensorTranspose, TensorConvertDepth, TensorTableLookup
        ├── node_misc.cpp        # Magnitude, Phase, TableLookup, Threshold, WeightedAverage, Select
        ├── immediate_benchmarks.cpp  # vxu* immediate-mode variants
        ├── pipeline_vision.cpp  # EdgeDetection, SobelMagnitudePhase, MorphologyOpen/Close, DualFilter
        └── pipeline_feature.cpp # HistogramEqualize, HarrisTracker, ThresholdedEdge
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
