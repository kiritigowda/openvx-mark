# Changelog

All notable changes to **openvx-mark** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows semantic versioning where the major version tracks backward compatibility of the JSON report schema.

## [Unreleased]

### Fixed

- **`LaplacianPyramid_S16` / `LaplacianReconstruct_S16` are kept, not
  removed.** An earlier commit (`e4f734a`, since reverted) deleted these
  two benchmarks under the false assumption that no implementation could
  support the S16 input path. CI evidence shows otherwise — rustVX runs
  both at ~10 ms (FHD) — so removing them lost real cross-impl signal.
  The benchmarks are restored. The cross-impl matrix we now observe in
  CI is documented inline in `src/benchmarks/node_multiscale.cpp`:
  - **rustVX** : runs the S16 path to completion (measured timings).
  - **Khronos sample** : runs the S16 path.
  - **AMD MIVisionX** : rejects at `vxVerifyGraph` with
    `VX_ERROR_INVALID_FORMAT` (-14). This is an *impl gap*, not a spec
    contradiction — the runner records it as a clean SKIP and the
    benchmark surfaces exactly the kind of cross-vendor difference this
    suite is designed to expose.
- **`verify_fn` of S16 Laplacian variants now also accepts
  `VX_ERROR_INVALID_FORMAT`** in addition to `VX_ERROR_NOT_SUPPORTED`,
  matching what AMD MIVisionX actually returns from `vxVerifyGraph`.
  The runner already handled this at the bench level (any non-
  `VX_SUCCESS` verify status → `supported=false` → SKIP); this change
  makes the standalone verify path consistent.

### Changed

- **`[VX LOG]` callback now deduplicates consecutive identical
  messages within a single benchmark.** Some drivers (notably AMD
  MIVisionX/AGO) log the same validate error on every call to
  `vxVerifyGraph` / `vxProcessGraph` — so a single skipped benchmark
  with `warmup=1 iterations=3` would produce 5 identical
  `status=-14: ERROR: agoVerifyGraph: ... ago_kernel_cmd_validate
  failed (-14)` lines, swamping the actual timings.

  The first occurrence is now always printed verbatim (full signal
  preserved); subsequent identical `(status, text)` pairs *within the
  same benchmark* are folded into a `(previous message repeated N
  more times)` line emitted at the next non-matching message or at
  the start of the next benchmark. `BenchmarkContext::resetLogDedup()`
  is called at the top of `runGraphMode` / `runImmediateMode` so each
  bench is guaranteed at least one verbatim copy of any driver log.

## [1.1.0] — OpenCV parity comparisons

### Added — Vision Conformance Feature Set completion (42/42) & per-spec input/output coverage

- **Registered the missing 42nd Vision-Conformance kernel.** The Vision
  Conformance Feature Set (vx_khr_feature_sets §3.2.2) lists 42 required
  kernels, but the registry only knew about 41 — `LaplacianReconstruct`
  (`VX_KERNEL_LAPLACIAN_RECONSTRUCT` / `vxLaplacianReconstructNode`)
  was missing. Now registered (1.1+ gated) and a benchmark added that
  exercises a full LaplacianPyramid → LaplacianReconstruct round-trip.
  `vision Conformance:` line now reports `PASS (42/42)` (was 41/41).
- **Full audit of every benchmark against OpenVX 1.3.1 §3.** Confirmed
  no benchmark uses non-conformant input/output formats, parameter
  values, or interpolation modes. Documented each per-kernel format
  contract inline with `[REQ-####]` spec citations.
- **Separate tests per spec-required input combination.** Where a
  single kernel has multiple required input/output type combinations
  (or multiple required parameter values), each is now exercised as
  its own benchmark — the conformance matcher recognises
  `Kernel_Suffix` as covering `Kernel`, so total kernel coverage
  stays 42/42 while every spec-required *feature* per kernel is now
  measured separately. New cases (19 total) cover:
  - Pixelwise: `AbsDiff_S16`, `Add_U8_U8_S16`, `Add_S16_S16_S16`,
    `Subtract_U8_U8_S16`, `Subtract_S16_S16_S16`, `Multiply_U8_U8_S16`,
    `Multiply_S16_S16_S16` (scale=1/255, NEAREST_EVEN per [REQ-0371])
  - Color/depth: `ConvertDepth_S16toU8`, `ColorConvert_RGB2YUV4`,
    `ColorConvert_IYUV2RGB`, `ChannelExtract_NV12_Y`,
    `ChannelExtract_IYUV_U`, `ChannelExtract_YUYV_Y`,
    `ChannelCombine_YUV4`
  - Filters: `NonLinearFilter_Min`, `NonLinearFilter_Max`,
    `CustomConvolution_U8_S16` (U8→S16 output path per [REQ-0147])
  - Geometric: `ScaleImage_Nearest_Half`, `ScaleImage_Area_Half`,
    `WarpAffine_Nearest`, `WarpPerspective_Nearest`, `Remap_Nearest`
  - Multiscale: `LaplacianReconstruct` (the missing 42nd kernel),
    `GaussianPyramid_ORB` (`VX_SCALE_PYRAMID_ORB` per [REQ-0189]),
    `LaplacianPyramid_S16`, `LaplacianReconstruct_S16`,
    `HalfScaleGaussian_1x1`, `HalfScaleGaussian_5x5`
    (kernel_size ∈ {1, 3, 5} per [REQ-0410])
  - Statistical: `MinMaxLoc_S16` (S16 input per [REQ-0315])
  - Misc: `TableLookup_S16` (S16 LUT path per [REQ-0422]),
    `Threshold_S16` (S16 input per [REQ-0493], 1.3-gated)
- **Graceful skip for genuinely unsupported impl paths.** S16
  Laplacian variants on AMD AGO return `VX_ERROR_NOT_SUPPORTED` from
  `vxVerifyGraph`; we now treat that status the same as `VX_SUCCESS`
  for verify purposes so a missing-feature impl bug is reported as
  a soft-skip rather than a falsified pass.

### Added — Enhanced Vision Feature Set coverage (19/19) on opencv-mark + rustVX integration

- **opencv-mark — 1:1 kernel-name parity for both feature sets.** All
  19 enhanced_vision kernels (per OpenVX 1.3.1 §7.2.2) now have an
  OpenCV counterpart, so `compare_reports.py` joins enhanced kernels
  too. Two new files (`cv_extraction.cpp`, `cv_tensor.cpp`) plus
  Min/Max/Copy/BilateralFilter/Select/ScalarOperation in existing
  files. `LBP` is a manual 3×3 inline impl (no native `cv::LBP`);
  `NonMaxSuppression` uses the canonical `cv::dilate`-as-local-max
  trick; `ScalarOperation` benchmarks a tight C++ scalar-add loop
  to match what the OpenVX kernel measures (framework dispatch cost).
- **opencv-mark — 6 previously-unmeasured openvx-mark enhanced_vision
  benchmarks added on both sides** so the cross-impl join is symmetric:
  `HOGCells`, `HOGFeatures`, `HoughLinesP`, `TensorMatMul`,
  `BilateralFilter`, `ScalarOperation`.
- **opencv-mark — CLI accepts `--feature-set enhanced_vision` and
  `--feature-set all`** (was rejecting both with `WARNING` in #18).
  New `--skip-pipelines` flag mirrors openvx-mark.
- **rustVX as a first-class third backend.** CMake `find_library` now
  also accepts `openvx_ffi` (rustVX's library name), de-duplicating
  the link list when the openvx/vxu names resolve to the same .so/
  .dylib (single-library backend, rustVX case). AMD MIVisionX and
  Khronos sample continue to auto-detect as before.
- **`include/openvx_optional_apis.h`** (new) — small dlsym shim because
  AMD MIVisionX *declares but does not export*
  `vxBilateralFilterNode`, `vxScalarOperationNode`, `vxHOGCellsNode`,
  `vxHOGFeaturesNode`, `vxHoughLinesPNode`, and
  `vxTensorMatrixMultiplyNode`. Without the shim, linking openvx-mark
  against MIVisionX after adding these benchmarks was a hard
  `ld: symbol not found` failure. `dlsym(RTLD_DEFAULT, …)` resolves
  them at first use; a null function pointer makes the benchmark
  gracefully report `"skipped (kernel not supported by impl)"`.
- **`scripts/build_rustvx.sh`** (new) — clones (or updates) rustVX,
  runs `cargo build --release` with the SIMD + parallel features
  that match rustVX upstream CI, honours `CARGO_TARGET_DIR`
  (IDE-style sandbox caches), and creates belt-and-suspenders
  `libopenvx.{so,dylib}` / `libvxu.{so,dylib}` symlinks for any tool
  that hard-codes the legacy names.
- **`scripts/three_way_summary.py`** (new) — N-way joined
  `(name, mode, resolution)` table. The existing `compare_reports.py`
  is rich (scores, win/loss, per-category geomean) but pairwise-only;
  this handles N ≥ 3 with one column-pair per impl and surfaces
  AMD-N/A rows explicitly.
- **`scripts/compare_three_way.sh`** (new) — end-to-end driver. Builds
  rustVX, configures + builds openvx-mark twice (once against AMD
  MIVisionX in `build/`, once against rustVX in `build-rustvx/`),
  runs each binary + opencv-mark with identical flags, then emits
  both the N-way summary and three pairwise drill-down reports
  (AMD-vs-rustVX, AMD-vs-OpenCV, rustVX-vs-OpenCV).

### Added — CI fairness, accuracy & timing audit

A single PR that closes the headline credibility gap surfaced when
adopting `opencv-mark` as the OpenCV baseline: "is OpenCV really
that much faster than MIVisionX, or am I unintentionally measuring
under-optimised MIVisionX code against optimised OpenCV?".

- **Optimized MIVisionX/Khronos builds in CI.** MIVisionX's stock
  `CMakeLists.txt` appends only `-msse4.2` to `CMAKE_CXX_FLAGS` —
  the AGO HAF kernels use `_mm256_*` AVX2 intrinsics directly, but
  the surrounding scalar code (dispatch, loop nests, address arith)
  is compiled SSE4.2-only because nothing widens the compile
  baseline. CI now passes `-DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG
  -march=x86-64-v3"` so the auto-vec / FMA / BMI2 paths unlock too.
  Same `CFLAGS`/`CXXFLAGS` upgrade applied to the Khronos sample's
  Python build script for cross-impl compile-baseline parity.
- **`--threads N` on both binaries** (default 1; 0 = leave impl's
  own default). `opencv-mark` calls `cv::setNumThreads(N)`; both
  binaries set `OMP_NUM_THREADS=N` for any OpenMP-using libs
  downstream. CI's Phase-2 compare now passes `--threads 1`
  explicitly so OpenCV doesn't get a silent `nproc`× boost from
  TBB default settings while the OpenVX impls run single-threaded
  per kernel.
- **`--validate-timing` self-test.** Measures the monotonic clock
  resolution and the timer's error against
  `std::this_thread::sleep_for(1ms / 10ms / 100ms)`. Runs as a gate
  at the top of every CI bench step — a borked runner clock fails
  loud before its measurements get propagated into a comparison
  report. Results land in JSON's new `timing_audit` block.
- **Cross-impl output verification.** New `--dump-outputs DIR`
  mode on both binaries dumps a curated sentinel set
  (`Box3x3`, `Gaussian3x3`, `Median3x3`, `Erode3x3`, `Dilate3x3`,
  `Sobel3x3`, `Add_U8_Saturate`, `Not_U8`, `ChannelExtract_R`)
  to raw `.bin` files plus a `manifest.json`. New
  `scripts/cross_verify_outputs.py` loads two such dumps, computes
  per-kernel max-abs-diff + mean-abs-diff + PSNR (with border-ring
  cropping for spatial filters where OpenVX `BORDER_UNDEFINED`
  leaves the outermost pixels uninitialised by spec), and gates
  on a per-kernel tolerance table. CI runs this against each
  `(OpenCV, OpenVX impl)` pair after the benchmarks; the verdict
  table appends to the existing Pairwise Comparison step summary.
- **Build & threading provenance in JSON.** Two new top-level
  blocks (`build`, `threading`) carry the benchmark-binary's
  `CMAKE_BUILD_TYPE`, compiler ID/version, `CXXFLAGS`/`CXXFLAGS_RELEASE`,
  `target_arch`, plus `requested_threads`, `opencv_threads`,
  `openmp_max_threads`, and `OMP_NUM_THREADS` env. Lets a reader
  audit at a glance whether a result was produced by an
  optimised binary at the threading policy they care about.
- **New artifact: `cross-verify-dumps`.** The raw sentinel `.bin`
  dumps from each impl uploaded so reviewers can re-run the
  verifier locally without rebuilding any binary.

## [1.0.0] — Framework Mark v1

The first major openvx-mark release that benchmarks the OpenVX **graph framework** itself, not just individual kernels. Adds a new family of *framework benchmarks* — scenarios that exercise the OpenVX graph runtime (verification, virtual-image fusion, parallel scheduling, async dispatch, per-node attribution) and that **no per-kernel benchmark can surface** — alongside the existing 60-kernel suite, which is unchanged.

### Added — Framework benchmarks (opt-in)

Run with `--feature-set framework` (only framework scenarios) or `--feature-set everything` (kernels + framework). Default `./openvx-mark` runs are unchanged.

- **`GraphDividend_Box3x3_x4`** and **`GraphDividend_MixedFilters`** — time the same N-node chain three ways (sum of immediate `vxu*` calls, graph with real intermediates, graph with virtual intermediates) and emit `sum_immediate_ms`, `graph_real_ms`, `graph_virtual_ms`, `graph_speedup`, `virtual_dividend`. The headline `graph_speedup > 1.0` is the framework dividend.
- **`VerifyChain_Box3x3`** — sweeps chain depths (configurable via `--framework-chain-depths`, default `1,4,16,64`) and reports per-N create / verify / first-process / steady-process timings, plus regression-derived `verify_per_node_ms`, `verify_intercept_ms`, and `first_process_overhead_ms`.
- **`ParallelBranches_Box3x3`** — K = 4 independent Box3x3 nodes sharing one input image, compared against K back-to-back `vxuBox3x3` immediate calls. Reports `parallelism_speedup` and `parallelism_efficiency` (where 1.0 = perfect K-way parallelism).
- **`Async_Single_Box3x3_x4`** — quantifies the per-call cost of `vxScheduleGraph` + `vxWaitGraph` vs `vxProcessGraph` on the same graph. Reports `async_overhead_ratio` (lower is better).
- **`Async_Concurrent_Box3x3_x2`** — schedules two independent graphs concurrently and reports `concurrency_speedup` — direct evidence of whether the runtime overlaps independent work.
- **Per-node `VX_NODE_PERFORMANCE` attribution** on both `GraphDividend_*` chains: emits `node_count`, `node_sum_ms`, `graph_perf_ms`, and `fusion_ratio` (`node_sum_ms / graph_perf_ms`). `≈ 1.0` = strict back-to-back, `> 1.0` = fusion / overlap detected, `≈ node_count` = the runtime reports graph time per node and isn't attributing per-node performance.

### Added — OpenVX Framework Score

A new dimensionless headline number, computed as the **equal-weight geometric mean** of every `graph_speedup`, `virtual_dividend`, `parallelism_efficiency`, and `concurrency_speedup` value produced by framework benchmarks. **`framework_score > 1.0` means the OpenVX graph framework adds aggregate value over a kernel-only baseline.** Lower-is-better metrics and the scenario-specific `fusion_ratio` are intentionally excluded so the score has a single monotonic interpretation. Only emitted when framework benchmarks are run.

Surfaced everywhere the Vision Score appears:

- Terminal summary: `OpenVX Framework Score: <x>x (geomean of <N> framework metrics)`.
- JSON `scores.framework_score` and `scores.framework_metric_count`.
- Markdown report's Composite Scores table plus a new dedicated **Framework Benchmarks** section listing every metric per scenario with its unit and direction.
- Both the C++ `--compare` path and `scripts/compare_reports.py` add a Framework Score row to **Conformance & Scores** and a new **Framework Metrics Comparison** table whose ratio column is direction-aware (so `> 1.00` always means the second implementation is better).

### Added — Plumbing

- New `FrameworkMetric` struct: `{name, value, unit, higher_is_better}`. `BenchmarkResult` gains a `framework_metrics` vector (empty for kernel results — backward-compatible).
- New `BenchmarkCase::framework_run` callback: framework benchmarks own their entire timing loop and return a populated `BenchmarkResult`. Existing 60-kernel codepath is untouched.
- New CLI flag `--framework-chain-depths` for `verify_chain` depth sweeps.
- New `--feature-set` values: `framework` (only) and `everything` (kernels + framework).
- CI workflow runs framework benchmarks for every vendor (Khronos sample-impl, MIVisionX) in a dedicated step and posts the headline metrics to the GitHub Actions job summary.

### Changed

- `BenchmarkRunner::runAll` dispatches to `framework_run` when set, with a pre-check for required kernels (so framework cases skip cleanly on implementations missing Box3x3 etc.).
- README adds a Framework Benchmarks section, glossary entries for every framework metric, and a Framework Score entry. Example terminal summary updated.
- JSON schema adds the `scores.framework_score`, `scores.framework_metric_count`, and per-result `framework_metrics` array. Existing kernel results emit an empty `framework_metrics` array. **No breaking change** for tools that consumed the previous schema.

### Notes for implementers

- `fusion_ratio` is implementation-quality-dependent: a value `≈ node_count` (e.g. `4.0` on a 4-node chain) usually means the runtime is reporting whole-graph time on every node. Useful cross-vendor signal in its own right; intentionally excluded from the Framework Score because not every conformant runtime populates `VX_NODE_PERFORMANCE` cleanly.
- `concurrency_speedup < 1.0` at small resolutions is expected and meaningful — it means async dispatch overhead exceeds concurrency gain at that work size.
- Pipelined streaming via the optional `vx_khr_pipelining` extension is intentionally out of scope for v1; only standard OpenVX APIs are used.

### v2 backlog (separate future PRs)

- `vxMapImagePatch` / `vxUnmapImagePatch` round-trip cost (host ↔ device tax).
- User-kernel dispatch tax via `vxAddUserKernel` no-op.
- Context lifecycle stress (`vxCreateContext` / `vxReleaseContext` × N).
- Determinism under load (single-graph CV% while K other graphs are scheduled).
- NN / extension-gated benchmarks.

See [`docs/framework-mark-plan.md`](docs/framework-mark-plan.md) for the full v1 design rationale.

---

## Pre-1.0

Earlier work — the kernel-only suite, output verification, MIVisionX CI, and version-independent build — landed in PRs #1–#4 on `main`. There is no formal changelog entry for those releases; see git history.
