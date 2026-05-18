# Changelog

All notable changes to **openvx-mark** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows semantic versioning where the major version tracks backward compatibility of the JSON report schema.

## [Unreleased]

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
