# Changelog

All notable changes to **openvx-mark** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows semantic versioning where the major version tracks backward compatibility of the JSON report schema.

## [Unreleased]

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
