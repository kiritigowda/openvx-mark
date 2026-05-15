# Framework Mark Plan

A concrete plan for adding a **framework benchmark suite** to `openvx-mark` that measures what only a graph framework can do — graph construction/verification cost, the "graph dividend" vs immediate mode, virtual-image savings, parallel/heterogeneous scheduling, async/streaming throughput, per-node attribution, user-kernel dispatch tax, and lifecycle costs.

This complements (not replaces) the existing per-kernel suite.

## 1. Goal & Non-goals

**Goal.** Add a *framework* benchmark suite that exercises the OpenVX graph runtime itself — the orchestration layer that differentiates OpenVX from a kernel library. The current suite measures `vxProcessGraph` of single-node graphs and short pipelines, which captures kernel performance but not framework value.

**Non-goals.**

- Not a replacement for the OpenVX-CTS conformance suite.
- Not a cross-*framework* benchmark (OpenVX vs OpenCV vs CUDA). That is a separate project.
- v1 will not require any vendor extensions. Anything that needs an extension (pipelining, NN, targets) is gated and clearly labeled.

## 2. Conceptual unit: "framework benchmark"

A framework benchmark is *not* "single kernel @ resolution → MP/s". It is a **scenario** that produces one or more **named scalar metrics**.

| Scenario | Reported metric(s) | Unit |
|:---|:---|:---|
| `verify_chain` (chain of N Box3x3 nodes, N ∈ {1,4,16,64}) | verify time, ms/node slope, first-process overhead | ms, ms/node |
| `graph_dividend` (4-node chain) | sum(immediate), graph latency, speedup, per-node overhead | ms, × ratio |
| `virtual_intermediate` | real-buffer time, virtual-buffer time, dividend | × ratio |
| `parallel_branches` | serial baseline, scheduled time, parallelism efficiency | × ratio |
| `async_streaming` | sync latency, async sustained throughput | ms, FPS |
| `user_kernel_overhead` | per-call dispatch cost above a no-op host function | ns/call |
| `context_lifecycle` | create+release time, build+teardown @ N | ms |

A framework benchmark therefore needs to emit a small set of typed metrics rather than a single MP/s value. That is the only real model change; everything else is additive.

## 3. v1 Scope

The four scenarios with the highest "value per line of code" and the cleanest cross-vendor story:

1. **`verify_chain`** — graph build + verify cost vs N.
2. **`graph_dividend`** — N-node chain timed three ways: sum of immediate `vxu*` calls, graph with real intermediates, graph with virtual intermediates.
3. **`parallel_branches`** — DAG with K independent branches feeding a join; default scheduler vs all-pinned-to-one-target (when targets are exposed via extension; otherwise just default + serial reference).
4. **`async_streaming`** — sync `vxProcessGraph` loop vs `vxScheduleGraph` + `vxWaitGraph` loop (and pipelined queue if the impl supports `vx_khr_pipelining`).

Each runs across the same `--resolution` set as kernels, so results scale with image size.

## 4. v2 Backlog (later, separate PRs)

- ~~Per-node `VX_NODE_PERFORMANCE` attribution (infer fusion when `sum(node_perf) > graph_perf`).~~ — landed in PR #7 as `node_count` / `node_sum_ms` / `graph_perf_ms` / `fusion_ratio` on `graph_dividend` results.
- `vxMapImagePatch` / `vxUnmapImagePatch` round-trip cost (host↔device tax).
- User-kernel dispatch tax via `vxAddUserKernel` no-op.
- Context lifecycle stress (`vxCreateContext` / `vxReleaseContext` × N; same graph built/torn down × N).
- Determinism under load (single-graph CV% while K other graphs are scheduled).
- NN / extension-gated benchmarks.

## 5. Data-model changes

`include/benchmark_stats.h` — extend `BenchmarkResult` (additive, won't break existing JSON/CSV/Markdown consumers):

```cpp
struct FrameworkMetric {
    std::string name;     // "verify_ms", "graph_speedup", "parallel_efficiency", ...
    double value;
    std::string unit;     // "ms", "ms/node", "x", "ns/call", "FPS"
    bool higher_is_better;
};

struct BenchmarkResult {
    // ... existing fields ...

    // For framework benchmarks: zero or more named scalars.
    // For kernel benchmarks: stays empty.
    std::vector<FrameworkMetric> framework_metrics;

    // For framework benchmarks the existing megapixels_per_sec/wall_clock
    // are interpreted as the "primary" timing if applicable, else 0.
};
```

Pre-existing kernel results emit an empty `framework_metrics` array. No schema break.

## 6. Runner changes

Today `BenchmarkRunner::runGraphMode` and `runImmediateMode` are hard-coded to "build graph, warmup, time `vxProcessGraph` × N." Framework benchmarks need their own execution loop because they may run *multiple graphs*, time *graph construction*, or compare two execution modes inside one case.

Cleanest extension to `include/benchmark_runner.h`:

```cpp
struct BenchmarkCase {
    // ... existing fields ...

    // Optional: for framework benchmarks. If set, this is called instead of
    // graph_setup / immediate_func, and is fully responsible for timing.
    using FrameworkRunFn = std::function<BenchmarkResult(
        vx_context ctx, const Resolution& res,
        const BenchmarkConfig& cfg, TestDataGenerator& gen)>;
    FrameworkRunFn framework_run;
};
```

In `BenchmarkRunner::runAll`, add a branch: if `bc.framework_run` is set, call it (skip graph mode / immediate mode); the returned `BenchmarkResult` already carries `framework_metrics`.

This keeps the existing 60-kernel codepath untouched.

## 7. New file: `src/benchmarks/framework_benchmarks.cpp`

One file with a `registerFrameworkBenchmarks()` function returning all v1 cases, mirroring `pipeline_vision.cpp`. Each case implements its own `framework_run` lambda. Sketch:

```cpp
// graph_dividend: 4-node chain (Gaussian -> Sobel -> Magnitude -> Threshold)
BenchmarkCase bc;
bc.name = "GraphDividend_4node";
bc.category = "framework_dividend";
bc.feature_set = "framework";
bc.required_kernels = { VX_KERNEL_GAUSSIAN_3x3, VX_KERNEL_SOBEL_3x3,
                        VX_KERNEL_MAGNITUDE,    VX_KERNEL_THRESHOLD };
bc.framework_run = [](vx_context ctx, const Resolution& res,
                      const BenchmarkConfig& cfg, TestDataGenerator& gen)
                   -> BenchmarkResult {
    BenchmarkResult r = makeFrameworkResult("GraphDividend_4node", res);

    double t_imm    = timeImmediateChain(ctx, res, cfg, gen);    // sum vxu*
    double t_graph  = timeGraphChain(ctx, res, cfg, gen, /*virtual=*/false);
    double t_virt   = timeGraphChain(ctx, res, cfg, gen, /*virtual=*/true);

    r.framework_metrics = {
        {"sum_immediate_ms",   t_imm   / 1e6, "ms",  false},
        {"graph_real_ms",      t_graph / 1e6, "ms",  false},
        {"graph_virtual_ms",   t_virt  / 1e6, "ms",  false},
        {"graph_speedup",      t_imm / t_graph, "x", true},
        {"virtual_dividend",   t_graph / t_virt, "x", true},
    };
    r.wall_clock.median_ns = t_virt;  // primary timing = best graph form
    r.megapixels_per_sec   = BenchmarkStats::computeThroughput(
                                 res.width, res.height, t_virt);
    return r;
};
```

The four v1 scenarios are each one ~50–80 line lambda. Helper functions (`timeImmediateChain`, `timeGraphChain`, `timeAsyncLoop`) live in the same file.

## 8. CLI changes (`src/main.cpp`)

Minimal additions, no breaking changes:

- New feature-set value: `--feature-set framework`.
- `--all` keeps current meaning (`vision,enhanced_vision`); introduce `--feature-set everything` (or `--include-framework`) for the kitchen-sink run so default kernel runs aren't perturbed.
- New category strings: `framework_compile`, `framework_dividend`, `framework_parallel`, `framework_async` (so users can `--category framework_dividend`).
- New flag: `--framework-chain-depths 1,4,16,64` (overrides the N-vector for `verify_chain`).
- `--list-framework` prints the framework scenarios with one-line descriptions.

`registerFrameworkBenchmarks()` is added to `runner.addCases(...)` in `main` only when the user opts in (or when `everything` is selected).

## 9. Reporting changes (`src/benchmark_report.cpp`)

- **JSON.** Each result already serializes; just add `"framework_metrics": [{name, value, unit, higher_is_better}, ...]`. Pre-existing kernel results emit an empty array. No schema break.
- **CSV.** Long-form rows — emit one row per `(benchmark, metric)` for framework benchmarks; existing kernel rows unchanged. Long-form is safer than wide-form with empty columns.
- **Markdown.** New section "Framework Benchmarks" with one table per scenario showing each metric per resolution, plus a one-line interpretation under each table (e.g., "graph_speedup > 1.0 means the graph form beats summing immediate-mode calls").
- **Composite score.** Add an **OpenVX Framework Score** = geomean of `graph_speedup`, `virtual_dividend`, `parallel_efficiency`, `async_speedup` across resolutions (only for those that produced valid values). Print alongside Vision Score in the terminal summary. Do **not** fold framework numbers into the existing Vision Score — keep the two scoreboards separate so existing comparisons stay valid.
- **Comparison (`compareReports`).** Extend the diff to print framework metrics side-by-side (vendor A vs vendor B, % delta).

## 10. Build (`CMakeLists.txt`)

One line:

```cmake
src/benchmarks/framework_benchmarks.cpp
```

added to `BENCHMARK_SOURCES`. No new dependencies.

## 11. README & glossary

- New "Framework Benchmarks" section explaining the philosophy: "kernel scores measure the kernel; framework scores measure what OpenVX adds *as* a framework."
- Glossary entries for: `graph_speedup`, `virtual_dividend`, `parallel_efficiency`, `async_speedup`, `verify_per_node_ms`, `OpenVX Framework Score`.
- Explicit caveat: framework metrics are most useful intra-vendor / cross-version; cross-vendor framework comparison still requires interpretation since vendors differ in target exposure.

## 12. Phased delivery (suggested PR slicing)

| PR | Scope | Mergeable independently? |
|:---|:---|:---|
| **#1 — Plumbing** | `FrameworkMetric` struct, `framework_run` field, runner branch, `--feature-set framework` flag, JSON `framework_metrics: []` (empty for kernels), CSV/Markdown unchanged. No new benchmarks. | Yes |
| **#2 — `graph_dividend`** | One scenario, helper functions, README section, terminal-summary entry. | Yes |
| **#3 — `verify_chain`** | Adds `--framework-chain-depths`, reports verify_ms and ms/node slope. | Yes |
| **#4 — `parallel_branches`** | Optional `VX_NODE_TARGET` pinning where supported. | Yes |
| **#5 — `async_streaming`** | `vxScheduleGraph` / `vxWaitGraph` loop; gated pipelining-extension probe. | Yes |
| **#6 — Framework Score** | Composite scoring, `comparison.md` extension. | After #2–#5 |
| **#7 — v2 backlog** | per-node perf, map/unmap cost, user-kernel dispatch, context lifecycle, determinism under load. | Each its own PR |

PRs #1 and #2 together are the minimum useful slice — they give the headline "graph dividend" number on day one.

## 13. Open questions to lock the plan

1. **Default behavior.** When the user runs bare `./openvx-mark`, should framework benchmarks be **opt-in** (recommended; current default stays exactly the same) or always on (changes default report)?
2. **Framework Score weight.** Geomean of {`graph_speedup`, `virtual_dividend`, `parallel_efficiency`, `async_speedup`} — equal weights, or weighted toward `graph_speedup` since it is the most directly cross-vendor-comparable?
3. **Chain content for `verify_chain` / `graph_dividend`.** Use `Box3x3` (cheapest, isolates framework cost) or a realistic 4-stage filter pipeline (closer to real workloads but mixes kernel cost into the number)? Recommendation: ship both — `Box3x3` for the *pure framework* signal, `Gaussian → Sobel → Magnitude → Threshold` for the *realistic* signal.
4. **Targets for `parallel_branches`.** Should we attempt to enumerate targets via `vxQueryContext` / vendor extensions, or just fix the scenario to "default scheduler vs single-threaded reference" to stay vendor-neutral in v1?
5. **Pipelining extension.** Detect via `vxQueryContext(VX_CONTEXT_EXTENSIONS)` lookup of `vx_khr_pipelining` and skip cleanly when missing — confirm this is acceptable rather than implementing a no-op fallback.
6. **Naming.** "OpenVX Framework Score" vs "Graph Score" vs "Orchestration Score" — the marketing message matters more than the code.
