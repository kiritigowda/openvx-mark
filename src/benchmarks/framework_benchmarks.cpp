////////////////////////////////////////////////////////////////////////////////
//
// MIT License
//
// Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////
//
// Framework benchmarks measure the OpenVX graph runtime itself rather than
// individual kernel throughput. The single most important framework metric is
// the "graph dividend": for an N-stage processing chain, how much faster is
// the graph form (one verified DAG, intermediates managed by the runtime) than
// the equivalent N back-to-back immediate-mode (vxu*) calls?
//
// This file implements graph_dividend across two chain shapes:
//   - GraphDividend_Box3x3_x4: 4 chained Box3x3 nodes. Same kernel everywhere
//     so the headline number isolates framework orchestration overhead from
//     per-kernel work mix.
//   - GraphDividend_MixedFilters: Gaussian3x3 -> Box3x3 -> Median3x3 ->
//     Erode3x3. Realistic mix of filter types; the dividend here reflects what
//     a real pipeline would see.
//
// For each chain we report:
//   sum_immediate_ms   - sum of N standalone vxu* calls per iteration
//   graph_real_ms      - one graph with real (non-virtual) intermediates
//   graph_virtual_ms   - one graph with virtual intermediates (the form most
//                        OpenVX implementations can optimize aggressively)
//   graph_speedup      - sum_immediate_ms / graph_virtual_ms (>1 = graph wins)
//   virtual_dividend   - graph_real_ms    / graph_virtual_ms (>1 = virtual
//                        intermediates help)
//
////////////////////////////////////////////////////////////////////////////////

#include "benchmark_runner.h"
#include "benchmark_stats.h"
#include "benchmark_timer.h"
#include "resource_tracker.h"
#include "test_data_generator.h"
#include <VX/vx.h>
#include <VX/vxu.h>
#include <functional>
#include <string>
#include <vector>

namespace {

// A single stage of a U8->U8 filter chain. Both forms (graph node + immediate
// function) take exactly one input image and one output image so the chain is
// trivially composable.
struct ChainStage {
    std::string kernel_name;
    std::function<vx_node(vx_graph, vx_image, vx_image)> make_node;
    std::function<vx_status(vx_context, vx_image, vx_image)> immediate;
};

// Time the chain executed as N back-to-back vxu* immediate-mode calls.
// Returns the median wall-clock time (ns) for one full chain pass.
double timeImmediateChain(vx_context ctx, uint32_t width, uint32_t height,
                          const std::vector<ChainStage>& stages,
                          int warmup, int iterations,
                          TestDataGenerator& gen) {
    ResourceTracker tracker;

    vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
    if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return 0.0;
    tracker.trackImage(input);

    // Reusable intermediates, one per stage boundary, plus the final output.
    std::vector<vx_image> buffers;
    buffers.reserve(stages.size());
    for (size_t i = 0; i < stages.size(); i++) {
        vx_image buf = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
        if (vxGetStatus((vx_reference)buf) != VX_SUCCESS) return 0.0;
        tracker.trackImage(buf);
        buffers.push_back(buf);
    }

    auto runOnce = [&]() -> vx_status {
        vx_image src = input;
        for (size_t i = 0; i < stages.size(); i++) {
            vx_status s = stages[i].immediate(ctx, src, buffers[i]);
            if (s != VX_SUCCESS) return s;
            src = buffers[i];
        }
        return VX_SUCCESS;
    };

    for (int i = 0; i < warmup; i++) runOnce();

    std::vector<double> samples;
    samples.reserve(iterations);
    BenchmarkTimer timer;
    for (int i = 0; i < iterations; i++) {
        timer.start();
        if (runOnce() != VX_SUCCESS) return 0.0;
        timer.stop();
        samples.push_back(timer.elapsed_ns());
    }

    return BenchmarkStats::compute(samples).median_ns;
}

// Time the chain executed as one verified graph. When use_virtual=true the
// intermediates are vxCreateVirtualImage so the runtime is free to fuse,
// alias, or tile them; when false they are real vxCreateImage objects with
// host-visible storage.
//
// Returns the median wall-clock time (ns) of one vxProcessGraph call.
double timeGraphChain(vx_context ctx, uint32_t width, uint32_t height,
                      const std::vector<ChainStage>& stages,
                      bool use_virtual,
                      int warmup, int iterations,
                      TestDataGenerator& gen) {
    ResourceTracker tracker;

    vx_graph graph = vxCreateGraph(ctx);
    if (vxGetStatus((vx_reference)graph) != VX_SUCCESS) return 0.0;
    tracker.trackGraph(graph);

    vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
    if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return 0.0;
    tracker.trackImage(input);

    // Final output is always real so the runtime has somewhere observable to
    // write to (otherwise dead-code elimination could in principle drop the
    // whole chain).
    vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
    if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return 0.0;
    tracker.trackImage(output);

    vx_image src = input;
    for (size_t i = 0; i < stages.size(); i++) {
        bool is_last = (i + 1 == stages.size());
        vx_image dst;
        if (is_last) {
            dst = output;
        } else if (use_virtual) {
            dst = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
        } else {
            dst = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
        }
        if (vxGetStatus((vx_reference)dst) != VX_SUCCESS) return 0.0;
        if (!is_last) tracker.trackImage(dst);

        vx_node node = stages[i].make_node(graph, src, dst);
        if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return 0.0;
        tracker.trackNode(node);

        src = dst;
    }

    if (vxVerifyGraph(graph) != VX_SUCCESS) return 0.0;

    for (int i = 0; i < warmup; i++) vxProcessGraph(graph);

    std::vector<double> samples;
    samples.reserve(iterations);
    BenchmarkTimer timer;
    for (int i = 0; i < iterations; i++) {
        timer.start();
        if (vxProcessGraph(graph) != VX_SUCCESS) return 0.0;
        timer.stop();
        samples.push_back(timer.elapsed_ns());
    }

    return BenchmarkStats::compute(samples).median_ns;
}

// Run all three timing modes for a chain and return a populated
// BenchmarkResult with framework_metrics filled. The runner backfills name /
// category / feature_set / resolution after this returns.
BenchmarkResult runGraphDividend(const std::vector<ChainStage>& stages,
                                 vx_context ctx, const Resolution& res,
                                 const BenchmarkConfig& cfg) {
    BenchmarkResult r;
    r.iterations = cfg.iterations;
    r.warmup = cfg.warmup;

    TestDataGenerator gen(cfg.seed);

    double t_imm = timeImmediateChain(ctx, res.width, res.height, stages,
                                      cfg.warmup, cfg.iterations, gen);
    double t_real = timeGraphChain(ctx, res.width, res.height, stages,
                                   /*use_virtual=*/false,
                                   cfg.warmup, cfg.iterations, gen);
    double t_virt = timeGraphChain(ctx, res.width, res.height, stages,
                                   /*use_virtual=*/true,
                                   cfg.warmup, cfg.iterations, gen);

    if (t_imm <= 0.0 || t_real <= 0.0 || t_virt <= 0.0) {
        r.supported = false;
        r.skip_reason = "chain timing failed (resource creation or graph verify error)";
        return r;
    }

    double speedup = t_imm / t_virt;
    double virt_div = t_real / t_virt;

    r.framework_metrics = {
        {"sum_immediate_ms", t_imm  / 1e6, "ms", false},
        {"graph_real_ms",    t_real / 1e6, "ms", false},
        {"graph_virtual_ms", t_virt / 1e6, "ms", false},
        {"graph_speedup",    speedup,      "x",  true},
        {"virtual_dividend", virt_div,     "x",  true},
    };

    // Surface the best graph form as the canonical wall-clock / MP/s so the
    // result aggregates sensibly in scaling and top-N views without polluting
    // the existing per-feature-set Vision Score (which only includes
    // feature_set == "vision" / "enhanced_vision").
    r.wall_clock.median_ns = t_virt;
    r.wall_clock.mean_ns = t_virt;
    r.wall_clock.min_ns = t_virt;
    r.wall_clock.max_ns = t_virt;
    r.wall_clock.sample_count = static_cast<size_t>(cfg.iterations);
    r.megapixels_per_sec = BenchmarkStats::computeThroughput(
        res.width, res.height, t_virt);

    return r;
}

// Time K back-to-back vxuBox3x3 calls writing to K independent outputs.
// This is the strict-serial baseline for the parallel_branches scenario:
// immediate-mode dispatch admits no scheduling parallelism even on
// multi-core hosts, so it is the right "no parallelism opportunity"
// reference to compare against the graph form.
double timeSerialImmediateBranches(vx_context ctx, uint32_t width, uint32_t height,
                                   int branches, int warmup, int iterations,
                                   TestDataGenerator& gen) {
    if (branches < 1) return 0.0;
    ResourceTracker tracker;

    vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
    if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return 0.0;
    tracker.trackImage(input);

    std::vector<vx_image> outputs;
    outputs.reserve(branches);
    for (int i = 0; i < branches; i++) {
        vx_image out = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
        if (vxGetStatus((vx_reference)out) != VX_SUCCESS) return 0.0;
        tracker.trackImage(out);
        outputs.push_back(out);
    }

    auto runOnce = [&]() -> vx_status {
        for (int i = 0; i < branches; i++) {
            vx_status s = vxuBox3x3(ctx, input, outputs[i]);
            if (s != VX_SUCCESS) return s;
        }
        return VX_SUCCESS;
    };

    for (int i = 0; i < warmup; i++) runOnce();

    std::vector<double> samples;
    samples.reserve(iterations);
    BenchmarkTimer timer;
    for (int i = 0; i < iterations; i++) {
        timer.start();
        if (runOnce() != VX_SUCCESS) return 0.0;
        timer.stop();
        samples.push_back(timer.elapsed_ns());
    }
    return BenchmarkStats::compute(samples).median_ns;
}

// Build one graph with K independent Box3x3 nodes, all reading the same
// input and writing to K independent real outputs. This is a textbook
// parallelism opportunity: the K nodes have no data dependency on each
// other, so a competent scheduler is free to dispatch them concurrently
// across cores / targets / queues.
double timeParallelGraphBranches(vx_context ctx, uint32_t width, uint32_t height,
                                 int branches, int warmup, int iterations,
                                 TestDataGenerator& gen) {
    if (branches < 1) return 0.0;
    ResourceTracker tracker;

    vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
    if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return 0.0;
    tracker.trackImage(input);

    vx_graph graph = vxCreateGraph(ctx);
    if (vxGetStatus((vx_reference)graph) != VX_SUCCESS) return 0.0;
    tracker.trackGraph(graph);

    for (int i = 0; i < branches; i++) {
        // Real (non-virtual) outputs ensure each branch produces an
        // observable side effect so dead-code elimination can't silently
        // drop branches.
        vx_image out = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
        if (vxGetStatus((vx_reference)out) != VX_SUCCESS) return 0.0;
        tracker.trackImage(out);

        vx_node node = vxBox3x3Node(graph, input, out);
        if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return 0.0;
        tracker.trackNode(node);
    }

    if (vxVerifyGraph(graph) != VX_SUCCESS) return 0.0;

    for (int i = 0; i < warmup; i++) vxProcessGraph(graph);

    std::vector<double> samples;
    samples.reserve(iterations);
    BenchmarkTimer timer;
    for (int i = 0; i < iterations; i++) {
        timer.start();
        if (vxProcessGraph(graph) != VX_SUCCESS) return 0.0;
        timer.stop();
        samples.push_back(timer.elapsed_ns());
    }
    return BenchmarkStats::compute(samples).median_ns;
}

// K = 4 independent branches is enough opportunity to expose any scheduling
// parallelism on every modern multi-core host, while keeping the work small
// enough that a single-core fallback still completes quickly. Future PRs
// can promote this to a CLI option if cross-machine variance demands it.
constexpr int kParallelBranchesCount = 4;

BenchmarkResult runParallelBranches(vx_context ctx, const Resolution& res,
                                    const BenchmarkConfig& cfg) {
    BenchmarkResult r;
    r.iterations = cfg.iterations;
    r.warmup = cfg.warmup;

    const int K = kParallelBranchesCount;
    TestDataGenerator gen(cfg.seed);

    double t_serial_imm = timeSerialImmediateBranches(
        ctx, res.width, res.height, K, cfg.warmup, cfg.iterations, gen);
    double t_parallel = timeParallelGraphBranches(
        ctx, res.width, res.height, K, cfg.warmup, cfg.iterations, gen);

    if (t_serial_imm <= 0.0 || t_parallel <= 0.0) {
        r.supported = false;
        r.skip_reason = "parallel branches timing failed";
        return r;
    }

    double speedup = t_serial_imm / t_parallel;
    double efficiency = speedup / static_cast<double>(K);

    r.framework_metrics = {
        {"branches",                static_cast<double>(K), "count", false},
        {"serial_immediate_ms",     t_serial_imm / 1e6,     "ms",    false},
        {"parallel_graph_ms",       t_parallel   / 1e6,     "ms",    false},
        {"parallelism_speedup",     speedup,                "x",     true},
        {"parallelism_efficiency",  efficiency,             "x",     true},
    };

    r.wall_clock.median_ns = t_parallel;
    r.wall_clock.mean_ns = t_parallel;
    r.wall_clock.min_ns = t_parallel;
    r.wall_clock.max_ns = t_parallel;
    r.wall_clock.sample_count = static_cast<size_t>(cfg.iterations);
    r.megapixels_per_sec = BenchmarkStats::computeThroughput(
        res.width, res.height, t_parallel);

    return r;
}

// Per-N timings collected by runVerifyChain; one of these is produced per
// chain depth and feeds both the per-N metrics and the slope regression.
struct VerifySample {
    int n;                      // chain depth (number of Box3x3 nodes)
    double create_ms;           // time to vxCreateGraph + add N nodes
    double verify_ms;           // time of vxVerifyGraph
    double first_process_ms;    // first vxProcessGraph (lazy alloc included)
    double steady_process_ms;   // median of subsequent vxProcessGraph calls
    bool ok;
};

// Build a graph of N Box3x3 nodes (input -> N-1 virtual intermediates ->
// output) and return per-phase timings.
VerifySample timeVerifyChain(vx_context ctx, uint32_t width, uint32_t height,
                             int n, int warmup, int iterations,
                             TestDataGenerator& gen) {
    VerifySample s{};
    s.n = n;
    if (n < 1) return s;

    ResourceTracker tracker;

    vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
    if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return s;
    tracker.trackImage(input);

    vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
    if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return s;
    tracker.trackImage(output);

    BenchmarkTimer timer;

    // Phase 1: graph construction (vxCreateGraph + N node creations).
    timer.start();
    vx_graph graph = vxCreateGraph(ctx);
    if (vxGetStatus((vx_reference)graph) != VX_SUCCESS) return s;
    tracker.trackGraph(graph);

    vx_image src = input;
    for (int i = 0; i < n; i++) {
        bool is_last = (i + 1 == n);
        vx_image dst = is_last
            ? output
            : vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
        if (vxGetStatus((vx_reference)dst) != VX_SUCCESS) return s;
        if (!is_last) tracker.trackImage(dst);

        vx_node node = vxBox3x3Node(graph, src, dst);
        if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return s;
        tracker.trackNode(node);

        src = dst;
    }
    timer.stop();
    s.create_ms = timer.elapsed_ms();

    // Phase 2: vxVerifyGraph. The headline framework metric.
    timer.start();
    if (vxVerifyGraph(graph) != VX_SUCCESS) return s;
    timer.stop();
    s.verify_ms = timer.elapsed_ms();

    // Phase 3: first vxProcessGraph. Often pays a one-shot tax (lazy
    // allocation of execution state, kernel JIT, target affinity selection)
    // beyond the steady-state cost; this number minus the steady median is
    // a useful "warm-up" signal.
    timer.start();
    if (vxProcessGraph(graph) != VX_SUCCESS) return s;
    timer.stop();
    s.first_process_ms = timer.elapsed_ms();

    // Phase 4: steady-state. Run cfg.warmup more then take median of
    // cfg.iterations samples.
    for (int i = 0; i < warmup; i++) vxProcessGraph(graph);
    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; i++) {
        timer.start();
        if (vxProcessGraph(graph) != VX_SUCCESS) return s;
        timer.stop();
        samples.push_back(timer.elapsed_ns());
    }
    s.steady_process_ms = BenchmarkStats::compute(samples).median_ns / 1e6;
    s.ok = true;
    return s;
}

// Linear regression over (n, verify_ms) samples returning slope and intercept
// of verify_ms = intercept + slope * n. Falls back to 0 / first-sample if
// fewer than 2 points are usable.
void verifyRegression(const std::vector<VerifySample>& samples,
                      double& slope_out, double& intercept_out) {
    slope_out = 0;
    intercept_out = 0;
    int count = 0;
    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    for (const auto& s : samples) {
        if (!s.ok) continue;
        double x = static_cast<double>(s.n);
        double y = s.verify_ms;
        sum_x += x; sum_y += y;
        sum_xx += x * x; sum_xy += x * y;
        count++;
    }
    if (count < 2) {
        if (count == 1) intercept_out = sum_y;
        return;
    }
    double denom = count * sum_xx - sum_x * sum_x;
    if (denom == 0) return;
    slope_out = (count * sum_xy - sum_x * sum_y) / denom;
    intercept_out = (sum_y - slope_out * sum_x) / count;
}

// Build a chain of N Box3x3 nodes for several N and report per-N create /
// verify / first-process / steady-process timings, plus regression-derived
// per-node verify slope and the lazy-alloc overhead at the deepest chain.
BenchmarkResult runVerifyChain(vx_context ctx, const Resolution& res,
                               const BenchmarkConfig& cfg) {
    BenchmarkResult r;
    r.iterations = cfg.iterations;
    r.warmup = cfg.warmup;

    const auto& depths = cfg.framework_chain_depths;
    if (depths.empty()) {
        r.supported = false;
        r.skip_reason = "no chain depths configured";
        return r;
    }

    TestDataGenerator gen(cfg.seed);
    std::vector<VerifySample> samples;
    samples.reserve(depths.size());
    for (int n : depths) {
        if (n < 1) continue;
        VerifySample s = timeVerifyChain(ctx, res.width, res.height, n,
                                         cfg.warmup, cfg.iterations, gen);
        if (!s.ok) {
            r.supported = false;
            r.skip_reason = "verify chain timing failed at depth " +
                            std::to_string(n);
            return r;
        }
        samples.push_back(s);
    }

    // Per-N metrics. Names embed the depth so downstream consumers can pick
    // them apart trivially.
    for (const auto& s : samples) {
        std::string p = "n" + std::to_string(s.n) + "_";
        r.framework_metrics.push_back({p + "create_ms",
                                       s.create_ms, "ms", false});
        r.framework_metrics.push_back({p + "verify_ms",
                                       s.verify_ms, "ms", false});
        r.framework_metrics.push_back({p + "first_process_ms",
                                       s.first_process_ms, "ms", false});
        r.framework_metrics.push_back({p + "steady_process_ms",
                                       s.steady_process_ms, "ms", false});
    }

    // Aggregates: linear-regression slope + intercept of verify cost vs N,
    // and the first-process overhead at the deepest chain.
    double slope_ms_per_node = 0, intercept_ms = 0;
    verifyRegression(samples, slope_ms_per_node, intercept_ms);

    r.framework_metrics.push_back({"verify_per_node_ms",
                                   slope_ms_per_node, "ms/node", false});
    r.framework_metrics.push_back({"verify_intercept_ms",
                                   intercept_ms, "ms", false});

    const auto& deepest = samples.back();
    double first_overhead = deepest.first_process_ms - deepest.steady_process_ms;
    if (first_overhead < 0) first_overhead = 0;
    r.framework_metrics.push_back({"first_process_overhead_ms",
                                   first_overhead, "ms", false});

    // Surface the deepest-chain steady-state time as the canonical
    // wall-clock so the row is sortable in scaling/top-N views without
    // polluting Vision Score (framework results are filtered out there).
    double primary_ns = deepest.steady_process_ms * 1e6;
    r.wall_clock.median_ns = primary_ns;
    r.wall_clock.mean_ns = primary_ns;
    r.wall_clock.min_ns = primary_ns;
    r.wall_clock.max_ns = primary_ns;
    r.wall_clock.sample_count = static_cast<size_t>(cfg.iterations);
    r.megapixels_per_sec = BenchmarkStats::computeThroughput(
        res.width, res.height, primary_ns);

    return r;
}

// Build the canonical "pure framework" chain: 4 Box3x3 nodes back-to-back.
std::vector<ChainStage> makeBox3x3Chain() {
    ChainStage box;
    box.kernel_name = "Box3x3";
    box.make_node = [](vx_graph g, vx_image in, vx_image out) {
        return vxBox3x3Node(g, in, out);
    };
    box.immediate = [](vx_context c, vx_image in, vx_image out) {
        return vxuBox3x3(c, in, out);
    };
    return {box, box, box, box};
}

// Build the "realistic" chain: Gaussian3x3 -> Box3x3 -> Median3x3 -> Erode3x3.
std::vector<ChainStage> makeMixedFilterChain() {
    ChainStage gauss;
    gauss.kernel_name = "Gaussian3x3";
    gauss.make_node = [](vx_graph g, vx_image in, vx_image out) {
        return vxGaussian3x3Node(g, in, out);
    };
    gauss.immediate = [](vx_context c, vx_image in, vx_image out) {
        return vxuGaussian3x3(c, in, out);
    };

    ChainStage box;
    box.kernel_name = "Box3x3";
    box.make_node = [](vx_graph g, vx_image in, vx_image out) {
        return vxBox3x3Node(g, in, out);
    };
    box.immediate = [](vx_context c, vx_image in, vx_image out) {
        return vxuBox3x3(c, in, out);
    };

    ChainStage median;
    median.kernel_name = "Median3x3";
    median.make_node = [](vx_graph g, vx_image in, vx_image out) {
        return vxMedian3x3Node(g, in, out);
    };
    median.immediate = [](vx_context c, vx_image in, vx_image out) {
        return vxuMedian3x3(c, in, out);
    };

    ChainStage erode;
    erode.kernel_name = "Erode3x3";
    erode.make_node = [](vx_graph g, vx_image in, vx_image out) {
        return vxErode3x3Node(g, in, out);
    };
    erode.immediate = [](vx_context c, vx_image in, vx_image out) {
        return vxuErode3x3(c, in, out);
    };

    return {gauss, box, median, erode};
}

}  // namespace

std::vector<BenchmarkCase> registerFrameworkBenchmarks() {
    std::vector<BenchmarkCase> cases;

    {
        BenchmarkCase bc;
        bc.name = "GraphDividend_Box3x3_x4";
        bc.category = "framework_dividend";
        bc.feature_set = "framework";
        bc.kernel_enum = VX_KERNEL_BOX_3x3;
        bc.required_kernels = {VX_KERNEL_BOX_3x3};
        bc.framework_run = [](vx_context ctx, const Resolution& res,
                              const BenchmarkConfig& cfg) -> BenchmarkResult {
            return runGraphDividend(makeBox3x3Chain(), ctx, res, cfg);
        };
        cases.push_back(bc);
    }

    {
        BenchmarkCase bc;
        bc.name = "GraphDividend_MixedFilters";
        bc.category = "framework_dividend";
        bc.feature_set = "framework";
        bc.kernel_enum = VX_KERNEL_GAUSSIAN_3x3;
        bc.required_kernels = {VX_KERNEL_GAUSSIAN_3x3, VX_KERNEL_BOX_3x3,
                               VX_KERNEL_MEDIAN_3x3,   VX_KERNEL_ERODE_3x3};
        bc.framework_run = [](vx_context ctx, const Resolution& res,
                              const BenchmarkConfig& cfg) -> BenchmarkResult {
            return runGraphDividend(makeMixedFilterChain(), ctx, res, cfg);
        };
        cases.push_back(bc);
    }

    {
        BenchmarkCase bc;
        bc.name = "VerifyChain_Box3x3";
        bc.category = "framework_compile";
        bc.feature_set = "framework";
        bc.kernel_enum = VX_KERNEL_BOX_3x3;
        bc.required_kernels = {VX_KERNEL_BOX_3x3};
        bc.framework_run = [](vx_context ctx, const Resolution& res,
                              const BenchmarkConfig& cfg) -> BenchmarkResult {
            return runVerifyChain(ctx, res, cfg);
        };
        cases.push_back(bc);
    }

    {
        BenchmarkCase bc;
        bc.name = "ParallelBranches_Box3x3";
        bc.category = "framework_parallel";
        bc.feature_set = "framework";
        bc.kernel_enum = VX_KERNEL_BOX_3x3;
        bc.required_kernels = {VX_KERNEL_BOX_3x3};
        bc.framework_run = [](vx_context ctx, const Resolution& res,
                              const BenchmarkConfig& cfg) -> BenchmarkResult {
            return runParallelBranches(ctx, res, cfg);
        };
        cases.push_back(bc);
    }

    return cases;
}
