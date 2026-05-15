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

    return cases;
}
