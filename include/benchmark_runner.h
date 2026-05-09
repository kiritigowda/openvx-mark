#ifndef BENCHMARK_RUNNER_H
#define BENCHMARK_RUNNER_H

#include "benchmark_config.h"
#include "benchmark_context.h"
#include "benchmark_stats.h"
#include "kernel_registry.h"
#include "test_data_generator.h"
#include "resource_tracker.h"
#include <VX/vx.h>
#include <functional>
#include <string>
#include <vector>

// A single benchmark case definition
struct BenchmarkCase {
    std::string name;
    std::string category;
    std::string feature_set;  // "vision" or "enhanced_vision"
    vx_enum kernel_enum;
    std::vector<vx_enum> required_kernels;  // all kernels needed (for pipelines)

    // Graph mode: sets up data objects + nodes on the provided graph
    // Returns true on success
    using GraphSetupFn = std::function<bool(
        vx_context ctx, vx_graph graph, uint32_t width, uint32_t height,
        TestDataGenerator& gen, ResourceTracker& tracker)>;
    GraphSetupFn graph_setup;

    // Immediate mode: runs the vxu* function once
    // Returns VX_SUCCESS on success
    using ImmediateFn = std::function<vx_status(
        vx_context ctx, uint32_t width, uint32_t height,
        TestDataGenerator& gen, ResourceTracker& tracker)>;
    ImmediateFn immediate_func;

    // Output verification: runs kernel on small known input, checks correctness
    using VerifyFn = std::function<bool(vx_context ctx)>;
    VerifyFn verify_fn;

    // When set, the runner builds + verifies + processes + tears down a
    // fresh graph for every measured iteration (and warm-up), and the
    // timer brackets the entire build+verify+process cycle.
    //
    // Use this for kernels where an implementation might hoist real
    // work out of `vxProcessGraph()` and into the kernel Initializer
    // (which runs once during `vxVerifyGraph()`), turning subsequent
    // `vxProcessGraph()` calls into ~1 µs no-op stubs. The Khronos
    // OpenVX sample-impl does exactly this for `vxLaplacianPyramidKernel`
    // / `vxLaplacianReconstructKernel`: the per-iteration kernel
    // callback is a 4-line `if (num == ...) status = VX_SUCCESS;`
    // stub, while the entire pyramid (Gaussian build, 5x5 upsample,
    // per-level subtract) is computed once in
    // `vxLaplacianPyramidInitializer`. With the default tight loop
    // this produces a misleading ~1.4 million MP/s reading on FHD
    // because `vxProcessGraph()` was no-opped.
    //
    // Forcing a fresh graph each iteration captures the total cost of
    // the kernel's computation regardless of which lifecycle phase the
    // implementation chooses to perform it in, and gives a fair
    // comparison against spec-compliant implementations that recompute
    // on every `vxProcessGraph()` call.
    bool rebuild_graph_per_iteration = false;
};

class BenchmarkRunner {
public:
    BenchmarkRunner(BenchmarkContext& context, const BenchmarkConfig& config,
                    KernelRegistry& registry);

    // Register benchmark cases
    void addCase(const BenchmarkCase& bc);
    void addCases(const std::vector<BenchmarkCase>& cases);

    // Run all registered benchmarks
    std::vector<BenchmarkResult> runAll();

private:
    BenchmarkResult runGraphMode(const BenchmarkCase& bc, const Resolution& res);
    BenchmarkResult runImmediateMode(const BenchmarkCase& bc, const Resolution& res);

    bool shouldRun(const BenchmarkCase& bc) const;

    BenchmarkContext& context_;
    const BenchmarkConfig& config_;
    KernelRegistry& registry_;
    std::vector<BenchmarkCase> cases_;
};

// Registration functions for each benchmark category
std::vector<BenchmarkCase> registerPixelwiseBenchmarks();
std::vector<BenchmarkCase> registerFilterBenchmarks();
std::vector<BenchmarkCase> registerColorBenchmarks();
std::vector<BenchmarkCase> registerGeometricBenchmarks();
std::vector<BenchmarkCase> registerStatisticalBenchmarks();
std::vector<BenchmarkCase> registerMultiscaleBenchmarks();
std::vector<BenchmarkCase> registerFeatureBenchmarks();
std::vector<BenchmarkCase> registerExtractionBenchmarks();
std::vector<BenchmarkCase> registerTensorBenchmarks();
std::vector<BenchmarkCase> registerMiscBenchmarks();
std::vector<BenchmarkCase> registerImmediateBenchmarks();
std::vector<BenchmarkCase> registerVisionPipelines();
std::vector<BenchmarkCase> registerFeaturePipelines();

#endif // BENCHMARK_RUNNER_H
