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
    std::string feature_set;  // "vision", "enhanced_vision", or "framework"
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

    // Framework benchmark: scenarios that exercise the OpenVX graph runtime
    // itself (verify cost, graph dividend, scheduling, async, etc.) rather
    // than a single kernel. When set, the runner skips graph_setup /
    // immediate_func and calls this function once per resolution; the
    // function is fully responsible for its own timing and returns a
    // populated BenchmarkResult (typically with framework_metrics filled).
    // The runner pre-fills name/category/feature_set/resolution before the
    // call; the function only needs to fill measurement fields.
    using FrameworkRunFn = std::function<BenchmarkResult(
        vx_context ctx, const Resolution& res,
        const BenchmarkConfig& cfg)>;
    FrameworkRunFn framework_run;
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
