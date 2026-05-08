#include "benchmark_runner.h"
#include "benchmark_timer.h"
#include <cstdio>
#include <algorithm>

BenchmarkRunner::BenchmarkRunner(BenchmarkContext& context, const BenchmarkConfig& config,
                                 KernelRegistry& registry)
    : context_(context), config_(config), registry_(registry) {}

void BenchmarkRunner::addCase(const BenchmarkCase& bc) {
    cases_.push_back(bc);
}

void BenchmarkRunner::addCases(const std::vector<BenchmarkCase>& cases) {
    cases_.insert(cases_.end(), cases.begin(), cases.end());
}

bool BenchmarkRunner::shouldRun(const BenchmarkCase& bc) const {
    // Category filter
    if (!config_.categories.empty()) {
        const KernelInfo* info = registry_.getInfo(bc.kernel_enum);
        std::string cat = info ? info->category : bc.category;
        bool found = false;
        for (const auto& c : config_.categories) {
            if (c == cat || c == bc.category) { found = true; break; }
        }
        if (!found) return false;
    }

    // Kernel name filter
    if (!config_.kernels.empty()) {
        bool found = false;
        for (const auto& k : config_.kernels) {
            if (k == bc.name) { found = true; break; }
        }
        if (!found) return false;
    }

    // Feature set filter
    if (!config_.feature_sets.empty()) {
        bool found = false;
        for (const auto& fs : config_.feature_sets) {
            if (fs == bc.feature_set) { found = true; break; }
        }
        if (!found) return false;
    }

    // Skip pipelines
    if (config_.skip_pipelines &&
        (bc.category == "pipeline_vision" || bc.category == "pipeline_feature")) return false;

    return true;
}

std::vector<BenchmarkResult> BenchmarkRunner::runAll() {
    std::vector<BenchmarkResult> results;

    int total_cases = 0;
    for (const auto& bc : cases_) {
        if (shouldRun(bc)) total_cases++;
    }

    int case_idx = 0;
    for (const auto& bc : cases_) {
        if (!shouldRun(bc)) continue;
        case_idx++;

        for (const auto& res : config_.resolutions) {
            // Graph mode
            if (config_.mode == BenchmarkConfig::Mode::GRAPH ||
                config_.mode == BenchmarkConfig::Mode::BOTH) {
                if (bc.graph_setup) {
                    if (!config_.quiet) {
                        printf("[%d/%d] %-30s %s (graph)...", case_idx, total_cases,
                               bc.name.c_str(), res.name.c_str());
                        fflush(stdout);
                    }
                    auto result = runGraphMode(bc, res);
                    if (!config_.quiet) {
                        if (!result.supported) {
                            printf(" SKIPPED (%s)\n", result.skip_reason.c_str());
                        } else if (!result.verified) {
                            printf(" VERIFY FAILED\n");
                        } else {
                            printf(" %.2f ms (%.1f MP/s)\n",
                                   result.wall_clock.median_ns / 1e6,
                                   result.megapixels_per_sec);
                        }
                    }
                    results.push_back(std::move(result));
                }
            }

            // Immediate mode
            if (config_.mode == BenchmarkConfig::Mode::IMMEDIATE ||
                config_.mode == BenchmarkConfig::Mode::BOTH) {
                if (bc.immediate_func) {
                    if (!config_.quiet) {
                        printf("[%d/%d] %-30s %s (immediate)...", case_idx, total_cases,
                               bc.name.c_str(), res.name.c_str());
                        fflush(stdout);
                    }
                    auto result = runImmediateMode(bc, res);
                    if (!config_.quiet) {
                        if (!result.supported) {
                            printf(" SKIPPED (%s)\n", result.skip_reason.c_str());
                        } else {
                            printf(" %.2f ms (%.1f MP/s)\n",
                                   result.wall_clock.median_ns / 1e6,
                                   result.megapixels_per_sec);
                        }
                    }
                    results.push_back(std::move(result));
                }
            }
        }
    }

    return results;
}

BenchmarkResult BenchmarkRunner::runGraphMode(const BenchmarkCase& bc, const Resolution& res) {
    BenchmarkResult result;
    result.name = bc.name;
    result.category = bc.category;
    result.feature_set = bc.feature_set;
    result.mode = "graph";
    result.resolution_name = res.name;
    result.width = res.width;
    result.height = res.height;
    result.iterations = config_.iterations;
    result.warmup = config_.warmup;

    // Check kernel availability
    std::vector<vx_enum> required = bc.required_kernels;
    if (required.empty() && bc.kernel_enum != 0) {
        required.push_back(bc.kernel_enum);
    }
    if (!registry_.allAvailable(required)) {
        result.supported = false;
        result.skip_reason = "kernel not available";
        return result;
    }

    vx_context ctx = context_.handle();
    TestDataGenerator gen(config_.seed);
    ResourceTracker tracker;

    // Create graph
    vx_graph graph = vxCreateGraph(ctx);
    if (vxGetStatus((vx_reference)graph) != VX_SUCCESS) {
        result.supported = false;
        result.skip_reason = "failed to create graph";
        return result;
    }
    tracker.trackGraph(graph);

    // Setup graph
    if (!bc.graph_setup(ctx, graph, res.width, res.height, gen, tracker)) {
        result.supported = false;
        result.skip_reason = "graph setup failed";
        return result;
    }

    // Verify graph
    vx_status verify_status = vxVerifyGraph(graph);
    if (verify_status != VX_SUCCESS) {
        result.supported = false;
        result.skip_reason = "vxVerifyGraph failed (kernel not fully supported)";
        return result;
    }

    // Warm-up
    for (int i = 0; i < config_.warmup; i++) {
        vxProcessGraph(graph);
    }

    // Output verification
    if (bc.verify_fn) {
        if (!bc.verify_fn(ctx)) {
            result.verified = false;
            result.skip_reason = "output verification failed";
        }
    }

    // Measurement
    std::vector<double> samples;
    samples.reserve(config_.iterations);
    BenchmarkTimer timer;

    for (int i = 0; i < config_.iterations; i++) {
        timer.start();
        vx_status s = vxProcessGraph(graph);
        timer.stop();
        if (s != VX_SUCCESS) {
            result.supported = false;
            result.skip_reason = "vxProcessGraph failed during measurement";
            return result;
        }
        samples.push_back(timer.elapsed_ns());
    }

    // Compute wall-clock stats
    result.wall_clock = BenchmarkStats::compute(samples);
    result.megapixels_per_sec = BenchmarkStats::computeThroughput(
        res.width, res.height, result.wall_clock.median_ns);

    // Stability gating: retry with 2x iterations if CV% exceeds threshold
    int retries_left = config_.max_retries;
    int current_iters = config_.iterations;
    while (result.wall_clock.cv_percent > config_.stability_threshold && retries_left > 0) {
        retries_left--;
        result.retry_count++;
        current_iters *= 2;

        if (config_.verbose) {
            printf("  RETRY %d: CV=%.1f%% > %.1f%%, re-running with %d iterations\n",
                   result.retry_count, result.wall_clock.cv_percent,
                   config_.stability_threshold, current_iters);
        }

        // Re-run warm-up
        for (int i = 0; i < config_.warmup; i++) {
            vxProcessGraph(graph);
        }

        // Re-measure
        samples.clear();
        samples.reserve(current_iters);
        for (int i = 0; i < current_iters; i++) {
            timer.start();
            vx_status s = vxProcessGraph(graph);
            timer.stop();
            if (s != VX_SUCCESS) {
                result.supported = false;
                result.skip_reason = "vxProcessGraph failed during retry";
                return result;
            }
            samples.push_back(timer.elapsed_ns());
        }

        result.wall_clock = BenchmarkStats::compute(samples);
        result.megapixels_per_sec = BenchmarkStats::computeThroughput(
            res.width, res.height, result.wall_clock.median_ns);
        result.iterations = current_iters;
    }

    // Flag unstable results
    if (result.wall_clock.cv_percent > config_.stability_threshold) {
        result.stability_warning = true;
    }

    // Query OpenVX perf
    vx_perf_t perf = {};
    if (BenchmarkTimer::queryGraphPerf(graph, perf)) {
        result.has_vx_perf = true;
        // Convert vx_perf_t to TimingStats (limited info)
        result.vx_perf.mean_ns = static_cast<double>(perf.avg);
        result.vx_perf.min_ns = static_cast<double>(perf.min);
        result.vx_perf.max_ns = static_cast<double>(perf.max);
        result.vx_perf.sample_count = perf.num;
        // Use avg as median approximation for vx_perf
        result.vx_perf.median_ns = static_cast<double>(perf.avg);
    }

    // Warn on high CV
    if (config_.verbose && result.wall_clock.cv_percent > config_.stability_threshold) {
        printf("  WARNING: CV=%.1f%% — consider more iterations for stable results\n",
               result.wall_clock.cv_percent);
    }

    return result;
}

BenchmarkResult BenchmarkRunner::runImmediateMode(const BenchmarkCase& bc, const Resolution& res) {
    BenchmarkResult result;
    result.name = bc.name;
    result.category = bc.category;
    result.feature_set = bc.feature_set;
    result.mode = "immediate";
    result.resolution_name = res.name;
    result.width = res.width;
    result.height = res.height;
    result.iterations = config_.iterations;
    result.warmup = config_.warmup;

    // Check kernel availability
    std::vector<vx_enum> required = bc.required_kernels;
    if (required.empty() && bc.kernel_enum != 0) {
        required.push_back(bc.kernel_enum);
    }
    if (!registry_.allAvailable(required)) {
        result.supported = false;
        result.skip_reason = "kernel not available";
        return result;
    }

    vx_context ctx = context_.handle();

    // Warm-up
    for (int i = 0; i < config_.warmup; i++) {
        TestDataGenerator gen(config_.seed);
        ResourceTracker tracker;
        bc.immediate_func(ctx, res.width, res.height, gen, tracker);
    }

    // Measurement
    std::vector<double> samples;
    samples.reserve(config_.iterations);
    BenchmarkTimer timer;

    for (int i = 0; i < config_.iterations; i++) {
        TestDataGenerator gen(config_.seed);
        ResourceTracker tracker;
        timer.start();
        vx_status s = bc.immediate_func(ctx, res.width, res.height, gen, tracker);
        timer.stop();
        if (s != VX_SUCCESS) {
            result.supported = false;
            result.skip_reason = "immediate function failed";
            return result;
        }
        samples.push_back(timer.elapsed_ns());
    }

    result.wall_clock = BenchmarkStats::compute(samples);
    result.megapixels_per_sec = BenchmarkStats::computeThroughput(
        res.width, res.height, result.wall_clock.median_ns);

    // Stability gating: retry with 2x iterations if CV% exceeds threshold
    int retries_left = config_.max_retries;
    int current_iters = config_.iterations;
    while (result.wall_clock.cv_percent > config_.stability_threshold && retries_left > 0) {
        retries_left--;
        result.retry_count++;
        current_iters *= 2;

        if (config_.verbose) {
            printf("  RETRY %d: CV=%.1f%% > %.1f%%, re-running with %d iterations\n",
                   result.retry_count, result.wall_clock.cv_percent,
                   config_.stability_threshold, current_iters);
        }

        // Re-run warm-up
        for (int i = 0; i < config_.warmup; i++) {
            TestDataGenerator gen_w(config_.seed);
            ResourceTracker tracker_w;
            bc.immediate_func(ctx, res.width, res.height, gen_w, tracker_w);
        }

        // Re-measure
        samples.clear();
        samples.reserve(current_iters);
        for (int i = 0; i < current_iters; i++) {
            TestDataGenerator gen_r(config_.seed);
            ResourceTracker tracker_r;
            timer.start();
            vx_status s = bc.immediate_func(ctx, res.width, res.height, gen_r, tracker_r);
            timer.stop();
            if (s != VX_SUCCESS) {
                result.supported = false;
                result.skip_reason = "immediate function failed during retry";
                return result;
            }
            samples.push_back(timer.elapsed_ns());
        }

        result.wall_clock = BenchmarkStats::compute(samples);
        result.megapixels_per_sec = BenchmarkStats::computeThroughput(
            res.width, res.height, result.wall_clock.median_ns);
        result.iterations = current_iters;
    }

    // Flag unstable results
    if (result.wall_clock.cv_percent > config_.stability_threshold) {
        result.stability_warning = true;
    }

    // Warn on high CV
    if (config_.verbose && result.wall_clock.cv_percent > config_.stability_threshold) {
        printf("  WARNING: CV=%.1f%% — consider more iterations for stable results\n",
               result.wall_clock.cv_percent);
    }

    return result;
}
