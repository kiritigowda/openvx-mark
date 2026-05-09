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

    // ----------------------------------------------------------------------
    // Rebuild-per-iteration path
    //
    // Builds a fresh graph for every measured iteration (and warmup) and
    // brackets the timer around the entire build + verify + process +
    // teardown cycle. Used by kernels that have implementations known to
    // hoist real work into the kernel Initializer (which runs once at
    // `vxVerifyGraph` time) and turn `vxProcessGraph` into a no-op stub —
    // see the comment on `BenchmarkCase::rebuild_graph_per_iteration` for
    // the Khronos LaplacianPyramid example.
    //
    // Tradeoff: this path includes graph-construction overhead in the
    // measured time, which is small relative to real FHD-scale kernel
    // work but would dominate ultra-cheap kernels — so it is opt-in and
    // every other case keeps the existing tight `vxProcessGraph`-only
    // measurement loop below.
    // ----------------------------------------------------------------------
    if (bc.rebuild_graph_per_iteration) {
        auto build_and_verify_once = [&](BenchmarkResult& r,
                                          ResourceTracker& tr) -> vx_graph {
            vx_graph g = vxCreateGraph(ctx);
            if (vxGetStatus((vx_reference)g) != VX_SUCCESS) {
                r.supported = false;
                r.skip_reason = "failed to create graph";
                return nullptr;
            }
            tr.trackGraph(g);
            if (!bc.graph_setup(ctx, g, res.width, res.height, gen, tr)) {
                r.supported = false;
                r.skip_reason = "graph setup failed";
                return nullptr;
            }
            if (vxVerifyGraph(g) != VX_SUCCESS) {
                r.supported = false;
                r.skip_reason = "vxVerifyGraph failed (kernel not fully supported)";
                return nullptr;
            }
            return g;
        };

        // Sanity-check the kernel is buildable on this implementation
        // before paying the per-iteration build cost. Any failure here
        // surfaces the same skip_reason as the default path so report
        // diffing remains stable.
        {
            ResourceTracker probe_tracker;
            if (build_and_verify_once(result, probe_tracker) == nullptr) {
                return result;
            }
        }

        // Output verification once (uses its own context-side resources;
        // the verify lambda is responsible for cleaning up).
        if (bc.verify_fn) {
            if (!bc.verify_fn(ctx)) {
                result.verified = false;
                result.skip_reason = "output verification failed";
            }
        }

        // Warm-up — same shape as the measured loop so the cycle (and
        // any allocator caches the impl maintains across vxReleaseGraph)
        // converges before measurement starts.
        for (int i = 0; i < config_.warmup; i++) {
            ResourceTracker warm_tracker;
            vx_graph g = build_and_verify_once(result, warm_tracker);
            if (!g) return result;
            vxProcessGraph(g);
        }

        std::vector<double> samples;
        samples.reserve(config_.iterations);
        BenchmarkTimer timer;
        vx_perf_t accum_perf = {};
        accum_perf.tmp = 0;
        bool any_vx_perf = false;

        auto measure_iter = [&](BenchmarkResult& r) -> bool {
            ResourceTracker iter_tracker;
            timer.start();
            vx_graph g = vxCreateGraph(ctx);
            if (vxGetStatus((vx_reference)g) != VX_SUCCESS) {
                timer.stop();
                r.supported = false;
                r.skip_reason = "failed to create graph during measurement";
                return false;
            }
            iter_tracker.trackGraph(g);
            if (!bc.graph_setup(ctx, g, res.width, res.height, gen, iter_tracker)) {
                timer.stop();
                r.supported = false;
                r.skip_reason = "graph setup failed during measurement";
                return false;
            }
            if (vxVerifyGraph(g) != VX_SUCCESS) {
                timer.stop();
                r.supported = false;
                r.skip_reason = "vxVerifyGraph failed during measurement";
                return false;
            }
            vx_status s = vxProcessGraph(g);
            timer.stop();
            if (s != VX_SUCCESS) {
                r.supported = false;
                r.skip_reason = "vxProcessGraph failed during measurement";
                return false;
            }
            // vx_perf_t snapshot before the graph is torn down.
            vx_perf_t perf = {};
            if (BenchmarkTimer::queryGraphPerf(g, perf)) {
                accum_perf.avg += perf.avg;
                accum_perf.min = (accum_perf.num == 0)
                    ? perf.min
                    : (perf.min < accum_perf.min ? perf.min : accum_perf.min);
                if (perf.max > accum_perf.max) accum_perf.max = perf.max;
                accum_perf.num += 1;
                any_vx_perf = true;
            }
            samples.push_back(timer.elapsed_ns());
            return true;
            // iter_tracker dtor releases graph + setup-tracked refs.
        };

        for (int i = 0; i < config_.iterations; i++) {
            if (!measure_iter(result)) return result;
        }

        result.wall_clock = BenchmarkStats::compute(samples);
        result.megapixels_per_sec = BenchmarkStats::computeThroughput(
            res.width, res.height, result.wall_clock.median_ns);

        int retries_left = config_.max_retries;
        int current_iters = config_.iterations;
        while (result.wall_clock.cv_percent > config_.stability_threshold && retries_left > 0) {
            retries_left--;
            result.retry_count++;
            current_iters *= 2;
            if (config_.verbose) {
                printf("  RETRY %d: CV=%.1f%% > %.1f%%, re-running with %d iterations (rebuild path)\n",
                       result.retry_count, result.wall_clock.cv_percent,
                       config_.stability_threshold, current_iters);
            }
            samples.clear();
            samples.reserve(current_iters);
            accum_perf = {};
            any_vx_perf = false;
            for (int i = 0; i < current_iters; i++) {
                if (!measure_iter(result)) return result;
            }
            result.wall_clock = BenchmarkStats::compute(samples);
            result.megapixels_per_sec = BenchmarkStats::computeThroughput(
                res.width, res.height, result.wall_clock.median_ns);
            result.iterations = current_iters;
        }

        if (result.wall_clock.cv_percent > config_.stability_threshold) {
            result.stability_warning = true;
        }

        if (any_vx_perf && accum_perf.num > 0) {
            result.has_vx_perf = true;
            // Average per-iteration (ns) — vx_perf_t::avg is per-iter
            // already, so a mean across iterations is the right summary
            // for the rebuild path's repeated single-shot graphs.
            result.vx_perf.mean_ns = static_cast<double>(accum_perf.avg) /
                                      static_cast<double>(accum_perf.num);
            result.vx_perf.median_ns = result.vx_perf.mean_ns;
            result.vx_perf.min_ns = static_cast<double>(accum_perf.min);
            result.vx_perf.max_ns = static_cast<double>(accum_perf.max);
            result.vx_perf.sample_count = accum_perf.num;
        }

        if (config_.verbose && result.wall_clock.cv_percent > config_.stability_threshold) {
            printf("  WARNING: CV=%.1f%% — consider more iterations for stable results\n",
                   result.wall_clock.cv_percent);
        }
        return result;
    }

    // ----------------------------------------------------------------------
    // Default path: build graph once, time only `vxProcessGraph()` per iter.
    // ----------------------------------------------------------------------
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
