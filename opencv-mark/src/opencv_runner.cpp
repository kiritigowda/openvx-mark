#include "opencv_runner.h"
#include "benchmark_timer.h"
#include <cstdio>

namespace opencv_mark {

OpenCVRunner::OpenCVRunner(const BenchmarkConfig& config) : config_(config) {}

void OpenCVRunner::addCase(const OpenCVBenchmarkCase& bc) { cases_.push_back(bc); }
void OpenCVRunner::addCases(const std::vector<OpenCVBenchmarkCase>& cases) {
    cases_.insert(cases_.end(), cases.begin(), cases.end());
}

bool OpenCVRunner::shouldRun(const OpenCVBenchmarkCase& bc) const {
    if (!config_.categories.empty()) {
        bool hit = false;
        for (const auto& c : config_.categories) {
            if (c == bc.category) { hit = true; break; }
        }
        if (!hit) return false;
    }
    if (!config_.kernels.empty()) {
        bool hit = false;
        for (const auto& k : config_.kernels) {
            if (k == bc.name) { hit = true; break; }
        }
        if (!hit) return false;
    }
    if (!config_.feature_sets.empty()) {
        bool hit = false;
        for (const auto& fs : config_.feature_sets) {
            if (fs == bc.feature_set) { hit = true; break; }
        }
        if (!hit) return false;
    }
    return true;
}

std::vector<BenchmarkResult> OpenCVRunner::runAll() {
    std::vector<BenchmarkResult> results;

    int total = 0;
    for (const auto& bc : cases_) if (shouldRun(bc)) ++total;

    int idx = 0;
    for (const auto& bc : cases_) {
        if (!shouldRun(bc)) continue;
        ++idx;
        for (const auto& res : config_.resolutions) {
            if (!config_.quiet) {
                printf("[%d/%d] %-30s %s (graph)...", idx, total,
                       bc.name.c_str(), res.name.c_str());
                fflush(stdout);
            }
            BenchmarkResult r = runOne(bc, res);
            if (!config_.quiet) {
                if (!r.supported) {
                    printf(" SKIPPED (%s)\n", r.skip_reason.c_str());
                } else if (!r.verified) {
                    printf(" VERIFY FAILED\n");
                } else {
                    printf(" %.2f ms (%.1f MP/s)\n",
                           r.wall_clock.median_ns / 1e6, r.megapixels_per_sec);
                }
            }
            results.push_back(std::move(r));
        }
    }
    return results;
}

BenchmarkResult OpenCVRunner::runOne(const OpenCVBenchmarkCase& bc, const Resolution& res) {
    BenchmarkResult result;
    result.name = bc.name;
    result.category = bc.category;
    result.feature_set = bc.feature_set;
    // Mode is intentionally "graph" so the (name, mode, resolution)
    // join key in scripts/compare_reports.py lines up with the typical
    // graph-mode entry from openvx-mark — see opencv_runner.h header
    // comment for the rationale.
    result.mode = "graph";
    result.resolution_name = res.name;
    result.width = res.width;
    result.height = res.height;
    result.iterations = config_.iterations;
    result.warmup = config_.warmup;

    OpenCVTestData gen(config_.seed);
    CaseBuffers bufs;

    if (!bc.setup_fn || !bc.run_fn) {
        result.supported = false;
        result.skip_reason = "benchmark missing setup or run function";
        return result;
    }

    try {
        if (!bc.setup_fn(res.width, res.height, gen, bufs)) {
            result.supported = false;
            result.skip_reason = "setup failed";
            return result;
        }
    } catch (const cv::Exception& e) {
        result.supported = false;
        result.skip_reason = std::string("cv::Exception in setup: ") + e.what();
        return result;
    }

    // Warm-up
    try {
        for (int i = 0; i < config_.warmup; ++i) bc.run_fn(bufs);
    } catch (const cv::Exception& e) {
        result.supported = false;
        result.skip_reason = std::string("cv::Exception in warmup: ") + e.what();
        return result;
    }

    if (bc.verify_fn) {
        bool ok = false;
        try { ok = bc.verify_fn(); }
        catch (const cv::Exception&) { ok = false; }
        if (!ok) {
            result.verified = false;
            result.skip_reason = "output verification failed";
        }
    }

    std::vector<double> samples;
    samples.reserve(config_.iterations);
    BenchmarkTimer timer;

    try {
        for (int i = 0; i < config_.iterations; ++i) {
            timer.start();
            bc.run_fn(bufs);
            timer.stop();
            samples.push_back(timer.elapsed_ns());
        }
    } catch (const cv::Exception& e) {
        result.supported = false;
        result.skip_reason = std::string("cv::Exception during measurement: ") + e.what();
        return result;
    }

    result.wall_clock = BenchmarkStats::compute(samples);
    result.megapixels_per_sec = BenchmarkStats::computeThroughput(
        res.width, res.height, result.wall_clock.median_ns);

    int retries_left = config_.max_retries;
    int current_iters = config_.iterations;
    while (result.wall_clock.cv_percent > config_.stability_threshold && retries_left > 0) {
        --retries_left;
        ++result.retry_count;
        current_iters *= 2;

        for (int i = 0; i < config_.warmup; ++i) bc.run_fn(bufs);

        samples.clear();
        samples.reserve(current_iters);
        for (int i = 0; i < current_iters; ++i) {
            timer.start();
            bc.run_fn(bufs);
            timer.stop();
            samples.push_back(timer.elapsed_ns());
        }
        result.wall_clock = BenchmarkStats::compute(samples);
        result.megapixels_per_sec = BenchmarkStats::computeThroughput(
            res.width, res.height, result.wall_clock.median_ns);
        result.iterations = current_iters;
    }
    if (result.wall_clock.cv_percent > config_.stability_threshold) {
        result.stability_warning = true;
    }

    return result;
}

} // namespace opencv_mark
