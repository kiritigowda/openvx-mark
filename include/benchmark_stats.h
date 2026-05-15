#ifndef BENCHMARK_STATS_H
#define BENCHMARK_STATS_H

#include <cstdint>
#include <string>
#include <vector>

struct TimingStats {
    double mean_ns = 0;
    double median_ns = 0;
    double min_ns = 0;
    double max_ns = 0;
    double stddev_ns = 0;
    double p5_ns = 0;
    double p95_ns = 0;
    double p99_ns = 0;
    double cv_percent = 0;  // coefficient of variation
    size_t sample_count = 0;
    size_t outliers_removed = 0;
};

// A named scalar metric emitted by a framework benchmark.
// Framework benchmarks (feature_set == "framework") may emit one or more
// of these in addition to (or instead of) megapixels_per_sec.
struct FrameworkMetric {
    std::string name;             // e.g. "graph_speedup", "verify_ms"
    double value = 0;
    std::string unit;             // e.g. "ms", "ms/node", "x", "ns/call", "FPS"
    bool higher_is_better = true; // direction for comparison reports
};

struct BenchmarkResult {
    std::string name;
    std::string category;
    std::string feature_set;  // "vision", "enhanced_vision", or "framework"
    std::string mode;  // "graph", "immediate", or "framework"
    std::string resolution_name;
    uint32_t width = 0;
    uint32_t height = 0;
    bool supported = true;
    bool verified = true;
    std::string skip_reason;

    TimingStats wall_clock;
    // OpenVX vx_perf_t timing (may be empty if not populated)
    bool has_vx_perf = false;
    TimingStats vx_perf;

    double megapixels_per_sec = 0;
    int iterations = 0;
    int warmup = 0;

    // Stability gating
    bool stability_warning = false;
    int retry_count = 0;

    // Framework benchmark metrics (empty for kernel benchmarks)
    std::vector<FrameworkMetric> framework_metrics;
};

class BenchmarkStats {
public:
    // Compute statistics from raw timing samples (in nanoseconds)
    static TimingStats compute(const std::vector<double>& samples_ns);

    // Compute throughput in megapixels/sec
    static double computeThroughput(uint32_t width, uint32_t height, double median_ns);

private:
    static double percentile(const std::vector<double>& sorted, double p);
    static std::vector<double> removeOutliers(const std::vector<double>& sorted);
};

#endif // BENCHMARK_STATS_H
