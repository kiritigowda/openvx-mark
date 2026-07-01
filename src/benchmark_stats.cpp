#include "benchmark_stats.h"
#include <algorithm>
#include <cmath>
#include <numeric>

// Internal helpers — declared here so compute() can use them above their
// definitions while keeping the file readable top-to-bottom.
static double mean(const std::vector<double>& v);
static double stddev(const std::vector<double>& v, double mean_val);

TimingStats BenchmarkStats::compute(const std::vector<double>& samples_ns,
                                    bool remove_outliers) {
    TimingStats stats;
    if (samples_ns.empty()) return stats;

    // Sort for percentile computation
    std::vector<double> sorted = samples_ns;
    std::sort(sorted.begin(), sorted.end());

    // Raw statistics over the full, unfiltered sample set.
    stats.raw_sample_count = sorted.size();
    stats.raw_mean_ns = mean(sorted);
    stats.raw_median_ns = percentile(sorted, 50.0);
    stats.raw_stddev_ns = stddev(sorted, stats.raw_mean_ns);
    stats.raw_cv_percent = (stats.raw_mean_ns > 0)
        ? (stats.raw_stddev_ns / stats.raw_mean_ns * 100.0) : 0.0;

    // Remove outliers via IQR when requested.
    std::vector<double> cleaned = remove_outliers ? removeOutliers(sorted) : sorted;
    if (cleaned.empty()) cleaned = sorted;  // fallback if all removed

    stats.sample_count = cleaned.size();
    stats.outliers_removed = samples_ns.size() - cleaned.size();

    // Min, max
    stats.min_ns = cleaned.front();
    stats.max_ns = cleaned.back();

    // Mean
    stats.mean_ns = mean(cleaned);

    // Median
    stats.median_ns = percentile(cleaned, 50.0);

    // Stddev
    stats.stddev_ns = stddev(cleaned, stats.mean_ns);

    // CV%
    stats.cv_percent = (stats.mean_ns > 0) ? (stats.stddev_ns / stats.mean_ns * 100.0) : 0.0;

    // Percentiles (from original sorted, not cleaned)
    stats.p5_ns = percentile(sorted, 5.0);
    stats.p95_ns = percentile(sorted, 95.0);
    stats.p99_ns = percentile(sorted, 99.0);

    return stats;
}

static double mean(const std::vector<double>& v) {
    if (v.empty()) return 0;
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / static_cast<double>(v.size());
}

static double stddev(const std::vector<double>& v, double mean_val) {
    if (v.empty()) return 0;
    double sq_sum = 0;
    for (double x : v) {
        double diff = x - mean_val;
        sq_sum += diff * diff;
    }
    return std::sqrt(sq_sum / static_cast<double>(v.size()));
}

double BenchmarkStats::computeThroughput(uint32_t width, uint32_t height, double median_ns) {
    if (median_ns <= 0) return 0;
    double pixels = static_cast<double>(width) * static_cast<double>(height);
    double seconds = median_ns / 1e9;
    return pixels / seconds / 1e6;  // megapixels/sec
}

double BenchmarkStats::percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0;
    if (sorted.size() == 1) return sorted[0];

    double rank = (p / 100.0) * static_cast<double>(sorted.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(rank));
    size_t upper = static_cast<size_t>(std::ceil(rank));
    if (lower == upper || upper >= sorted.size()) return sorted[lower];

    double frac = rank - static_cast<double>(lower);
    return sorted[lower] * (1.0 - frac) + sorted[upper] * frac;
}

std::vector<double> BenchmarkStats::removeOutliers(const std::vector<double>& sorted) {
    if (sorted.size() < 4) return sorted;

    double q1 = percentile(sorted, 25.0);
    double q3 = percentile(sorted, 75.0);
    double iqr = q3 - q1;
    double lower = q1 - 1.5 * iqr;
    double upper = q3 + 1.5 * iqr;

    std::vector<double> result;
    result.reserve(sorted.size());
    for (double v : sorted) {
        if (v >= lower && v <= upper) {
            result.push_back(v);
        }
    }
    return result;
}
