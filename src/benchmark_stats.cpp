#include "benchmark_stats.h"
#include <algorithm>
#include <cmath>
#include <numeric>

TimingStats BenchmarkStats::compute(const std::vector<double>& samples_ns) {
    TimingStats stats;
    if (samples_ns.empty()) return stats;

    // Sort for percentile computation
    std::vector<double> sorted = samples_ns;
    std::sort(sorted.begin(), sorted.end());

    // Remove outliers via IQR
    std::vector<double> cleaned = removeOutliers(sorted);
    if (cleaned.empty()) cleaned = sorted;  // fallback if all removed

    stats.sample_count = cleaned.size();
    stats.outliers_removed = samples_ns.size() - cleaned.size();

    // Min, max
    stats.min_ns = cleaned.front();
    stats.max_ns = cleaned.back();

    // Mean
    double sum = std::accumulate(cleaned.begin(), cleaned.end(), 0.0);
    stats.mean_ns = sum / static_cast<double>(cleaned.size());

    // Median
    stats.median_ns = percentile(cleaned, 50.0);

    // Stddev
    double sq_sum = 0;
    for (double v : cleaned) {
        double diff = v - stats.mean_ns;
        sq_sum += diff * diff;
    }
    stats.stddev_ns = std::sqrt(sq_sum / static_cast<double>(cleaned.size()));

    // CV%
    stats.cv_percent = (stats.mean_ns > 0) ? (stats.stddev_ns / stats.mean_ns * 100.0) : 0.0;

    // Percentiles (from original sorted, not cleaned)
    stats.p5_ns = percentile(sorted, 5.0);
    stats.p95_ns = percentile(sorted, 95.0);
    stats.p99_ns = percentile(sorted, 99.0);

    return stats;
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
