#ifndef BENCHMARK_REPORT_H
#define BENCHMARK_REPORT_H

#include "benchmark_config.h"
#include "benchmark_stats.h"
#include "kernel_registry.h"
#include "system_info.h"
#include <map>
#include <string>
#include <vector>

// Composite scoring (Feature 1)
struct CompositeScores {
    double overall_vision_score = 0;     // geometric mean of MP/s across all passing graph-mode vision benchmarks
    double enhanced_vision_score = 0;    // geometric mean when enhanced_vision benchmarks run
    std::map<std::string, double> category_scores;  // per-category geometric mean
    int vision_count = 0;
    int enhanced_count = 0;
};

// Conformance checking (Feature 7)
struct ConformanceResult {
    std::string feature_set;
    int passed = 0;
    int total = 0;
    std::vector<std::string> missing_kernels;
    bool pass = false;
};

// Multi-resolution scaling (Feature 5)
struct ScalingEntry {
    std::string name;
    std::string mode;
    std::string low_res;
    std::string high_res;
    double low_mps = 0;     // MP/s at low resolution
    double high_mps = 0;    // MP/s at high resolution
    double scaling_efficiency = 0;  // high_mps / low_mps (1.0 = perfect)
};

class BenchmarkReport {
public:
    BenchmarkReport(const SystemInfo& sys_info, const BenchmarkConfig& config,
                    const KernelRegistry& registry);

    void generate(const std::vector<BenchmarkResult>& results);

    // Individual format generators
    void writeJSON(const std::vector<BenchmarkResult>& results, const std::string& path);
    void writeCSV(const std::vector<BenchmarkResult>& results, const std::string& path);
    void writeMarkdown(const std::vector<BenchmarkResult>& results, const std::string& path);

    // Analytics
    static CompositeScores computeScores(const std::vector<BenchmarkResult>& results);
    static std::vector<ScalingEntry> computeScaling(const std::vector<BenchmarkResult>& results);
    static std::vector<ConformanceResult> checkConformance(const std::vector<BenchmarkResult>& results,
                                                            const KernelRegistry& registry);

    // Cross-vendor comparison
    static void compareReports(const std::vector<std::string>& json_files,
                               const std::string& output_path);

private:
    SystemInfo sys_info_;
    BenchmarkConfig config_;
    const KernelRegistry& registry_;
};

#endif // BENCHMARK_REPORT_H
