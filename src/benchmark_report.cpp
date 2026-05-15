#include "benchmark_report.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <sys/stat.h>

BenchmarkReport::BenchmarkReport(const SystemInfo& sys_info, const BenchmarkConfig& config,
                                 const KernelRegistry& registry)
    : sys_info_(sys_info), config_(config), registry_(registry) {}

static void ensureDir(const std::string& path) {
    struct stat st = {};
    if (stat(path.c_str(), &st) != 0) {
#ifdef _WIN32
        _mkdir(path.c_str());
#else
        mkdir(path.c_str(), 0755);
#endif
    }
}

void BenchmarkReport::generate(const std::vector<BenchmarkResult>& results) {
    ensureDir(config_.output_dir);

    if (config_.output_json) {
        writeJSON(results, config_.output_dir + "/benchmark_results.json");
    }
    if (config_.output_csv) {
        writeCSV(results, config_.output_dir + "/benchmark_results.csv");
    }
    if (config_.output_markdown) {
        writeMarkdown(results, config_.output_dir + "/benchmark_results.md");
    }
}

// Escape a string for JSON
static std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
                break;
        }
    }
    return out;
}

static void writeTimingJSON(std::ofstream& f, const std::string& prefix, const TimingStats& ts) {
    f << prefix << "\"mean_ms\": " << std::fixed << std::setprecision(4) << ts.mean_ns / 1e6 << ",\n";
    f << prefix << "\"median_ms\": " << ts.median_ns / 1e6 << ",\n";
    f << prefix << "\"min_ms\": " << ts.min_ns / 1e6 << ",\n";
    f << prefix << "\"max_ms\": " << ts.max_ns / 1e6 << ",\n";
    f << prefix << "\"stddev_ms\": " << ts.stddev_ns / 1e6 << ",\n";
    f << prefix << "\"p5_ms\": " << ts.p5_ns / 1e6 << ",\n";
    f << prefix << "\"p95_ms\": " << ts.p95_ns / 1e6 << ",\n";
    f << prefix << "\"p99_ms\": " << ts.p99_ns / 1e6 << ",\n";
    f << prefix << "\"cv_percent\": " << std::setprecision(2) << ts.cv_percent << ",\n";
    f << prefix << "\"sample_count\": " << ts.sample_count << ",\n";
    f << prefix << "\"outliers_removed\": " << ts.outliers_removed;
}

// ============================================================
// Feature 1: Composite Scoring
// ============================================================
CompositeScores BenchmarkReport::computeScores(const std::vector<BenchmarkResult>& results) {
    CompositeScores scores;

    // Collect MP/s values for graph-mode passing benchmarks, grouped by feature_set and category
    std::vector<double> vision_mps;
    std::vector<double> enhanced_mps;
    std::map<std::string, std::vector<double>> cat_mps;

    for (const auto& r : results) {
        if (!r.supported || !r.verified || r.mode != "graph") continue;
        if (r.megapixels_per_sec <= 0) continue;

        // Key category scores by "feature_set/category" to keep them separate
        cat_mps[r.feature_set + "/" + r.category].push_back(r.megapixels_per_sec);

        if (r.feature_set == "vision") {
            vision_mps.push_back(r.megapixels_per_sec);
        } else if (r.feature_set == "enhanced_vision") {
            enhanced_mps.push_back(r.megapixels_per_sec);
        }
    }

    // Geometric mean helper
    auto geomean = [](const std::vector<double>& v) -> double {
        if (v.empty()) return 0;
        double log_sum = 0;
        for (double x : v) log_sum += std::log(x);
        return std::exp(log_sum / static_cast<double>(v.size()));
    };

    scores.vision_count = static_cast<int>(vision_mps.size());
    scores.enhanced_count = static_cast<int>(enhanced_mps.size());

    if (!vision_mps.empty()) {
        scores.overall_vision_score = geomean(vision_mps);
    }

    if (!enhanced_mps.empty()) {
        // Enhanced Vision Score = geometric mean of only enhanced_vision graph benchmarks
        scores.enhanced_vision_score = geomean(enhanced_mps);
    }

    for (const auto& [cat, vals] : cat_mps) {
        scores.category_scores[cat] = geomean(vals);
    }

    // Framework Score: equal-weight geometric mean of "higher is better"
    // dimensionless framework metrics. We restrict to a curated allow-list so
    // adding new framework metrics later does not silently shift the score.
    static const std::set<std::string> kFrameworkScoreMetrics = {
        "graph_speedup",          // graph_dividend
        "virtual_dividend",       // graph_dividend
        "parallelism_efficiency", // parallel_branches
        "concurrency_speedup",    // async_streaming
    };

    std::vector<double> fw_values;
    for (const auto& r : results) {
        if (!r.supported || !r.verified) continue;
        for (const auto& fm : r.framework_metrics) {
            if (kFrameworkScoreMetrics.find(fm.name) == kFrameworkScoreMetrics.end()) continue;
            if (fm.value <= 0) continue;
            fw_values.push_back(fm.value);
        }
    }
    scores.framework_metric_count = static_cast<int>(fw_values.size());
    if (!fw_values.empty()) {
        scores.framework_score = geomean(fw_values);
    }

    return scores;
}

// ============================================================
// Feature 5: Multi-Resolution Scaling Analysis
// ============================================================
std::vector<ScalingEntry> BenchmarkReport::computeScaling(const std::vector<BenchmarkResult>& results) {
    std::vector<ScalingEntry> entries;

    // Group results by (name, mode) -> list of (total_pixels, resolution_name, MP/s)
    struct ResPoint {
        uint64_t pixels;
        std::string res_name;
        double mps;
    };
    std::map<std::string, std::vector<ResPoint>> groups;

    for (const auto& r : results) {
        if (!r.supported || !r.verified || r.megapixels_per_sec <= 0) continue;
        std::string key = r.name + "|" + r.mode;
        groups[key].push_back({
            static_cast<uint64_t>(r.width) * r.height,
            r.resolution_name,
            r.megapixels_per_sec
        });
    }

    for (const auto& [key, points] : groups) {
        if (points.size() < 2) continue;

        // Find lowest and highest resolution
        auto low = std::min_element(points.begin(), points.end(),
            [](const ResPoint& a, const ResPoint& b) { return a.pixels < b.pixels; });
        auto high = std::max_element(points.begin(), points.end(),
            [](const ResPoint& a, const ResPoint& b) { return a.pixels < b.pixels; });

        if (low->pixels == high->pixels) continue;

        ScalingEntry e;
        size_t sep = key.find('|');
        e.name = key.substr(0, sep);
        e.mode = key.substr(sep + 1);
        e.low_res = low->res_name;
        e.high_res = high->res_name;
        e.low_mps = low->mps;
        e.high_mps = high->mps;
        e.scaling_efficiency = high->mps / low->mps;

        entries.push_back(e);
    }

    // Sort by scaling efficiency ascending (worst scaling first)
    std::sort(entries.begin(), entries.end(),
        [](const ScalingEntry& a, const ScalingEntry& b) {
            return a.scaling_efficiency < b.scaling_efficiency;
        });

    return entries;
}

// ============================================================
// Feature 7: Conformance Checking
// ============================================================
std::vector<ConformanceResult> BenchmarkReport::checkConformance(
        const std::vector<BenchmarkResult>& results,
        const KernelRegistry& registry) {
    std::vector<ConformanceResult> conformance;

    // Collect all kernel names that produced valid graph-mode results, grouped by feature_set
    std::map<std::string, std::set<std::string>> benchmarked;
    for (const auto& r : results) {
        if (r.mode == "graph" && r.supported && r.verified) {
            benchmarked[r.feature_set].insert(r.name);
        }
    }

    // For each feature set, check which available kernels have benchmark results
    std::map<std::string, std::vector<std::string>> fs_available;
    for (const auto& [e, info] : registry.allKernels()) {
        if (info.available) {
            fs_available[info.feature_set].push_back(info.name);
        }
    }

    // Only check feature sets that have benchmarked results (i.e., were selected to run)
    // If no results exist at all, fall back to checking all available feature sets
    std::set<std::string> active_fs;
    for (const auto& r : results) {
        active_fs.insert(r.feature_set);
    }

    for (const auto& [fs, kernels] : fs_available) {
        if (!active_fs.empty() && active_fs.find(fs) == active_fs.end()) continue;
        ConformanceResult cr;
        cr.feature_set = fs;
        cr.total = static_cast<int>(kernels.size());

        const auto& bench_set = benchmarked[fs];
        for (const auto& kname : kernels) {
            // Check if any benchmark result name matches this kernel name
            // Match: exact name, or benchmark name starts with kernel name + '_'
            bool found = false;
            for (const auto& bname : bench_set) {
                if (bname == kname ||
                    (bname.length() > kname.length() &&
                     bname.compare(0, kname.length(), kname) == 0 &&
                     bname[kname.length()] == '_')) {
                    found = true;
                    break;
                }
            }
            if (found) {
                cr.passed++;
            } else {
                cr.missing_kernels.push_back(kname);
            }
        }
        cr.pass = (cr.passed == cr.total);
        conformance.push_back(cr);
    }

    return conformance;
}

// ============================================================
// JSON Report (updated with all features)
// ============================================================
void BenchmarkReport::writeJSON(const std::vector<BenchmarkResult>& results,
                                const std::string& path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        printf("ERROR: Cannot open %s for writing\n", path.c_str());
        return;
    }

    f << "{\n";
    // System info
    f << "  \"system\": {\n";
    f << "    \"hostname\": \"" << jsonEscape(sys_info_.hostname) << "\",\n";
    f << "    \"os_name\": \"" << jsonEscape(sys_info_.os_name) << "\",\n";
    f << "    \"os_version\": \"" << jsonEscape(sys_info_.os_version) << "\",\n";
    f << "    \"cpu_model\": \"" << jsonEscape(sys_info_.cpu_model) << "\",\n";
    f << "    \"cpu_cores\": " << sys_info_.cpu_cores << ",\n";
    f << "    \"ram_gb\": " << std::fixed << std::setprecision(1)
      << (sys_info_.ram_bytes / (1024.0 * 1024.0 * 1024.0)) << ",\n";
    f << "    \"timestamp\": \"" << sys_info_.timestamp_iso8601 << "\"\n";
    f << "  },\n";

    // OpenVX info
    f << "  \"openvx\": {\n";
    f << "    \"implementation\": \"" << jsonEscape(sys_info_.vx_implementation) << "\",\n";
    f << "    \"vendor_id\": " << sys_info_.vx_vendor_id << ",\n";
    f << "    \"version\": " << sys_info_.vx_version << ",\n";
    f << "    \"num_kernels\": " << sys_info_.vx_num_kernels << ",\n";
    f << "    \"extensions\": \"" << jsonEscape(sys_info_.vx_extensions) << "\"\n";
    f << "  },\n";

    // Benchmark version
    f << "  \"benchmark\": {\n";
    f << "    \"version\": \"" << jsonEscape(sys_info_.benchmark_version) << "\",\n";
    f << "    \"git_commit\": \"" << jsonEscape(sys_info_.benchmark_git_commit) << "\"\n";
    f << "  },\n";

    // Config
    f << "  \"config\": {\n";
    f << "    \"iterations\": " << config_.iterations << ",\n";
    f << "    \"warmup\": " << config_.warmup << ",\n";
    f << "    \"seed\": " << config_.seed << ",\n";
    f << "    \"stability_threshold\": " << std::fixed << std::setprecision(1)
      << config_.stability_threshold << ",\n";
    f << "    \"max_retries\": " << config_.max_retries << ",\n";
    f << "    \"resolutions\": [";
    for (size_t i = 0; i < config_.resolutions.size(); i++) {
        if (i > 0) f << ", ";
        f << "\"" << config_.resolutions[i].name << "\"";
    }
    f << "]\n";
    f << "  },\n";

    // Feature set availability
    f << "  \"feature_set_availability\": {\n";
    auto fs_summary = registry_.featureSetSummary();
    for (size_t i = 0; i < fs_summary.size(); i++) {
        const auto& s = fs_summary[i];
        f << "    \"" << jsonEscape(s.feature_set) << "\": {"
          << "\"available\": " << s.available << ", \"total\": " << s.total << "}";
        if (i + 1 < fs_summary.size()) f << ",";
        f << "\n";
    }
    f << "  },\n";

    // Kernel availability
    f << "  \"kernel_availability\": {\n";
    bool first_k = true;
    for (const auto& [e, info] : registry_.allKernels()) {
        if (!first_k) f << ",\n";
        first_k = false;
        f << "    \"" << jsonEscape(info.name) << "\": " << (info.available ? "true" : "false");
    }
    f << "\n  },\n";

    // Feature 1: Composite Scores
    auto scores = computeScores(results);
    f << "  \"scores\": {\n";
    f << "    \"overall_vision_score\": " << std::fixed << std::setprecision(2)
      << scores.overall_vision_score << ",\n";
    f << "    \"vision_benchmark_count\": " << scores.vision_count << ",\n";
    f << "    \"enhanced_vision_score\": " << std::setprecision(2)
      << scores.enhanced_vision_score << ",\n";
    f << "    \"enhanced_benchmark_count\": " << scores.enhanced_count << ",\n";
    // Group category scores by feature set
    f << "    \"category_scores\": {\n";
    {
        // Parse "feature_set/category" keys into nested structure
        std::map<std::string, std::map<std::string, double>> fs_cat;
        for (const auto& [key, sc] : scores.category_scores) {
            size_t sep = key.find('/');
            std::string fs = key.substr(0, sep);
            std::string cat = key.substr(sep + 1);
            fs_cat[fs][cat] = sc;
        }
        bool first_fs = true;
        for (const auto& [fs, cat_scores] : fs_cat) {
            if (!first_fs) f << ",\n";
            first_fs = false;
            f << "      \"" << jsonEscape(fs) << "\": {\n";
            bool first_cs = true;
            for (const auto& [cat, sc] : cat_scores) {
                if (!first_cs) f << ",\n";
                first_cs = false;
                f << "        \"" << jsonEscape(cat) << "\": " << std::setprecision(2) << sc;
            }
            f << "\n      }";
        }
        f << "\n";
    }
    f << "    },\n";
    f << "    \"framework_score\": " << std::setprecision(3)
      << scores.framework_score << ",\n";
    f << "    \"framework_metric_count\": " << scores.framework_metric_count << "\n";
    f << "  },\n";

    // Feature 7: Conformance
    auto conformance = checkConformance(results, registry_);
    f << "  \"conformance\": [\n";
    for (size_t i = 0; i < conformance.size(); i++) {
        const auto& cr = conformance[i];
        f << "    {\n";
        f << "      \"feature_set\": \"" << jsonEscape(cr.feature_set) << "\",\n";
        f << "      \"passed\": " << cr.passed << ",\n";
        f << "      \"total\": " << cr.total << ",\n";
        f << "      \"pass\": " << (cr.pass ? "true" : "false") << ",\n";
        f << "      \"missing_kernels\": [";
        for (size_t j = 0; j < cr.missing_kernels.size(); j++) {
            if (j > 0) f << ", ";
            f << "\"" << jsonEscape(cr.missing_kernels[j]) << "\"";
        }
        f << "]\n";
        f << "    }";
        if (i + 1 < conformance.size()) f << ",";
        f << "\n";
    }
    f << "  ],\n";

    // Feature 5: Scaling Analysis
    auto scaling = computeScaling(results);
    f << "  \"scaling_analysis\": [\n";
    for (size_t i = 0; i < scaling.size(); i++) {
        const auto& se = scaling[i];
        f << "    {\n";
        f << "      \"name\": \"" << jsonEscape(se.name) << "\",\n";
        f << "      \"mode\": \"" << jsonEscape(se.mode) << "\",\n";
        f << "      \"low_resolution\": \"" << jsonEscape(se.low_res) << "\",\n";
        f << "      \"high_resolution\": \"" << jsonEscape(se.high_res) << "\",\n";
        f << "      \"low_mps\": " << std::setprecision(2) << se.low_mps << ",\n";
        f << "      \"high_mps\": " << std::setprecision(2) << se.high_mps << ",\n";
        f << "      \"scaling_efficiency\": " << std::setprecision(4) << se.scaling_efficiency << "\n";
        f << "    }";
        if (i + 1 < scaling.size()) f << ",";
        f << "\n";
    }
    f << "  ],\n";

    // Results
    f << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        f << "    {\n";
        f << "      \"name\": \"" << jsonEscape(r.name) << "\",\n";
        f << "      \"category\": \"" << jsonEscape(r.category) << "\",\n";
        f << "      \"feature_set\": \"" << jsonEscape(r.feature_set) << "\",\n";
        f << "      \"mode\": \"" << jsonEscape(r.mode) << "\",\n";
        f << "      \"resolution\": \"" << jsonEscape(r.resolution_name) << "\",\n";
        f << "      \"width\": " << r.width << ",\n";
        f << "      \"height\": " << r.height << ",\n";
        f << "      \"supported\": " << (r.supported ? "true" : "false") << ",\n";
        f << "      \"verified\": " << (r.verified ? "true" : "false") << ",\n";

        if (!r.supported || !r.verified) {
            f << "      \"skip_reason\": \"" << jsonEscape(r.skip_reason) << "\"\n";
        } else {
            f << "      \"iterations\": " << r.iterations << ",\n";
            f << "      \"warmup\": " << r.warmup << ",\n";
            f << "      \"megapixels_per_sec\": " << std::fixed << std::setprecision(2)
              << r.megapixels_per_sec << ",\n";

            // Feature 6: Peak vs Sustained
            double peak_ms = r.wall_clock.min_ns / 1e6;
            double sustained_ms = r.wall_clock.median_ns / 1e6;
            double sustained_ratio = (r.wall_clock.median_ns > 0)
                ? r.wall_clock.min_ns / r.wall_clock.median_ns : 0;
            f << "      \"peak_ms\": " << std::setprecision(4) << peak_ms << ",\n";
            f << "      \"sustained_ms\": " << sustained_ms << ",\n";
            f << "      \"sustained_ratio\": " << std::setprecision(4) << sustained_ratio << ",\n";

            // Feature 2: Stability fields
            f << "      \"stability_warning\": " << (r.stability_warning ? "true" : "false") << ",\n";
            f << "      \"retry_count\": " << r.retry_count << ",\n";

            f << "      \"wall_clock\": {\n";
            writeTimingJSON(f, "        ", r.wall_clock);
            f << "\n      }";
            if (r.has_vx_perf) {
                f << ",\n      \"vx_perf\": {\n";
                writeTimingJSON(f, "        ", r.vx_perf);
                f << "\n      }";
            }
            f << ",\n      \"framework_metrics\": [";
            for (size_t fm_i = 0; fm_i < r.framework_metrics.size(); fm_i++) {
                const auto& fm = r.framework_metrics[fm_i];
                if (fm_i > 0) f << ",";
                f << "\n        {\"name\": \"" << jsonEscape(fm.name)
                  << "\", \"value\": " << std::fixed << std::setprecision(6) << fm.value
                  << ", \"unit\": \"" << jsonEscape(fm.unit)
                  << "\", \"higher_is_better\": " << (fm.higher_is_better ? "true" : "false")
                  << "}";
            }
            if (!r.framework_metrics.empty()) f << "\n      ";
            f << "]\n";
        }

        f << "    }";
        if (i + 1 < results.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";

    printf("  JSON report: %s\n", path.c_str());
}

// ============================================================
// CSV Report (updated with stability_warning, peak_ms, sustained_ratio)
// ============================================================
void BenchmarkReport::writeCSV(const std::vector<BenchmarkResult>& results,
                               const std::string& path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        printf("ERROR: Cannot open %s for writing\n", path.c_str());
        return;
    }

    // Metadata
    f << "# openvx-mark v" << sys_info_.benchmark_version
      << " (" << sys_info_.benchmark_git_commit << ")"
      << " | " << sys_info_.vx_implementation
      << " | " << sys_info_.timestamp_iso8601 << "\n";

    // Header
    f << "name,category,feature_set,mode,resolution,width,height,supported,verified,"
      << "median_ms,mean_ms,min_ms,max_ms,stddev_ms,p5_ms,p95_ms,p99_ms,"
      << "cv_percent,megapixels_per_sec,samples,outliers_removed,"
      << "vx_perf_avg_ms,vx_perf_min_ms,vx_perf_max_ms,"
      << "stability_warning,peak_ms,sustained_ratio\n";

    for (const auto& r : results) {
        f << r.name << "," << r.category << "," << r.feature_set << "," << r.mode << ","
          << r.resolution_name << "," << r.width << "," << r.height << ","
          << (r.supported ? "true" : "false") << ","
          << (r.verified ? "true" : "false") << ",";

        if (r.supported && r.verified) {
            f << std::fixed << std::setprecision(4)
              << r.wall_clock.median_ns / 1e6 << ","
              << r.wall_clock.mean_ns / 1e6 << ","
              << r.wall_clock.min_ns / 1e6 << ","
              << r.wall_clock.max_ns / 1e6 << ","
              << r.wall_clock.stddev_ns / 1e6 << ","
              << r.wall_clock.p5_ns / 1e6 << ","
              << r.wall_clock.p95_ns / 1e6 << ","
              << r.wall_clock.p99_ns / 1e6 << ","
              << std::setprecision(2) << r.wall_clock.cv_percent << ","
              << r.megapixels_per_sec << ","
              << r.wall_clock.sample_count << ","
              << r.wall_clock.outliers_removed << ",";

            if (r.has_vx_perf) {
                f << std::setprecision(4)
                  << r.vx_perf.mean_ns / 1e6 << ","
                  << r.vx_perf.min_ns / 1e6 << ","
                  << r.vx_perf.max_ns / 1e6;
            } else {
                f << ",,";
            }

            // Stability warning, peak_ms, sustained_ratio
            double peak_ms = r.wall_clock.min_ns / 1e6;
            double sustained_ratio = (r.wall_clock.median_ns > 0)
                ? r.wall_clock.min_ns / r.wall_clock.median_ns : 0;
            f << "," << (r.stability_warning ? "true" : "false")
              << "," << std::setprecision(4) << peak_ms
              << "," << std::setprecision(4) << sustained_ratio;
        } else {
            f << ",,,,,,,,,,,,,,,,,";
        }
        f << "\n";
    }

    printf("  CSV report:  %s\n", path.c_str());
}

// ============================================================
// Markdown Report (updated with all features)
// ============================================================
void BenchmarkReport::writeMarkdown(const std::vector<BenchmarkResult>& results,
                                    const std::string& path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        printf("ERROR: Cannot open %s for writing\n", path.c_str());
        return;
    }

    f << "# OpenVX Benchmark Results\n\n";

    // System info
    f << "## System Information\n\n";
    f << "| Property | Value |\n";
    f << "|:---|:---|\n";
    f << "| Hostname | " << sys_info_.hostname << " |\n";
    f << "| OS | " << sys_info_.os_name << " " << sys_info_.os_version << " |\n";
    f << "| CPU | " << sys_info_.cpu_model << " |\n";
    f << "| Cores | " << sys_info_.cpu_cores << " |\n";
    f << "| RAM | " << std::fixed << std::setprecision(1)
      << (sys_info_.ram_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB |\n";
    f << "| OpenVX Implementation | " << sys_info_.vx_implementation << " |\n";
    f << "| OpenVX Version | " << sys_info_.vx_version << " |\n";
    f << "| Available Kernels | " << registry_.availableCount()
      << " / " << registry_.totalCount() << " |\n";
    f << "| Benchmark Version | " << sys_info_.benchmark_version << " |\n";
    f << "| Git Commit | " << sys_info_.benchmark_git_commit << " |\n";
    f << "| Timestamp | " << sys_info_.timestamp_iso8601 << " |\n\n";

    // Config
    f << "## Configuration\n\n";
    f << "- Iterations: " << config_.iterations << "\n";
    f << "- Warmup: " << config_.warmup << "\n";
    f << "- Seed: " << config_.seed << "\n";
    f << "- Stability Threshold: " << std::fixed << std::setprecision(1)
      << config_.stability_threshold << "% CV\n";
    f << "- Max Retries: " << config_.max_retries << "\n";
    f << "- Resolutions: ";
    for (size_t i = 0; i < config_.resolutions.size(); i++) {
        if (i > 0) f << ", ";
        f << config_.resolutions[i].name
          << " (" << config_.resolutions[i].width << "x" << config_.resolutions[i].height << ")";
    }
    f << "\n\n";

    // Feature set availability
    f << "## Feature Set Availability\n\n";
    auto fs_summary_md = registry_.featureSetSummary();
    f << "| Feature Set | Available | Total |\n";
    f << "|:---|---:|---:|\n";
    for (const auto& s : fs_summary_md) {
        f << "| " << s.feature_set << " | " << s.available << " | " << s.total << " |\n";
    }
    f << "\n";

    // Kernel availability summary
    f << "## Kernel Availability\n\n";
    auto summary = registry_.categorySummary();
    f << "| Category | Available | Total |\n";
    f << "|:---|---:|---:|\n";
    for (const auto& s : summary) {
        f << "| " << s.category << " | " << s.available << " | " << s.total << " |\n";
    }
    f << "\n";

    // Feature 1: Composite Scores
    auto scores = computeScores(results);
    f << "## Composite Scores\n\n";
    f << "| Score | Value | Unit | Benchmarks |\n";
    f << "|:---|---:|:---|---:|\n";
    f << "| OpenVX Vision Score | " << std::fixed << std::setprecision(2)
      << scores.overall_vision_score << " | MP/s | " << scores.vision_count << " |\n";
    if (scores.enhanced_count > 0) {
        f << "| Enhanced Vision Score | " << scores.enhanced_vision_score
          << " | MP/s | " << scores.enhanced_count << " |\n";
    }
    if (scores.framework_metric_count > 0) {
        f << "| OpenVX Framework Score | " << std::setprecision(3)
          << scores.framework_score << " | x (geomean) | "
          << scores.framework_metric_count << " |\n";
    }
    f << "\n";

    if (scores.framework_metric_count > 0) {
        f << "> **Framework Score** is the equal-weight geometric mean of "
          << "`graph_speedup`, `virtual_dividend`, `parallelism_efficiency`, and "
          << "`concurrency_speedup` across all framework benchmarks. Values >1.0x "
          << "indicate the OpenVX graph framework adds aggregate value over a "
          << "kernel-only baseline.\n\n";
    }

    // Category Sub-Scores separated by feature set
    if (!scores.category_scores.empty()) {
        // Group by feature set prefix
        std::map<std::string, std::map<std::string, double>> fs_cat_scores;
        for (const auto& [key, sc] : scores.category_scores) {
            size_t sep = key.find('/');
            std::string fs = key.substr(0, sep);
            std::string cat = key.substr(sep + 1);
            fs_cat_scores[fs][cat] = sc;
        }

        for (const auto& [fs, cat_scores] : fs_cat_scores) {
            f << "### " << fs << " Category Sub-Scores\n\n";
            f << "| Category | Score (MP/s) |\n";
            f << "|:---|---:|\n";
            for (const auto& [cat, sc] : cat_scores) {
                f << "| " << cat << " | " << std::setprecision(2) << sc << " |\n";
            }
            f << "\n";
        }
    }

    // Feature 7: Conformance Summary
    auto conformance = checkConformance(results, registry_);
    f << "## Conformance Summary\n\n";
    f << "| Feature Set | Status | Passed | Total | Missing |\n";
    f << "|:---|:---|---:|---:|:---|\n";
    for (const auto& cr : conformance) {
        f << "| " << cr.feature_set << " | "
          << (cr.pass ? "PASS" : "FAIL") << " | "
          << cr.passed << " | " << cr.total << " | ";
        if (cr.missing_kernels.empty()) {
            f << "- |\n";
        } else {
            for (size_t j = 0; j < cr.missing_kernels.size(); j++) {
                if (j > 0) f << ", ";
                f << cr.missing_kernels[j];
            }
            f << " |\n";
        }
    }
    f << "\n";

    // Feature 4: Performance Summary Tables — separated by feature set
    {
        // Group passing results by feature set
        std::map<std::string, std::vector<const BenchmarkResult*>> fs_passing;
        for (const auto& r : results) {
            if (r.supported && r.verified && r.megapixels_per_sec > 0) {
                fs_passing[r.feature_set].push_back(&r);
            }
        }

        for (const auto& [fs, passing] : fs_passing) {
            // Top-10 highest throughput
            std::vector<const BenchmarkResult*> by_throughput = passing;
            std::sort(by_throughput.begin(), by_throughput.end(),
                [](const BenchmarkResult* a, const BenchmarkResult* b) {
                    return a->megapixels_per_sec > b->megapixels_per_sec;
                });

            size_t top_n = std::min(by_throughput.size(), size_t(10));
            f << "## " << fs << " — Top-" << top_n << " Highest Throughput\n\n";
            f << "| Rank | Benchmark | Mode | Resolution | MP/s | Median (ms) |\n";
            f << "|---:|:---|:---|:---|---:|---:|\n";
            for (size_t i = 0; i < top_n; i++) {
                const auto* r = by_throughput[i];
                f << "| " << (i + 1) << " | " << r->name << " | " << r->mode
                  << " | " << r->resolution_name << " | "
                  << std::setprecision(1) << r->megapixels_per_sec << " | "
                  << std::setprecision(3) << r->wall_clock.median_ns / 1e6 << " |\n";
            }
            f << "\n";

            // Top-10 highest latency (slowest)
            std::vector<const BenchmarkResult*> by_latency = passing;
            std::sort(by_latency.begin(), by_latency.end(),
                [](const BenchmarkResult* a, const BenchmarkResult* b) {
                    return a->wall_clock.median_ns > b->wall_clock.median_ns;
                });

            top_n = std::min(by_latency.size(), size_t(10));
            f << "## " << fs << " — Top-" << top_n << " Highest Latency\n\n";
            f << "| Rank | Benchmark | Mode | Resolution | Median (ms) | MP/s |\n";
            f << "|---:|:---|:---|:---|---:|---:|\n";
            for (size_t i = 0; i < top_n; i++) {
                const auto* r = by_latency[i];
                f << "| " << (i + 1) << " | " << r->name << " | " << r->mode
                  << " | " << r->resolution_name << " | "
                  << std::setprecision(3) << r->wall_clock.median_ns / 1e6 << " | "
                  << std::setprecision(1) << r->megapixels_per_sec << " |\n";
            }
            f << "\n";
        }
    }

    // Feature 6: Peak vs Sustained Performance — separated by feature set
    {
        std::map<std::string, std::vector<const BenchmarkResult*>> fs_passing;
        for (const auto& r : results) {
            if (r.supported && r.verified && r.wall_clock.median_ns > 0) {
                fs_passing[r.feature_set].push_back(&r);
            }
        }

        for (const auto& [fs, passing] : fs_passing) {
            f << "## " << fs << " — Peak vs Sustained Performance\n\n";
            f << "| Benchmark | Mode | Resolution | Peak (ms) | Sustained (ms) | Sustained Ratio |\n";
            f << "|:---|:---|:---|---:|---:|---:|\n";
            for (const auto* r : passing) {
                double peak_ms = r->wall_clock.min_ns / 1e6;
                double sustained_ms = r->wall_clock.median_ns / 1e6;
                double sustained_ratio = r->wall_clock.min_ns / r->wall_clock.median_ns;
                f << "| " << r->name << " | " << r->mode << " | " << r->resolution_name
                  << " | " << std::setprecision(3) << peak_ms
                  << " | " << sustained_ms
                  << " | " << std::setprecision(4) << sustained_ratio << " |\n";
            }
            f << "\n";
        }
    }

    // Feature 5: Multi-Resolution Scaling Analysis
    auto scaling = computeScaling(results);
    if (!scaling.empty()) {
        f << "## Multi-Resolution Scaling Analysis\n\n";
        f << "| Benchmark | Mode | Low Res | High Res | Low MP/s | High MP/s | Scaling Efficiency |\n";
        f << "|:---|:---|:---|:---|---:|---:|---:|\n";
        for (const auto& se : scaling) {
            f << "| " << se.name << " | " << se.mode << " | " << se.low_res
              << " | " << se.high_res << " | " << std::setprecision(1) << se.low_mps
              << " | " << se.high_mps << " | " << std::setprecision(4)
              << se.scaling_efficiency << " |\n";
        }
        f << "\n";
    }

    // Framework Benchmarks: per-scenario framework metrics
    {
        std::vector<const BenchmarkResult*> framework_results;
        for (const auto& r : results) {
            if (!r.framework_metrics.empty()) framework_results.push_back(&r);
        }
        if (!framework_results.empty()) {
            f << "## Framework Benchmarks\n\n";
            f << "Per-scenario metrics that characterize the OpenVX *graph framework* "
              << "(orchestration, scheduling, async, verification) rather than raw kernel "
              << "throughput.\n\n";
            f << "| Benchmark | Resolution | Metric | Value | Unit | Direction |\n";
            f << "|:---|:---|:---|---:|:---|:---|\n";
            for (const auto* r : framework_results) {
                for (const auto& fm : r->framework_metrics) {
                    f << "| " << r->name << " | " << r->resolution_name
                      << " | `" << fm.name << "` | "
                      << std::fixed << std::setprecision(3) << fm.value
                      << " | " << (fm.unit.empty() ? "—" : fm.unit)
                      << " | " << (fm.higher_is_better ? "higher is better" : "lower is better")
                      << " |\n";
                }
            }
            f << "\n";
        }
    }

    // Feature 2: Stability Warnings
    {
        std::vector<const BenchmarkResult*> unstable;
        for (const auto& r : results) {
            if (r.stability_warning) {
                unstable.push_back(&r);
            }
        }

        if (!unstable.empty()) {
            f << "## Stability Warnings\n\n";
            f << "| Benchmark | Mode | Resolution | CV% | Retries |\n";
            f << "|:---|:---|:---|---:|---:|\n";
            for (const auto* r : unstable) {
                f << "| " << r->name << " | " << r->mode << " | " << r->resolution_name
                  << " | " << std::setprecision(1) << r->wall_clock.cv_percent
                  << " | " << r->retry_count << " |\n";
            }
            f << "\n";
        }
    }

    // Group results by feature set then category
    std::map<std::string, std::map<std::string, std::vector<const BenchmarkResult*>>> by_fs_cat;
    for (const auto& r : results) {
        by_fs_cat[r.feature_set][r.category].push_back(&r);
    }

    // Each feature set gets its own top-level section
    for (const auto& [fs, cat_map] : by_fs_cat) {
        f << "## " << fs << " Results\n\n";
        for (const auto& [cat, cat_results] : cat_map) {
            f << "### " << cat << "\n\n";
            f << "| Benchmark | Mode | Resolution | Median (ms) | Mean (ms) | Min (ms) | "
              << "StdDev (ms) | CV% | MP/s |\n";
            f << "|:---|:---|:---|---:|---:|---:|---:|---:|---:|\n";

            for (const auto* r : cat_results) {
                f << "| " << r->name << " | " << r->mode << " | " << r->resolution_name << " | ";
                if (!r->supported) {
                    f << "SKIPPED | | | | | |\n";
                } else if (!r->verified) {
                    f << "VERIFY FAILED | | | | | |\n";
                } else {
                    f << std::fixed << std::setprecision(3)
                      << r->wall_clock.median_ns / 1e6 << " | "
                      << r->wall_clock.mean_ns / 1e6 << " | "
                      << r->wall_clock.min_ns / 1e6 << " | "
                      << r->wall_clock.stddev_ns / 1e6 << " | "
                      << std::setprecision(1) << r->wall_clock.cv_percent << " | "
                      << std::setprecision(1) << r->megapixels_per_sec << " |\n";
                }
            }
            f << "\n";
        }
    }

    // Glossary
    f << "## Glossary\n\n";
    f << "| Term | Description |\n";
    f << "|:---|:---|\n";
    f << "| **Median (ms)** | Median execution time in milliseconds across all measured iterations. "
      << "More robust than mean as it is not affected by outliers. Used as the primary latency metric. |\n";
    f << "| **Mean (ms)** | Arithmetic mean of execution time in milliseconds. "
      << "Can be skewed by occasional long runs. |\n";
    f << "| **Min (ms)** | Minimum (fastest) execution time observed. "
      << "Represents peak hardware capability under ideal conditions. |\n";
    f << "| **Max (ms)** | Maximum (slowest) execution time observed. |\n";
    f << "| **StdDev (ms)** | Standard deviation of execution times. "
      << "Lower values indicate more consistent performance. |\n";
    f << "| **CV%** | Coefficient of Variation (StdDev / Mean x 100). "
      << "Measures result stability as a percentage. Below 5% is excellent, 5-15% is acceptable, "
      << "above 15% suggests system interference or insufficient iterations. |\n";
    f << "| **MP/s** | Megapixels per second throughput = (width x height) / median_time. "
      << "Higher is better. Primary cross-resolution performance metric. |\n";
    f << "| **Peak (ms)** | Same as Min (ms). The best-case single-iteration execution time. |\n";
    f << "| **Sustained (ms)** | Same as Median (ms). Typical execution time under repeated load. |\n";
    f << "| **Sustained Ratio** | Peak / Sustained (min_ms / median_ms). "
      << "1.0 = perfectly consistent performance. Values below 0.9 indicate thermal throttling, "
      << "cache pressure, or scheduling jitter. |\n";
    f << "| **Scaling Efficiency** | (MP/s at high resolution) / (MP/s at low resolution). "
      << "1.0 = perfect linear scaling. Values below 1.0 indicate memory bandwidth or cache limitations "
      << "at higher resolutions. |\n";
    f << "| **Vision Score** | Geometric mean of MP/s across all passing graph-mode vision benchmarks. "
      << "A single composite number for cross-vendor comparison. |\n";
    f << "| **Enhanced Vision Score** | Geometric mean of MP/s across all passing graph-mode "
      << "enhanced_vision benchmarks. |\n";
    f << "| **Category Sub-Score** | Geometric mean of MP/s within a single benchmark category "
      << "(e.g., filters, pixelwise). Identifies relative strengths and weaknesses. |\n";
    f << "| **Conformance** | Whether all available kernels in a feature set produced valid graph-mode results. "
      << "PASS = all tested successfully. FAIL = some kernels skipped or unavailable. |\n";
    f << "| **Stability Warning** | Flagged when CV% exceeds the stability threshold (default 15%). "
      << "Results may not be reliable for comparison. |\n";
    f << "| **graph** | OpenVX graph execution mode. The graph is verified and optimized once, "
      << "then executed repeatedly. Most efficient mode. |\n";
    f << "| **immediate** | OpenVX immediate execution mode (vxu* functions). "
      << "Each call creates a temporary graph internally, adding overhead. "
      << "Useful for measuring single-call latency. |\n";
    f << "| **P5 / P95 / P99** | 5th, 95th, and 99th percentile execution times. "
      << "P5 approximates best-case, P95 and P99 capture tail latency. |\n";
    f << "\n";

    printf("  Markdown:    %s\n", path.c_str());
}

// ============================================================
// Feature 3: Baseline Comparison (C++ implementation)
// ============================================================

static std::string extractJsonString(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\": \"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) {
        search = "\"" + key + "\":\"";
        pos = json.find(search);
    }
    if (pos == std::string::npos) return "";
    pos += search.size();
    size_t end = json.find('"', pos);
    if (end == std::string::npos) return "";
    return json.substr(pos, end - pos);
}

static double extractJsonNumber(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\": ";
    size_t pos = json.find(search);
    if (pos == std::string::npos) {
        search = "\"" + key + "\":";
        pos = json.find(search);
    }
    if (pos == std::string::npos) return 0;
    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    size_t end = pos;
    while (end < json.size() && (json[end] == '-' || json[end] == '+' ||
           json[end] == '.' || json[end] == 'e' || json[end] == 'E' ||
           (json[end] >= '0' && json[end] <= '9'))) end++;
    if (end == pos) return 0;
    return std::stod(json.substr(pos, end - pos));
}

static bool extractJsonBool(const std::string& json, const std::string& key, bool default_val = false) {
    std::string search_true = "\"" + key + "\": true";
    std::string search_true2 = "\"" + key + "\":true";
    if (json.find(search_true) != std::string::npos || json.find(search_true2) != std::string::npos)
        return true;
    std::string search_false = "\"" + key + "\": false";
    std::string search_false2 = "\"" + key + "\":false";
    if (json.find(search_false) != std::string::npos || json.find(search_false2) != std::string::npos)
        return false;
    return default_val;
}

static std::vector<std::string> extractResultObjects(const std::string& json) {
    std::vector<std::string> objects;
    std::string marker = "\"results\": [";
    size_t pos = json.find(marker);
    if (pos == std::string::npos) {
        marker = "\"results\":[";
        pos = json.find(marker);
    }
    if (pos == std::string::npos) return objects;
    pos += marker.size();

    int depth = 0;
    size_t obj_start = 0;
    for (size_t i = pos; i < json.size(); i++) {
        if (json[i] == '{') {
            if (depth == 0) obj_start = i;
            depth++;
        } else if (json[i] == '}') {
            depth--;
            if (depth == 0) {
                objects.push_back(json.substr(obj_start, i - obj_start + 1));
            }
        } else if (json[i] == ']' && depth == 0) {
            break;
        }
    }
    return objects;
}

static std::string extractJsonSection(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\": {";
    size_t pos = json.find(search);
    if (pos == std::string::npos) {
        search = "\"" + key + "\":{";
        pos = json.find(search);
    }
    if (pos == std::string::npos) return "";
    pos = json.find('{', pos);
    int depth = 0;
    for (size_t i = pos; i < json.size(); i++) {
        if (json[i] == '{') depth++;
        else if (json[i] == '}') {
            depth--;
            if (depth == 0) return json.substr(pos, i - pos + 1);
        }
    }
    return "";
}

static std::map<std::string, double> extractCategoryScores(const std::string& json) {
    std::map<std::string, double> scores;
    std::string section = extractJsonSection(json, "category_scores");
    if (section.empty()) return scores;

    // Parse nested: {"vision": {"color": 123.4, ...}, "enhanced_vision": {...}}
    // Look for inner objects
    for (const std::string& fs : {"vision", "enhanced_vision"}) {
        std::string inner = extractJsonSection(section, fs);
        if (inner.empty()) continue;
        size_t pos = 0;
        while (pos < inner.size()) {
            size_t q1 = inner.find('"', pos);
            if (q1 == std::string::npos) break;
            size_t q2 = inner.find('"', q1 + 1);
            if (q2 == std::string::npos) break;
            std::string cat = inner.substr(q1 + 1, q2 - q1 - 1);
            double val = extractJsonNumber(inner.substr(q2), cat);
            if (val > 0) {
                scores[fs + "/" + cat] = val;
            }
            pos = q2 + 1;
            size_t comma = inner.find(',', pos);
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }
    return scores;
}

struct ReportInfo {
    std::string impl_name;
    std::string cpu_model;
    int cpu_cores = 0;
    double ram_gb = 0;
    std::string os_name;
    std::string os_version;
    std::string timestamp;
    std::string benchmark_version;
    std::string git_commit;
    double vision_score = 0;
    double enhanced_vision_score = 0;
    std::map<std::string, double> category_scores;
    bool conformance_pass = false;
    int conformance_passed = 0;
    int conformance_total = 0;
    double framework_score = 0;
    int framework_metric_count = 0;
};

static ReportInfo extractReportInfo(const std::string& json) {
    ReportInfo info;
    std::string sys = extractJsonSection(json, "system");
    info.cpu_model = extractJsonString(sys, "cpu_model");
    info.cpu_cores = (int)extractJsonNumber(sys, "cpu_cores");
    info.ram_gb = extractJsonNumber(sys, "ram_gb");
    info.os_name = extractJsonString(sys, "os_name");
    info.os_version = extractJsonString(sys, "os_version");
    info.timestamp = extractJsonString(sys, "timestamp");

    std::string openvx = extractJsonSection(json, "openvx");
    info.impl_name = extractJsonString(openvx, "implementation");

    std::string bench = extractJsonSection(json, "benchmark");
    info.benchmark_version = extractJsonString(bench, "version");
    info.git_commit = extractJsonString(bench, "git_commit");

    std::string scores_section = extractJsonSection(json, "scores");
    info.vision_score = extractJsonNumber(scores_section, "overall_vision_score");
    info.enhanced_vision_score = extractJsonNumber(scores_section, "enhanced_vision_score");
    info.framework_score = extractJsonNumber(scores_section, "framework_score");
    info.framework_metric_count = (int)extractJsonNumber(scores_section, "framework_metric_count");
    info.category_scores = extractCategoryScores(json);

    // Parse first conformance entry
    size_t conf_pos = json.find("\"conformance\"");
    if (conf_pos != std::string::npos) {
        std::string conf_section = json.substr(conf_pos, std::min((size_t)500, json.size() - conf_pos));
        info.conformance_pass = extractJsonBool(conf_section, "pass");
        info.conformance_passed = (int)extractJsonNumber(conf_section, "passed");
        info.conformance_total = (int)extractJsonNumber(conf_section, "total");
    }

    return info;
}

struct ComparisonEntry {
    std::string name;
    std::string category;
    std::string mode;
    std::string resolution;
    double baseline_median_ms = 0;
    double current_median_ms = 0;
    double baseline_mps = 0;
    double current_mps = 0;
    double baseline_cv = 0;
    double current_cv = 0;
    double speedup = 0;
    bool baseline_verified = false;
    bool current_verified = false;
    bool baseline_supported = false;
    bool current_supported = false;
};

void BenchmarkReport::compareReports(const std::vector<std::string>& json_files,
                                     const std::string& output_path) {
    if (json_files.size() < 2) {
        printf("ERROR: Need at least 2 JSON files for comparison\n");
        return;
    }

    auto readFile = [](const std::string& path) -> std::string {
        std::ifstream f(path);
        if (!f.is_open()) return "";
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    };

    std::string baseline_json = readFile(json_files[0]);
    std::string current_json = readFile(json_files[1]);

    if (baseline_json.empty()) {
        printf("ERROR: Cannot read baseline file: %s\n", json_files[0].c_str());
        return;
    }
    if (current_json.empty()) {
        printf("ERROR: Cannot read current file: %s\n", json_files[1].c_str());
        return;
    }

    ReportInfo info_a = extractReportInfo(baseline_json);
    ReportInfo info_b = extractReportInfo(current_json);

    std::string name_a = info_a.impl_name.empty() ? "Baseline" : info_a.impl_name;
    std::string name_b = info_b.impl_name.empty() ? "Current" : info_b.impl_name;

    auto baseline_objs = extractResultObjects(baseline_json);
    auto current_objs = extractResultObjects(current_json);

    struct FwMetric {
        double value = 0;
        std::string unit;
        bool higher_is_better = true;
    };
    struct ParsedResult {
        std::string name;
        std::string category;
        std::string mode;
        std::string resolution;
        double median_ms;
        double mps;
        double cv_percent;
        bool supported;
        bool verified;
        std::map<std::string, FwMetric> framework_metrics;
    };

    auto parseResults = [](const std::vector<std::string>& objs) {
        std::map<std::string, ParsedResult> map;
        for (const auto& obj : objs) {
            ParsedResult pr;
            pr.supported = !(obj.find("\"supported\": false") != std::string::npos ||
                             obj.find("\"supported\":false") != std::string::npos);
            pr.verified = !(obj.find("\"verified\": false") != std::string::npos ||
                            obj.find("\"verified\":false") != std::string::npos);

            pr.name = extractJsonString(obj, "name");
            pr.category = extractJsonString(obj, "category");
            pr.mode = extractJsonString(obj, "mode");
            pr.resolution = extractJsonString(obj, "resolution");

            if (pr.name.empty()) continue;

            size_t wc_pos = obj.find("\"wall_clock\"");
            if (wc_pos != std::string::npos) {
                std::string wc_section = obj.substr(wc_pos);
                pr.median_ms = extractJsonNumber(wc_section, "median_ms");
                pr.cv_percent = extractJsonNumber(wc_section, "cv_percent");
            }
            pr.mps = extractJsonNumber(obj, "megapixels_per_sec");

            // Parse framework_metrics array (each entry:
            //   {"name": "...", "value": ..., "unit": "...", "higher_is_better": ...})
            size_t fm_pos = obj.find("\"framework_metrics\"");
            if (fm_pos != std::string::npos) {
                size_t arr_open = obj.find('[', fm_pos);
                if (arr_open != std::string::npos) {
                    int depth = 0;
                    size_t entry_start = 0;
                    bool in_entry = false;
                    for (size_t i = arr_open; i < obj.size(); i++) {
                        char c = obj[i];
                        if (c == '{') {
                            if (!in_entry) { entry_start = i; in_entry = true; }
                            depth++;
                        } else if (c == '}') {
                            depth--;
                            if (depth == 0 && in_entry) {
                                std::string entry = obj.substr(entry_start, i - entry_start + 1);
                                std::string nm = extractJsonString(entry, "name");
                                FwMetric m;
                                m.value = extractJsonNumber(entry, "value");
                                m.unit = extractJsonString(entry, "unit");
                                m.higher_is_better = extractJsonBool(entry, "higher_is_better", true);
                                if (!nm.empty()) pr.framework_metrics[nm] = m;
                                in_entry = false;
                            }
                        } else if (c == ']' && depth == 0) {
                            break;
                        }
                    }
                }
            }

            std::string key = pr.name + "|" + pr.mode + "|" + pr.resolution;
            map[key] = pr;
        }
        return map;
    };

    auto baseline_map = parseResults(baseline_objs);
    auto current_map = parseResults(current_objs);

    // Match and compare — collect all keys from both maps
    std::set<std::string> all_keys;
    for (const auto& [key, _] : baseline_map) all_keys.insert(key);
    for (const auto& [key, _] : current_map) all_keys.insert(key);

    std::vector<ComparisonEntry> comparisons;
    std::vector<std::string> only_in_baseline, only_in_current;

    for (const auto& key : all_keys) {
        auto it_a = baseline_map.find(key);
        auto it_b = current_map.find(key);

        if (it_a != baseline_map.end() && it_b == current_map.end()) {
            only_in_baseline.push_back(key);
            continue;
        }
        if (it_a == baseline_map.end() && it_b != current_map.end()) {
            only_in_current.push_back(key);
            continue;
        }

        const auto& base = it_a->second;
        const auto& curr = it_b->second;
        ComparisonEntry ce;
        ce.name = curr.name;
        ce.category = curr.category.empty() ? base.category : curr.category;
        ce.mode = curr.mode;
        ce.resolution = curr.resolution;
        ce.baseline_median_ms = base.median_ms;
        ce.current_median_ms = curr.median_ms;
        ce.baseline_mps = base.mps;
        ce.current_mps = curr.mps;
        ce.baseline_cv = base.cv_percent;
        ce.current_cv = curr.cv_percent;
        ce.baseline_verified = base.verified;
        ce.current_verified = curr.verified;
        ce.baseline_supported = base.supported;
        ce.current_supported = curr.supported;

        if (base.median_ms > 0 && curr.median_ms > 0) {
            ce.speedup = base.median_ms / curr.median_ms;
        }

        comparisons.push_back(ce);
    }

    // Sort by speedup ascending (slowest relative performance first, then fastest)
    std::sort(comparisons.begin(), comparisons.end(),
        [](const ComparisonEntry& a, const ComparisonEntry& b) {
            return a.speedup < b.speedup;
        });

    // Write comparison markdown
    std::string md_path = output_path + ".md";
    ensureDir(output_path.substr(0, output_path.find_last_of('/')));

    std::ofstream f(md_path);
    if (!f.is_open()) {
        printf("ERROR: Cannot open %s for writing\n", md_path.c_str());
        return;
    }

    f << "# OpenVX Benchmark Comparison\n\n";
    f << "**" << name_a << "** vs **" << name_b << "**\n\n";

    // --- System Info ---
    bool hw_match = (!info_a.cpu_model.empty() && info_a.cpu_model == info_b.cpu_model
                     && info_a.cpu_cores == info_b.cpu_cores);

    f << "## System Info\n\n";
    if (hw_match) {
        f << "| Property | Value |\n";
        f << "|:---|:---|\n";
        f << "| CPU | " << info_a.cpu_model << " |\n";
        f << "| Cores | " << info_a.cpu_cores << " |\n";
        if (info_a.ram_gb > 0)
            f << "| RAM | " << std::fixed << std::setprecision(1) << info_a.ram_gb << " GB |\n";
        f << "| OS | " << info_a.os_name << " " << info_a.os_version << " |\n";
        f << "\n> Same hardware — both benchmarks ran on identical hardware.\n\n";
    } else {
        f << "| Property | " << name_a << " | " << name_b << " |\n";
        f << "|:---|:---|:---|\n";
        f << "| CPU | " << (info_a.cpu_model.empty() ? "N/A" : info_a.cpu_model)
          << " | " << (info_b.cpu_model.empty() ? "N/A" : info_b.cpu_model) << " |\n";
        f << "| Cores | " << info_a.cpu_cores << " | " << info_b.cpu_cores << " |\n";
        f << "| OS | " << info_a.os_name << " " << info_a.os_version
          << " | " << info_b.os_name << " " << info_b.os_version << " |\n";
        f << "\n> **Warning:** Benchmarks ran on different hardware — results may not be directly comparable.\n\n";
    }

    // --- Conformance & Scores ---
    f << "## Conformance & Scores\n\n";
    f << "| Metric | " << name_a << " | " << name_b << " |\n";
    f << "|:---|---:|---:|\n";
    f << "| Vision Score (MP/s) | " << std::fixed << std::setprecision(2)
      << info_a.vision_score << " | " << info_b.vision_score << " |\n";
    if (info_a.enhanced_vision_score > 0 || info_b.enhanced_vision_score > 0) {
        f << "| Enhanced Vision Score (MP/s) | " << info_a.enhanced_vision_score
          << " | " << info_b.enhanced_vision_score << " |\n";
    }
    if (info_a.framework_metric_count > 0 || info_b.framework_metric_count > 0) {
        f << "| Framework Score (x, geomean) | " << std::setprecision(3)
          << info_a.framework_score << " | " << info_b.framework_score << " |\n";
    }
    f << "| Conformance | " << (info_a.conformance_pass ? "PASS" : "FAIL")
      << " (" << info_a.conformance_passed << "/" << info_a.conformance_total << ")"
      << " | " << (info_b.conformance_pass ? "PASS" : "FAIL")
      << " (" << info_b.conformance_passed << "/" << info_b.conformance_total << ")"
      << " |\n\n";

    // --- Category Sub-Scores ---
    std::set<std::string> all_cats;
    for (const auto& [k, _] : info_a.category_scores) all_cats.insert(k);
    for (const auto& [k, _] : info_b.category_scores) all_cats.insert(k);

    if (!all_cats.empty()) {
        f << "## Category Sub-Scores\n\n";
        f << "| Category | " << name_a << " (MP/s) | " << name_b << " (MP/s) | Change % |\n";
        f << "|:---|---:|---:|---:|\n";
        for (const auto& cat : all_cats) {
            double a_val = 0, b_val = 0;
            auto it_a = info_a.category_scores.find(cat);
            auto it_b = info_b.category_scores.find(cat);
            if (it_a != info_a.category_scores.end()) a_val = it_a->second;
            if (it_b != info_b.category_scores.end()) b_val = it_b->second;
            double change = 0;
            if (a_val > 0) change = ((b_val - a_val) / a_val) * 100.0;
            // Strip feature_set prefix for display
            std::string display_cat = cat;
            size_t slash = cat.find('/');
            if (slash != std::string::npos) display_cat = cat.substr(slash + 1);
            f << "| " << display_cat << " | " << std::setprecision(2) << a_val
              << " | " << b_val << " | "
              << (change >= 0 ? "+" : "") << std::setprecision(1) << change << " |\n";
        }
        f << "\n";
    }

    // --- Framework Metrics Comparison ---
    //
    // Layout intent (uniform & intuitive — mirrors scripts/compare_reports.py):
    //
    //   * One H3 subsection per (benchmark, resolution) scenario.
    //   * Within each scenario, metrics are split by direction into up
    //     to three H4 sub-tables: ↑ higher-is-better, ↓ lower-is-better,
    //     · descriptive. Each sub-table is internally uniform — every
    //     row uses the SAME ratio formula and the SAME "what counts as
    //     a win" rule.
    //   * The ratio column is ALWAYS the literal raw `B/A`. No
    //     per-row inversion. The H4 heading tells the reader whether
    //     bigger or smaller is the winning direction:
    //       - ↑ table: bold when ratio > 1.00x (B is bigger = better)
    //       - ↓ table: bold when ratio < 1.00x (B is smaller = better)
    //   * Descriptive metrics (unit=="count") get a stripped-down
    //     no-ratio sub-table.
    //   * Per-scenario summary line aggregating B's win count across
    //     all comparable metrics.
    {
        struct FwKey { std::string display; };
        std::map<std::string, FwKey> fw_keys;
        std::map<std::string, std::set<std::string>> fw_metric_names;
        auto noteFw = [&](const std::map<std::string, ParsedResult>& m) {
            for (const auto& [k, pr] : m) {
                if (pr.framework_metrics.empty()) continue;
                std::string key = pr.name + "|" + pr.resolution;
                fw_keys[key] = FwKey{pr.name + " @ " + pr.resolution};
                for (const auto& [nm, _] : pr.framework_metrics) {
                    fw_metric_names[key].insert(nm);
                }
            }
        };
        noteFw(baseline_map);
        noteFw(current_map);

        // Classify a metric: 0 = higher-better, 1 = lower-better, 2 = descriptive.
        auto classify = [](const FwMetric* m) -> int {
            if (!m) return 0;
            if (m->unit == "count") return 2;
            return m->higher_is_better ? 0 : 1;
        };

        // Render one sub-table for a given direction band (0 or 1) and
        // accumulate B's win count into the per-scenario totals.
        struct Row {
            std::string name;
            const FwMetric* a;
            const FwMetric* b;
        };
        auto write_table = [&](std::ostream& os, const std::vector<Row>& rows,
                               int direction, int& wins_b_out, int& comparable_out) {
            const bool descriptive = (direction == 2);
            if (descriptive) {
                os << "| Metric | Unit | " << name_a << " | " << name_b << " |\n";
                os << "|:---|:---|---:|---:|\n";
            } else {
                os << "| Metric | Unit | " << name_a << " | " << name_b
                   << " | B/A Ratio |\n";
                os << "|:---|:---|---:|---:|---:|\n";
            }
            for (const auto& row : rows) {
                std::string unit = row.a ? row.a->unit
                                   : (row.b ? row.b->unit : std::string());
                double a_val = row.a ? row.a->value : 0.0;
                double b_val = row.b ? row.b->value : 0.0;

                os << "| `" << row.name << "` | "
                   << (unit.empty() ? "—" : unit) << " | ";
                if (row.a) os << std::fixed << std::setprecision(3) << a_val;
                else os << "—";
                os << " | ";
                if (row.b) os << std::fixed << std::setprecision(3) << b_val;
                else os << "—";

                if (descriptive) {
                    os << " |\n";
                    continue;
                }

                os << " | ";
                if (row.a && row.b && a_val > 0 && b_val > 0) {
                    double ratio = b_val / a_val;  // always literal B/A
                    comparable_out++;
                    bool b_wins = (direction == 0) ? (ratio > 1.0) : (ratio < 1.0);
                    if (b_wins) {
                        wins_b_out++;
                        os << "**" << std::fixed << std::setprecision(2)
                           << ratio << "x**";
                    } else {
                        os << std::fixed << std::setprecision(2) << ratio << "x";
                    }
                } else {
                    os << "—";
                }
                os << " |\n";
            }
        };

        if (!fw_keys.empty()) {
            f << "## Framework Metrics Comparison\n\n";
            f << "> Per-scenario framework metrics — graph orchestration, "
              << "scheduling, async streaming, verify cost. Each scenario "
              << "is split into up to three sub-tables grouped by metric "
              << "direction so every row in a given table follows the same "
              << "rule:\n>\n"
              << ">   * **↑ Higher-is-better** — throughput, speedup, fusion ratio. Ratio = "
              << name_b << " / " << name_a << "; **" << name_b
              << " wins when ratio > 1.00x**.\n"
              << ">   * **↓ Lower-is-better** — latency, overhead in ms. Ratio = "
              << name_b << " / " << name_a << " (raw, no inversion); **"
              << name_b << " wins when ratio < 1.00x** because lower is better here.\n"
              << ">   * **· Descriptive** — counts and structural sizes; no ratio shown.\n>\n"
              << "> **Bold** ratios mark cells where " << name_b
              << " wins, regardless of direction.\n\n";

            for (const auto& [key, fk] : fw_keys) {
                std::string base_key, curr_key;
                for (const auto& [k, pr] : baseline_map) {
                    if (pr.name + "|" + pr.resolution == key) { base_key = k; break; }
                }
                for (const auto& [k, pr] : current_map) {
                    if (pr.name + "|" + pr.resolution == key) { curr_key = k; break; }
                }
                const ParsedResult* a_pr = base_key.empty() ? nullptr : &baseline_map.at(base_key);
                const ParsedResult* b_pr = curr_key.empty() ? nullptr : &current_map.at(curr_key);

                f << "### " << fk.display << "\n\n";

                // Bucket this scenario's metrics by direction.
                std::vector<Row> higher, lower, descriptive;
                for (const auto& nm : fw_metric_names[key]) {
                    const FwMetric* a_m = nullptr;
                    const FwMetric* b_m = nullptr;
                    if (a_pr) {
                        auto it = a_pr->framework_metrics.find(nm);
                        if (it != a_pr->framework_metrics.end()) a_m = &it->second;
                    }
                    if (b_pr) {
                        auto it = b_pr->framework_metrics.find(nm);
                        if (it != b_pr->framework_metrics.end()) b_m = &it->second;
                    }
                    Row row{nm, a_m, b_m};
                    int dir = classify(a_m ? a_m : b_m);
                    if (dir == 0) higher.push_back(row);
                    else if (dir == 1) lower.push_back(row);
                    else descriptive.push_back(row);
                }

                int total_wins = 0;
                int total_comparable = 0;

                if (!higher.empty()) {
                    f << "#### ↑ Higher-is-better metrics\n\n";
                    int w = 0, c = 0;
                    write_table(f, higher, 0, w, c);
                    total_wins += w;
                    total_comparable += c;
                    f << "\n_" << name_b << " wins " << w << "/" << c
                      << " in this category (**bold** = " << name_b
                      << " better, i.e. ratio > 1.00x)._\n\n";
                }

                if (!lower.empty()) {
                    f << "#### ↓ Lower-is-better metrics\n\n";
                    int w = 0, c = 0;
                    write_table(f, lower, 1, w, c);
                    total_wins += w;
                    total_comparable += c;
                    f << "\n_" << name_b << " wins " << w << "/" << c
                      << " in this category (**bold** = " << name_b
                      << " better, i.e. ratio < 1.00x because " << name_b
                      << " is smaller / faster)._\n\n";
                }

                if (!descriptive.empty()) {
                    f << "#### · Descriptive metrics\n\n";
                    int w = 0, c = 0;
                    write_table(f, descriptive, 2, w, c);
                    f << "\n";
                }

                if (total_comparable > 0) {
                    f << "_**Per-scenario summary:** " << name_b << " wins **"
                      << total_wins << "/" << total_comparable
                      << "** comparable metrics in this scenario._\n\n";
                } else {
                    f << "_No comparable metrics in this scenario._\n\n";
                }
            }
        }
    }

    // --- Summary ---
    int both_verified = 0, a_only_verified = 0, b_only_verified = 0, neither_verified = 0;
    for (const auto& ce : comparisons) {
        bool a_ok = ce.baseline_supported && ce.baseline_verified;
        bool b_ok = ce.current_supported && ce.current_verified;
        if (a_ok && b_ok) both_verified++;
        else if (a_ok) a_only_verified++;
        else if (b_ok) b_only_verified++;
        else neither_verified++;
    }

    f << "## Summary\n\n";
    f << "| Metric | Count |\n";
    f << "|:---|---:|\n";
    f << "| Total benchmarks compared | " << comparisons.size() << " |\n";
    f << "| Both verified | " << both_verified << " |\n";
    if (a_only_verified > 0)
        f << "| Verified only in " << name_a << " | " << a_only_verified << " |\n";
    if (b_only_verified > 0)
        f << "| Verified only in " << name_b << " | " << b_only_verified << " |\n";
    if (neither_verified > 0)
        f << "| Neither verified | " << neither_verified << " |\n";
    if (!only_in_baseline.empty())
        f << "| Only in " << name_a << " | " << only_in_baseline.size() << " |\n";
    if (!only_in_current.empty())
        f << "| Only in " << name_b << " | " << only_in_current.size() << " |\n";
    f << "\n";

    // --- Detailed Results ---
    f << "## Detailed Comparison\n\n";
    f << "> Speedup = " << name_b << " throughput / " << name_a
      << " throughput. Values >1.00 mean " << name_b << " is faster.\n\n";
    f << "| Benchmark | Mode | Resolution | " << name_a << " (ms) | " << name_a << " (MP/s) | "
      << name_a << " Verified | " << name_b << " (ms) | " << name_b << " (MP/s) | "
      << name_b << " Verified | Speedup |\n";
    f << "|:---|:---|:---|---:|---:|:---:|---:|---:|:---:|---:|\n";

    for (const auto& ce : comparisons) {
        std::string flag;
        if (ce.baseline_cv > 15.0 || ce.current_cv > 15.0) flag = " *";

        f << "| " << ce.name << " | " << ce.mode << " | " << ce.resolution << " | ";

        // Baseline columns
        if (!ce.baseline_supported) {
            f << "N/A | N/A | N/A | ";
        } else {
            f << std::fixed << std::setprecision(3) << ce.baseline_median_ms
              << " | " << std::setprecision(1) << ce.baseline_mps
              << " | " << (ce.baseline_verified ? "PASS" : "FAIL") << " | ";
        }

        // Current columns
        if (!ce.current_supported) {
            f << "N/A | N/A | N/A | ";
        } else {
            f << std::fixed << std::setprecision(3) << ce.current_median_ms
              << " | " << std::setprecision(1) << ce.current_mps
              << " | " << (ce.current_verified ? "PASS" : "FAIL") << " | ";
        }

        // Speedup
        if (ce.speedup > 0 && ce.baseline_supported && ce.current_supported
            && ce.baseline_verified && ce.current_verified) {
            f << std::setprecision(2) << ce.speedup << "x" << flag;
        } else {
            f << "N/A";
        }
        f << " |\n";
    }
    f << "\n";

    // Stability caveat
    bool has_unstable = false;
    for (const auto& ce : comparisons) {
        if (ce.baseline_cv > 15.0 || ce.current_cv > 15.0) { has_unstable = true; break; }
    }
    if (has_unstable) {
        f << "> \\* High variability (CV% > 15%) — comparison may not be reliable for these benchmarks. "
          << "Consider increasing iterations.\n\n";
    }

    // --- Benchmarks Only In One Report ---
    if (!only_in_baseline.empty() || !only_in_current.empty()) {
        f << "## Benchmarks Only In One Report\n\n";
        if (!only_in_baseline.empty()) {
            f << "### Only in " << name_a << "\n\n";
            f << "| Benchmark | Mode | Resolution |\n";
            f << "|:---|:---|:---|\n";
            for (const auto& key : only_in_baseline) {
                size_t p1 = key.find('|');
                size_t p2 = key.find('|', p1 + 1);
                f << "| " << key.substr(0, p1) << " | " << key.substr(p1 + 1, p2 - p1 - 1)
                  << " | " << key.substr(p2 + 1) << " |\n";
            }
            f << "\n";
        }
        if (!only_in_current.empty()) {
            f << "### Only in " << name_b << "\n\n";
            f << "| Benchmark | Mode | Resolution |\n";
            f << "|:---|:---|:---|\n";
            for (const auto& key : only_in_current) {
                size_t p1 = key.find('|');
                size_t p2 = key.find('|', p1 + 1);
                f << "| " << key.substr(0, p1) << " | " << key.substr(p1 + 1, p2 - p1 - 1)
                  << " | " << key.substr(p2 + 1) << " |\n";
            }
            f << "\n";
        }
    }

    printf("  Comparison:  %s\n", md_path.c_str());
    printf("  %zu benchmarks compared, %d both verified, %zu only in %s, %zu only in %s\n",
           comparisons.size(), both_verified,
           only_in_baseline.size(), name_a.c_str(),
           only_in_current.size(), name_b.c_str());

    // --- CSV Output ---
    std::string csv_path = output_path + ".csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "benchmark,category,mode,resolution,"
            << name_a << "_median_ms," << name_a << "_mp_per_sec," << name_a << "_verified,"
            << name_b << "_median_ms," << name_b << "_mp_per_sec," << name_b << "_verified,"
            << "speedup\n";
        for (const auto& ce : comparisons) {
            csv << ce.name << "," << ce.category << "," << ce.mode << "," << ce.resolution << ",";
            if (ce.baseline_supported) {
                csv << std::fixed << std::setprecision(4) << ce.baseline_median_ms << ","
                    << std::setprecision(2) << ce.baseline_mps << ","
                    << (ce.baseline_verified ? "PASS" : "FAIL") << ",";
            } else {
                csv << ",,,";
            }
            if (ce.current_supported) {
                csv << std::fixed << std::setprecision(4) << ce.current_median_ms << ","
                    << std::setprecision(2) << ce.current_mps << ","
                    << (ce.current_verified ? "PASS" : "FAIL") << ",";
            } else {
                csv << ",,,";
            }
            if (ce.speedup > 0 && ce.baseline_verified && ce.current_verified) {
                csv << std::setprecision(4) << ce.speedup;
            }
            csv << "\n";
        }
        printf("  Comparison CSV: %s\n", csv_path.c_str());
    }
}
