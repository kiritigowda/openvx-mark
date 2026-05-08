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
    f << "    }\n";
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
            f << "\n";
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
    f << "| Score | Value (MP/s) | Benchmarks |\n";
    f << "|:---|---:|---:|\n";
    f << "| OpenVX Vision Score | " << std::fixed << std::setprecision(2)
      << scores.overall_vision_score << " | " << scores.vision_count << " |\n";
    if (scores.enhanced_count > 0) {
        f << "| Enhanced Vision Score | " << scores.enhanced_vision_score
          << " | " << scores.enhanced_count << " |\n";
    }
    f << "\n";

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
    double change_percent = 0;
    std::string status;
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
    };

    auto parseResults = [](const std::vector<std::string>& objs) {
        std::map<std::string, ParsedResult> map;
        for (const auto& obj : objs) {
            ParsedResult pr;
            pr.supported = !extractJsonBool(obj, "supported", true) ? false : true;
            pr.supported = !(obj.find("\"supported\": false") != std::string::npos ||
                             obj.find("\"supported\":false") != std::string::npos);
            pr.verified = !(obj.find("\"verified\": false") != std::string::npos ||
                            obj.find("\"verified\":false") != std::string::npos);

            if (!pr.supported || !pr.verified) continue;

            pr.name = extractJsonString(obj, "name");
            pr.category = extractJsonString(obj, "category");
            pr.mode = extractJsonString(obj, "mode");
            pr.resolution = extractJsonString(obj, "resolution");

            size_t wc_pos = obj.find("\"wall_clock\"");
            if (wc_pos != std::string::npos) {
                std::string wc_section = obj.substr(wc_pos);
                pr.median_ms = extractJsonNumber(wc_section, "median_ms");
                pr.cv_percent = extractJsonNumber(wc_section, "cv_percent");
            } else {
                continue;
            }
            pr.mps = extractJsonNumber(obj, "megapixels_per_sec");

            if (pr.name.empty() || pr.median_ms <= 0) continue;

            std::string key = pr.name + "|" + pr.mode + "|" + pr.resolution;
            map[key] = pr;
        }
        return map;
    };

    auto baseline_map = parseResults(baseline_objs);
    auto current_map = parseResults(current_objs);

    // Match and compare
    std::vector<ComparisonEntry> comparisons;
    int regressions = 0, improvements = 0, same_count = 0;
    std::map<std::string, int> cat_regressions, cat_improvements;

    for (const auto& [key, curr] : current_map) {
        auto it = baseline_map.find(key);
        if (it == baseline_map.end()) continue;

        const auto& base = it->second;
        ComparisonEntry ce;
        ce.name = curr.name;
        ce.category = curr.category;
        ce.mode = curr.mode;
        ce.resolution = curr.resolution;
        ce.baseline_median_ms = base.median_ms;
        ce.current_median_ms = curr.median_ms;
        ce.baseline_mps = base.mps;
        ce.current_mps = curr.mps;
        ce.baseline_cv = base.cv_percent;
        ce.current_cv = curr.cv_percent;

        if (base.median_ms > 0) {
            ce.change_percent = ((curr.median_ms - base.median_ms) / base.median_ms) * 100.0;
        }

        if (ce.change_percent > 5.0) {
            ce.status = "REGRESSION";
            regressions++;
            cat_regressions[ce.category]++;
        } else if (ce.change_percent < -5.0) {
            ce.status = "IMPROVEMENT";
            improvements++;
            cat_improvements[ce.category]++;
        } else {
            ce.status = "same";
            same_count++;
        }

        comparisons.push_back(ce);
    }

    // Find benchmarks only in one report
    std::vector<std::string> only_in_baseline, only_in_current;
    for (const auto& [key, _] : baseline_map) {
        if (current_map.find(key) == current_map.end())
            only_in_baseline.push_back(key);
    }
    for (const auto& [key, _] : current_map) {
        if (baseline_map.find(key) == baseline_map.end())
            only_in_current.push_back(key);
    }

    // Sort by change_percent descending (worst regressions first)
    std::sort(comparisons.begin(), comparisons.end(),
        [](const ComparisonEntry& a, const ComparisonEntry& b) {
            return a.change_percent > b.change_percent;
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

    // --- Summary ---
    f << "## Summary\n\n";
    f << "| Metric | Count |\n";
    f << "|:---|---:|\n";
    f << "| Total compared | " << comparisons.size() << " |\n";
    f << "| Regressions (>5% slower) | " << regressions << " |\n";
    f << "| Improvements (>5% faster) | " << improvements << " |\n";
    f << "| Unchanged | " << same_count << " |\n\n";

    if (!cat_regressions.empty() || !cat_improvements.empty()) {
        f << "### By Category\n\n";
        f << "| Category | Regressions | Improvements |\n";
        f << "|:---|---:|---:|\n";
        std::set<std::string> summary_cats;
        for (const auto& [c, _] : cat_regressions) summary_cats.insert(c);
        for (const auto& [c, _] : cat_improvements) summary_cats.insert(c);
        for (const auto& c : summary_cats) {
            int reg = cat_regressions.count(c) ? cat_regressions[c] : 0;
            int imp = cat_improvements.count(c) ? cat_improvements[c] : 0;
            f << "| " << c << " | " << reg << " | " << imp << " |\n";
        }
        f << "\n";
    }

    // --- Detailed Results ---
    f << "## Detailed Comparison\n\n";
    f << "> Change % is based on median latency. Positive = slower (regression), negative = faster (improvement).\n\n";
    f << "| Benchmark | Mode | Resolution | " << name_a << " (ms) | " << name_a << " (MP/s) | "
      << name_b << " (ms) | " << name_b << " (MP/s) | Change % | Status |\n";
    f << "|:---|:---|:---|---:|---:|---:|---:|---:|:---|\n";

    for (const auto& ce : comparisons) {
        std::string flag;
        if (ce.baseline_cv > 15.0 || ce.current_cv > 15.0) flag = " *";
        f << "| " << ce.name << " | " << ce.mode << " | " << ce.resolution
          << " | " << std::fixed << std::setprecision(3) << ce.baseline_median_ms
          << " | " << std::setprecision(1) << ce.baseline_mps
          << " | " << std::setprecision(3) << ce.current_median_ms
          << " | " << std::setprecision(1) << ce.current_mps
          << " | " << std::setprecision(1)
          << (ce.change_percent >= 0 ? "+" : "") << ce.change_percent
          << " | " << ce.status << flag << " |\n";
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
    printf("  %d regressions, %d improvements, %d unchanged out of %zu compared\n",
           regressions, improvements, same_count, comparisons.size());

    // --- CSV Output ---
    std::string csv_path = output_path + ".csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "benchmark,category,mode,resolution,"
            << name_a << "_median_ms," << name_a << "_mp_per_sec,"
            << name_b << "_median_ms," << name_b << "_mp_per_sec,"
            << "change_percent,status\n";
        for (const auto& ce : comparisons) {
            csv << ce.name << "," << ce.category << "," << ce.mode << "," << ce.resolution << ","
                << std::fixed << std::setprecision(4) << ce.baseline_median_ms << ","
                << std::setprecision(2) << ce.baseline_mps << ","
                << std::setprecision(4) << ce.current_median_ms << ","
                << std::setprecision(2) << ce.current_mps << ","
                << std::setprecision(2) << ce.change_percent << ","
                << ce.status << "\n";
        }
        printf("  Comparison CSV: %s\n", csv_path.c_str());
    }
}
