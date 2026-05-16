// opencv-mark — OpenCV baseline companion to openvx-mark.
//
// CLI mirrors openvx-mark's so the same shell scripts and CI steps
// can drive both binaries with identical flags. The output JSON
// schema is identical too (BenchmarkReport from the shared
// `bench_core` static library), so the resulting JSON drops straight
// into scripts/compare_reports.py for cross-vendor comparison
// alongside any OpenVX implementation report.

#include "benchmark_catalog.h"
#include "benchmark_config.h"
#include "benchmark_report.h"
#include "benchmark_stats.h"
#include "opencv_context.h"
#include "opencv_runner.h"
#include "opencv_test_data.h"
#include "opencv_verify.h"
#include "system_info.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/core.hpp>
#include <set>
#include <string>
#include <vector>

namespace {

void printUsage(const char* prog) {
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("opencv-mark v%s (%s) — OpenCV Baseline Benchmark Suite\n",
           OPENCV_MARK_VERSION, GIT_COMMIT_SHA);
    printf("Linked against OpenCV %s\n\n", OPENCV_VERSION_STRING);

    printf("Benchmark Selection:\n");
    printf("  --feature-set SET[,SET,...]   Currently only 'vision' is supported (PR1 scope)\n");
    printf("  --category CAT[,CAT,...]      Filter by category (filters,color,geometric)\n");
    printf("  --kernel NAME[,NAME,...]      Filter by kernel name\n\n");

    printf("Resolution:\n");
    printf("  --resolution RES[,RES,...]    VGA,HD,FHD,4K,8K (default: VGA,FHD,4K)\n");
    printf("  --width W --height H          Custom resolution\n\n");

    printf("Timing:\n");
    printf("  --iterations N                Measurement iterations (default: 100)\n");
    printf("  --warmup N                    Warm-up iterations (default: 10)\n");
    printf("  --seed N                      PRNG seed (default: 42)\n");
    printf("  --stability-threshold N       CV%% threshold (default: 15)\n");
    printf("  --max-retries N               Max retries for unstable benchmarks (default: 0)\n\n");

    printf("Output:\n");
    printf("  --output-dir DIR              Output directory (default: ./benchmark_results)\n");
    printf("  --format json,csv,markdown    Output formats (default: all three)\n");
    printf("  --verbose / --quiet           Output verbosity\n\n");

    printf("Comparison:\n");
    printf("  --compare F1,F2[,...]         Compare existing JSON reports — produces\n");
    printf("                                a comparison.md / .csv (uses the same shared\n");
    printf("                                BenchmarkReport::compareReports as openvx-mark)\n\n");

    printf("Other:\n");
    printf("  --help                        Show this help\n");
}

std::vector<std::string> splitComma(const std::string& s) {
    std::vector<std::string> result;
    std::string token;
    for (char c : s) {
        if (c == ',') {
            if (!token.empty()) result.push_back(token);
            token.clear();
        } else { token += c; }
    }
    if (!token.empty()) result.push_back(token);
    return result;
}

bool parseArgs(int argc, char* argv[], BenchmarkConfig& config) {
    bool custom_res = false;
    uint32_t custom_w = 0, custom_h = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            // `--help` is a successful invocation by UNIX convention —
            // exit 0 directly so CI steps using `set -e` don't trip on
            // it. parseArgs returning `false` is reserved for actual
            // parse failures (mapped to exit 1 by main).
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--feature-set" && i + 1 < argc) {
            // PR1 only supports the "vision" feature set on the OpenCV
            // side; "framework" and "enhanced_vision" are intentionally
            // not registered yet (see the umbrella PR plan for follow-up).
            auto sets = splitComma(argv[++i]);
            config.feature_sets.clear();
            for (const auto& s : sets) {
                if (s == "vision" || s == "all" || s == "everything") {
                    config.feature_sets.push_back("vision");
                } else if (s == "enhanced_vision" || s == "framework") {
                    printf("WARNING: feature-set '%s' has no opencv-mark "
                           "implementation yet — skipping\n", s.c_str());
                } else {
                    printf("WARNING: unknown feature-set '%s'\n", s.c_str());
                }
            }
            if (config.feature_sets.empty()) {
                printf("ERROR: no valid feature-sets specified\n");
                return false;
            }
        } else if (arg == "--category" && i + 1 < argc) {
            config.categories = splitComma(argv[++i]);
        } else if (arg == "--kernel" && i + 1 < argc) {
            config.kernels = splitComma(argv[++i]);
        } else if (arg == "--resolution" && i + 1 < argc) {
            auto names = splitComma(argv[++i]);
            config.resolutions.clear();
            const auto& presets = getResolutionPresets();
            for (const auto& name : names) {
                auto it = presets.find(name);
                if (it != presets.end()) config.resolutions.push_back(it->second);
                else printf("WARNING: unknown resolution '%s'\n", name.c_str());
            }
            if (config.resolutions.empty()) {
                printf("ERROR: no valid resolutions specified\n");
                return false;
            }
        } else if (arg == "--width" && i + 1 < argc) {
            custom_w = static_cast<uint32_t>(atoi(argv[++i])); custom_res = true;
        } else if (arg == "--height" && i + 1 < argc) {
            custom_h = static_cast<uint32_t>(atoi(argv[++i])); custom_res = true;
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.iterations = atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            config.warmup = atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = static_cast<uint64_t>(atoll(argv[++i]));
        } else if (arg == "--output-dir" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--format" && i + 1 < argc) {
            auto fmts = splitComma(argv[++i]);
            config.output_json = false;
            config.output_csv = false;
            config.output_markdown = false;
            for (const auto& fmt : fmts) {
                if (fmt == "json") config.output_json = true;
                else if (fmt == "csv") config.output_csv = true;
                else if (fmt == "markdown" || fmt == "md") config.output_markdown = true;
            }
        } else if (arg == "--verbose") { config.verbose = true; config.quiet = false; }
        else if (arg == "--quiet")     { config.quiet = true; config.verbose = false; }
        else if (arg == "--stability-threshold" && i + 1 < argc) {
            config.stability_threshold = atof(argv[++i]);
        } else if (arg == "--max-retries" && i + 1 < argc) {
            config.max_retries = atoi(argv[++i]);
        } else if (arg == "--compare" && i + 1 < argc) {
            config.compare_files = splitComma(argv[++i]);
        } else {
            printf("Unknown option: %s\n", arg.c_str());
            printUsage(argv[0]);
            return false;
        }
    }

    if (custom_res && custom_w > 0 && custom_h > 0) {
        std::string name = std::to_string(custom_w) + "x" + std::to_string(custom_h);
        config.resolutions = {{custom_w, custom_h, name}};
    }

    if (config.feature_sets.empty()) config.feature_sets = {"vision"};
    return true;
}

// Build a `BenchmarkCatalog` snapshot from the runner's registered
// case list — analogous to what `KernelRegistry::snapshot()` does in
// openvx-mark, but driven by the cv:: benchmark registration list
// instead of OpenVX kernel-enum probing. This preserves identical
// JSON schema (kernel_availability, feature_set_availability) across
// both binaries.
BenchmarkCatalog buildCatalog(const std::vector<opencv_mark::OpenCVBenchmarkCase>& cases) {
    BenchmarkCatalog cat;
    std::map<std::string, std::pair<int, int>> fs_map;     // available, total
    std::map<std::string, std::pair<int, int>> cat_map;    // available, total

    for (const auto& bc : cases) {
        cat.kernels.push_back({bc.name, bc.feature_set, /*available=*/true});
        auto& f = fs_map[bc.feature_set]; f.first++; f.second++;
        auto& c = cat_map[bc.category];   c.first++; c.second++;
        cat.available_count++;
        cat.total_count++;
    }
    for (const auto& [fs, p] : fs_map) {
        cat.feature_sets.push_back({fs, p.first, p.second});
    }
    for (const auto& [cn, p] : cat_map) {
        cat.categories.push_back({cn, p.first, p.second});
    }
    return cat;
}

}  // namespace

int main(int argc, char* argv[]) {
    BenchmarkConfig config;
    if (!parseArgs(argc, argv, config)) return 1;

    // Comparison mode delegates to the shared reporter so the output
    // is byte-identical to running the same comparison from
    // openvx-mark — proves the schema parity by construction.
    if (!config.compare_files.empty()) {
        BenchmarkReport::compareReports(config.compare_files,
                                        config.output_dir + "/comparison");
        return 0;
    }

    printf("=============================================================\n");
    printf("  opencv-mark v%s (%s) — OpenCV Baseline Benchmark Suite\n",
           OPENCV_MARK_VERSION, GIT_COMMIT_SHA);
    printf("=============================================================\n\n");

    printf("[1/4] Initialising OpenCV context...\n");
    opencv_mark::OpenCVContext cv_ctx;
    printf("  Implementation: %s\n", cv_ctx.implementation().c_str());
    printf("  Version:        %d.%d\n", (cv_ctx.version() >> 8) & 0xFF,
           cv_ctx.version() & 0xFF);
    printf("  Build flags:    %s\n", cv_ctx.buildOptions().c_str());

    printf("\n[2/4] Collecting system information...\n");
    SystemInfo sys_info = collectSystemInfo();
    sys_info.vx_implementation = cv_ctx.implementation();
    sys_info.vx_vendor_id      = cv_ctx.vendorId();
    sys_info.vx_version        = cv_ctx.version();
    sys_info.vx_extensions     = cv_ctx.buildOptions();
    sys_info.benchmark_version = OPENCV_MARK_VERSION;
    sys_info.benchmark_git_commit = GIT_COMMIT_SHA;
    printf("  Host: %s (%s %s)\n", sys_info.hostname.c_str(),
           sys_info.os_name.c_str(), sys_info.os_version.c_str());
    printf("  CPU:  %s (%d cores)\n\n", sys_info.cpu_model.c_str(), sys_info.cpu_cores);

    printf("[3/4] Running benchmarks...\n");
    printf("  Resolutions: ");
    for (size_t i = 0; i < config.resolutions.size(); ++i) {
        if (i) printf(", ");
        printf("%s (%ux%u)", config.resolutions[i].name.c_str(),
               config.resolutions[i].width, config.resolutions[i].height);
    }
    printf("\n  Iterations: %d (warmup %d)\n", config.iterations, config.warmup);
    printf("  Mode:       graph (single cv:: call on pre-allocated buffers)\n\n");

    opencv_mark::OpenCVRunner runner(config);
    runner.addCases(opencv_mark::registerCvFilterBenchmarks());
    runner.addCases(opencv_mark::registerCvColorBenchmarks());
    runner.addCases(opencv_mark::registerCvGeometricBenchmarks());

    // Now we know how many kernels are registered; reflect that into
    // the SystemInfo so the JSON's `openvx.num_kernels` field carries
    // a meaningful value for opencv-mark reports too.
    cv_ctx.setNumKernels(static_cast<uint32_t>(runner.caseCount()));
    sys_info.vx_num_kernels = cv_ctx.numKernels();

    auto results = runner.runAll();

    printf("\n[4/4] Generating reports...\n");
    BenchmarkCatalog catalog = buildCatalog(runner.cases());
    BenchmarkReport report(sys_info, config, catalog);
    report.generate(results);

    int total = static_cast<int>(results.size());
    int passed = 0, failed = 0, skipped = 0;
    for (const auto& r : results) {
        if (!r.supported) ++skipped;
        else if (!r.verified) ++failed;
        else ++passed;
    }
    printf("\n=============================================================\n");
    printf("  Summary: %d total | %d passed | %d skipped | %d failed\n",
           total, passed, skipped, failed);
    auto scores = BenchmarkReport::computeScores(results);
    if (scores.vision_count > 0) {
        printf("  OpenCV Vision Score: %.2f MP/s (%d benchmarks)\n",
               scores.overall_vision_score, scores.vision_count);
    }
    printf("=============================================================\n");
    return 0;
}
