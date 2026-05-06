#include "benchmark_config.h"
#include "benchmark_context.h"
#include "benchmark_runner.h"
#include "benchmark_report.h"
#include "benchmark_stats.h"
#include "kernel_registry.h"
#include "system_info.h"
#include <VX/vx.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void printUsage(const char* prog) {
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("openvx-mark v%s (%s) — OpenVX Benchmark Suite\n\n",
           OPENVX_MARK_VERSION, GIT_COMMIT_SHA);

    printf("Benchmark Selection:\n");
    printf("  --all                         Run all benchmarks (default)\n");
    printf("  --feature-set SET[,SET,...]   Feature sets: vision,enhanced_vision,all\n");
    printf("                                (default: vision)\n");
    printf("  --category CAT[,CAT,...]      Filter by category (pixelwise,filters,color,\n");
    printf("                                geometric,statistical,multiscale,feature,\n");
    printf("                                extraction,tensor,misc,immediate,\n");
    printf("                                pipeline_vision,pipeline_feature)\n");
    printf("  --kernel NAME[,NAME,...]      Filter by kernel name\n");
    printf("  --mode graph|immediate|both   Execution mode (default: graph)\n");
    printf("  --skip-pipelines              Skip multi-node pipelines\n\n");

    printf("Resolution:\n");
    printf("  --resolution RES[,RES,...]    VGA,HD,FHD,4K,8K (default: VGA,FHD,4K)\n");
    printf("  --width W --height H          Custom resolution\n\n");

    printf("Timing:\n");
    printf("  --iterations N                Measurement iterations (default: 100)\n");
    printf("  --warmup N                    Warm-up iterations (default: 10)\n");
    printf("  --seed N                      PRNG seed (default: 42)\n");
    printf("  --stability-threshold N       CV%% threshold for stability warning (default: 15)\n");
    printf("  --max-retries N               Max retries for unstable benchmarks (default: 0)\n\n");

    printf("Output:\n");
    printf("  --output-dir DIR              Output directory (default: ./benchmark_results)\n");
    printf("  --format json,csv,markdown    Output formats (default: all three)\n");
    printf("  --verbose                     Verbose output\n");
    printf("  --quiet                       Minimal output\n\n");

    printf("Other:\n");
    printf("  --help                        Show this help\n");
}

static std::vector<std::string> splitComma(const std::string& s) {
    std::vector<std::string> result;
    std::string token;
    for (char c : s) {
        if (c == ',') {
            if (!token.empty()) result.push_back(token);
            token.clear();
        } else {
            token += c;
        }
    }
    if (!token.empty()) result.push_back(token);
    return result;
}

static bool parseArgs(int argc, char* argv[], BenchmarkConfig& config) {
    bool custom_res = false;
    uint32_t custom_w = 0, custom_h = 0;
    bool format_specified = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return false;
        } else if (arg == "--all") {
            config.feature_sets = {"vision", "enhanced_vision"};
        } else if (arg == "--category" && i + 1 < argc) {
            config.categories = splitComma(argv[++i]);
        } else if (arg == "--kernel" && i + 1 < argc) {
            config.kernels = splitComma(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "graph") config.mode = BenchmarkConfig::Mode::GRAPH;
            else if (mode == "immediate") config.mode = BenchmarkConfig::Mode::IMMEDIATE;
            else config.mode = BenchmarkConfig::Mode::BOTH;
        } else if (arg == "--skip-pipelines") {
            config.skip_pipelines = true;
        } else if (arg == "--feature-set" && i + 1 < argc) {
            auto sets = splitComma(argv[++i]);
            config.feature_sets.clear();
            for (const auto& s : sets) {
                if (s == "all") {
                    config.feature_sets = {"vision", "enhanced_vision"};
                    break;
                } else if (s == "vision" || s == "enhanced_vision") {
                    config.feature_sets.push_back(s);
                } else {
                    printf("WARNING: Unknown feature set '%s', skipping\n", s.c_str());
                }
            }
            if (config.feature_sets.empty()) {
                printf("ERROR: No valid feature sets specified\n");
                return false;
            }
        } else if (arg == "--resolution" && i + 1 < argc) {
            auto names = splitComma(argv[++i]);
            config.resolutions.clear();
            const auto& presets = getResolutionPresets();
            for (const auto& name : names) {
                auto it = presets.find(name);
                if (it != presets.end()) {
                    config.resolutions.push_back(it->second);
                } else {
                    printf("WARNING: Unknown resolution '%s', skipping\n", name.c_str());
                }
            }
            if (config.resolutions.empty()) {
                printf("ERROR: No valid resolutions specified\n");
                return false;
            }
        } else if (arg == "--width" && i + 1 < argc) {
            custom_w = static_cast<uint32_t>(atoi(argv[++i]));
            custom_res = true;
        } else if (arg == "--height" && i + 1 < argc) {
            custom_h = static_cast<uint32_t>(atoi(argv[++i]));
            custom_res = true;
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
            format_specified = true;
            config.output_json = false;
            config.output_csv = false;
            config.output_markdown = false;
            for (const auto& fmt : fmts) {
                if (fmt == "json") config.output_json = true;
                else if (fmt == "csv") config.output_csv = true;
                else if (fmt == "markdown" || fmt == "md") config.output_markdown = true;
            }
        } else if (arg == "--verbose") {
            config.verbose = true;
            config.quiet = false;
        } else if (arg == "--quiet") {
            config.quiet = true;
            config.verbose = false;
        } else if (arg == "--stability-threshold" && i + 1 < argc) {
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

    return true;
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config;
    if (!parseArgs(argc, argv, config)) {
        return 1;
    }

    // Handle comparison mode
    if (!config.compare_files.empty()) {
        BenchmarkReport::compareReports(config.compare_files, config.output_dir + "/comparison");
        return 0;
    }

    printf("=============================================================\n");
    printf("  openvx-mark v%s (%s) — OpenVX Benchmark Suite\n",
           OPENVX_MARK_VERSION, GIT_COMMIT_SHA);
    printf("=============================================================\n\n");

    // Create OpenVX context
    printf("[1/5] Creating OpenVX context...\n");
    BenchmarkContext context;
    if (!context.isValid()) {
        printf("FATAL: Failed to create OpenVX context\n");
        return 1;
    }
    printf("  Implementation: %s\n", context.implementation().c_str());
    printf("  Vendor ID:      0x%04X\n", context.vendorId());
    printf("  Version:        %d.%d\n", (context.version() >> 8) & 0xFF,
           context.version() & 0xFF);
    printf("  Kernels:        %u\n\n", context.numKernels());

    // Probe kernel availability
    printf("[2/5] Probing kernel availability...\n");
    KernelRegistry registry;
    registry.probe(context.handle());
    printf("  Available: %d / %d standard kernels\n", registry.availableCount(),
           registry.totalCount());

    // Feature set availability
    auto fs_summary = registry.featureSetSummary();
    for (const auto& s : fs_summary) {
        printf("    %-20s %d / %d\n", s.feature_set.c_str(), s.available, s.total);
    }

    if (config.verbose) {
        auto summary = registry.categorySummary();
        for (const auto& s : summary) {
            printf("    %-20s %d / %d\n", s.category.c_str(), s.available, s.total);
        }
    }
    printf("\n");

    // Collect system info
    printf("[3/5] Collecting system information...\n");
    SystemInfo sys_info = collectSystemInfo();
    sys_info.vx_implementation = context.implementation();
    sys_info.vx_vendor_id = context.vendorId();
    sys_info.vx_version = context.version();
    sys_info.vx_num_kernels = context.numKernels();
    sys_info.vx_extensions = context.extensions();
    sys_info.benchmark_version = OPENVX_MARK_VERSION;
    sys_info.benchmark_git_commit = GIT_COMMIT_SHA;
    printf("  Host: %s (%s %s)\n", sys_info.hostname.c_str(),
           sys_info.os_name.c_str(), sys_info.os_version.c_str());
    printf("  CPU:  %s (%d cores)\n\n", sys_info.cpu_model.c_str(), sys_info.cpu_cores);

    // Register and run benchmarks
    printf("[4/5] Running benchmarks...\n");
    printf("  Resolutions: ");
    for (size_t i = 0; i < config.resolutions.size(); i++) {
        if (i > 0) printf(", ");
        printf("%s (%ux%u)", config.resolutions[i].name.c_str(),
               config.resolutions[i].width, config.resolutions[i].height);
    }
    printf("\n");
    printf("  Iterations:  %d (warmup: %d)\n", config.iterations, config.warmup);
    printf("  Mode:        %s\n",
           config.mode == BenchmarkConfig::Mode::GRAPH ? "graph" :
           config.mode == BenchmarkConfig::Mode::IMMEDIATE ? "immediate" : "both");
    printf("  Feature set: ");
    for (size_t i = 0; i < config.feature_sets.size(); i++) {
        if (i > 0) printf(", ");
        printf("%s", config.feature_sets[i].c_str());
    }
    printf("\n\n");

    BenchmarkRunner runner(context, config, registry);

    // Register all benchmark categories
    runner.addCases(registerPixelwiseBenchmarks());
    runner.addCases(registerFilterBenchmarks());
    runner.addCases(registerColorBenchmarks());
    runner.addCases(registerGeometricBenchmarks());
    runner.addCases(registerStatisticalBenchmarks());
    runner.addCases(registerMultiscaleBenchmarks());
    runner.addCases(registerFeatureBenchmarks());
    runner.addCases(registerExtractionBenchmarks());
    runner.addCases(registerTensorBenchmarks());
    runner.addCases(registerMiscBenchmarks());

    if (config.mode == BenchmarkConfig::Mode::IMMEDIATE ||
        config.mode == BenchmarkConfig::Mode::BOTH) {
        runner.addCases(registerImmediateBenchmarks());
    }

    if (!config.skip_pipelines) {
        runner.addCases(registerVisionPipelines());
        runner.addCases(registerFeaturePipelines());
    }

    auto results = runner.runAll();

    // Generate reports
    printf("\n[5/5] Generating reports...\n");
    BenchmarkReport report(sys_info, config, registry);
    report.generate(results);

    // Summary
    int total = static_cast<int>(results.size());
    int passed = 0, skipped = 0, failed = 0;
    int stability_warnings = 0;
    for (const auto& r : results) {
        if (!r.supported) skipped++;
        else if (!r.verified) failed++;
        else passed++;
        if (r.stability_warning) stability_warnings++;
    }

    printf("\n=============================================================\n");
    printf("  Summary: %d total | %d passed | %d skipped | %d failed\n",
           total, passed, skipped, failed);

    // Feature 1: Composite Scores
    auto scores = BenchmarkReport::computeScores(results);
    if (scores.vision_count > 0) {
        printf("  OpenVX Vision Score: %.2f MP/s (%d benchmarks)\n",
               scores.overall_vision_score, scores.vision_count);
    }
    if (scores.enhanced_count > 0) {
        printf("  Enhanced Vision Score: %.2f MP/s (%d benchmarks)\n",
               scores.enhanced_vision_score, scores.enhanced_count);
    }

    // Feature 2: Stability warning count
    if (stability_warnings > 0) {
        printf("  Stability Warnings: %d benchmarks with CV%% > %.1f%%\n",
               stability_warnings, config.stability_threshold);
    }

    // Feature 7: Conformance per feature set
    auto conformance = BenchmarkReport::checkConformance(results, registry);
    for (const auto& cr : conformance) {
        if (cr.total > 0) {
            printf("  %s Conformance: %s (%d/%d",
                   cr.feature_set.c_str(),
                   cr.pass ? "PASS" : "FAIL",
                   cr.passed, cr.total);
            if (!cr.missing_kernels.empty()) {
                printf(" -- missing:");
                size_t show = std::min(cr.missing_kernels.size(), size_t(5));
                for (size_t i = 0; i < show; i++) {
                    printf(" %s", cr.missing_kernels[i].c_str());
                    if (i + 1 < show) printf(",");
                }
                if (cr.missing_kernels.size() > 5) {
                    printf(" +%zu more", cr.missing_kernels.size() - 5);
                }
            }
            printf(")\n");
        }
    }

    // Feature 4: Top-5 fastest and top-5 slowest — separated by feature set
    {
        std::map<std::string, std::vector<const BenchmarkResult*>> fs_passing;
        for (const auto& r : results) {
            if (r.supported && r.verified && r.megapixels_per_sec > 0) {
                fs_passing[r.feature_set].push_back(&r);
            }
        }

        for (const auto& [fs, passing] : fs_passing) {
            if (passing.size() < 2) continue;

            // Top-5 fastest (highest throughput)
            std::vector<const BenchmarkResult*> by_tp = passing;
            std::sort(by_tp.begin(), by_tp.end(),
                [](const BenchmarkResult* a, const BenchmarkResult* b) {
                    return a->megapixels_per_sec > b->megapixels_per_sec;
                });

            size_t top_n = std::min(by_tp.size(), size_t(5));
            printf("  %s Top-%zu Fastest:\n", fs.c_str(), top_n);
            for (size_t i = 0; i < top_n; i++) {
                printf("    %zu. %-28s %8.1f MP/s (%s, %s)\n",
                       i + 1, by_tp[i]->name.c_str(),
                       by_tp[i]->megapixels_per_sec,
                       by_tp[i]->mode.c_str(),
                       by_tp[i]->resolution_name.c_str());
            }

            // Top-5 slowest (highest latency)
            std::vector<const BenchmarkResult*> by_lat = passing;
            std::sort(by_lat.begin(), by_lat.end(),
                [](const BenchmarkResult* a, const BenchmarkResult* b) {
                    return a->wall_clock.median_ns > b->wall_clock.median_ns;
                });

            top_n = std::min(by_lat.size(), size_t(5));
            printf("  %s Top-%zu Slowest:\n", fs.c_str(), top_n);
            for (size_t i = 0; i < top_n; i++) {
                printf("    %zu. %-28s %8.3f ms (%s, %s)\n",
                       i + 1, by_lat[i]->name.c_str(),
                       by_lat[i]->wall_clock.median_ns / 1e6,
                       by_lat[i]->mode.c_str(),
                       by_lat[i]->resolution_name.c_str());
            }
        }
    }

    printf("=============================================================\n");

    return (failed > 0) ? 1 : 0;
}
