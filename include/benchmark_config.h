#ifndef BENCHMARK_CONFIG_H
#define BENCHMARK_CONFIG_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>

struct Resolution {
    uint32_t width;
    uint32_t height;
    std::string name;
};

inline const std::map<std::string, Resolution>& getResolutionPresets() {
    static const std::map<std::string, Resolution> presets = {
        {"VGA",  {640,  480,  "VGA"}},
        {"HD",   {1280, 720,  "HD"}},
        {"FHD",  {1920, 1080, "FHD"}},
        {"4K",   {3840, 2160, "4K"}},
        {"8K",   {7680, 4320, "8K"}}
    };
    return presets;
}

struct BenchmarkConfig {
    // Resolutions to test
    std::vector<Resolution> resolutions = {
        {640, 480, "VGA"}, {1920, 1080, "FHD"}, {3840, 2160, "4K"}
    };

    // Timing
    int iterations = 100;
    int warmup = 10;
    uint64_t seed = 42;

    // Execution mode
    enum class Mode { GRAPH, IMMEDIATE, BOTH };
    Mode mode = Mode::GRAPH;

    // Filters
    std::vector<std::string> categories;
    std::vector<std::string> kernels;
    std::vector<std::string> feature_sets = {"vision"};  // default: vision only
    bool skip_pipelines = false;

    // Output
    std::string output_dir = "./benchmark_results";
    bool output_json = true;
    bool output_csv = true;
    bool output_markdown = true;

    // Verbosity
    bool verbose = false;
    bool quiet = false;

    // Stability gating
    double stability_threshold = 15.0;  // CV% threshold for stability warning
    int max_retries = 0;                // 0 = no retries

    // Comparison
    std::vector<std::string> compare_files;
};

// Default tensor dimensions for benchmarks
constexpr int DEFAULT_TENSOR_DIM = 256;
constexpr int DEFAULT_NUM_BINS = 256;
constexpr int DEFAULT_LUT_COUNT = 256;

// Pyramid defaults
constexpr int DEFAULT_PYRAMID_LEVELS = 4;
constexpr float DEFAULT_PYRAMID_SCALE = 0.5f;

// Feature detection defaults
constexpr int DEFAULT_MAX_CORNERS = 10000;
constexpr int DEFAULT_MAX_KEYPOINTS = 10000;

// Optical flow defaults
constexpr int DEFAULT_OPTFLOW_POINTS = 1000;
constexpr int DEFAULT_OPTFLOW_WINSIZE = 9;

#endif // BENCHMARK_CONFIG_H
