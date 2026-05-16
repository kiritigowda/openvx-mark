#ifndef SYSTEM_INFO_H
#define SYSTEM_INFO_H

#include <cstdint>
#include <string>

struct SystemInfo {
    std::string hostname;
    std::string os_name;
    std::string os_version;
    std::string cpu_model;
    int cpu_cores = 0;
    uint64_t ram_bytes = 0;
    std::string timestamp_iso8601;

    // OpenVX context info
    std::string vx_implementation;
    uint16_t vx_vendor_id = 0;
    uint16_t vx_version = 0;
    uint32_t vx_num_kernels = 0;
    std::string vx_extensions;

    // Benchmark version tracking
    std::string benchmark_version;
    std::string benchmark_git_commit;

    // Build environment of the *benchmark binary itself* — not the
    // underlying impl. Surfaced into JSON `build` block so a reader can
    // tell at a glance whether a result was produced by an
    // optimisation-enabled binary or a Debug build. The corresponding
    // build flags of the OpenVX impl / OpenCV library are not exposed
    // here (they live in the impl's own runtime info).
    std::string build_type;          // CMAKE_BUILD_TYPE at configure time
    std::string compiler_id;         // e.g. "GNU", "Clang", "AppleClang"
    std::string compiler_version;    // e.g. "11.4.0"
    std::string cxx_flags;           // CMAKE_CXX_FLAGS as seen by the build
    std::string cxx_flags_release;   // CMAKE_CXX_FLAGS_RELEASE
    std::string target_arch;         // CMAKE_SYSTEM_PROCESSOR

    // Threading policy applied at run time — see BenchmarkConfig::threads.
    int requested_threads = 0;       // value of --threads (0 = unset / single)
    int opencv_threads = 0;          // cv::getNumThreads() at startup (opencv-mark only)
    int openmp_max_threads = 0;      // omp_get_max_threads() if OpenMP linked, else 0
    std::string omp_num_threads_env; // value of $OMP_NUM_THREADS at startup

    // Timer self-test result — fields populated only when `--validate-timing`
    // is requested; left zero otherwise so the JSON schema stays compact for
    // regular runs.
    bool timing_validated = false;
    double timer_resolution_ns = 0.0;     // measured smallest non-zero delta
    double timer_sleep_1ms_err_pct = 0.0; // |measured - 1ms| / 1ms × 100
    double timer_sleep_10ms_err_pct = 0.0;
    double timer_sleep_100ms_err_pct = 0.0;
};

SystemInfo collectSystemInfo();

// Compile-time build metadata, expanded into the SystemInfo by the
// per-binary main() so the parent CMake doesn't need to touch
// system_info.cpp. The defines themselves are injected via
// add_definitions() / target_compile_definitions() in the parent
// CMakeLists.txt — see SystemInfoHelper-defines comment there.
void populateBuildInfo(SystemInfo& info);

#endif // SYSTEM_INFO_H
