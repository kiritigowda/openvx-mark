#include "system_info.h"
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/utsname.h>
#include <unistd.h>
#elif defined(__linux__)
#include <sys/utsname.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

SystemInfo collectSystemInfo() {
    SystemInfo info;

    // Timestamp
    time_t now = time(nullptr);
    struct tm* t = gmtime(&now);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", t);
    info.timestamp_iso8601 = buf;

    // Hostname
    char hostname[256] = {};
#ifdef _WIN32
    DWORD size = sizeof(hostname);
    GetComputerNameA(hostname, &size);
#else
    gethostname(hostname, sizeof(hostname));
#endif
    info.hostname = hostname;

    // OS info
#ifdef __APPLE__
    struct utsname uts;
    if (uname(&uts) == 0) {
        info.os_name = uts.sysname;
        info.os_version = uts.release;
    }

    // CPU model
    char cpu_brand[256] = {};
    size_t cpu_size = sizeof(cpu_brand);
    if (sysctlbyname("machdep.cpu.brand_string", cpu_brand, &cpu_size, nullptr, 0) == 0) {
        info.cpu_model = cpu_brand;
    }

    // Core count
    int ncores = 0;
    size_t ncores_size = sizeof(ncores);
    if (sysctlbyname("hw.ncpu", &ncores, &ncores_size, nullptr, 0) == 0) {
        info.cpu_cores = ncores;
    }

    // RAM
    int64_t memsize = 0;
    size_t memsize_size = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &memsize_size, nullptr, 0) == 0) {
        info.ram_bytes = static_cast<uint64_t>(memsize);
    }

#elif defined(__linux__)
    struct utsname uts;
    if (uname(&uts) == 0) {
        info.os_name = uts.sysname;
        info.os_version = uts.release;
    }

    // CPU model from /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                info.cpu_model = line.substr(pos + 2);
            }
            break;
        }
    }

    // Core count
    info.cpu_cores = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));

    // RAM from /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal") != std::string::npos) {
            uint64_t kb = 0;
            sscanf(line.c_str(), "MemTotal: %lu kB", &kb);
            info.ram_bytes = kb * 1024;
            break;
        }
    }

#elif defined(_WIN32)
    info.os_name = "Windows";
    OSVERSIONINFOA osvi = {};
    osvi.dwOSVersionInfoSize = sizeof(osvi);
    info.os_version = "Unknown";

    SYSTEM_INFO si;
    GetSystemInfo(&si);
    info.cpu_cores = static_cast<int>(si.dwNumberOfProcessors);

    MEMORYSTATUSEX memstat;
    memstat.dwLength = sizeof(memstat);
    if (GlobalMemoryStatusEx(&memstat)) {
        info.ram_bytes = memstat.ullTotalPhys;
    }
#endif

    return info;
}

// ----------------------------------------------------------------------------
// populateBuildInfo
//
// Reads the compile-time macros injected by the parent CMakeLists.txt:
//
//   BENCH_BUILD_TYPE        — CMAKE_BUILD_TYPE                ("Release")
//   BENCH_COMPILER_ID       — CMAKE_CXX_COMPILER_ID           ("GNU")
//   BENCH_COMPILER_VERSION  — CMAKE_CXX_COMPILER_VERSION      ("11.4.0")
//   BENCH_CXX_FLAGS         — CMAKE_CXX_FLAGS                 (project-wide)
//   BENCH_CXX_FLAGS_RELEASE — CMAKE_CXX_FLAGS_RELEASE         (per-config)
//   BENCH_TARGET_ARCH       — CMAKE_SYSTEM_PROCESSOR          ("x86_64")
//
// These are the build environment of the openvx-mark / opencv-mark binary
// itself — the OpenVX implementation library's build flags are separate
// runtime concerns and not visible here.
//
// $OMP_NUM_THREADS is read at startup so the JSON records what the user's
// shell told the kernel libs to do, independent of any --threads CLI knob
// (those two settings interact unpredictably across impls).
// ----------------------------------------------------------------------------
#ifndef BENCH_BUILD_TYPE
#define BENCH_BUILD_TYPE "Unknown"
#endif
#ifndef BENCH_COMPILER_ID
#define BENCH_COMPILER_ID "Unknown"
#endif
#ifndef BENCH_COMPILER_VERSION
#define BENCH_COMPILER_VERSION "Unknown"
#endif
#ifndef BENCH_CXX_FLAGS
#define BENCH_CXX_FLAGS ""
#endif
#ifndef BENCH_CXX_FLAGS_RELEASE
#define BENCH_CXX_FLAGS_RELEASE ""
#endif
#ifndef BENCH_TARGET_ARCH
#define BENCH_TARGET_ARCH "Unknown"
#endif

void populateBuildInfo(SystemInfo& info) {
    info.build_type = BENCH_BUILD_TYPE;
    info.compiler_id = BENCH_COMPILER_ID;
    info.compiler_version = BENCH_COMPILER_VERSION;
    info.cxx_flags = BENCH_CXX_FLAGS;
    info.cxx_flags_release = BENCH_CXX_FLAGS_RELEASE;
    info.target_arch = BENCH_TARGET_ARCH;

    const char* env_omp = std::getenv("OMP_NUM_THREADS");
    if (env_omp != nullptr) {
        info.omp_num_threads_env = env_omp;
    }
}
