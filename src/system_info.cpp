#include "system_info.h"
#include <ctime>
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
