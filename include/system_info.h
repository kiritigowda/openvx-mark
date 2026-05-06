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
};

SystemInfo collectSystemInfo();

#endif // SYSTEM_INFO_H
