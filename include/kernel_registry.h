#ifndef KERNEL_REGISTRY_H
#define KERNEL_REGISTRY_H

#include "benchmark_catalog.h"
#include <VX/vx.h>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct KernelInfo {
    vx_enum kernel_enum;
    std::string name;
    std::string display_name;
    std::string category;
    std::string feature_set;  // "vision" or "enhanced_vision"
    bool available = false;
};

class KernelRegistry {
public:
    // Probe all standard kernels for availability
    void probe(vx_context context);

    bool isAvailable(vx_enum kernel_enum) const;
    bool allAvailable(const std::vector<vx_enum>& enums) const;
    const KernelInfo* getInfo(vx_enum kernel_enum) const;

    const std::map<vx_enum, KernelInfo>& allKernels() const { return kernels_; }
    int availableCount() const;
    int totalCount() const;

    // Get a summary of available/unavailable kernels by category
    struct CategorySummary {
        std::string category;
        int available;
        int total;
    };
    std::vector<CategorySummary> categorySummary() const;

    // Get a summary of available/unavailable kernels by feature set
    struct FeatureSetSummary {
        std::string feature_set;
        int available;
        int total;
    };
    std::vector<FeatureSetSummary> featureSetSummary() const;

    // Build an implementation-agnostic catalog snapshot for the
    // shared `bench_core` reporter (JSON / Markdown / cross-vendor
    // comparison). See benchmark_catalog.h for the rationale.
    BenchmarkCatalog snapshot() const;

private:
    void registerKernel(vx_enum e, const std::string& name,
                        const std::string& display, const std::string& cat,
                        const std::string& feature_set);
    void initCatalog();

    std::map<vx_enum, KernelInfo> kernels_;
};

#endif // KERNEL_REGISTRY_H
