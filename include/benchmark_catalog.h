#ifndef BENCHMARK_CATALOG_H
#define BENCHMARK_CATALOG_H

// POD snapshot of the per-implementation kernel catalog used by the
// JSON / CSV / Markdown reporters and the cross-vendor comparator.
//
// Why a POD snapshot rather than a live registry pointer?
// `BenchmarkReport` lives in the implementation-agnostic `bench_core`
// static library so both `openvx-mark` and `opencv-mark` (and any future
// `<other>-mark` companion binary) can share JSON schema, comparison
// logic, and stats math. The OpenVX-specific `KernelRegistry` couldn't
// be exposed there without dragging in `<VX/vx.h>`. A flat POD snapshot
// decouples the reporter from the registry implementation: each
// implementation-mark binary populates its own snapshot from whatever
// kernel catalog it has (vx_enum probing for openvx-mark, list of
// registered cv:: benchmarks for opencv-mark, etc.) and hands it to
// the shared reporter.

#include <string>
#include <vector>

struct CatalogFeatureSetSummary {
    std::string feature_set;
    int available = 0;
    int total = 0;
};

struct CatalogCategorySummary {
    std::string category;
    int available = 0;
    int total = 0;
};

struct CatalogKernelEntry {
    std::string name;
    std::string feature_set;
    bool available = false;
};

struct BenchmarkCatalog {
    int available_count = 0;
    int total_count = 0;
    std::vector<CatalogFeatureSetSummary> feature_sets;
    std::vector<CatalogCategorySummary> categories;
    std::vector<CatalogKernelEntry> kernels;
};

#endif // BENCHMARK_CATALOG_H
