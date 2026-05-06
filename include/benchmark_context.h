#ifndef BENCHMARK_CONTEXT_H
#define BENCHMARK_CONTEXT_H

#include <VX/vx.h>
#include <cstdint>
#include <string>

class BenchmarkContext {
public:
    BenchmarkContext();
    ~BenchmarkContext();

    // Non-copyable
    BenchmarkContext(const BenchmarkContext&) = delete;
    BenchmarkContext& operator=(const BenchmarkContext&) = delete;

    vx_context handle() const { return context_; }

    // Vendor information
    uint16_t vendorId() const { return vendor_id_; }
    uint16_t version() const { return version_; }
    const std::string& implementation() const { return implementation_; }
    uint32_t numKernels() const { return num_kernels_; }
    const std::string& extensions() const { return extensions_; }

    bool isValid() const { return context_ != nullptr; }

private:
    static void VX_CALLBACK logCallback(vx_context context, vx_reference ref,
                                        vx_status status, const vx_char string[]);
    void queryVendorInfo();

    vx_context context_ = nullptr;
    uint16_t vendor_id_ = 0;
    uint16_t version_ = 0;
    std::string implementation_;
    uint32_t num_kernels_ = 0;
    std::string extensions_;
};

#endif // BENCHMARK_CONTEXT_H
