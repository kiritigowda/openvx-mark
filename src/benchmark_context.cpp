#include "benchmark_context.h"
#include <cstdio>
#include <cstring>
#include <vector>

void VX_CALLBACK BenchmarkContext::logCallback(vx_context /*context*/, vx_reference /*ref*/,
                                               vx_status status, const vx_char string[]) {
    if (status == VX_SUCCESS) {
        printf("[VX LOG] %s\n", string);
    } else {
        printf("[VX LOG] status=%d: %s\n", status, string);
    }
}

BenchmarkContext::BenchmarkContext() {
    context_ = vxCreateContext();
    if (vxGetStatus((vx_reference)context_) != VX_SUCCESS) {
        printf("ERROR: vxCreateContext() failed\n");
        context_ = nullptr;
        return;
    }

    // Register log callback
    vxRegisterLogCallback(context_, logCallback, vx_false_e);

    // Enable performance counters
    vxDirective((vx_reference)context_, VX_DIRECTIVE_ENABLE_PERFORMANCE);

    // Query vendor info
    queryVendorInfo();
}

BenchmarkContext::~BenchmarkContext() {
    if (context_) {
        vxReleaseContext(&context_);
    }
}

void BenchmarkContext::queryVendorInfo() {
    if (!context_) return;

    vxQueryContext(context_, VX_CONTEXT_VENDOR_ID, &vendor_id_, sizeof(vendor_id_));
    vxQueryContext(context_, VX_CONTEXT_VERSION, &version_, sizeof(version_));

    vx_char impl[VX_MAX_IMPLEMENTATION_NAME];
    memset(impl, 0, sizeof(impl));
    if (vxQueryContext(context_, VX_CONTEXT_IMPLEMENTATION, impl, sizeof(impl)) == VX_SUCCESS) {
        implementation_ = impl;
    }

    vxQueryContext(context_, VX_CONTEXT_UNIQUE_KERNELS, &num_kernels_, sizeof(num_kernels_));

    vx_size ext_size = 0;
    vxQueryContext(context_, VX_CONTEXT_EXTENSIONS_SIZE, &ext_size, sizeof(ext_size));
    if (ext_size > 0) {
        std::vector<char> ext_buf(ext_size + 1, 0);
        if (vxQueryContext(context_, VX_CONTEXT_EXTENSIONS, ext_buf.data(), ext_size) == VX_SUCCESS) {
            extensions_ = ext_buf.data();
        }
    }
}
