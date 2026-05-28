#include "benchmark_context.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Per-process dedup state for the VX log callback.
//
// Background: some drivers (notably AMD MIVisionX/AGO) emit a fresh
// [VX LOG] line every time vxVerifyGraph is called for an unsupported
// kernel parameter combination. The runner verifies once per (warmup +
// measured) iteration, so a single skipped benchmark can produce e.g.
// 5 identical "status=-14: ERROR: agoVerifyGraph: kernel ...:
// ago_kernel_cmd_validate failed (-14)" lines that visually swamp the
// bench output and obscure the actual numbers (see LaplacianPyramid_S16
// / LaplacianReconstruct_S16).
//
// We keep the first occurrence per-benchmark verbatim (signal preserved)
// and, when the same (status, string) pair fires again *within the same
// benchmark*, suppress it. The runner calls resetLogDedup() before each
// benchmark to flush state so each bench gets its own first-line copy.
//
// Thread-safety: VX log callbacks for a given context are delivered
// serially; the bench runner is single-threaded across cases. Simple
// statics suffice.
int g_last_status = VX_SUCCESS;
std::string g_last_text;
int g_suppressed_count = 0;

void flush_suppressed() {
    if (g_suppressed_count > 0) {
        printf("[VX LOG] (previous message repeated %d more time%s)\n",
               g_suppressed_count, g_suppressed_count == 1 ? "" : "s");
        g_suppressed_count = 0;
    }
}

} // namespace

void BenchmarkContext::resetLogDedup() {
    flush_suppressed();
    g_last_status = VX_SUCCESS;
    g_last_text.clear();
}

void VX_CALLBACK BenchmarkContext::logCallback(vx_context /*context*/, vx_reference /*ref*/,
                                               vx_status status, const vx_char string[]) {
    const std::string text = string ? string : "";

    if (status == g_last_status && !g_last_text.empty() && text == g_last_text) {
        ++g_suppressed_count;
        return;
    }

    flush_suppressed();

    if (status == VX_SUCCESS) {
        printf("[VX LOG] %s\n", text.c_str());
    } else {
        printf("[VX LOG] status=%d: %s\n", status, text.c_str());
    }
    g_last_status = status;
    g_last_text = text;
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
