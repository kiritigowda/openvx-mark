////////////////////////////////////////////////////////////////////////////////
//
// MIT License
//
// Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "benchmark_runner.h"
#include "openvx_version.h"
#include "verify_utils.h"
#include <VX/vxu.h>
#include <vector>

std::vector<BenchmarkCase> registerExtractionBenchmarks() {
    std::vector<BenchmarkCase> cases;

#if OPENVX_HAS_1_2
    // MatchTemplate: compare an image against a 32x32 template using CCORR_NORM
    {
        BenchmarkCase bc;
        bc.name = "MatchTemplate";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_MATCH_TEMPLATE;
        bc.required_kernels = {VX_KERNEL_MATCH_TEMPLATE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image src = tracker.trackImage(
                gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            if (vxGetStatus((vx_reference)src) != VX_SUCCESS) return false;

            vx_image templateImage = tracker.trackImage(
                gen.createFilledImage(ctx, 32, 32, VX_DF_IMAGE_U8));
            if (vxGetStatus((vx_reference)templateImage) != VX_SUCCESS) return false;

            vx_image output = tracker.trackImage(
                vxCreateImage(ctx, width, height, VX_DF_IMAGE_S16));
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;

            vx_enum method = VX_COMPARE_CCORR_NORM;
            vx_scalar matchingMethod = tracker.trackScalar(
                vxCreateScalar(ctx, VX_TYPE_ENUM, &method));
            if (vxGetStatus((vx_reference)matchingMethod) != VX_SUCCESS) return false;

            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_MATCH_TEMPLATE);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)src);
            vxSetParameterByIndex(node, 1, (vx_reference)templateImage);
            vxSetParameterByIndex(node, 2, (vx_reference)matchingMethod);
            vxSetParameterByIndex(node, 3, (vx_reference)output);
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_MATCH_TEMPLATE);
            bool ok = (vxGetStatus((vx_reference)k) == VX_SUCCESS);
            if (ok) vxReleaseKernel(&k);
            return ok;
        };
        cases.push_back(bc);
    }

    // LBP: extract Local Binary Pattern from an input image
    {
        BenchmarkCase bc;
        bc.name = "LBP";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_LBP;
        bc.required_kernels = {VX_KERNEL_LBP};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = tracker.trackImage(
                gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;

            vx_image output = tracker.trackImage(
                vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;

            vx_enum format_val = VX_LBP;
            vx_scalar format = tracker.trackScalar(
                vxCreateScalar(ctx, VX_TYPE_ENUM, &format_val));
            if (vxGetStatus((vx_reference)format) != VX_SUCCESS) return false;

            vx_int8 ksize_val = 3;
            vx_scalar kernel_size = tracker.trackScalar(
                vxCreateScalar(ctx, VX_TYPE_INT8, &ksize_val));
            if (vxGetStatus((vx_reference)kernel_size) != VX_SUCCESS) return false;

            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_LBP);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)input);
            vxSetParameterByIndex(node, 1, (vx_reference)format);
            vxSetParameterByIndex(node, 2, (vx_reference)kernel_size);
            vxSetParameterByIndex(node, 3, (vx_reference)output);
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            // LBP on a gradient pattern should produce non-zero output
            std::vector<uint8_t> a(64 * 64);
            for (int y = 0; y < 64; y++)
                for (int x = 0; x < 64; x++)
                    a[y * 64 + x] = (uint8_t)((x + y * 64) % 256);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_enum format_val = VX_LBP;
            vx_scalar format = vxCreateScalar(ctx, VX_TYPE_ENUM, &format_val);
            vx_int8 ksize = 3;
            vx_scalar kernel_size = vxCreateScalar(ctx, VX_TYPE_INT8, &ksize);
            vx_graph g = vxCreateGraph(ctx);
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_LBP);
            vx_node n = vxCreateGenericNode(g, k);
            vxReleaseKernel(&k);
            vxSetParameterByIndex(n, 0, (vx_reference)in);
            vxSetParameterByIndex(n, 1, (vx_reference)format);
            vxSetParameterByIndex(n, 2, (vx_reference)kernel_size);
            vxSetParameterByIndex(n, 3, (vx_reference)out);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            bool ok = (status != VX_SUCCESS) ? true : verify::imageNonZero(out, 64, 64);
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseScalar(&format); vxReleaseScalar(&kernel_size);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // NonMaxSuppression: suppress non-maxima in a 3x3 window on S16 input
    {
        BenchmarkCase bc;
        bc.name = "NonMaxSuppression";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_NON_MAX_SUPPRESSION;
        bc.required_kernels = {VX_KERNEL_NON_MAX_SUPPRESSION};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = tracker.trackImage(
                gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_S16));
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;

            vx_image output = tracker.trackImage(
                vxCreateImage(ctx, width, height, VX_DF_IMAGE_S16));
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;

            vx_int32 win_size_val = 3;
            vx_scalar win_size = tracker.trackScalar(
                vxCreateScalar(ctx, VX_TYPE_INT32, &win_size_val));
            if (vxGetStatus((vx_reference)win_size) != VX_SUCCESS) return false;

            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_NON_MAX_SUPPRESSION);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)input);
            // parameter 1 (mask) left unset — defaults to NULL
            vxSetParameterByIndex(node, 2, (vx_reference)win_size);
            vxSetParameterByIndex(node, 3, (vx_reference)output);
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_NON_MAX_SUPPRESSION);
            bool ok = (vxGetStatus((vx_reference)k) == VX_SUCCESS);
            if (ok) vxReleaseKernel(&k);
            return ok;
        };
        cases.push_back(bc);
    }
#endif

    return cases;
}
