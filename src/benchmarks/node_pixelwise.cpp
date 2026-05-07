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
#include <VX/vx_nodes.h>
#include <VX/vxu.h>
#include <vector>

std::vector<BenchmarkCase> registerPixelwiseBenchmarks()
{
    std::vector<BenchmarkCase> cases;

    // ---- And ----
    {
        BenchmarkCase bc;
        bc.name        = "And";
        bc.category    = "pixelwise";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_AND;
        bc.required_kernels = {VX_KERNEL_AND};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image in1 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image in2 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image out = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_node node = vxAndNode(graph, in1, in2, out);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 0xF0);
            std::vector<uint8_t> b(N, 0x33);
            vx_image in1 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image in2 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, b.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vxuAnd(ctx, in1, in2, out);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[0] == 0x30);
            vxReleaseImage(&in1); vxReleaseImage(&in2); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- Or ----
    {
        BenchmarkCase bc;
        bc.name        = "Or";
        bc.category    = "pixelwise";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_OR;
        bc.required_kernels = {VX_KERNEL_OR};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image in1 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image in2 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image out = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_node node = vxOrNode(graph, in1, in2, out);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 0xF0);
            std::vector<uint8_t> b(N, 0x33);
            vx_image in1 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image in2 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, b.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vxuOr(ctx, in1, in2, out);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[0] == 0xF3);
            vxReleaseImage(&in1); vxReleaseImage(&in2); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- Xor ----
    {
        BenchmarkCase bc;
        bc.name        = "Xor";
        bc.category    = "pixelwise";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_XOR;
        bc.required_kernels = {VX_KERNEL_XOR};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image in1 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image in2 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image out = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_node node = vxXorNode(graph, in1, in2, out);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 0xF0);
            std::vector<uint8_t> b(N, 0x33);
            vx_image in1 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image in2 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, b.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vxuXor(ctx, in1, in2, out);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[0] == 0xC3);
            vxReleaseImage(&in1); vxReleaseImage(&in2); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- Not ----
    {
        BenchmarkCase bc;
        bc.name        = "Not";
        bc.category    = "pixelwise";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_NOT;
        bc.required_kernels = {VX_KERNEL_NOT};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_node node = vxNotNode(graph, input, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 0xA5);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vxuNot(ctx, in, out);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[0] == 0x5A);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- AbsDiff ----
    {
        BenchmarkCase bc;
        bc.name        = "AbsDiff";
        bc.category    = "pixelwise";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_ABSDIFF;
        bc.required_kernels = {VX_KERNEL_ABSDIFF};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image in1 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image in2 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image out = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_node node = vxAbsDiffNode(graph, in1, in2, out);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 200);
            std::vector<uint8_t> b(N, 100);
            vx_image in1 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image in2 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, b.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vxuAbsDiff(ctx, in1, in2, out);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[0] == 100);
            vxReleaseImage(&in1); vxReleaseImage(&in2); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- Add ----
    {
        BenchmarkCase bc;
        bc.name        = "Add";
        bc.category    = "pixelwise";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_ADD;
        bc.required_kernels = {VX_KERNEL_ADD};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image in1 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image in2 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image out = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_node node = vxAddNode(graph, in1, in2, VX_CONVERT_POLICY_SATURATE, out);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 100);
            std::vector<uint8_t> b(N, 50);
            vx_image in1 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image in2 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, b.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vxuAdd(ctx, in1, in2, VX_CONVERT_POLICY_SATURATE, out);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[0] == 150);
            vxReleaseImage(&in1); vxReleaseImage(&in2); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- Subtract ----
    {
        BenchmarkCase bc;
        bc.name        = "Subtract";
        bc.category    = "pixelwise";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_SUBTRACT;
        bc.required_kernels = {VX_KERNEL_SUBTRACT};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image in1 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image in2 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image out = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_node node = vxSubtractNode(graph, in1, in2, VX_CONVERT_POLICY_SATURATE, out);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 200);
            std::vector<uint8_t> b(N, 100);
            vx_image in1 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image in2 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, b.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vxuSubtract(ctx, in1, in2, VX_CONVERT_POLICY_SATURATE, out);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[0] == 100);
            vxReleaseImage(&in1); vxReleaseImage(&in2); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- Multiply ----
    {
        BenchmarkCase bc;
        bc.name        = "Multiply";
        bc.category    = "pixelwise";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_MULTIPLY;
        bc.required_kernels = {VX_KERNEL_MULTIPLY};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image in1 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image in2 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image out = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_float32 scale_val = 1.0f;
            vx_scalar scale = tracker.trackScalar(
                vxCreateScalar(ctx, VX_TYPE_FLOAT32, &scale_val));
            vx_node node = vxMultiplyNode(graph, in1, in2, scale,
                                          VX_CONVERT_POLICY_SATURATE,
                                          VX_ROUND_POLICY_TO_ZERO, out);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 10);
            std::vector<uint8_t> b(N, 5);
            vx_image in1 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image in2 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, b.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vxuMultiply(ctx, in1, in2, 1.0f, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_ZERO, out);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[0] == 50);
            vxReleaseImage(&in1); vxReleaseImage(&in2); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

#if OPENVX_HAS_1_3
    // ---- Min ----
    {
        BenchmarkCase bc;
        bc.name        = "Min";
        bc.category    = "pixelwise";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_MIN;
        bc.required_kernels = {VX_KERNEL_MIN};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image in1 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image in2 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image out = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_MIN);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)in1);
            vxSetParameterByIndex(node, 1, (vx_reference)in2);
            vxSetParameterByIndex(node, 2, (vx_reference)out);
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 100);
            std::vector<uint8_t> b(N, 150);
            vx_image in1 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image in2 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, b.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_graph g = vxCreateGraph(ctx);
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_MIN);
            vx_node n = vxCreateGenericNode(g, k);
            vxReleaseKernel(&k);
            vxSetParameterByIndex(n, 0, (vx_reference)in1);
            vxSetParameterByIndex(n, 1, (vx_reference)in2);
            vxSetParameterByIndex(n, 2, (vx_reference)out);
            vxVerifyGraph(g);
            vxProcessGraph(g);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[0] == 100);
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseImage(&in1); vxReleaseImage(&in2); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- Max ----
    {
        BenchmarkCase bc;
        bc.name        = "Max";
        bc.category    = "pixelwise";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_MAX;
        bc.required_kernels = {VX_KERNEL_MAX};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image in1 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image in2 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image out = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_MAX);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)in1);
            vxSetParameterByIndex(node, 1, (vx_reference)in2);
            vxSetParameterByIndex(node, 2, (vx_reference)out);
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 100);
            std::vector<uint8_t> b(N, 150);
            vx_image in1 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image in2 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, b.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_graph g = vxCreateGraph(ctx);
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_MAX);
            vx_node n = vxCreateGenericNode(g, k);
            vxReleaseKernel(&k);
            vxSetParameterByIndex(n, 0, (vx_reference)in1);
            vxSetParameterByIndex(n, 1, (vx_reference)in2);
            vxSetParameterByIndex(n, 2, (vx_reference)out);
            vxVerifyGraph(g);
            vxProcessGraph(g);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[0] == 150);
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseImage(&in1); vxReleaseImage(&in2); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }
#endif

#if OPENVX_HAS_1_2
    // ---- Copy ----
    {
        BenchmarkCase bc;
        bc.name        = "Copy";
        bc.category    = "pixelwise";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_COPY;
        bc.required_kernels = {VX_KERNEL_COPY};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_node node = vxCopyNode(graph, (vx_reference)input, (vx_reference)output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 42);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = vxCopyNode(g, (vx_reference)in, (vx_reference)out);
            vxVerifyGraph(g); vxProcessGraph(g);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[0] == 42);
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }
#endif

    return cases;
}
