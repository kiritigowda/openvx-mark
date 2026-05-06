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
#include <VX/vx.h>
#include <VX/vx_nodes.h>
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
        cases.push_back(bc);
    }

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
        cases.push_back(bc);
    }

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
        cases.push_back(bc);
    }

    return cases;
}
