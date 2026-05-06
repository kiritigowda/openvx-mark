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

std::vector<BenchmarkCase> registerMiscBenchmarks()
{
    std::vector<BenchmarkCase> cases;

    // ---- Magnitude ----
    {
        BenchmarkCase bc;
        bc.name        = "Magnitude";
        bc.category    = "misc";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_MAGNITUDE;
        bc.required_kernels = {VX_KERNEL_MAGNITUDE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image grad_x = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_S16));
            vx_image grad_y = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_S16));
            vx_image mag    = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_S16));
            vx_node node = vxMagnitudeNode(graph, grad_x, grad_y, mag);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // ---- Phase ----
    {
        BenchmarkCase bc;
        bc.name        = "Phase";
        bc.category    = "misc";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_PHASE;
        bc.required_kernels = {VX_KERNEL_PHASE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image grad_x      = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_S16));
            vx_image grad_y      = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_S16));
            vx_image orientation = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_node node = vxPhaseNode(graph, grad_x, grad_y, orientation);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // ---- TableLookup ----
    {
        BenchmarkCase bc;
        bc.name        = "TableLookup";
        bc.category    = "misc";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_TABLE_LOOKUP;
        bc.required_kernels = {VX_KERNEL_TABLE_LOOKUP};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_lut lut = tracker.trackLUT(gen.createLUT(ctx));
            vx_node node = vxTableLookupNode(graph, input, lut, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // ---- Threshold_Binary ----
    {
        BenchmarkCase bc;
        bc.name        = "Threshold_Binary";
        bc.category    = "misc";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_THRESHOLD;
        bc.required_kernels = {VX_KERNEL_THRESHOLD};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_threshold thresh = tracker.trackThreshold(gen.createBinaryThreshold(ctx, 128));
            vx_node node = vxThresholdNode(graph, input, thresh, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // ---- Threshold_Range ----
    {
        BenchmarkCase bc;
        bc.name        = "Threshold_Range";
        bc.category    = "misc";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_THRESHOLD;
        bc.required_kernels = {VX_KERNEL_THRESHOLD};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_threshold thresh = tracker.trackThreshold(gen.createRangeThreshold(ctx, 80, 200));
            vx_node node = vxThresholdNode(graph, input, thresh, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // ---- WeightedAverage ----
    {
        BenchmarkCase bc;
        bc.name        = "WeightedAverage";
        bc.category    = "misc";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_WEIGHTED_AVERAGE;
        bc.required_kernels = {VX_KERNEL_WEIGHTED_AVERAGE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image img1   = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image img2   = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_float32 alpha_val = 0.5f;
            vx_scalar alpha = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_FLOAT32, &alpha_val));
            vx_node node = vxWeightedAverageNode(graph, img1, alpha, img2, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // ---- Select ----
    {
        BenchmarkCase bc;
        bc.name        = "Select";
        bc.category    = "misc";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_SELECT;
        bc.required_kernels = {VX_KERNEL_SELECT};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_bool cond_val = vx_true_e;
            vx_scalar condition = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_BOOL, &cond_val));
            vx_image true_img  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image false_img = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output    = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_SELECT);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)condition);
            vxSetParameterByIndex(node, 1, (vx_reference)true_img);
            vxSetParameterByIndex(node, 2, (vx_reference)false_img);
            vxSetParameterByIndex(node, 3, (vx_reference)output);
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    return cases;
}
