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

std::vector<BenchmarkCase> registerStatisticalBenchmarks()
{
    std::vector<BenchmarkCase> cases;

    // ---- Histogram ----
    {
        BenchmarkCase bc;
        bc.name        = "Histogram";
        bc.category    = "statistical";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_HISTOGRAM;
        bc.required_kernels = {VX_KERNEL_HISTOGRAM};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_distribution distribution = tracker.trackDistribution(
                gen.createDistribution(ctx, 256, 0, 256));
            vx_node node = vxHistogramNode(graph, input, distribution);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // ---- EqualizeHist ----
    {
        BenchmarkCase bc;
        bc.name        = "EqualizeHist";
        bc.category    = "statistical";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_EQUALIZE_HISTOGRAM;
        bc.required_kernels = {VX_KERNEL_EQUALIZE_HISTOGRAM};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_node node = vxEqualizeHistNode(graph, input, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // ---- MeanStdDev ----
    {
        BenchmarkCase bc;
        bc.name        = "MeanStdDev";
        bc.category    = "statistical";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_MEAN_STDDEV;
        bc.required_kernels = {VX_KERNEL_MEAN_STDDEV};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_float32 mean_val = 0.0f;
            vx_scalar mean_scalar = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_FLOAT32, &mean_val));
            vx_float32 stddev_val = 0.0f;
            vx_scalar stddev_scalar = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_FLOAT32, &stddev_val));
            vx_node node = vxMeanStdDevNode(graph, input, mean_scalar, stddev_scalar);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // ---- MinMaxLoc ----
    {
        BenchmarkCase bc;
        bc.name        = "MinMaxLoc";
        bc.category    = "statistical";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_MINMAXLOC;
        bc.required_kernels = {VX_KERNEL_MINMAXLOC};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));

            vx_uint8 min_init = 0;
            vx_scalar minVal = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_UINT8, &min_init));
            vx_uint8 max_init = 0;
            vx_scalar maxVal = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_UINT8, &max_init));

            vx_array minLoc = tracker.trackArray(
                vxCreateArray(ctx, VX_TYPE_COORDINATES2D, 1));
            vx_array maxLoc = tracker.trackArray(
                vxCreateArray(ctx, VX_TYPE_COORDINATES2D, 1));

            vx_uint32 count_init = 0;
            vx_scalar minCount = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_UINT32, &count_init));
            vx_scalar maxCount = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_UINT32, &count_init));

            vx_node node = vxMinMaxLocNode(graph, input,
                                           minVal, maxVal,
                                           minLoc, maxLoc,
                                           minCount, maxCount);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // ---- IntegralImage ----
    {
        BenchmarkCase bc;
        bc.name        = "IntegralImage";
        bc.category    = "statistical";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_INTEGRAL_IMAGE;
        bc.required_kernels = {VX_KERNEL_INTEGRAL_IMAGE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U32));
            vx_node node = vxIntegralImageNode(graph, input, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    return cases;
}
