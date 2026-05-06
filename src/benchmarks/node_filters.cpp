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

std::vector<BenchmarkCase> registerFilterBenchmarks() {
    std::vector<BenchmarkCase> cases;

    // Box3x3 — U8 input, U8 output
    {
        BenchmarkCase bc;
        bc.name = "Box3x3";
        bc.category = "filters";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_BOX_3x3;
        bc.required_kernels = {VX_KERNEL_BOX_3x3};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen,
                            ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;
            tracker.trackImage(output);

            vx_node node = vxBox3x3Node(graph, input, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // Gaussian3x3 — U8 input, U8 output
    {
        BenchmarkCase bc;
        bc.name = "Gaussian3x3";
        bc.category = "filters";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_GAUSSIAN_3x3;
        bc.required_kernels = {VX_KERNEL_GAUSSIAN_3x3};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen,
                            ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;
            tracker.trackImage(output);

            vx_node node = vxGaussian3x3Node(graph, input, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // Median3x3 — U8 input, U8 output
    {
        BenchmarkCase bc;
        bc.name = "Median3x3";
        bc.category = "filters";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_MEDIAN_3x3;
        bc.required_kernels = {VX_KERNEL_MEDIAN_3x3};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen,
                            ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;
            tracker.trackImage(output);

            vx_node node = vxMedian3x3Node(graph, input, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // Erode3x3 — U8 input, U8 output
    {
        BenchmarkCase bc;
        bc.name = "Erode3x3";
        bc.category = "filters";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_ERODE_3x3;
        bc.required_kernels = {VX_KERNEL_ERODE_3x3};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen,
                            ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;
            tracker.trackImage(output);

            vx_node node = vxErode3x3Node(graph, input, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // Dilate3x3 — U8 input, U8 output
    {
        BenchmarkCase bc;
        bc.name = "Dilate3x3";
        bc.category = "filters";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_DILATE_3x3;
        bc.required_kernels = {VX_KERNEL_DILATE_3x3};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen,
                            ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;
            tracker.trackImage(output);

            vx_node node = vxDilate3x3Node(graph, input, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // Sobel3x3 — U8 input, S16 output_x and output_y
    {
        BenchmarkCase bc;
        bc.name = "Sobel3x3";
        bc.category = "filters";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_SOBEL_3x3;
        bc.required_kernels = {VX_KERNEL_SOBEL_3x3};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen,
                            ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            vx_image output_x = vxCreateImage(ctx, width, height, VX_DF_IMAGE_S16);
            if (vxGetStatus((vx_reference)output_x) != VX_SUCCESS) return false;
            tracker.trackImage(output_x);

            vx_image output_y = vxCreateImage(ctx, width, height, VX_DF_IMAGE_S16);
            if (vxGetStatus((vx_reference)output_y) != VX_SUCCESS) return false;
            tracker.trackImage(output_y);

            vx_node node = vxSobel3x3Node(graph, input, output_x, output_y);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // CustomConvolution — U8 input, U8 output, 3x3 convolution kernel
    {
        BenchmarkCase bc;
        bc.name = "CustomConvolution";
        bc.category = "filters";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_CUSTOM_CONVOLUTION;
        bc.required_kernels = {VX_KERNEL_CUSTOM_CONVOLUTION};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen,
                            ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;
            tracker.trackImage(output);

            vx_convolution conv = gen.createConvolution3x3(ctx);
            if (vxGetStatus((vx_reference)conv) != VX_SUCCESS) return false;
            tracker.trackConvolution(conv);

            vx_node node = vxConvolveNode(graph, input, conv, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // NonLinearFilter — U8 input, U8 output, median mode with mask
    {
        BenchmarkCase bc;
        bc.name = "NonLinearFilter";
        bc.category = "filters";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_NON_LINEAR_FILTER;
        bc.required_kernels = {VX_KERNEL_NON_LINEAR_FILTER};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen,
                            ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;
            tracker.trackImage(output);

            vx_matrix mask = gen.createNonLinearMask(ctx);
            if (vxGetStatus((vx_reference)mask) != VX_SUCCESS) return false;
            tracker.trackMatrix(mask);

            vx_node node = vxNonLinearFilterNode(graph, VX_NONLINEAR_FILTER_MEDIAN,
                                                 input, mask, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    return cases;
}
