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

std::vector<BenchmarkCase> registerGeometricBenchmarks() {
    std::vector<BenchmarkCase> cases;

    // ScaleImage_Half: scale down by 2x using bilinear interpolation
    {
        BenchmarkCase bc;
        bc.name = "ScaleImage_Half";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_SCALE_IMAGE;
        bc.required_kernels = {VX_KERNEL_SCALE_IMAGE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(input);

            vx_image output_half = vxCreateImage(ctx, width / 2, height / 2, VX_DF_IMAGE_U8);
            tracker.trackImage(output_half);

            vx_node node = vxScaleImageNode(graph, input, output_half, VX_INTERPOLATION_BILINEAR);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // ScaleImage_Double: scale up by 2x using bilinear interpolation
    {
        BenchmarkCase bc;
        bc.name = "ScaleImage_Double";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_SCALE_IMAGE;
        bc.required_kernels = {VX_KERNEL_SCALE_IMAGE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(input);

            vx_image output_double = vxCreateImage(ctx, width * 2, height * 2, VX_DF_IMAGE_U8);
            tracker.trackImage(output_double);

            vx_node node = vxScaleImageNode(graph, input, output_double, VX_INTERPOLATION_BILINEAR);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // WarpAffine: apply an affine warp using bilinear interpolation
    {
        BenchmarkCase bc;
        bc.name = "WarpAffine";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_WARP_AFFINE;
        bc.required_kernels = {VX_KERNEL_WARP_AFFINE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(output);

            vx_matrix matrix = gen.createAffineMatrix(ctx);
            tracker.trackMatrix(matrix);

            vx_node node = vxWarpAffineNode(graph, input, matrix, VX_INTERPOLATION_BILINEAR, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // WarpPerspective: apply a perspective warp using bilinear interpolation
    {
        BenchmarkCase bc;
        bc.name = "WarpPerspective";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_WARP_PERSPECTIVE;
        bc.required_kernels = {VX_KERNEL_WARP_PERSPECTIVE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(output);

            vx_matrix matrix = gen.createPerspectiveMatrix(ctx);
            tracker.trackMatrix(matrix);

            vx_node node = vxWarpPerspectiveNode(graph, input, matrix, VX_INTERPOLATION_BILINEAR, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // Remap: apply a remap table using bilinear interpolation
    {
        BenchmarkCase bc;
        bc.name = "Remap";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_REMAP;
        bc.required_kernels = {VX_KERNEL_REMAP};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(output);

            vx_remap remap_table = gen.createRemap(ctx, width, height, width, height);
            tracker.trackRemap(remap_table);

            vx_node node = vxRemapNode(graph, input, remap_table, VX_INTERPOLATION_BILINEAR, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    return cases;
}
