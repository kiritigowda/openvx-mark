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

std::vector<BenchmarkCase> registerVisionPipelines() {
    std::vector<BenchmarkCase> cases;

    // 1. EdgeDetection: ColorConvert(RGB->IYUV) -> ChannelExtract(Y) -> Gaussian3x3 -> CannyEdgeDetector
    {
        BenchmarkCase bc;
        bc.name = "EdgeDetection";
        bc.category = "pipeline_vision";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_COLOR_CONVERT;
        bc.required_kernels = {VX_KERNEL_COLOR_CONVERT, VX_KERNEL_CHANNEL_EXTRACT,
                               VX_KERNEL_GAUSSIAN_3x3, VX_KERNEL_CANNY_EDGE_DETECTOR};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            // Width/height must be even for YUV formats
            uint32_t w = width & ~1u;
            uint32_t h = height & ~1u;
            if (w == 0 || h == 0) return false;

            // Input: RGB image
            vx_image input_rgb = gen.createFilledImage(ctx, w, h, VX_DF_IMAGE_RGB);
            if (vxGetStatus((vx_reference)input_rgb) != VX_SUCCESS) return false;
            tracker.trackImage(input_rgb);

            // Virtual intermediate: IYUV
            vx_image virt_iyuv = vxCreateVirtualImage(graph, w, h, VX_DF_IMAGE_IYUV);
            if (vxGetStatus((vx_reference)virt_iyuv) != VX_SUCCESS) return false;
            tracker.trackImage(virt_iyuv);

            // Virtual intermediate: U8 luma channel
            vx_image virt_luma = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)virt_luma) != VX_SUCCESS) return false;
            tracker.trackImage(virt_luma);

            // Virtual intermediate: U8 blurred
            vx_image virt_blurred = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)virt_blurred) != VX_SUCCESS) return false;
            tracker.trackImage(virt_blurred);

            // Output: U8 edges
            vx_image output_edges = vxCreateImage(ctx, w, h, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output_edges) != VX_SUCCESS) return false;
            tracker.trackImage(output_edges);

            // Threshold for Canny
            vx_threshold hyst = gen.createRangeThreshold(ctx, 80, 100);
            if (vxGetStatus((vx_reference)hyst) != VX_SUCCESS) return false;
            tracker.trackThreshold(hyst);

            // Node 1: ColorConvert RGB -> IYUV
            vx_node node1 = vxColorConvertNode(graph, input_rgb, virt_iyuv);
            if (vxGetStatus((vx_reference)node1) != VX_SUCCESS) return false;
            tracker.trackNode(node1);

            // Node 2: ChannelExtract Y channel
            vx_node node2 = vxChannelExtractNode(graph, virt_iyuv, VX_CHANNEL_Y, virt_luma);
            if (vxGetStatus((vx_reference)node2) != VX_SUCCESS) return false;
            tracker.trackNode(node2);

            // Node 3: Gaussian3x3
            vx_node node3 = vxGaussian3x3Node(graph, virt_luma, virt_blurred);
            if (vxGetStatus((vx_reference)node3) != VX_SUCCESS) return false;
            tracker.trackNode(node3);

            // Node 4: CannyEdgeDetector
            vx_node node4 = vxCannyEdgeDetectorNode(graph, virt_blurred, hyst,
                                                     3, VX_NORM_L1, output_edges);
            if (vxGetStatus((vx_reference)node4) != VX_SUCCESS) return false;
            tracker.trackNode(node4);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // 2. SobelMagnitudePhase: Sobel3x3 -> Magnitude + Phase
    {
        BenchmarkCase bc;
        bc.name = "SobelMagnitudePhase";
        bc.category = "pipeline_vision";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_SOBEL_3x3;
        bc.required_kernels = {VX_KERNEL_SOBEL_3x3, VX_KERNEL_MAGNITUDE, VX_KERNEL_PHASE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            // Input: U8
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            // Virtual intermediates: S16 gradients
            vx_image virt_grad_x = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_S16);
            if (vxGetStatus((vx_reference)virt_grad_x) != VX_SUCCESS) return false;
            tracker.trackImage(virt_grad_x);

            vx_image virt_grad_y = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_S16);
            if (vxGetStatus((vx_reference)virt_grad_y) != VX_SUCCESS) return false;
            tracker.trackImage(virt_grad_y);

            // Outputs: S16 magnitude, U8 phase
            vx_image output_mag = vxCreateImage(ctx, width, height, VX_DF_IMAGE_S16);
            if (vxGetStatus((vx_reference)output_mag) != VX_SUCCESS) return false;
            tracker.trackImage(output_mag);

            vx_image output_phase = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output_phase) != VX_SUCCESS) return false;
            tracker.trackImage(output_phase);

            // Node 1: Sobel3x3 producing grad_x and grad_y
            vx_node node1 = vxSobel3x3Node(graph, input, virt_grad_x, virt_grad_y);
            if (vxGetStatus((vx_reference)node1) != VX_SUCCESS) return false;
            tracker.trackNode(node1);

            // Node 2: Magnitude(grad_x, grad_y) -> mag_out
            vx_node node2 = vxMagnitudeNode(graph, virt_grad_x, virt_grad_y, output_mag);
            if (vxGetStatus((vx_reference)node2) != VX_SUCCESS) return false;
            tracker.trackNode(node2);

            // Node 3: Phase(grad_x, grad_y) -> phase_out
            vx_node node3 = vxPhaseNode(graph, virt_grad_x, virt_grad_y, output_phase);
            if (vxGetStatus((vx_reference)node3) != VX_SUCCESS) return false;
            tracker.trackNode(node3);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // 3. MorphologyOpen: Erode3x3 -> Dilate3x3
    {
        BenchmarkCase bc;
        bc.name = "MorphologyOpen";
        bc.category = "pipeline_vision";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_ERODE_3x3;
        bc.required_kernels = {VX_KERNEL_ERODE_3x3, VX_KERNEL_DILATE_3x3};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            // Input: U8
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            // Virtual intermediate: U8 eroded
            vx_image virt_eroded = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)virt_eroded) != VX_SUCCESS) return false;
            tracker.trackImage(virt_eroded);

            // Output: U8
            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;
            tracker.trackImage(output);

            // Node 1: Erode3x3
            vx_node node1 = vxErode3x3Node(graph, input, virt_eroded);
            if (vxGetStatus((vx_reference)node1) != VX_SUCCESS) return false;
            tracker.trackNode(node1);

            // Node 2: Dilate3x3
            vx_node node2 = vxDilate3x3Node(graph, virt_eroded, output);
            if (vxGetStatus((vx_reference)node2) != VX_SUCCESS) return false;
            tracker.trackNode(node2);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // 4. MorphologyClose: Dilate3x3 -> Erode3x3
    {
        BenchmarkCase bc;
        bc.name = "MorphologyClose";
        bc.category = "pipeline_vision";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_DILATE_3x3;
        bc.required_kernels = {VX_KERNEL_DILATE_3x3, VX_KERNEL_ERODE_3x3};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            // Input: U8
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            // Virtual intermediate: U8 dilated
            vx_image virt_dilated = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)virt_dilated) != VX_SUCCESS) return false;
            tracker.trackImage(virt_dilated);

            // Output: U8
            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;
            tracker.trackImage(output);

            // Node 1: Dilate3x3
            vx_node node1 = vxDilate3x3Node(graph, input, virt_dilated);
            if (vxGetStatus((vx_reference)node1) != VX_SUCCESS) return false;
            tracker.trackNode(node1);

            // Node 2: Erode3x3
            vx_node node2 = vxErode3x3Node(graph, virt_dilated, output);
            if (vxGetStatus((vx_reference)node2) != VX_SUCCESS) return false;
            tracker.trackNode(node2);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // 5. DualFilter: Box3x3 -> Median3x3
    {
        BenchmarkCase bc;
        bc.name = "DualFilter";
        bc.category = "pipeline_vision";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_BOX_3x3;
        bc.required_kernels = {VX_KERNEL_BOX_3x3, VX_KERNEL_MEDIAN_3x3};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            // Input: U8
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            // Virtual intermediate: U8 boxed
            vx_image virt_boxed = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)virt_boxed) != VX_SUCCESS) return false;
            tracker.trackImage(virt_boxed);

            // Output: U8
            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;
            tracker.trackImage(output);

            // Node 1: Box3x3
            vx_node node1 = vxBox3x3Node(graph, input, virt_boxed);
            if (vxGetStatus((vx_reference)node1) != VX_SUCCESS) return false;
            tracker.trackNode(node1);

            // Node 2: Median3x3
            vx_node node2 = vxMedian3x3Node(graph, virt_boxed, output);
            if (vxGetStatus((vx_reference)node2) != VX_SUCCESS) return false;
            tracker.trackNode(node2);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    return cases;
}
