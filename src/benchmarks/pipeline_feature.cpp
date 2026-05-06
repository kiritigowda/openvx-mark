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

std::vector<BenchmarkCase> registerFeaturePipelines() {
    std::vector<BenchmarkCase> cases;

    // 1. HistogramEqualize: ColorConvert(RGB->IYUV) -> ChannelExtract(Y) -> EqualizeHist
    {
        BenchmarkCase bc;
        bc.name = "HistogramEqualize";
        bc.category = "pipeline_feature";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_COLOR_CONVERT;
        bc.required_kernels = {VX_KERNEL_COLOR_CONVERT, VX_KERNEL_CHANNEL_EXTRACT,
                               VX_KERNEL_EQUALIZE_HISTOGRAM};
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

            // Output: U8 equalized
            vx_image output_eq = vxCreateImage(ctx, w, h, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output_eq) != VX_SUCCESS) return false;
            tracker.trackImage(output_eq);

            // Node 1: ColorConvert RGB -> IYUV
            vx_node node1 = vxColorConvertNode(graph, input_rgb, virt_iyuv);
            if (vxGetStatus((vx_reference)node1) != VX_SUCCESS) return false;
            tracker.trackNode(node1);

            // Node 2: ChannelExtract Y channel
            vx_node node2 = vxChannelExtractNode(graph, virt_iyuv, VX_CHANNEL_Y, virt_luma);
            if (vxGetStatus((vx_reference)node2) != VX_SUCCESS) return false;
            tracker.trackNode(node2);

            // Node 3: EqualizeHist
            vx_node node3 = vxEqualizeHistNode(graph, virt_luma, output_eq);
            if (vxGetStatus((vx_reference)node3) != VX_SUCCESS) return false;
            tracker.trackNode(node3);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // 2. HarrisTracker: ColorConvert(RGB->IYUV) -> ChannelExtract(Y) -> HarrisCorners
    {
        BenchmarkCase bc;
        bc.name = "HarrisTracker";
        bc.category = "pipeline_feature";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_COLOR_CONVERT;
        bc.required_kernels = {VX_KERNEL_COLOR_CONVERT, VX_KERNEL_CHANNEL_EXTRACT,
                               VX_KERNEL_HARRIS_CORNERS};
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

            // Harris parameters
            vx_float32 strength_val = 0.0005f;
            vx_float32 min_dist_val = 5.0f;
            vx_float32 sensitivity_val = 0.04f;
            vx_scalar strength_thresh = gen.createScalar(ctx, VX_TYPE_FLOAT32, &strength_val);
            vx_scalar min_distance = gen.createScalar(ctx, VX_TYPE_FLOAT32, &min_dist_val);
            vx_scalar sensitivity = gen.createScalar(ctx, VX_TYPE_FLOAT32, &sensitivity_val);
            tracker.trackScalar(strength_thresh);
            tracker.trackScalar(min_distance);
            tracker.trackScalar(sensitivity);

            // Output: array of keypoints
            vx_array corners = gen.createKeypointArray(ctx, DEFAULT_MAX_CORNERS);
            tracker.trackArray(corners);

            vx_size zero_size = 0;
            vx_scalar num_corners = gen.createScalar(ctx, VX_TYPE_SIZE, &zero_size);
            tracker.trackScalar(num_corners);

            // Node 1: ColorConvert RGB -> IYUV
            vx_node node1 = vxColorConvertNode(graph, input_rgb, virt_iyuv);
            if (vxGetStatus((vx_reference)node1) != VX_SUCCESS) return false;
            tracker.trackNode(node1);

            // Node 2: ChannelExtract Y channel
            vx_node node2 = vxChannelExtractNode(graph, virt_iyuv, VX_CHANNEL_Y, virt_luma);
            if (vxGetStatus((vx_reference)node2) != VX_SUCCESS) return false;
            tracker.trackNode(node2);

            // Node 3: HarrisCorners
            vx_node node3 = vxHarrisCornersNode(graph, virt_luma, strength_thresh,
                                                 min_distance, sensitivity,
                                                 3, 3, corners, num_corners);
            if (vxGetStatus((vx_reference)node3) != VX_SUCCESS) return false;
            tracker.trackNode(node3);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // 3. ThresholdedEdge: Sobel3x3 -> Magnitude -> ConvertDepth(S16->U8) -> Threshold
    {
        BenchmarkCase bc;
        bc.name = "ThresholdedEdge";
        bc.category = "pipeline_feature";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_SOBEL_3x3;
        bc.required_kernels = {VX_KERNEL_SOBEL_3x3, VX_KERNEL_MAGNITUDE,
                               VX_KERNEL_CONVERTDEPTH, VX_KERNEL_THRESHOLD};
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

            // Virtual intermediate: S16 magnitude
            vx_image virt_mag = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_S16);
            if (vxGetStatus((vx_reference)virt_mag) != VX_SUCCESS) return false;
            tracker.trackImage(virt_mag);

            // Virtual intermediate: U8 magnitude after depth conversion
            vx_image virt_mag_u8 = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)virt_mag_u8) != VX_SUCCESS) return false;
            tracker.trackImage(virt_mag_u8);

            // Output: U8 thresholded
            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;
            tracker.trackImage(output);

            // Shift scalar for ConvertDepth (value 0)
            vx_int32 shift_val = 0;
            vx_scalar shift = vxCreateScalar(ctx, VX_TYPE_INT32, &shift_val);
            if (vxGetStatus((vx_reference)shift) != VX_SUCCESS) return false;
            tracker.trackScalar(shift);

            // Binary threshold (value 100)
            vx_threshold thresh = gen.createBinaryThreshold(ctx, 100);
            if (vxGetStatus((vx_reference)thresh) != VX_SUCCESS) return false;
            tracker.trackThreshold(thresh);

            // Node 1: Sobel3x3 producing grad_x and grad_y
            vx_node node1 = vxSobel3x3Node(graph, input, virt_grad_x, virt_grad_y);
            if (vxGetStatus((vx_reference)node1) != VX_SUCCESS) return false;
            tracker.trackNode(node1);

            // Node 2: Magnitude(grad_x, grad_y) -> mag
            vx_node node2 = vxMagnitudeNode(graph, virt_grad_x, virt_grad_y, virt_mag);
            if (vxGetStatus((vx_reference)node2) != VX_SUCCESS) return false;
            tracker.trackNode(node2);

            // Node 3: ConvertDepth S16 -> U8
            vx_node node3 = vxConvertDepthNode(graph, virt_mag, virt_mag_u8,
                                                VX_CONVERT_POLICY_SATURATE, shift);
            if (vxGetStatus((vx_reference)node3) != VX_SUCCESS) return false;
            tracker.trackNode(node3);

            // Node 4: Threshold
            vx_node node4 = vxThresholdNode(graph, virt_mag_u8, thresh, output);
            if (vxGetStatus((vx_reference)node4) != VX_SUCCESS) return false;
            tracker.trackNode(node4);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    return cases;
}
