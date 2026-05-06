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
#include "benchmark_config.h"
#include <VX/vx.h>
#include <VX/vx_nodes.h>
#include <vector>

std::vector<BenchmarkCase> registerFeatureBenchmarks() {
    std::vector<BenchmarkCase> cases;

    // CannyEdgeDetector — U8 input, U8 output
    {
        BenchmarkCase bc;
        bc.name = "CannyEdgeDetector";
        bc.category = "feature";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_CANNY_EDGE_DETECTOR;
        bc.required_kernels = {VX_KERNEL_CANNY_EDGE_DETECTOR};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;
            tracker.trackImage(output);

            vx_threshold hyst = gen.createRangeThreshold(ctx, 80, 100);
            if (vxGetStatus((vx_reference)hyst) != VX_SUCCESS) return false;
            tracker.trackThreshold(hyst);

            vx_node node = vxCannyEdgeDetectorNode(graph, input, hyst, 3, VX_NORM_L1, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // HarrisCorners — U8 input
    {
        BenchmarkCase bc;
        bc.name = "HarrisCorners";
        bc.category = "feature";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_HARRIS_CORNERS;
        bc.required_kernels = {VX_KERNEL_HARRIS_CORNERS};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            vx_float32 strength_val = 0.0005f;
            vx_float32 min_dist_val = 5.0f;
            vx_float32 sensitivity_val = 0.04f;
            vx_scalar strength_thresh = gen.createScalar(ctx, VX_TYPE_FLOAT32, &strength_val);
            vx_scalar min_distance = gen.createScalar(ctx, VX_TYPE_FLOAT32, &min_dist_val);
            vx_scalar sensitivity = gen.createScalar(ctx, VX_TYPE_FLOAT32, &sensitivity_val);
            tracker.trackScalar(strength_thresh);
            tracker.trackScalar(min_distance);
            tracker.trackScalar(sensitivity);

            vx_array corners = gen.createKeypointArray(ctx, DEFAULT_MAX_CORNERS);
            tracker.trackArray(corners);

            vx_size zero_size = 0;
            vx_scalar num_corners = gen.createScalar(ctx, VX_TYPE_SIZE, &zero_size);
            tracker.trackScalar(num_corners);

            vx_node node = vxHarrisCornersNode(graph, input, strength_thresh,
                                               min_distance, sensitivity,
                                               3, 3, corners, num_corners);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // FastCorners — U8 input
    {
        BenchmarkCase bc;
        bc.name = "FastCorners";
        bc.category = "feature";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_FAST_CORNERS;
        bc.required_kernels = {VX_KERNEL_FAST_CORNERS};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            vx_float32 strength_val = 50.0f;
            vx_scalar strength_thresh = gen.createScalar(ctx, VX_TYPE_FLOAT32, &strength_val);
            tracker.trackScalar(strength_thresh);

            vx_array corners = gen.createKeypointArray(ctx, DEFAULT_MAX_KEYPOINTS);
            tracker.trackArray(corners);

            vx_size zero_size = 0;
            vx_scalar num_corners = gen.createScalar(ctx, VX_TYPE_SIZE, &zero_size);
            tracker.trackScalar(num_corners);

            vx_node node = vxFastCornersNode(graph, input, strength_thresh,
                                             vx_true_e, corners, num_corners);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // OpticalFlowPyrLK — multi-node pipeline with gaussian pyramids
    {
        BenchmarkCase bc;
        bc.name = "OpticalFlowPyrLK";
        bc.category = "feature";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_OPTICAL_FLOW_PYR_LK;
        bc.required_kernels = {VX_KERNEL_OPTICAL_FLOW_PYR_LK, VX_KERNEL_GAUSSIAN_PYRAMID};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            // Create two filled input images
            vx_image input1 = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            vx_image input2 = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(input1);
            tracker.trackImage(input2);

            // Create pyramids
            vx_pyramid pyr1 = gen.createPyramid(ctx, DEFAULT_PYRAMID_LEVELS,
                                                 VX_SCALE_PYRAMID_HALF,
                                                 width, height, VX_DF_IMAGE_U8);
            vx_pyramid pyr2 = gen.createPyramid(ctx, DEFAULT_PYRAMID_LEVELS,
                                                 VX_SCALE_PYRAMID_HALF,
                                                 width, height, VX_DF_IMAGE_U8);
            tracker.trackPyramid(pyr1);
            tracker.trackPyramid(pyr2);

            // Build pyramids using gaussian pyramid nodes first
            vx_node pyr_node1 = vxGaussianPyramidNode(graph, input1, pyr1);
            vx_node pyr_node2 = vxGaussianPyramidNode(graph, input2, pyr2);
            tracker.trackNode(pyr_node1);
            tracker.trackNode(pyr_node2);

            // Create point arrays — pass old_points as new_points_estimates
            // (same pattern as apps/optical_flow when use_initial_estimate is false)
            vx_array old_points = vxCreateArray(ctx, VX_TYPE_KEYPOINT, DEFAULT_OPTFLOW_POINTS);
            vx_array new_points = vxCreateArray(ctx, VX_TYPE_KEYPOINT, DEFAULT_OPTFLOW_POINTS);
            tracker.trackArray(old_points);
            tracker.trackArray(new_points);

            // Fill old_points with keypoints spread across the image
            vx_keypoint_t pts[100];
            for (int i = 0; i < 100; i++) {
                pts[i].x = (i % 10) * (width / 10);
                pts[i].y = (i / 10) * (height / 10);
                pts[i].strength = 1.0f;
                pts[i].tracking_status = 1;
                pts[i].scale = 0.0f;
                pts[i].orientation = 0.0f;
                pts[i].error = 0.0f;
            }
            vxAddArrayItems(old_points, 100, pts, sizeof(vx_keypoint_t));

            vx_float32 eps_val = 0.01f;
            vx_scalar epsilon = gen.createScalar(ctx, VX_TYPE_FLOAT32, &eps_val);
            tracker.trackScalar(epsilon);

            vx_uint32 iter_val = 5;
            vx_scalar num_iterations = gen.createScalar(ctx, VX_TYPE_UINT32, &iter_val);
            tracker.trackScalar(num_iterations);

            vx_bool use_init = vx_false_e;
            vx_scalar use_initial_estimate = gen.createScalar(ctx, VX_TYPE_BOOL, &use_init);
            tracker.trackScalar(use_initial_estimate);

            vx_node node = vxOpticalFlowPyrLKNode(graph, pyr1, pyr2,
                                                   old_points, old_points, new_points,
                                                   VX_TERM_CRITERIA_BOTH,
                                                   epsilon, num_iterations,
                                                   use_initial_estimate,
                                                   DEFAULT_OPTFLOW_WINSIZE);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    return cases;
}
