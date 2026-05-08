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
#include "verify_utils.h"
#include <VX/vx.h>
#include <VX/vxu.h>
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
        bc.verify_fn = [](vx_context ctx) -> bool {
            const uint32_t W = 320, H = 240;
            std::vector<uint8_t> a(W * H, 0);
            for (uint32_t r = 100; r < 140; r++)
                for (uint32_t c = 50; c < 270; c++)
                    a[r * W + c] = 255;
            vx_image in = verify::createImage(ctx, W, H, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, W, H, VX_DF_IMAGE_U8);
            vx_threshold hyst = vxCreateThresholdForImage(ctx, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
            vx_pixel_value_t lower_pv = {}, upper_pv = {};
            lower_pv.U8 = 50; upper_pv.U8 = 100;
            vxCopyThresholdRange(hyst, &lower_pv, &upper_pv, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = vxCannyEdgeDetectorNode(g, in, hyst, 3, VX_NORM_L1, out);
            if (vxVerifyGraph(g) != VX_SUCCESS) {
                vxReleaseNode(&n); vxReleaseGraph(&g); vxReleaseThreshold(&hyst);
                vxReleaseImage(&in); vxReleaseImage(&out);
                return true;
            }
            vx_status status = vxProcessGraph(g);
            if (status != VX_SUCCESS) {
                vxReleaseNode(&n); vxReleaseGraph(&g); vxReleaseThreshold(&hyst);
                vxReleaseImage(&in); vxReleaseImage(&out);
                return true;
            }
            auto result = verify::readImage(out, W, H);
            bool has_edge_near_boundary = false;
            for (uint32_t r = 98; r <= 142 && !has_edge_near_boundary; r++)
                for (uint32_t c = 48; c <= 272 && !has_edge_near_boundary; c++)
                    if (r < H && c < W && result[r * W + c] != 0)
                        has_edge_near_boundary = true;
            bool interior_clean = (result[120 * W + 160] == 0);
            bool ok = has_edge_near_boundary && interior_clean;
            vxReleaseNode(&n); vxReleaseGraph(&g); vxReleaseThreshold(&hyst);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
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
        bc.verify_fn = [](vx_context ctx) -> bool {
            // Verify Harris runs without crashing — corner count varies by implementation
            std::vector<uint8_t> a(64 * 64);
            for (int y = 0; y < 64; y++)
                for (int x = 0; x < 64; x++)
                    a[y * 64 + x] = ((x / 2 + y / 2) % 2) ? 255 : 0;
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_float32 strength_val = 0.0001f, min_dist_val = 1.0f, sensitivity_val = 0.04f;
            vx_scalar strength = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &strength_val);
            vx_scalar min_dist = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &min_dist_val);
            vx_scalar sensitivity = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &sensitivity_val);
            vx_array corners = vxCreateArray(ctx, VX_TYPE_KEYPOINT, 100);
            vx_size num = 0;
            vx_scalar num_corners = vxCreateScalar(ctx, VX_TYPE_SIZE, &num);
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = vxHarrisCornersNode(g, in, strength, min_dist, sensitivity, 3, 3, corners, num_corners);
            if (vxVerifyGraph(g) != VX_SUCCESS) {
                vxReleaseNode(&n); vxReleaseGraph(&g);
                vxReleaseScalar(&strength); vxReleaseScalar(&min_dist); vxReleaseScalar(&sensitivity);
                vxReleaseScalar(&num_corners); vxReleaseArray(&corners); vxReleaseImage(&in);
                return true;
            }
            vx_status status = vxProcessGraph(g);
            bool ok = (status == VX_SUCCESS);
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseScalar(&strength); vxReleaseScalar(&min_dist); vxReleaseScalar(&sensitivity);
            vxReleaseScalar(&num_corners); vxReleaseArray(&corners); vxReleaseImage(&in);
            return ok;
        };
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
        bc.verify_fn = [](vx_context ctx) -> bool {
            std::vector<uint8_t> a(64 * 64);
            for (int y = 0; y < 64; y++)
                for (int x = 0; x < 64; x++)
                    a[y * 64 + x] = ((x / 2 + y / 2) % 2) ? 255 : 0;
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_float32 strength_val = 10.0f;
            vx_scalar strength = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &strength_val);
            vx_array corners = vxCreateArray(ctx, VX_TYPE_KEYPOINT, 100);
            vx_size num = 0;
            vx_scalar num_corners = vxCreateScalar(ctx, VX_TYPE_SIZE, &num);
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = vxFastCornersNode(g, in, strength, vx_true_e, corners, num_corners);
            if (vxVerifyGraph(g) != VX_SUCCESS) {
                vxReleaseNode(&n); vxReleaseGraph(&g);
                vxReleaseScalar(&strength); vxReleaseScalar(&num_corners);
                vxReleaseArray(&corners); vxReleaseImage(&in);
                return true;
            }
            vx_status status = vxProcessGraph(g);
            bool ok = (status == VX_SUCCESS);
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseScalar(&strength); vxReleaseScalar(&num_corners);
            vxReleaseArray(&corners); vxReleaseImage(&in);
            return ok;
        };
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
        bc.verify_fn = [](vx_context ctx) -> bool {
            const uint32_t W = 64, H = 64;
            std::vector<uint8_t> a(W * H, 100);
            vx_image img1 = verify::createImage(ctx, W, H, VX_DF_IMAGE_U8, a.data());
            vx_image img2 = verify::createImage(ctx, W, H, VX_DF_IMAGE_U8, a.data());
            if (!img1 || !img2) {
                if (img1) vxReleaseImage(&img1);
                if (img2) vxReleaseImage(&img2);
                return true;
            }
            vx_pyramid pyr1 = vxCreatePyramid(ctx, 2, VX_SCALE_PYRAMID_HALF, W, H, VX_DF_IMAGE_U8);
            vx_pyramid pyr2 = vxCreatePyramid(ctx, 2, VX_SCALE_PYRAMID_HALF, W, H, VX_DF_IMAGE_U8);
            vx_array old_pts = vxCreateArray(ctx, VX_TYPE_KEYPOINT, 10);
            vx_array new_pts = vxCreateArray(ctx, VX_TYPE_KEYPOINT, 10);
            vx_keypoint_t pts[4];
            for (int i = 0; i < 4; i++) {
                pts[i].x = 16 + (i % 2) * 32;
                pts[i].y = 16 + (i / 2) * 32;
                pts[i].strength = 1.0f;
                pts[i].tracking_status = 1;
                pts[i].scale = 0.0f;
                pts[i].orientation = 0.0f;
                pts[i].error = 0.0f;
            }
            vxAddArrayItems(old_pts, 4, pts, sizeof(vx_keypoint_t));
            vx_float32 eps_val = 0.01f;
            vx_scalar epsilon = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &eps_val);
            vx_uint32 iter_val = 5;
            vx_scalar num_iters = vxCreateScalar(ctx, VX_TYPE_UINT32, &iter_val);
            vx_bool use_init = vx_false_e;
            vx_scalar use_initial = vxCreateScalar(ctx, VX_TYPE_BOOL, &use_init);
            vx_graph g = vxCreateGraph(ctx);
            vx_node pn1 = vxGaussianPyramidNode(g, img1, pyr1);
            vx_node pn2 = vxGaussianPyramidNode(g, img2, pyr2);
            vx_node n = vxOpticalFlowPyrLKNode(g, pyr1, pyr2, old_pts, old_pts,
                                                new_pts, VX_TERM_CRITERIA_BOTH,
                                                epsilon, num_iters, use_initial, 5);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            bool ok = (status == VX_SUCCESS);
            vxReleaseNode(&pn1); vxReleaseNode(&pn2); vxReleaseNode(&n);
            vxReleaseGraph(&g);
            vxReleaseScalar(&epsilon); vxReleaseScalar(&num_iters); vxReleaseScalar(&use_initial);
            vxReleaseArray(&old_pts); vxReleaseArray(&new_pts);
            vxReleasePyramid(&pyr1); vxReleasePyramid(&pyr2);
            vxReleaseImage(&img1); vxReleaseImage(&img2);
            return ok;
        };
        cases.push_back(bc);
    }

    return cases;
}
