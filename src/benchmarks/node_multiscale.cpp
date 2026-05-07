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
#include "openvx_version.h"
#include "verify_utils.h"
#include <VX/vxu.h>
#include <VX/vx_nodes.h>
#include <cmath>
#include <vector>

std::vector<BenchmarkCase> registerMultiscaleBenchmarks() {
    std::vector<BenchmarkCase> cases;

    // GaussianPyramid — U8 input, Gaussian pyramid output
    {
        BenchmarkCase bc;
        bc.name = "GaussianPyramid";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_GAUSSIAN_PYRAMID;
        bc.required_kernels = {VX_KERNEL_GAUSSIAN_PYRAMID};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen,
                            ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            vx_pyramid pyramid = gen.createPyramid(ctx, DEFAULT_PYRAMID_LEVELS,
                                                   VX_SCALE_PYRAMID_HALF,
                                                   width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)pyramid) != VX_SUCCESS) return false;
            tracker.trackPyramid(pyramid);

            vx_node node = vxGaussianPyramidNode(graph, input, pyramid);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            uint8_t a[64];
            for (int i = 0; i < 64; i++) a[i] = 100;
            vx_image in = verify::createImage(ctx, 8, 8, VX_DF_IMAGE_U8, a);
            vx_pyramid pyr = vxCreatePyramid(ctx, 4, VX_SCALE_PYRAMID_HALF, 8, 8, VX_DF_IMAGE_U8);
            vxuGaussianPyramid(ctx, in, pyr);
            vx_image level0 = vxGetPyramidLevel(pyr, 0);
            auto result = verify::readImage(level0, 8, 8);
            bool ok = (result[0] == 100);
            vxReleaseImage(&level0); vxReleasePyramid(&pyr); vxReleaseImage(&in);
            return ok;
        };
        cases.push_back(bc);
    }

#if OPENVX_HAS_1_1
    // LaplacianPyramid — U8 input, Laplacian pyramid + S16 remainder output
    {
        BenchmarkCase bc;
        bc.name = "LaplacianPyramid";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_LAPLACIAN_PYRAMID;
        bc.required_kernels = {VX_KERNEL_LAPLACIAN_PYRAMID};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen,
                            ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            vx_pyramid laplacian_pyr = vxCreatePyramid(ctx, DEFAULT_PYRAMID_LEVELS - 1,
                                                       VX_SCALE_PYRAMID_HALF,
                                                       width, height, VX_DF_IMAGE_S16);
            if (vxGetStatus((vx_reference)laplacian_pyr) != VX_SUCCESS) return false;
            tracker.trackPyramid(laplacian_pyr);

            vx_uint32 out_w = width >> (DEFAULT_PYRAMID_LEVELS - 1);
            vx_uint32 out_h = height >> (DEFAULT_PYRAMID_LEVELS - 1);
            if (out_w < 1) out_w = 1;
            if (out_h < 1) out_h = 1;
            vx_image output = vxCreateImage(ctx, out_w, out_h, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;
            tracker.trackImage(output);

            vx_node node = vxLaplacianPyramidNode(graph, input, laplacian_pyr, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            uint8_t a[64];
            for (int i = 0; i < 64; i++) a[i] = 100;
            vx_image in = verify::createImage(ctx, 8, 8, VX_DF_IMAGE_U8, a);
            vx_pyramid lap = vxCreatePyramid(ctx, 3, VX_SCALE_PYRAMID_HALF, 8, 8, VX_DF_IMAGE_S16);
            vx_image remainder = vxCreateImage(ctx, 1, 1, VX_DF_IMAGE_U8);
            vxuLaplacianPyramid(ctx, in, lap, remainder);
            // For uniform input, remainder should be close to 100
            auto result = verify::readImage(remainder, 1, 1);
            bool ok = (std::abs((int)result[0] - 100) <= 5);
            vxReleaseImage(&remainder); vxReleasePyramid(&lap); vxReleaseImage(&in);
            return ok;
        };
        cases.push_back(bc);
    }
#endif

    // HalfScaleGaussian — U8 input, U8 output at half resolution, kernel_size=3
    {
        BenchmarkCase bc;
        bc.name = "HalfScaleGaussian";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_HALFSCALE_GAUSSIAN;
        bc.required_kernels = {VX_KERNEL_HALFSCALE_GAUSSIAN};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen,
                            ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width / 2, height / 2, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;
            tracker.trackImage(output);

            vx_node node = vxHalfScaleGaussianNode(graph, input, output, 3);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            uint8_t a[16];
            for (int i = 0; i < 16; i++) a[i] = 100;
            vx_image in = verify::createImage(ctx, 4, 4, VX_DF_IMAGE_U8, a);
            vx_image out = vxCreateImage(ctx, 2, 2, VX_DF_IMAGE_U8);
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = vxHalfScaleGaussianNode(g, in, out, 3);
            vxVerifyGraph(g);
            vxProcessGraph(g);
            auto result = verify::readImage(out, 2, 2);
            bool ok = verify::compareU8(result, {100, 100, 100, 100}, 1);
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    return cases;
}
