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
#include "verify_utils.h"
#include "openvx_version.h"
#include <VX/vx_nodes.h>
#include <VX/vxu.h>
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
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vxuBox3x3(ctx, in, out);
            auto result = verify::readImage(out, 64, 64);
            // Center pixel should be 100 since all neighbors are 100
            bool ok = (result[32 * 64 + 32] == 100);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
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
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vxuGaussian3x3(ctx, in, out);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[32 * 64 + 32] == 100);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
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
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vxuMedian3x3(ctx, in, out);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[32 * 64 + 32] == 100);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
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
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vxuErode3x3(ctx, in, out);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[32 * 64 + 32] == 100);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
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
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vxuDilate3x3(ctx, in, out);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[32 * 64 + 32] == 100);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
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
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image out_x = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_S16);
            vx_image out_y = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_S16);
            vxuSobel3x3(ctx, in, out_x, out_y);
            auto gx = verify::readImageS16(out_x, 64, 64);
            auto gy = verify::readImageS16(out_y, 64, 64);
            // Center pixel gradients should be 0 for uniform input
            bool ok = (gx[32 * 64 + 32] == 0 && gy[32 * 64 + 32] == 0);
            vxReleaseImage(&in); vxReleaseImage(&out_x); vxReleaseImage(&out_y);
            return ok;
        };
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
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_convolution conv = vxCreateConvolution(ctx, 3, 3);
            vx_int16 kernel[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
            vxCopyConvolutionCoefficients(conv, kernel, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_uint32 scale = 1;
            vxSetConvolutionAttribute(conv, VX_CONVOLUTION_SCALE, &scale, sizeof(scale));
            vxuConvolve(ctx, in, conv, out);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[32 * 64 + 32] == 100);
            vxReleaseConvolution(&conv);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

#if OPENVX_HAS_1_1
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
        bc.verify_fn = [](vx_context ctx) -> bool {
            const int N = 64 * 64;
            std::vector<uint8_t> a(N, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            // 3x3 all-ones mask
            vx_uint8 mask_data[9];
            for (int i = 0; i < 9; i++) mask_data[i] = 255;
            vx_matrix mask = vxCreateMatrix(ctx, VX_TYPE_UINT8, 3, 3);
            vxCopyMatrix(mask, mask_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vxuNonLinearFilter(ctx, VX_NONLINEAR_FILTER_MEDIAN, in, mask, out);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[32 * 64 + 32] == 100);
            vxReleaseMatrix(&mask);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }
#endif

    return cases;
}
