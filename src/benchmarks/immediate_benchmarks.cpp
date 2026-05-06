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
#include <VX/vxu.h>
#include <vector>

std::vector<BenchmarkCase> registerImmediateBenchmarks()
{
    std::vector<BenchmarkCase> cases;

    // ---- And_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "And_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_AND;
        bc.required_kernels = {VX_KERNEL_AND};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image in1 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image in2 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image out = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            return vxuAnd(ctx, in1, in2, out);
        };
        cases.push_back(bc);
    }

    // ---- Or_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "Or_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_OR;
        bc.required_kernels = {VX_KERNEL_OR};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image in1 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image in2 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image out = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            return vxuOr(ctx, in1, in2, out);
        };
        cases.push_back(bc);
    }

    // ---- Not_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "Not_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_NOT;
        bc.required_kernels = {VX_KERNEL_NOT};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            return vxuNot(ctx, input, output);
        };
        cases.push_back(bc);
    }

    // ---- AbsDiff_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "AbsDiff_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_ABSDIFF;
        bc.required_kernels = {VX_KERNEL_ABSDIFF};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image in1 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image in2 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image out = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            return vxuAbsDiff(ctx, in1, in2, out);
        };
        cases.push_back(bc);
    }

    // ---- Add_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "Add_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_ADD;
        bc.required_kernels = {VX_KERNEL_ADD};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image in1 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image in2 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image out = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            return vxuAdd(ctx, in1, in2, VX_CONVERT_POLICY_SATURATE, out);
        };
        cases.push_back(bc);
    }

    // ---- Subtract_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "Subtract_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_SUBTRACT;
        bc.required_kernels = {VX_KERNEL_SUBTRACT};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image in1 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image in2 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image out = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            return vxuSubtract(ctx, in1, in2, VX_CONVERT_POLICY_SATURATE, out);
        };
        cases.push_back(bc);
    }

    // ---- Multiply_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "Multiply_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_MULTIPLY;
        bc.required_kernels = {VX_KERNEL_MULTIPLY};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image in1 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image in2 = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image out = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            return vxuMultiply(ctx, in1, in2, 1.0f, VX_CONVERT_POLICY_SATURATE,
                               VX_ROUND_POLICY_TO_ZERO, out);
        };
        cases.push_back(bc);
    }

    // ---- Box3x3_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "Box3x3_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_BOX_3x3;
        bc.required_kernels = {VX_KERNEL_BOX_3x3};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            return vxuBox3x3(ctx, input, output);
        };
        cases.push_back(bc);
    }

    // ---- Gaussian3x3_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "Gaussian3x3_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_GAUSSIAN_3x3;
        bc.required_kernels = {VX_KERNEL_GAUSSIAN_3x3};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            return vxuGaussian3x3(ctx, input, output);
        };
        cases.push_back(bc);
    }

    // ---- Median3x3_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "Median3x3_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_MEDIAN_3x3;
        bc.required_kernels = {VX_KERNEL_MEDIAN_3x3};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            return vxuMedian3x3(ctx, input, output);
        };
        cases.push_back(bc);
    }

    // ---- Erode3x3_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "Erode3x3_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_ERODE_3x3;
        bc.required_kernels = {VX_KERNEL_ERODE_3x3};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            return vxuErode3x3(ctx, input, output);
        };
        cases.push_back(bc);
    }

    // ---- Dilate3x3_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "Dilate3x3_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_DILATE_3x3;
        bc.required_kernels = {VX_KERNEL_DILATE_3x3};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            return vxuDilate3x3(ctx, input, output);
        };
        cases.push_back(bc);
    }

    // ---- Sobel3x3_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "Sobel3x3_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_SOBEL_3x3;
        bc.required_kernels = {VX_KERNEL_SOBEL_3x3};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image input    = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output_x = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_S16));
            vx_image output_y = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_S16));
            return vxuSobel3x3(ctx, input, output_x, output_y);
        };
        cases.push_back(bc);
    }

    // ---- EqualizeHist_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "EqualizeHist_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_EQUALIZE_HISTOGRAM;
        bc.required_kernels = {VX_KERNEL_EQUALIZE_HISTOGRAM};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            return vxuEqualizeHist(ctx, input, output);
        };
        cases.push_back(bc);
    }

    // ---- Threshold_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "Threshold_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_THRESHOLD;
        bc.required_kernels = {VX_KERNEL_THRESHOLD};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image input     = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_threshold thresh = tracker.trackThreshold(gen.createBinaryThreshold(ctx, 128));
            vx_image output    = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            return vxuThreshold(ctx, input, thresh, output);
        };
        cases.push_back(bc);
    }

    // ---- ColorConvert_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "ColorConvert_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_COLOR_CONVERT;
        bc.required_kernels = {VX_KERNEL_COLOR_CONVERT};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            // Width/height must be even for YUV formats
            uint32_t w = width & ~1u;
            uint32_t h = height & ~1u;
            if (w == 0 || h == 0) return VX_ERROR_INVALID_DIMENSION;

            vx_image rgb_input   = tracker.trackImage(gen.createFilledImage(ctx, w, h, VX_DF_IMAGE_RGB));
            vx_image iyuv_output = tracker.trackImage(vxCreateImage(ctx, w, h, VX_DF_IMAGE_IYUV));
            return vxuColorConvert(ctx, rgb_input, iyuv_output);
        };
        cases.push_back(bc);
    }

    // ---- ScaleImage_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "ScaleImage_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_SCALE_IMAGE;
        bc.required_kernels = {VX_KERNEL_SCALE_IMAGE};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image input       = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output_half = tracker.trackImage(vxCreateImage(ctx, width / 2, height / 2, VX_DF_IMAGE_U8));
            return vxuScaleImage(ctx, input, output_half, VX_INTERPOLATION_BILINEAR);
        };
        cases.push_back(bc);
    }

    // ---- CannyEdgeDetector_Immediate ----
    {
        BenchmarkCase bc;
        bc.name        = "CannyEdgeDetector_Immediate";
        bc.category    = "immediate";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_CANNY_EDGE_DETECTOR;
        bc.required_kernels = {VX_KERNEL_CANNY_EDGE_DETECTOR};
        bc.graph_setup = nullptr;
        bc.immediate_func = [](vx_context ctx, uint32_t width, uint32_t height,
                               TestDataGenerator& gen, ResourceTracker& tracker) -> vx_status {
            vx_image input     = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_threshold hyst  = tracker.trackThreshold(gen.createRangeThreshold(ctx, 80, 100));
            vx_image output    = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            return vxuCannyEdgeDetector(ctx, input, hyst, 3, VX_NORM_L1, output);
        };
        cases.push_back(bc);
    }

    return cases;
}
