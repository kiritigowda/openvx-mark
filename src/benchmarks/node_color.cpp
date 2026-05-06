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

std::vector<BenchmarkCase> registerColorBenchmarks() {
    std::vector<BenchmarkCase> cases;

    // 1. ColorConvert_RGB2IYUV
    {
        BenchmarkCase bc;
        bc.name = "ColorConvert_RGB2IYUV";
        bc.category = "color";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_COLOR_CONVERT;
        bc.required_kernels = {VX_KERNEL_COLOR_CONVERT};
        bc.graph_setup = [](vx_context ctx, vx_graph graph, uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            // Width/height must be even for YUV formats
            uint32_t w = width & ~1u;
            uint32_t h = height & ~1u;
            if (w == 0 || h == 0) return false;

            vx_image input_rgb = gen.createFilledImage(ctx, w, h, VX_DF_IMAGE_RGB);
            if (vxGetStatus((vx_reference)input_rgb) != VX_SUCCESS) return false;
            tracker.trackImage(input_rgb);

            vx_image output_iyuv = vxCreateImage(ctx, w, h, VX_DF_IMAGE_IYUV);
            if (vxGetStatus((vx_reference)output_iyuv) != VX_SUCCESS) return false;
            tracker.trackImage(output_iyuv);

            vx_node node = vxColorConvertNode(graph, input_rgb, output_iyuv);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // 2. ColorConvert_RGB2NV12
    {
        BenchmarkCase bc;
        bc.name = "ColorConvert_RGB2NV12";
        bc.category = "color";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_COLOR_CONVERT;
        bc.required_kernels = {VX_KERNEL_COLOR_CONVERT};
        bc.graph_setup = [](vx_context ctx, vx_graph graph, uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            // Width/height must be even for NV12 format
            uint32_t w = width & ~1u;
            uint32_t h = height & ~1u;
            if (w == 0 || h == 0) return false;

            vx_image input_rgb = gen.createFilledImage(ctx, w, h, VX_DF_IMAGE_RGB);
            if (vxGetStatus((vx_reference)input_rgb) != VX_SUCCESS) return false;
            tracker.trackImage(input_rgb);

            vx_image output_nv12 = vxCreateImage(ctx, w, h, VX_DF_IMAGE_NV12);
            if (vxGetStatus((vx_reference)output_nv12) != VX_SUCCESS) return false;
            tracker.trackImage(output_nv12);

            vx_node node = vxColorConvertNode(graph, input_rgb, output_nv12);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // 3. ChannelExtract
    {
        BenchmarkCase bc;
        bc.name = "ChannelExtract";
        bc.category = "color";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_CHANNEL_EXTRACT;
        bc.required_kernels = {VX_KERNEL_CHANNEL_EXTRACT};
        bc.graph_setup = [](vx_context ctx, vx_graph graph, uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input_rgb = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_RGB);
            if (vxGetStatus((vx_reference)input_rgb) != VX_SUCCESS) return false;
            tracker.trackImage(input_rgb);

            vx_image output_u8 = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output_u8) != VX_SUCCESS) return false;
            tracker.trackImage(output_u8);

            vx_node node = vxChannelExtractNode(graph, input_rgb, VX_CHANNEL_R, output_u8);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // 4. ChannelCombine
    {
        BenchmarkCase bc;
        bc.name = "ChannelCombine";
        bc.category = "color";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_CHANNEL_COMBINE;
        bc.required_kernels = {VX_KERNEL_CHANNEL_COMBINE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph, uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image ch0 = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)ch0) != VX_SUCCESS) return false;
            tracker.trackImage(ch0);

            vx_image ch1 = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)ch1) != VX_SUCCESS) return false;
            tracker.trackImage(ch1);

            vx_image ch2 = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)ch2) != VX_SUCCESS) return false;
            tracker.trackImage(ch2);

            vx_image output_rgb = vxCreateImage(ctx, width, height, VX_DF_IMAGE_RGB);
            if (vxGetStatus((vx_reference)output_rgb) != VX_SUCCESS) return false;
            tracker.trackImage(output_rgb);

            vx_node node = vxChannelCombineNode(graph, ch0, ch1, ch2, nullptr, output_rgb);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    // 5. ConvertDepth
    {
        BenchmarkCase bc;
        bc.name = "ConvertDepth";
        bc.category = "color";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_CONVERTDEPTH;
        bc.required_kernels = {VX_KERNEL_CONVERTDEPTH};
        bc.graph_setup = [](vx_context ctx, vx_graph graph, uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input_u8 = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input_u8) != VX_SUCCESS) return false;
            tracker.trackImage(input_u8);

            vx_image output_s16 = vxCreateImage(ctx, width, height, VX_DF_IMAGE_S16);
            if (vxGetStatus((vx_reference)output_s16) != VX_SUCCESS) return false;
            tracker.trackImage(output_s16);

            vx_int32 shift_val = 0;
            vx_scalar shift = vxCreateScalar(ctx, VX_TYPE_INT32, &shift_val);
            if (vxGetStatus((vx_reference)shift) != VX_SUCCESS) return false;
            tracker.trackScalar(shift);

            vx_node node = vxConvertDepthNode(graph, input_u8, output_s16,
                                              VX_CONVERT_POLICY_SATURATE, shift);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        cases.push_back(bc);
    }

    return cases;
}
