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
#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_nodes.h>
#include <cmath>
#include <vector>

std::vector<BenchmarkCase> registerStatisticalBenchmarks()
{
    std::vector<BenchmarkCase> cases;

    // ---- Histogram ----
    {
        BenchmarkCase bc;
        bc.name        = "Histogram";
        bc.category    = "statistical";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_HISTOGRAM;
        bc.required_kernels = {VX_KERNEL_HISTOGRAM};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_distribution distribution = tracker.trackDistribution(
                gen.createDistribution(ctx, 256, 0, 256));
            vx_node node = vxHistogramNode(graph, input, distribution);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_distribution dist = vxCreateDistribution(ctx, 256, 0, 256);
            vxuHistogram(ctx, in, dist);
            vx_int32 bins[256] = {};
            vxCopyDistribution(dist, bins, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
            bool ok = (bins[100] == 64 * 64);
            for (int i = 0; i < 256; i++) {
                if (i != 100 && bins[i] != 0) ok = false;
            }
            vxReleaseDistribution(&dist);
            vxReleaseImage(&in);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- EqualizeHist ----
    {
        BenchmarkCase bc;
        bc.name        = "EqualizeHist";
        bc.category    = "statistical";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_EQUALIZE_HISTOGRAM;
        bc.required_kernels = {VX_KERNEL_EQUALIZE_HISTOGRAM};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_node node = vxEqualizeHistNode(graph, input, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vxuEqualizeHist(ctx, in, out);
            auto result = verify::readImage(out, 64, 64);
            // All pixels same value means equalization should produce uniform output
            // The exact value depends on implementation but all pixels should be the same
            bool ok = true;
            for (size_t i = 1; i < result.size(); i++) {
                if (result[i] != result[0]) { ok = false; break; }
            }
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- MeanStdDev ----
    {
        BenchmarkCase bc;
        bc.name        = "MeanStdDev";
        bc.category    = "statistical";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_MEAN_STDDEV;
        bc.required_kernels = {VX_KERNEL_MEAN_STDDEV};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_float32 mean_val = 0.0f;
            vx_scalar mean_scalar = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_FLOAT32, &mean_val));
            vx_float32 stddev_val = 0.0f;
            vx_scalar stddev_scalar = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_FLOAT32, &stddev_val));
            vx_node node = vxMeanStdDevNode(graph, input, mean_scalar, stddev_scalar);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_float32 mean = 0, stddev = 0;
            vxuMeanStdDev(ctx, in, &mean, &stddev);
            bool ok = (std::abs(mean - 100.0f) < 0.01f) && (stddev < 0.01f);
            vxReleaseImage(&in);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- MinMaxLoc ----
    {
        BenchmarkCase bc;
        bc.name        = "MinMaxLoc";
        bc.category    = "statistical";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_MINMAXLOC;
        bc.required_kernels = {VX_KERNEL_MINMAXLOC};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));

            vx_uint8 min_init = 0;
            vx_scalar minVal = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_UINT8, &min_init));
            vx_uint8 max_init = 0;
            vx_scalar maxVal = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_UINT8, &max_init));

            vx_array minLoc = tracker.trackArray(
                vxCreateArray(ctx, VX_TYPE_COORDINATES2D, 1));
            vx_array maxLoc = tracker.trackArray(
                vxCreateArray(ctx, VX_TYPE_COORDINATES2D, 1));

            vx_uint32 count_init = 0;
            vx_scalar minCount = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_UINT32, &count_init));
            vx_scalar maxCount = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_UINT32, &count_init));

            vx_node node = vxMinMaxLocNode(graph, input,
                                           minVal, maxVal,
                                           minLoc, maxLoc,
                                           minCount, maxCount);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            std::vector<uint8_t> a(64 * 64, 100);
            a[0] = 50;
            a[1] = 200;
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_uint8 min_val = 0, max_val = 0;
            vx_uint32 min_count = 0, max_count = 0;
            vx_scalar s_min = vxCreateScalar(ctx, VX_TYPE_UINT8, &min_val);
            vx_scalar s_max = vxCreateScalar(ctx, VX_TYPE_UINT8, &max_val);
            vx_scalar s_min_count = vxCreateScalar(ctx, VX_TYPE_UINT32, &min_count);
            vx_scalar s_max_count = vxCreateScalar(ctx, VX_TYPE_UINT32, &max_count);
            vx_array min_loc = vxCreateArray(ctx, VX_TYPE_COORDINATES2D, 4);
            vx_array max_loc = vxCreateArray(ctx, VX_TYPE_COORDINATES2D, 4);
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = vxMinMaxLocNode(g, in, s_min, s_max, min_loc, max_loc, s_min_count, s_max_count);
            vxVerifyGraph(g);
            vxProcessGraph(g);
            vxCopyScalar(s_min, &min_val, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
            vxCopyScalar(s_max, &max_val, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
            bool ok = (min_val == 50 && max_val == 200);
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseScalar(&s_min); vxReleaseScalar(&s_max);
            vxReleaseScalar(&s_min_count); vxReleaseScalar(&s_max_count);
            vxReleaseArray(&min_loc); vxReleaseArray(&max_loc);
            vxReleaseImage(&in);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- IntegralImage ----
    {
        BenchmarkCase bc;
        bc.name        = "IntegralImage";
        bc.category    = "statistical";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_INTEGRAL_IMAGE;
        bc.required_kernels = {VX_KERNEL_INTEGRAL_IMAGE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U32));
            vx_node node = vxIntegralImageNode(graph, input, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            std::vector<uint8_t> a(64 * 64, 1);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U32);
            vxuIntegralImage(ctx, in, out);
            // Read U32 output
            vx_rectangle_t rect = {0, 0, 64, 64};
            vx_imagepatch_addressing_t addr = {};
            void* ptr = nullptr;
            vx_map_id map_id;
            vxMapImagePatch(out, &rect, 0, &map_id, &addr, &ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
            uint32_t* data = static_cast<uint32_t*>(ptr);
            uint32_t stride = addr.stride_y / sizeof(uint32_t);
            // Expected integral: [0,0]=1, [1,0]=2, [0,1]=65 (64+1)
            bool ok = (data[0] == 1 && data[1] == 2 && data[stride] == 65);
            vxUnmapImagePatch(out, map_id);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    return cases;
}
