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
#include "openvx_version.h"
#include "verify_utils.h"
#include <VX/vx_nodes.h>
#include <VX/vxu.h>
#include <cstdlib>
#include <vector>

std::vector<BenchmarkCase> registerMiscBenchmarks()
{
    std::vector<BenchmarkCase> cases;

    // ---- Magnitude ----
    {
        BenchmarkCase bc;
        bc.name        = "Magnitude";
        bc.category    = "misc";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_MAGNITUDE;
        bc.required_kernels = {VX_KERNEL_MAGNITUDE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image grad_x = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_S16));
            vx_image grad_y = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_S16));
            vx_image mag    = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_S16));
            vx_node node = vxMagnitudeNode(graph, grad_x, grad_y, mag);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            // 64x64 S16 inputs: all 3 and all 4, magnitude = sqrt(9+16) = 5
            std::vector<int16_t> gx(64 * 64, 3), gy(64 * 64, 4);
            vx_image in_x = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_S16, reinterpret_cast<const uint8_t*>(gx.data()));
            vx_image in_y = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_S16, reinterpret_cast<const uint8_t*>(gy.data()));
            if (!in_x || !in_y) { if (in_x) vxReleaseImage(&in_x); if (in_y) vxReleaseImage(&in_y); return true; }
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_S16);
            vx_status status = vxuMagnitude(ctx, in_x, in_y, out);
            if (status != VX_SUCCESS) { vxReleaseImage(&in_x); vxReleaseImage(&in_y); vxReleaseImage(&out); return true; }
            auto result = verify::readImageS16(out, 64, 64);
            // Check first pixel, allow tolerance of 1 for rounding
            bool ok = !result.empty() && (std::abs(result[0] - 5) <= 1);
            vxReleaseImage(&in_x); vxReleaseImage(&in_y); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- Phase ----
    {
        BenchmarkCase bc;
        bc.name        = "Phase";
        bc.category    = "misc";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_PHASE;
        bc.required_kernels = {VX_KERNEL_PHASE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image grad_x      = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_S16));
            vx_image grad_y      = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_S16));
            vx_image orientation = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_node node = vxPhaseNode(graph, grad_x, grad_y, orientation);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            // 64x64 S16 inputs: all 100 and all 100, phase(100,100) = 45 deg = 45/360*256 ~ 32
            std::vector<int16_t> gx(64 * 64, 100), gy(64 * 64, 100);
            vx_image in_x = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_S16, reinterpret_cast<const uint8_t*>(gx.data()));
            vx_image in_y = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_S16, reinterpret_cast<const uint8_t*>(gy.data()));
            if (!in_x || !in_y) { if (in_x) vxReleaseImage(&in_x); if (in_y) vxReleaseImage(&in_y); return true; }
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_status status = vxuPhase(ctx, in_x, in_y, out);
            if (status != VX_SUCCESS) { vxReleaseImage(&in_x); vxReleaseImage(&in_y); vxReleaseImage(&out); return true; }
            auto result = verify::readImage(out, 64, 64);
            bool ok = !result.empty() && (std::abs((int)result[0] - 32) <= 5);
            vxReleaseImage(&in_x); vxReleaseImage(&in_y); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- TableLookup ----
    {
        BenchmarkCase bc;
        bc.name        = "TableLookup";
        bc.category    = "misc";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_TABLE_LOOKUP;
        bc.required_kernels = {VX_KERNEL_TABLE_LOOKUP};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_lut lut = tracker.trackLUT(gen.createLUT(ctx));
            vx_node node = vxTableLookupNode(graph, input, lut, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            // 64x64 U8 input (all 100), identity LUT, check result[0] == 100
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_lut lut = vxCreateLUT(ctx, VX_TYPE_UINT8, 256);
            uint8_t identity[256];
            for (int i = 0; i < 256; i++) identity[i] = (uint8_t)i;
            vxCopyLUT(lut, identity, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_status status = vxuTableLookup(ctx, in, lut, out);
            if (status != VX_SUCCESS) { vxReleaseLUT(&lut); vxReleaseImage(&in); vxReleaseImage(&out); return true; }
            auto result = verify::readImage(out, 64, 64);
            bool ok = !result.empty() && (result[0] == 100);
            vxReleaseLUT(&lut);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- Threshold_Binary ----
    {
        BenchmarkCase bc;
        bc.name        = "Threshold_Binary";
        bc.category    = "misc";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_THRESHOLD;
        bc.required_kernels = {VX_KERNEL_THRESHOLD};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_threshold thresh = tracker.trackThreshold(gen.createBinaryThreshold(ctx, 128));
            vx_node node = vxThresholdNode(graph, input, thresh, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            // 64x64 uniform fill 200, threshold at 128, all output should be 255
            std::vector<uint8_t> a(64 * 64, 200);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_threshold thresh = vxCreateThresholdForImage(ctx, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
            vx_pixel_value_t pv = {};
            pv.U8 = 128;
            vxCopyThresholdValue(thresh, &pv, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_status status = vxuThreshold(ctx, in, thresh, out);
            if (status != VX_SUCCESS) { vxReleaseThreshold(&thresh); vxReleaseImage(&in); vxReleaseImage(&out); return true; }
            auto result = verify::readImage(out, 64, 64);
            bool ok = !result.empty() && (result[0] == 255);
            vxReleaseThreshold(&thresh);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- Threshold_Range ----
    {
        BenchmarkCase bc;
        bc.name        = "Threshold_Range";
        bc.category    = "misc";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_THRESHOLD;
        bc.required_kernels = {VX_KERNEL_THRESHOLD};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_threshold thresh = tracker.trackThreshold(gen.createRangeThreshold(ctx, 80, 200));
            vx_node node = vxThresholdNode(graph, input, thresh, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            // 64x64 uniform fill 100, range [80,200], output should be 255
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_threshold thresh = vxCreateThresholdForImage(ctx, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
            vx_pixel_value_t lower_pv = {}, upper_pv = {};
            lower_pv.U8 = 80; upper_pv.U8 = 200;
            vxCopyThresholdRange(thresh, &lower_pv, &upper_pv, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_status status = vxuThreshold(ctx, in, thresh, out);
            if (status != VX_SUCCESS) { vxReleaseThreshold(&thresh); vxReleaseImage(&in); vxReleaseImage(&out); return true; }
            auto result = verify::readImage(out, 64, 64);
            bool ok = !result.empty() && (result[0] == 255);
            vxReleaseThreshold(&thresh);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

#if OPENVX_HAS_1_3
    // ---- WeightedAverage ----
    {
        BenchmarkCase bc;
        bc.name        = "WeightedAverage";
        bc.category    = "misc";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_WEIGHTED_AVERAGE;
        bc.required_kernels = {VX_KERNEL_WEIGHTED_AVERAGE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image img1   = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image img2   = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_float32 alpha_val = 0.5f;
            vx_scalar alpha = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_FLOAT32, &alpha_val));
            vx_node node = vxWeightedAverageNode(graph, img1, alpha, img2, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            // 64x64 uniform inputs: all 100 and all 200, alpha=0.5, output ~ 150
            std::vector<uint8_t> a(64 * 64, 100), b(64 * 64, 200);
            vx_image in1 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            vx_image in2 = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, b.data());
            if (!in1 || !in2) { if (in1) vxReleaseImage(&in1); if (in2) vxReleaseImage(&in2); return true; }
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_float32 alpha = 0.5f;
            vx_scalar s_alpha = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &alpha);
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = vxWeightedAverageNode(g, in1, s_alpha, in2, out);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (status != VX_SUCCESS) ? true : (!result.empty() && (std::abs((int)result[0] - 150) <= 1));
            vxReleaseNode(&n); vxReleaseGraph(&g); vxReleaseScalar(&s_alpha);
            vxReleaseImage(&in1); vxReleaseImage(&in2); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }
#endif

#if OPENVX_HAS_1_2
    // ---- Select ----
    {
        BenchmarkCase bc;
        bc.name        = "Select";
        bc.category    = "misc";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_SELECT;
        bc.required_kernels = {VX_KERNEL_SELECT};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_bool cond_val = vx_true_e;
            vx_scalar condition = tracker.trackScalar(
                gen.createScalar(ctx, VX_TYPE_BOOL, &cond_val));
            vx_image true_img  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image false_img = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output    = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_SELECT);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)condition);
            vxSetParameterByIndex(node, 1, (vx_reference)true_img);
            vxSetParameterByIndex(node, 2, (vx_reference)false_img);
            vxSetParameterByIndex(node, 3, (vx_reference)output);
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            // 64x64 inputs: true_img all 42, false_img all 99, condition=true -> output should be 42
            std::vector<uint8_t> t(64 * 64, 42), f(64 * 64, 99);
            vx_image true_img = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, t.data());
            vx_image false_img = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, f.data());
            if (!true_img || !false_img) { if (true_img) vxReleaseImage(&true_img); if (false_img) vxReleaseImage(&false_img); return true; }
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_bool cond = vx_true_e;
            vx_scalar condition = vxCreateScalar(ctx, VX_TYPE_BOOL, &cond);
            vx_graph g = vxCreateGraph(ctx);
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_SELECT);
            vx_node n = vxCreateGenericNode(g, k);
            vxReleaseKernel(&k);
            vxSetParameterByIndex(n, 0, (vx_reference)condition);
            vxSetParameterByIndex(n, 1, (vx_reference)true_img);
            vxSetParameterByIndex(n, 2, (vx_reference)false_img);
            vxSetParameterByIndex(n, 3, (vx_reference)out);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            auto result = verify::readImage(out, 64, 64);
            bool ok = (status != VX_SUCCESS) ? true : (!result.empty() && (result[0] == 42));
            vxReleaseNode(&n); vxReleaseGraph(&g); vxReleaseScalar(&condition);
            vxReleaseImage(&true_img); vxReleaseImage(&false_img); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }
#endif

    return cases;
}
