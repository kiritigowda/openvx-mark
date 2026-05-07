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
            int16_t gx[] = {3, 0, 4, 100};
            int16_t gy[] = {4, 0, 3, 0};
            // magnitude = sqrt(gx^2 + gy^2), integer truncation: sqrt(9+16)=5, sqrt(0)=0, sqrt(16+9)=5, sqrt(10000)=100
            vx_image in_x = verify::createImage(ctx, 2, 2, VX_DF_IMAGE_S16, reinterpret_cast<const uint8_t*>(gx));
            vx_image in_y = verify::createImage(ctx, 2, 2, VX_DF_IMAGE_S16, reinterpret_cast<const uint8_t*>(gy));
            vx_image out = vxCreateImage(ctx, 2, 2, VX_DF_IMAGE_S16);
            vxuMagnitude(ctx, in_x, in_y, out);
            auto result = verify::readImageS16(out, 2, 2);
            // Allow tolerance of 1 for rounding
            bool ok = verify::compareS16(result, {5, 0, 5, 100}, 1);
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
            int16_t gx[] = {100, 0, -100, 0};
            int16_t gy[] = {0, 100, 0, -100};
            // phase(100,0)=0, phase(0,100)=64 (~pi/2), phase(-100,0)=128 (~pi), phase(0,-100)=192 (~3pi/2)
            vx_image in_x = verify::createImage(ctx, 2, 2, VX_DF_IMAGE_S16, reinterpret_cast<const uint8_t*>(gx));
            vx_image in_y = verify::createImage(ctx, 2, 2, VX_DF_IMAGE_S16, reinterpret_cast<const uint8_t*>(gy));
            vx_image out = vxCreateImage(ctx, 2, 2, VX_DF_IMAGE_U8);
            vxuPhase(ctx, in_x, in_y, out);
            auto result = verify::readImage(out, 2, 2);
            uint8_t exp[] = {0, 64, 128, 192};
            bool ok = verify::compareU8(result, {exp, exp + 4}, 2);
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
            uint8_t a[] = {0, 128, 255, 42};
            vx_image in = verify::createImage(ctx, 2, 2, VX_DF_IMAGE_U8, a);
            vx_image out = vxCreateImage(ctx, 2, 2, VX_DF_IMAGE_U8);
            vx_lut lut = vxCreateLUT(ctx, VX_TYPE_UINT8, 256);
            uint8_t identity[256];
            for (int i = 0; i < 256; i++) identity[i] = (uint8_t)i;
            vxCopyLUT(lut, identity, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vxuTableLookup(ctx, in, lut, out);
            auto result = verify::readImage(out, 2, 2);
            bool ok = verify::compareU8(result, {a, a + 4});
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
            uint8_t a[] = {0, 100, 200, 255};
            uint8_t exp[] = {0, 0, 255, 255};
            vx_image in = verify::createImage(ctx, 2, 2, VX_DF_IMAGE_U8, a);
            vx_image out = vxCreateImage(ctx, 2, 2, VX_DF_IMAGE_U8);
            vx_threshold thresh = vxCreateThresholdForImage(ctx, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
            vx_pixel_value_t pv = {};
            pv.U8 = 128;
            vxCopyThresholdValue(thresh, &pv, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vxuThreshold(ctx, in, thresh, out);
            auto result = verify::readImage(out, 2, 2);
            bool ok = verify::compareU8(result, {exp, exp + 4});
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
            uint8_t a[] = {0, 80, 200, 255};
            uint8_t exp[] = {0, 255, 255, 0};
            vx_image in = verify::createImage(ctx, 2, 2, VX_DF_IMAGE_U8, a);
            vx_image out = vxCreateImage(ctx, 2, 2, VX_DF_IMAGE_U8);
            vx_threshold thresh = vxCreateThresholdForImage(ctx, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
            vx_pixel_value_t lower_pv = {}, upper_pv = {};
            lower_pv.U8 = 80; upper_pv.U8 = 200;
            vxCopyThresholdRange(thresh, &lower_pv, &upper_pv, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vxuThreshold(ctx, in, thresh, out);
            auto result = verify::readImage(out, 2, 2);
            bool ok = verify::compareU8(result, {exp, exp + 4});
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
            uint8_t a[] = {100, 200, 0, 255};
            uint8_t b[] = {200, 100, 0, 0};
            // alpha=0.5: out = 0.5*a + 0.5*b = {150, 150, 0, 127/128}
            uint8_t exp[] = {150, 150, 0, 128};
            vx_image in1 = verify::createImage(ctx, 2, 2, VX_DF_IMAGE_U8, a);
            vx_image in2 = verify::createImage(ctx, 2, 2, VX_DF_IMAGE_U8, b);
            vx_image out = vxCreateImage(ctx, 2, 2, VX_DF_IMAGE_U8);
            vx_float32 alpha = 0.5f;
            vx_scalar s_alpha = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &alpha);
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = vxWeightedAverageNode(g, in1, s_alpha, in2, out);
            vxVerifyGraph(g);
            vxProcessGraph(g);
            auto result = verify::readImage(out, 2, 2);
            bool ok = verify::compareU8(result, {exp, exp + 4}, 1);
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
            uint8_t t[] = {10, 20, 30, 40};
            uint8_t f[] = {90, 80, 70, 60};
            vx_image true_img = verify::createImage(ctx, 2, 2, VX_DF_IMAGE_U8, t);
            vx_image false_img = verify::createImage(ctx, 2, 2, VX_DF_IMAGE_U8, f);
            vx_image out = vxCreateImage(ctx, 2, 2, VX_DF_IMAGE_U8);
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
            vxVerifyGraph(g);
            vxProcessGraph(g);
            auto result = verify::readImage(out, 2, 2);
            bool ok = verify::compareU8(result, {t, t + 4});
            vxReleaseNode(&n); vxReleaseGraph(&g); vxReleaseScalar(&condition);
            vxReleaseImage(&true_img); vxReleaseImage(&false_img); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }
#endif

    return cases;
}
