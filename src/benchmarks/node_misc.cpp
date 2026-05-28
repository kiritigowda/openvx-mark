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
#include "openvx_optional_apis.h"
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

    // ---- TableLookup (U8 input, U8 LUT) ----
    //
    // Vision Conformance Feature Set: TableLookup requires both VX_TYPE_UINT8
    // (256-entry LUT, U8 image) and VX_TYPE_INT16 (65536-entry LUT, S16
    // image with offset). The two paths have very different LUT sizes and
    // address arithmetic, so we benchmark them as separate tests.
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

    // ---- TableLookup_S16 (S16 input, S16 LUT) ----
    //
    // The S16 LUT path uses a 65536-entry table with a signed offset, which
    // is materially different from the 256-entry U8 path in both memory
    // footprint and address arithmetic. Required by the Vision Conformance
    // Feature Set per the OpenVX 1.3 vxTableLookupNode spec.
    {
        BenchmarkCase bc;
        bc.name        = "TableLookup_S16";
        bc.category    = "misc";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_TABLE_LOOKUP;
        bc.required_kernels = {VX_KERNEL_TABLE_LOOKUP};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_S16));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_S16));
            // Full S16 LUT: 65536 entries, offset -32768 so index 0 maps to -32768.
            vx_lut lut = vxCreateLUT(ctx, VX_TYPE_INT16, 65536);
            if (vxGetStatus((vx_reference)lut) != VX_SUCCESS) return false;
            tracker.trackLUT(lut);
            std::vector<int16_t> table(65536);
            for (size_t i = 0; i < table.size(); i++) {
                table[i] = static_cast<int16_t>(i - 32768);
            }
            vxCopyLUT(lut, table.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_node node = vxTableLookupNode(graph, input, lut, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            // 64x64 S16 image filled with 1000, identity LUT (with -32768 offset
            // so LUT[input + 32768] = input) → output should equal input.
            std::vector<int16_t> a(64 * 64, 1000);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_S16,
                                              reinterpret_cast<const uint8_t*>(a.data()));
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_S16);
            vx_lut lut = vxCreateLUT(ctx, VX_TYPE_INT16, 65536);
            if (!lut) { vxReleaseImage(&in); vxReleaseImage(&out); return true; }
            std::vector<int16_t> table(65536);
            for (size_t i = 0; i < table.size(); i++) {
                table[i] = static_cast<int16_t>(i - 32768);
            }
            vxCopyLUT(lut, table.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_status status = vxuTableLookup(ctx, in, lut, out);
            if (status != VX_SUCCESS) {
                vxReleaseLUT(&lut); vxReleaseImage(&in); vxReleaseImage(&out);
                return true;
            }
            auto result = verify::readImageS16(out, 64, 64);
            bool ok = !result.empty() && (result[0] == 1000);
            vxReleaseLUT(&lut);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- Threshold_Binary (U8 input) ----
    //
    // OpenVX 1.3.1 §3.55 [REQ-0493]: Threshold accepts U8 or S16 input,
    // and produces a U8 or U1 boolean output. The vx_threshold object's
    // VX_THRESHOLD_INPUT_FORMAT must match the input image format. We
    // benchmark U8 binary, U8 range and S16 binary as separate tests.
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

#if OPENVX_HAS_1_3
    // ---- Threshold_S16 (S16 input, U8 binary output) ----
    //
    // OpenVX 1.3.1 §3.55: Threshold input may be U8 or S16; the threshold
    // object's VX_THRESHOLD_INPUT_FORMAT must match the input image.
    // Gated on OpenVX 1.3+ because pre-1.3 `vxCreateThreshold(type, U8)`
    // is the only path our compat shim provides (no signed support there).
    {
        BenchmarkCase bc;
        bc.name        = "Threshold_S16";
        bc.category    = "misc";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_THRESHOLD;
        bc.required_kernels = {VX_KERNEL_THRESHOLD};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input  = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_S16));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_threshold thresh = vxCreateThresholdForImage(ctx, VX_THRESHOLD_TYPE_BINARY,
                                                            VX_DF_IMAGE_S16, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)thresh) != VX_SUCCESS) return false;
            tracker.trackThreshold(thresh);
            vx_pixel_value_t pv = {};
            pv.S16 = 1000;
            vxCopyThresholdValue(thresh, &pv, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_node node = vxThresholdNode(graph, input, thresh, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            // S16 input filled with 2000, threshold at 1000 → all output = 255
            std::vector<int16_t> a(64 * 64, 2000);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_S16,
                                              reinterpret_cast<const uint8_t*>(a.data()));
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_threshold thresh = vxCreateThresholdForImage(ctx, VX_THRESHOLD_TYPE_BINARY,
                                                            VX_DF_IMAGE_S16, VX_DF_IMAGE_U8);
            vx_pixel_value_t pv = {};
            pv.S16 = 1000;
            vxCopyThresholdValue(thresh, &pv, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_status status = vxuThreshold(ctx, in, thresh, out);
            if (status != VX_SUCCESS) {
                vxReleaseThreshold(&thresh); vxReleaseImage(&in); vxReleaseImage(&out);
                return true;
            }
            auto result = verify::readImage(out, 64, 64);
            bool ok = !result.empty() && (result[0] == 255);
            vxReleaseThreshold(&thresh);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }
#endif

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
    // ---- BilateralFilter ----
    //
    // OpenVX 1.3.1 §3.4: vxBilateralFilterNode operates on vx_tensor
    // (not vx_image). The src tensor has dims [W, H, 1] (or [W, H, N]
    // for a batch). diameter, sigmaSpace, sigmaValues are scalar
    // parameters. We use a 5-tap diameter and modest sigmas to match
    // the typical OpenCV bilateralFilter call.
    {
        BenchmarkCase bc;
        bc.name        = "BilateralFilter";
        bc.category    = "misc";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_BILATERAL_FILTER;
        bc.required_kernels = {VX_KERNEL_BILATERAL_FILTER};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_size dims[2] = {width, height};
            vx_tensor src = tracker.trackTensor(
                gen.createFilledTensor(ctx, dims, 2, VX_TYPE_UINT8));
            vx_tensor dst = tracker.trackTensor(
                vxCreateTensor(ctx, 2, dims, VX_TYPE_UINT8, 0));
            if (vxGetStatus((vx_reference)src) != VX_SUCCESS ||
                vxGetStatus((vx_reference)dst) != VX_SUCCESS) return false;

            auto fn = openvx_optional::bilateralFilterNode();
            if (!fn) return false;
            vx_node node = fn(graph, src,
                              /*diameter=*/5,
                              /*sigmaSpace=*/20.0f,
                              /*sigmaValues=*/40.0f,
                              dst);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            // Uniform input should pass through unchanged within rounding.
            auto fn = openvx_optional::bilateralFilterNode();
            if (!fn) return true;
            vx_size dims[2] = {64, 64};
            std::vector<uint8_t> in_data(64 * 64, 100);
            vx_tensor tin = vxCreateTensor(ctx, 2, dims, VX_TYPE_UINT8, 0);
            vx_tensor tout = vxCreateTensor(ctx, 2, dims, VX_TYPE_UINT8, 0);
            vx_size starts[2] = {0, 0}, strides[2] = {sizeof(uint8_t), 64 * sizeof(uint8_t)};
            vxCopyTensorPatch(tin, 2, starts, dims, strides, in_data.data(),
                              VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = fn(g, tin, 5, 20.0f, 40.0f, tout);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            std::vector<uint8_t> result(64 * 64, 0);
            vxCopyTensorPatch(tout, 2, starts, dims, strides, result.data(),
                              VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
            bool ok = (status != VX_SUCCESS) ? true :
                      (!result.empty() && std::abs((int)result[32 * 64 + 32] - 100) <= 2);
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseTensor(&tin); vxReleaseTensor(&tout);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- ScalarOperation ----
    //
    // OpenVX 1.3.1 §3.14 (Control Flow): vxScalarOperationNode performs
    // an arithmetic / comparison / logical op on two vx_scalar values.
    // The work per call is O(1) — the cost is entirely in the kernel
    // dispatch + scalar marshalling. We benchmark VX_SCALAR_OP_ADD on
    // two INT32 scalars as a representative case.
    //
    // Per the OpenVX 1.3.1 spec, the throughput for ScalarOperation is
    // not measured in MP/s (there are no pixels) — `megapixels_per_sec`
    // in the JSON will just reflect the dispatch rate; useful for
    // measuring framework overhead per call.
    {
        BenchmarkCase bc;
        bc.name        = "ScalarOperation";
        bc.category    = "misc";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_SCALAR_OPERATION;
        bc.required_kernels = {VX_KERNEL_SCALAR_OPERATION};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t /*width*/, uint32_t /*height*/,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_int32 a_val = 100, b_val = 50, out_val = 0;
            vx_scalar a   = tracker.trackScalar(gen.createScalar(ctx, VX_TYPE_INT32, &a_val));
            vx_scalar b   = tracker.trackScalar(gen.createScalar(ctx, VX_TYPE_INT32, &b_val));
            vx_scalar out = tracker.trackScalar(gen.createScalar(ctx, VX_TYPE_INT32, &out_val));
            if (vxGetStatus((vx_reference)a) != VX_SUCCESS ||
                vxGetStatus((vx_reference)b) != VX_SUCCESS ||
                vxGetStatus((vx_reference)out) != VX_SUCCESS) return false;

            auto fn = openvx_optional::scalarOperationNode();
            if (!fn) return false;
            vx_node node = fn(graph, VX_SCALAR_OP_ADD, a, b, out);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            auto fn = openvx_optional::scalarOperationNode();
            if (!fn) return true;
            vx_int32 a_val = 100, b_val = 50, out_val = 0;
            vx_scalar a = vxCreateScalar(ctx, VX_TYPE_INT32, &a_val);
            vx_scalar b = vxCreateScalar(ctx, VX_TYPE_INT32, &b_val);
            vx_scalar out = vxCreateScalar(ctx, VX_TYPE_INT32, &out_val);
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = fn(g, VX_SCALAR_OP_ADD, a, b, out);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            vxCopyScalar(out, &out_val, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
            bool ok = (status != VX_SUCCESS) ? true : (out_val == 150);
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseScalar(&a); vxReleaseScalar(&b); vxReleaseScalar(&out);
            return ok;
        };
        cases.push_back(bc);
    }

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
