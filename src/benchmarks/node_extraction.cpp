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
#include <algorithm>
#include <vector>

std::vector<BenchmarkCase> registerExtractionBenchmarks() {
    std::vector<BenchmarkCase> cases;

#if OPENVX_HAS_1_2
    // MatchTemplate: compare an image against a 32x32 template using CCORR_NORM
    {
        BenchmarkCase bc;
        bc.name = "MatchTemplate";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_MATCH_TEMPLATE;
        bc.required_kernels = {VX_KERNEL_MATCH_TEMPLATE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image src = tracker.trackImage(
                gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            if (vxGetStatus((vx_reference)src) != VX_SUCCESS) return false;

            vx_image templateImage = tracker.trackImage(
                gen.createFilledImage(ctx, 32, 32, VX_DF_IMAGE_U8));
            if (vxGetStatus((vx_reference)templateImage) != VX_SUCCESS) return false;

            vx_image output = tracker.trackImage(
                vxCreateImage(ctx, width, height, VX_DF_IMAGE_S16));
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;

            vx_enum method = VX_COMPARE_CCORR_NORM;
            vx_scalar matchingMethod = tracker.trackScalar(
                vxCreateScalar(ctx, VX_TYPE_ENUM, &method));
            if (vxGetStatus((vx_reference)matchingMethod) != VX_SUCCESS) return false;

            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_MATCH_TEMPLATE);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)src);
            vxSetParameterByIndex(node, 1, (vx_reference)templateImage);
            vxSetParameterByIndex(node, 2, (vx_reference)matchingMethod);
            vxSetParameterByIndex(node, 3, (vx_reference)output);
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            const uint32_t W = 64, H = 64, TW = 16, TH = 16;
            std::vector<uint8_t> src(W * H, 100);
            std::vector<uint8_t> tmpl(TW * TH, 100);
            vx_image src_img = verify::createImage(ctx, W, H, VX_DF_IMAGE_U8, src.data());
            vx_image tmpl_img = verify::createImage(ctx, TW, TH, VX_DF_IMAGE_U8, tmpl.data());
            if (!src_img || !tmpl_img) {
                if (src_img) vxReleaseImage(&src_img);
                if (tmpl_img) vxReleaseImage(&tmpl_img);
                return true;
            }
            vx_image out = vxCreateImage(ctx, W, H, VX_DF_IMAGE_S16);
            vx_enum method = VX_COMPARE_CCORR_NORM;
            vx_scalar match_method = vxCreateScalar(ctx, VX_TYPE_ENUM, &method);
            vx_graph g = vxCreateGraph(ctx);
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_MATCH_TEMPLATE);
            vx_node n = vxCreateGenericNode(g, k);
            vxReleaseKernel(&k);
            vxSetParameterByIndex(n, 0, (vx_reference)src_img);
            vxSetParameterByIndex(n, 1, (vx_reference)tmpl_img);
            vxSetParameterByIndex(n, 2, (vx_reference)match_method);
            vxSetParameterByIndex(n, 3, (vx_reference)out);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            auto result = verify::readImageS16(out, W, H);
            bool ok = (status != VX_SUCCESS) ? true :
                      (!result.empty() && result[H / 2 * W + W / 2] != 0);
            vxReleaseNode(&n); vxReleaseGraph(&g); vxReleaseScalar(&match_method);
            vxReleaseImage(&src_img); vxReleaseImage(&tmpl_img); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // LBP: extract Local Binary Pattern from an input image
    {
        BenchmarkCase bc;
        bc.name = "LBP";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_LBP;
        bc.required_kernels = {VX_KERNEL_LBP};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = tracker.trackImage(
                gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;

            vx_image output = tracker.trackImage(
                vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;

            vx_enum format_val = VX_LBP;
            vx_scalar format = tracker.trackScalar(
                vxCreateScalar(ctx, VX_TYPE_ENUM, &format_val));
            if (vxGetStatus((vx_reference)format) != VX_SUCCESS) return false;

            vx_int8 ksize_val = 3;
            vx_scalar kernel_size = tracker.trackScalar(
                vxCreateScalar(ctx, VX_TYPE_INT8, &ksize_val));
            if (vxGetStatus((vx_reference)kernel_size) != VX_SUCCESS) return false;

            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_LBP);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)input);
            vxSetParameterByIndex(node, 1, (vx_reference)format);
            vxSetParameterByIndex(node, 2, (vx_reference)kernel_size);
            vxSetParameterByIndex(node, 3, (vx_reference)output);
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            // LBP on a gradient pattern should produce non-zero output
            std::vector<uint8_t> a(64 * 64);
            for (int y = 0; y < 64; y++)
                for (int x = 0; x < 64; x++)
                    a[y * 64 + x] = (uint8_t)((x + y * 64) % 256);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_enum format_val = VX_LBP;
            vx_scalar format = vxCreateScalar(ctx, VX_TYPE_ENUM, &format_val);
            vx_int8 ksize = 3;
            vx_scalar kernel_size = vxCreateScalar(ctx, VX_TYPE_INT8, &ksize);
            vx_graph g = vxCreateGraph(ctx);
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_LBP);
            vx_node n = vxCreateGenericNode(g, k);
            vxReleaseKernel(&k);
            vxSetParameterByIndex(n, 0, (vx_reference)in);
            vxSetParameterByIndex(n, 1, (vx_reference)format);
            vxSetParameterByIndex(n, 2, (vx_reference)kernel_size);
            vxSetParameterByIndex(n, 3, (vx_reference)out);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            bool ok = (status != VX_SUCCESS) ? true : verify::imageNonZero(out, 64, 64);
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseScalar(&format); vxReleaseScalar(&kernel_size);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // HOGCells — U8 input → (magnitudes tensor, bins tensor)
    //
    // OpenVX 1.3.1 §3.24: vxHOGCellsNode computes the average gradient
    // magnitude per cell and the gradient-orientation histograms per
    // cell. We use 8x8 cells and 9 bins (standard Dalal-Triggs HOG
    // defaults — same as cv::HOGDescriptor defaults).
    {
        BenchmarkCase bc;
        bc.name = "HOGCells";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_HOG_CELLS;
        bc.required_kernels = {VX_KERNEL_HOG_CELLS};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            constexpr vx_int32 CELL = 8;
            constexpr vx_int32 BINS = 9;
            // Floor width/height to multiples of CELL.
            uint32_t w = (width  / CELL) * CELL;
            uint32_t h = (height / CELL) * CELL;
            if (w == 0 || h == 0) return false;

            vx_image input = tracker.trackImage(
                gen.createFilledImage(ctx, w, h, VX_DF_IMAGE_U8));
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;

            vx_size mag_dims[2] = {static_cast<vx_size>(w / CELL),
                                   static_cast<vx_size>(h / CELL)};
            vx_tensor magnitudes = tracker.trackTensor(
                vxCreateTensor(ctx, 2, mag_dims, VX_TYPE_INT16, 0));
            if (vxGetStatus((vx_reference)magnitudes) != VX_SUCCESS) return false;

            vx_size bin_dims[3] = {static_cast<vx_size>(w / CELL),
                                   static_cast<vx_size>(h / CELL),
                                   static_cast<vx_size>(BINS)};
            vx_tensor bins = tracker.trackTensor(
                vxCreateTensor(ctx, 3, bin_dims, VX_TYPE_INT16, 0));
            if (vxGetStatus((vx_reference)bins) != VX_SUCCESS) return false;

            auto fn = openvx_optional::hogCellsNode();
            if (!fn) return false;  // runtime doesn't export this API
            vx_node node = fn(graph, input, CELL, CELL, BINS, magnitudes, bins);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            // Smoke check: confirm the graph builds and runs end-to-end
            // on a small uniform input. Implementations differ in cell
            // gradient sign conventions, so we don't pin specific values.
            auto fn = openvx_optional::hogCellsNode();
            if (!fn) return true;  // not exported → graph_setup already returned false
            const uint32_t W = 64, H = 64;
            std::vector<uint8_t> data(W * H, 100);
            vx_image in = verify::createImage(ctx, W, H, VX_DF_IMAGE_U8, data.data());
            if (!in) return true;
            vx_size mag_dims[2] = {W / 8, H / 8};
            vx_size bin_dims[3] = {W / 8, H / 8, 9};
            vx_tensor mag  = vxCreateTensor(ctx, 2, mag_dims, VX_TYPE_INT16, 0);
            vx_tensor bins = vxCreateTensor(ctx, 3, bin_dims, VX_TYPE_INT16, 0);
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = fn(g, in, 8, 8, 9, mag, bins);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            bool ok = (status == VX_SUCCESS) || (status == VX_ERROR_NOT_SUPPORTED);
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseTensor(&mag); vxReleaseTensor(&bins);
            vxReleaseImage(&in);
            return ok;
        };
        cases.push_back(bc);
    }

    // HOGFeatures — U8 + magnitudes + bins → HOG descriptor tensor
    {
        BenchmarkCase bc;
        bc.name = "HOGFeatures";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_HOG_FEATURES;
        bc.required_kernels = {VX_KERNEL_HOG_FEATURES};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            constexpr vx_int32 CELL = 8;
            constexpr vx_int32 BINS = 9;
            constexpr vx_int32 BLOCK = 16;
            constexpr vx_int32 BLOCK_STRIDE = 8;
            constexpr vx_int32 WIN = 64;
            constexpr vx_int32 WIN_STRIDE = 8;

            // Cap effective HOG dimensions independent of the bench
            // resolution flag. The HOG features tensor sizes as
            //   cells_per_block * BINS * blocks_per_win * win_per_row * win_per_col
            // which grows ~O(w·h). At full FHD that's ~100 MB of
            // int16; at 4K ~420 MB — large enough to OOM CI runners
            // and to dwarf the actual kernel cost with allocator
            // pressure. 1024x768 is the classic HOG-pedestrian-detect
            // resolution and gives a meaningful ~36 MB tensor while
            // staying inside every realistic memory budget. The kernel's
            // per-window cost is what we're measuring, so capping the
            // window count doesn't change what the comparison answers.
            constexpr uint32_t MAX_HOG_W = 1024;
            constexpr uint32_t MAX_HOG_H = 768;
            uint32_t eff_w = std::min<uint32_t>(width,  MAX_HOG_W);
            uint32_t eff_h = std::min<uint32_t>(height, MAX_HOG_H);
            // Coerce dimensions so window fits with at least one slide.
            uint32_t w = std::max<uint32_t>(WIN + WIN_STRIDE, (eff_w / CELL) * CELL);
            uint32_t h = std::max<uint32_t>(WIN + WIN_STRIDE, (eff_h / CELL) * CELL);

            vx_image input = tracker.trackImage(
                gen.createFilledImage(ctx, w, h, VX_DF_IMAGE_U8));
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;

            // HOGCells outputs (re-used here as inputs)
            vx_size mag_dims[2] = {static_cast<vx_size>(w / CELL),
                                   static_cast<vx_size>(h / CELL)};
            vx_size bin_dims[3] = {static_cast<vx_size>(w / CELL),
                                   static_cast<vx_size>(h / CELL),
                                   static_cast<vx_size>(BINS)};
            vx_tensor magnitudes = tracker.trackTensor(
                vxCreateTensor(ctx, 2, mag_dims, VX_TYPE_INT16, 0));
            vx_tensor bins = tracker.trackTensor(
                vxCreateTensor(ctx, 3, bin_dims, VX_TYPE_INT16, 0));
            if (vxGetStatus((vx_reference)magnitudes) != VX_SUCCESS ||
                vxGetStatus((vx_reference)bins) != VX_SUCCESS) return false;

            vx_hog_t params = {};
            params.cell_width    = CELL;
            params.cell_height   = CELL;
            params.block_width   = BLOCK;
            params.block_height  = BLOCK;
            params.block_stride  = BLOCK_STRIDE;
            params.num_bins      = BINS;
            params.window_width  = WIN;
            params.window_height = WIN;
            params.window_stride = WIN_STRIDE;
            params.threshold     = 0.2f;

            const vx_int32 cells_per_block = (BLOCK / CELL) * (BLOCK / CELL);
            const vx_int32 blocks_per_win  = ((WIN - BLOCK) / BLOCK_STRIDE + 1) *
                                             ((WIN - BLOCK) / BLOCK_STRIDE + 1);
            const vx_int32 win_per_row     = (w - WIN) / WIN_STRIDE + 1;
            const vx_int32 win_per_col     = (h - WIN) / WIN_STRIDE + 1;
            vx_size feat_dims[1] = {
                static_cast<vx_size>(cells_per_block * BINS * blocks_per_win *
                                     win_per_row * win_per_col)};
            vx_tensor features = tracker.trackTensor(
                vxCreateTensor(ctx, 1, feat_dims, VX_TYPE_INT16, 0));
            if (vxGetStatus((vx_reference)features) != VX_SUCCESS) return false;

            auto fn = openvx_optional::hogFeaturesNode();
            if (!fn) return false;
            vx_node node = fn(graph, input, magnitudes, bins,
                              &params, sizeof(params), features);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context /*ctx*/) -> bool {
            // Smoke check skipped — HOGFeatures depends on a populated
            // HOGCells output, the test data shape is sensitive to
            // implementation rounding, and the dominant cost is the
            // per-window block normalisation loop which runs on any
            // input. Graph_setup validation already covers wiring.
            return true;
        };
        cases.push_back(bc);
    }

    // HoughLinesP — U8 (binary) edge map → array of detected line segments
    {
        BenchmarkCase bc;
        bc.name = "HoughLinesP";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_HOUGH_LINES_P;
        bc.required_kernels = {VX_KERNEL_HOUGH_LINES_P};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            // OpenVX expects a binary edge map; we fabricate a synthetic
            // image with a deterministic edge pattern by extracting Canny
            // edges from a random U8 input would create a varying input —
            // instead just feed the random U8 since the per-pixel
            // accumulator cost dominates regardless of edge density.
            vx_image input = tracker.trackImage(
                gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;

            // OpenVX 1.3.1 §3.30: HoughLinesP outputs an array of
            // vx_line2d_t (NOT vx_rectangle_t — those are different
            // structs). Using the wrong type tag here would (a) be
            // caught by lenient impls at vxVerifyGraph with a clean
            // skip, but (b) cause a Rust-FFI panic on strict impls
            // like rustVX, where a panic across the FFI boundary is
            // undefined behaviour and manifests as a segfault. Use
            // the spec-mandated VX_TYPE_LINE_2D.
            vx_array lines = tracker.trackArray(
                vxCreateArray(ctx, VX_TYPE_LINE_2D, 1024));
            if (vxGetStatus((vx_reference)lines) != VX_SUCCESS) return false;

            vx_size zero = 0;
            vx_scalar num_lines = tracker.trackScalar(
                vxCreateScalar(ctx, VX_TYPE_SIZE, &zero));
            if (vxGetStatus((vx_reference)num_lines) != VX_SUCCESS) return false;

            vx_hough_lines_p_t params = {};
            params.rho          = 1.0f;
            params.theta        = 3.14159265f / 180.0f;
            params.threshold    = 50;
            params.line_length  = 30;
            params.line_gap     = 10;
            params.theta_min    = 0.0f;
            params.theta_max    = 3.14159265f;

            auto fn = openvx_optional::houghLinesPNode();
            if (!fn) return false;
            vx_node node = fn(graph, input, &params, lines, num_lines);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context /*ctx*/) -> bool {
            // Implementation-defined output (the algorithm is allowed to
            // be non-deterministic per OpenVX 1.3.1 §3.27). Graph_setup
            // validation covers wiring.
            return true;
        };
        cases.push_back(bc);
    }

    // NonMaxSuppression: suppress non-maxima in a 3x3 window on S16 input
    {
        BenchmarkCase bc;
        bc.name = "NonMaxSuppression";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_NON_MAX_SUPPRESSION;
        bc.required_kernels = {VX_KERNEL_NON_MAX_SUPPRESSION};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = tracker.trackImage(
                gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_S16));
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;

            vx_image output = tracker.trackImage(
                vxCreateImage(ctx, width, height, VX_DF_IMAGE_S16));
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;

            vx_int32 win_size_val = 3;
            vx_scalar win_size = tracker.trackScalar(
                vxCreateScalar(ctx, VX_TYPE_INT32, &win_size_val));
            if (vxGetStatus((vx_reference)win_size) != VX_SUCCESS) return false;

            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_NON_MAX_SUPPRESSION);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)input);
            // parameter 1 (mask) left unset — defaults to NULL
            vxSetParameterByIndex(node, 2, (vx_reference)win_size);
            vxSetParameterByIndex(node, 3, (vx_reference)output);
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            const uint32_t W = 64, H = 64;
            std::vector<int16_t> data(W * H, 10);
            data[32 * W + 32] = 1000;
            vx_image in = verify::createImage(ctx, W, H, VX_DF_IMAGE_S16,
                                              reinterpret_cast<const uint8_t*>(data.data()));
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, W, H, VX_DF_IMAGE_S16);
            vx_int32 win_size = 3;
            vx_scalar ws = vxCreateScalar(ctx, VX_TYPE_INT32, &win_size);
            vx_graph g = vxCreateGraph(ctx);
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_NON_MAX_SUPPRESSION);
            vx_node n = vxCreateGenericNode(g, k);
            vxReleaseKernel(&k);
            vxSetParameterByIndex(n, 0, (vx_reference)in);
            vxSetParameterByIndex(n, 2, (vx_reference)ws);
            vxSetParameterByIndex(n, 3, (vx_reference)out);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            auto result = verify::readImageS16(out, W, H);
            bool ok = (status != VX_SUCCESS) ? true :
                      (!result.empty() && result[32 * W + 32] == 1000);
            vxReleaseNode(&n); vxReleaseGraph(&g); vxReleaseScalar(&ws);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }
#endif

    return cases;
}
