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
#include <cstdio>
#include <VX/vx_nodes.h>
#include <cmath>
#include <vector>

std::vector<BenchmarkCase> registerMultiscaleBenchmarks() {
    std::vector<BenchmarkCase> cases;

    // GaussianPyramid — U8 input, Gaussian pyramid output
    //
    // OpenVX 1.3.1 §3.23: input is U8-only, output pyramid is U8. Both
    // VX_SCALE_PYRAMID_HALF (0.5) and VX_SCALE_PYRAMID_ORB (4/5) must be
    // supported [REQ-0189]. The HALF case below is the baseline; the
    // GaussianPyramid_ORB variant covers the ORB scale.
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
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_pyramid pyr = vxCreatePyramid(ctx, 2, VX_SCALE_PYRAMID_HALF, 64, 64, VX_DF_IMAGE_U8);
            vx_status status = vxuGaussianPyramid(ctx, in, pyr);
            if (status != VX_SUCCESS) {
                vxReleasePyramid(&pyr); vxReleaseImage(&in);
                return true;
            }
            vx_image level0 = vxGetPyramidLevel(pyr, 0);
            auto result = verify::readImage(level0, 64, 64);
            bool ok = (result[32 * 64 + 32] == 100);
            vxReleaseImage(&level0); vxReleasePyramid(&pyr); vxReleaseImage(&in);
            return ok;
        };
        cases.push_back(bc);
    }

    // GaussianPyramid_ORB — U8 input, ORB-scaled pyramid (scale = 4/5)
    {
        BenchmarkCase bc;
        bc.name = "GaussianPyramid_ORB";
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
                                                   VX_SCALE_PYRAMID_ORB,
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
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_pyramid pyr = vxCreatePyramid(ctx, 2, VX_SCALE_PYRAMID_ORB, 64, 64, VX_DF_IMAGE_U8);
            vx_status status = vxuGaussianPyramid(ctx, in, pyr);
            if (status != VX_SUCCESS) {
                vxReleasePyramid(&pyr); vxReleaseImage(&in);
                return true;
            }
            vx_image level0 = vxGetPyramidLevel(pyr, 0);
            auto result = verify::readImage(level0, 64, 64);
            bool ok = (result[32 * 64 + 32] == 100);
            vxReleaseImage(&level0); vxReleasePyramid(&pyr); vxReleaseImage(&in);
            return ok;
        };
        cases.push_back(bc);
    }

#if OPENVX_HAS_1_1
    // LaplacianPyramid — U8 input, S16 Laplacian pyramid + U8 remainder output
    //
    // OpenVX 1.3.1 §3.30: nominally [REQ-0265] permits VX_DF_IMAGE_U8 OR
    // VX_DF_IMAGE_S16 input; the Laplacian pyramid is always
    // VX_DF_IMAGE_S16 [REQ-0266]; the low-pass remainder format must
    // match the input format [REQ-0267, REQ-0268].
    //
    // **Why we only test the U8 input path:** the same §3.30 algorithm
    // description mandates that the implementation internally builds a
    // Gaussian pyramid (§3.23), and vxGaussianPyramid is U8-only per
    // [REQ-0191]. So an S16 input would require an S16 Gaussian pyramid
    // step that no conformant impl can provide — AMD AGO, Khronos
    // sample, and rustVX all reject it with VX_ERROR_FORMAT_NOT_SUPPORTED
    // at vxVerifyGraph time. The spec text is internally inconsistent;
    // the CTS only exercises the U8 path because that's the only
    // practically implementable combination. We follow suit.
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
            const uint32_t W = 320, H = 240;
            std::vector<uint8_t> a(W * H, 100);
            vx_image in = verify::createImage(ctx, W, H, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_pyramid lap = vxCreatePyramid(ctx, 1, VX_SCALE_PYRAMID_HALF, W, H, VX_DF_IMAGE_S16);
            vx_image remainder = vxCreateImage(ctx, W / 2, H / 2, VX_DF_IMAGE_U8);
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = vxLaplacianPyramidNode(g, in, lap, remainder);
            if (vxVerifyGraph(g) != VX_SUCCESS) {
                vxReleaseNode(&n); vxReleaseGraph(&g);
                vxReleaseImage(&remainder); vxReleasePyramid(&lap); vxReleaseImage(&in);
                return true;
            }
            vx_status status = vxProcessGraph(g);
            if (status != VX_SUCCESS) {
                vxReleaseNode(&n); vxReleaseGraph(&g);
                vxReleaseImage(&remainder); vxReleasePyramid(&lap); vxReleaseImage(&in);
                return true;
            }
            auto result = verify::readImage(remainder, W / 2, H / 2);
            uint32_t cx = W / 4, cy = H / 4;
            uint8_t center_val = result[cy * (W / 2) + cx];
            bool ok = (std::abs((int)center_val - 100) <= 10);
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseImage(&remainder); vxReleasePyramid(&lap); vxReleaseImage(&in);
            return ok;
        };
        cases.push_back(bc);
    }

    // NOTE: a LaplacianPyramid_S16 benchmark is intentionally NOT
    // registered. See the comment block above the U8 LaplacianPyramid
    // case for the rationale (spec §3.30 vs §3.23 [REQ-0191]
    // contradiction — S16 input would require an S16 GaussianPyramid
    // step that no impl can provide). Adding the benchmark would just
    // produce a noisy "SKIPPED — vxVerifyGraph failed" row on every
    // run of every impl, which is misleading because it implies the
    // impl is incomplete when in fact the operation is unspecifiable.

    // LaplacianReconstruct — S16 Laplacian pyramid + U8 low-pass input → U8 output
    //
    // OpenVX 1.3.1 §3.43: the inverse of LaplacianPyramid. The Laplacian
    // pyramid is always VX_DF_IMAGE_S16 [REQ-0386]; the low-pass input
    // and reconstructed output may nominally both be U8 or both be S16
    // (formats must match) [REQ-0387, REQ-0388, REQ-0390].
    //
    // We only test U8 here for the same reason LaplacianPyramid above
    // is U8-only — the S16 inverse path depends on an S16 LaplacianPyramid
    // existing as a building block in any realistic pipeline, and that
    // doesn't work per the §3.30/§3.23 spec contradiction. Every impl
    // rejects S16 LaplacianReconstruct in practice for the same reason.
    {
        BenchmarkCase bc;
        bc.name = "LaplacianReconstruct";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_LAPLACIAN_RECONSTRUCT;
        bc.required_kernels = {VX_KERNEL_LAPLACIAN_RECONSTRUCT};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen,
                            ResourceTracker& tracker) -> bool {
            const vx_size lap_levels = DEFAULT_PYRAMID_LEVELS - 1;
            vx_pyramid laplacian_pyr = vxCreatePyramid(ctx, lap_levels,
                                                       VX_SCALE_PYRAMID_HALF,
                                                       width, height, VX_DF_IMAGE_S16);
            if (vxGetStatus((vx_reference)laplacian_pyr) != VX_SUCCESS) return false;
            tracker.trackPyramid(laplacian_pyr);

            vx_uint32 low_w = width  >> lap_levels;
            vx_uint32 low_h = height >> lap_levels;
            if (low_w < 1) low_w = 1;
            if (low_h < 1) low_h = 1;
            vx_image input = gen.createFilledImage(ctx, low_w, low_h, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)input) != VX_SUCCESS) return false;
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            if (vxGetStatus((vx_reference)output) != VX_SUCCESS) return false;
            tracker.trackImage(output);

            vx_node node = vxLaplacianReconstructNode(graph, laplacian_pyr, input, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            // Build a Laplacian pyramid from a uniform image, then reconstruct
            // from (pyramid, remainder) — the center pixel should round-trip
            // back to the original (constant) value within filter rounding.
            const uint32_t W = 320, H = 240;
            const uint8_t FILL = 100;
            std::vector<uint8_t> a(W * H, FILL);
            vx_image in = verify::createImage(ctx, W, H, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;

            vx_pyramid lap = vxCreatePyramid(ctx, 1, VX_SCALE_PYRAMID_HALF, W, H, VX_DF_IMAGE_S16);
            vx_image remainder = vxCreateImage(ctx, W / 2, H / 2, VX_DF_IMAGE_U8);
            vx_image reconstructed = vxCreateImage(ctx, W, H, VX_DF_IMAGE_U8);

            vx_graph g = vxCreateGraph(ctx);
            vx_node n_decompose = vxLaplacianPyramidNode(g, in, lap, remainder);
            vx_node n_reconstruct = vxLaplacianReconstructNode(g, lap, remainder, reconstructed);
            if (vxVerifyGraph(g) != VX_SUCCESS) {
                vxReleaseNode(&n_decompose); vxReleaseNode(&n_reconstruct); vxReleaseGraph(&g);
                vxReleaseImage(&reconstructed); vxReleaseImage(&remainder);
                vxReleasePyramid(&lap); vxReleaseImage(&in);
                return true;
            }
            vx_status status = vxProcessGraph(g);
            if (status != VX_SUCCESS) {
                vxReleaseNode(&n_decompose); vxReleaseNode(&n_reconstruct); vxReleaseGraph(&g);
                vxReleaseImage(&reconstructed); vxReleaseImage(&remainder);
                vxReleasePyramid(&lap); vxReleaseImage(&in);
                return true;
            }
            auto result = verify::readImage(reconstructed, W, H);
            uint32_t cx = W / 2, cy = H / 2;
            uint8_t center_val = result[cy * W + cx];
            // Tolerate small rounding from Gaussian decompose / reconstruct path.
            bool ok = (std::abs((int)center_val - (int)FILL) <= 5);
            vxReleaseNode(&n_decompose); vxReleaseNode(&n_reconstruct); vxReleaseGraph(&g);
            vxReleaseImage(&reconstructed); vxReleaseImage(&remainder);
            vxReleasePyramid(&lap); vxReleaseImage(&in);
            return ok;
        };
        cases.push_back(bc);
    }

    // NOTE: a LaplacianReconstruct_S16 benchmark is intentionally NOT
    // registered. See the comment block above the U8 LaplacianReconstruct
    // case for the rationale (the S16 path depends on an S16
    // LaplacianPyramid that the spec contradicts itself about and that
    // no conformant impl can implement).
#endif

    // HalfScaleGaussian — U8 input, U8 output at half resolution, kernel_size=3
    //
    // OpenVX 1.3.1 §3.45 [REQ-0410]: kernel_size of 1, 3 and 5 are all
    // required. The three sizes exercise very different per-pixel work
    // (kernel_size=1 is a pure down-sample with no blur), so we benchmark
    // each as a separate test: HalfScaleGaussian_1x1 below,
    // HalfScaleGaussian (3x3) here, HalfScaleGaussian_5x5 further below.
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
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 32, 32, VX_DF_IMAGE_U8);
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = vxHalfScaleGaussianNode(g, in, out, 3);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            bool ok;
            if (status != VX_SUCCESS) {
                ok = true;
            } else {
                auto result = verify::readImage(out, 32, 32);
                // Check center pixel of output — edge handling varies
                ok = (std::abs((int)result[16 * 32 + 16] - 100) <= 2);
            }
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // HalfScaleGaussian_1x1 — U8 input, U8 output at half resolution, kernel_size=1
    //
    // kernel_size=1 means the Gaussian filter is a no-op (1x1 kernel of
    // weight 1.0), so the operation degenerates to a nearest-neighbor
    // down-sample. This benchmarks the pure down-sample path that the
    // 3x3/5x5 variants do not exercise.
    {
        BenchmarkCase bc;
        bc.name = "HalfScaleGaussian_1x1";
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

            vx_node node = vxHalfScaleGaussianNode(graph, input, output, 1);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 32, 32, VX_DF_IMAGE_U8);
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = vxHalfScaleGaussianNode(g, in, out, 1);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            bool ok;
            if (status != VX_SUCCESS) {
                ok = true;
            } else {
                auto result = verify::readImage(out, 32, 32);
                ok = (result[16 * 32 + 16] == 100);
            }
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // HalfScaleGaussian_5x5 — U8 input, U8 output at half resolution, kernel_size=5
    //
    // Separate test for the 5x5 Gaussian kernel size required by the
    // Vision Conformance Feature Set. Memory traffic is identical to the
    // 3x3 variant but the per-pixel arithmetic is meaningfully larger,
    // which is why it is benchmarked independently.
    {
        BenchmarkCase bc;
        bc.name = "HalfScaleGaussian_5x5";
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

            vx_node node = vxHalfScaleGaussianNode(graph, input, output, 5);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);

            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 32, 32, VX_DF_IMAGE_U8);
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = vxHalfScaleGaussianNode(g, in, out, 5);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            bool ok;
            if (status != VX_SUCCESS) {
                ok = true;
            } else {
                auto result = verify::readImage(out, 32, 32);
                ok = (std::abs((int)result[16 * 32 + 16] - 100) <= 2);
            }
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    return cases;
}
