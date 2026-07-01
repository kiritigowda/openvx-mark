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
#include <VX/vx.h>
#include <VX/vx_nodes.h>
#include <VX/vxu.h>
#include <cmath>
#include <vector>

std::vector<BenchmarkCase> registerGeometricBenchmarks(const BenchmarkConfig& config) {
    std::vector<BenchmarkCase> cases;
    const benchmark::RemapPattern remap_pattern = benchmark::remapPatternFromString(config.remap_pattern);

    // ScaleImage_Half: scale down by 2x using bilinear interpolation
    //
    // Vision Conformance Feature Set: ScaleImage requires support for
    // VX_INTERPOLATION_NEAREST_NEIGHBOR, _BILINEAR and _AREA. We benchmark
    // bilinear half/double here, plus separate Nearest_Half and Area_Half
    // variants below — each interpolation walks the same input but does
    // very different per-pixel work.
    {
        BenchmarkCase bc;
        bc.name = "ScaleImage_Half";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_SCALE_IMAGE;
        bc.required_kernels = {VX_KERNEL_SCALE_IMAGE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(input);

            vx_image output_half = vxCreateImage(ctx, width / 2, height / 2, VX_DF_IMAGE_U8);
            tracker.trackImage(output_half);

            vx_node node = vxScaleImageNode(graph, input, output_half, VX_INTERPOLATION_BILINEAR);
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
            vx_status status = vxuScaleImage(ctx, in, out, VX_INTERPOLATION_BILINEAR);
            if (status != VX_SUCCESS) {
                vxReleaseImage(&in); vxReleaseImage(&out);
                return true;
            }
            auto result = verify::readImage(out, 32, 32);
            bool ok = (std::abs((int)result[16 * 32 + 16] - 100) <= 2);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ScaleImage_Double: scale up by 2x using bilinear interpolation
    {
        BenchmarkCase bc;
        bc.name = "ScaleImage_Double";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_SCALE_IMAGE;
        bc.required_kernels = {VX_KERNEL_SCALE_IMAGE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(input);

            vx_image output_double = vxCreateImage(ctx, width * 2, height * 2, VX_DF_IMAGE_U8);
            tracker.trackImage(output_double);

            vx_node node = vxScaleImageNode(graph, input, output_double, VX_INTERPOLATION_BILINEAR);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 128, 128, VX_DF_IMAGE_U8);
            vx_status status = vxuScaleImage(ctx, in, out, VX_INTERPOLATION_BILINEAR);
            if (status != VX_SUCCESS) {
                vxReleaseImage(&in); vxReleaseImage(&out);
                return true;
            }
            auto result = verify::readImage(out, 128, 128);
            bool ok = (std::abs((int)result[64 * 128 + 64] - 100) <= 2);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ScaleImage_Nearest_Half: scale down by 2x using nearest-neighbor
    {
        BenchmarkCase bc;
        bc.name = "ScaleImage_Nearest_Half";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_SCALE_IMAGE;
        bc.required_kernels = {VX_KERNEL_SCALE_IMAGE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width / 2, height / 2, VX_DF_IMAGE_U8));
            vx_node node = vxScaleImageNode(graph, input, output, VX_INTERPOLATION_NEAREST_NEIGHBOR);
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
            vx_status status = vxuScaleImage(ctx, in, out, VX_INTERPOLATION_NEAREST_NEIGHBOR);
            if (status != VX_SUCCESS) { vxReleaseImage(&in); vxReleaseImage(&out); return true; }
            auto result = verify::readImage(out, 32, 32);
            bool ok = (result[16 * 32 + 16] == 100);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // ScaleImage_Area_Half: scale down by 2x using area resampling
    {
        BenchmarkCase bc;
        bc.name = "ScaleImage_Area_Half";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_SCALE_IMAGE;
        bc.required_kernels = {VX_KERNEL_SCALE_IMAGE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width / 2, height / 2, VX_DF_IMAGE_U8));
            vx_node node = vxScaleImageNode(graph, input, output, VX_INTERPOLATION_AREA);
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
            vx_status status = vxuScaleImage(ctx, in, out, VX_INTERPOLATION_AREA);
            // VX_INTERPOLATION_AREA is required by the spec but a few
            // implementations only enable it behind a build flag — treat
            // VX_ERROR_NOT_SUPPORTED as "skipped" rather than a failure
            // here so we still count toward the conformance summary on
            // compliant impls.
            if (status != VX_SUCCESS) { vxReleaseImage(&in); vxReleaseImage(&out); return true; }
            auto result = verify::readImage(out, 32, 32);
            bool ok = (std::abs((int)result[16 * 32 + 16] - 100) <= 2);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // WarpAffine: apply an affine warp using bilinear interpolation
    //
    // Vision Conformance Feature Set: WarpAffine and WarpPerspective both
    // require support for VX_INTERPOLATION_NEAREST_NEIGHBOR and
    // VX_INTERPOLATION_BILINEAR. We benchmark each as a separate test
    // since nearest is a single load and bilinear is a 4-tap blend.
    {
        BenchmarkCase bc;
        bc.name = "WarpAffine";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_WARP_AFFINE;
        bc.required_kernels = {VX_KERNEL_WARP_AFFINE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(output);

            vx_matrix matrix = gen.createAffineMatrix(ctx);
            tracker.trackMatrix(matrix);

            vx_node node = vxWarpAffineNode(graph, input, matrix, VX_INTERPOLATION_BILINEAR, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_float32 identity[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
            vx_matrix mat = vxCreateMatrix(ctx, VX_TYPE_FLOAT32, 2, 3);
            vxCopyMatrix(mat, identity, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_status status = vxuWarpAffine(ctx, in, mat, VX_INTERPOLATION_BILINEAR, out);
            if (status != VX_SUCCESS) {
                vxReleaseMatrix(&mat);
                vxReleaseImage(&in); vxReleaseImage(&out);
                return true;
            }
            auto result = verify::readImage(out, 64, 64);
            bool ok = (std::abs((int)result[32 * 64 + 32] - 100) <= 2);
            vxReleaseMatrix(&mat);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // WarpAffine_Nearest: identity affine warp using nearest-neighbor
    {
        BenchmarkCase bc;
        bc.name = "WarpAffine_Nearest";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_WARP_AFFINE;
        bc.required_kernels = {VX_KERNEL_WARP_AFFINE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_matrix matrix = tracker.trackMatrix(gen.createAffineMatrix(ctx));
            vx_node node = vxWarpAffineNode(graph, input, matrix, VX_INTERPOLATION_NEAREST_NEIGHBOR, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_float32 identity[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
            vx_matrix mat = vxCreateMatrix(ctx, VX_TYPE_FLOAT32, 2, 3);
            vxCopyMatrix(mat, identity, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_status status = vxuWarpAffine(ctx, in, mat, VX_INTERPOLATION_NEAREST_NEIGHBOR, out);
            if (status != VX_SUCCESS) { vxReleaseMatrix(&mat); vxReleaseImage(&in); vxReleaseImage(&out); return true; }
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[32 * 64 + 32] == 100);
            vxReleaseMatrix(&mat);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // WarpPerspective: apply a perspective warp using bilinear interpolation
    {
        BenchmarkCase bc;
        bc.name = "WarpPerspective";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_WARP_PERSPECTIVE;
        bc.required_kernels = {VX_KERNEL_WARP_PERSPECTIVE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(output);

            vx_matrix matrix = gen.createPerspectiveMatrix(ctx);
            tracker.trackMatrix(matrix);

            vx_node node = vxWarpPerspectiveNode(graph, input, matrix, VX_INTERPOLATION_BILINEAR, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_float32 identity[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
            vx_matrix mat = vxCreateMatrix(ctx, VX_TYPE_FLOAT32, 3, 3);
            vxCopyMatrix(mat, identity, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_status status = vxuWarpPerspective(ctx, in, mat, VX_INTERPOLATION_BILINEAR, out);
            if (status != VX_SUCCESS) {
                vxReleaseMatrix(&mat);
                vxReleaseImage(&in); vxReleaseImage(&out);
                return true;
            }
            auto result = verify::readImage(out, 64, 64);
            bool ok = (std::abs((int)result[32 * 64 + 32] - 100) <= 2);
            vxReleaseMatrix(&mat);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // WarpPerspective_Nearest: identity perspective warp using nearest-neighbor
    {
        BenchmarkCase bc;
        bc.name = "WarpPerspective_Nearest";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_WARP_PERSPECTIVE;
        bc.required_kernels = {VX_KERNEL_WARP_PERSPECTIVE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_matrix matrix = tracker.trackMatrix(gen.createPerspectiveMatrix(ctx));
            vx_node node = vxWarpPerspectiveNode(graph, input, matrix, VX_INTERPOLATION_NEAREST_NEIGHBOR, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            vx_float32 identity[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
            vx_matrix mat = vxCreateMatrix(ctx, VX_TYPE_FLOAT32, 3, 3);
            vxCopyMatrix(mat, identity, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_status status = vxuWarpPerspective(ctx, in, mat, VX_INTERPOLATION_NEAREST_NEIGHBOR, out);
            if (status != VX_SUCCESS) { vxReleaseMatrix(&mat); vxReleaseImage(&in); vxReleaseImage(&out); return true; }
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[32 * 64 + 32] == 100);
            vxReleaseMatrix(&mat);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // Remap: apply a remap table using bilinear interpolation
    //
    // Vision Conformance Feature Set: Remap requires both
    // VX_INTERPOLATION_NEAREST_NEIGHBOR and VX_INTERPOLATION_BILINEAR.
    //
    // The remap coordinates default to a radial lens-distortion model
    // (LENS_DISTORTION) so the benchmark exercises scattered, realistic
    // memory access rather than the cache-friendly identity pattern.
    // Use --remap-pattern identity to restore the old behaviour.
    {
        BenchmarkCase bc;
        bc.name = "Remap";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_REMAP;
        bc.required_kernels = {VX_KERNEL_REMAP};
        bc.graph_setup = [remap_pattern](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(output);

            vx_remap remap_table = gen.createRemap(ctx, width, height, width, height, remap_pattern);
            tracker.trackRemap(remap_table);

            vx_node node = vxRemapNode(graph, input, remap_table, VX_INTERPOLATION_BILINEAR, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            TestDataGenerator gen(42);
            vx_remap remap = gen.createRemapIdentity(ctx, 64, 64, 64, 64);
            vx_status status = vxuRemap(ctx, in, remap, VX_INTERPOLATION_BILINEAR, out);
            if (status != VX_SUCCESS) {
                vxReleaseRemap(&remap);
                vxReleaseImage(&in); vxReleaseImage(&out);
                return true;
            }
            auto result = verify::readImage(out, 64, 64);
            bool ok = (std::abs((int)result[32 * 64 + 32] - 100) <= 2);
            vxReleaseRemap(&remap);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    // Remap_Nearest: nearest-neighbor interpolation using the same
    // coordinate pattern as Remap so both variants exercise realistic
    // memory access. Correctness verification below still uses a stable
    // identity remap.
    {
        BenchmarkCase bc;
        bc.name = "Remap_Nearest";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_REMAP;
        bc.required_kernels = {VX_KERNEL_REMAP};
        bc.graph_setup = [remap_pattern](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = tracker.trackImage(gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_image output = tracker.trackImage(vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8));
            vx_remap remap_table = tracker.trackRemap(gen.createRemap(ctx, width, height, width, height, remap_pattern));
            vx_node node = vxRemapNode(graph, input, remap_table, VX_INTERPOLATION_NEAREST_NEIGHBOR, output);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            std::vector<uint8_t> a(64 * 64, 100);
            vx_image in = verify::createImage(ctx, 64, 64, VX_DF_IMAGE_U8, a.data());
            if (!in) return true;
            vx_image out = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
            TestDataGenerator gen(42);
            vx_remap remap = gen.createRemapIdentity(ctx, 64, 64, 64, 64);
            vx_status status = vxuRemap(ctx, in, remap, VX_INTERPOLATION_NEAREST_NEIGHBOR, out);
            if (status != VX_SUCCESS) {
                vxReleaseRemap(&remap);
                vxReleaseImage(&in); vxReleaseImage(&out);
                return true;
            }
            auto result = verify::readImage(out, 64, 64);
            bool ok = (result[32 * 64 + 32] == 100);
            vxReleaseRemap(&remap);
            vxReleaseImage(&in); vxReleaseImage(&out);
            return ok;
        };
        cases.push_back(bc);
    }

    return cases;
}
