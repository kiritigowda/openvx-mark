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

std::vector<BenchmarkCase> registerGeometricBenchmarks() {
    std::vector<BenchmarkCase> cases;

    // ScaleImage_Half: scale down by 2x using bilinear interpolation
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

    // WarpAffine: apply an affine warp using bilinear interpolation
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

    // Remap: apply a remap table using bilinear interpolation
    {
        BenchmarkCase bc;
        bc.name = "Remap";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.kernel_enum = VX_KERNEL_REMAP;
        bc.required_kernels = {VX_KERNEL_REMAP};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_image input = gen.createFilledImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(input);

            vx_image output = vxCreateImage(ctx, width, height, VX_DF_IMAGE_U8);
            tracker.trackImage(output);

            vx_remap remap_table = gen.createRemap(ctx, width, height, width, height);
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
            vx_remap remap = vxCreateRemap(ctx, 64, 64, 64, 64);
#if OPENVX_USE_SET_REMAP_POINT
            for (vx_uint32 y = 0; y < 64; y++)
                for (vx_uint32 x = 0; x < 64; x++)
                    vxSetRemapPoint(remap, x, y, (vx_float32)x, (vx_float32)y);
#else
            vx_rectangle_t rect = {0, 0, 64, 64};
            vx_size stride = 64 * sizeof(vx_coordinates2df_t);
            std::vector<vx_coordinates2df_t> coords(64 * 64);
            for (vx_uint32 y = 0; y < 64; y++)
                for (vx_uint32 x = 0; x < 64; x++) {
                    coords[y * 64 + x].x = (vx_float32)x;
                    coords[y * 64 + x].y = (vx_float32)y;
                }
            vxCopyRemapPatch(remap, &rect, stride, coords.data(),
                             VX_TYPE_COORDINATES2DF, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
#endif
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

    return cases;
}
