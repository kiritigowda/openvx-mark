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
#include <VX/vx_nodes.h>
#include <vector>

std::vector<BenchmarkCase> registerTensorBenchmarks()
{
    std::vector<BenchmarkCase> cases;

#if OPENVX_HAS_1_2
    // ---- TensorAdd ----
    {
        BenchmarkCase bc;
        bc.name        = "TensorAdd";
        bc.category    = "tensor";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_TENSOR_ADD;
        bc.required_kernels = {VX_KERNEL_TENSOR_ADD};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_size tw = (width > 1024) ? 1024 : width;
            vx_size th = (height > 1024) ? 1024 : height;
            vx_size dims[2] = {tw, th};
            vx_tensor input1 = tracker.trackTensor(gen.createFilledTensor(ctx, dims, 2, VX_TYPE_INT16));
            vx_tensor input2 = tracker.trackTensor(gen.createFilledTensor(ctx, dims, 2, VX_TYPE_INT16));
            vx_tensor output = tracker.trackTensor(vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0));
            vx_enum policy_val = VX_CONVERT_POLICY_SATURATE;
            vx_scalar policy = tracker.trackScalar(gen.createScalar(ctx, VX_TYPE_ENUM, &policy_val));
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_ADD);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)input1);
            vxSetParameterByIndex(node, 1, (vx_reference)input2);
            vxSetParameterByIndex(node, 2, (vx_reference)policy);
            vxSetParameterByIndex(node, 3, (vx_reference)output);
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            vx_size dims[2] = {64, 64};
            std::vector<int16_t> a_data(64 * 64, 10), b_data(64 * 64, 20);
            vx_tensor t1 = vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0);
            vx_tensor t2 = vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0);
            vx_tensor tout = vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0);
            vx_size starts[2] = {0, 0}, strides[2] = {sizeof(int16_t), 64 * sizeof(int16_t)};
            vxCopyTensorPatch(t1, 2, starts, dims, strides, a_data.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vxCopyTensorPatch(t2, 2, starts, dims, strides, b_data.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_enum policy_val = VX_CONVERT_POLICY_SATURATE;
            vx_scalar policy = vxCreateScalar(ctx, VX_TYPE_ENUM, &policy_val);
            vx_graph g = vxCreateGraph(ctx);
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_ADD);
            vx_node n = vxCreateGenericNode(g, k);
            vxSetParameterByIndex(n, 0, (vx_reference)t1);
            vxSetParameterByIndex(n, 1, (vx_reference)t2);
            vxSetParameterByIndex(n, 2, (vx_reference)policy);
            vxSetParameterByIndex(n, 3, (vx_reference)tout);
            vxVerifyGraph(g); vxProcessGraph(g);
            std::vector<int16_t> result(64 * 64, 0);
            vxCopyTensorPatch(tout, 2, starts, dims, strides, result.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
            bool ok = (result[0] == 30);
            vxReleaseKernel(&k); vxReleaseNode(&n); vxReleaseGraph(&g); vxReleaseScalar(&policy);
            vxReleaseTensor(&t1); vxReleaseTensor(&t2); vxReleaseTensor(&tout);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- TensorSub ----
    {
        BenchmarkCase bc;
        bc.name        = "TensorSub";
        bc.category    = "tensor";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_TENSOR_SUBTRACT;
        bc.required_kernels = {VX_KERNEL_TENSOR_SUBTRACT};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_size tw = (width > 1024) ? 1024 : width;
            vx_size th = (height > 1024) ? 1024 : height;
            vx_size dims[2] = {tw, th};
            vx_tensor input1 = tracker.trackTensor(gen.createFilledTensor(ctx, dims, 2, VX_TYPE_INT16));
            vx_tensor input2 = tracker.trackTensor(gen.createFilledTensor(ctx, dims, 2, VX_TYPE_INT16));
            vx_tensor output = tracker.trackTensor(vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0));
            vx_enum policy_val = VX_CONVERT_POLICY_SATURATE;
            vx_scalar policy = tracker.trackScalar(gen.createScalar(ctx, VX_TYPE_ENUM, &policy_val));
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_SUBTRACT);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)input1);
            vxSetParameterByIndex(node, 1, (vx_reference)input2);
            vxSetParameterByIndex(node, 2, (vx_reference)policy);
            vxSetParameterByIndex(node, 3, (vx_reference)output);
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            vx_size dims[2] = {64, 64};
            std::vector<int16_t> a_data(64 * 64, 30), b_data(64 * 64, 10);
            vx_tensor t1 = vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0);
            vx_tensor t2 = vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0);
            vx_tensor tout = vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0);
            vx_size starts[2] = {0, 0}, strides[2] = {sizeof(int16_t), 64 * sizeof(int16_t)};
            vxCopyTensorPatch(t1, 2, starts, dims, strides, a_data.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vxCopyTensorPatch(t2, 2, starts, dims, strides, b_data.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_enum policy_val = VX_CONVERT_POLICY_SATURATE;
            vx_scalar policy = vxCreateScalar(ctx, VX_TYPE_ENUM, &policy_val);
            vx_graph g = vxCreateGraph(ctx);
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_SUBTRACT);
            vx_node n = vxCreateGenericNode(g, k);
            vxSetParameterByIndex(n, 0, (vx_reference)t1);
            vxSetParameterByIndex(n, 1, (vx_reference)t2);
            vxSetParameterByIndex(n, 2, (vx_reference)policy);
            vxSetParameterByIndex(n, 3, (vx_reference)tout);
            vxVerifyGraph(g); vxProcessGraph(g);
            std::vector<int16_t> result(64 * 64, 0);
            vxCopyTensorPatch(tout, 2, starts, dims, strides, result.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
            bool ok = (result[0] == 20);
            vxReleaseKernel(&k); vxReleaseNode(&n); vxReleaseGraph(&g); vxReleaseScalar(&policy);
            vxReleaseTensor(&t1); vxReleaseTensor(&t2); vxReleaseTensor(&tout);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- TensorMul ----
    {
        BenchmarkCase bc;
        bc.name        = "TensorMul";
        bc.category    = "tensor";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_TENSOR_MULTIPLY;
        bc.required_kernels = {VX_KERNEL_TENSOR_MULTIPLY};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_size tw = (width > 1024) ? 1024 : width;
            vx_size th = (height > 1024) ? 1024 : height;
            vx_size dims[2] = {tw, th};
            vx_tensor input1 = tracker.trackTensor(gen.createFilledTensor(ctx, dims, 2, VX_TYPE_INT16));
            vx_tensor input2 = tracker.trackTensor(gen.createFilledTensor(ctx, dims, 2, VX_TYPE_INT16));
            vx_tensor output = tracker.trackTensor(vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0));
            vx_float32 scale_val = 1.0f;
            vx_scalar scale = tracker.trackScalar(gen.createScalar(ctx, VX_TYPE_FLOAT32, &scale_val));
            vx_enum overflow_val = VX_CONVERT_POLICY_SATURATE;
            vx_scalar overflow_policy = tracker.trackScalar(gen.createScalar(ctx, VX_TYPE_ENUM, &overflow_val));
            vx_enum rounding_val = VX_ROUND_POLICY_TO_ZERO;
            vx_scalar rounding_policy = tracker.trackScalar(gen.createScalar(ctx, VX_TYPE_ENUM, &rounding_val));
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_MULTIPLY);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)input1);
            vxSetParameterByIndex(node, 1, (vx_reference)input2);
            vxSetParameterByIndex(node, 2, (vx_reference)scale);
            vxSetParameterByIndex(node, 3, (vx_reference)overflow_policy);
            vxSetParameterByIndex(node, 4, (vx_reference)rounding_policy);
            vxSetParameterByIndex(node, 5, (vx_reference)output);
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_MULTIPLY);
            bool ok = (vxGetStatus((vx_reference)k) == VX_SUCCESS);
            if (ok) vxReleaseKernel(&k);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- TensorTranspose ----
    {
        BenchmarkCase bc;
        bc.name        = "TensorTranspose";
        bc.category    = "tensor";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_TENSOR_TRANSPOSE;
        bc.required_kernels = {VX_KERNEL_TENSOR_TRANSPOSE};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_size tw = (width > 1024) ? 1024 : width;
            vx_size th = (height > 1024) ? 1024 : height;
            vx_size in_dims[2] = {tw, th};
            vx_size out_dims[2] = {th, tw};
            vx_tensor input  = tracker.trackTensor(gen.createFilledTensor(ctx, in_dims, 2, VX_TYPE_INT16));
            vx_tensor output = tracker.trackTensor(vxCreateTensor(ctx, 2, out_dims, VX_TYPE_INT16, 0));
            vx_size dim1_val = 0;
            vx_size dim2_val = 1;
            vx_scalar dim1 = tracker.trackScalar(gen.createScalar(ctx, VX_TYPE_SIZE, &dim1_val));
            vx_scalar dim2 = tracker.trackScalar(gen.createScalar(ctx, VX_TYPE_SIZE, &dim2_val));
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_TRANSPOSE);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)input);
            vxSetParameterByIndex(node, 1, (vx_reference)output);
            vxSetParameterByIndex(node, 2, (vx_reference)dim1);
            vxSetParameterByIndex(node, 3, (vx_reference)dim2);
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_TRANSPOSE);
            bool ok = (vxGetStatus((vx_reference)k) == VX_SUCCESS);
            if (ok) vxReleaseKernel(&k);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- TensorConvertDepth ----
    {
        BenchmarkCase bc;
        bc.name        = "TensorConvertDepth";
        bc.category    = "tensor";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_TENSOR_CONVERT_DEPTH;
        bc.required_kernels = {VX_KERNEL_TENSOR_CONVERT_DEPTH};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_size tw = (width > 1024) ? 1024 : width;
            vx_size th = (height > 1024) ? 1024 : height;
            vx_size dims[2] = {tw, th};
            vx_tensor input  = tracker.trackTensor(gen.createFilledTensor(ctx, dims, 2, VX_TYPE_INT16));
            vx_tensor output = tracker.trackTensor(vxCreateTensor(ctx, 2, dims, VX_TYPE_UINT8, 0));
            vx_enum policy_val = VX_CONVERT_POLICY_SATURATE;
            vx_scalar policy = tracker.trackScalar(gen.createScalar(ctx, VX_TYPE_ENUM, &policy_val));
            vx_float32 norm_val = 1.0f;
            vx_float32 offset_val = 0.0f;
            vx_scalar norm_scalar   = tracker.trackScalar(gen.createScalar(ctx, VX_TYPE_FLOAT32, &norm_val));
            vx_scalar offset_scalar = tracker.trackScalar(gen.createScalar(ctx, VX_TYPE_FLOAT32, &offset_val));
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_CONVERT_DEPTH);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)input);
            vxSetParameterByIndex(node, 1, (vx_reference)policy);
            vxSetParameterByIndex(node, 2, (vx_reference)norm_scalar);
            vxSetParameterByIndex(node, 3, (vx_reference)offset_scalar);
            vxSetParameterByIndex(node, 4, (vx_reference)output);
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_CONVERT_DEPTH);
            bool ok = (vxGetStatus((vx_reference)k) == VX_SUCCESS);
            if (ok) vxReleaseKernel(&k);
            return ok;
        };
        cases.push_back(bc);
    }

    // NOTE: TensorMatMul benchmark removed -- vxTensorMatrixMultiplyNode takes
    // a vx_tensor_matrix_multiply_params_t struct pointer which cannot be
    // passed through the generic vxSetParameterByIndex interface.

    // ---- TensorTableLookup ----
    {
        BenchmarkCase bc;
        bc.name        = "TensorTableLookup";
        bc.category    = "tensor";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_TENSOR_TABLE_LOOKUP;
        bc.required_kernels = {VX_KERNEL_TENSOR_TABLE_LOOKUP};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            vx_size tw = (width > 1024) ? 1024 : width;
            vx_size th = (height > 1024) ? 1024 : height;
            vx_size dims[2] = {tw, th};
            vx_tensor input  = tracker.trackTensor(gen.createFilledTensor(ctx, dims, 2, VX_TYPE_UINT8));
            vx_tensor output = tracker.trackTensor(vxCreateTensor(ctx, 2, dims, VX_TYPE_UINT8, 0));
            vx_lut lut = tracker.trackLUT(gen.createLUT(ctx));
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_TABLE_LOOKUP);
            if (vxGetStatus((vx_reference)k) != VX_SUCCESS) return false;
            vx_node node = vxCreateGenericNode(graph, k);
            vxReleaseKernel(&k);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            vxSetParameterByIndex(node, 0, (vx_reference)input);
            vxSetParameterByIndex(node, 1, (vx_reference)lut);
            vxSetParameterByIndex(node, 2, (vx_reference)output);
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_TABLE_LOOKUP);
            bool ok = (vxGetStatus((vx_reference)k) == VX_SUCCESS);
            if (ok) vxReleaseKernel(&k);
            return ok;
        };
        cases.push_back(bc);
    }
#endif

    return cases;
}
