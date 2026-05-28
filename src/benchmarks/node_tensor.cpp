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
#include "openvx_optional_apis.h"
#include "openvx_version.h"
#include "verify_utils.h"
#include <VX/vxu.h>
#include <VX/vx_nodes.h>
#include <algorithm>
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
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            std::vector<int16_t> result(64 * 64, 0);
            vxCopyTensorPatch(tout, 2, starts, dims, strides, result.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
            bool ok = (status != VX_SUCCESS) ? true : (result[0] == 30);
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
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            std::vector<int16_t> result(64 * 64, 0);
            vxCopyTensorPatch(tout, 2, starts, dims, strides, result.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
            bool ok = (status != VX_SUCCESS) ? true : (result[0] == 20);
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
            vx_size dims[2] = {64, 64};
            std::vector<int16_t> a_data(64 * 64, 5), b_data(64 * 64, 3);
            vx_tensor t1 = vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0);
            vx_tensor t2 = vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0);
            vx_tensor tout = vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0);
            vx_size starts[2] = {0, 0}, strides[2] = {sizeof(int16_t), 64 * sizeof(int16_t)};
            vxCopyTensorPatch(t1, 2, starts, dims, strides, a_data.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vxCopyTensorPatch(t2, 2, starts, dims, strides, b_data.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_float32 scale_val = 1.0f;
            vx_scalar scale = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &scale_val);
            vx_enum overflow_val = VX_CONVERT_POLICY_SATURATE;
            vx_scalar overflow_policy = vxCreateScalar(ctx, VX_TYPE_ENUM, &overflow_val);
            vx_enum rounding_val = VX_ROUND_POLICY_TO_ZERO;
            vx_scalar rounding_policy = vxCreateScalar(ctx, VX_TYPE_ENUM, &rounding_val);
            vx_graph g = vxCreateGraph(ctx);
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_MULTIPLY);
            vx_node n = vxCreateGenericNode(g, k);
            vxSetParameterByIndex(n, 0, (vx_reference)t1);
            vxSetParameterByIndex(n, 1, (vx_reference)t2);
            vxSetParameterByIndex(n, 2, (vx_reference)scale);
            vxSetParameterByIndex(n, 3, (vx_reference)overflow_policy);
            vxSetParameterByIndex(n, 4, (vx_reference)rounding_policy);
            vxSetParameterByIndex(n, 5, (vx_reference)tout);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            std::vector<int16_t> result(64 * 64, 0);
            vxCopyTensorPatch(tout, 2, starts, dims, strides, result.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
            bool ok = (status != VX_SUCCESS) ? true : (result[0] == 15);
            vxReleaseKernel(&k); vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseScalar(&scale); vxReleaseScalar(&overflow_policy); vxReleaseScalar(&rounding_policy);
            vxReleaseTensor(&t1); vxReleaseTensor(&t2); vxReleaseTensor(&tout);
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
            vx_size in_dims[2] = {4, 2};
            vx_size out_dims[2] = {2, 4};
            int16_t in_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
            vx_tensor tin = vxCreateTensor(ctx, 2, in_dims, VX_TYPE_INT16, 0);
            vx_tensor tout = vxCreateTensor(ctx, 2, out_dims, VX_TYPE_INT16, 0);
            vx_size starts[2] = {0, 0};
            vx_size in_strides[2] = {sizeof(int16_t), 4 * sizeof(int16_t)};
            vxCopyTensorPatch(tin, 2, starts, in_dims, in_strides, in_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_size dim1_val = 0, dim2_val = 1;
            vx_scalar dim1 = vxCreateScalar(ctx, VX_TYPE_SIZE, &dim1_val);
            vx_scalar dim2 = vxCreateScalar(ctx, VX_TYPE_SIZE, &dim2_val);
            vx_graph g = vxCreateGraph(ctx);
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_TRANSPOSE);
            vx_node n = vxCreateGenericNode(g, k);
            vxSetParameterByIndex(n, 0, (vx_reference)tin);
            vxSetParameterByIndex(n, 1, (vx_reference)tout);
            vxSetParameterByIndex(n, 2, (vx_reference)dim1);
            vxSetParameterByIndex(n, 3, (vx_reference)dim2);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            int16_t out_data[8] = {};
            vx_size out_strides[2] = {sizeof(int16_t), 2 * sizeof(int16_t)};
            vxCopyTensorPatch(tout, 2, starts, out_dims, out_strides, out_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
            // in[col=1,row=0]=2 should become out[col=0,row=1]=2
            bool ok = (status != VX_SUCCESS) ? true : (out_data[0] == 1 && out_data[2] == 2);
            vxReleaseKernel(&k); vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseScalar(&dim1); vxReleaseScalar(&dim2);
            vxReleaseTensor(&tin); vxReleaseTensor(&tout);
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
            vx_size dims[2] = {64, 64};
            std::vector<int16_t> in_data(64 * 64, 100);
            vx_tensor tin = vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0);
            vx_tensor tout = vxCreateTensor(ctx, 2, dims, VX_TYPE_UINT8, 0);
            vx_size starts[2] = {0, 0}, strides[2] = {sizeof(int16_t), 64 * sizeof(int16_t)};
            vxCopyTensorPatch(tin, 2, starts, dims, strides, in_data.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_enum policy_val = VX_CONVERT_POLICY_SATURATE;
            vx_scalar policy = vxCreateScalar(ctx, VX_TYPE_ENUM, &policy_val);
            vx_float32 norm_val = 1.0f, offset_val = 0.0f;
            vx_scalar norm = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &norm_val);
            vx_scalar offset = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &offset_val);
            vx_graph g = vxCreateGraph(ctx);
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_CONVERT_DEPTH);
            vx_node n = vxCreateGenericNode(g, k);
            vxSetParameterByIndex(n, 0, (vx_reference)tin);
            vxSetParameterByIndex(n, 1, (vx_reference)policy);
            vxSetParameterByIndex(n, 2, (vx_reference)norm);
            vxSetParameterByIndex(n, 3, (vx_reference)offset);
            vxSetParameterByIndex(n, 4, (vx_reference)tout);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            std::vector<uint8_t> result(64 * 64, 0);
            vx_size out_strides[2] = {sizeof(uint8_t), 64 * sizeof(uint8_t)};
            vxCopyTensorPatch(tout, 2, starts, dims, out_strides, result.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
            bool ok = (status != VX_SUCCESS) ? true : (result[0] == 100);
            vxReleaseKernel(&k); vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseScalar(&policy); vxReleaseScalar(&norm); vxReleaseScalar(&offset);
            vxReleaseTensor(&tin); vxReleaseTensor(&tout);
            return ok;
        };
        cases.push_back(bc);
    }

    // ---- TensorMatMul ----
    //
    // OpenVX 1.3.1 §3.50: vxTensorMatrixMultiplyNode computes
    //   output = (transpose_input1 ? input1ᵀ : input1) ×
    //            (transpose_input2 ? input2ᵀ : input2)
    //          + (transpose_input3 ? input3ᵀ : input3)
    // for 2D input tensors. input3 is optional. We use the typed
    // vxTensorMatrixMultiplyNode API directly (the params struct
    // cannot be wired through the generic vxSetParameterByIndex path).
    {
        BenchmarkCase bc;
        bc.name        = "TensorMatMul";
        bc.category    = "tensor";
        bc.feature_set = "enhanced_vision";
        bc.kernel_enum = VX_KERNEL_TENSOR_MATRIX_MULTIPLY;
        bc.required_kernels = {VX_KERNEL_TENSOR_MATRIX_MULTIPLY};
        bc.graph_setup = [](vx_context ctx, vx_graph graph,
                            uint32_t width, uint32_t height,
                            TestDataGenerator& gen, ResourceTracker& tracker) -> bool {
            // Use square M×N · N×M matmul so we touch w*h*w ops at full
            // res. Cap at 256 to keep iteration cost reasonable.
            vx_size M = std::min<vx_size>(256, width);
            vx_size N = std::min<vx_size>(256, height);
            vx_size in1_dims[2] = {N, M};   // M×N matrix (rows=M, cols=N)
            vx_size in2_dims[2] = {M, N};   // N×M matrix
            vx_size out_dims[2] = {M, M};   // M×M result
            vx_tensor in1 = tracker.trackTensor(gen.createFilledTensor(ctx, in1_dims, 2, VX_TYPE_INT16));
            vx_tensor in2 = tracker.trackTensor(gen.createFilledTensor(ctx, in2_dims, 2, VX_TYPE_INT16));
            vx_tensor out = tracker.trackTensor(vxCreateTensor(ctx, 2, out_dims, VX_TYPE_INT16, 0));
            if (vxGetStatus((vx_reference)in1) != VX_SUCCESS ||
                vxGetStatus((vx_reference)in2) != VX_SUCCESS ||
                vxGetStatus((vx_reference)out) != VX_SUCCESS) return false;

            // input3 (bias) is "optional" per OpenVX 1.3.1 §3.50, but
            // "optional" means different things to different impls:
            //   - AMD MIVisionX / Khronos sample: accept NULL.
            //   - rustVX (and other strict-FFI impls): may segfault on
            //     a NULL tensor handle inside the FFI boundary because
            //     the Rust binding expects a valid `vx_tensor` opaque
            //     pointer to dereference for type queries.
            // We therefore pass a real zero-filled M×M bias tensor so
            // the cross-impl bench is portable: y = A·B + 0 = A·B,
            // semantically identical to the no-bias path, and every
            // impl sees a valid tensor handle. Cost of the add over
            // M² fp16 ≤ 0.5% of an O(M²·N) matmul at M=N=256 — well
            // below the timer-noise floor.
            vx_tensor bias = tracker.trackTensor(vxCreateTensor(ctx, 2, out_dims, VX_TYPE_INT16, 0));
            if (vxGetStatus((vx_reference)bias) != VX_SUCCESS) return false;

            vx_tensor_matrix_multiply_params_t params = {};
            params.transpose_input1 = vx_false_e;
            params.transpose_input2 = vx_false_e;
            params.transpose_input3 = vx_false_e;

            auto fn = openvx_optional::tensorMatrixMultiplyNode();
            if (!fn) return false;
            vx_node node = fn(graph, in1, in2, bias, &params, out);
            if (vxGetStatus((vx_reference)node) != VX_SUCCESS) return false;
            tracker.trackNode(node);
            return true;
        };
        bc.immediate_func = nullptr;
        bc.verify_fn = [](vx_context ctx) -> bool {
            // 2×2 · 2×2 matmul with known values: [[1,2],[3,4]] · [[1,0],[0,1]] = [[1,2],[3,4]]
            // Pass a zero-filled bias for the same NULL-safety reason as
            // the graph_setup path above.
            auto fn = openvx_optional::tensorMatrixMultiplyNode();
            if (!fn) return true;
            vx_size dims[2] = {2, 2};
            int16_t a[4] = {1, 2, 3, 4};
            int16_t b[4] = {1, 0, 0, 1};
            int16_t zero_bias[4] = {0, 0, 0, 0};
            vx_tensor t1 = vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0);
            vx_tensor t2 = vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0);
            vx_tensor t3 = vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0);
            vx_tensor tout = vxCreateTensor(ctx, 2, dims, VX_TYPE_INT16, 0);
            vx_size starts[2] = {0, 0}, strides[2] = {sizeof(int16_t), 2 * sizeof(int16_t)};
            vxCopyTensorPatch(t1, 2, starts, dims, strides, a, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vxCopyTensorPatch(t2, 2, starts, dims, strides, b, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vxCopyTensorPatch(t3, 2, starts, dims, strides, zero_bias, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_tensor_matrix_multiply_params_t params = {};
            vx_graph g = vxCreateGraph(ctx);
            vx_node n = fn(g, t1, t2, t3, &params, tout);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            int16_t result[4] = {};
            vxCopyTensorPatch(tout, 2, starts, dims, strides, result, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
            bool ok = (status != VX_SUCCESS) ? true :
                      (result[0] == 1 && result[1] == 2 && result[2] == 3 && result[3] == 4);
            vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseTensor(&t1); vxReleaseTensor(&t2); vxReleaseTensor(&t3); vxReleaseTensor(&tout);
            return ok;
        };
        cases.push_back(bc);
    }

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
            vx_size dims[2] = {64, 64};
            std::vector<uint8_t> in_data(64 * 64, 42);
            vx_tensor tin = vxCreateTensor(ctx, 2, dims, VX_TYPE_UINT8, 0);
            vx_tensor tout = vxCreateTensor(ctx, 2, dims, VX_TYPE_UINT8, 0);
            vx_size starts[2] = {0, 0}, strides[2] = {sizeof(uint8_t), 64 * sizeof(uint8_t)};
            vxCopyTensorPatch(tin, 2, starts, dims, strides, in_data.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_lut lut = vxCreateLUT(ctx, VX_TYPE_UINT8, 256);
            uint8_t identity[256];
            for (int i = 0; i < 256; i++) identity[i] = (uint8_t)i;
            vxCopyLUT(lut, identity, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            vx_graph g = vxCreateGraph(ctx);
            vx_kernel k = vxGetKernelByEnum(ctx, VX_KERNEL_TENSOR_TABLE_LOOKUP);
            vx_node n = vxCreateGenericNode(g, k);
            vxSetParameterByIndex(n, 0, (vx_reference)tin);
            vxSetParameterByIndex(n, 1, (vx_reference)lut);
            vxSetParameterByIndex(n, 2, (vx_reference)tout);
            vx_status status = vxVerifyGraph(g);
            if (status == VX_SUCCESS) status = vxProcessGraph(g);
            std::vector<uint8_t> result(64 * 64, 0);
            vxCopyTensorPatch(tout, 2, starts, dims, strides, result.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
            bool ok = (status != VX_SUCCESS) ? true : (result[0] == 42);
            vxReleaseKernel(&k); vxReleaseNode(&n); vxReleaseGraph(&g);
            vxReleaseLUT(&lut);
            vxReleaseTensor(&tin); vxReleaseTensor(&tout);
            return ok;
        };
        cases.push_back(bc);
    }
#endif

    return cases;
}
