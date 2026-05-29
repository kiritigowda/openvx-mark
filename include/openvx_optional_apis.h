////////////////////////////////////////////////////////////////////////////////
//
// MIT License
//
// Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc.
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

#ifndef OPENVX_OPTIONAL_APIS_H
#define OPENVX_OPTIONAL_APIS_H

// Several OpenVX 1.2+ Enhanced Vision kernel APIs are *declared* by the
// vendor headers but are *not exported* by every OpenVX runtime — most
// notably AMD MIVisionX/AGO does not export `vxBilateralFilterNode`,
// `vxScalarOperationNode`, `vxHOGCellsNode`, `vxHOGFeaturesNode`,
// `vxHoughLinesPNode`, or `vxTensorMatrixMultiplyNode`. Linking
// against them directly produces a "symbol not found" error at link
// time, which would prevent the entire benchmark binary from building
// on those runtimes.
//
// This header soft-resolves those APIs via `dlsym(RTLD_DEFAULT, ...)`
// the first time each is requested. If the symbol is not present in
// the dynamically-linked OpenVX runtime, the lookup returns nullptr
// and the corresponding benchmark's graph_setup returns false so the
// runner records it as "skipped (kernel not supported by impl)" —
// exactly the behaviour of `vxGetKernelByEnum` returning a missing
// kernel. The trade-off is that we avoid hard link failures while
// still reporting accurate per-impl coverage in the JSON report.

#include "openvx_version.h"
#include <VX/vx.h>
#include <VX/vx_nodes.h>
#include <VX/vx_types.h>
#include <dlfcn.h>

namespace openvx_optional {

#if OPENVX_HAS_1_2

// vxBilateralFilterNode(graph, src_tensor, diameter, sigmaSpace,
//                       sigmaValues, dst_tensor) → vx_node
using BilateralFilterNodeFn = vx_node (*)(vx_graph, vx_tensor, vx_int32,
                                          vx_float32, vx_float32, vx_tensor);
inline BilateralFilterNodeFn bilateralFilterNode() {
    static BilateralFilterNodeFn fn = reinterpret_cast<BilateralFilterNodeFn>(
        dlsym(RTLD_DEFAULT, "vxBilateralFilterNode"));
    return fn;
}

// vxScalarOperationNode(graph, scalar_op_e, a, b, output) → vx_node
using ScalarOperationNodeFn = vx_node (*)(vx_graph, vx_enum, vx_scalar,
                                          vx_scalar, vx_scalar);
inline ScalarOperationNodeFn scalarOperationNode() {
    static ScalarOperationNodeFn fn = reinterpret_cast<ScalarOperationNodeFn>(
        dlsym(RTLD_DEFAULT, "vxScalarOperationNode"));
    return fn;
}

// vxHOGCellsNode(graph, input_U8, cell_w, cell_h, num_bins,
//                magnitudes_tensor, bins_tensor) → vx_node
using HOGCellsNodeFn = vx_node (*)(vx_graph, vx_image, vx_int32, vx_int32,
                                   vx_int32, vx_tensor, vx_tensor);
inline HOGCellsNodeFn hogCellsNode() {
    static HOGCellsNodeFn fn = reinterpret_cast<HOGCellsNodeFn>(
        dlsym(RTLD_DEFAULT, "vxHOGCellsNode"));
    return fn;
}

// vxHOGFeaturesNode(graph, input_U8, magnitudes, bins, params,
//                   hog_param_size, features_tensor) → vx_node
using HOGFeaturesNodeFn = vx_node (*)(vx_graph, vx_image, vx_tensor,
                                      vx_tensor, const vx_hog_t*,
                                      vx_size, vx_tensor);
inline HOGFeaturesNodeFn hogFeaturesNode() {
    static HOGFeaturesNodeFn fn = reinterpret_cast<HOGFeaturesNodeFn>(
        dlsym(RTLD_DEFAULT, "vxHOGFeaturesNode"));
    return fn;
}

// vxHoughLinesPNode(graph, input_U8, params, lines_array, num_lines_scalar) → vx_node
using HoughLinesPNodeFn = vx_node (*)(vx_graph, vx_image,
                                      const vx_hough_lines_p_t*,
                                      vx_array, vx_scalar);
inline HoughLinesPNodeFn houghLinesPNode() {
    static HoughLinesPNodeFn fn = reinterpret_cast<HoughLinesPNodeFn>(
        dlsym(RTLD_DEFAULT, "vxHoughLinesPNode"));
    return fn;
}

// vxTensorMatrixMultiplyNode(graph, in1, in2, in3_optional,
//                            params, output) → vx_node
using TensorMatrixMultiplyNodeFn = vx_node (*)(vx_graph, vx_tensor, vx_tensor,
                                               vx_tensor,
                                               const vx_tensor_matrix_multiply_params_t*,
                                               vx_tensor);
inline TensorMatrixMultiplyNodeFn tensorMatrixMultiplyNode() {
    static TensorMatrixMultiplyNodeFn fn = reinterpret_cast<TensorMatrixMultiplyNodeFn>(
        dlsym(RTLD_DEFAULT, "vxTensorMatrixMultiplyNode"));
    return fn;
}

// vxTensorTransposeNode(graph, input, output, dim1, dim2) → vx_node
//
// The typed helper is the only portable way to construct this node:
// OpenVX 1.3.1 §3.51 only defines the helper signature; the underlying
// kernel's parameter index order is impl-defined. Going through
// vxGetKernelByEnum + vxSetParameterByIndex with an assumed order
// (e.g. [input, output, dim1, dim2]) works on AMD AGO but breaks on
// rustVX (which uses [input, dim1, dim2, output]).
using TensorTransposeNodeFn = vx_node (*)(vx_graph, vx_tensor, vx_tensor,
                                          vx_size, vx_size);
inline TensorTransposeNodeFn tensorTransposeNode() {
    static TensorTransposeNodeFn fn = reinterpret_cast<TensorTransposeNodeFn>(
        dlsym(RTLD_DEFAULT, "vxTensorTransposeNode"));
    return fn;
}

// vxTensorConvertDepthNode(graph, input, policy_enum, norm_scalar,
//                          offset_scalar, output) → vx_node
//
// Same rationale as vxTensorTransposeNode: prefer the typed helper so
// each impl can dispatch through its own param-order convention.
using TensorConvertDepthNodeFn = vx_node (*)(vx_graph, vx_tensor, vx_enum,
                                             vx_scalar, vx_scalar, vx_tensor);
inline TensorConvertDepthNodeFn tensorConvertDepthNode() {
    static TensorConvertDepthNodeFn fn = reinterpret_cast<TensorConvertDepthNodeFn>(
        dlsym(RTLD_DEFAULT, "vxTensorConvertDepthNode"));
    return fn;
}

#endif // OPENVX_HAS_1_2

} // namespace openvx_optional

#endif // OPENVX_OPTIONAL_APIS_H
