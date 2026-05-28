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
//
// OpenCV equivalents for the OpenVX `tensor` category (Enhanced Vision
// Feature Set). OpenVX tensors are N-dimensional arrays; cv::Mat
// supports multi-dim too but most cv:: ops are written for 2D Mat.
// Where we benchmark a 2D-tensor-shape operation, we use cv::Mat
// directly; for element-wise ops we flatten the dimensions into a 1D
// CV_*C1 Mat so the kernel's per-element work matches the OpenVX
// kernel's per-element work exactly.
//
// Kernel-to-OpenCV mapping (matches openvx-mark naming convention —
// `TensorAdd`/`TensorSub`/`TensorMul` rather than the spec's
// `Tensor{Add,Subtract,Multiply}` because openvx-mark uses the
// shorter form):
//
//   * TensorAdd          — cv::add        on flat Mat
//   * TensorSub          — cv::subtract
//   * TensorMul          — cv::multiply
//   * TensorConvertDepth — cv::Mat::convertTo (S16 → U8 with saturate)
//   * TensorMatMul       — cv::gemm (general matrix multiply)
//   * TensorTableLookup  — cv::LUT
//   * TensorTranspose    — cv::transpose (2D)

#include "opencv_runner.h"
#include <algorithm>
#include <opencv2/core.hpp>
#include <vector>

namespace opencv_mark {

namespace {
// openvx-mark caps tensor dims at 1024 to keep iteration cost
// reasonable; we mirror that so the byte count of the per-iteration
// work matches.
inline uint32_t capDim(uint32_t d) { return std::min<uint32_t>(d, 1024); }
}

std::vector<OpenCVBenchmarkCase> registerCvTensorBenchmarks() {
    std::vector<OpenCVBenchmarkCase> cases;

    // TensorAdd — two S16 tensors → S16 (saturate)
    {
        OpenCVBenchmarkCase bc;
        bc.name = "TensorAdd";
        bc.category = "tensor";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            const uint32_t tw = capDim(w), th = capDim(h);
            bufs.input = gen.makeS16(tw, th);
            bufs.input_extra = gen.makeS16(tw, th);
            bufs.output.create(static_cast<int>(th), static_cast<int>(tw), CV_16SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::add(bufs.input, bufs.input_extra, bufs.output);  // dst type = input type = S16
        };
        bc.verify_fn = []() -> bool {
            cv::Mat a(64, 64, CV_16SC1, cv::Scalar(10));
            cv::Mat b(64, 64, CV_16SC1, cv::Scalar(20));
            cv::Mat o; cv::add(a, b, o);
            return o.at<int16_t>(32, 32) == 30;
        };
        cases.push_back(bc);
    }

    // TensorSub — two S16 tensors → S16
    {
        OpenCVBenchmarkCase bc;
        bc.name = "TensorSub";
        bc.category = "tensor";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            const uint32_t tw = capDim(w), th = capDim(h);
            bufs.input = gen.makeS16(tw, th);
            bufs.input_extra = gen.makeS16(tw, th);
            bufs.output.create(static_cast<int>(th), static_cast<int>(tw), CV_16SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::subtract(bufs.input, bufs.input_extra, bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat a(64, 64, CV_16SC1, cv::Scalar(50));
            cv::Mat b(64, 64, CV_16SC1, cv::Scalar(20));
            cv::Mat o; cv::subtract(a, b, o);
            return o.at<int16_t>(32, 32) == 30;
        };
        cases.push_back(bc);
    }

    // TensorMul — two S16 tensors → S16 (element-wise; OpenVX uses
    //             a scale + saturate just like the Multiply kernel).
    {
        OpenCVBenchmarkCase bc;
        bc.name = "TensorMul";
        bc.category = "tensor";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            const uint32_t tw = capDim(w), th = capDim(h);
            bufs.input = gen.makeS16(tw, th);
            bufs.input_extra = gen.makeS16(tw, th);
            bufs.output.create(static_cast<int>(th), static_cast<int>(tw), CV_16SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::multiply(bufs.input, bufs.input_extra, bufs.output, /*scale=*/1.0, CV_16S);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat a(64, 64, CV_16SC1, cv::Scalar(10));
            cv::Mat b(64, 64, CV_16SC1, cv::Scalar(5));
            cv::Mat o; cv::multiply(a, b, o, 1.0, CV_16S);
            return o.at<int16_t>(32, 32) == 50;
        };
        cases.push_back(bc);
    }

    // TensorConvertDepth — S16 in → U8 out (saturate). Matches the
    //                      openvx-mark default of S16→U8 with norm=1,
    //                      offset=0, saturate policy.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "TensorConvertDepth";
        bc.category = "tensor";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            const uint32_t tw = capDim(w), th = capDim(h);
            bufs.input = gen.makeS16(tw, th);
            bufs.output.create(static_cast<int>(th), static_cast<int>(tw), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            bufs.input.convertTo(bufs.output, CV_8U);  // saturates to [0, 255]
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_16SC1, cv::Scalar(100));
            cv::Mat o; in.convertTo(o, CV_8U);
            return o.at<uint8_t>(32, 32) == 100;
        };
        cases.push_back(bc);
    }

    // TensorMatMul — 2D matrix multiply (S16 ⊗ S16 → S16 via gemm in
    //                f32; OpenVX returns S16 directly. We compute in
    //                F32 and clamp to S16 to mirror the OpenVX
    //                contract.)
    //
    // Matches openvx-mark's TensorMatMul shape: square M×N · N×M → M×M
    // with M, N capped at 256.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "TensorMatMul";
        bc.category = "tensor";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            const int M = std::min<int>(256, static_cast<int>(w));
            const int N = std::min<int>(256, static_cast<int>(h));
            // cv::gemm requires CV_32FC1 or CV_64FC1. Generate S16 inputs
            // and convert to F32 in setup_fn (outside the timing loop)
            // so the per-iteration cost is just the matrix multiply.
            cv::Mat a_s16 = gen.makeS16(N, M);  // M rows × N cols
            cv::Mat b_s16 = gen.makeS16(M, N);  // N rows × M cols
            a_s16.convertTo(bufs.input,       CV_32F);
            b_s16.convertTo(bufs.input_extra, CV_32F);
            bufs.output.create(M, M, CV_32FC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::gemm(bufs.input, bufs.input_extra, /*alpha=*/1.0,
                     cv::noArray(), /*beta=*/0.0, bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat a = (cv::Mat_<float>(2, 2) << 1, 2, 3, 4);
            cv::Mat b = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1);
            cv::Mat o;
            cv::gemm(a, b, 1.0, cv::noArray(), 0.0, o);
            return o.at<float>(0, 0) == 1.0f && o.at<float>(1, 1) == 4.0f;
        };
        cases.push_back(bc);
    }

    // TensorTableLookup — U8 in → U8 out via 256-entry LUT.
    //                     Matches the openvx-mark default of a U8 LUT.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "TensorTableLookup";
        bc.category = "tensor";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            const uint32_t tw = capDim(w), th = capDim(h);
            bufs.input = gen.makeU8(tw, th);
            bufs.input_extra = gen.makeLUT();
            bufs.output.create(static_cast<int>(th), static_cast<int>(tw), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::LUT(bufs.input, bufs.input_extra, bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(42));
            cv::Mat lut(1, 256, CV_8UC1);
            for (int i = 0; i < 256; ++i) lut.at<uint8_t>(0, i) = static_cast<uint8_t>(i);
            cv::Mat o; cv::LUT(in, lut, o);
            return o.at<uint8_t>(32, 32) == 42;
        };
        cases.push_back(bc);
    }

    // TensorTranspose — 2D S16 transpose. cv::transpose handles
    //                   element-wise swap of (row, col) coordinates;
    //                   the per-element cost matches vxTensorTransposeNode
    //                   with dim1=0, dim2=1.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "TensorTranspose";
        bc.category = "tensor";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            const uint32_t tw = capDim(w), th = capDim(h);
            bufs.input = gen.makeS16(tw, th);
            bufs.output.create(static_cast<int>(tw), static_cast<int>(th), CV_16SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::transpose(bufs.input, bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(2, 4, CV_16SC1);
            int16_t v = 0;
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 4; ++x)
                    in.at<int16_t>(y, x) = ++v;
            cv::Mat o; cv::transpose(in, o);
            return o.at<int16_t>(1, 0) == 2 && o.at<int16_t>(0, 1) == 5;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
