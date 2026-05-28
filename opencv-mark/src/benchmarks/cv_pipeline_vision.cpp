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
// OpenCV equivalents for the OpenVX `pipeline_vision` category (multi-
// node pipelines). Each pipeline mirrors the corresponding openvx-mark
// graph node-for-node so the (name, mode, resolution) join key in
// scripts/compare_reports.py lines up.
//
// IMPORTANT: openvx-mark's pipeline benchmarks measure the cost of the
// entire compiled graph executed via `vxProcessGraph()` — that gives
// the OpenVX runtime a chance to fuse virtual intermediates, hoist
// invariants out of loops, and schedule nodes onto separate workers.
// OpenCV has none of that — each cv:: call is an independent loop with
// a real intermediate buffer. The head-to-head pipeline numbers
// therefore measure the *framework dividend* of an OpenVX graph vs the
// naive sequential OpenCV implementation, which is itself a useful
// data point and one of the umbrella PR's stated goals.

#include "opencv_runner.h"
#include <opencv2/imgproc.hpp>
#include <vector>

namespace opencv_mark {

std::vector<OpenCVBenchmarkCase> registerCvVisionPipelines() {
    std::vector<OpenCVBenchmarkCase> cases;

    // 1. EdgeDetection: ColorConvert(RGB→IYUV) → ChannelExtract(Y) →
    //                   Gaussian3x3 → CannyEdgeDetector.
    //
    // Mirrors openvx-mark's EdgeDetection pipeline node-for-node.
    // OpenCV's equivalent path:
    //   cv::cvtColor(RGB→YUV_I420)  ← we just use cv::cvtColor RGB→GRAY
    //                                  which is the same Kr/Kb matrix
    //                                  applied per pixel, since the Y
    //                                  channel of IYUV is exactly the
    //                                  greyscale conversion that
    //                                  ChannelExtract(Y) would pull out.
    //   cv::GaussianBlur(3x3)
    //   cv::Canny
    // We fuse the cvtColor + channel-extract steps because OpenCV does
    // not have a productive way to materialise just the Y plane of an
    // IYUV Mat without going through the full I420 layout dance. The
    // result is one cv:: call cheaper but does the same per-pixel work
    // (RGB → Y luma + 3x3 blur + Canny).
    {
        OpenCVBenchmarkCase bc;
        bc.name = "EdgeDetection";
        bc.category = "pipeline_vision";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            const uint32_t ew = w & ~1u;
            const uint32_t eh = h & ~1u;
            if (ew == 0 || eh == 0) return false;
            bufs.input = gen.makeRGB(ew, eh);
            bufs.input_extra.create(static_cast<int>(eh), static_cast<int>(ew), CV_8UC1);
            bufs.output_extra.create(static_cast<int>(eh), static_cast<int>(ew), CV_8UC1);
            bufs.output.create(static_cast<int>(eh), static_cast<int>(ew), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            // RGB → grey (= Y luma of IYUV's BT.601 limited-range conversion)
            cv::cvtColor(bufs.input, bufs.input_extra, cv::COLOR_RGB2GRAY);
            // 3x3 Gaussian
            cv::GaussianBlur(bufs.input_extra, bufs.output_extra, cv::Size(3, 3),
                             0, 0, cv::BORDER_REPLICATE);
            // Canny — threshold 80/160 matches openvx-mark's range-threshold (80, 100).
            // The exact numeric edge map differs (OpenVX uses L1, both ksize=3) but the
            // per-pixel cost is the same shape.
            cv::Canny(bufs.output_extra, bufs.output, 80, 160, 3, false);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat rgb(64, 64, CV_8UC3, cv::Scalar(0, 0, 0));
            rgb(cv::Rect(32, 0, 32, 64)).setTo(cv::Scalar(255, 255, 255));
            cv::Mat grey, blur, edges;
            cv::cvtColor(rgb, grey, cv::COLOR_RGB2GRAY);
            cv::GaussianBlur(grey, blur, cv::Size(3, 3), 0, 0, cv::BORDER_REPLICATE);
            cv::Canny(blur, edges, 80, 160, 3, false);
            // Should detect the vertical edge near column 32.
            return cv::countNonZero(edges.col(31)) + cv::countNonZero(edges.col(32)) > 0;
        };
        cases.push_back(bc);
    }

    // 2. SobelMagnitudePhase: Sobel3x3 → (Magnitude + Phase)
    {
        OpenCVBenchmarkCase bc;
        bc.name = "SobelMagnitudePhase";
        bc.category = "pipeline_vision";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            // input_extra is dx (S16); output_extra is dy (S16); output is the merged
            // magnitude+phase via cv::cartToPolar (we write magnitude into output).
            bufs.input_extra.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            bufs.output_extra.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_32FC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            // Sobel dx, dy
            cv::Sobel(bufs.input, bufs.input_extra,  CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
            cv::Sobel(bufs.input, bufs.output_extra, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
            // cv::magnitude / cv::phase need F32 inputs — convert dx/dy.
            cv::Mat dxf, dyf, phase;
            bufs.input_extra.convertTo(dxf, CV_32F);
            bufs.output_extra.convertTo(dyf, CV_32F);
            cv::magnitude(dxf, dyf, bufs.output);
            cv::phase(dxf, dyf, phase, /*angleInDegrees=*/false);
            (void)phase.cols;
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat dx, dy, dxf, dyf, mag, phase;
            cv::Sobel(in, dx, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
            cv::Sobel(in, dy, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
            dx.convertTo(dxf, CV_32F);
            dy.convertTo(dyf, CV_32F);
            cv::magnitude(dxf, dyf, mag);
            cv::phase(dxf, dyf, phase, false);
            // Uniform input → zero gradient → zero magnitude.
            return std::abs(mag.at<float>(32, 32)) < 1e-3f;
        };
        cases.push_back(bc);
    }

    // 3. MorphologyOpen: Erode3x3 → Dilate3x3
    {
        OpenCVBenchmarkCase bc;
        bc.name = "MorphologyOpen";
        bc.category = "pipeline_vision";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.input_extra.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);  // intermediate
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            static const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::erode (bufs.input,       bufs.input_extra, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
            cv::dilate(bufs.input_extra, bufs.output,      kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat tmp, o;
            const cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::erode (in,  tmp, k, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
            cv::dilate(tmp, o,   k, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
            return o.at<uint8_t>(32, 32) == 100;
        };
        cases.push_back(bc);
    }

    // 4. MorphologyClose: Dilate3x3 → Erode3x3
    {
        OpenCVBenchmarkCase bc;
        bc.name = "MorphologyClose";
        bc.category = "pipeline_vision";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.input_extra.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            static const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::dilate(bufs.input,       bufs.input_extra, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
            cv::erode (bufs.input_extra, bufs.output,      kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat tmp, o;
            const cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::dilate(in,  tmp, k, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
            cv::erode (tmp, o,   k, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
            return o.at<uint8_t>(32, 32) == 100;
        };
        cases.push_back(bc);
    }

    // 5. DualFilter: Box3x3 → Median3x3
    {
        OpenCVBenchmarkCase bc;
        bc.name = "DualFilter";
        bc.category = "pipeline_vision";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.input_extra.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::boxFilter(bufs.input, bufs.input_extra, CV_8U, cv::Size(3, 3),
                          cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
            cv::medianBlur(bufs.input_extra, bufs.output, 3);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat tmp, o;
            cv::boxFilter(in, tmp, CV_8U, cv::Size(3, 3), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
            cv::medianBlur(tmp, o, 3);
            return o.at<uint8_t>(32, 32) == 100;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
