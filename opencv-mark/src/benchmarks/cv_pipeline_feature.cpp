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
// OpenCV equivalents for the OpenVX `pipeline_feature` category. Same
// "OpenVX graph dividend vs sequential cv:: calls" rationale as the
// pipeline_vision pipelines.

#include "opencv_runner.h"
#include "benchmark_config.h"
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace opencv_mark {

std::vector<OpenCVBenchmarkCase> registerCvFeaturePipelines() {
    std::vector<OpenCVBenchmarkCase> cases;

    // 1. HistogramEqualize: ColorConvert(RGB→IYUV) → ChannelExtract(Y)
    //                       → EqualizeHist.
    //
    // OpenCV equivalent: cv::cvtColor(RGB→GRAY) (which yields the same
    // luma the Y channel of IYUV would carry) → cv::equalizeHist.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "HistogramEqualize";
        bc.category = "pipeline_feature";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            const uint32_t ew = w & ~1u;
            const uint32_t eh = h & ~1u;
            if (ew == 0 || eh == 0) return false;
            bufs.input = gen.makeRGB(ew, eh);
            bufs.input_extra.create(static_cast<int>(eh), static_cast<int>(ew), CV_8UC1);
            bufs.output.create(static_cast<int>(eh), static_cast<int>(ew), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::cvtColor(bufs.input, bufs.input_extra, cv::COLOR_RGB2GRAY);
            cv::equalizeHist(bufs.input_extra, bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat rgb(1, 256, CV_8UC3);
            for (int i = 0; i < 256; ++i) {
                rgb.at<cv::Vec3b>(0, i) = cv::Vec3b(static_cast<uint8_t>(i),
                                                    static_cast<uint8_t>(i),
                                                    static_cast<uint8_t>(i));
            }
            cv::Mat grey, o;
            cv::cvtColor(rgb, grey, cv::COLOR_RGB2GRAY);
            cv::equalizeHist(grey, o);
            return o.at<uint8_t>(0, 0) == 0 && o.at<uint8_t>(0, 255) == 255;
        };
        cases.push_back(bc);
    }

    // 2. HarrisTracker: ColorConvert(RGB→IYUV) → ChannelExtract(Y) →
    //                   HarrisCorners.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "HarrisTracker";
        bc.category = "pipeline_feature";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            const uint32_t ew = w & ~1u;
            const uint32_t eh = h & ~1u;
            if (ew == 0 || eh == 0) return false;
            bufs.input = gen.makeRGB(ew, eh);
            bufs.input_extra.create(static_cast<int>(eh), static_cast<int>(ew), CV_8UC1);
            bufs.output.create(static_cast<int>(eh), static_cast<int>(ew), CV_32FC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::cvtColor(bufs.input, bufs.input_extra, cv::COLOR_RGB2GRAY);
            cv::cornerHarris(bufs.input_extra, bufs.output,
                             /*blockSize=*/3, /*ksize=*/3, /*k=*/0.04,
                             cv::BORDER_REPLICATE);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat rgb(64, 64, CV_8UC3, cv::Scalar(0, 0, 0));
            rgb(cv::Rect(0, 0, 32, 32)).setTo(cv::Scalar(255, 255, 255));
            cv::Mat grey, o;
            cv::cvtColor(rgb, grey, cv::COLOR_RGB2GRAY);
            cv::cornerHarris(grey, o, 3, 3, 0.04, cv::BORDER_REPLICATE);
            double mn, mx;
            cv::minMaxLoc(o, &mn, &mx);
            return mx > 0.0;  // L-shape produces a non-zero corner response
        };
        cases.push_back(bc);
    }

    // 3. ThresholdedEdge: Sobel3x3 → Magnitude → ConvertDepth(S16→U8) → Threshold.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "ThresholdedEdge";
        bc.category = "pipeline_feature";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            // input_extra = dx (S16), output_extra = dy (S16),
            // output = final U8 thresholded edges.
            bufs.input_extra.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            bufs.output_extra.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::Sobel(bufs.input, bufs.input_extra,  CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
            cv::Sobel(bufs.input, bufs.output_extra, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
            cv::Mat dxf, dyf, magf, magu8;
            bufs.input_extra.convertTo(dxf, CV_32F);
            bufs.output_extra.convertTo(dyf, CV_32F);
            cv::magnitude(dxf, dyf, magf);
            magf.convertTo(magu8, CV_8U);  // saturate to U8
            cv::threshold(magu8, bufs.output, 100, 255, cv::THRESH_BINARY);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(0));
            in(cv::Rect(32, 0, 32, 64)).setTo(cv::Scalar(255));
            cv::Mat dx, dy, dxf, dyf, magf, magu8, o;
            cv::Sobel(in, dx, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
            cv::Sobel(in, dy, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
            dx.convertTo(dxf, CV_32F);
            dy.convertTo(dyf, CV_32F);
            cv::magnitude(dxf, dyf, magf);
            magf.convertTo(magu8, CV_8U);
            cv::threshold(magu8, o, 100, 255, cv::THRESH_BINARY);
            // Vertical edge near column 32 → at least some thresholded pixels there.
            return cv::countNonZero(o.col(31)) + cv::countNonZero(o.col(32)) > 0;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
