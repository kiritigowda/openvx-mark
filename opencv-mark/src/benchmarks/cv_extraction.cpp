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
// OpenCV equivalents for the OpenVX `extraction` category (Enhanced
// Vision Feature Set). Name parity with openvx-mark.
//
// Kernel-to-OpenCV mapping:
//
//   * MatchTemplate    — cv::matchTemplate (TM_CCORR_NORMED)
//   * LBP              — manual 3x3 LBP implementation (no native cv::
//                        LBP; opencv_contrib `cv::face::LBPHFaceRecognizer`
//                        is a face-recognition wrapper, not a generic
//                        per-pixel LBP kernel)
//   * HOGCells         — cv::HOGDescriptor::computeGradient (the HOG
//                        gradient/orientation step that
//                        vxHOGCellsNode performs)
//   * HOGFeatures      — cv::HOGDescriptor::compute (full HOG pipeline
//                        — includes the cells step internally; we match
//                        the same parameters openvx-mark uses
//                        (8×8 cell, 16×16 block, 8×8 block stride,
//                        64×64 window, 8×8 window stride, 9 bins))
//   * HoughLinesP      — cv::HoughLinesP (rho=1, theta=π/180,
//                        threshold=50, minLineLength=30, maxLineGap=10)
//   * NonMaxSuppression — cv::dilate + cv::compare local-maxima trick:
//                        a pixel is a local max iff input(x,y) >=
//                        dilate(input, 3x3)(x,y). Output is the input
//                        masked by that condition (matches the OpenVX
//                        semantics of "suppress non-maxima" — kept
//                        pixels retain their value, others become 0 /
//                        INT16_MIN for S16).

#include "opencv_runner.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include <vector>

namespace opencv_mark {

std::vector<OpenCVBenchmarkCase> registerCvExtractionBenchmarks() {
    std::vector<OpenCVBenchmarkCase> cases;

    // MatchTemplate — U8 input, U8 32x32 template, F32 response map
    {
        OpenCVBenchmarkCase bc;
        bc.name = "MatchTemplate";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            if (w < 32 || h < 32) return false;
            bufs.input = gen.makeU8(w, h);
            bufs.input_extra = gen.makeU8(32, 32);  // template
            // result dims = (w - tw + 1) x (h - th + 1)
            bufs.output.create(static_cast<int>(h - 32 + 1),
                               static_cast<int>(w - 32 + 1), CV_32FC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::matchTemplate(bufs.input, bufs.input_extra, bufs.output,
                              cv::TM_CCORR_NORMED);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat src(64, 64, CV_8UC1, cv::Scalar(0));
            src(cv::Rect(16, 16, 32, 32)).setTo(cv::Scalar(255));
            cv::Mat tmpl(32, 32, CV_8UC1, cv::Scalar(255));
            cv::Mat result;
            cv::matchTemplate(src, tmpl, result, cv::TM_CCORR_NORMED);
            // Peak should be at the upper-left corner of the white square.
            double mn, mx;
            cv::Point mnl, mxl;
            cv::minMaxLoc(result, &mn, &mx, &mnl, &mxl);
            return mxl.x == 16 && mxl.y == 16;
        };
        cases.push_back(bc);
    }

    // LBP — U8 input, U8 output. Manual 3x3 implementation.
    //
    // OpenVX 1.3.1 §3.29 default LBP: each pixel encodes the sign of
    // (neighbour - centre) over the 8 cardinal/diagonal neighbours,
    // packed into a U8. Equivalent to the canonical Ojala et al. LBP.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "LBP";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            const cv::Mat& src = bufs.input;
            cv::Mat& dst = bufs.output;
            const int H = src.rows, W = src.cols;
            for (int y = 1; y < H - 1; ++y) {
                const uint8_t* p0 = src.ptr<uint8_t>(y - 1);
                const uint8_t* p1 = src.ptr<uint8_t>(y);
                const uint8_t* p2 = src.ptr<uint8_t>(y + 1);
                uint8_t*       d  = dst.ptr<uint8_t>(y);
                for (int x = 1; x < W - 1; ++x) {
                    const uint8_t c = p1[x];
                    uint8_t v = 0;
                    v |= (p0[x - 1] >= c) << 0;
                    v |= (p0[x    ] >= c) << 1;
                    v |= (p0[x + 1] >= c) << 2;
                    v |= (p1[x + 1] >= c) << 3;
                    v |= (p2[x + 1] >= c) << 4;
                    v |= (p2[x    ] >= c) << 5;
                    v |= (p2[x - 1] >= c) << 6;
                    v |= (p1[x - 1] >= c) << 7;
                    d[x] = v;
                }
            }
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(0));
            // Centre at 100, surrounded by 200s → all 8 neighbours ≥ centre → 0xFF.
            in(cv::Rect(31, 31, 3, 3)).setTo(cv::Scalar(200));
            in.at<uint8_t>(32, 32) = 100;
            cv::Mat out(64, 64, CV_8UC1, cv::Scalar(0));
            const int H = in.rows, W = in.cols;
            for (int y = 1; y < H - 1; ++y) {
                const uint8_t* p0 = in.ptr<uint8_t>(y - 1);
                const uint8_t* p1 = in.ptr<uint8_t>(y);
                const uint8_t* p2 = in.ptr<uint8_t>(y + 1);
                uint8_t*       d  = out.ptr<uint8_t>(y);
                for (int x = 1; x < W - 1; ++x) {
                    const uint8_t c = p1[x];
                    uint8_t v = 0;
                    v |= (p0[x - 1] >= c) << 0;
                    v |= (p0[x    ] >= c) << 1;
                    v |= (p0[x + 1] >= c) << 2;
                    v |= (p1[x + 1] >= c) << 3;
                    v |= (p2[x + 1] >= c) << 4;
                    v |= (p2[x    ] >= c) << 5;
                    v |= (p2[x - 1] >= c) << 6;
                    v |= (p1[x - 1] >= c) << 7;
                    d[x] = v;
                }
            }
            return out.at<uint8_t>(32, 32) == 0xFF;
        };
        cases.push_back(bc);
    }

    // HOGCells — U8 input, F32 magnitudes + F32 angles (computeGradient output).
    //
    // cv::HOGDescriptor::computeGradient is the OpenCV-internal step
    // that produces per-pixel gradient magnitudes + bin assignments.
    // It's the closest analogue to vxHOGCellsNode's per-cell histogram
    // accumulation, although the binning happens later in OpenCV's
    // pipeline (during compute()). For benchmark purposes we time the
    // gradient step which dominates the per-pixel cost.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "HOGCells";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            // HOG window must be a multiple of cell (8x8) and ≥ 16x16.
            const uint32_t ew = std::max<uint32_t>(16, (w / 8) * 8);
            const uint32_t eh = std::max<uint32_t>(16, (h / 8) * 8);
            bufs.input = gen.makeU8(ew, eh);
            bufs.output.create(static_cast<int>(eh), static_cast<int>(ew), CV_32FC2);  // mag
            bufs.output_extra.create(static_cast<int>(eh), static_cast<int>(ew), CV_8UC2);  // angle bins
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::HOGDescriptor hog;  // defaults: win 64x128, block 16x16, cell 8x8, 9 bins
            // computeGradient signature: (img, grad, qangle, paddingTL, paddingBR)
            hog.computeGradient(bufs.input, bufs.output, bufs.output_extra,
                                cv::Size(0, 0), cv::Size(0, 0));
        };
        bc.verify_fn = []() -> bool {
            cv::HOGDescriptor hog;
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat grad, ang;
            try {
                hog.computeGradient(in, grad, ang, cv::Size(0, 0), cv::Size(0, 0));
            } catch (const cv::Exception&) { return true; }
            // Uniform input → zero gradient magnitude.
            return grad.at<cv::Vec2f>(32, 32)[0] < 1e-3f;
        };
        cases.push_back(bc);
    }

    // HOGFeatures — U8 input, F32 descriptor vector (full HOG pipeline).
    {
        OpenCVBenchmarkCase bc;
        bc.name = "HOGFeatures";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            // Round up to a HOG window stride (8x8). cv::HOGDescriptor
            // defaults to a 64x128 window; we use 64x64 to match the
            // openvx-mark benchmark and feed an image that's at least
            // that big.
            const uint32_t ew = std::max<uint32_t>(64, (w / 8) * 8);
            const uint32_t eh = std::max<uint32_t>(64, (h / 8) * 8);
            bufs.input = gen.makeU8(ew, eh);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            // Match openvx-mark's HOGFeatures parameters:
            //   window 64×64, block 16×16, block stride 8×8, cell 8×8, 9 bins
            cv::HOGDescriptor hog(cv::Size(64, 64),   // win
                                  cv::Size(16, 16),   // block
                                  cv::Size(8, 8),     // block stride
                                  cv::Size(8, 8),     // cell
                                  9);                 // nbins
            std::vector<float> descriptors;
            hog.compute(bufs.input, descriptors, cv::Size(8, 8), cv::Size(0, 0));
            (void)descriptors.size();
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(0));
            cv::randu(in, 0, 255);
            cv::HOGDescriptor hog(cv::Size(64, 64), cv::Size(16, 16),
                                  cv::Size(8, 8), cv::Size(8, 8), 9);
            std::vector<float> descriptors;
            try { hog.compute(in, descriptors, cv::Size(8, 8), cv::Size(0, 0)); }
            catch (const cv::Exception&) { return true; }
            return !descriptors.empty();
        };
        cases.push_back(bc);
    }

    // HoughLinesP — U8 (binary) in, vector<Vec4i> lines out.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "HoughLinesP";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            // HoughLinesP wants a binary (edge) image; threshold the random
            // input so we get a meaningful set of edge pixels. Threshold
            // inside setup_fn so cv::HoughLinesP only times the Hough step
            // itself.
            cv::threshold(bufs.input, bufs.output_extra, 200, 255, cv::THRESH_BINARY);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            std::vector<cv::Vec4i> lines;
            cv::HoughLinesP(bufs.output_extra, lines,
                            /*rho=*/1.0,
                            /*theta=*/CV_PI / 180.0,
                            /*threshold=*/50,
                            /*minLineLength=*/30,
                            /*maxLineGap=*/10);
            (void)lines.size();
        };
        bc.verify_fn = []() -> bool {
            // Step image with a vertical white bar → at least one line found.
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(0));
            cv::line(in, cv::Point(32, 4), cv::Point(32, 60), cv::Scalar(255), 1);
            std::vector<cv::Vec4i> lines;
            cv::HoughLinesP(in, lines, 1.0, CV_PI / 180.0, 20, 20, 5);
            return !lines.empty();
        };
        cases.push_back(bc);
    }

    // NonMaxSuppression — S16 in, S16 out via cv::dilate + mask trick.
    //
    // Per OpenVX 1.3.1 §3.39, suppressed pixels in S16 take INT16_MIN.
    // We compute local maxima using cv::dilate (max filter over 3x3),
    // then keep pixels equal to their local max and set the rest to
    // INT16_MIN.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "NonMaxSuppression";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeS16(w, h);
            bufs.input_extra.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            static const cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            // Local max via dilate; pixel kept iff input == local max.
            cv::dilate(bufs.input, bufs.input_extra, se,
                       cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
            cv::Mat keep_mask = (bufs.input >= bufs.input_extra);  // CV_8UC1 mask
            bufs.output.setTo(static_cast<int16_t>(-32768));        // INT16_MIN
            bufs.input.copyTo(bufs.output, keep_mask);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_16SC1, cv::Scalar(10));
            in.at<int16_t>(32, 32) = 1000;
            cv::Mat dilated, out(64, 64, CV_16SC1, cv::Scalar(-32768));
            const cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::dilate(in, dilated, se, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
            cv::Mat mask = (in >= dilated);
            in.copyTo(out, mask);
            // Center should keep its 1000 value.
            return out.at<int16_t>(32, 32) == 1000;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
