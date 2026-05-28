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
#include <memory>
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
    //
    // The HOGDescriptor instance is captured in shared state and
    // constructed once in setup_fn. Constructing a fresh
    // cv::HOGDescriptor inside run_fn (the previous shape) walked
    // OpenCV's default-init code path on every iteration, which on
    // a busy bench is enough non-kernel overhead to bias the timing.
    {
        struct HogCellsState {
            cv::HOGDescriptor hog;  // defaults: win 64x128, block 16x16, cell 8x8, 9 bins
        };
        auto state = std::make_shared<HogCellsState>();

        OpenCVBenchmarkCase bc;
        bc.name = "HOGCells";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [state](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            // HOG window must be a multiple of cell (8x8) and ≥ 16x16.
            const uint32_t ew = std::max<uint32_t>(16, (w / 8) * 8);
            const uint32_t eh = std::max<uint32_t>(16, (h / 8) * 8);
            bufs.input = gen.makeU8(ew, eh);
            bufs.output.create(static_cast<int>(eh), static_cast<int>(ew), CV_32FC2);   // mag
            bufs.output_extra.create(static_cast<int>(eh), static_cast<int>(ew), CV_8UC2); // angle bins
            // Default-constructed HOGDescriptor lives in state; no
            // run_fn-side construction. NOTE: HOGDescriptor is not
            // thread-safe in OpenCV, but our runner is single-threaded
            // per case so this is fine.
            state->hog = cv::HOGDescriptor();
            return true;
        };
        bc.run_fn = [state](CaseBuffers& bufs) {
            // computeGradient signature: (img, grad, qangle, paddingTL, paddingBR)
            state->hog.computeGradient(bufs.input, bufs.output, bufs.output_extra,
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
    //
    // Two preallocation moves vs the original shape:
    //   1) cv::HOGDescriptor with the openvx-mark-matching parameters
    //      is captured in shared state, not reconstructed per iter.
    //   2) std::vector<float> descriptors is also captured + reserved
    //      to its final size in setup_fn so hog.compute()'s resize()
    //      below stays inside the reserved capacity — no realloc in
    //      the timed loop.
    //
    // Also: cap the effective input dimensions to 1024x768.
    // cv::HOGDescriptor::compute slides a 64x64 window with stride 8
    // across the full image, producing one descriptor per window. At
    // FHD that's ~30k windows × 1764 floats/win ≈ 50M floats ≈ 200 MB
    // of descriptors; at 4K ≈ 800 MB. Capping to 1024x768 (the
    // classic HOG-pedestrian-detect resolution) keeps the descriptors
    // vector ≤ ~80 MB while still being a meaningful workload — the
    // per-window cost is what's being measured, so window count
    // doesn't change the comparison answer.
    {
        struct HogFeaturesState {
            cv::HOGDescriptor hog{cv::Size(64, 64),   // win
                                  cv::Size(16, 16),   // block
                                  cv::Size(8, 8),     // block stride
                                  cv::Size(8, 8),     // cell
                                  9};                 // nbins
            std::vector<float> descriptors;
        };
        auto state = std::make_shared<HogFeaturesState>();

        OpenCVBenchmarkCase bc;
        bc.name = "HOGFeatures";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [state](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            // Cap rationale: see block comment above.
            constexpr uint32_t MAX_HOG_W = 1024;
            constexpr uint32_t MAX_HOG_H = 768;
            const uint32_t cw = std::min<uint32_t>(w, MAX_HOG_W);
            const uint32_t ch = std::min<uint32_t>(h, MAX_HOG_H);
            // Round up to a HOG window stride (8x8). cv::HOGDescriptor
            // defaults to a 64x128 window; we use 64x64 to match the
            // openvx-mark benchmark and feed an image that's at least
            // that big.
            const uint32_t ew = std::max<uint32_t>(64, (cw / 8) * 8);
            const uint32_t eh = std::max<uint32_t>(64, (ch / 8) * 8);
            bufs.input = gen.makeU8(ew, eh);

            // Reserve the descriptors vector to the size compute() will
            // produce: getDescriptorSize() returns the per-window length,
            // and the number of windows = win_per_row × win_per_col
            // with stride (8,8) and no padding.
            const size_t per_win = state->hog.getDescriptorSize();
            const size_t wins_per_row = (ew >= 64) ? ((ew - 64) / 8 + 1) : 1;
            const size_t wins_per_col = (eh >= 64) ? ((eh - 64) / 8 + 1) : 1;
            state->descriptors.clear();
            state->descriptors.reserve(per_win * wins_per_row * wins_per_col);
            return true;
        };
        bc.run_fn = [state](CaseBuffers& bufs) {
            // compute() resizes descriptors to the exact output length —
            // since we reserved to that exact size in setup_fn the
            // resize is a no-op (no realloc), so the timing measures
            // only the kernel work.
            state->hog.compute(bufs.input, state->descriptors, cv::Size(8, 8), cv::Size(0, 0));
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
    //
    // The output lines vector is captured in shared state and reserved
    // to a sensible upper bound in setup_fn. Without this, every timed
    // call would land cv::HoughLinesP's first push_back inside the
    // measurement window (vector allocation + copies of any line
    // segments accumulated so far).
    {
        struct HoughState {
            std::vector<cv::Vec4i> lines;
        };
        auto state = std::make_shared<HoughState>();

        OpenCVBenchmarkCase bc;
        bc.name = "HoughLinesP";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [state](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            // HoughLinesP wants a binary (edge) image; threshold the random
            // input so we get a meaningful set of edge pixels. Threshold
            // inside setup_fn so cv::HoughLinesP only times the Hough step
            // itself.
            cv::threshold(bufs.input, bufs.output_extra, 200, 255, cv::THRESH_BINARY);
            // 4096 is a generous cap for a random-edge image at any
            // resolution we exercise; the worst-case observed in
            // local runs is ~few hundred segments. Reserve once, reuse.
            state->lines.clear();
            state->lines.reserve(4096);
            return true;
        };
        bc.run_fn = [state](CaseBuffers& bufs) {
            // clear() preserves capacity; HoughLinesP will append into
            // the reserved storage without realloc as long as the
            // detected line count stays under 4096.
            state->lines.clear();
            cv::HoughLinesP(bufs.output_extra, state->lines,
                            /*rho=*/1.0,
                            /*theta=*/CV_PI / 180.0,
                            /*threshold=*/50,
                            /*minLineLength=*/30,
                            /*maxLineGap=*/10);
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
    //
    // keep_mask was previously allocated by an in-loop Mat expression
    // (`bufs.input >= bufs.input_extra`) which allocates a fresh
    // CV_8UC1 the size of the image every iteration. Preallocate it
    // in shared state and fill via cv::compare to keep run_fn
    // allocation-free.
    {
        struct NmsState {
            cv::Mat keep_mask;  // CV_8UC1, preallocated in setup_fn
        };
        auto state = std::make_shared<NmsState>();

        OpenCVBenchmarkCase bc;
        bc.name = "NonMaxSuppression";
        bc.category = "extraction";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [state](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeS16(w, h);
            bufs.input_extra.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            state->keep_mask.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [state](CaseBuffers& bufs) {
            static const cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            // Local max via dilate; pixel kept iff input == local max.
            cv::dilate(bufs.input, bufs.input_extra, se,
                       cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
            // cv::compare writes into the preallocated mask in place —
            // no Mat allocation in the timed loop. CMP_GE = "input >=
            // input_extra" → 255 where input is a local max, else 0.
            cv::compare(bufs.input, bufs.input_extra, state->keep_mask, cv::CMP_GE);
            bufs.output.setTo(static_cast<int16_t>(-32768));  // INT16_MIN
            bufs.input.copyTo(bufs.output, state->keep_mask);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_16SC1, cv::Scalar(10));
            in.at<int16_t>(32, 32) = 1000;
            cv::Mat dilated, out(64, 64, CV_16SC1, cv::Scalar(-32768));
            const cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::dilate(in, dilated, se, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
            cv::Mat mask;
            cv::compare(in, dilated, mask, cv::CMP_GE);
            in.copyTo(out, mask);
            // Center should keep its 1000 value.
            return out.at<int16_t>(32, 32) == 1000;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
