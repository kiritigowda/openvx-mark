// OpenCV equivalents for the OpenVX `misc` category.
//
// PR2 set: Magnitude, Phase, TableLookup, Threshold_Binary,
//          Threshold_Range, WeightedAverage.
//
// Parameter mapping notes:
//
//   * Magnitude: cv::magnitude takes two CV_32F inputs and produces a
//     CV_32F output. OpenVX vxMagnitudeNode takes two S16 inputs and
//     produces an S16 output. We emulate the OpenVX shape: convert
//     the two S16 inputs to F32 inside setup_fn (outside the timing
//     budget), call cv::magnitude on the F32 buffers, and write the
//     F32 result to a pre-allocated F32 output. This keeps the
//     measured cost as "cv::magnitude only" rather than including
//     the conversion.
//   * Phase: same shape — atan2 on S16 dx/dy via F32 intermediates.
//     cv::phase produces a CV_32F angle in radians by default;
//     openvx-mark's Phase produces a U8 angle in [0, 255]. We emit
//     F32 here because the measured cost is the cv::phase call;
//     the post-quantisation step would be a separate kernel anyway.
//   * TableLookup: cv::LUT(src, lut, dst) with a 256-entry CV_8UC1
//     LUT. Same per-pixel single-fetch contract as vxTableLookupNode.
//   * Threshold_Binary: cv::threshold(THRESH_BINARY, thresh=128).
//   * Threshold_Range: cv::inRange(low=64, high=192).
//   * WeightedAverage: cv::addWeighted(alpha=0.7, beta=0.3, gamma=0).
//     OpenVX vxWeightedAverageNode accepts a single alpha and computes
//     out = alpha * imgA + (1 - alpha) * imgB; we configure
//     cv::addWeighted with the same coefficients.

#include "opencv_runner.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace opencv_mark {

std::vector<OpenCVBenchmarkCase> registerCvMiscBenchmarks() {
    std::vector<OpenCVBenchmarkCase> cases;

    // Magnitude — two S16 inputs (dx, dy as F32) → F32 output.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Magnitude";
        bc.category = "misc";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            cv::Mat dx_s16 = gen.makeS16(w, h);
            cv::Mat dy_s16 = gen.makeS16(w, h);
            dx_s16.convertTo(bufs.input, CV_32F);
            dy_s16.convertTo(bufs.input_extra, CV_32F);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_32FC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::magnitude(bufs.input, bufs.input_extra, bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat dx(4, 4, CV_32FC1, cv::Scalar(3.0f));
            cv::Mat dy(4, 4, CV_32FC1, cv::Scalar(4.0f));
            cv::Mat o;
            cv::magnitude(dx, dy, o);
            return std::abs(o.at<float>(0, 0) - 5.0f) < 1e-3f;
        };
        cases.push_back(bc);
    }

    // Phase — two F32 inputs (dx, dy) → F32 angle output.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Phase";
        bc.category = "misc";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            cv::Mat dx_s16 = gen.makeS16(w, h);
            cv::Mat dy_s16 = gen.makeS16(w, h);
            dx_s16.convertTo(bufs.input, CV_32F);
            dy_s16.convertTo(bufs.input_extra, CV_32F);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_32FC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::phase(bufs.input, bufs.input_extra, bufs.output, /*angleInDegrees=*/false);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat dx(4, 4, CV_32FC1, cv::Scalar(1.0f));
            cv::Mat dy(4, 4, CV_32FC1, cv::Scalar(1.0f));
            cv::Mat o;
            cv::phase(dx, dy, o, false);
            // atan2(1, 1) == pi/4 ≈ 0.7854
            return std::abs(o.at<float>(0, 0) - 0.7854f) < 1e-2f;
        };
        cases.push_back(bc);
    }

    // TableLookup — U8 in, U8 out, 256-entry LUT.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "TableLookup";
        bc.category = "misc";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.input_extra = gen.makeLUT();
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::LUT(bufs.input, bufs.input_extra, bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat lut(1, 256, CV_8UC1);
            for (int i = 0; i < 256; ++i) lut.at<uint8_t>(0, i) = static_cast<uint8_t>(255 - i);
            cv::Mat o;
            cv::LUT(in, lut, o);
            return o.at<uint8_t>(32, 32) == 155;  // 255 - 100
        };
        cases.push_back(bc);
    }

    // Threshold_Binary — U8 in, U8 out, threshold=128.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Threshold_Binary";
        bc.category = "misc";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::threshold(bufs.input, bufs.output, /*thresh=*/128, /*maxval=*/255,
                          cv::THRESH_BINARY);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(200));
            cv::Mat o;
            cv::threshold(in, o, 128, 255, cv::THRESH_BINARY);
            return o.at<uint8_t>(32, 32) == 255;
        };
        cases.push_back(bc);
    }

    // Threshold_Range — U8 in, U8 out, low=64, high=192.
    //
    // OpenVX VX_THRESHOLD_TYPE_RANGE outputs 255 inside the range and
    // 0 outside; cv::inRange has the same contract.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Threshold_Range";
        bc.category = "misc";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::inRange(bufs.input, cv::Scalar(64), cv::Scalar(192), bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat o;
            cv::inRange(in, cv::Scalar(64), cv::Scalar(192), o);
            return o.at<uint8_t>(32, 32) == 255;
        };
        cases.push_back(bc);
    }

    // WeightedAverage — two U8 inputs, U8 out, alpha=0.7.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "WeightedAverage";
        bc.category = "misc";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.input_extra = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::addWeighted(bufs.input, 0.7, bufs.input_extra, 0.3, 0.0,
                            bufs.output, CV_8U);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat a(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat b(64, 64, CV_8UC1, cv::Scalar(200));
            cv::Mat o;
            cv::addWeighted(a, 0.7, b, 0.3, 0.0, o, CV_8U);
            // 0.7*100 + 0.3*200 = 70 + 60 = 130
            return std::abs(o.at<uint8_t>(32, 32) - 130) <= 1;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
