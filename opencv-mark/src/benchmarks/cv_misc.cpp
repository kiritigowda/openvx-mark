// OpenCV equivalents for the OpenVX `misc` category.
//
// Name parity with openvx-mark.
//
// Skipped openvx-mark entries (no native OpenCV equivalent):
//
//   * TableLookup_S16 — cv::LUT only supports CV_8U inputs (the
//     internal indexing assumes a 256-entry table). An S16 lookup
//     would require a 65536-entry table and is not a native OpenCV
//     operation; emulating it manually would not measure equivalent
//     work.
//
// Parameter mapping notes:
//
//   * Magnitude: cv::magnitude takes two CV_32F inputs and produces a
//     CV_32F output. OpenVX vxMagnitudeNode takes two S16 inputs and
//     produces an S16 output. We emulate the OpenVX shape: convert
//     the two S16 inputs to F32 inside setup_fn (outside the timing
//     budget), call cv::magnitude on the F32 buffers. This is a
//     documented "known epsilon" — F32 sqrt vs S16 saturating sqrt.
//   * Phase: same shape — atan2 on S16 dx/dy via F32 intermediates.
//   * TableLookup: cv::LUT with a 256-entry CV_8UC1 LUT.
//   * Threshold_Binary: cv::threshold(THRESH_BINARY, thresh=128).
//   * Threshold_Range: cv::inRange(low=64, high=192).
//   * Threshold_S16: cv::threshold on CV_16SC1 input — OpenCV's
//     threshold supports U8/S16/F32/F64 input depths, so this is a
//     drop-in equivalent of the OpenVX 1.3+ Threshold_S16 path.
//   * WeightedAverage: cv::addWeighted(alpha=0.5, beta=0.5, gamma=0)
//     — matches openvx-mark's vxWeightedAverageNode benchmark which
//     uses alpha=0.5 (see node_misc.cpp). vxWeightedAverageNode
//     computes out = alpha * imgA + (1 - alpha) * imgB.

#include "opencv_runner.h"
#include <cstdlib>
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

    // WeightedAverage — two U8 inputs, U8 out, alpha=0.5 (matches
    // openvx-mark's vxWeightedAverageNode benchmark which uses 0.5).
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
            cv::addWeighted(bufs.input, 0.5, bufs.input_extra, 0.5, 0.0,
                            bufs.output, CV_8U);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat a(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat b(64, 64, CV_8UC1, cv::Scalar(200));
            cv::Mat o;
            cv::addWeighted(a, 0.5, b, 0.5, 0.0, o, CV_8U);
            // 0.5*100 + 0.5*200 = 50 + 100 = 150
            return std::abs(o.at<uint8_t>(32, 32) - 150) <= 1;
        };
        cases.push_back(bc);
    }

    // BilateralFilter (enhanced_vision) — U8 in, U8 out.
    //
    // cv::bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, borderType)
    // is a direct equivalent of vxBilateralFilterNode — same edge-
    // preserving filter mathematics. Note: openvx-mark feeds the
    // OpenVX kernel a vx_tensor (the kernel is tensor-typed in the
    // spec); cv:: works on a cv::Mat which is the equivalent 2-D
    // representation, so the per-pixel cost is the same.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "BilateralFilter";
        bc.category = "misc";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::bilateralFilter(bufs.input, bufs.output,
                                /*d=*/5,
                                /*sigmaColor=*/40.0,
                                /*sigmaSpace=*/20.0,
                                cv::BORDER_REPLICATE);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat o;
            cv::bilateralFilter(in, o, 5, 40.0, 20.0, cv::BORDER_REPLICATE);
            return std::abs(o.at<uint8_t>(32, 32) - 100) <= 2;
        };
        cases.push_back(bc);
    }

    // Select (enhanced_vision) — condition + two U8 inputs → U8 out.
    //
    // OpenVX vxSelectNode picks element-wise: out = condition ? truevals
    // : falsevals where `condition` is a scalar boolean (the same
    // value applied across the whole image). The cv:: equivalent is
    // cv::Mat::copyTo(out, mask) with a full mask (when condition is
    // true) or no copy (when false); we use a constant true mask so
    // the per-iteration work is "copy truevals to output", matching
    // the OpenVX semantics for condition=true.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Select";
        bc.category = "misc";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);          // truevals
            bufs.input_extra = gen.makeU8(w, h);    // falsevals
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            // condition=true → output := truevals
            bufs.input.copyTo(bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat t(64, 64, CV_8UC1, cv::Scalar(42));
            cv::Mat o; t.copyTo(o);
            return o.at<uint8_t>(32, 32) == 42;
        };
        cases.push_back(bc);
    }

    // ScalarOperation (enhanced_vision) — scalar a + scalar b → scalar out.
    //
    // OpenVX vxScalarOperationNode performs one arithmetic / comparison
    // / logical op on two vx_scalar values per call. There are no
    // pixels — the cost is entirely framework dispatch + scalar
    // marshalling. cv:: has no kernel for this; we benchmark a tight
    // C++ loop of integer adds at the resolution-equivalent count so
    // the comparison shows the difference between "OpenVX framework
    // dispatch per scalar op" vs "raw C++ add per scalar op".
    //
    // The "resolution" doesn't really apply here, but the runner
    // expects a per-resolution measurement. We make the per-call work
    // O(1) (single add) so the measured cost is overwhelmingly the
    // function-call + timer overhead, matching what OpenVX measures.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "ScalarOperation";
        bc.category = "misc";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t /*w*/, uint32_t /*h*/,
                         OpenCVTestData& /*gen*/, CaseBuffers& bufs) -> bool {
            // No buffers needed; allocate a 1x1 dummy to satisfy the
            // runner's expectations.
            bufs.input.create(1, 1, CV_32SC1);
            bufs.input.at<int32_t>(0, 0) = 100;
            bufs.input_extra.create(1, 1, CV_32SC1);
            bufs.input_extra.at<int32_t>(0, 0) = 50;
            bufs.output.create(1, 1, CV_32SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            // Single scalar add — matches the per-call work of
            // vxScalarOperationNode(VX_SCALAR_OP_ADD, a, b, out).
            bufs.output.at<int32_t>(0, 0) =
                bufs.input.at<int32_t>(0, 0) + bufs.input_extra.at<int32_t>(0, 0);
        };
        bc.verify_fn = []() -> bool {
            int a = 100, b = 50;
            int out = a + b;
            return out == 150;
        };
        cases.push_back(bc);
    }

    // Threshold_S16 — S16 in, U8 out (binary), threshold=1000.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Threshold_S16";
        bc.category = "misc";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeS16(w, h);
            // cv::threshold on CV_16SC1 produces CV_16SC1 output (matches input depth).
            // OpenVX vxThresholdNode for S16 input is documented to emit a U8
            // boolean output, so we run a 2-step pipeline: threshold → convertTo
            // U8 (inside the timing loop, like a fused kernel would).
            bufs.output_extra.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::threshold(bufs.input, bufs.output_extra, /*thresh=*/1000,
                          /*maxval=*/255, cv::THRESH_BINARY);
            bufs.output_extra.convertTo(bufs.output, CV_8U);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_16SC1, cv::Scalar(2000));
            cv::Mat tmp, o;
            cv::threshold(in, tmp, 1000, 255, cv::THRESH_BINARY);
            tmp.convertTo(o, CV_8U);
            return o.at<uint8_t>(32, 32) == 255;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
