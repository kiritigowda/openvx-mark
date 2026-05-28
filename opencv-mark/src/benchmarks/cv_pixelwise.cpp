// OpenCV equivalents for the OpenVX `pixelwise` category.
//
// Name and (input,output) format parity with openvx-mark — the join
// key in scripts/compare_reports.py is (name, mode, resolution), so
// every benchmark below mirrors an openvx-mark entry by name AND uses
// the same cv::Mat depths as the matching OpenVX vx_df_image. Where
// the data types are signed/wider, the cv:: equivalent is forced
// into the same depth via the dst-type argument (not via convertTo
// inside the timing loop), keeping the per-call work apples-to-apples.
//
// Parameter mapping notes:
//
//   * Bitwise ops (And/Or/Xor/Not) operate on CV_8UC1 to match
//     openvx-mark's U8 vxAndNode etc.
//   * Saturation arithmetic (Add/Subtract/Multiply U8→U8): cv:: defaults
//     to saturation when dst type is CV_8U, matching
//     VX_CONVERT_POLICY_SATURATE.
//   * Multiply scale=1 / saturate matches openvx-mark's Multiply (U8→U8).
//   * Multiply_S16_S16_S16 uses scale=1/255 + ROUND_NEAREST_EVEN to
//     match the openvx-mark variant — this is one of the two required
//     scale/rounding combinations from OpenVX 1.3.1 §3.42 [REQ-0371].
//   * AbsDiff_S16 / Add_S16_S16_S16 / Subtract_S16_S16_S16: same
//     operation as the U8 variants but on CV_16SC1 (S16) buffers,
//     matching the (S16,S16)→S16 path required by [REQ-0011..0027].
//   * (U8,U8)→S16 variants for Add/Subtract/Multiply use wrap
//     convention (no saturation) so out-of-U8-range values are
//     representable in the S16 output — same shape openvx-mark uses
//     for its _U8_U8_S16 variants.

#include "opencv_runner.h"
#include <cstdlib>
#include <opencv2/core.hpp>
#include <vector>

namespace opencv_mark {

namespace {
// Helper: builds a benchmark case whose run_fn calls one of the simple
// "one or two U8 inputs in, U8 out" cv:: functions. Eliminates the
// boilerplate that would otherwise duplicate across 8 nearly-identical
// kernels.
template <typename RunFn>
OpenCVBenchmarkCase makeBinaryU8Case(const char* name, RunFn run) {
    OpenCVBenchmarkCase bc;
    bc.name = name;
    bc.category = "pixelwise";
    bc.feature_set = "vision";
    bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
        bufs.input = gen.makeU8(w, h);
        bufs.input_extra = gen.makeU8(w, h);
        bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
        return true;
    };
    bc.run_fn = run;
    return bc;
}
}  // namespace

std::vector<OpenCVBenchmarkCase> registerCvPixelwiseBenchmarks() {
    std::vector<OpenCVBenchmarkCase> cases;

    cases.push_back(makeBinaryU8Case("And", [](CaseBuffers& bufs) {
        cv::bitwise_and(bufs.input, bufs.input_extra, bufs.output);
    }));
    cases.back().verify_fn = []() -> bool {
        cv::Mat a(64, 64, CV_8UC1, cv::Scalar(0xF0));
        cv::Mat b(64, 64, CV_8UC1, cv::Scalar(0x33));
        cv::Mat o; cv::bitwise_and(a, b, o);
        return o.at<uint8_t>(32, 32) == 0x30;
    };

    cases.push_back(makeBinaryU8Case("Or", [](CaseBuffers& bufs) {
        cv::bitwise_or(bufs.input, bufs.input_extra, bufs.output);
    }));
    cases.back().verify_fn = []() -> bool {
        cv::Mat a(64, 64, CV_8UC1, cv::Scalar(0xF0));
        cv::Mat b(64, 64, CV_8UC1, cv::Scalar(0x33));
        cv::Mat o; cv::bitwise_or(a, b, o);
        return o.at<uint8_t>(32, 32) == 0xF3;
    };

    cases.push_back(makeBinaryU8Case("Xor", [](CaseBuffers& bufs) {
        cv::bitwise_xor(bufs.input, bufs.input_extra, bufs.output);
    }));
    cases.back().verify_fn = []() -> bool {
        cv::Mat a(64, 64, CV_8UC1, cv::Scalar(0xF0));
        cv::Mat b(64, 64, CV_8UC1, cv::Scalar(0x33));
        cv::Mat o; cv::bitwise_xor(a, b, o);
        return o.at<uint8_t>(32, 32) == 0xC3;
    };

    // Not is unary — single input, no input_extra.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Not";
        bc.category = "pixelwise";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) { cv::bitwise_not(bufs.input, bufs.output); };
        bc.verify_fn = []() -> bool {
            cv::Mat a(64, 64, CV_8UC1, cv::Scalar(0x55));
            cv::Mat o; cv::bitwise_not(a, o);
            return o.at<uint8_t>(32, 32) == 0xAA;
        };
        cases.push_back(bc);
    }

    cases.push_back(makeBinaryU8Case("AbsDiff", [](CaseBuffers& bufs) {
        cv::absdiff(bufs.input, bufs.input_extra, bufs.output);
    }));
    cases.back().verify_fn = []() -> bool {
        cv::Mat a(64, 64, CV_8UC1, cv::Scalar(150));
        cv::Mat b(64, 64, CV_8UC1, cv::Scalar(50));
        cv::Mat o; cv::absdiff(a, b, o);
        return o.at<uint8_t>(32, 32) == 100;
    };

    cases.push_back(makeBinaryU8Case("Add", [](CaseBuffers& bufs) {
        // dst type CV_8U → cv::add saturates to [0, 255], matching
        // OpenVX VX_CONVERT_POLICY_SATURATE.
        cv::add(bufs.input, bufs.input_extra, bufs.output, cv::noArray(), CV_8U);
    }));
    cases.back().verify_fn = []() -> bool {
        cv::Mat a(64, 64, CV_8UC1, cv::Scalar(200));
        cv::Mat b(64, 64, CV_8UC1, cv::Scalar(100));
        cv::Mat o; cv::add(a, b, o, cv::noArray(), CV_8U);
        return o.at<uint8_t>(32, 32) == 255;  // saturated
    };

    cases.push_back(makeBinaryU8Case("Subtract", [](CaseBuffers& bufs) {
        cv::subtract(bufs.input, bufs.input_extra, bufs.output, cv::noArray(), CV_8U);
    }));
    cases.back().verify_fn = []() -> bool {
        cv::Mat a(64, 64, CV_8UC1, cv::Scalar(50));
        cv::Mat b(64, 64, CV_8UC1, cv::Scalar(100));
        cv::Mat o; cv::subtract(a, b, o, cv::noArray(), CV_8U);
        return o.at<uint8_t>(32, 32) == 0;  // saturated
    };

    cases.push_back(makeBinaryU8Case("Multiply", [](CaseBuffers& bufs) {
        cv::multiply(bufs.input, bufs.input_extra, bufs.output, /*scale=*/1.0, CV_8U);
    }));
    cases.back().verify_fn = []() -> bool {
        cv::Mat a(64, 64, CV_8UC1, cv::Scalar(20));
        cv::Mat b(64, 64, CV_8UC1, cv::Scalar(5));
        cv::Mat o; cv::multiply(a, b, o, 1.0, CV_8U);
        return o.at<uint8_t>(32, 32) == 100;
    };

    // ---- AbsDiff_S16 (S16 in, S16 out) ----
    {
        OpenCVBenchmarkCase bc;
        bc.name = "AbsDiff_S16";
        bc.category = "pixelwise";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeS16(w, h);
            bufs.input_extra = gen.makeS16(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::absdiff(bufs.input, bufs.input_extra, bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat a(64, 64, CV_16SC1, cv::Scalar(-200));
            cv::Mat b(64, 64, CV_16SC1, cv::Scalar( 300));
            cv::Mat o; cv::absdiff(a, b, o);
            return o.at<int16_t>(32, 32) == 500;
        };
        cases.push_back(bc);
    }

    // ---- Add_U8_U8_S16 (U8 + U8 → S16, wrap) ----
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Add_U8_U8_S16";
        bc.category = "pixelwise";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.input_extra = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::add(bufs.input, bufs.input_extra, bufs.output, cv::noArray(), CV_16S);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat a(64, 64, CV_8UC1, cv::Scalar(200));
            cv::Mat b(64, 64, CV_8UC1, cv::Scalar(200));
            cv::Mat o; cv::add(a, b, o, cv::noArray(), CV_16S);
            // 200 + 200 = 400 (fits in S16; would saturate to 255 in U8)
            return o.at<int16_t>(32, 32) == 400;
        };
        cases.push_back(bc);
    }

    // ---- Add_S16_S16_S16 (S16 + S16 → S16, saturate) ----
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Add_S16_S16_S16";
        bc.category = "pixelwise";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeS16(w, h);
            bufs.input_extra = gen.makeS16(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::add(bufs.input, bufs.input_extra, bufs.output, cv::noArray(), CV_16S);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat a(64, 64, CV_16SC1, cv::Scalar(-1000));
            cv::Mat b(64, 64, CV_16SC1, cv::Scalar( 4000));
            cv::Mat o; cv::add(a, b, o, cv::noArray(), CV_16S);
            return o.at<int16_t>(32, 32) == 3000;
        };
        cases.push_back(bc);
    }

    // ---- Subtract_U8_U8_S16 (U8 - U8 → S16, wrap) ----
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Subtract_U8_U8_S16";
        bc.category = "pixelwise";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.input_extra = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::subtract(bufs.input, bufs.input_extra, bufs.output, cv::noArray(), CV_16S);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat a(64, 64, CV_8UC1, cv::Scalar(50));
            cv::Mat b(64, 64, CV_8UC1, cv::Scalar(200));
            cv::Mat o; cv::subtract(a, b, o, cv::noArray(), CV_16S);
            // 50 - 200 = -150 (representable in S16; would underflow to 0 in U8)
            return o.at<int16_t>(32, 32) == -150;
        };
        cases.push_back(bc);
    }

    // ---- Subtract_S16_S16_S16 (S16 - S16 → S16, saturate) ----
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Subtract_S16_S16_S16";
        bc.category = "pixelwise";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeS16(w, h);
            bufs.input_extra = gen.makeS16(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::subtract(bufs.input, bufs.input_extra, bufs.output, cv::noArray(), CV_16S);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat a(64, 64, CV_16SC1, cv::Scalar(5000));
            cv::Mat b(64, 64, CV_16SC1, cv::Scalar(2000));
            cv::Mat o; cv::subtract(a, b, o, cv::noArray(), CV_16S);
            return o.at<int16_t>(32, 32) == 3000;
        };
        cases.push_back(bc);
    }

    // ---- Multiply_U8_U8_S16 (U8 * U8 → S16, scale=1, wrap) ----
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Multiply_U8_U8_S16";
        bc.category = "pixelwise";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.input_extra = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::multiply(bufs.input, bufs.input_extra, bufs.output, /*scale=*/1.0, CV_16S);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat a(64, 64, CV_8UC1, cv::Scalar(200));
            cv::Mat b(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat o; cv::multiply(a, b, o, 1.0, CV_16S);
            // 200 * 100 = 20000 (fits in S16; would saturate in U8)
            return o.at<int16_t>(32, 32) == 20000;
        };
        cases.push_back(bc);
    }

    // ---- Min (enhanced_vision) ----
    //
    // OpenVX 1.3.1 §3.36: vxMinNode is per-pixel min on U8 or S16
    // images with both inputs and output sharing the same depth.
    // cv::min is the direct equivalent.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Min";
        bc.category = "pixelwise";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.input_extra = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::min(bufs.input, bufs.input_extra, bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat a(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat b(64, 64, CV_8UC1, cv::Scalar(150));
            cv::Mat o; cv::min(a, b, o);
            return o.at<uint8_t>(32, 32) == 100;
        };
        cases.push_back(bc);
    }

    // ---- Max (enhanced_vision) ----
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Max";
        bc.category = "pixelwise";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.input_extra = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::max(bufs.input, bufs.input_extra, bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat a(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat b(64, 64, CV_8UC1, cv::Scalar(150));
            cv::Mat o; cv::max(a, b, o);
            return o.at<uint8_t>(32, 32) == 150;
        };
        cases.push_back(bc);
    }

    // ---- Copy (enhanced_vision) ----
    //
    // OpenVX 1.3.1 §3.17: vxCopyNode copies a data object. For images
    // the equivalent is cv::Mat::copyTo (full memcpy with no
    // arithmetic). The cost is purely memory traffic.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Copy";
        bc.category = "pixelwise";
        bc.feature_set = "enhanced_vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            bufs.input.copyTo(bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(42));
            cv::Mat out; in.copyTo(out);
            return out.at<uint8_t>(32, 32) == 42;
        };
        cases.push_back(bc);
    }

    // ---- Multiply_S16_S16_S16 (S16 * S16 → S16, scale=1/255, NEAREST_EVEN) ----
    //
    // OpenVX 1.3.1 §3.42 [REQ-0371]: scale=1/255 must be supported with
    // VX_ROUND_POLICY_TO_NEAREST_EVEN. cv::multiply uses banker's
    // rounding for floating-point intermediates, which is the same as
    // round-half-to-even, so this is a true apples-to-apples comparison.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Multiply_S16_S16_S16";
        bc.category = "pixelwise";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeS16(w, h);
            bufs.input_extra = gen.makeS16(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::multiply(bufs.input, bufs.input_extra, bufs.output, /*scale=*/1.0 / 255.0, CV_16S);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat a(64, 64, CV_16SC1, cv::Scalar(255));
            cv::Mat b(64, 64, CV_16SC1, cv::Scalar(200));
            cv::Mat o; cv::multiply(a, b, o, 1.0 / 255.0, CV_16S);
            // 255 * 200 / 255 = 200 (±1 for rounding)
            return std::abs(o.at<int16_t>(32, 32) - 200) <= 1;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
