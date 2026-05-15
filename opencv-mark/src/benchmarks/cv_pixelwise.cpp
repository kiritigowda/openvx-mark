// OpenCV equivalents for the OpenVX `pixelwise` category.
//
// PR2 set: And, Or, Xor, Not, Add, Subtract, Multiply, AbsDiff.
//
// Parameter mapping notes:
//
//   * Bitwise ops (And/Or/Xor/Not) operate on CV_8UC1 inputs to match
//     openvx-mark which exclusively benchmarks the U8 variant. cv::
//     equivalents are vectorized over the same byte-per-pixel layout
//     so this is the cleanest apples-to-apples comparison in the suite.
//   * Arithmetic ops (Add/Subtract) use saturated U8→U8. OpenVX
//     supports a saturate flag (VX_CONVERT_POLICY_SATURATE) and
//     openvx-mark uses it. cv::add / cv::subtract default to saturation
//     when the dst type is U8.
//   * Multiply uses scale=1.0 / saturation = on, matching openvx-mark's
//     vxMultiplyNode default. The integer-rounding semantics differ
//     subtly between OpenVX (truncate) and OpenCV (round-to-nearest);
//     this is documented and accepted as a "known epsilon" in the
//     PSNR-based cross-impl tolerance check. Self-verify uses constant
//     inputs so rounding doesn't trip the check.
//   * AbsDiff is straightforward — same operation in both libraries.

#include "opencv_runner.h"
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

    return cases;
}

} // namespace opencv_mark
