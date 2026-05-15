// OpenCV equivalents for the OpenVX `color` category.
//
// PR1 sentinel set: ColorConvert_RGB2IYUV (RGB -> YUV 4:2:0 planar).
//
// Parameter mapping notes:
//
//   * OpenVX's vxColorConvertNode(RGB -> IYUV) is BT.601 limited-range
//     by default in MIVisionX/Khronos sample/rustVX. OpenCV's
//     `cv::COLOR_RGB2YUV_I420` is also BT.601 limited-range — same
//     conversion matrix. The verify check accepts a small tolerance
//     because integer rounding differs by ±1 grey level between
//     implementations.
//   * IYUV (== YUV_I420) requires even width and height. We coerce
//     the resolution to even values inside setup_fn, mirroring what
//     openvx-mark does for the same kernel.

#include "opencv_runner.h"
#include "opencv_verify.h"
#include <opencv2/imgproc.hpp>
#include <vector>

namespace opencv_mark {

std::vector<OpenCVBenchmarkCase> registerCvColorBenchmarks() {
    std::vector<OpenCVBenchmarkCase> cases;

    {
        OpenCVBenchmarkCase bc;
        bc.name = "ColorConvert_RGB2IYUV";
        bc.category = "color";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            const uint32_t ew = w & ~1u;
            const uint32_t eh = h & ~1u;
            if (ew == 0 || eh == 0) return false;
            bufs.input = gen.makeRGB(ew, eh);
            // YUV_I420 layout: Y is full-resolution, U and V are
            // quarter-resolution and stacked vertically. OpenCV stores
            // the result in a single CV_8UC1 Mat of height = h * 3 / 2.
            bufs.output.create(static_cast<int>(eh + eh / 2),
                               static_cast<int>(ew), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::cvtColor(bufs.input, bufs.output, cv::COLOR_RGB2YUV_I420);
        };
        bc.verify_fn = []() -> bool {
            // Constant RGB(200, 100, 50) — Y ≈ 124 (BT.601). Accept
            // the same band openvx-mark does so OpenVX vs OpenCV
            // numerical differences (BT.601 vs BT.709, integer
            // rounding) don't trip self-verification here.
            cv::Mat in(64, 64, CV_8UC3, cv::Scalar(200, 100, 50));
            cv::Mat out;
            cv::cvtColor(in, out, cv::COLOR_RGB2YUV_I420);
            const uint8_t y = out.at<uint8_t>(32, 32);
            return y >= 115 && y <= 130;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
