// OpenCV equivalents for the OpenVX `color` category.
//
// PR1 set: ColorConvert_RGB2IYUV.
// PR2 set: ChannelExtract, ChannelCombine, ConvertDepth.
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
//   * ChannelExtract: cv::extractChannel(src, dst, channel) — we
//     extract channel 0 (R) from a 3-channel RGB image, matching the
//     openvx-mark default of "extract R from VX_DF_IMAGE_RGB".
//   * ChannelCombine: cv::merge(channels, dst) — three U8 inputs into
//     a single CV_8UC3, matching vxChannelCombineNode for RGB output.
//   * ConvertDepth: cv::Mat::convertTo(dst, CV_16S) — same up-cast
//     U8→S16 vxConvertDepthNode does. Down-cast (S16→U8 with
//     saturation) is omitted from the sentinel set; openvx-mark's
//     ConvertDepth benchmark only times the U8→S16 direction so we
//     match.
//   * ColorConvert_RGB2NV12 is intentionally NOT included: OpenCV has
//     no direct cvtColor for the forward RGB→NV12 path; emulating it
//     would require a manual U/V interleave step that isn't an
//     apples-to-apples OpenCV kernel call. Documented in the umbrella
//     PR rationale.

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

    // ChannelExtract — RGB in, U8 out (extracts channel 0 = R)
    {
        OpenCVBenchmarkCase bc;
        bc.name = "ChannelExtract";
        bc.category = "color";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeRGB(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::extractChannel(bufs.input, bufs.output, /*coi=*/0);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC3, cv::Scalar(200, 100, 50));
            cv::Mat out;
            cv::extractChannel(in, out, 0);
            return out.at<uint8_t>(32, 32) == 200;
        };
        cases.push_back(bc);
    }

    // ChannelCombine — three U8 in, RGB out
    {
        OpenCVBenchmarkCase bc;
        bc.name = "ChannelCombine";
        bc.category = "color";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.input_extra = gen.makeU8(w, h);
            // Stash third plane in the output Mat slot — `output` will
            // be repurposed as the merged 3-channel Mat by run_fn.
            bufs.output_extra = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC3);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            const cv::Mat planes[3] = {bufs.input, bufs.input_extra, bufs.output_extra};
            cv::merge(planes, 3, bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat r(64, 64, CV_8UC1, cv::Scalar(200));
            cv::Mat g(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat b(64, 64, CV_8UC1, cv::Scalar(50));
            const cv::Mat planes[3] = {r, g, b};
            cv::Mat out;
            cv::merge(planes, 3, out);
            const auto px = out.at<cv::Vec3b>(32, 32);
            return px[0] == 200 && px[1] == 100 && px[2] == 50;
        };
        cases.push_back(bc);
    }

    // ConvertDepth — U8 in, S16 out (up-cast)
    {
        OpenCVBenchmarkCase bc;
        bc.name = "ConvertDepth";
        bc.category = "color";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            bufs.input.convertTo(bufs.output, CV_16S, /*alpha=*/1.0, /*beta=*/0.0);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(200));
            cv::Mat out;
            in.convertTo(out, CV_16S, 1.0, 0.0);
            return out.at<int16_t>(32, 32) == 200;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
