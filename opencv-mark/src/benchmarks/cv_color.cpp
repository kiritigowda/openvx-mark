// OpenCV equivalents for the OpenVX `color` category.
//
// Name parity with openvx-mark — every benchmark below shares its
// name with an openvx-mark entry so scripts/compare_reports.py joins
// them on (name, mode, resolution).
//
// Skipped openvx-mark entries (no native OpenCV equivalent):
//
//   * ColorConvert_RGB2NV12 — OpenCV has no direct RGB→NV12
//     cv::cvtColor (only NV12/NV21 → RGB). Emulating it via cvtColor
//     to YUV plus manual U/V interleave would compare two kernels'
//     worth of OpenCV work against one kernel's worth of OpenVX work,
//     which isn't apples-to-apples.
//   * ChannelExtract_NV12_Y — NV12 isn't a native OpenCV `cv::Mat`
//     layout. Y-plane extraction from an NV12 buffer is a memcpy of
//     the first w*h bytes; not a real benchmark.
//   * ChannelExtract_IYUV_U — same: extracting the U plane from an
//     IYUV/I420 buffer is a memcpy of a sub-region, not measurable
//     per-pixel work.
//
// Parameter mapping notes:
//
//   * OpenVX's vxColorConvertNode(RGB → IYUV) is BT.601 limited-range
//     by default in MIVisionX/Khronos sample/rustVX. OpenCV's
//     `cv::COLOR_RGB2YUV_I420` is also BT.601 limited-range. Verify
//     checks accept a small tolerance because integer rounding
//     differs by ±1 grey level between implementations.
//   * IYUV (== YUV_I420) and NV12 require even width/height; we coerce
//     the resolution inside setup_fn, mirroring openvx-mark.
//   * ColorConvert_RGB2YUV4: OpenVX's YUV4 is the 3-plane planar
//     no-chroma-subsampling format. OpenCV's `cv::COLOR_RGB2YUV`
//     produces a 3-channel interleaved YUV image with the same per-
//     pixel arithmetic (full Kr/Kb matrix multiply); the storage
//     layout differs but the per-pixel cost matches.
//   * ColorConvert_IYUV2RGB: cv::COLOR_YUV2RGB_I420 does the inverse
//     of the RGB2IYUV path with chroma upsampling.
//   * ChannelExtract: extract channel 0 (R) from RGB — matches
//     openvx-mark default of "extract R from VX_DF_IMAGE_RGB".
//   * ChannelExtract_YUYV_Y: YUYV is a packed CV_8UC2 layout where
//     channel 0 is Y. cv::extractChannel on a CV_8UC2 view picks up
//     every other byte — the same per-pixel work openvx-mark times.
//   * ChannelCombine / ChannelCombine_YUV4: cv::merge of 3 U8 planes.
//     For the YUV4 case the per-pixel work is identical to the RGB
//     case — OpenVX VX_DF_IMAGE_YUV4 is a 3-plane planar U8 format;
//     OpenCV stores it as interleaved CV_8UC3 (same byte count, same
//     per-pixel arithmetic = 0; the cost is purely memory traffic).
//   * ConvertDepth: cv::Mat::convertTo for U8↔S16 conversions. The
//     U8→S16 case is a zero-extend; the S16→U8 case clamps to
//     [0,255] (matching VX_CONVERT_POLICY_SATURATE).

#include "opencv_runner.h"
#include "opencv_verify.h"
#include <cstring>
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

    // ConvertDepth_S16toU8 — S16 in, U8 out (down-cast with saturation)
    {
        OpenCVBenchmarkCase bc;
        bc.name = "ConvertDepth_S16toU8";
        bc.category = "color";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeS16(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            // convertTo with CV_8U saturates to [0, 255] — same semantics as
            // VX_CONVERT_POLICY_SATURATE on the OpenVX path.
            bufs.input.convertTo(bufs.output, CV_8U, /*alpha=*/1.0, /*beta=*/0.0);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_16SC1, cv::Scalar(1000));  // > 255 → must saturate
            cv::Mat out;
            in.convertTo(out, CV_8U, 1.0, 0.0);
            return out.at<uint8_t>(32, 32) == 255;
        };
        cases.push_back(bc);
    }

    // ColorConvert_RGB2YUV4 — RGB in, 3-channel interleaved YUV out
    {
        OpenCVBenchmarkCase bc;
        bc.name = "ColorConvert_RGB2YUV4";
        bc.category = "color";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeRGB(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC3);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::cvtColor(bufs.input, bufs.output, cv::COLOR_RGB2YUV);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC3, cv::Scalar(200, 100, 50));
            cv::Mat out;
            cv::cvtColor(in, out, cv::COLOR_RGB2YUV);
            const auto px = out.at<cv::Vec3b>(32, 32);
            // Y channel for RGB(200,100,50) ≈ 124 (BT.601 / JFIF).
            return px[0] >= 115 && px[0] <= 130;
        };
        cases.push_back(bc);
    }

    // ColorConvert_IYUV2RGB — IYUV (I420) in, RGB out
    {
        OpenCVBenchmarkCase bc;
        bc.name = "ColorConvert_IYUV2RGB";
        bc.category = "color";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            const uint32_t ew = w & ~1u;
            const uint32_t eh = h & ~1u;
            if (ew == 0 || eh == 0) return false;
            // I420 layout is a single-channel Mat of size (h * 3/2) × w with
            // Y on top and U/V stacked below as quarter-resolution planes.
            bufs.input.create(static_cast<int>(eh + eh / 2),
                              static_cast<int>(ew), CV_8UC1);
            bufs.input.setTo(cv::Scalar(128));  // mid-grey luma + neutral chroma
            bufs.output.create(static_cast<int>(eh), static_cast<int>(ew), CV_8UC3);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::cvtColor(bufs.input, bufs.output, cv::COLOR_YUV2RGB_I420);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(96, 64, CV_8UC1, cv::Scalar(128));  // 64x64 I420 = 64x96
            cv::Mat out;
            cv::cvtColor(in, out, cv::COLOR_YUV2RGB_I420);
            const auto px = out.at<cv::Vec3b>(32, 32);
            // Y=128, U=V=128 → mid-grey RGB. Accept wide tolerance because
            // BT.601 vs BT.709 vs JFIF differ by ±25 LSB at mid-grey.
            return px[0] >= 95 && px[0] <= 160;
        };
        cases.push_back(bc);
    }

    // ChannelExtract_YUYV_Y — YUYV (packed) in, U8 (Y plane) out.
    //
    // YUYV is byte-interleaved: Y0 U0 Y1 V0 Y2 U1 Y3 V1 ... — every
    // other byte is Y. We view the input buffer as CV_8UC2 (so each
    // pixel pair is one 2-byte element) and extract channel 0 = Y.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "ChannelExtract_YUYV_Y";
        bc.category = "color";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            const uint32_t ew = w & ~1u;
            if (ew == 0 || h == 0) return false;
            // CV_8UC2 of width=W/2 has the same byte footprint as a YUYV
            // image of width W: each pair (Y, U) or (Y, V) is one 2-byte
            // pixel. Channel 0 = Y, channel 1 = U/V alternating.
            // But for the Y extraction case we want a CV_8UC2 of width=W
            // so channel 0 extraction gives a W-wide Y plane.
            bufs.input.create(static_cast<int>(h), static_cast<int>(ew), CV_8UC2);
            cv::Mat tmp = gen.makeU8(ew * 2, h);  // 2x bytes for YUYV
            std::memcpy(bufs.input.data, tmp.data,
                        static_cast<size_t>(ew) * 2 * h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(ew), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::extractChannel(bufs.input, bufs.output, /*coi=*/0);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC2, cv::Scalar(42, 99));
            cv::Mat out;
            cv::extractChannel(in, out, 0);
            // Channel 0 should pick up the 42 byte from every pixel pair.
            return out.at<uint8_t>(32, 32) == 42;
        };
        cases.push_back(bc);
    }

    // ChannelCombine_YUV4 — three U8 planes in, YUV4 (3-ch interleaved) out.
    //
    // OpenVX YUV4 is 3-plane planar U8. OpenCV stores it as
    // interleaved CV_8UC3 (same per-pixel byte count, same merge cost
    // — only the cache stride pattern differs). cv::merge is the same
    // call we use for ChannelCombine → RGB.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "ChannelCombine_YUV4";
        bc.category = "color";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.input_extra = gen.makeU8(w, h);
            bufs.output_extra = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC3);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            const cv::Mat planes[3] = {bufs.input, bufs.input_extra, bufs.output_extra};
            cv::merge(planes, 3, bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat y(64, 64, CV_8UC1, cv::Scalar(128));
            cv::Mat u(64, 64, CV_8UC1, cv::Scalar(64));
            cv::Mat v(64, 64, CV_8UC1, cv::Scalar(192));
            const cv::Mat planes[3] = {y, u, v};
            cv::Mat out;
            cv::merge(planes, 3, out);
            const auto px = out.at<cv::Vec3b>(32, 32);
            return px[0] == 128 && px[1] == 64 && px[2] == 192;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
