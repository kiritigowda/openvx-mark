// OpenCV equivalents for the OpenVX `geometric` category.
//
// Name parity with openvx-mark — bilinear is the default variant
// (e.g. ScaleImage_Half), nearest-neighbour has a `_Nearest` suffix
// (ScaleImage_Nearest_Half, WarpAffine_Nearest, ...). Area
// resampling for ScaleImage is the `_Area_Half` suffix.
//
// Parameter mapping notes:
//
//   * Interpolation: INTER_LINEAR matches VX_INTERPOLATION_BILINEAR,
//     INTER_NEAREST matches VX_INTERPOLATION_NEAREST_NEIGHBOR, and
//     INTER_AREA matches VX_INTERPOLATION_AREA (only valid for
//     vxScaleImageNode per OpenVX 1.3.1 §3.45 — not supported by
//     vxWarpAffineNode / vxWarpPerspectiveNode / vxRemapNode).
//   * Border mode = BORDER_CONSTANT(0). OpenVX defaults to UNDEFINED;
//     in practice MIVisionX/Khronos sample treat out-of-image samples
//     as 0 so the per-pixel work matches.
//   * The 2x3 affine matrix comes from OpenCVTestData::makeAffineMatrix
//     which mirrors openvx-mark's own ~5° rotation + small translation
//     so the per-pixel resampling work is the same across both binaries.
//   * The 3x3 perspective matrix comes from
//     OpenCVTestData::makePerspectiveMatrix — small homography close to
//     identity with a non-zero perspective term so warpPerspective
//     doesn't degenerate into an affine fast path.
//   * Remap: a CV_32FC1 mapX + CV_32FC1 mapY pair generated as
//     identity-with-tiny-offset so the sampling path does real work.

#include "opencv_runner.h"
#include "opencv_verify.h"
#include <opencv2/imgproc.hpp>
#include <vector>

namespace opencv_mark {

std::vector<OpenCVBenchmarkCase> registerCvGeometricBenchmarks(const BenchmarkConfig& config) {
    std::vector<OpenCVBenchmarkCase> cases;
    const RemapPattern remap_pattern = remapPatternFromString(config.remap_pattern);

    {
        OpenCVBenchmarkCase bc;
        bc.name = "WarpAffine";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            bufs.input_extra = gen.makeAffineMatrix();
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::warpAffine(bufs.input, bufs.output, bufs.input_extra,
                           cv::Size(bufs.output.cols, bufs.output.rows),
                           cv::INTER_LINEAR,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
        };
        bc.verify_fn = []() -> bool {
            // Identity warp on a constant-100 image: the centre pixel
            // should round-trip to ~100. Allow ±2 grey levels because
            // bilinear sampling at integer positions can pull adjacent
            // border-replicated samples in some OpenCV versions.
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat out;
            cv::Mat identity = (cv::Mat_<float>(2, 3) << 1, 0, 0,  0, 1, 0);
            cv::warpAffine(in, out, identity, cv::Size(64, 64),
                           cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
            const int v = out.at<uint8_t>(32, 32);
            return std::abs(v - 100) <= 2;
        };
        cases.push_back(bc);
    }

    // ScaleImage_Half — U8 in (W x H) → U8 out (W/2 x H/2)
    {
        OpenCVBenchmarkCase bc;
        bc.name = "ScaleImage_Half";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h / 2), static_cast<int>(w / 2), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::resize(bufs.input, bufs.output,
                       cv::Size(bufs.output.cols, bufs.output.rows),
                       0, 0, cv::INTER_LINEAR);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat out;
            cv::resize(in, out, cv::Size(32, 32), 0, 0, cv::INTER_LINEAR);
            return out.at<uint8_t>(16, 16) == 100;
        };
        cases.push_back(bc);
    }

    // ScaleImage_Double — U8 in (W x H) → U8 out (2W x 2H)
    {
        OpenCVBenchmarkCase bc;
        bc.name = "ScaleImage_Double";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h * 2), static_cast<int>(w * 2), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::resize(bufs.input, bufs.output,
                       cv::Size(bufs.output.cols, bufs.output.rows),
                       0, 0, cv::INTER_LINEAR);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat out;
            cv::resize(in, out, cv::Size(128, 128), 0, 0, cv::INTER_LINEAR);
            return out.at<uint8_t>(64, 64) == 100;
        };
        cases.push_back(bc);
    }

    // WarpPerspective — U8 in, U8 out, 3x3 perspective matrix
    {
        OpenCVBenchmarkCase bc;
        bc.name = "WarpPerspective";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            bufs.input_extra = gen.makePerspectiveMatrix();
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::warpPerspective(bufs.input, bufs.output, bufs.input_extra,
                                cv::Size(bufs.output.cols, bufs.output.rows),
                                cv::INTER_LINEAR,
                                cv::BORDER_CONSTANT, cv::Scalar(0));
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat out;
            cv::Mat identity = (cv::Mat_<float>(3, 3) <<
                                1, 0, 0,
                                0, 1, 0,
                                0, 0, 1);
            cv::warpPerspective(in, out, identity, cv::Size(64, 64),
                                cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
            const int v = out.at<uint8_t>(32, 32);
            return std::abs(v - 100) <= 2;
        };
        cases.push_back(bc);
    }

    // Remap — U8 in, U8 out, 32FC1 mapX + CV_32FC1 mapY
    //
    // Note: cv::remap uses a separate output for the mapY (we reuse
    // CaseBuffers.output_extra to hold mapY since input_extra is
    // already taken by mapX). The actual output image lives in
    // CaseBuffers.output.
    //
    // The map coordinates default to a radial lens-distortion model
    // (LENS_DISTORTION) so the benchmark exercises scattered, realistic
    // memory access. Use --remap-pattern identity to restore the old
    // cache-friendly behaviour.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Remap";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.setup_fn = [remap_pattern](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            gen.makeRemap(w, h, w, h, bufs.input_extra, bufs.output_extra, remap_pattern);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::remap(bufs.input, bufs.output,
                      bufs.input_extra, bufs.output_extra,
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        };
        bc.verify_fn = []() -> bool {
            // Identity remap should round-trip a uniform image.
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat mx(64, 64, CV_32FC1), my(64, 64, CV_32FC1);
            for (int y = 0; y < 64; ++y) {
                auto* px = mx.ptr<float>(y);
                auto* py = my.ptr<float>(y);
                for (int x = 0; x < 64; ++x) {
                    px[x] = static_cast<float>(x);
                    py[x] = static_cast<float>(y);
                }
            }
            cv::Mat out;
            cv::remap(in, out, mx, my, cv::INTER_LINEAR,
                      cv::BORDER_CONSTANT, cv::Scalar(0));
            return out.at<uint8_t>(32, 32) == 100;
        };
        cases.push_back(bc);
    }

    // ScaleImage_Nearest_Half — INTER_NEAREST, 0.5x
    {
        OpenCVBenchmarkCase bc;
        bc.name = "ScaleImage_Nearest_Half";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h / 2), static_cast<int>(w / 2), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::resize(bufs.input, bufs.output,
                       cv::Size(bufs.output.cols, bufs.output.rows),
                       0, 0, cv::INTER_NEAREST);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat out;
            cv::resize(in, out, cv::Size(32, 32), 0, 0, cv::INTER_NEAREST);
            return out.at<uint8_t>(16, 16) == 100;
        };
        cases.push_back(bc);
    }

    // ScaleImage_Area_Half — INTER_AREA, 0.5x
    {
        OpenCVBenchmarkCase bc;
        bc.name = "ScaleImage_Area_Half";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h / 2), static_cast<int>(w / 2), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::resize(bufs.input, bufs.output,
                       cv::Size(bufs.output.cols, bufs.output.rows),
                       0, 0, cv::INTER_AREA);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat out;
            cv::resize(in, out, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
            return out.at<uint8_t>(16, 16) == 100;
        };
        cases.push_back(bc);
    }

    // WarpAffine_Nearest — INTER_NEAREST
    {
        OpenCVBenchmarkCase bc;
        bc.name = "WarpAffine_Nearest";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            bufs.input_extra = gen.makeAffineMatrix();
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::warpAffine(bufs.input, bufs.output, bufs.input_extra,
                           cv::Size(bufs.output.cols, bufs.output.rows),
                           cv::INTER_NEAREST,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat out;
            cv::Mat identity = (cv::Mat_<float>(2, 3) << 1, 0, 0,  0, 1, 0);
            cv::warpAffine(in, out, identity, cv::Size(64, 64),
                           cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
            return out.at<uint8_t>(32, 32) == 100;
        };
        cases.push_back(bc);
    }

    // WarpPerspective_Nearest — INTER_NEAREST
    {
        OpenCVBenchmarkCase bc;
        bc.name = "WarpPerspective_Nearest";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            bufs.input_extra = gen.makePerspectiveMatrix();
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::warpPerspective(bufs.input, bufs.output, bufs.input_extra,
                                cv::Size(bufs.output.cols, bufs.output.rows),
                                cv::INTER_NEAREST,
                                cv::BORDER_CONSTANT, cv::Scalar(0));
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat out;
            cv::Mat identity = (cv::Mat_<float>(3, 3) <<
                                1, 0, 0,
                                0, 1, 0,
                                0, 0, 1);
            cv::warpPerspective(in, out, identity, cv::Size(64, 64),
                                cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
            return out.at<uint8_t>(32, 32) == 100;
        };
        cases.push_back(bc);
    }

    // Remap_Nearest — INTER_NEAREST. Uses the same coordinate pattern
    // as Remap so both variants exercise realistic memory access.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Remap_Nearest";
        bc.category = "geometric";
        bc.feature_set = "vision";
        bc.setup_fn = [remap_pattern](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            gen.makeRemap(w, h, w, h, bufs.input_extra, bufs.output_extra, remap_pattern);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::remap(bufs.input, bufs.output,
                      bufs.input_extra, bufs.output_extra,
                      cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat mx(64, 64, CV_32FC1), my(64, 64, CV_32FC1);
            for (int y = 0; y < 64; ++y) {
                auto* px = mx.ptr<float>(y);
                auto* py = my.ptr<float>(y);
                for (int x = 0; x < 64; ++x) {
                    px[x] = static_cast<float>(x);
                    py[x] = static_cast<float>(y);
                }
            }
            cv::Mat out;
            cv::remap(in, out, mx, my, cv::INTER_NEAREST,
                      cv::BORDER_CONSTANT, cv::Scalar(0));
            return out.at<uint8_t>(32, 32) == 100;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
