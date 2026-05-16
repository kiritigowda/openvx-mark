// OpenCV equivalents for the OpenVX `geometric` category.
//
// PR1 sentinel set: WarpAffine.
//
// Parameter mapping notes:
//
//   * Interpolation = INTER_LINEAR matches OpenVX's
//     VX_INTERPOLATION_BILINEAR.
//   * Border mode = BORDER_CONSTANT(0). OpenVX defaults to UNDEFINED
//     border behaviour; in practice MIVisionX/Khronos sample treat
//     out-of-image samples as 0 so the per-pixel work matches.
//   * The 2x3 affine matrix comes from OpenCVTestData::makeAffineMatrix
//     which mirrors openvx-mark's own ~5° rotation + small translation
//     so the per-pixel resampling work is the same.

#include "opencv_runner.h"
#include "opencv_verify.h"
#include <opencv2/imgproc.hpp>
#include <vector>

namespace opencv_mark {

std::vector<OpenCVBenchmarkCase> registerCvGeometricBenchmarks() {
    std::vector<OpenCVBenchmarkCase> cases;

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

    return cases;
}

} // namespace opencv_mark
