// OpenCV equivalents for the OpenVX `filters` category.
//
// PR1 sentinel set: Box3x3, Gaussian3x3, Sobel3x3.
//
// Parameter mapping notes (so a reader of a future cross-impl
// comparison report knows what is — and isn't — apples-to-apples):
//
//   * Border mode is `cv::BORDER_REPLICATE` for every filter here.
//     OpenVX's default is VX_BORDER_UNDEFINED but every real
//     implementation we have seen treats border samples as
//     replicate-edge in practice; matching that minimises edge-pixel
//     numerical differences.
//   * `cv::boxFilter` is configured `normalize=true, ksize=3` to match
//     the OpenVX `vxBox3x3Node` definition (mean of 3×3 neighbours).
//   * `cv::GaussianBlur` is called with `sigmaX=0` so OpenCV derives
//     sigma from ksize=3, which yields the same canonical
//     [1 2 1; 2 4 2; 1 2 1]/16 weights OpenVX's Gaussian3x3 uses.
//   * `cv::Sobel` is called twice (one dx, one dy) with ksize=3,
//     ddepth=CV_16S — same dual-output S16 contract as
//     `vxSobel3x3Node`. Both calls are inside the timing budget so
//     the apples-to-apples cost matches OpenVX's two-output node.

#include "opencv_runner.h"
#include "opencv_verify.h"
#include <opencv2/imgproc.hpp>
#include <vector>

namespace opencv_mark {

std::vector<OpenCVBenchmarkCase> registerCvFilterBenchmarks() {
    std::vector<OpenCVBenchmarkCase> cases;

    // Box3x3 — U8 in, U8 out
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Box3x3";
        bc.category = "filters";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::boxFilter(bufs.input, bufs.output, CV_8U, cv::Size(3, 3),
                          cv::Point(-1, -1), /*normalize=*/true, cv::BORDER_REPLICATE);
        };
        bc.verify_fn = []() -> bool {
            // Constant-100 input → constant-100 output for a normalized
            // 3×3 box filter (same self-check shape openvx-mark uses).
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat out;
            cv::boxFilter(in, out, CV_8U, cv::Size(3, 3),
                          cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
            return out.at<uint8_t>(32, 32) == 100;
        };
        cases.push_back(bc);
    }

    // Gaussian3x3 — U8 in, U8 out
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Gaussian3x3";
        bc.category = "filters";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::GaussianBlur(bufs.input, bufs.output, cv::Size(3, 3),
                             /*sigmaX=*/0, /*sigmaY=*/0, cv::BORDER_REPLICATE);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat out;
            cv::GaussianBlur(in, out, cv::Size(3, 3), 0, 0, cv::BORDER_REPLICATE);
            return out.at<uint8_t>(32, 32) == 100;
        };
        cases.push_back(bc);
    }

    // Sobel3x3 — U8 in, S16 dx + S16 dy out (both timed together to
    // match vxSobel3x3Node's single-call dual-output contract).
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Sobel3x3";
        bc.category = "filters";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            bufs.output_extra.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::Sobel(bufs.input, bufs.output,       CV_16S, /*dx=*/1, /*dy=*/0,
                      /*ksize=*/3, /*scale=*/1, /*delta=*/0, cv::BORDER_REPLICATE);
            cv::Sobel(bufs.input, bufs.output_extra, CV_16S, /*dx=*/0, /*dy=*/1,
                      /*ksize=*/3, /*scale=*/1, /*delta=*/0, cv::BORDER_REPLICATE);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat dx, dy;
            cv::Sobel(in, dx, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
            cv::Sobel(in, dy, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
            return dx.at<int16_t>(32, 32) == 0 && dy.at<int16_t>(32, 32) == 0;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
