// OpenCV equivalents for the OpenVX `filters` category.
//
// PR1 set: Box3x3, Gaussian3x3, Sobel3x3.
// PR2 set: Median3x3, Erode3x3, Dilate3x3, CustomConvolution.
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
//   * `cv::medianBlur` with ksize=3 matches `vxMedian3x3Node`.
//   * `cv::erode` / `cv::dilate` use a 3x3 rectangular structuring
//     element (`cv::MORPH_RECT`) with anchor at centre to match
//     `vxErode3x3Node` / `vxDilate3x3Node` semantics.
//   * `cv::filter2D` for CustomConvolution uses a 3x3 CV_16S kernel
//     identical to the one openvx-mark feeds vxConvolveNode, with
//     ddepth=CV_8U, anchor at centre, delta=0, BORDER_REPLICATE.

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

    // Median3x3 — U8 in, U8 out
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Median3x3";
        bc.category = "filters";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::medianBlur(bufs.input, bufs.output, /*ksize=*/3);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat out;
            cv::medianBlur(in, out, 3);
            return out.at<uint8_t>(32, 32) == 100;
        };
        cases.push_back(bc);
    }

    // Erode3x3 — U8 in, U8 out (3x3 rectangular structuring element)
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Erode3x3";
        bc.category = "filters";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            static const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::erode(bufs.input, bufs.output, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat out;
            const cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::erode(in, out, k, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
            return out.at<uint8_t>(32, 32) == 100;
        };
        cases.push_back(bc);
    }

    // Dilate3x3 — U8 in, U8 out (3x3 rectangular structuring element)
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Dilate3x3";
        bc.category = "filters";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            static const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::dilate(bufs.input, bufs.output, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat out;
            const cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::dilate(in, out, k, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
            return out.at<uint8_t>(32, 32) == 100;
        };
        cases.push_back(bc);
    }

    // CustomConvolution — U8 in, U8 out (3x3 CV_16S kernel)
    {
        OpenCVBenchmarkCase bc;
        bc.name = "CustomConvolution";
        bc.category = "filters";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            bufs.input_extra = gen.makeConvolution3x3();
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::filter2D(bufs.input, bufs.output, CV_8U, bufs.input_extra,
                         cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        };
        bc.verify_fn = []() -> bool {
            // Identity kernel (all zeros except centre = 1) → output == input.
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat k = cv::Mat::zeros(3, 3, CV_16SC1);
            k.at<int16_t>(1, 1) = 1;
            cv::Mat out;
            cv::filter2D(in, out, CV_8U, k, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
            return out.at<uint8_t>(32, 32) == 100;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
