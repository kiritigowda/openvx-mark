// OpenCV equivalents for the OpenVX `statistical` category.
//
// PR2 set: Histogram, EqualizeHist, MeanStdDev, MinMaxLoc, IntegralImage.
//
// Parameter mapping notes:
//
//   * Histogram uses 256 bins over [0, 256), matching the OpenVX
//     vxHistogramNode default and openvx-mark's benchmark setup.
//     cv::calcHist returns a CV_32F count vector — we discard the
//     return shape since only timing matters.
//   * EqualizeHist is a single cv::equalizeHist call. OpenVX's
//     vxEqualizeHistNode also produces an in-place CDF-equalized U8
//     output; same operation in both libs.
//   * MeanStdDev returns two scalars; we wrap them in cv::Scalar
//     output objects to mirror openvx-mark's vxMeanStdDevNode which
//     produces two vx_scalar outputs.
//   * MinMaxLoc returns four values (min, max, minLoc, maxLoc). We
//     time the cv:: call once per iteration; same single-call
//     contract as openvx-mark.
//   * IntegralImage produces a CV_32S output (CV_32SC1). cv::integral
//     can also produce a CV_64F squared-sum output but openvx-mark
//     only times the integer integral so we match.

#include "opencv_runner.h"
#include <opencv2/imgproc.hpp>
#include <vector>

namespace opencv_mark {

std::vector<OpenCVBenchmarkCase> registerCvStatisticalBenchmarks() {
    std::vector<OpenCVBenchmarkCase> cases;

    // Histogram — U8 in, 256-bin CV_32F histogram out.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "Histogram";
        bc.category = "statistical";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            // cv::calcHist allocates the output histogram itself; we
            // reserve space here so the per-iteration call doesn't have
            // to grow the buffer.
            bufs.output.create(256, 1, CV_32FC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            const int channels[] = {0};
            const int hist_size[] = {256};
            const float range[] = {0, 256};
            const float* ranges[] = {range};
            cv::calcHist(&bufs.input, 1, channels, cv::Mat(),
                         bufs.output, 1, hist_size, ranges,
                         /*uniform=*/true, /*accumulate=*/false);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat hist;
            const int channels[] = {0};
            const int hs[] = {256};
            const float range[] = {0, 256};
            const float* ranges[] = {range};
            cv::calcHist(&in, 1, channels, cv::Mat(), hist, 1, hs, ranges, true, false);
            // All 4096 pixels should fall in bin 100.
            return hist.at<float>(100) == static_cast<float>(64 * 64);
        };
        cases.push_back(bc);
    }

    // EqualizeHist — U8 in, U8 out.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "EqualizeHist";
        bc.category = "statistical";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::equalizeHist(bufs.input, bufs.output);
        };
        bc.verify_fn = []() -> bool {
            // Ramp 0..255 across the image — equalised output should
            // remain a (mostly) ramp (CDF of a uniform distribution
            // is approximately the identity transform).
            cv::Mat in(1, 256, CV_8UC1);
            for (int i = 0; i < 256; ++i) in.at<uint8_t>(0, i) = static_cast<uint8_t>(i);
            cv::Mat out;
            cv::equalizeHist(in, out);
            // First pixel should map to 0, last to 255 (or very close).
            return out.at<uint8_t>(0, 0) == 0 && out.at<uint8_t>(0, 255) == 255;
        };
        cases.push_back(bc);
    }

    // MeanStdDev — U8 in, two scalar outs (mean, stddev).
    {
        OpenCVBenchmarkCase bc;
        bc.name = "MeanStdDev";
        bc.category = "statistical";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::Scalar mean_v, stddev_v;
            cv::meanStdDev(bufs.input, mean_v, stddev_v);
            // Touch the values to defeat dead-code elimination; cost
            // is negligible (two doubles) and ensures the call isn't
            // optimised away by an aggressive compiler.
            (void)mean_v[0];
            (void)stddev_v[0];
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Scalar mean_v, stddev_v;
            cv::meanStdDev(in, mean_v, stddev_v);
            return std::abs(mean_v[0] - 100.0) < 0.5 && stddev_v[0] < 0.5;
        };
        cases.push_back(bc);
    }

    // MinMaxLoc — U8 in, four scalar outs (min, max, minLoc, maxLoc).
    {
        OpenCVBenchmarkCase bc;
        bc.name = "MinMaxLoc";
        bc.category = "statistical";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            double min_v = 0.0, max_v = 0.0;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(bufs.input, &min_v, &max_v, &min_loc, &max_loc);
            (void)min_v; (void)max_v; (void)min_loc.x; (void)max_loc.x;
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            in.at<uint8_t>(10, 10) = 50;
            in.at<uint8_t>(20, 20) = 200;
            double mn, mx;
            cv::Point mnl, mxl;
            cv::minMaxLoc(in, &mn, &mx, &mnl, &mxl);
            return mn == 50 && mx == 200;
        };
        cases.push_back(bc);
    }

    // IntegralImage — U8 in, S32 out (one extra row/column per
    // OpenCV's convention; openvx-mark trims to W x H).
    {
        OpenCVBenchmarkCase bc;
        bc.name = "IntegralImage";
        bc.category = "statistical";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            // cv::integral wants (W+1) x (H+1) destination; allocate
            // outside timing so the per-iteration call doesn't grow it.
            bufs.output.create(static_cast<int>(h + 1), static_cast<int>(w + 1), CV_32SC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::integral(bufs.input, bufs.output, CV_32S);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(4, 4, CV_8UC1, cv::Scalar(1));
            cv::Mat out;
            cv::integral(in, out, CV_32S);
            // bottom-right of integral image == sum of all pixels = 16.
            return out.at<int32_t>(4, 4) == 16;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
