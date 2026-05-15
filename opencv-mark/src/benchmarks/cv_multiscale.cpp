// OpenCV equivalents for the OpenVX `multiscale` category.
//
// PR2 set: GaussianPyramid, LaplacianPyramid, HalfScaleGaussian.
//
// Parameter mapping notes:
//
//   * Pyramid level count = 4 (DEFAULT_PYRAMID_LEVELS in
//     benchmark_config.h). cv::buildPyramid takes a maxlevel
//     parameter — we pass `levels - 1` because cv::buildPyramid
//     treats level 0 as the input image itself, matching openvx-mark's
//     vxGaussianPyramidNode which also reports the input as level 0.
//   * cv::pyrDown uses a fixed 5x5 Gaussian kernel and a halving
//     downsample, identical to vxHalfScaleGaussianNode and to one
//     level of vxGaussianPyramidNode.
//   * LaplacianPyramid is implemented by openvx-mark as a chain of
//     pyrDown → pyrUp → subtract per level. We implement the same
//     chain with cv:: calls so the per-level work is comparable.

#include "opencv_runner.h"
#include "benchmark_config.h"  // DEFAULT_PYRAMID_LEVELS
#include <opencv2/imgproc.hpp>
#include <vector>

namespace opencv_mark {

std::vector<OpenCVBenchmarkCase> registerCvMultiscaleBenchmarks() {
    std::vector<OpenCVBenchmarkCase> cases;

    // GaussianPyramid — U8 in, vector<cv::Mat> out (DEFAULT_PYRAMID_LEVELS).
    {
        OpenCVBenchmarkCase bc;
        bc.name = "GaussianPyramid";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            std::vector<cv::Mat> levels;
            cv::buildPyramid(bufs.input, levels, DEFAULT_PYRAMID_LEVELS - 1);
            // Touch to defeat DCE.
            (void)levels.size();
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            std::vector<cv::Mat> lv;
            cv::buildPyramid(in, lv, 3);
            // Centre pixel of every level should remain ~100 for a
            // uniform input.
            return lv.size() == 4 && lv[3].at<uint8_t>(4, 4) == 100;
        };
        cases.push_back(bc);
    }

    // LaplacianPyramid — U8 in, chain of pyrDown / pyrUp / subtract
    // per level. We reuse `output` as the working level0 buffer and
    // `output_extra` as the upsampled level1 buffer.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "LaplacianPyramid";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::Mat current = bufs.input;
            for (int i = 0; i < DEFAULT_PYRAMID_LEVELS - 1; ++i) {
                cv::Mat down, up, diff;
                cv::pyrDown(current, down);
                cv::pyrUp(down, up, current.size());
                cv::subtract(current, up, diff);
                current = down;
            }
            (void)current.cols;
        };
        bc.verify_fn = []() -> bool {
            // Just verify that the chain runs without throwing on a
            // small uniform input — exact Laplacian values depend on
            // the pyrDown/pyrUp Gaussian kernels and aren't worth
            // hard-coding here.
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat down, up, diff;
            cv::pyrDown(in, down);
            cv::pyrUp(down, up, in.size());
            cv::subtract(in, up, diff);
            return diff.cols == 64 && diff.rows == 64;
        };
        cases.push_back(bc);
    }

    // HalfScaleGaussian — U8 in (W x H), U8 out (W/2 x H/2), single
    // pyrDown call.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "HalfScaleGaussian";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h / 2), static_cast<int>(w / 2), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::pyrDown(bufs.input, bufs.output,
                        cv::Size(bufs.output.cols, bufs.output.rows));
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat o;
            cv::pyrDown(in, o);
            return o.at<uint8_t>(16, 16) == 100;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
