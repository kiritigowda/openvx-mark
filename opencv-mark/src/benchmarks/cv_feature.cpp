// OpenCV equivalents for the OpenVX `feature` category.
//
// PR2 set: CannyEdgeDetector, HarrisCorners, FastCorners.
//
// Parameter mapping notes:
//
//   * Canny: low=80, high=160, ksize=3 (sobel aperture), L2gradient=false.
//     OpenVX vxCannyEdgeDetectorNode takes the same parameters.
//   * HarrisCorners: cv::cornerHarris produces a CV_32F response map
//     of the same size as input. OpenVX vxHarrisCornersNode produces a
//     keypoint list — different output shape but the per-pixel cost of
//     the Harris response computation (the dominant cost) is the same.
//     The non-max-suppression + thresholding stage that converts the
//     response map to keypoints in OpenVX is structurally separate;
//     timing the cv::cornerHarris call alone gives the most apples-to-
//     apples per-pixel comparison.
//     Parameters: blockSize=2, ksize=3, k=0.04 — standard defaults
//     matching openvx-mark's HarrisCorners benchmark.
//   * FastCorners: cv::FAST returns a vector<cv::KeyPoint>. The
//     dominant cost is the per-pixel detector loop; output extraction
//     is sub-dominant. We pass nonmaxSuppression=true to match
//     openvx-mark's vxFastCornersNode default.

#include "opencv_runner.h"
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace opencv_mark {

std::vector<OpenCVBenchmarkCase> registerCvFeatureBenchmarks() {
    std::vector<OpenCVBenchmarkCase> cases;

    // CannyEdgeDetector — U8 in, U8 out (binary edge map).
    {
        OpenCVBenchmarkCase bc;
        bc.name = "CannyEdgeDetector";
        bc.category = "feature";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::Canny(bufs.input, bufs.output, /*threshold1=*/80, /*threshold2=*/160,
                      /*apertureSize=*/3, /*L2gradient=*/false);
        };
        bc.verify_fn = []() -> bool {
            // Step image: half black, half white. Expect at least
            // one strong edge along the boundary.
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(0));
            in(cv::Rect(32, 0, 32, 64)).setTo(cv::Scalar(255));
            cv::Mat o;
            cv::Canny(in, o, 80, 160, 3, false);
            return cv::countNonZero(o.col(31)) + cv::countNonZero(o.col(32)) > 0;
        };
        cases.push_back(bc);
    }

    // HarrisCorners — U8 in, F32 response map out.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "HarrisCorners";
        bc.category = "feature";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_32FC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::cornerHarris(bufs.input, bufs.output,
                             /*blockSize=*/2, /*ksize=*/3, /*k=*/0.04,
                             cv::BORDER_REPLICATE);
        };
        bc.verify_fn = []() -> bool {
            // L-shape image — should produce a non-zero response at
            // the corner.
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(0));
            in(cv::Rect(0, 0, 32, 32)).setTo(cv::Scalar(255));
            cv::Mat o;
            cv::cornerHarris(in, o, 2, 3, 0.04, cv::BORDER_REPLICATE);
            // Some pixel near the corner should have a noticeable
            // response.
            double mn, mx;
            cv::minMaxLoc(o, &mn, &mx);
            return mx > 0.0;
        };
        cases.push_back(bc);
    }

    // FastCorners — U8 in, vector<KeyPoint> out (not stored — only
    // the detector's per-pixel cost is being timed).
    {
        OpenCVBenchmarkCase bc;
        bc.name = "FastCorners";
        bc.category = "feature";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            std::vector<cv::KeyPoint> kp;
            cv::FAST(bufs.input, kp, /*threshold=*/30, /*nonmaxSuppression=*/true);
            (void)kp.size();
        };
        bc.verify_fn = []() -> bool {
            // Random uniform noise input — natural variation produces
            // plenty of FAST keypoints. Lower threshold than the run
            // configuration to make this test robust across OpenCV
            // versions that may apply slightly different non-max
            // suppression rules.
            cv::Mat in(64, 64, CV_8UC1);
            cv::randu(in, cv::Scalar(0), cv::Scalar(255));
            std::vector<cv::KeyPoint> kp;
            cv::FAST(in, kp, /*threshold=*/10, /*nonmaxSuppression=*/false);
            return !kp.empty();
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
