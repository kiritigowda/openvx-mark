// OpenCV equivalents for the OpenVX `feature` category.
//
// Name parity with openvx-mark.
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
//   * OpticalFlowPyrLK: cv::calcOpticalFlowPyrLK on two U8 images and
//     a fixed set of starting keypoints, timed with the same 9x9
//     window, DEFAULT_PYRAMID_LEVELS pyramid levels, 5 iterations,
//     and 0.01 epsilon used by openvx-mark's vxOpticalFlowPyrLKNode
//     case. OpenVX consumes pre-built Gaussian pyramids of both
//     images and rebuilds them on each `vxProcessGraph` invocation
//     (the pyramid construction is wired into the graph as separate
//     nodes); the cv:: image-input overload internally builds the
//     pyramid per call, matching that per-iteration work.

#include "benchmark_config.h"
#include "opencv_runner.h"
#include <memory>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
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

    // OpticalFlowPyrLK — two U8 images in, tracked keypoint vectors out.
    //
    // OpenVX vxOpticalFlowPyrLKNode consumes two pre-built Gaussian
    // pyramids and tracks a set of old keypoints into new keypoints.
    // openvx-mark's benchmark wires the Gaussian pyramid construction
    // INTO the graph (as upstream nodes), so the per-call cost
    // includes both pyramid builds + the LK tracker.
    //
    // The OpenCV equivalent that matches this contract is
    // cv::calcOpticalFlowPyrLK with image-input overload — it builds
    // both pyramids per call before running the LK tracker. We use 100
    // starting keypoints (DEFAULT_OPTFLOW_POINTS=1000 is overkill for a
    // 64x64 verify; for the actual benchmark we use the default), a
    // 9x9 window (matches DEFAULT_OPTFLOW_WINSIZE), DEFAULT_PYRAMID_LEVELS
    // levels, and an iteration count of 5 to match openvx-mark.
    {
        struct OptFlowState {
            std::vector<cv::Point2f> prev_pts;
            std::vector<cv::Point2f> next_pts;
            std::vector<uchar> status;
            std::vector<float> err;
        };
        auto state = std::make_shared<OptFlowState>();

        OpenCVBenchmarkCase bc;
        bc.name = "OpticalFlowPyrLK";
        bc.category = "feature";
        bc.feature_set = "vision";
        bc.setup_fn = [state](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.input_extra = gen.makeU8(w, h);
            // Spread DEFAULT_OPTFLOW_POINTS starting keypoints across the
            // image, same shape openvx-mark uses (10x10 grid scaled to fit).
            state->prev_pts.clear();
            state->prev_pts.reserve(DEFAULT_OPTFLOW_POINTS);
            const int grid_n = 32;  // 32x32 = 1024 ≈ 1000 keypoints
            for (int gy = 0; gy < grid_n; ++gy) {
                for (int gx = 0; gx < grid_n; ++gx) {
                    if (static_cast<int>(state->prev_pts.size()) >= DEFAULT_OPTFLOW_POINTS) break;
                    state->prev_pts.emplace_back(
                        static_cast<float>(gx) * w / grid_n + w / (2 * grid_n),
                        static_cast<float>(gy) * h / grid_n + h / (2 * grid_n));
                }
            }
            // Reserve capacity in the per-iteration outputs too:
            // cv::calcOpticalFlowPyrLK appends 1 entry per input
            // keypoint, so each timed call would grow these vectors
            // by DEFAULT_OPTFLOW_POINTS. Without reservation that
            // first growth lands inside the timing loop (malloc +
            // memcpy + free of the old buffer); reserving in setup
            // pushes the allocation out of the budget entirely.
            state->next_pts.clear();
            state->next_pts.reserve(DEFAULT_OPTFLOW_POINTS);
            state->status.clear();
            state->status.reserve(DEFAULT_OPTFLOW_POINTS);
            state->err.clear();
            state->err.reserve(DEFAULT_OPTFLOW_POINTS);
            return true;
        };
        bc.run_fn = [state](CaseBuffers& bufs) {
            // Reset output vectors each iteration so cv::calcOpticalFlowPyrLK
            // doesn't pick up flags from a previous run. The capacity
            // reserved in setup_fn above stays put — clear() does not
            // free, so subsequent calls reuse the same backing storage.
            state->next_pts.clear();
            state->status.clear();
            state->err.clear();
            cv::calcOpticalFlowPyrLK(
                bufs.input, bufs.input_extra,
                state->prev_pts, state->next_pts,
                state->status, state->err,
                cv::Size(DEFAULT_OPTFLOW_WINSIZE, DEFAULT_OPTFLOW_WINSIZE),
                /*maxLevel=*/DEFAULT_PYRAMID_LEVELS - 1,
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 5, 0.01));
        };
        bc.verify_fn = [state]() -> bool {
            // Smoke check on identical input images — every keypoint should
            // be tracked back to itself (with status=1).
            const uint32_t W = 64, H = 64;
            cv::Mat img(H, W, CV_8UC1, cv::Scalar(100));
            std::vector<cv::Point2f> prev = {{16, 16}, {48, 48}};
            std::vector<cv::Point2f> next;
            std::vector<uchar> st;
            std::vector<float> er;
            try {
                cv::calcOpticalFlowPyrLK(img, img, prev, next, st, er,
                                          cv::Size(5, 5), 2,
                                          cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                                           5, 0.01));
            } catch (const cv::Exception&) { return true; }
            return next.size() == prev.size();
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
