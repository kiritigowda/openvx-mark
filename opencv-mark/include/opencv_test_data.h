#ifndef OPENCV_MARK_OPENCV_TEST_DATA_H
#define OPENCV_MARK_OPENCV_TEST_DATA_H

// Deterministic synthetic image generator for opencv-mark — the cv::
// analogue of openvx-mark's TestDataGenerator. Same seed-driven RNG
// (mt19937_64) so a given (--seed N, width, height) produces an
// equivalent random fill across both binaries; this keeps cross-impl
// timing comparisons honest (same input distribution → same per-call
// work for the kernel).

#include <cstdint>
#include <opencv2/core.hpp>
#include <random>

namespace opencv_mark {

class OpenCVTestData {
public:
    explicit OpenCVTestData(uint64_t seed = 42);

    // Random U8 grayscale image (CV_8UC1). Filled uniformly in [0, 255].
    cv::Mat makeU8(uint32_t width, uint32_t height);

    // Random RGB image (CV_8UC3, channel order R-G-B to match
    // openvx-mark's vx_df_image=VX_DF_IMAGE_RGB convention).
    cv::Mat makeRGB(uint32_t width, uint32_t height);

    // 2x3 affine warp matrix matching what openvx-mark generates for
    // its warp_affine benchmark (slight rotation + translation), so
    // the per-pixel resampling work is comparable.
    cv::Mat makeAffineMatrix();

    void reseed(uint64_t seed);

private:
    std::mt19937_64 rng_;
};

} // namespace opencv_mark

#endif // OPENCV_MARK_OPENCV_TEST_DATA_H
