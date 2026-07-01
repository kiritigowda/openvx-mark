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
#include <string>

namespace opencv_mark {

// Coordinate generation patterns for cv::remap. Keep in sync with
// openvx-mark's benchmark::RemapPattern.
enum class RemapPattern {
    IDENTITY,        // (x,y) -> (x,y): sequential, not representative
    LENS_DISTORTION, // radial distortion model (default for benchmarks)
    RANDOM_OFFSETS   // small random jitter around each pixel
};

// Parse a user-supplied remap pattern string. Returns LENS_DISTORTION for
// unknown values so benchmarks always run with a realistic default.
inline RemapPattern remapPatternFromString(const std::string& s) {
    if (s == "identity") return RemapPattern::IDENTITY;
    if (s == "random_offsets") return RemapPattern::RANDOM_OFFSETS;
    return RemapPattern::LENS_DISTORTION;
}

class OpenCVTestData {
public:
    explicit OpenCVTestData(uint64_t seed = 42);

    // Random U8 grayscale image (CV_8UC1). Filled uniformly in [0, 255].
    cv::Mat makeU8(uint32_t width, uint32_t height);

    // Random S16 image (CV_16SC1). Filled uniformly in [-1024, 1024].
    // Used by Magnitude / Phase / Add / Subtract benchmarks that take
    // signed 16-bit inputs (typically the output of a Sobel stage).
    cv::Mat makeS16(uint32_t width, uint32_t height);

    // Random RGB image (CV_8UC3, channel order R-G-B to match
    // openvx-mark's vx_df_image=VX_DF_IMAGE_RGB convention).
    cv::Mat makeRGB(uint32_t width, uint32_t height);

    // 2x3 affine warp matrix matching what openvx-mark generates for
    // its warp_affine benchmark (slight rotation + translation), so
    // the per-pixel resampling work is comparable.
    cv::Mat makeAffineMatrix();

    // 3x3 perspective warp matrix — small homography close to identity
    // so cv::warpPerspective does meaningful work without producing a
    // mostly-empty output. Mirrors openvx-mark's perspective matrix
    // generator.
    cv::Mat makePerspectiveMatrix();

    // Two CV_32FC1 maps (mapX, mapY) for cv::remap, sized
    // (dst_w, dst_h). Default pattern is LENS_DISTORTION so the
    // benchmark exercises scattered, realistic memory access; pass
    // IDENTITY to reproduce the old cache-friendly behaviour.
    void makeRemap(uint32_t src_w, uint32_t src_h,
                   uint32_t dst_w, uint32_t dst_h,
                   cv::Mat& mapX, cv::Mat& mapY,
                   RemapPattern pattern = RemapPattern::LENS_DISTORTION);

    // Identity remap maps for deterministic correctness checks.
    static void makeRemapIdentity(uint32_t src_w, uint32_t src_h,
                                  uint32_t dst_w, uint32_t dst_h,
                                  cv::Mat& mapX, cv::Mat& mapY);

    // 3x3 CV_16SC1 convolution kernel for cv::filter2D — same
    // signed-int weights openvx-mark uses for its CustomConvolution
    // benchmark so per-pixel arithmetic is the same.
    cv::Mat makeConvolution3x3();

    // 256-entry CV_8UC1 lookup table for cv::LUT.
    cv::Mat makeLUT();

    void reseed(uint64_t seed);

private:
    std::mt19937_64 rng_;
    uint64_t seed_;
};

} // namespace opencv_mark

#endif // OPENCV_MARK_OPENCV_TEST_DATA_H
