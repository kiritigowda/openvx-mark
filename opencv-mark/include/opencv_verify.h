#ifndef OPENCV_MARK_OPENCV_VERIFY_H
#define OPENCV_MARK_OPENCV_VERIFY_H

// Output verification helpers for opencv-mark.
//
// Two flavours of check, sized to two distinct purposes:
//
// 1. **Self-verification (PR #1 scope).** A given cv:: kernel is run
//    on a small fixed input and the output is compared against a
//    hand-computed reference array hard-coded in the benchmark file.
//    Used by every benchmark's `verify_fn` to catch a broken cv::
//    call (wrong borders, swapped channels, etc.) before any
//    timing happens. Threshold is exact (max-diff = 0) for
//    deterministic kernels and `maxAbsDiff <= tolerance` for
//    interpolating kernels (warps, resizes).
//
// 2. **Cross-implementation tolerance check (used by future PRs).**
//    PSNR and max-abs-diff helpers are exposed here so a follow-up
//    "compare-images" tool (or a future opencv-mark / openvx-mark
//    cross-verification phase) can decide whether two implementations
//    of the same kernel produced "equivalent enough" output, given
//    that OpenVX and OpenCV legitimately differ on border modes,
//    rounding, and integer kernel weight quantisation. The chosen
//    bounds — PSNR >= 30 dB OR max-abs-diff <= 5 grey levels — were
//    picked per the project's verification policy as "egregious
//    mismatch detector, not a bit-exact compliance test".

#include <cstdint>
#include <opencv2/core.hpp>

namespace opencv_mark {

// Maximum absolute pixel difference between `a` and `b`. Inputs must
// have identical shape and type (CV_8U or CV_16S supported in PR1).
double maxAbsDiff(const cv::Mat& a, const cv::Mat& b);

// Peak-signal-to-noise ratio in dB. Returns +inf for bit-exact match.
// Used by the cross-impl tolerance check.
double psnrDb(const cv::Mat& a, const cv::Mat& b);

// Project-wide cross-impl tolerance thresholds (see header rationale).
constexpr double kCrossImplPsnrMinDb = 30.0;
constexpr double kCrossImplMaxAbsDiff = 5.0;

// `true` iff `actual` matches `expected` within the project's
// cross-impl tolerance — PSNR >= 30 dB OR max-abs-diff <= 5.
bool matchesWithinCrossImplTolerance(const cv::Mat& actual, const cv::Mat& expected);

} // namespace opencv_mark

#endif // OPENCV_MARK_OPENCV_VERIFY_H
