#include "opencv_verify.h"
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>

namespace opencv_mark {

double maxAbsDiff(const cv::Mat& a, const cv::Mat& b) {
    if (a.size() != b.size() || a.type() != b.type()) {
        return std::numeric_limits<double>::infinity();
    }
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    double minVal = 0.0, maxVal = 0.0;
    cv::minMaxLoc(diff.reshape(1), &minVal, &maxVal);
    return maxVal;
}

double psnrDb(const cv::Mat& a, const cv::Mat& b) {
    if (a.size() != b.size() || a.type() != b.type() || a.empty()) {
        return -1.0;
    }
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    const cv::Scalar s = cv::sum(diff.reshape(1));
    double sse = 0.0;
    for (int i = 0; i < a.channels(); ++i) sse += s[i];
    if (sse <= 1e-12) return std::numeric_limits<double>::infinity();
    const double mse = sse / static_cast<double>(a.total() * a.channels());
    // Assume 8-bit dynamic range (255) — all PR1 kernels operate on
    // CV_8U outputs; broaden if/when 16-bit kernels enter the suite.
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

bool matchesWithinCrossImplTolerance(const cv::Mat& actual, const cv::Mat& expected) {
    if (actual.size() != expected.size() || actual.type() != expected.type()) return false;
    if (maxAbsDiff(actual, expected) <= kCrossImplMaxAbsDiff) return true;
    return psnrDb(actual, expected) >= kCrossImplPsnrMinDb;
}

} // namespace opencv_mark
