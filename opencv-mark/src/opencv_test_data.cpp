#include "opencv_test_data.h"
#include <cmath>

namespace opencv_mark {

OpenCVTestData::OpenCVTestData(uint64_t seed) : rng_(seed) {}

void OpenCVTestData::reseed(uint64_t seed) { rng_.seed(seed); }

cv::Mat OpenCVTestData::makeU8(uint32_t width, uint32_t height) {
    cv::Mat m(static_cast<int>(height), static_cast<int>(width), CV_8UC1);
    std::uniform_int_distribution<int> dist(0, 255);
    auto* p = m.data;
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    for (size_t i = 0; i < n; ++i) p[i] = static_cast<uint8_t>(dist(rng_));
    return m;
}

cv::Mat OpenCVTestData::makeRGB(uint32_t width, uint32_t height) {
    cv::Mat m(static_cast<int>(height), static_cast<int>(width), CV_8UC3);
    std::uniform_int_distribution<int> dist(0, 255);
    auto* p = m.data;
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    for (size_t i = 0; i < n; ++i) p[i] = static_cast<uint8_t>(dist(rng_));
    return m;
}

cv::Mat OpenCVTestData::makeAffineMatrix() {
    // Mirror openvx-mark's TestDataGenerator::createAffineMatrix —
    // ~5° rotation + small translation, deterministic so the same
    // per-pixel sampling work happens whether OpenVX or OpenCV runs it.
    const float angle = static_cast<float>(5.0 * M_PI / 180.0);
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    cv::Mat m(2, 3, CV_32FC1);
    m.at<float>(0, 0) = c;
    m.at<float>(0, 1) = -s;
    m.at<float>(0, 2) = 1.0f;  // tx
    m.at<float>(1, 0) = s;
    m.at<float>(1, 1) = c;
    m.at<float>(1, 2) = 1.0f;  // ty
    return m;
}

} // namespace opencv_mark
