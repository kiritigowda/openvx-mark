#include "opencv_test_data.h"
#include <cmath>
#include <cstring>

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

cv::Mat OpenCVTestData::makeS16(uint32_t width, uint32_t height) {
    cv::Mat m(static_cast<int>(height), static_cast<int>(width), CV_16SC1);
    std::uniform_int_distribution<int> dist(-1024, 1024);
    auto* p = reinterpret_cast<int16_t*>(m.data);
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    for (size_t i = 0; i < n; ++i) p[i] = static_cast<int16_t>(dist(rng_));
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

cv::Mat OpenCVTestData::makePerspectiveMatrix() {
    // Small near-identity perspective transform. Same shape openvx-mark
    // generates for its warp_perspective benchmark — bottom row has a
    // non-zero perspective term so cv::warpPerspective doesn't degenerate
    // into an affine fast path.
    cv::Mat m(3, 3, CV_32FC1);
    m.at<float>(0, 0) = 1.001f; m.at<float>(0, 1) = 0.001f; m.at<float>(0, 2) = 0.0f;
    m.at<float>(1, 0) = 0.001f; m.at<float>(1, 1) = 1.001f; m.at<float>(1, 2) = 0.0f;
    m.at<float>(2, 0) = 0.00001f; m.at<float>(2, 1) = 0.00001f; m.at<float>(2, 2) = 1.0f;
    return m;
}

void OpenCVTestData::makeRemap(uint32_t src_w, uint32_t src_h,
                               uint32_t dst_w, uint32_t dst_h,
                               cv::Mat& mapX, cv::Mat& mapY) {
    // Identity map with a tiny per-pixel offset so the kernel does
    // real bilinear sampling instead of degenerate copy. Mirrors
    // openvx-mark's TestDataGenerator::createRemap behaviour.
    mapX.create(static_cast<int>(dst_h), static_cast<int>(dst_w), CV_32FC1);
    mapY.create(static_cast<int>(dst_h), static_cast<int>(dst_w), CV_32FC1);
    const float sx = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float sy = static_cast<float>(src_h) / static_cast<float>(dst_h);
    for (int y = 0; y < mapX.rows; ++y) {
        auto* mx = mapX.ptr<float>(y);
        auto* my = mapY.ptr<float>(y);
        for (int x = 0; x < mapX.cols; ++x) {
            mx[x] = (x + 0.5f) * sx + 0.25f;
            my[x] = (y + 0.5f) * sy + 0.25f;
        }
    }
}

cv::Mat OpenCVTestData::makeConvolution3x3() {
    // Sharpen-ish 3x3 kernel with non-trivial signed weights, matching
    // openvx-mark's TestDataGenerator::createConvolution3x3 weights so
    // per-pixel arithmetic cost is the same.
    cv::Mat k(3, 3, CV_16SC1);
    const int16_t weights[9] = { 0, -1,  0,
                                -1,  5, -1,
                                 0, -1,  0 };
    std::memcpy(k.data, weights, sizeof(weights));
    return k;
}

cv::Mat OpenCVTestData::makeLUT() {
    // Identity-with-noise LUT — every entry is `i XOR low_byte_of_rng`
    // so cv::LUT does the full table fetch per pixel.
    cv::Mat lut(1, 256, CV_8UC1);
    std::uniform_int_distribution<int> dist(0, 255);
    for (int i = 0; i < 256; ++i) {
        lut.at<uint8_t>(0, i) = static_cast<uint8_t>(i ^ (dist(rng_) & 0x0F));
    }
    return lut;
}

} // namespace opencv_mark
