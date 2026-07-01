#include "opencv_test_data.h"
#include <cmath>
#include <cstring>

namespace opencv_mark {

OpenCVTestData::OpenCVTestData(uint64_t seed) : rng_(seed), seed_(seed) {}

void OpenCVTestData::reseed(uint64_t seed) { rng_.seed(seed); seed_ = seed; }

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

void OpenCVTestData::makeRemapIdentity(uint32_t src_w, uint32_t src_h,
                                       uint32_t dst_w, uint32_t dst_h,
                                       cv::Mat& mapX, cv::Mat& mapY) {
    mapX.create(static_cast<int>(dst_h), static_cast<int>(dst_w), CV_32FC1);
    mapY.create(static_cast<int>(dst_h), static_cast<int>(dst_w), CV_32FC1);
    const float sx = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float sy = static_cast<float>(src_h) / static_cast<float>(dst_h);
    for (int y = 0; y < mapX.rows; ++y) {
        auto* mx = mapX.ptr<float>(y);
        auto* my = mapY.ptr<float>(y);
        for (int x = 0; x < mapX.cols; ++x) {
            mx[x] = (x + 0.5f) * sx - 0.5f;
            my[x] = (y + 0.5f) * sy - 0.5f;
        }
    }
}

void OpenCVTestData::makeRemap(uint32_t src_w, uint32_t src_h,
                               uint32_t dst_w, uint32_t dst_h,
                               cv::Mat& mapX, cv::Mat& mapY,
                               RemapPattern pattern) {
    // Build the requested pattern. By default use a radial lens-distortion
    // model so the benchmark exercises scattered, realistic memory access
    // rather than the cache-friendly identity path.
    if (pattern == RemapPattern::IDENTITY) {
        makeRemapIdentity(src_w, src_h, dst_w, dst_h, mapX, mapY);
        return;
    }

    mapX.create(static_cast<int>(dst_h), static_cast<int>(dst_w), CV_32FC1);
    mapY.create(static_cast<int>(dst_h), static_cast<int>(dst_w), CV_32FC1);
    const float sx = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float sy = static_cast<float>(src_h) / static_cast<float>(dst_h);
    const float dst_wf = static_cast<float>(dst_w);
    const float dst_hf = static_cast<float>(dst_h);

    if (pattern == RemapPattern::LENS_DISTORTION) {
        const float cx = dst_wf * 0.5f;
        const float cy = dst_hf * 0.5f;
        const float max_radius = 0.5f * std::sqrt(dst_wf * dst_wf + dst_hf * dst_hf);
        const float inv_max_r2 = 1.0f / (max_radius * max_radius + 1e-6f);
        const float k1 = 0.08f;
        const float k2 = 0.01f;
        for (int y = 0; y < mapX.rows; ++y) {
            auto* mx = mapX.ptr<float>(y);
            auto* my = mapY.ptr<float>(y);
            for (int x = 0; x < mapX.cols; ++x) {
                const float xf = static_cast<float>(x);
                const float yf = static_cast<float>(y);
                const float dx = xf - cx;
                const float dy = yf - cy;
                const float r2 = (dx * dx + dy * dy) * inv_max_r2;
                const float r4 = r2 * r2;
                const float scale = 1.0f + k1 * r2 + k2 * r4;
                const float src_x = ((xf * sx) - cx) * scale + cx;
                const float src_y = ((yf * sy) - cy) * scale + cy;
                // cv::remap expects subpixel coordinates; (x+0.5)*scale-0.5
                // is the standard convention, but here src_x/src_y already
                // represent destination-to-source mapping coordinates and
                // are used directly to match openvx-mark's convention.
                mx[x] = src_x;
                my[x] = src_y;
            }
        }
    } else { // RANDOM_OFFSETS
        // Seed a dedicated, deterministic RNG for this pattern so offsets are
        // reproducible regardless of how many other benchmarks ran before
        // this one. Mix the original global seed with dimensions and a
        // pattern tag so the map is stable across run order.
        const uint64_t seed = seed_ + static_cast<uint64_t>(src_w) * 73856093u +
                              static_cast<uint64_t>(src_h) * 19349663u +
                              static_cast<uint64_t>(dst_w) * 83492791u +
                              static_cast<uint64_t>(dst_h) * 4256233u +
                              0x9e3779b97f4a7c15ULL;
        std::mt19937_64 pattern_rng(seed);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        const float src_wf = static_cast<float>(src_w);
        const float src_hf = static_cast<float>(src_h);
        for (int y = 0; y < mapX.rows; ++y) {
            auto* mx = mapX.ptr<float>(y);
            auto* my = mapY.ptr<float>(y);
            for (int x = 0; x < mapX.cols; ++x) {
                float rx = (x + 0.5f) * sx - 0.5f + dist(pattern_rng);
                float ry = (y + 0.5f) * sy - 0.5f + dist(pattern_rng);
                // Clamp to valid source bounds so border handling stays
                // consistent across implementations and runs.
                mx[x] = std::max(-0.5f, std::min(rx, src_wf - 0.5f));
                my[x] = std::max(-0.5f, std::min(ry, src_hf - 0.5f));
            }
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
