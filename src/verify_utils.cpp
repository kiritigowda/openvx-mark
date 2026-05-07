#include "verify_utils.h"
#include <cstring>
#include <cmath>

namespace verify {

vx_image createImage(vx_context ctx, uint32_t w, uint32_t h,
                     vx_df_image format, const uint8_t* data) {
    vx_image img = vxCreateImage(ctx, w, h, format);
    if (vxGetStatus((vx_reference)img) != VX_SUCCESS) return img;

    vx_rectangle_t rect = {0, 0, w, h};
    vx_imagepatch_addressing_t addr = {};
    void* ptr = nullptr;
    vx_map_id map_id;

    if (vxMapImagePatch(img, &rect, 0, &map_id, &addr, &ptr,
                        VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0) == VX_SUCCESS) {
        uint32_t stride = addr.stride_y;
        uint32_t row_bytes = w * addr.stride_x;
        for (uint32_t y = 0; y < h; y++) {
            std::memcpy(static_cast<uint8_t*>(ptr) + y * stride,
                       data + y * row_bytes, row_bytes);
        }
        vxUnmapImagePatch(img, map_id);
    }
    return img;
}

std::vector<uint8_t> readImage(vx_image img, uint32_t w, uint32_t h) {
    std::vector<uint8_t> result(w * h, 0);
    vx_rectangle_t rect = {0, 0, w, h};
    vx_imagepatch_addressing_t addr = {};
    void* ptr = nullptr;
    vx_map_id map_id;

    if (vxMapImagePatch(img, &rect, 0, &map_id, &addr, &ptr,
                        VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0) == VX_SUCCESS) {
        uint32_t stride = addr.stride_y;
        for (uint32_t y = 0; y < h; y++) {
            std::memcpy(result.data() + y * w,
                       static_cast<uint8_t*>(ptr) + y * stride, w);
        }
        vxUnmapImagePatch(img, map_id);
    }
    return result;
}

std::vector<int16_t> readImageS16(vx_image img, uint32_t w, uint32_t h) {
    std::vector<int16_t> result(w * h, 0);
    vx_rectangle_t rect = {0, 0, w, h};
    vx_imagepatch_addressing_t addr = {};
    void* ptr = nullptr;
    vx_map_id map_id;

    if (vxMapImagePatch(img, &rect, 0, &map_id, &addr, &ptr,
                        VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0) == VX_SUCCESS) {
        uint32_t stride = addr.stride_y;
        for (uint32_t y = 0; y < h; y++) {
            std::memcpy(result.data() + y * w,
                       static_cast<uint8_t*>(ptr) + y * stride, w * sizeof(int16_t));
        }
        vxUnmapImagePatch(img, map_id);
    }
    return result;
}

bool compareU8(const std::vector<uint8_t>& actual,
               const std::vector<uint8_t>& expected, int tolerance) {
    if (actual.size() != expected.size()) return false;
    for (size_t i = 0; i < actual.size(); i++) {
        if (std::abs(static_cast<int>(actual[i]) - static_cast<int>(expected[i])) > tolerance)
            return false;
    }
    return true;
}

bool compareS16(const std::vector<int16_t>& actual,
                const std::vector<int16_t>& expected, int tolerance) {
    if (actual.size() != expected.size()) return false;
    for (size_t i = 0; i < actual.size(); i++) {
        if (std::abs(static_cast<int>(actual[i]) - static_cast<int>(expected[i])) > tolerance)
            return false;
    }
    return true;
}

bool imageNonZero(vx_image img, uint32_t w, uint32_t h) {
    auto data = readImage(img, w, h);
    for (auto v : data) {
        if (v != 0) return true;
    }
    return false;
}

} // namespace verify
