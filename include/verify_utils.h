#ifndef VERIFY_UTILS_H
#define VERIFY_UTILS_H

#include <VX/vx.h>
#include <cstdint>
#include <vector>

namespace verify {

vx_image createImage(vx_context ctx, uint32_t w, uint32_t h,
                     vx_df_image format, const uint8_t* data);

std::vector<uint8_t> readImage(vx_image img, uint32_t w, uint32_t h);

std::vector<int16_t> readImageS16(vx_image img, uint32_t w, uint32_t h);

bool compareU8(const std::vector<uint8_t>& actual,
               const std::vector<uint8_t>& expected, int tolerance = 0);

bool compareS16(const std::vector<int16_t>& actual,
                const std::vector<int16_t>& expected, int tolerance = 0);

bool imageNonZero(vx_image img, uint32_t w, uint32_t h);

} // namespace verify

#endif // VERIFY_UTILS_H
