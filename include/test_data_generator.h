#ifndef TEST_DATA_GENERATOR_H
#define TEST_DATA_GENERATOR_H

#include "openvx_version.h"
#include <cstdint>
#include <random>

class TestDataGenerator {
public:
    explicit TestDataGenerator(uint64_t seed = 42);

    // Image creation + fill
    vx_image createFilledImage(vx_context ctx, uint32_t width, uint32_t height, vx_df_image format);
    void fillImageRandom(vx_image image, uint32_t width, uint32_t height, vx_df_image format);

    // Tensor creation + fill (OpenVX 1.2+)
#if OPENVX_HAS_1_2
    vx_tensor createFilledTensor(vx_context ctx, const vx_size* dims, vx_size num_dims,
                                 vx_enum data_type);
#endif

    // Auxiliary objects
    vx_threshold createBinaryThreshold(vx_context ctx, vx_int32 value);
    vx_threshold createRangeThreshold(vx_context ctx, vx_int32 lower, vx_int32 upper);
    vx_matrix createAffineMatrix(vx_context ctx);
    vx_matrix createPerspectiveMatrix(vx_context ctx);
    vx_remap createRemap(vx_context ctx, uint32_t src_w, uint32_t src_h,
                         uint32_t dst_w, uint32_t dst_h);
    vx_convolution createConvolution3x3(vx_context ctx);
    vx_lut createLUT(vx_context ctx);
    vx_distribution createDistribution(vx_context ctx, vx_size num_bins,
                                       vx_int32 offset, vx_uint32 range);
    vx_pyramid createPyramid(vx_context ctx, vx_size levels, vx_float32 scale,
                             vx_uint32 width, vx_uint32 height, vx_df_image format);
    vx_scalar createScalar(vx_context ctx, vx_enum type, const void* value);
    vx_array createKeypointArray(vx_context ctx, vx_size capacity);
    vx_matrix createNonLinearMask(vx_context ctx);

    void reseed(uint64_t seed);

private:
    std::mt19937_64 rng_;
};

#endif // TEST_DATA_GENERATOR_H
