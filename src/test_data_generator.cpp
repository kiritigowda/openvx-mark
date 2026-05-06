#include "test_data_generator.h"
#include <VX/vx_compatibility.h>
#include <cstring>
#include <vector>
#include <cmath>

TestDataGenerator::TestDataGenerator(uint64_t seed) : rng_(seed) {}

void TestDataGenerator::reseed(uint64_t seed) {
    rng_.seed(seed);
}

vx_image TestDataGenerator::createFilledImage(vx_context ctx, uint32_t width, uint32_t height,
                                              vx_df_image format) {
    vx_image img = vxCreateImage(ctx, width, height, format);
    if (vxGetStatus((vx_reference)img) != VX_SUCCESS) return img;
    fillImageRandom(img, width, height, format);
    return img;
}

void TestDataGenerator::fillImageRandom(vx_image image, uint32_t width, uint32_t height,
                                        vx_df_image format) {
    // Determine number of planes
    vx_size num_planes = 0;
    switch (format) {
        case VX_DF_IMAGE_U8:  num_planes = 1; break;
        case VX_DF_IMAGE_S16: num_planes = 1; break;
        case VX_DF_IMAGE_U16: num_planes = 1; break;
        case VX_DF_IMAGE_S32: num_planes = 1; break;
        case VX_DF_IMAGE_U32: num_planes = 1; break;
        case VX_DF_IMAGE_RGB: num_planes = 1; break;
        case VX_DF_IMAGE_RGBX: num_planes = 1; break;
        case VX_DF_IMAGE_IYUV: num_planes = 3; break;
        case VX_DF_IMAGE_NV12: num_planes = 2; break;
        case VX_DF_IMAGE_NV21: num_planes = 2; break;
        case VX_DF_IMAGE_YUYV: num_planes = 1; break;
        case VX_DF_IMAGE_UYVY: num_planes = 1; break;
        case VX_DF_IMAGE_YUV4: num_planes = 3; break;
        default: num_planes = 1; break;
    }

    std::uniform_int_distribution<int> dist(0, 255);

    for (vx_uint32 plane = 0; plane < num_planes; plane++) {
        // Get plane dimensions
        uint32_t pw = width, ph = height;
        if (format == VX_DF_IMAGE_IYUV && plane > 0) {
            pw = width / 2;
            ph = height / 2;
        } else if ((format == VX_DF_IMAGE_NV12 || format == VX_DF_IMAGE_NV21) && plane > 0) {
            pw = width;
            ph = height / 2;
        }

        // Determine bytes per pixel for this plane
        vx_size bpp = 1;
        if (plane == 0) {
            switch (format) {
                case VX_DF_IMAGE_U8:   bpp = 1; break;
                case VX_DF_IMAGE_S16:  bpp = 2; break;
                case VX_DF_IMAGE_U16:  bpp = 2; break;
                case VX_DF_IMAGE_S32:  bpp = 4; break;
                case VX_DF_IMAGE_U32:  bpp = 4; break;
                case VX_DF_IMAGE_RGB:  bpp = 3; break;
                case VX_DF_IMAGE_RGBX: bpp = 4; break;
                case VX_DF_IMAGE_YUYV: bpp = 2; break;
                case VX_DF_IMAGE_UYVY: bpp = 2; break;
                default: bpp = 1; break;
            }
        } else if (format == VX_DF_IMAGE_NV12 || format == VX_DF_IMAGE_NV21) {
            bpp = 2;  // interleaved UV
        }

        // Generate random data and copy via vxCopyImagePatch
        vx_rectangle_t rect = {0, 0, pw, ph};
        vx_imagepatch_addressing_t addr = {};
        addr.dim_x = pw;
        addr.dim_y = ph;
        addr.stride_x = static_cast<vx_int32>(bpp);
        addr.stride_y = static_cast<vx_int32>(pw * bpp);

        std::vector<uint8_t> data(pw * ph * bpp);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = static_cast<uint8_t>(dist(rng_));
        }

        vxCopyImagePatch(image, &rect, plane, &addr, data.data(),
                         VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    }
}

vx_tensor TestDataGenerator::createFilledTensor(vx_context ctx, const vx_size* dims,
                                                vx_size num_dims, vx_enum data_type) {
    vx_tensor tensor = vxCreateTensor(ctx, num_dims, dims, data_type, 0);
    if (vxGetStatus((vx_reference)tensor) != VX_SUCCESS) return tensor;

    // Compute total size
    vx_size total = 1;
    for (vx_size i = 0; i < num_dims; i++) total *= dims[i];

    // Fill based on data type
    vx_size starts[4] = {0, 0, 0, 0};
    vx_size strides[4];
    vx_size elem_size = 0;
    switch (data_type) {
        case VX_TYPE_INT8:  elem_size = 1; break;
        case VX_TYPE_UINT8: elem_size = 1; break;
        case VX_TYPE_INT16: elem_size = 2; break;
        case VX_TYPE_INT32: elem_size = 4; break;
        case VX_TYPE_FLOAT32: elem_size = 4; break;
        default: elem_size = 1; break;
    }
    strides[0] = elem_size;
    for (vx_size i = 1; i < num_dims; i++) strides[i] = strides[i - 1] * dims[i - 1];

    std::vector<uint8_t> data(total * elem_size);
    std::uniform_int_distribution<int> dist(0, 127);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = static_cast<uint8_t>(dist(rng_));
    }

    vxCopyTensorPatch(tensor, num_dims, starts, const_cast<vx_size*>(dims), strides,
                      data.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    return tensor;
}

vx_threshold TestDataGenerator::createBinaryThreshold(vx_context ctx, vx_int32 value) {
    vx_threshold thresh = vxCreateThresholdForImage(ctx, VX_THRESHOLD_TYPE_BINARY,
                                                    VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    vx_pixel_value_t pv = {};
    pv.U8 = static_cast<vx_uint8>(value);
    vxCopyThresholdValue(thresh, &pv, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    return thresh;
}

vx_threshold TestDataGenerator::createRangeThreshold(vx_context ctx, vx_int32 lower, vx_int32 upper) {
    vx_threshold thresh = vxCreateThresholdForImage(ctx, VX_THRESHOLD_TYPE_RANGE,
                                                    VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    vx_pixel_value_t lower_pv = {}, upper_pv = {};
    lower_pv.U8 = static_cast<vx_uint8>(lower);
    upper_pv.U8 = static_cast<vx_uint8>(upper);
    vxCopyThresholdRange(thresh, &lower_pv, &upper_pv, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    return thresh;
}

vx_matrix TestDataGenerator::createAffineMatrix(vx_context ctx) {
    vx_matrix mat = vxCreateMatrix(ctx, VX_TYPE_FLOAT32, 2, 3);
    // Identity with slight rotation
    vx_float32 data[6] = {
        0.9f, 0.1f, 10.0f,
        -0.1f, 0.9f, 5.0f
    };
    vxCopyMatrix(mat, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    return mat;
}

vx_matrix TestDataGenerator::createPerspectiveMatrix(vx_context ctx) {
    vx_matrix mat = vxCreateMatrix(ctx, VX_TYPE_FLOAT32, 3, 3);
    vx_float32 data[9] = {
        1.0f, 0.1f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0001f, 0.0001f, 1.0f
    };
    vxCopyMatrix(mat, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    return mat;
}

vx_remap TestDataGenerator::createRemap(vx_context ctx, uint32_t src_w, uint32_t src_h,
                                        uint32_t dst_w, uint32_t dst_h) {
    vx_remap remap = vxCreateRemap(ctx, src_w, src_h, dst_w, dst_h);

    // Build identity mapping data and copy via vxCopyRemapPatch
    vx_rectangle_t rect = {0, 0, dst_w, dst_h};
    vx_size stride = dst_w * sizeof(vx_coordinates2df_t);
    std::vector<vx_coordinates2df_t> coords(dst_w * dst_h);
    for (vx_uint32 y = 0; y < dst_h; y++) {
        for (vx_uint32 x = 0; x < dst_w; x++) {
            coords[y * dst_w + x].x = static_cast<vx_float32>(x * src_w) / static_cast<vx_float32>(dst_w);
            coords[y * dst_w + x].y = static_cast<vx_float32>(y * src_h) / static_cast<vx_float32>(dst_h);
        }
    }
    vxCopyRemapPatch(remap, &rect, stride, coords.data(),
                     VX_TYPE_COORDINATES2DF, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    return remap;
}

vx_convolution TestDataGenerator::createConvolution3x3(vx_context ctx) {
    vx_convolution conv = vxCreateConvolution(ctx, 3, 3);
    // Sharpening kernel
    vx_int16 data[9] = {
        0, -1, 0,
        -1,  5, -1,
        0, -1, 0
    };
    vxCopyConvolutionCoefficients(conv, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vx_uint32 scale = 1;
    vxSetConvolutionAttribute(conv, VX_CONVOLUTION_SCALE, &scale, sizeof(scale));
    return conv;
}

vx_lut TestDataGenerator::createLUT(vx_context ctx) {
    vx_lut lut = vxCreateLUT(ctx, VX_TYPE_UINT8, 256);
    // Gamma correction curve
    vx_uint8 data[256];
    for (int i = 0; i < 256; i++) {
        data[i] = static_cast<vx_uint8>(255.0 * std::pow(i / 255.0, 0.8));
    }
    vxCopyLUT(lut, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    return lut;
}

vx_distribution TestDataGenerator::createDistribution(vx_context ctx, vx_size num_bins,
                                                      vx_int32 offset, vx_uint32 range) {
    return vxCreateDistribution(ctx, num_bins, offset, range);
}

vx_pyramid TestDataGenerator::createPyramid(vx_context ctx, vx_size levels, vx_float32 scale,
                                            vx_uint32 width, vx_uint32 height, vx_df_image format) {
    return vxCreatePyramid(ctx, levels, scale, width, height, format);
}

vx_scalar TestDataGenerator::createScalar(vx_context ctx, vx_enum type, const void* value) {
    return vxCreateScalar(ctx, type, value);
}

vx_array TestDataGenerator::createKeypointArray(vx_context ctx, vx_size capacity) {
    return vxCreateArray(ctx, VX_TYPE_KEYPOINT, capacity);
}

vx_matrix TestDataGenerator::createNonLinearMask(vx_context ctx) {
    // 3x3 cross pattern mask for non-linear filter
    vx_matrix mask = vxCreateMatrix(ctx, VX_TYPE_UINT8, 3, 3);
    vx_uint8 data[9] = {
        0, 255, 0,
        255, 255, 255,
        0, 255, 0
    };
    vxCopyMatrix(mask, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    return mask;
}
