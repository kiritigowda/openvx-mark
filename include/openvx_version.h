////////////////////////////////////////////////////////////////////////////////
//
// MIT License
//
// Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef OPENVX_VERSION_H
#define OPENVX_VERSION_H

#include <VX/vx.h>

// VX_VERSION_1_X macros are only defined in headers of that version or later.
// Use their presence to detect the header version at compile time.
#ifdef VX_VERSION_1_1
#define OPENVX_HAS_1_1 1
#else
#define OPENVX_HAS_1_1 0
#endif

#ifdef VX_VERSION_1_2
#define OPENVX_HAS_1_2 1
#else
#define OPENVX_HAS_1_2 0
#endif

#ifdef VX_VERSION_1_3
#define OPENVX_HAS_1_3 1
#else
#define OPENVX_HAS_1_3 0
#endif

// Compatibility wrappers: OpenVX 1.3 APIs that map to 1.1 equivalents
#if !OPENVX_HAS_1_3

static inline vx_threshold vxCreateThresholdForImage(vx_context ctx,
    vx_enum thresh_type, vx_df_image /*in_fmt*/, vx_df_image /*out_fmt*/) {
    return vxCreateThreshold(ctx, thresh_type, VX_TYPE_UINT8);
}

static inline vx_status vxCopyThresholdValue(vx_threshold thresh,
    vx_pixel_value_t *value, vx_enum usage, vx_enum /*mem_type*/) {
    if (usage == VX_WRITE_ONLY) {
        vx_int32 v = value->U8;
        return vxSetThresholdAttribute(thresh,
            VX_THRESHOLD_THRESHOLD_VALUE, &v, sizeof(v));
    }
    return VX_ERROR_NOT_SUPPORTED;
}

static inline vx_status vxCopyThresholdRange(vx_threshold thresh,
    vx_pixel_value_t *lower, vx_pixel_value_t *upper,
    vx_enum usage, vx_enum /*mem_type*/) {
    if (usage == VX_WRITE_ONLY) {
        vx_int32 lo = lower->U8, hi = upper->U8;
        vx_status s = vxSetThresholdAttribute(thresh,
            VX_THRESHOLD_THRESHOLD_LOWER, &lo, sizeof(lo));
        if (s != VX_SUCCESS) return s;
        return vxSetThresholdAttribute(thresh,
            VX_THRESHOLD_THRESHOLD_UPPER, &hi, sizeof(hi));
    }
    return VX_ERROR_NOT_SUPPORTED;
}

// OpenVX 1.1 uses vxSetRemapPoint per pixel instead of vxCopyRemapPatch
#define OPENVX_USE_SET_REMAP_POINT 1

#else
#define OPENVX_USE_SET_REMAP_POINT 0
#endif

#endif // OPENVX_VERSION_H
