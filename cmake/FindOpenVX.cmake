################################################################################
#
# MIT License
#
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
# FindOpenVX.cmake - Vendor-agnostic OpenVX discovery
#
# User can set:
#   OPENVX_INCLUDES  - Path to directory containing VX/vx.h
#   OPENVX_LIB_DIR   - Path to directory containing libopenvx and libvxu
#
# This module defines:
#   OpenVX_FOUND       - TRUE if OpenVX was found
#   OpenVX_INCLUDE_DIR - Include directory
#   OpenVX_LIBRARIES   - Libraries to link

# Search paths for includes
find_path(OpenVX_INCLUDE_DIR
    NAMES VX/vx.h
    HINTS
        ${OPENVX_INCLUDES}
        $ENV{OPENVX_DIR}/include
        $ENV{OPENVX_INCLUDES}
    PATHS
        /opt/rocm/include/mivisionx
        /opt/rocm/include
        /usr/local/include
        /usr/include
)

# Determine library search directory
if(OPENVX_LIB_DIR)
    set(_openvx_lib_search ${OPENVX_LIB_DIR})
else()
    set(_openvx_lib_search
        $ENV{OPENVX_DIR}/lib
        $ENV{OPENVX_LIB_DIR}
        /opt/rocm/lib
        /usr/local/lib
        /usr/lib
    )
endif()

find_library(OpenVX_openvx_LIBRARY
    NAMES openvx
    HINTS ${_openvx_lib_search}
)

find_library(OpenVX_vxu_LIBRARY
    NAMES vxu
    HINTS ${_openvx_lib_search}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenVX
    REQUIRED_VARS OpenVX_INCLUDE_DIR OpenVX_openvx_LIBRARY OpenVX_vxu_LIBRARY
)

if(OpenVX_FOUND)
    set(OpenVX_LIBRARIES ${OpenVX_openvx_LIBRARY} ${OpenVX_vxu_LIBRARY})
    mark_as_advanced(OpenVX_INCLUDE_DIR OpenVX_openvx_LIBRARY OpenVX_vxu_LIBRARY)
endif()
