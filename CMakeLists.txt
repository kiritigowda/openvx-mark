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
cmake_minimum_required(VERSION 3.10)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

project(openvx-mark VERSION 1.0.0)

# Git commit SHA for version tracking
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE GIT_RESULT
)
if(NOT GIT_RESULT EQUAL 0)
    set(GIT_COMMIT_SHA "unknown")
endif()

# Add custom find module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Find OpenVX (vendor-agnostic)
if(OPENVX_INCLUDES AND OPENVX_LIB_DIR)
    # User provided explicit paths
    set(OpenVX_INCLUDE_DIR ${OPENVX_INCLUDES})
    find_library(OpenVX_openvx_LIBRARY NAMES openvx HINTS ${OPENVX_LIB_DIR})
    find_library(OpenVX_vxu_LIBRARY NAMES vxu HINTS ${OPENVX_LIB_DIR})
    if(OpenVX_openvx_LIBRARY AND OpenVX_vxu_LIBRARY)
        set(OpenVX_FOUND TRUE)
        set(OpenVX_LIBRARIES ${OpenVX_openvx_LIBRARY} ${OpenVX_vxu_LIBRARY})
    else()
        message(FATAL_ERROR "Could not find openvx/vxu libraries in ${OPENVX_LIB_DIR}")
    endif()
else()
    # Try FindOpenVX module with standard search paths
    find_package(OpenVX QUIET)
    if(NOT OpenVX_FOUND)
        # Fallback to ROCM_PATH defaults
        set(OpenVX_INCLUDE_DIR ${ROCM_PATH}/include/mivisionx)
        set(OpenVX_LIBRARIES openvx vxu)
        link_directories(${ROCM_PATH}/lib)
        message("-- OpenVX: Using default ROCM_PATH=${ROCM_PATH}")
    endif()
endif()

message("-- OpenVX Include: ${OpenVX_INCLUDE_DIR}")
message("-- OpenVX Libraries: ${OpenVX_LIBRARIES}")

# Collect sources
set(BENCHMARK_SOURCES
    src/main.cpp
    src/benchmark_context.cpp
    src/benchmark_runner.cpp
    src/benchmark_stats.cpp
    src/benchmark_report.cpp
    src/benchmark_timer.cpp
    src/kernel_registry.cpp
    src/test_data_generator.cpp
    src/system_info.cpp
    src/benchmarks/node_pixelwise.cpp
    src/benchmarks/node_filters.cpp
    src/benchmarks/node_color.cpp
    src/benchmarks/node_geometric.cpp
    src/benchmarks/node_statistical.cpp
    src/benchmarks/node_multiscale.cpp
    src/benchmarks/node_feature.cpp
    src/benchmarks/node_extraction.cpp
    src/benchmarks/node_tensor.cpp
    src/benchmarks/node_misc.cpp
    src/benchmarks/immediate_benchmarks.cpp
    src/benchmarks/pipeline_vision.cpp
    src/benchmarks/pipeline_feature.cpp
)

add_executable(openvx-mark ${BENCHMARK_SOURCES})
target_include_directories(openvx-mark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenVX_INCLUDE_DIR}
)
target_compile_definitions(openvx-mark PRIVATE
    OPENVX_MARK_VERSION="${PROJECT_VERSION}"
    GIT_COMMIT_SHA="${GIT_COMMIT_SHA}"
)
target_link_libraries(openvx-mark ${OpenVX_LIBRARIES})
