################################################################################
#
# MIT License
#
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc.
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
#
# opencv-mark — OpenCV baseline companion binary for openvx-mark.
#
# Mirrors the openvx-mark CLI, kernel naming, and JSON schema so a user
# can run the same benchmark suite against an OpenCV implementation and
# feed the resulting JSON into the existing scripts/compare_reports.py
# (or the in-tool `--compare` flag) for side-by-side comparison with any
# OpenVX implementation. See ../README.md for the broader context.
#
# OpenCV is treated as the de facto vision baseline. The point is to
# answer "does adopting OpenVX actually pay off vs the cv:: code I
# already have?" at both the per-kernel level and (in future PRs) the
# pipeline / graph-equivalent level.
#
# This binary depends on:
#   * OpenCV 4 (core + imgproc — no GPU/CUDA/OpenCL requirements)
#   * bench_core static library (built from the parent project — provides
#     BenchmarkResult, BenchmarkStats, BenchmarkTimer, BenchmarkReport,
#     SystemInfo — i.e. the implementation-agnostic infrastructure that
#     makes the JSON schema match openvx-mark byte-for-byte)
#
################################################################################

find_package(OpenCV 4 QUIET COMPONENTS core imgproc)
if(NOT OpenCV_FOUND)
    message(STATUS "opencv-mark: OpenCV 4 not found — skipping companion binary build.")
    message(STATUS "             Install opencv4 (e.g. apt install libopencv-dev / brew install opencv) to enable.")
    return()
endif()

message(STATUS "opencv-mark: OpenCV ${OpenCV_VERSION} (${OpenCV_INCLUDE_DIRS})")

set(OPENCV_MARK_SOURCES
    src/main.cpp
    src/opencv_context.cpp
    src/opencv_runner.cpp
    src/opencv_test_data.cpp
    src/opencv_verify.cpp
    src/benchmarks/cv_filters.cpp
    src/benchmarks/cv_color.cpp
    src/benchmarks/cv_geometric.cpp
)

add_executable(opencv-mark ${OPENCV_MARK_SOURCES})
target_include_directories(opencv-mark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_compile_definitions(opencv-mark PRIVATE
    OPENCV_MARK_VERSION="${PROJECT_VERSION}"
    GIT_COMMIT_SHA="${GIT_COMMIT_SHA}"
    OPENCV_VERSION_STRING="${OpenCV_VERSION}"
)
target_link_libraries(opencv-mark PRIVATE
    bench_core
    ${OpenCV_LIBS}
)
