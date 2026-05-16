#ifndef OPENCV_MARK_OPENCV_CONTEXT_H
#define OPENCV_MARK_OPENCV_CONTEXT_H

// Light-weight wrapper around the OpenCV runtime metadata so the
// `opencv-mark` main loop can populate `SystemInfo` / `BenchmarkCatalog`
// in the same shape `openvx-mark` does — keeping the JSON schema
// identical across the two binaries so cross-comparison "just works"
// via scripts/compare_reports.py.
//
// OpenCV itself is stateless from our perspective (we never construct
// a cv::Context); this class exists purely so the main loop reads
// symmetrically with openvx-mark's BenchmarkContext usage.

#include <cstdint>
#include <string>

namespace opencv_mark {

class OpenCVContext {
public:
    OpenCVContext();

    // Mirrors openvx_mark::BenchmarkContext::implementation() — returns
    // a human-readable identifier such as "OpenCV 4.10.0".
    const std::string& implementation() const { return implementation_; }

    // Encoded version: (major << 8) | minor — same shape openvx-mark
    // emits for vx_version so downstream tooling reading the JSON
    // doesn't need to special-case OpenCV.
    uint16_t version() const { return version_encoded_; }

    // We have no analogue of a vendor ID in OpenCV. Reported as 0 and
    // documented in the JSON schema as "0 = OpenCV (no vendor)" so
    // diff tools don't trip over it.
    uint16_t vendorId() const { return 0; }

    // Number of benchmark kernels registered in this binary — used in
    // place of OpenVX's "number of available kernel enums" so the
    // SystemInfo.vx_num_kernels field carries a meaningful count when
    // the JSON is read back by compare_reports.py.
    uint32_t numKernels() const { return num_kernels_; }
    void setNumKernels(uint32_t n) { num_kernels_ = n; }

    // OpenCV build flag summary (e.g. "WITH_IPP, ENABLE_NEON, AVX2"),
    // surfaced into the JSON so users reading a comparison can see
    // which OpenCV configuration produced the baseline numbers.
    const std::string& buildOptions() const { return build_options_; }

private:
    std::string implementation_;
    std::string build_options_;
    uint16_t version_encoded_ = 0;
    uint32_t num_kernels_ = 0;
};

} // namespace opencv_mark

#endif // OPENCV_MARK_OPENCV_CONTEXT_H
