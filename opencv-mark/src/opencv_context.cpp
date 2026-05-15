#include "opencv_context.h"
#include <opencv2/core.hpp>
#include <regex>
#include <sstream>

namespace opencv_mark {

namespace {
// Cherry-pick a small set of cv::getBuildInformation() flags worth
// surfacing into the JSON so users can tell whether the baseline they
// are comparing against was built with IPP / SIMD / parallelism. We
// deliberately avoid dumping the full multi-kilobyte build info
// string into the JSON because it would dominate the artifact size
// and rarely changes between runs on a given host.
std::string extractBuildSummary() {
    const std::string info = cv::getBuildInformation();
    std::vector<std::string> hits;

    // Each `(label, regex)` matches a `Label: value` pair we care about.
    const std::pair<std::string, std::regex> patterns[] = {
        {"IPP",         std::regex(R"(\n\s*Use IPP:\s+([^\n]+))")},
        {"TBB",         std::regex(R"(\n\s*Use TBB:\s+([^\n]+))")},
        {"Parallel",    std::regex(R"(\n\s*Parallel framework:\s+([^\n]+))")},
        {"NEON",        std::regex(R"(\n\s*NEON:\s+([^\n]+))")},
        {"AVX2",        std::regex(R"(\n\s*AVX2:\s+([^\n]+))")},
        {"AVX512",      std::regex(R"(\n\s*AVX512[^:]*:\s+([^\n]+))")},
    };
    for (const auto& [label, re] : patterns) {
        std::smatch m;
        if (std::regex_search(info, m, re) && m.size() > 1) {
            std::string val = m[1].str();
            // Trim trailing whitespace.
            while (!val.empty() && std::isspace(static_cast<unsigned char>(val.back()))) {
                val.pop_back();
            }
            if (!val.empty() && val != "NO" && val.find("NO ") != 0) {
                hits.push_back(label + "=" + val);
            }
        }
    }
    if (hits.empty()) return "default";

    std::ostringstream out;
    for (size_t i = 0; i < hits.size(); ++i) {
        if (i) out << ", ";
        out << hits[i];
    }
    return out.str();
}
}  // namespace

OpenCVContext::OpenCVContext() {
    std::ostringstream impl;
    impl << "OpenCV " << CV_VERSION;
    implementation_ = impl.str();

    // Pack OpenCV's (major, minor) into the same 16-bit shape that
    // openvx-mark uses for vx_version: high byte = major, low byte = minor.
    version_encoded_ = static_cast<uint16_t>((CV_MAJOR_VERSION << 8) | (CV_MINOR_VERSION & 0xFF));

    build_options_ = extractBuildSummary();
}

} // namespace opencv_mark
