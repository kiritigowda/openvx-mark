// cv_output_dumper — OpenCV-side counterpart of
// src/openvx_output_dumper.cpp. Dumps the raw pixel bytes of the SAME
// sentinel kernel set on the OpenCV path, using the SAME deterministic
// input (same mt19937_64 seed, same fill function) so the produced
// .bin files are byte-comparable to the OpenVX-side dumps.
//
// Matching kernel ↔ kernel mapping (kept in lock-step with
// openvx_output_dumper.cpp — any new kernel added there MUST get an
// equivalent here or scripts/cross_verify_outputs.py will skip it):
//
//   Box3x3                  ↔  cv::boxFilter(ksize=3, normalize=true)
//   Gaussian3x3             ↔  cv::GaussianBlur(ksize=3, sigma=0)
//   Median3x3               ↔  cv::medianBlur(ksize=3)
//   Erode3x3                ↔  cv::erode(3x3 rect)
//   Dilate3x3               ↔  cv::dilate(3x3 rect)
//   Sobel3x3_dx/_dy         ↔  cv::Sobel(ksize=3, ddepth=CV_16S)
//   Add_U8_Saturate         ↔  cv::add (saturate by default for U8)
//   ColorConvert_RGB2Gray   ↔  cv::cvtColor(COLOR_RGB2GRAY) — BT.601
//
// The verifier (scripts/cross_verify_outputs.py) is the authoritative
// place where tolerances live, since the tolerances depend on the
// PAIR of impls being compared (e.g. MIVisionX vs OpenCV may
// disagree on Sobel border handling by 1 LSB even though both are
// "right" per their own spec).

#include "opencv_test_data.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace opencv_mark {

namespace {

struct DumpRecord {
    std::string name;
    std::string file;
    uint32_t width = 0;
    uint32_t height = 0;
    int channels = 0;
    std::string dtype;
    size_t byte_size = 0;
};

void ensureDir(const std::string& path) {
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) {
        mkdir(path.c_str(), 0755);
    }
}

const char* dtypeFromMat(const cv::Mat& m) {
    switch (m.depth()) {
        case CV_8U:  return "u8";
        case CV_8S:  return "s8";
        case CV_16U: return "u16";
        case CV_16S: return "s16";
        case CV_32S: return "s32";
        case CV_32F: return "f32";
        default:     return "u8";
    }
}

// Write a Mat as a tightly-packed h × w × channels buffer (no stride
// padding), matching the layout openvx_output_dumper.cpp emits so the
// verifier can mmap both sides as numpy arrays without per-side
// special-casing.
bool dumpMat(const cv::Mat& m, const std::string& dir,
             const std::string& name, DumpRecord& rec) {
    if (!m.data) {
        std::fprintf(stderr, "  cv dump %s: empty mat\n", name.c_str());
        return false;
    }
    const size_t elem_size = m.elemSize();
    const size_t row_bytes = static_cast<size_t>(m.cols) * elem_size;
    std::vector<uint8_t> tight(static_cast<size_t>(m.rows) * row_bytes);
    for (int y = 0; y < m.rows; ++y) {
        std::memcpy(tight.data() + y * row_bytes,
                    m.ptr<uint8_t>(y), row_bytes);
    }
    std::string path = dir + "/" + name + ".bin";
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "  cv dump %s: cannot open %s\n",
                     name.c_str(), path.c_str());
        return false;
    }
    f.write(reinterpret_cast<const char*>(tight.data()), tight.size());

    rec.name = name;
    rec.file = name + ".bin";
    rec.width = static_cast<uint32_t>(m.cols);
    rec.height = static_cast<uint32_t>(m.rows);
    rec.channels = m.channels();
    rec.dtype = dtypeFromMat(m);
    rec.byte_size = tight.size();
    return true;
}

const uint32_t kDumpW = 640;
const uint32_t kDumpH = 480;

void writeManifest(const std::string& dir, uint64_t seed,
                   const std::vector<DumpRecord>& records) {
    std::string path = dir + "/manifest.json";
    std::ofstream f(path);
    if (!f) {
        std::fprintf(stderr, "FATAL: cannot write manifest %s\n", path.c_str());
        return;
    }
    f << "{\n";
    f << "  \"binary\": \"opencv-mark\",\n";
    f << "  \"impl\": \"OpenCV\",\n";
    f << "  \"seed\": " << seed << ",\n";
    f << "  \"width\": " << kDumpW << ",\n";
    f << "  \"height\": " << kDumpH << ",\n";
    f << "  \"kernels\": [\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        f << "    {";
        f << "\"name\": \"" << r.name << "\", ";
        f << "\"file\": \"" << r.file << "\", ";
        f << "\"width\": " << r.width << ", ";
        f << "\"height\": " << r.height << ", ";
        f << "\"channels\": " << r.channels << ", ";
        f << "\"dtype\": \"" << r.dtype << "\", ";
        f << "\"byte_size\": " << r.byte_size;
        f << "}";
        if (i + 1 < records.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";
}

}  // namespace

int runDumpMode(const std::string& dir, uint64_t seed) {
    ensureDir(dir);

    std::printf("=============================================================\n");
    std::printf("  opencv-mark output dump for cross-impl verification\n");
    std::printf("=============================================================\n");
    std::printf("  Output dir : %s\n", dir.c_str());
    std::printf("  Resolution : %ux%u (sentinel — fixed)\n", kDumpW, kDumpH);
    std::printf("  Seed       : %lu\n\n", static_cast<unsigned long>(seed));

    std::vector<DumpRecord> records;
    int passes = 0, fails = 0;

    auto run = [&](bool ok, const char* kname) {
        if (ok) { ++passes; std::printf("  [ok]  %s\n", kname); }
        else    { ++fails;  std::printf("  [FAIL]%s\n", kname); }
    };

    // _input_u8 — emitted ONCE so the verifier can hash-compare input
    // bytes across the OpenCV dump and the OpenVX dump. If those don't
    // match, every per-kernel diff downstream is meaningless.
    {
        OpenCVTestData gen(seed);
        cv::Mat in = gen.makeU8(kDumpW, kDumpH);
        DumpRecord rec;
        run(dumpMat(in, dir, "_input_u8", rec), "_input_u8");
        if (rec.byte_size) records.push_back(rec);
    }

    // Box3x3 — OpenVX vxuBox3x3 = "normalized 3x3 box filter, border
    // UNDEFINED". cv::boxFilter(normalize=true) matches the maths; the
    // 1px UNDEFINED border in OpenVX vs cv::BORDER_DEFAULT (reflect)
    // will create small diffs in the outermost ring of pixels — the
    // verifier excludes the border ring or applies a tolerance that
    // accounts for that.
    {
        OpenCVTestData gen(seed);
        cv::Mat in = gen.makeU8(kDumpW, kDumpH);
        cv::Mat out;
        cv::boxFilter(in, out, CV_8U, cv::Size(3, 3),
                      cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
        DumpRecord rec;
        run(dumpMat(out, dir, "Box3x3", rec), "Box3x3");
        if (rec.byte_size) records.push_back(rec);
    }

    // Gaussian3x3 — OpenVX uses fixed [1 2 1] separable kernel ÷ 16.
    // cv::GaussianBlur(3x3, sigma=0) uses (1 2 1)/4 separable ÷ 4 along
    // each axis = same overall (1 2 1)x(1 2 1)/16 = identical maths.
    {
        OpenCVTestData gen(seed);
        cv::Mat in = gen.makeU8(kDumpW, kDumpH);
        cv::Mat out;
        cv::GaussianBlur(in, out, cv::Size(3, 3), 0.0, 0.0,
                         cv::BORDER_REPLICATE);
        DumpRecord rec;
        run(dumpMat(out, dir, "Gaussian3x3", rec), "Gaussian3x3");
        if (rec.byte_size) records.push_back(rec);
    }

    // Median3x3 — selection-based, identical maths in both libs.
    {
        OpenCVTestData gen(seed);
        cv::Mat in = gen.makeU8(kDumpW, kDumpH);
        cv::Mat out;
        cv::medianBlur(in, out, 3);
        DumpRecord rec;
        run(dumpMat(out, dir, "Median3x3", rec), "Median3x3");
        if (rec.byte_size) records.push_back(rec);
    }

    // Erode3x3 / Dilate3x3 — 3x3 rectangular structuring element.
    {
        OpenCVTestData gen(seed);
        cv::Mat in = gen.makeU8(kDumpW, kDumpH);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat e_out, d_out;
        cv::erode(in, e_out, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
        cv::dilate(in, d_out, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
        DumpRecord rec_e, rec_d;
        run(dumpMat(e_out, dir, "Erode3x3", rec_e), "Erode3x3");
        run(dumpMat(d_out, dir, "Dilate3x3", rec_d), "Dilate3x3");
        if (rec_e.byte_size) records.push_back(rec_e);
        if (rec_d.byte_size) records.push_back(rec_d);
    }

    // Sobel3x3 — CV_16S outputs (signed, raw gradient — no abs, no
    // shift) to match OpenVX vxuSobel3x3 which produces vx_int16.
    {
        OpenCVTestData gen(seed);
        cv::Mat in = gen.makeU8(kDumpW, kDumpH);
        cv::Mat dx, dy;
        cv::Sobel(in, dx, CV_16S, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
        cv::Sobel(in, dy, CV_16S, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
        DumpRecord rec_dx, rec_dy;
        run(dumpMat(dx, dir, "Sobel3x3_dx", rec_dx), "Sobel3x3_dx");
        run(dumpMat(dy, dir, "Sobel3x3_dy", rec_dy), "Sobel3x3_dy");
        if (rec_dx.byte_size) records.push_back(rec_dx);
        if (rec_dy.byte_size) records.push_back(rec_dy);
    }

    // Add_U8_Saturate — two distinct inputs from one generator
    // (subsequent makeU8 calls consume more RNG output, identical
    // pattern to dumpAddSaturate in openvx_output_dumper.cpp).
    {
        OpenCVTestData gen(seed);
        cv::Mat a = gen.makeU8(kDumpW, kDumpH);
        cv::Mat b = gen.makeU8(kDumpW, kDumpH);
        cv::Mat out;
        cv::add(a, b, out);  // U8 + U8 → U8 saturate by default
        DumpRecord rec;
        run(dumpMat(out, dir, "Add_U8_Saturate", rec), "Add_U8_Saturate");
        if (rec.byte_size) records.push_back(rec);
    }

    // Not_U8 — bitwise NOT. No border, no rounding, no fixed-point.
    // Sentinel: any difference here means generators or dumpers
    // diverged.
    {
        OpenCVTestData gen(seed);
        cv::Mat in = gen.makeU8(kDumpW, kDumpH);
        cv::Mat out;
        cv::bitwise_not(in, out);
        DumpRecord rec;
        run(dumpMat(out, dir, "Not_U8", rec), "Not_U8");
        if (rec.byte_size) records.push_back(rec);
    }

    // ChannelExtract_R from packed RGB. cv::extractChannel(0) pulls
    // the R plane. OpenVX vxuChannelExtract(VX_CHANNEL_R, RGB) does
    // the same byte-pluck — no arithmetic. Replaces the dropped
    // ColorConvert_RGB2Gray (which had a BT.601 narrow-range vs
    // full-range trap — see openvx_output_dumper.cpp for the
    // rationale).
    {
        OpenCVTestData gen(seed);
        cv::Mat rgb = gen.makeRGB(kDumpW, kDumpH);
        cv::Mat r;
        cv::extractChannel(rgb, r, 0);
        DumpRecord rec;
        run(dumpMat(r, dir, "ChannelExtract_R", rec), "ChannelExtract_R");
        if (rec.byte_size) records.push_back(rec);
    }

    writeManifest(dir, seed, records);

    std::printf("\n  Dumped %d records (%d ok, %d fail) to %s\n",
                static_cast<int>(records.size()), passes, fails, dir.c_str());
    std::printf("=============================================================\n");
    return (fails == 0) ? 0 : 1;
}

}  // namespace opencv_mark
