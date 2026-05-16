// openvx_output_dumper — dumps the raw pixel bytes of a small,
// hand-curated sentinel kernel set so cross-impl numerical
// verification can compare OpenVX outputs against the OpenCV
// baseline (and against each other).
//
// Design constraints:
//
//   * Same input bytes on both sides. Both binaries use
//     std::mt19937_64(seed) + uniform_int_distribution<int>(0,255)
//     for U8 image fills, so the produced bytes are identical
//     for the same seed. The dumper hash-fingerprints the input
//     before any kernel runs so a mismatched MD5 in the manifest
//     would immediately surface a generator skew.
//
//   * Single resolution, single seed — VGA (640×480) with seed 42.
//     Each kernel runs ONCE — no warmup, no timing, no iterations.
//     Bytes go directly to disk; nothing is benchmarked here.
//
//   * Sentinel kernels chosen so a) every implementation actually
//     ships them (no point dumping kernels MIVisionX has but rustVX
//     hasn't), b) they exercise a representative mix of computational
//     shapes (3×3 box/gauss = separable, sobel = signed-output stencil,
//     add+saturate = pixelwise arithmetic, RGB→gray = color convert).
//     This set is small enough to read at a glance in the manifest
//     and large enough to catch a real semantic divergence.
//
//   * The companion script scripts/cross_verify_outputs.py reads two
//     dump directories (typically opencv vs <openvx-impl>), matches
//     kernels by name, computes max-abs-diff + PSNR + mean-abs-diff,
//     and gates with per-kernel tolerance thresholds. A kernel exceeding
//     tolerance is a real correctness divergence worth investigating —
//     not a CI-flake.

#include "benchmark_config.h"
#include "test_data_generator.h"
#include "resource_tracker.h"
#include <VX/vx.h>
#include <VX/vxu.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace openvx_dump {

namespace {

// Public dump record — one entry per dumped kernel. We don't link a
// JSON lib so we serialise the manifest by hand. Fields kept flat for
// easy diff in CI logs.
struct DumpRecord {
    std::string name;
    std::string file;
    uint32_t width = 0;
    uint32_t height = 0;
    int channels = 0;
    std::string dtype;   // "u8" or "s16"
    size_t byte_size = 0;
};

void ensureDir(const std::string& path) {
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) {
        mkdir(path.c_str(), 0755);
    }
}

// Map an image plane and dump it into <dir>/<name>.bin. Returns false
// on any vxMap error — caller should record the failure but keep going.
bool dumpImagePlane(vx_image img, const std::string& dir,
                    const std::string& name, vx_uint32 plane_idx,
                    DumpRecord& rec) {
    vx_uint32 w = 0, h = 0;
    vx_df_image fmt = 0;
    vxQueryImage(img, VX_IMAGE_WIDTH, &w, sizeof(w));
    vxQueryImage(img, VX_IMAGE_HEIGHT, &h, sizeof(h));
    vxQueryImage(img, VX_IMAGE_FORMAT, &fmt, sizeof(fmt));

    int channels = 1;
    int bpp = 1;
    const char* dtype = "u8";
    switch (fmt) {
        case VX_DF_IMAGE_U8:   bpp = 1; channels = 1; dtype = "u8";  break;
        case VX_DF_IMAGE_S16:  bpp = 2; channels = 1; dtype = "s16"; break;
        case VX_DF_IMAGE_U16:  bpp = 2; channels = 1; dtype = "u16"; break;
        case VX_DF_IMAGE_RGB:  bpp = 3; channels = 3; dtype = "u8";  break;
        case VX_DF_IMAGE_RGBX: bpp = 4; channels = 4; dtype = "u8";  break;
        default:               bpp = 1; channels = 1; dtype = "u8";  break;
    }

    vx_rectangle_t rect{0, 0, w, h};
    vx_imagepatch_addressing_t addr{};
    void* ptr = nullptr;
    vx_map_id map_id;
    vx_status s = vxMapImagePatch(img, &rect, plane_idx, &map_id, &addr,
                                  &ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST,
                                  VX_NOGAP_X);
    if (s != VX_SUCCESS) {
        std::fprintf(stderr, "  dump %s: vxMapImagePatch failed (status=%d)\n",
                     name.c_str(), s);
        return false;
    }

    // OpenVX may stride wider than width × bpp; copy row-by-row so the
    // dumped .bin is a tightly-packed h × w × channels buffer that the
    // verifier can mmap as a numpy array without stride bookkeeping.
    const size_t row_bytes = static_cast<size_t>(w) * static_cast<size_t>(bpp);
    std::vector<uint8_t> tight(static_cast<size_t>(h) * row_bytes);
    const uint8_t* src = static_cast<const uint8_t*>(ptr);
    for (uint32_t y = 0; y < h; ++y) {
        std::memcpy(tight.data() + y * row_bytes,
                    src + y * addr.stride_y, row_bytes);
    }
    vxUnmapImagePatch(img, map_id);

    std::string path = dir + "/" + name + ".bin";
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "  dump %s: cannot open %s for writing\n",
                     name.c_str(), path.c_str());
        return false;
    }
    f.write(reinterpret_cast<const char*>(tight.data()), tight.size());
    f.close();

    rec.name = name;
    rec.file = name + ".bin";
    rec.width = w;
    rec.height = h;
    rec.channels = channels;
    rec.dtype = dtype;
    rec.byte_size = tight.size();
    return true;
}

// ----- Per-kernel dump routines -----
//
// Each takes a fresh TestDataGenerator(seed) so the random sequence
// resets per kernel — matches what BenchmarkRunner does in runGraphMode
// (it constructs a new generator for every graph setup). Output image
// gets dumped via dumpImagePlane. Input dumps are optional and only
// done for the first kernel (Box3x3) so the verifier can sanity-check
// that both binaries produced the same input bytes.

const uint32_t kDumpW = 640;
const uint32_t kDumpH = 480;

bool dumpBox3x3(vx_context ctx, const std::string& dir,
                std::vector<DumpRecord>& records, uint64_t seed) {
    TestDataGenerator gen(seed);
    vx_image in = gen.createFilledImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_image out = vxCreateImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    // Dump the input ONCE — same seed across all kernels, so this proves
    // input parity for the whole suite without dumping per kernel.
    DumpRecord in_rec;
    if (dumpImagePlane(in, dir, "_input_u8", 0, in_rec)) {
        records.push_back(in_rec);
    }
    vx_status s = vxuBox3x3(ctx, in, out);
    DumpRecord rec;
    if (s == VX_SUCCESS && dumpImagePlane(out, dir, "Box3x3", 0, rec)) {
        records.push_back(rec);
    } else {
        std::fprintf(stderr, "  Box3x3: vxuBox3x3 failed (status=%d)\n", s);
    }
    vxReleaseImage(&in);
    vxReleaseImage(&out);
    return s == VX_SUCCESS;
}

bool dumpGaussian3x3(vx_context ctx, const std::string& dir,
                     std::vector<DumpRecord>& records, uint64_t seed) {
    TestDataGenerator gen(seed);
    vx_image in = gen.createFilledImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_image out = vxCreateImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_status s = vxuGaussian3x3(ctx, in, out);
    DumpRecord rec;
    if (s == VX_SUCCESS && dumpImagePlane(out, dir, "Gaussian3x3", 0, rec)) {
        records.push_back(rec);
    } else {
        std::fprintf(stderr, "  Gaussian3x3: vxuGaussian3x3 failed (status=%d)\n", s);
    }
    vxReleaseImage(&in);
    vxReleaseImage(&out);
    return s == VX_SUCCESS;
}

bool dumpMedian3x3(vx_context ctx, const std::string& dir,
                   std::vector<DumpRecord>& records, uint64_t seed) {
    TestDataGenerator gen(seed);
    vx_image in = gen.createFilledImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_image out = vxCreateImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_status s = vxuMedian3x3(ctx, in, out);
    DumpRecord rec;
    if (s == VX_SUCCESS && dumpImagePlane(out, dir, "Median3x3", 0, rec)) {
        records.push_back(rec);
    } else {
        std::fprintf(stderr, "  Median3x3: vxuMedian3x3 failed (status=%d)\n", s);
    }
    vxReleaseImage(&in);
    vxReleaseImage(&out);
    return s == VX_SUCCESS;
}

bool dumpErode3x3(vx_context ctx, const std::string& dir,
                  std::vector<DumpRecord>& records, uint64_t seed) {
    TestDataGenerator gen(seed);
    vx_image in = gen.createFilledImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_image out = vxCreateImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_status s = vxuErode3x3(ctx, in, out);
    DumpRecord rec;
    if (s == VX_SUCCESS && dumpImagePlane(out, dir, "Erode3x3", 0, rec)) {
        records.push_back(rec);
    } else {
        std::fprintf(stderr, "  Erode3x3: vxuErode3x3 failed (status=%d)\n", s);
    }
    vxReleaseImage(&in);
    vxReleaseImage(&out);
    return s == VX_SUCCESS;
}

bool dumpDilate3x3(vx_context ctx, const std::string& dir,
                   std::vector<DumpRecord>& records, uint64_t seed) {
    TestDataGenerator gen(seed);
    vx_image in = gen.createFilledImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_image out = vxCreateImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_status s = vxuDilate3x3(ctx, in, out);
    DumpRecord rec;
    if (s == VX_SUCCESS && dumpImagePlane(out, dir, "Dilate3x3", 0, rec)) {
        records.push_back(rec);
    } else {
        std::fprintf(stderr, "  Dilate3x3: vxuDilate3x3 failed (status=%d)\n", s);
    }
    vxReleaseImage(&in);
    vxReleaseImage(&out);
    return s == VX_SUCCESS;
}

bool dumpSobel3x3(vx_context ctx, const std::string& dir,
                  std::vector<DumpRecord>& records, uint64_t seed) {
    TestDataGenerator gen(seed);
    vx_image in = gen.createFilledImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_image dx = vxCreateImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_S16);
    vx_image dy = vxCreateImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_S16);
    vx_status s = vxuSobel3x3(ctx, in, dx, dy);
    DumpRecord rec_dx, rec_dy;
    if (s == VX_SUCCESS) {
        if (dumpImagePlane(dx, dir, "Sobel3x3_dx", 0, rec_dx)) records.push_back(rec_dx);
        if (dumpImagePlane(dy, dir, "Sobel3x3_dy", 0, rec_dy)) records.push_back(rec_dy);
    } else {
        std::fprintf(stderr, "  Sobel3x3: vxuSobel3x3 failed (status=%d)\n", s);
    }
    vxReleaseImage(&in);
    vxReleaseImage(&dx);
    vxReleaseImage(&dy);
    return s == VX_SUCCESS;
}

bool dumpAddSaturate(vx_context ctx, const std::string& dir,
                     std::vector<DumpRecord>& records, uint64_t seed) {
    // Two distinct inputs by reseeding mid-generate. Same as how the
    // opencv-mark dump constructs its operands (see cv_dump.cpp).
    TestDataGenerator gen(seed);
    vx_image a = gen.createFilledImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_image b = gen.createFilledImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_image out = vxCreateImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_status s = vxuAdd(ctx, a, b, VX_CONVERT_POLICY_SATURATE, out);
    DumpRecord rec;
    if (s == VX_SUCCESS && dumpImagePlane(out, dir, "Add_U8_Saturate", 0, rec)) {
        records.push_back(rec);
    } else {
        std::fprintf(stderr, "  Add_U8_Saturate: vxuAdd failed (status=%d)\n", s);
    }
    vxReleaseImage(&a);
    vxReleaseImage(&b);
    vxReleaseImage(&out);
    return s == VX_SUCCESS;
}

bool dumpNotU8(vx_context ctx, const std::string& dir,
               std::vector<DumpRecord>& records, uint64_t seed) {
    // Bitwise NOT — picked as a "if THIS doesn't match, your dumper or
    // input-fill is broken" sentinel. No border, no rounding, no
    // saturation, no fixed-point — must be bitwise identical between
    // any two implementations that ship vxuNot / cv::bitwise_not.
    TestDataGenerator gen(seed);
    vx_image in = gen.createFilledImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_image out = vxCreateImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_status s = vxuNot(ctx, in, out);
    DumpRecord rec;
    if (s == VX_SUCCESS && dumpImagePlane(out, dir, "Not_U8", 0, rec)) {
        records.push_back(rec);
    } else {
        std::fprintf(stderr, "  Not_U8: vxuNot failed (status=%d)\n", s);
    }
    vxReleaseImage(&in);
    vxReleaseImage(&out);
    return s == VX_SUCCESS;
}

bool dumpChannelExtractR(vx_context ctx, const std::string& dir,
                         std::vector<DumpRecord>& records, uint64_t seed) {
    // ChannelExtract R from a packed VX_DF_IMAGE_RGB. No arithmetic,
    // no rounding. Should be byte-identical to cv::extractChannel(0)
    // on the same RGB input.
    //
    // Why deliberately NOT ColorConvert_RGB2Gray: OpenVX 1.x's
    // vxuColorConvert RGB→IYUV produces narrow-range BT.601 luma
    // (Y in [16, 235]); cv::cvtColor(COLOR_RGB2GRAY) emits full-range
    // luma (Y in [0, 255]). That's a real ~16% systematic offset that
    // would dwarf any actual implementation skew — useless as a
    // verifier signal. ChannelExtract has no such trap.
    TestDataGenerator gen(seed);
    vx_image in = gen.createFilledImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_RGB);
    vx_image out = vxCreateImage(ctx, kDumpW, kDumpH, VX_DF_IMAGE_U8);
    vx_status s = vxuChannelExtract(ctx, in, VX_CHANNEL_R, out);
    DumpRecord rec;
    if (s == VX_SUCCESS && dumpImagePlane(out, dir, "ChannelExtract_R", 0, rec)) {
        records.push_back(rec);
    } else {
        std::fprintf(stderr, "  ChannelExtract_R: status=%d\n", s);
    }
    vxReleaseImage(&in);
    vxReleaseImage(&out);
    return s == VX_SUCCESS;
}

void writeManifest(const std::string& dir, const std::string& impl_name,
                   uint64_t seed,
                   const std::vector<DumpRecord>& records) {
    std::string path = dir + "/manifest.json";
    std::ofstream f(path);
    if (!f) {
        std::fprintf(stderr, "FATAL: cannot write manifest %s\n", path.c_str());
        return;
    }
    f << "{\n";
    f << "  \"binary\": \"openvx-mark\",\n";
    f << "  \"impl\": \"" << impl_name << "\",\n";
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

int runDumpMode(const BenchmarkConfig& config, vx_context ctx) {
    const std::string& dir = config.dump_outputs_dir;
    ensureDir(dir);

    std::printf("=============================================================\n");
    std::printf("  openvx-mark output dump for cross-impl verification\n");
    std::printf("=============================================================\n");
    std::printf("  Output dir : %s\n", dir.c_str());
    std::printf("  Resolution : %ux%u (sentinel — fixed)\n", kDumpW, kDumpH);
    std::printf("  Seed       : %lu\n\n", static_cast<unsigned long>(config.seed));

    // Discover the impl name for the manifest — we re-query rather than
    // plumbing it through from main() to keep the dump module loosely
    // coupled to the rest of openvx-mark's startup.
    char impl_buf[VX_MAX_IMPLEMENTATION_NAME] = {};
    vxQueryContext(ctx, VX_CONTEXT_IMPLEMENTATION, impl_buf, sizeof(impl_buf));

    std::vector<DumpRecord> records;
    int passes = 0, fails = 0;

    auto run = [&](bool ok, const char* kname) {
        if (ok) { ++passes; std::printf("  [ok]  %s\n", kname); }
        else    { ++fails;  std::printf("  [FAIL]%s\n", kname); }
    };

    run(dumpBox3x3(ctx, dir, records, config.seed),               "Box3x3");
    run(dumpGaussian3x3(ctx, dir, records, config.seed),          "Gaussian3x3");
    run(dumpMedian3x3(ctx, dir, records, config.seed),            "Median3x3");
    run(dumpErode3x3(ctx, dir, records, config.seed),             "Erode3x3");
    run(dumpDilate3x3(ctx, dir, records, config.seed),            "Dilate3x3");
    run(dumpSobel3x3(ctx, dir, records, config.seed),             "Sobel3x3");
    run(dumpAddSaturate(ctx, dir, records, config.seed),          "Add_U8_Saturate");
    run(dumpNotU8(ctx, dir, records, config.seed),                "Not_U8");
    run(dumpChannelExtractR(ctx, dir, records, config.seed),      "ChannelExtract_R");

    writeManifest(dir, impl_buf, config.seed, records);

    std::printf("\n  Dumped %d records (%d ok, %d fail) to %s\n",
                static_cast<int>(records.size()), passes, fails, dir.c_str());
    std::printf("=============================================================\n");
    return (fails == 0) ? 0 : 1;
}

}  // namespace openvx_dump
