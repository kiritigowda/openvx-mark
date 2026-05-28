// OpenCV equivalents for the OpenVX `multiscale` category.
//
// Name parity with openvx-mark — same kernel name and same set of
// variants (kernel_size for HalfScaleGaussian, scale for
// GaussianPyramid, U8 vs S16 for Laplacian{Pyramid,Reconstruct}).
//
// Parameter mapping notes:
//
//   * Pyramid level count = DEFAULT_PYRAMID_LEVELS (= 4) from
//     benchmark_config.h. cv::buildPyramid takes a maxlevel parameter;
//     we pass `levels - 1` so the OpenCV pyramid level count matches
//     openvx-mark's (both treat the input image as level 0).
//   * GaussianPyramid uses cv::buildPyramid which always uses the
//     5x5 Gaussian pyrDown filter at scale 0.5x — matches OpenVX's
//     VX_SCALE_PYRAMID_HALF (5x5 Gaussian per spec §3.23).
//   * GaussianPyramid_ORB: VX_SCALE_PYRAMID_ORB = 0.8408964 (= 2^-(1/4))
//     per the OpenVX 1.3.1 spec [REQ-0189]. OpenCV doesn't have a
//     dedicated `cv::buildOrbPyramid`, so we mimic the ORB scale via a
//     manual loop: per level, apply cv::GaussianBlur (5x5) and then
//     cv::resize at the ORB ratio. The per-level work is the same set
//     of operations OpenVX does internally for ORB scale.
//   * LaplacianPyramid: chain of pyrDown → pyrUp → subtract per level.
//     The S16 variant runs the same chain on CV_16SC1 buffers
//     (cv::pyrDown and cv::pyrUp both support CV_16S).
//   * LaplacianReconstruct: chain of pyrUp → add per level (inverse).
//   * HalfScaleGaussian variants by kernel_size:
//       - kernel_size=1 → no blur, just downsample by 2 using
//         INTER_NEAREST (matching OpenVX semantics — kernel_size=1 in
//         OpenVX means a 1x1 Gaussian = no blur, downsample with
//         VX_INTERPOLATION_NEAREST_NEIGHBOR per spec §3.45)
//       - kernel_size=3 → cv::GaussianBlur(3x3) + cv::resize NEAREST 0.5x
//       - kernel_size=5 → cv::pyrDown (built-in 5x5 Gaussian + downsample)

#include "opencv_runner.h"
#include "benchmark_config.h"  // DEFAULT_PYRAMID_LEVELS
#include <algorithm>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace opencv_mark {

std::vector<OpenCVBenchmarkCase> registerCvMultiscaleBenchmarks() {
    std::vector<OpenCVBenchmarkCase> cases;

    // GaussianPyramid — U8 in, vector<cv::Mat> out (DEFAULT_PYRAMID_LEVELS).
    {
        OpenCVBenchmarkCase bc;
        bc.name = "GaussianPyramid";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            std::vector<cv::Mat> levels;
            cv::buildPyramid(bufs.input, levels, DEFAULT_PYRAMID_LEVELS - 1);
            // Touch to defeat DCE.
            (void)levels.size();
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            std::vector<cv::Mat> lv;
            cv::buildPyramid(in, lv, 3);
            // Centre pixel of every level should remain ~100 for a
            // uniform input.
            return lv.size() == 4 && lv[3].at<uint8_t>(4, 4) == 100;
        };
        cases.push_back(bc);
    }

    // LaplacianPyramid — U8 in, chain of pyrDown / pyrUp / subtract
    // per level. We reuse `output` as the working level0 buffer and
    // `output_extra` as the upsampled level1 buffer.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "LaplacianPyramid";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::Mat current = bufs.input;
            for (int i = 0; i < DEFAULT_PYRAMID_LEVELS - 1; ++i) {
                cv::Mat down, up, diff;
                cv::pyrDown(current, down);
                cv::pyrUp(down, up, current.size());
                cv::subtract(current, up, diff);
                current = down;
            }
            (void)current.cols;
        };
        bc.verify_fn = []() -> bool {
            // Just verify that the chain runs without throwing on a
            // small uniform input — exact Laplacian values depend on
            // the pyrDown/pyrUp Gaussian kernels and aren't worth
            // hard-coding here.
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat down, up, diff;
            cv::pyrDown(in, down);
            cv::pyrUp(down, up, in.size());
            cv::subtract(in, up, diff);
            return diff.cols == 64 && diff.rows == 64;
        };
        cases.push_back(bc);
    }

    // HalfScaleGaussian — U8 in, U8 out at half resolution, kernel_size=3.
    //
    // Matches openvx-mark's HalfScaleGaussian (which uses kernel_size=3
    // — a 3x3 Gaussian followed by nearest-neighbour 0.5x downsample
    // per OpenVX 1.3.1 §3.45). NOT the same as cv::pyrDown (which uses
    // a 5x5 Gaussian); cv::pyrDown corresponds to HalfScaleGaussian_5x5
    // below.
    {
        OpenCVBenchmarkCase bc;
        bc.name = "HalfScaleGaussian";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            // We reuse output_extra as the blurred-but-not-downsampled
            // intermediate so the per-iteration call doesn't allocate.
            bufs.output_extra.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            bufs.output.create(static_cast<int>(h / 2), static_cast<int>(w / 2), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::GaussianBlur(bufs.input, bufs.output_extra, cv::Size(3, 3),
                             /*sigmaX=*/0, /*sigmaY=*/0, cv::BORDER_REPLICATE);
            cv::resize(bufs.output_extra, bufs.output,
                       cv::Size(bufs.output.cols, bufs.output.rows),
                       0, 0, cv::INTER_NEAREST);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat blurred, o;
            cv::GaussianBlur(in, blurred, cv::Size(3, 3), 0, 0, cv::BORDER_REPLICATE);
            cv::resize(blurred, o, cv::Size(32, 32), 0, 0, cv::INTER_NEAREST);
            return o.at<uint8_t>(16, 16) == 100;
        };
        cases.push_back(bc);
    }

    // HalfScaleGaussian_1x1 — kernel_size=1 (no blur, just downsample).
    {
        OpenCVBenchmarkCase bc;
        bc.name = "HalfScaleGaussian_1x1";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h / 2), static_cast<int>(w / 2), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::resize(bufs.input, bufs.output,
                       cv::Size(bufs.output.cols, bufs.output.rows),
                       0, 0, cv::INTER_NEAREST);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat o;
            cv::resize(in, o, cv::Size(32, 32), 0, 0, cv::INTER_NEAREST);
            return o.at<uint8_t>(16, 16) == 100;
        };
        cases.push_back(bc);
    }

    // HalfScaleGaussian_5x5 — kernel_size=5 (cv::pyrDown is exactly this).
    {
        OpenCVBenchmarkCase bc;
        bc.name = "HalfScaleGaussian_5x5";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.setup_fn = [](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeU8(w, h);
            bufs.output.create(static_cast<int>(h / 2), static_cast<int>(w / 2), CV_8UC1);
            return true;
        };
        bc.run_fn = [](CaseBuffers& bufs) {
            cv::pyrDown(bufs.input, bufs.output,
                        cv::Size(bufs.output.cols, bufs.output.rows));
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat o;
            cv::pyrDown(in, o);
            return o.at<uint8_t>(16, 16) == 100;
        };
        cases.push_back(bc);
    }

    // GaussianPyramid_ORB — U8 in, manual loop at ORB scale (4/5 ≈ 0.84).
    //
    // OpenVX 1.3.1 §3.23 [REQ-0189] requires both VX_SCALE_PYRAMID_HALF
    // (0.5) and VX_SCALE_PYRAMID_ORB (0.8408964 = 2^-(1/4)) to be
    // supported. cv::buildPyramid only does 0.5x; we mimic the ORB
    // ratio with a manual loop of Gaussian blur + resize per level.
    //
    // Per-level scratch buffers (`blurred[i]`, `downsampled[i]`) are
    // preallocated to their exact final sizes in setup_fn so the
    // timed run_fn does zero cv::Mat allocations. Without this every
    // iteration of every level would call cv::Mat::create() and
    // potentially malloc, which on a deep ORB chain at FHD is
    // ~25 MB of per-iter allocator traffic — enough to swamp the
    // actual Gaussian + resize cost.
    {
        struct OrbPyramidState {
            std::vector<cv::Mat> blurred;
            std::vector<cv::Mat> downsampled;
        };
        auto state = std::make_shared<OrbPyramidState>();

        OpenCVBenchmarkCase bc;
        bc.name = "GaussianPyramid_ORB";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.setup_fn = [state](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            constexpr float ORB_SCALE = 0.8408964f;
            bufs.input = gen.makeU8(w, h);

            state->blurred.assign(DEFAULT_PYRAMID_LEVELS - 1, cv::Mat());
            state->downsampled.assign(DEFAULT_PYRAMID_LEVELS - 1, cv::Mat());

            // Walk the same shrink chain as run_fn to compute each
            // level's dims and allocate the matching scratch Mats.
            int cur_w = static_cast<int>(w);
            int cur_h = static_cast<int>(h);
            for (int i = 0; i < DEFAULT_PYRAMID_LEVELS - 1; ++i) {
                state->blurred[i].create(cur_h, cur_w, CV_8UC1);
                int new_w = std::max(1, static_cast<int>(cur_w * ORB_SCALE));
                int new_h = std::max(1, static_cast<int>(cur_h * ORB_SCALE));
                state->downsampled[i].create(new_h, new_w, CV_8UC1);
                cur_w = new_w;
                cur_h = new_h;
            }
            return true;
        };
        bc.run_fn = [state](CaseBuffers& bufs) {
            const cv::Mat* current = &bufs.input;
            for (int i = 0; i < DEFAULT_PYRAMID_LEVELS - 1; ++i) {
                cv::GaussianBlur(*current, state->blurred[i], cv::Size(5, 5),
                                 /*sigmaX=*/0, /*sigmaY=*/0, cv::BORDER_REPLICATE);
                cv::resize(state->blurred[i], state->downsampled[i],
                           state->downsampled[i].size(),
                           0, 0, cv::INTER_LINEAR);
                current = &state->downsampled[i];
            }
            (void)current->cols;
        };
        bc.verify_fn = []() -> bool {
            // Just verify the chain runs end-to-end on a small uniform
            // input — the exact center value depends on the resize
            // interpolation method and isn't worth pinning here.
            cv::Mat in(64, 64, CV_8UC1, cv::Scalar(100));
            cv::Mat blurred, down;
            cv::GaussianBlur(in, blurred, cv::Size(5, 5), 0, 0, cv::BORDER_REPLICATE);
            cv::resize(blurred, down, cv::Size(54, 54), 0, 0, cv::INTER_LINEAR);
            return down.cols == 54;
        };
        cases.push_back(bc);
    }

    // LaplacianPyramid_S16 — S16 in, S16 working buffers throughout.
    //
    // Per-level `down[i]`, `up[i]`, `diff[i]` Mats are preallocated to
    // their final sizes in setup_fn — see LaplacianReconstruct below
    // for the same rationale: keeping run_fn allocation-free is what
    // the cross-impl OpenVX-graph comparison demands.
    {
        struct LapPyrState {
            std::vector<cv::Mat> down;
            std::vector<cv::Mat> up;
            std::vector<cv::Mat> diff;
        };
        auto state = std::make_shared<LapPyrState>();

        OpenCVBenchmarkCase bc;
        bc.name = "LaplacianPyramid_S16";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.setup_fn = [state](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            bufs.input = gen.makeS16(w, h);
            const int levels = DEFAULT_PYRAMID_LEVELS - 1;
            state->down.assign(levels, cv::Mat());
            state->up.assign(levels, cv::Mat());
            state->diff.assign(levels, cv::Mat());

            int cur_w = static_cast<int>(w);
            int cur_h = static_cast<int>(h);
            for (int i = 0; i < levels; ++i) {
                int dw = std::max(1, cur_w / 2);
                int dh = std::max(1, cur_h / 2);
                state->down[i].create(dh, dw, CV_16SC1);
                state->up[i].create(cur_h, cur_w, CV_16SC1);
                state->diff[i].create(cur_h, cur_w, CV_16SC1);
                cur_w = dw;
                cur_h = dh;
            }
            return true;
        };
        bc.run_fn = [state](CaseBuffers& bufs) {
            const cv::Mat* current = &bufs.input;
            const int levels = DEFAULT_PYRAMID_LEVELS - 1;
            for (int i = 0; i < levels; ++i) {
                cv::pyrDown(*current, state->down[i], state->down[i].size());
                cv::pyrUp(state->down[i], state->up[i], current->size());
                cv::subtract(*current, state->up[i], state->diff[i]);  // CV_16S in/out
                current = &state->down[i];
            }
            (void)current->cols;
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(64, 64, CV_16SC1, cv::Scalar(500));
            cv::Mat down, up, diff;
            cv::pyrDown(in, down);
            cv::pyrUp(down, up, in.size());
            cv::subtract(in, up, diff);
            return diff.type() == CV_16SC1 && diff.cols == 64 && diff.rows == 64;
        };
        cases.push_back(bc);
    }

    // LaplacianReconstruct — inverse of LaplacianPyramid: chain of
    // pyrUp + add per level, reconstructing the full-resolution image
    // from a lowest-resolution input plus the Laplacian residuals.
    //
    // For the benchmark we don't have a Laplacian pyramid laying around,
    // so we synthesise the input shape: a low-resolution starting image
    // and per-level residual buffers, then time the pyrUp+add chain. The
    // per-iteration work is the dominant cost of vxLaplacianReconstructNode.
    //
    // Per-level upsample buffers are preallocated in setup_fn. The
    // previous version allocated a fresh cv::Mat (up) plus a
    // cv::Mat::zeros() residual every level every iteration; at
    // DEFAULT_PYRAMID_LEVELS=4 + FHD that's ~20 MB of allocator
    // traffic per iteration, more than the pyrUp work itself for
    // small levels. Now: one `up[i]` per level, one shared
    // `zero_residual` sized to the largest level — both preallocated.
    {
        struct LapReconState {
            std::vector<cv::Mat> up;        // upsampled buffer per reconstruction step
            cv::Mat              zero_residual;  // shared, sized to largest level
        };
        auto state = std::make_shared<LapReconState>();

        OpenCVBenchmarkCase bc;
        bc.name = "LaplacianReconstruct";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.setup_fn = [state](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            // Low-resolution starting image at width >> (levels-1).
            const int shift = DEFAULT_PYRAMID_LEVELS - 1;
            uint32_t low_w = std::max<uint32_t>(1, w >> shift);
            uint32_t low_h = std::max<uint32_t>(1, h >> shift);
            bufs.input = gen.makeU8(low_w, low_h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);

            // Preallocate one `up[i]` per reconstruction step, sized to
            // its target dims (matches the up_size computed in run_fn).
            state->up.assign(shift, cv::Mat());
            for (int i = shift - 1, k = 0; i >= 0; --i, ++k) {
                int uw = std::max(1, bufs.output.cols >> i);
                int uh = std::max(1, bufs.output.rows >> i);
                state->up[k].create(uh, uw, CV_8UC1);
            }
            // The zero residual just needs to be at least as large as
            // the largest up buffer. cv::add ignores extra rows/cols
            // when given an explicit dst, so a max-size zero buffer
            // works as a "scratch zero" for every level.
            state->zero_residual = cv::Mat::zeros(static_cast<int>(h),
                                                  static_cast<int>(w), CV_8UC1);
            return true;
        };
        bc.run_fn = [state](CaseBuffers& bufs) {
            const cv::Mat* current = &bufs.input;
            const int shift = DEFAULT_PYRAMID_LEVELS - 1;
            for (int i = shift - 1, k = 0; i >= 0; --i, ++k) {
                cv::pyrUp(*current, state->up[k], state->up[k].size());
                // Residual addition (using a zero residual here — measures
                // pyrUp + add cost, matching vxLaplacianReconstructNode).
                // Use a sub-roi of zero_residual sized to up[k] to avoid
                // any per-iter allocation while keeping cv::add happy.
                cv::add(state->up[k],
                        state->zero_residual(cv::Rect(0, 0,
                                                       state->up[k].cols,
                                                       state->up[k].rows)),
                        state->up[k]);
                current = &state->up[k];
            }
            current->copyTo(bufs.output);
        };
        bc.verify_fn = []() -> bool {
            // Smoke check: chain runs end-to-end on a 16x12 input upsampled
            // through 3 pyrUp stages back to 128x96.
            cv::Mat in(12, 16, CV_8UC1, cv::Scalar(100));
            cv::Mat cur = in;
            for (int i = 0; i < 3; ++i) {
                cv::Mat up;
                cv::pyrUp(cur, up);
                cur = up;
            }
            return cur.cols == 128 && cur.rows == 96;
        };
        cases.push_back(bc);
    }

    // LaplacianReconstruct_S16 — same chain on CV_16SC1 buffers,
    // same preallocation pattern as the U8 variant above.
    {
        struct LapReconS16State {
            std::vector<cv::Mat> up;
            cv::Mat              zero_residual;
        };
        auto state = std::make_shared<LapReconS16State>();

        OpenCVBenchmarkCase bc;
        bc.name = "LaplacianReconstruct_S16";
        bc.category = "multiscale";
        bc.feature_set = "vision";
        bc.setup_fn = [state](uint32_t w, uint32_t h, OpenCVTestData& gen, CaseBuffers& bufs) -> bool {
            const int shift = DEFAULT_PYRAMID_LEVELS - 1;
            uint32_t low_w = std::max<uint32_t>(1, w >> shift);
            uint32_t low_h = std::max<uint32_t>(1, h >> shift);
            bufs.input = gen.makeS16(low_w, low_h);
            bufs.output.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);

            state->up.assign(shift, cv::Mat());
            for (int i = shift - 1, k = 0; i >= 0; --i, ++k) {
                int uw = std::max(1, bufs.output.cols >> i);
                int uh = std::max(1, bufs.output.rows >> i);
                state->up[k].create(uh, uw, CV_16SC1);
            }
            state->zero_residual = cv::Mat::zeros(static_cast<int>(h),
                                                  static_cast<int>(w), CV_16SC1);
            return true;
        };
        bc.run_fn = [state](CaseBuffers& bufs) {
            const cv::Mat* current = &bufs.input;
            const int shift = DEFAULT_PYRAMID_LEVELS - 1;
            for (int i = shift - 1, k = 0; i >= 0; --i, ++k) {
                cv::pyrUp(*current, state->up[k], state->up[k].size());
                cv::add(state->up[k],
                        state->zero_residual(cv::Rect(0, 0,
                                                       state->up[k].cols,
                                                       state->up[k].rows)),
                        state->up[k]);
                current = &state->up[k];
            }
            current->copyTo(bufs.output);
        };
        bc.verify_fn = []() -> bool {
            cv::Mat in(12, 16, CV_16SC1, cv::Scalar(500));
            cv::Mat cur = in;
            for (int i = 0; i < 3; ++i) {
                cv::Mat up;
                cv::pyrUp(cur, up);
                cur = up;
            }
            return cur.cols == 128 && cur.rows == 96 && cur.type() == CV_16SC1;
        };
        cases.push_back(bc);
    }

    return cases;
}

} // namespace opencv_mark
