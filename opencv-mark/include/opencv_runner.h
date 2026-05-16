#ifndef OPENCV_MARK_OPENCV_RUNNER_H
#define OPENCV_MARK_OPENCV_RUNNER_H

// `OpenCVRunner` is the cv::* analogue of openvx-mark's BenchmarkRunner.
// It runs each registered cv:: benchmark with identical warm-up /
// measurement / stability-retry semantics so head-to-head comparisons
// against an OpenVX implementation reduce to "same pipeline, different
// kernel call" rather than "different harnesses".
//
// Mode model (PR #1)
// -------------------
// OpenCV is immediate-mode only — there is no graph compile step to
// time. To make cross-comparison with `openvx-mark` natural we record
// each benchmark with `mode = "graph"`: the cv:: kernel is timed as a
// single repeated call on pre-allocated cv::Mat buffers, which is the
// closest analogue of `vxProcessGraph()` on a verified graph (graph
// construction + verification are excluded from the timing budget,
// matching openvx-mark's graph-mode policy). This means the (name,
// "graph", resolution) join key in `compare_reports.py` will line up
// the typical OpenVX graph-mode benchmark against the OpenCV baseline
// without any tooling changes.
//
// (A second pass that emits `mode = "immediate"` rows — for matching
// against vxu* immediate-mode benchmarks — is deferred to PR #2.)

#include "benchmark_config.h"
#include "benchmark_stats.h"
#include "opencv_context.h"
#include "opencv_test_data.h"
#include <functional>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace opencv_mark {

// One opencv-mark benchmark case definition.
//
// `setup_fn` is called once per resolution before warm-up/timing and is
// expected to allocate the cv::Mat inputs/outputs into the supplied
// CaseBuffers struct. `run_fn` is called repeatedly inside the timing
// loop and should run the cv:: kernel on the pre-allocated buffers
// only — any allocation done inside `run_fn` ends up in the timing
// budget, which would unfairly bias against OpenCV vs OpenVX graphs
// that pre-allocate. `verify_fn` runs once (after warm-up) on a small
// fixed input and returns true iff the output matches the documented
// reference within tolerance.
struct CaseBuffers {
    cv::Mat input;
    cv::Mat input_extra;   // optional second input (e.g. warp matrix)
    cv::Mat output;
    cv::Mat output_extra;  // optional second output (e.g. Sobel dy)
};

struct OpenCVBenchmarkCase {
    std::string name;
    std::string category;
    std::string feature_set;  // mirrors openvx-mark; PR1 only emits "vision"

    using SetupFn = std::function<bool(uint32_t width, uint32_t height,
                                       OpenCVTestData& gen, CaseBuffers& bufs)>;
    SetupFn setup_fn;

    using RunFn = std::function<void(CaseBuffers& bufs)>;
    RunFn run_fn;

    using VerifyFn = std::function<bool()>;
    VerifyFn verify_fn;  // optional — when null, no output verification is performed
};

class OpenCVRunner {
public:
    OpenCVRunner(const BenchmarkConfig& config);

    void addCase(const OpenCVBenchmarkCase& bc);
    void addCases(const std::vector<OpenCVBenchmarkCase>& cases);

    std::vector<BenchmarkResult> runAll();

    // Read-only view of the registered benchmark cases — used by
    // main.cpp to build a `BenchmarkCatalog` snapshot for the JSON
    // header without having to re-call the per-category register
    // functions a second time.
    const std::vector<OpenCVBenchmarkCase>& cases() const { return cases_; }
    size_t caseCount() const { return cases_.size(); }

private:
    BenchmarkResult runOne(const OpenCVBenchmarkCase& bc, const Resolution& res);
    bool shouldRun(const OpenCVBenchmarkCase& bc) const;

    const BenchmarkConfig& config_;
    std::vector<OpenCVBenchmarkCase> cases_;
};

// Per-category registration entry points — each cv_*.cpp file in
// src/benchmarks/ defines one of these. Mirrors openvx-mark's
// `registerXxxBenchmarks()` factories.
std::vector<OpenCVBenchmarkCase> registerCvFilterBenchmarks();
std::vector<OpenCVBenchmarkCase> registerCvColorBenchmarks();
std::vector<OpenCVBenchmarkCase> registerCvGeometricBenchmarks();
std::vector<OpenCVBenchmarkCase> registerCvPixelwiseBenchmarks();
std::vector<OpenCVBenchmarkCase> registerCvStatisticalBenchmarks();
std::vector<OpenCVBenchmarkCase> registerCvMiscBenchmarks();
std::vector<OpenCVBenchmarkCase> registerCvMultiscaleBenchmarks();
std::vector<OpenCVBenchmarkCase> registerCvFeatureBenchmarks();

} // namespace opencv_mark

#endif // OPENCV_MARK_OPENCV_RUNNER_H
