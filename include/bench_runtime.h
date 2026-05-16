#ifndef BENCH_RUNTIME_H
#define BENCH_RUNTIME_H

// bench_runtime — implementation-agnostic helpers that live in
// `bench_core` so both openvx-mark and opencv-mark get identical
// threading-policy plumbing and timer self-test behaviour without
// either binary needing to roll its own.
//
// Three knobs, all surfaced through the JSON `system` block so a
// reader can audit what the binary actually did:
//
//   1. applyThreadingPolicy(N, info)
//        - Sets OMP_NUM_THREADS=N (via setenv) before any kernel lib
//          spins up its thread pool.
//        - Caller (opencv-mark only) is expected to also call
//          cv::setNumThreads(N) so OpenCV's TBB/OpenMP pool aligns.
//        - Records the resolved values into SystemInfo so JSON can
//          report "you asked for N, here's what each library saw".
//        - N=0 means "don't touch anything" — for headline max-perf
//          runs where each impl uses its own default.
//
//   2. runTimerValidation(info)
//        - Measures the monotonic clock resolution (smallest non-zero
//          inter-sample delta from a tight loop) and the timer's error
//          against std::this_thread::sleep_for(1ms/10ms/100ms).
//        - Prints a small PASS/FAIL report.
//        - Returns 0 on PASS (all errors <5%) / 1 on FAIL, so main()
//          can exit with that code.
//        - Populates SystemInfo timer_* fields for the JSON.
//
//   3. detectOpenMPThreads()
//        - Returns omp_get_max_threads() if OpenMP is linked into the
//          binary, 0 otherwise. Used to fill SystemInfo at startup.
//
// The header is dependency-free (no OpenVX / no OpenCV) so it ships
// inside bench_core unchanged for every <impl>-mark companion binary.

#include "system_info.h"

namespace bench_runtime {

// Returns 0 = OpenMP not linked / not enabled, else omp_get_max_threads().
int detectOpenMPThreads();

// Apply the requested threading policy and fill SystemInfo fields.
// `requested` is BenchmarkConfig::threads (0 = leave defaults). After
// this returns, info.requested_threads / info.openmp_max_threads /
// info.omp_num_threads_env are populated. opencv_threads is left for
// the caller (only opencv-mark links cv::setNumThreads).
void applyThreadingPolicy(int requested, SystemInfo& info);

// Run the timer self-test. Populates info.timing_validated +
// timer_resolution_ns + timer_sleep_*ms_err_pct.
// Returns 0 on PASS, 1 on FAIL.
int runTimerValidation(SystemInfo& info);

}  // namespace bench_runtime

#endif  // BENCH_RUNTIME_H
