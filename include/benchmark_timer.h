#ifndef BENCHMARK_TIMER_H
#define BENCHMARK_TIMER_H

// Pure C++ wall-clock timer — no implementation-specific dependencies.
// Lives in the shared `bench_core` static library so both openvx-mark
// and opencv-mark (and any future implementation-mark companion binary)
// can use the same timing primitive without dragging in OpenVX.
//
// OpenVX-specific performance queries (vxQueryGraph / vxQueryNode with
// VX_*_PERFORMANCE) live in include/openvx_perf_query.h, owned solely
// by openvx-mark.

#include <chrono>
#include <cstdint>

class BenchmarkTimer {
public:
    void start();
    void stop();
    double elapsed_ns() const;
    double elapsed_ms() const;

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point stop_time_;
};

#endif // BENCHMARK_TIMER_H
