#include "benchmark_timer.h"
#include <atomic>

void BenchmarkTimer::start() {
    // Prevent compiler/CPU reordering of work before the timing window
    // into the measurement window.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    start_time_ = std::chrono::high_resolution_clock::now();
}

void BenchmarkTimer::stop() {
    stop_time_ = std::chrono::high_resolution_clock::now();
    // Prevent reordering of the timing call's conclusion before any
    // work that conceptually belongs after the measurement window.
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

double BenchmarkTimer::elapsed_ns() const {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time_ - start_time_).count());
}

double BenchmarkTimer::elapsed_ms() const {
    return elapsed_ns() / 1e6;
}
