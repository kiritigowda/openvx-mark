#include "benchmark_timer.h"

void BenchmarkTimer::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
}

void BenchmarkTimer::stop() {
    stop_time_ = std::chrono::high_resolution_clock::now();
}

double BenchmarkTimer::elapsed_ns() const {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time_ - start_time_).count());
}

double BenchmarkTimer::elapsed_ms() const {
    return elapsed_ns() / 1e6;
}
