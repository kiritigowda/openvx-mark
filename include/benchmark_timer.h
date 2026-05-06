#ifndef BENCHMARK_TIMER_H
#define BENCHMARK_TIMER_H

#include <VX/vx.h>
#include <chrono>
#include <cstdint>

class BenchmarkTimer {
public:
    void start();
    void stop();
    double elapsed_ns() const;
    double elapsed_ms() const;

    // Query OpenVX performance from a graph
    static bool queryGraphPerf(vx_graph graph, vx_perf_t& perf);
    // Query OpenVX performance from a node
    static bool queryNodePerf(vx_node node, vx_perf_t& perf);

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point stop_time_;
};

#endif // BENCHMARK_TIMER_H
