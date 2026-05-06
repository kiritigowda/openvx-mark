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

bool BenchmarkTimer::queryGraphPerf(vx_graph graph, vx_perf_t& perf) {
    vx_status status = vxQueryGraph(graph, VX_GRAPH_PERFORMANCE, &perf, sizeof(perf));
    return (status == VX_SUCCESS && perf.num > 0);
}

bool BenchmarkTimer::queryNodePerf(vx_node node, vx_perf_t& perf) {
    vx_status status = vxQueryNode(node, VX_NODE_PERFORMANCE, &perf, sizeof(perf));
    return (status == VX_SUCCESS && perf.num > 0);
}
