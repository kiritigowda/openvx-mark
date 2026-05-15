#ifndef OPENVX_PERF_QUERY_H
#define OPENVX_PERF_QUERY_H

// OpenVX-specific helpers for reading vx_perf_t timing off graphs and
// nodes. Split out of benchmark_timer.h so the wall-clock timer in the
// shared `bench_core` library doesn't drag in <VX/vx.h>, which would
// force every implementation-mark companion (opencv-mark, future
// halide-mark, etc.) to depend on OpenVX headers transitively.

#include <VX/vx.h>

namespace openvx_perf {

bool queryGraphPerf(vx_graph graph, vx_perf_t& perf);
bool queryNodePerf(vx_node node, vx_perf_t& perf);

} // namespace openvx_perf

#endif // OPENVX_PERF_QUERY_H
