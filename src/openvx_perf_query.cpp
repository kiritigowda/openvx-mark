#include "openvx_perf_query.h"

namespace openvx_perf {

bool queryGraphPerf(vx_graph graph, vx_perf_t& perf) {
    vx_status status = vxQueryGraph(graph, VX_GRAPH_PERFORMANCE, &perf, sizeof(perf));
    return (status == VX_SUCCESS && perf.num > 0);
}

bool queryNodePerf(vx_node node, vx_perf_t& perf) {
    vx_status status = vxQueryNode(node, VX_NODE_PERFORMANCE, &perf, sizeof(perf));
    return (status == VX_SUCCESS && perf.num > 0);
}

} // namespace openvx_perf
