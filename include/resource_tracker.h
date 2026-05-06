#ifndef RESOURCE_TRACKER_H
#define RESOURCE_TRACKER_H

#include <VX/vx.h>
#include <functional>
#include <vector>

// RAII wrapper that tracks OpenVX resources and releases them in reverse order
class ResourceTracker {
public:
    ~ResourceTracker() { releaseAll(); }

    // Track a generic release function
    void track(std::function<void()> release_fn) {
        release_fns_.push_back(std::move(release_fn));
    }

    // Convenience: track an image
    vx_image trackImage(vx_image img) {
        release_fns_.push_back([img]() mutable { vxReleaseImage(&img); });
        return img;
    }

    vx_graph trackGraph(vx_graph g) {
        release_fns_.push_back([g]() mutable { vxReleaseGraph(&g); });
        return g;
    }

    vx_node trackNode(vx_node n) {
        release_fns_.push_back([n]() mutable { vxReleaseNode(&n); });
        return n;
    }

    vx_scalar trackScalar(vx_scalar s) {
        release_fns_.push_back([s]() mutable { vxReleaseScalar(&s); });
        return s;
    }

    vx_threshold trackThreshold(vx_threshold t) {
        release_fns_.push_back([t]() mutable { vxReleaseThreshold(&t); });
        return t;
    }

    vx_matrix trackMatrix(vx_matrix m) {
        release_fns_.push_back([m]() mutable { vxReleaseMatrix(&m); });
        return m;
    }

    vx_convolution trackConvolution(vx_convolution c) {
        release_fns_.push_back([c]() mutable { vxReleaseConvolution(&c); });
        return c;
    }

    vx_pyramid trackPyramid(vx_pyramid p) {
        release_fns_.push_back([p]() mutable { vxReleasePyramid(&p); });
        return p;
    }

    vx_remap trackRemap(vx_remap r) {
        release_fns_.push_back([r]() mutable { vxReleaseRemap(&r); });
        return r;
    }

    vx_distribution trackDistribution(vx_distribution d) {
        release_fns_.push_back([d]() mutable { vxReleaseDistribution(&d); });
        return d;
    }

    vx_lut trackLUT(vx_lut l) {
        release_fns_.push_back([l]() mutable { vxReleaseLUT(&l); });
        return l;
    }

    vx_array trackArray(vx_array a) {
        release_fns_.push_back([a]() mutable { vxReleaseArray(&a); });
        return a;
    }

    vx_tensor trackTensor(vx_tensor t) {
        release_fns_.push_back([t]() mutable { vxReleaseTensor(&t); });
        return t;
    }

    void releaseAll() {
        for (auto it = release_fns_.rbegin(); it != release_fns_.rend(); ++it) {
            (*it)();
        }
        release_fns_.clear();
    }

private:
    std::vector<std::function<void()>> release_fns_;
};

#endif // RESOURCE_TRACKER_H
