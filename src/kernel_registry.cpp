#include "kernel_registry.h"
#include "openvx_version.h"

void KernelRegistry::registerKernel(vx_enum e, const std::string& name,
                                    const std::string& display, const std::string& cat,
                                    const std::string& feature_set) {
    kernels_[e] = {e, name, display, cat, feature_set, false};
}

void KernelRegistry::initCatalog() {
    // Pixel-wise — Vision feature set
    registerKernel(VX_KERNEL_AND, "And", "Bitwise AND", "pixelwise", "vision");
    registerKernel(VX_KERNEL_OR, "Or", "Bitwise OR", "pixelwise", "vision");
    registerKernel(VX_KERNEL_XOR, "Xor", "Bitwise XOR", "pixelwise", "vision");
    registerKernel(VX_KERNEL_NOT, "Not", "Bitwise NOT", "pixelwise", "vision");
    registerKernel(VX_KERNEL_ABSDIFF, "AbsDiff", "Absolute Difference", "pixelwise", "vision");
    registerKernel(VX_KERNEL_ADD, "Add", "Addition", "pixelwise", "vision");
    registerKernel(VX_KERNEL_SUBTRACT, "Subtract", "Subtraction", "pixelwise", "vision");
    registerKernel(VX_KERNEL_MULTIPLY, "Multiply", "Pixelwise Multiply", "pixelwise", "vision");
    // Pixel-wise — Enhanced Vision feature set
#if OPENVX_HAS_1_3
    registerKernel(VX_KERNEL_MIN, "Min", "Pixel-wise Minimum", "pixelwise", "enhanced_vision");
    registerKernel(VX_KERNEL_MAX, "Max", "Pixel-wise Maximum", "pixelwise", "enhanced_vision");
#endif
#if OPENVX_HAS_1_2
    registerKernel(VX_KERNEL_COPY, "Copy", "Data Object Copy", "pixelwise", "enhanced_vision");
#endif

    // Filters — Vision feature set
    registerKernel(VX_KERNEL_BOX_3x3, "Box3x3", "Box Filter 3x3", "filters", "vision");
    registerKernel(VX_KERNEL_GAUSSIAN_3x3, "Gaussian3x3", "Gaussian Filter 3x3", "filters", "vision");
    registerKernel(VX_KERNEL_MEDIAN_3x3, "Median3x3", "Median Filter 3x3", "filters", "vision");
    registerKernel(VX_KERNEL_ERODE_3x3, "Erode3x3", "Erode 3x3", "filters", "vision");
    registerKernel(VX_KERNEL_DILATE_3x3, "Dilate3x3", "Dilate 3x3", "filters", "vision");
    registerKernel(VX_KERNEL_SOBEL_3x3, "Sobel3x3", "Sobel 3x3", "filters", "vision");
    registerKernel(VX_KERNEL_CUSTOM_CONVOLUTION, "CustomConvolution", "Custom Convolution", "filters", "vision");
#if OPENVX_HAS_1_1
    registerKernel(VX_KERNEL_NON_LINEAR_FILTER, "NonLinearFilter", "Non-linear Filter", "filters", "vision");
#endif

    // Color — Vision feature set
    registerKernel(VX_KERNEL_COLOR_CONVERT, "ColorConvert", "Color Space Conversion", "color", "vision");
    registerKernel(VX_KERNEL_CHANNEL_EXTRACT, "ChannelExtract", "Channel Extraction", "color", "vision");
    registerKernel(VX_KERNEL_CHANNEL_COMBINE, "ChannelCombine", "Channel Combine", "color", "vision");
    registerKernel(VX_KERNEL_CONVERTDEPTH, "ConvertDepth", "Bit-depth Conversion", "color", "vision");

    // Geometric — Vision feature set
    registerKernel(VX_KERNEL_SCALE_IMAGE, "ScaleImage", "Scale Image", "geometric", "vision");
    registerKernel(VX_KERNEL_WARP_AFFINE, "WarpAffine", "Warp Affine", "geometric", "vision");
    registerKernel(VX_KERNEL_WARP_PERSPECTIVE, "WarpPerspective", "Warp Perspective", "geometric", "vision");
    registerKernel(VX_KERNEL_REMAP, "Remap", "Remap", "geometric", "vision");

    // Statistical — Vision feature set
    registerKernel(VX_KERNEL_HISTOGRAM, "Histogram", "Histogram", "statistical", "vision");
    registerKernel(VX_KERNEL_EQUALIZE_HISTOGRAM, "EqualizeHist", "Histogram Equalization", "statistical", "vision");
    registerKernel(VX_KERNEL_MEAN_STDDEV, "MeanStdDev", "Mean and Std Dev", "statistical", "vision");
    registerKernel(VX_KERNEL_MINMAXLOC, "MinMaxLoc", "Min/Max Location", "statistical", "vision");
    registerKernel(VX_KERNEL_INTEGRAL_IMAGE, "IntegralImage", "Integral Image", "statistical", "vision");

    // Multi-scale — Vision feature set
    registerKernel(VX_KERNEL_GAUSSIAN_PYRAMID, "GaussianPyramid", "Gaussian Pyramid", "multiscale", "vision");
#if OPENVX_HAS_1_1
    registerKernel(VX_KERNEL_LAPLACIAN_PYRAMID, "LaplacianPyramid", "Laplacian Pyramid", "multiscale", "vision");
#endif
    registerKernel(VX_KERNEL_HALFSCALE_GAUSSIAN, "HalfScaleGaussian", "Half-Scale Gaussian", "multiscale", "vision");

    // Feature detection — Vision feature set
    registerKernel(VX_KERNEL_CANNY_EDGE_DETECTOR, "CannyEdgeDetector", "Canny Edge Detector", "feature", "vision");
    registerKernel(VX_KERNEL_HARRIS_CORNERS, "HarrisCorners", "Harris Corners", "feature", "vision");
    registerKernel(VX_KERNEL_FAST_CORNERS, "FastCorners", "FAST Corners", "feature", "vision");
    registerKernel(VX_KERNEL_OPTICAL_FLOW_PYR_LK, "OpticalFlowPyrLK", "Optical Flow Pyramid LK", "feature", "vision");

    // Feature extraction — Enhanced Vision feature set (OpenVX 1.2+)
#if OPENVX_HAS_1_2
    registerKernel(VX_KERNEL_MATCH_TEMPLATE, "MatchTemplate", "Match Template", "extraction", "enhanced_vision");
    registerKernel(VX_KERNEL_LBP, "LBP", "Local Binary Pattern", "extraction", "enhanced_vision");
    registerKernel(VX_KERNEL_HOG_CELLS, "HOGCells", "HOG Cells", "extraction", "enhanced_vision");
    registerKernel(VX_KERNEL_HOG_FEATURES, "HOGFeatures", "HOG Features", "extraction", "enhanced_vision");
    registerKernel(VX_KERNEL_HOUGH_LINES_P, "HoughLinesP", "Hough Lines Probability", "extraction", "enhanced_vision");
    registerKernel(VX_KERNEL_NON_MAX_SUPPRESSION, "NonMaxSuppression", "Non-Max Suppression", "extraction", "enhanced_vision");

    // Tensor — Enhanced Vision feature set (OpenVX 1.2+)
    registerKernel(VX_KERNEL_TENSOR_ADD, "TensorAdd", "Tensor Add", "tensor", "enhanced_vision");
    registerKernel(VX_KERNEL_TENSOR_SUBTRACT, "TensorSub", "Tensor Subtract", "tensor", "enhanced_vision");
    registerKernel(VX_KERNEL_TENSOR_MULTIPLY, "TensorMul", "Tensor Multiply", "tensor", "enhanced_vision");
    registerKernel(VX_KERNEL_TENSOR_TRANSPOSE, "TensorTranspose", "Tensor Transpose", "tensor", "enhanced_vision");
    registerKernel(VX_KERNEL_TENSOR_CONVERT_DEPTH, "TensorConvertDepth", "Tensor Convert Depth", "tensor", "enhanced_vision");
    registerKernel(VX_KERNEL_TENSOR_MATRIX_MULTIPLY, "TensorMatMul", "Tensor Matrix Multiply", "tensor", "enhanced_vision");
    registerKernel(VX_KERNEL_TENSOR_TABLE_LOOKUP, "TensorTableLookup", "Tensor Table Lookup", "tensor", "enhanced_vision");
#endif

    // Misc — Vision feature set
    registerKernel(VX_KERNEL_MAGNITUDE, "Magnitude", "Magnitude", "misc", "vision");
    registerKernel(VX_KERNEL_PHASE, "Phase", "Phase", "misc", "vision");
    registerKernel(VX_KERNEL_TABLE_LOOKUP, "TableLookup", "Table Lookup", "misc", "vision");
    registerKernel(VX_KERNEL_THRESHOLD, "Threshold", "Threshold", "misc", "vision");
#if OPENVX_HAS_1_3
    registerKernel(VX_KERNEL_WEIGHTED_AVERAGE, "WeightedAverage", "Weighted Average", "misc", "vision");
#endif
    // Misc — Enhanced Vision feature set (OpenVX 1.2+)
#if OPENVX_HAS_1_2
    registerKernel(VX_KERNEL_BILATERAL_FILTER, "BilateralFilter", "Bilateral Filter", "misc", "enhanced_vision");
    registerKernel(VX_KERNEL_SELECT, "Select", "Select", "misc", "enhanced_vision");
    registerKernel(VX_KERNEL_SCALAR_OPERATION, "ScalarOperation", "Scalar Operation", "misc", "enhanced_vision");
#endif
}

void KernelRegistry::probe(vx_context context) {
    initCatalog();

    for (auto& [e, info] : kernels_) {
        vx_kernel kernel = vxGetKernelByEnum(context, e);
        if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS) {
            info.available = true;
            vxReleaseKernel(&kernel);
        }
    }
}

bool KernelRegistry::isAvailable(vx_enum kernel_enum) const {
    auto it = kernels_.find(kernel_enum);
    return (it != kernels_.end()) && it->second.available;
}

bool KernelRegistry::allAvailable(const std::vector<vx_enum>& enums) const {
    for (vx_enum e : enums) {
        if (!isAvailable(e)) return false;
    }
    return true;
}

const KernelInfo* KernelRegistry::getInfo(vx_enum kernel_enum) const {
    auto it = kernels_.find(kernel_enum);
    return (it != kernels_.end()) ? &it->second : nullptr;
}

int KernelRegistry::availableCount() const {
    int count = 0;
    for (const auto& [_, info] : kernels_) {
        if (info.available) count++;
    }
    return count;
}

int KernelRegistry::totalCount() const {
    return static_cast<int>(kernels_.size());
}

std::vector<KernelRegistry::CategorySummary> KernelRegistry::categorySummary() const {
    std::map<std::string, std::pair<int, int>> cat_map;  // available, total
    for (const auto& [_, info] : kernels_) {
        auto& p = cat_map[info.category];
        p.second++;
        if (info.available) p.first++;
    }

    std::vector<CategorySummary> result;
    for (const auto& [cat, counts] : cat_map) {
        result.push_back({cat, counts.first, counts.second});
    }
    return result;
}

std::vector<KernelRegistry::FeatureSetSummary> KernelRegistry::featureSetSummary() const {
    std::map<std::string, std::pair<int, int>> fs_map;  // available, total
    for (const auto& [_, info] : kernels_) {
        auto& p = fs_map[info.feature_set];
        p.second++;
        if (info.available) p.first++;
    }

    std::vector<FeatureSetSummary> result;
    for (const auto& [fs, counts] : fs_map) {
        result.push_back({fs, counts.first, counts.second});
    }
    return result;
}
