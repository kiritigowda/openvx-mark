# Features To Add

## Verification Audit Recommendations

Recommendations from a verification audit of all 52 benchmark verify functions.

## Priority 1: Strengthen Filter Tests with Non-Uniform Input

All 8 filter verify functions use constant-value input (all pixels = 100), making them unable to distinguish a working filter from a simple copy or no-op.

**Affected benchmarks:** Box3x3, Gaussian3x3, Median3x3, Erode3x3, Dilate3x3, Sobel3x3, CustomConvolution, NonLinearFilter

**Recommended fixes:**
- **Box3x3** — Input with a single bright pixel (255) surrounded by zeros. Output center should be ~28 (255/9).
- **Gaussian3x3** — Input with a single bright pixel. Output center should be less than 255 due to Gaussian weighting.
- **Median3x3** — Input with salt-and-pepper noise. Output should be smoother than input.
- **Erode3x3** — Input with an isolated bright pixel in a dark field. Erode should remove it (output = 0 at that position).
- **Dilate3x3** — Input with an isolated dark pixel in a bright field. Dilate should fill it (output = 255 at that position).
- **Sobel3x3** — Input with a horizontal edge (top half = 0, bottom half = 255). Verify dy gradient is non-zero at the edge.
- **CustomConvolution** — Use a non-identity kernel (e.g., edge-detect) and verify output differs from input.
- **NonLinearFilter** — Use a pattern where min/median/max produce distinct, verifiable results.

## Priority 2: Use Non-Identity Geometric Transforms

WarpAffine, WarpPerspective, and Remap all use identity transforms, so output trivially equals input. A copy operation would pass.

**Affected benchmarks:** WarpAffine, WarpPerspective, Remap

**Recommended fixes:**
- **WarpAffine** — Use a known translation (e.g., shift by 10 pixels) and verify the pixel value appears at the expected offset.
- **WarpPerspective** — Use a known simple perspective transform and verify pixel displacement.
- **Remap** — Use a coordinate mapping that flips or shifts the image and verify output positions.

## Priority 3: Verify Feature Detector Output

HarrisCorners, FastCorners, and OpticalFlowPyrLK only check that graph execution succeeds without verifying detected features.

**Affected benchmarks:** HarrisCorners, FastCorners, OpticalFlowPyrLK

**Recommended fixes:**
- **HarrisCorners** — Use a checkerboard or cross pattern with obvious corners. Verify the output array is non-empty.
- **FastCorners** — Same approach. Verify at least one corner is detected on a known pattern.
- **OpticalFlowPyrLK** — Verify that tracked keypoint positions shift in the expected direction between frames.

## Priority 4: Multi-Pixel Sampling for Single-Pixel Checks

Several tests only check a single output pixel. A bug affecting other regions would go undetected.

**Affected benchmarks:** ChannelExtract, ChannelCombine, Phase, ScaleImage_Half, ScaleImage_Double

**Recommended fixes:**
- Sample at least 3-4 positions (e.g., center, corners, mid-edges) to verify the operation is consistent across the image.

## Priority 5: Strengthen Remaining Weak Checks

- **LBP** — Currently only checks `imageNonZero`. Should verify specific LBP pattern values for a known input.
- **EqualizeHist** — Currently checks all pixels are equal +/-1. Could additionally verify the output value matches the expected equalized level for uniform input (should map to ~128 for full-range equalization).

## Comparison Report Enhancements

Features implemented in the polished comparison report (both C++ `--compare` and Python `compare_reports.py`):

### Implemented

- **System info section** — Shows CPU, cores, RAM, OS. Detects same vs different hardware with a mismatch warning.
- **Conformance & Scores table** — Side-by-side Vision Score (geometric mean MP/s), conformance PASS/FAIL with kernel counts.
- **Category sub-scores** — Per-category geometric mean comparison with % change column.
- **Summary with per-category breakdown** — Regression/improvement/unchanged counts, broken down by category (e.g., "3 regressions in filters").
- **Detailed results with MP/s** — Both median latency (ms) and throughput (MP/s) for each implementation, plus change % and status.
- **Benchmarks only in one report** — Lists benchmarks present in one file but not the other, so nothing is silently dropped.
- **Stability caveat flags** — Marks rows where either side had CV% > 15%, with a footnote explaining unreliable comparisons.
- **CSV output from C++** — Generates both `.md` and `.csv` from the C++ `--compare` path.
- **Vision Score from JSON** — Python script reads precomputed `overall_vision_score` from JSON instead of incorrectly summing MP/s.
- **Missing kernels detail** — Shows missing kernel lists side by side when conformance differs.

### Future Enhancements

- **Configurable regression threshold** — The 5% threshold for regression/improvement is hardcoded. Add a `--threshold` CLI option to both C++ and Python.
- **Statistical significance testing** — When iterations > 1, perform confidence interval or t-test analysis to determine if differences are statistically meaningful.
- **Multi-resolution scaling comparison** — Compare scaling efficiency between implementations (how well each handles higher resolutions).
- **Chart/graph output** — Generate bar charts or SVG visualizations for throughput comparison.
- **N-way comparison** — Support comparing 3+ implementations in a single report (currently optimized for pairwise).
- **Grouped-by-category view** — Option to group the detailed results table by category instead of sorting by change %.
- **Historical trend tracking** — Compare against a series of reports over time to detect gradual regressions.
