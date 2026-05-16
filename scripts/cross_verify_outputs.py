#!/usr/bin/env python3
"""
cross_verify_outputs — numerical verification of two OpenVX/OpenCV
output dumps.

Given two directories produced by `<impl>-mark --dump-outputs DIR`,
load every matching kernel pair (matched by `name` in each side's
manifest.json), compute max-abs-diff + mean-abs-diff + PSNR, and
report PASS/FAIL per kernel against a per-kernel-class tolerance.

Why a separate verifier instead of bolting numerical compare into
each binary:

  1. The dumpers (one per binary) ship the raw bytes ONLY — no
     numerical compare logic. The verifier is the single source
     of truth for tolerances, which depend on the PAIR of impls
     (e.g. Sobel border-mode skew between MIVisionX and OpenCV
     is well-known and shouldn't fail the gate at 1 LSB).

  2. Lets us run the verifier in CI against any pair of dumps
     produced by any impl, without rebuilding any binary.

  3. PSNR / max-abs-diff are standard image-quality metrics that
     reviewers can interpret without reading C++ — a high PSNR
     (>40 dB for U8 outputs) means "perceptually identical even
     if not byte-identical", which is the right framing for a
     vision-library cross-check.

Output:

  - A Markdown report on stdout (PASS/FAIL table). Suitable for
    appending straight into a GitHub Actions step summary.
  - Optional --json FILE writes the same data as machine-readable
    JSON for downstream tooling.
  - Process exit code is 0 if every kernel passes its tolerance,
    1 otherwise — so CI can gate on this.

Tolerances (per-kernel, max-abs-diff in raw integer units) live in
TOLERANCES below. They reflect known-acceptable disagreements
between OpenCV and the OpenVX implementations after auditing the
spec for each kernel — primarily border-mode behaviour, rounding
direction in fixed-point ops, and the small ±1 LSB drift from
RGB→Gray colour-matrix arithmetic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Per-kernel comparison rules.
#
#   max_diff    — largest raw-unit disagreement we tolerate
#   border_crop — N pixels stripped off each edge before compare. OpenVX
#                 uses BORDER_UNDEFINED by default which leaves the
#                 outermost ring uninitialised; our OpenCV dumper uses
#                 BORDER_REPLICATE which fills it. The interior is the
#                 part the kernel actually computes the same way on both
#                 sides, so we crop the border before comparing rather
#                 than try to align border behaviour (which the OpenVX
#                 1.x spec doesn't promise anyway).
#
# Comments call out WHY each tolerance is what it is so future readers
# don't loosen them without understanding what they're papering over.
RULES: dict[str, dict[str, int]] = {
    # 3x3 separable box: identical fixed-point divide-by-9 on both sides.
    # After cropping the 1px border ring the only diff is rounding
    # direction (OpenCV adds bias before divide, OpenVX truncates).
    "Box3x3":      {"max_diff": 1, "border_crop": 1},
    "Gaussian3x3": {"max_diff": 1, "border_crop": 1},

    # Selection-based, no rounding — interior must be bitwise identical.
    "Median3x3":   {"max_diff": 0, "border_crop": 1},
    "Erode3x3":    {"max_diff": 0, "border_crop": 1},
    "Dilate3x3":   {"max_diff": 0, "border_crop": 1},

    # Sobel: OpenVX produces vx_int16 raw [-1020, 1020]; OpenCV cv::Sobel
    # ddepth=CV_16S matches that scale. Interior must be bitwise
    # identical after border crop.
    "Sobel3x3_dx": {"max_diff": 0, "border_crop": 1},
    "Sobel3x3_dy": {"max_diff": 0, "border_crop": 1},

    # Saturating add — fully specified pixelwise op, no border, no
    # rounding. Must be bitwise identical.
    "Add_U8_Saturate":         {"max_diff": 0, "border_crop": 0},

    # Bitwise NOT — totally spec'd, no border, no rounding, no colour
    # matrix. Picked specifically as a "if THIS doesn't match, your
    # dumper is broken" sentinel.
    "Not_U8":                  {"max_diff": 0, "border_crop": 0},

    # Channel extract — splat a fixed input channel out. No arithmetic,
    # no border. Same byte-pattern on both sides by construction.
    "ChannelExtract_R":        {"max_diff": 0, "border_crop": 0},

    # _input_u8 — generator parity check. ANY non-zero diff means the
    # mt19937_64 + uniform_int_distribution streams diverged between
    # the binaries; every downstream kernel diff would be meaningless.
    "_input_u8":               {"max_diff": 0, "border_crop": 0},
}

# Back-compat: TOLERANCES kept as a derived view so old callers/tests
# referencing TOLERANCES[name] still work (just returns the max_diff).
TOLERANCES: dict[str, int] = {k: v["max_diff"] for k, v in RULES.items()}

# PSNR floor — any kernel falling below this dB threshold is flagged as
# a perceptual-level divergence even if its max-abs-diff is within
# tolerance. 40 dB is the standard "looks identical" floor; below that
# a human reviewer would actually see the difference.
PSNR_FLOOR_DB = 40.0


def load_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest.json in {directory} — was --dump-outputs run there?"
        )
    with open(manifest_path) as f:
        return json.load(f)


def load_kernel(directory: Path, entry: dict[str, Any]) -> np.ndarray:
    """Load a kernel dump as a tightly-packed numpy array. The dumpers
    write h × w × channels with no stride padding (see openvx_output_dumper
    and cv_output_dumper for the tight-pack guarantee)."""
    dtype_map = {
        "u8": np.uint8, "s8": np.int8,
        "u16": np.uint16, "s16": np.int16,
        "u32": np.uint32, "s32": np.int32,
        "f32": np.float32,
    }
    dtype = dtype_map.get(entry["dtype"])
    if dtype is None:
        raise ValueError(f"Unknown dtype '{entry['dtype']}' in manifest entry "
                         f"for kernel '{entry['name']}'")
    path = directory / entry["file"]
    data = np.fromfile(path, dtype=dtype)
    h, w, c = entry["height"], entry["width"], entry["channels"]
    expected_count = h * w * c
    if data.size != expected_count:
        raise ValueError(
            f"{entry['name']}: file has {data.size} elements, manifest says "
            f"h*w*c = {h}*{w}*{c} = {expected_count}"
        )
    if c == 1:
        return data.reshape(h, w)
    return data.reshape(h, w, c)


def compute_psnr_db(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR for raw integer arrays. We use the dtype's full range as
    the peak signal — for u8 that's 255, for s16 it's 32767. This is
    the standard interpretation that matches OpenCV's `cv::PSNR`.
    Returns +inf when the two arrays are bitwise identical."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    mse = float(np.mean((af - bf) ** 2))
    if mse == 0.0:
        return float("inf")
    if a.dtype == np.uint8 or a.dtype == np.int8:
        peak = 255.0
    elif a.dtype == np.uint16 or a.dtype == np.int16:
        peak = 32767.0
    elif a.dtype == np.uint32 or a.dtype == np.int32:
        peak = 2_147_483_647.0
    else:
        peak = float(np.max(np.abs(af)))
    if peak == 0:
        return float("inf") if mse == 0 else 0.0
    return 10.0 * math.log10((peak * peak) / mse)


def crop_border(a: np.ndarray, n: int) -> np.ndarray:
    """Crop n pixels off each edge. n=0 returns a unchanged. Handles
    both (h, w) and (h, w, c) layouts."""
    if n <= 0:
        return a
    if a.ndim == 2:
        if a.shape[0] <= 2 * n or a.shape[1] <= 2 * n:
            return a  # too small to crop meaningfully; let the diff fail loudly
        return a[n:-n, n:-n]
    if a.ndim == 3:
        if a.shape[0] <= 2 * n or a.shape[1] <= 2 * n:
            return a
        return a[n:-n, n:-n, :]
    return a


def compare_kernel(a: np.ndarray, b: np.ndarray, border_crop: int = 0) -> dict[str, Any]:
    """Compute the diagnostic suite for one kernel pair. We do
    arithmetic in int64 because s16-s16 can overflow int16.

    `border_crop` strips N pixels off each edge before comparison —
    used for spatial filters where the border behaviour differs by
    spec between OpenVX (UNDEFINED) and OpenCV (REPLICATE)."""
    if a.shape != b.shape:
        return {
            "ok": False, "reason": f"shape mismatch {a.shape} vs {b.shape}",
            "max_abs_diff": -1, "mean_abs_diff": -1.0, "psnr_db": -1.0,
            "frac_exact": 0.0, "border_crop": border_crop,
        }
    a_c = crop_border(a, border_crop)
    b_c = crop_border(b, border_crop)
    diff = a_c.astype(np.int64) - b_c.astype(np.int64)
    abs_diff = np.abs(diff)
    max_abs = int(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    frac_exact = float(np.mean(abs_diff == 0))
    return {
        "ok": True, "reason": "",
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "psnr_db": compute_psnr_db(a_c, b_c),
        "frac_exact": frac_exact,
        "border_crop": border_crop,
    }


def verify_pair(left_dir: Path, right_dir: Path) -> tuple[list[dict[str, Any]], bool]:
    """Returns (per-kernel results, overall_pass)."""
    lm = load_manifest(left_dir)
    rm = load_manifest(right_dir)
    left_kernels = {k["name"]: k for k in lm["kernels"]}
    right_kernels = {k["name"]: k for k in rm["kernels"]}

    common = sorted(set(left_kernels) & set(right_kernels))
    only_left = sorted(set(left_kernels) - set(right_kernels))
    only_right = sorted(set(right_kernels) - set(left_kernels))

    results: list[dict[str, Any]] = []
    all_pass = True

    for name in common:
        left = load_kernel(left_dir, left_kernels[name])
        right = load_kernel(right_dir, right_kernels[name])
        rule = RULES.get(name, {"max_diff": 0, "border_crop": 0})
        tolerance = rule["max_diff"]
        cmp = compare_kernel(left, right, border_crop=rule["border_crop"])
        passed = (
            cmp["ok"]
            and cmp["max_abs_diff"] <= tolerance
            and (cmp["psnr_db"] >= PSNR_FLOOR_DB or cmp["max_abs_diff"] == 0)
        )
        if not passed:
            all_pass = False
        # _input_u8 must always be byte-identical — if it's not, every
        # downstream per-kernel diff is meaningless. Surface that
        # specifically.
        if name == "_input_u8" and cmp["max_abs_diff"] != 0:
            all_pass = False
            cmp["reason"] = "INPUT BYTES DIFFER — generator skew, downstream diffs are meaningless"
        results.append({
            "name": name,
            "tolerance": tolerance,
            "passed": passed,
            **cmp,
        })

    # Surface presence-only mismatches (kernel dumped on one side and
    # not the other) so the report shows them even though they can't
    # be numerically compared.
    for name in only_left:
        results.append({"name": name, "passed": False, "tolerance": 0,
                        "max_abs_diff": -1, "mean_abs_diff": -1,
                        "psnr_db": -1, "frac_exact": 0,
                        "reason": f"only in {left_dir.name}"})
        all_pass = False
    for name in only_right:
        results.append({"name": name, "passed": False, "tolerance": 0,
                        "max_abs_diff": -1, "mean_abs_diff": -1,
                        "psnr_db": -1, "frac_exact": 0,
                        "reason": f"only in {right_dir.name}"})
        all_pass = False

    return results, all_pass


def render_markdown(left_label: str, right_label: str,
                    left_dir: Path, right_dir: Path,
                    results: list[dict[str, Any]], all_pass: bool) -> str:
    out: list[str] = []
    out.append(f"## Cross-impl output verification: **{left_label}** vs **{right_label}**\n")
    out.append(f"_Dumps: `{left_dir}` ↔ `{right_dir}`_\n")
    out.append("")
    out.append(f"**Verdict:** {'**PASS**' if all_pass else '**FAIL**'} — "
               f"{sum(1 for r in results if r['passed'])}/{len(results)} kernels within tolerance.\n")
    out.append("")
    out.append("| Kernel | max-abs-diff | mean-abs-diff | PSNR (dB) | exact % | tol | border-crop | status |")
    out.append("|--------|-------------:|--------------:|----------:|--------:|----:|------------:|:------:|")
    for r in results:
        if r["max_abs_diff"] < 0:
            row = (f"| `{r['name']}` | — | — | — | — | — | — | "
                   f"FAIL — {r.get('reason', 'missing')} |")
        else:
            psnr = "∞" if math.isinf(r["psnr_db"]) else f"{r['psnr_db']:.1f}"
            status = "ok" if r["passed"] else "FAIL"
            extra = f" ({r['reason']})" if r.get("reason") else ""
            row = (f"| `{r['name']}` | {r['max_abs_diff']} | "
                   f"{r['mean_abs_diff']:.3f} | {psnr} | "
                   f"{r['frac_exact']*100:.1f} | {r['tolerance']} | "
                   f"{r.get('border_crop', 0)} | "
                   f"{status}{extra} |")
        out.append(row)
    out.append("")
    out.append("**Tolerance philosophy:** `max-abs-diff` is the LARGEST raw-unit "
               "disagreement between corresponding pixels AFTER cropping the "
               "border-ring (where OpenVX `BORDER_UNDEFINED` produces uninitialised "
               "output by spec). Tolerances are tuned per kernel to absorb known-"
               "acceptable spec differences (border modes, fixed-point rounding "
               "direction) — see `RULES` in `scripts/cross_verify_outputs.py` for "
               "the per-kernel rationale.")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left",  type=Path, help="first dump directory")
    parser.add_argument("right", type=Path, help="second dump directory")
    parser.add_argument("--left-label",  default=None,
                        help="label for the first dump (defaults to impl from manifest)")
    parser.add_argument("--right-label", default=None,
                        help="label for the second dump (defaults to impl from manifest)")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write machine-readable results to this path")
    parser.add_argument("--no-gate", action="store_true",
                        help="never exit non-zero (still prints PASS/FAIL)")
    args = parser.parse_args()

    if not args.left.is_dir():
        print(f"ERROR: {args.left} is not a directory", file=sys.stderr)
        return 2
    if not args.right.is_dir():
        print(f"ERROR: {args.right} is not a directory", file=sys.stderr)
        return 2

    lm = load_manifest(args.left)
    rm = load_manifest(args.right)
    left_label = args.left_label or lm.get("impl", args.left.name)
    right_label = args.right_label or rm.get("impl", args.right.name)

    results, all_pass = verify_pair(args.left, args.right)
    md = render_markdown(left_label, right_label,
                         args.left, args.right, results, all_pass)
    print(md)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "left_label": left_label,
                "right_label": right_label,
                "left_dir": str(args.left),
                "right_dir": str(args.right),
                "all_pass": all_pass,
                "results": results,
            }, f, indent=2)

    if args.no_gate:
        return 0
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
