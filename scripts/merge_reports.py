#!/usr/bin/env python3
"""
merge_reports.py — merge multiple openvx-mark / opencv-mark JSON
benchmark reports produced by separate invocations into a single
JSON file whose schema matches the original output format.

WHY THIS EXISTS
---------------
openvx-mark writes its JSON report ONLY at the end of a successful
benchmark run. If the linked OpenVX implementation crashes mid-run
(SIGSEGV, hang, abort, etc.) we lose ALL the data for the benchmarks
that hadn't been measured yet. The classic case is the Khronos sample
implementation's enhanced_vision tensor kernels — its TensorAdd /
TensorSub / TensorMul kernels have known buggy implementations that
SIGSEGV the moment we invoke `vxProcessGraph`, taking the whole bench
process down with them.

Workaround: split a single bench invocation into N invocations along
feature-set lines (e.g. `vision,framework` first, then `enhanced_vision`
second), each writing to its own output directory. If the
enhanced_vision invocation crashes, the vision+framework JSON is
already on disk and the comparison report can be built from it.

This script merges the JSONs from those split invocations back into
one report whose schema matches what a single full invocation would
have produced. The downstream comparison + summary scripts
(compare_reports.py, three_way_summary.py, ci_pairwise_summary.py)
then operate on the merged JSON without needing any awareness of the
split.

MERGE SEMANTICS
---------------
  * `results`                 : concat from every input (this is the
                                per-bench measurements array — the
                                main signal we care about).
  * `feature_set_availability`,
    `kernel_availability`,
    `conformance`             : per-key union from every input. When a
                                key (e.g. "vision") exists in only
                                one input, take that value; when in
                                multiple, take the most-permissive
                                value (true > false).
  * `scores`, `scaling_analysis`
                              : pick from whichever input ran the
                                relevant feature set. We prefer the
                                LAST input that produced a non-empty
                                value (CI orders feature-set runs
                                most-likely-to-succeed first so the
                                last successful one wins).
  * everything else (`system`, `openvx`, `benchmark`, `build`,
    `threading`, `timing_audit`, `config`)
                              : taken from the first input, since
                                these describe the test environment
                                and don't differ between split runs
                                of the same binary.

USAGE
-----
  scripts/merge_reports.py \
      results-vision/benchmark_results.json \
      results-enhanced/benchmark_results.json \
      --output merged/benchmark_results.json

Any input that does not exist (e.g. because that invocation crashed
before writing its JSON) is silently skipped. As long as at least
ONE input exists, the merge produces a valid output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_optional(path: Path) -> Optional[Dict[str, Any]]:
    """Return parsed JSON or None if file missing/empty/malformed.

    Empty/missing inputs are valid here — they mean "this invocation
    crashed before writing", which is the case we exist to handle.
    """
    if not path.exists():
        return None
    try:
        text = path.read_text()
        if not text.strip():
            return None
        return json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: skipping {path}: {exc}", file=sys.stderr)
        return None


def _merge_bool_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two {string: bool} maps (or nested maps of bools).

    Resolution rule: for any key present in both, OR the values so the
    merged map reflects "this feature is available in AT LEAST one of
    the invocations". For nested dicts, recurse.
    """
    out = dict(a)
    for k, v in b.items():
        if k not in out:
            out[k] = v
        elif isinstance(v, dict) and isinstance(out[k], dict):
            out[k] = _merge_bool_dict(out[k], v)
        elif isinstance(v, bool) and isinstance(out[k], bool):
            out[k] = out[k] or v
        else:
            # Last-write-wins for non-bool/non-dict values; this
            # branch is reached only for unexpected schema drift.
            out[k] = v
    return out


def merge(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge N benchmark JSON reports into one."""
    assert reports, "merge() requires at least one report"

    # Start from the first report — its `system`, `openvx`,
    # `benchmark`, `build`, `threading`, `timing_audit`, `config`
    # blocks describe the test environment and are taken verbatim.
    merged: Dict[str, Any] = dict(reports[0])

    # Concat the `results` arrays from every input (this is the main
    # signal — per-bench measurements).
    all_results: List[Any] = []
    for r in reports:
        all_results.extend(r.get("results", []))
    merged["results"] = all_results

    # Merge feature-set/kernel availability + conformance via union.
    for key in ("feature_set_availability", "kernel_availability", "conformance"):
        merged_val: Dict[str, Any] = {}
        for r in reports:
            v = r.get(key)
            if isinstance(v, dict):
                merged_val = _merge_bool_dict(merged_val, v)
        if merged_val:
            merged[key] = merged_val

    # Scores + scaling_analysis: prefer the LAST input that produced a
    # non-empty value (most-recent-successful-run wins).
    for key in ("scores", "scaling_analysis"):
        for r in reversed(reports):
            v = r.get(key)
            if v:
                merged[key] = v
                break

    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="One or more openvx-mark JSON report files. "
                             "Files that don't exist are silently skipped "
                             "(useful when an invocation crashed before "
                             "writing).")
    parser.add_argument("--output", "-o", required=True, type=Path,
                        help="Path to write the merged JSON.")
    args = parser.parse_args()

    reports = [r for r in (_load_optional(p) for p in args.inputs) if r]
    if not reports:
        print("ERROR: no valid input reports — every input file was "
              "missing/empty/malformed.", file=sys.stderr)
        return 1

    merged = merge(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2))
    print(f"merged {len(reports)} report(s) into {args.output} "
          f"({len(merged.get('results', []))} total benchmark rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
