#!/usr/bin/env python3
"""Check an openvx-mark/opencv-mark JSON report for failures.

Returns non-zero if any benchmark result in the report is unsupported
or unverified. Use --allow-feature-set to scope the check to specific
feature sets (e.g. "vision,framework"), or --warn-only to print a
summary without failing.
"""

import argparse
import json
import sys


def load_report(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Verify benchmark report integrity")
    parser.add_argument("report", help="Path to benchmark_results.json")
    parser.add_argument(
        "--allow-feature-set",
        type=str,
        default="",
        help="Comma-separated feature sets to check (default: all)",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Print summary but do not exit with failure",
    )
    args = parser.parse_args()

    report = load_report(args.report)
    results = report.get("results", [])

    allowed_sets = set()
    if args.allow_feature_set:
        allowed_sets = {s.strip() for s in args.allow_feature_set.split(",")}

    unsupported = []
    unverified = []

    for r in results:
        if allowed_sets and r.get("feature_set") not in allowed_sets:
            continue
        if not r.get("supported", True):
            unsupported.append(r)
        elif not r.get("verified", True):
            unverified.append(r)

    total_checked = len(
        [r for r in results if not allowed_sets or r.get("feature_set") in allowed_sets]
    )

    print(
        f"check_report: {total_checked} result(s) checked, "
        f"{len(unsupported)} unsupported, {len(unverified)} unverified"
    )

    for r in unsupported[:5]:
        print(
            f"  UNSUPPORTED: {r.get('name')} @ {r.get('resolution')} "
            f"({r.get('mode')}) — {r.get('skip_reason', 'no reason')}"
        )
    if len(unsupported) > 5:
        print(f"  ... and {len(unsupported) - 5} more unsupported")

    for r in unverified[:5]:
        print(
            f"  UNVERIFIED:  {r.get('name')} @ {r.get('resolution')} "
            f"({r.get('mode')}) — {r.get('skip_reason', 'output verification failed')}"
        )
    if len(unverified) > 5:
        print(f"  ... and {len(unverified) - 5} more unverified")

    if (unsupported or unverified) and not args.warn_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
