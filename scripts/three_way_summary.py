#!/usr/bin/env python3
################################################################################
#
# MIT License
#
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
"""
N-way side-by-side benchmark summary.

The existing `compare_reports.py` is a rich 2-way comparator (scores,
category breakdowns, conformance, win/loss counts, etc.) but doesn't
generalise to N implementations. This script handles the N≥3 case in
a more focused way: one row per `(name, mode, resolution)` join key,
one column per implementation, showing `median_ms` and `MP/s`. That's
the table users actually want when they ask "rustVX vs MIVisionX vs
OpenCV on my hardware".

Usage:

    python3 scripts/three_way_summary.py \\
        --label "AMD MIVisionX" amd.json \\
        --label "rustVX"        rustvx.json \\
        --label "OpenCV"        opencv.json \\
        --output build/three_way

Generates:

    build/three_way.md        # combined markdown table
    build/three_way.csv       # combined CSV (one column per impl × {ms, MP/s})

A row is marked `N/A` for any implementation that didn't produce a
result for that (kernel, mode, resolution) — e.g. AMD MIVisionX
typically shows N/A on enhanced_vision rows because its libopenvx
runtime doesn't actually implement those kernels even though the
headers declare them.
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def load(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_index(report):
    """Build a dict keyed by (name, mode, resolution) → {ms, mps, ok}."""
    idx = {}
    for r in report.get('results', []):
        key = (r['name'], r['mode'], r['resolution'])
        ms = r.get('wall_clock', {}).get('median_ns', 0) / 1e6
        mps = r.get('megapixels_per_sec', 0.0)
        verified = r.get('verified', True) and r.get('supported', True)
        idx[key] = {
            'ms': ms,
            'mps': mps,
            'verified': verified,
            'feature_set': r.get('feature_set', 'vision'),
            'category': r.get('category', ''),
        }
    return idx


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--label', action='append', default=[],
                   help='Display label for the next --report argument')
    p.add_argument('reports', nargs='+',
                   help='Benchmark JSON files (one per implementation)')
    p.add_argument('--output', default='three_way_summary',
                   help='Output base path (writes .md and .csv)')
    args = p.parse_args()

    # If --label was passed, use those names; otherwise derive from each
    # report's `openvx.implementation` field, falling back to filename.
    labels = list(args.label)
    while len(labels) < len(args.reports):
        # Pad with auto-derived names
        report = load(args.reports[len(labels)])
        name = (report.get('openvx', {}).get('implementation')
                or os.path.basename(args.reports[len(labels)]))
        labels.append(name)
    labels = labels[:len(args.reports)]

    reports = [load(p) for p in args.reports]
    indexes = [build_index(r) for r in reports]

    # Union of all keys across all reports.
    all_keys = set()
    for idx in indexes:
        all_keys.update(idx.keys())
    all_keys = sorted(all_keys)

    # Group by feature_set for nicer markdown sectioning. Take the
    # feature_set from whichever report has the key (they should agree
    # but we don't enforce it).
    def key_feature_set(k):
        for idx in indexes:
            if k in idx:
                return idx[k]['feature_set']
        return 'unknown'

    keys_by_fs = defaultdict(list)
    for k in all_keys:
        keys_by_fs[key_feature_set(k)].append(k)

    # --------------------------------------------------------------
    # Markdown report
    # --------------------------------------------------------------
    md_path = args.output + '.md'
    with open(md_path, 'w') as f:
        f.write('# N-way OpenVX Benchmark Summary\n\n')
        f.write('Implementations compared: ' + ', '.join(f'**{l}**' for l in labels) + '\n\n')
        f.write(f'Total benchmark keys joined: **{len(all_keys)}**\n\n')

        # Per-impl coverage line.
        f.write('## Coverage\n\n')
        f.write('| Implementation | Verified results | Skipped / N/A |\n')
        f.write('|:---|---:|---:|\n')
        for label, idx in zip(labels, indexes):
            verified = sum(1 for k in all_keys if k in idx and idx[k]['verified'])
            na = len(all_keys) - verified
            f.write(f'| {label} | {verified} | {na} |\n')
        f.write('\n')

        # Per-feature-set sections.
        for fs in ('vision', 'enhanced_vision', 'framework'):
            keys = keys_by_fs.get(fs, [])
            if not keys:
                continue
            f.write(f'## {fs}\n\n')
            f.write('| Benchmark | Mode | Resolution |')
            for l in labels:
                f.write(f' {l} ms | {l} MP/s |')
            f.write('\n|:---|:---|:---|' + ' ---: | ---: |' * len(labels) + '\n')

            for key in keys:
                name, mode, res = key
                f.write(f'| {name} | {mode} | {res} |')
                for idx in indexes:
                    if key in idx and idx[key]['verified']:
                        f.write(f' {idx[key]["ms"]:.3f} | {idx[key]["mps"]:.1f} |')
                    else:
                        f.write(' N/A | N/A |')
                f.write('\n')
            f.write('\n')

        # Other (anything that didn't fall into the three known FS).
        keys = []
        for fs, ks in keys_by_fs.items():
            if fs not in ('vision', 'enhanced_vision', 'framework'):
                keys.extend(ks)
        if keys:
            f.write('## other\n\n')
            f.write('| Benchmark | Mode | Resolution |')
            for l in labels:
                f.write(f' {l} ms | {l} MP/s |')
            f.write('\n|:---|:---|:---|' + ' ---: | ---: |' * len(labels) + '\n')
            for key in sorted(keys):
                name, mode, res = key
                f.write(f'| {name} | {mode} | {res} |')
                for idx in indexes:
                    if key in idx and idx[key]['verified']:
                        f.write(f' {idx[key]["ms"]:.3f} | {idx[key]["mps"]:.1f} |')
                    else:
                        f.write(' N/A | N/A |')
                f.write('\n')
            f.write('\n')

    # --------------------------------------------------------------
    # CSV
    # --------------------------------------------------------------
    csv_path = args.output + '.csv'
    with open(csv_path, 'w') as f:
        header = ['name', 'mode', 'resolution', 'feature_set', 'category']
        for l in labels:
            header.append(f'{l}_ms')
            header.append(f'{l}_mps')
            header.append(f'{l}_verified')
        f.write(','.join(header) + '\n')

        for key in all_keys:
            name, mode, res = key
            fs = key_feature_set(key)
            cat = next((idx[key]['category'] for idx in indexes if key in idx), '')
            row = [name, mode, res, fs, cat]
            for idx in indexes:
                if key in idx:
                    row.append(f'{idx[key]["ms"]:.6f}')
                    row.append(f'{idx[key]["mps"]:.3f}')
                    row.append('1' if idx[key]['verified'] else '0')
                else:
                    row.extend(['', '', '0'])
            f.write(','.join(str(c) for c in row) + '\n')

    # --------------------------------------------------------------
    # Console summary
    # --------------------------------------------------------------
    print(f'Wrote {md_path}')
    print(f'Wrote {csv_path}')
    print(f'Joined {len(all_keys)} unique (name, mode, resolution) keys across '
          f'{len(labels)} implementations.')


if __name__ == '__main__':
    sys.exit(main() or 0)
