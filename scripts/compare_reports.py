#!/usr/bin/env python3
################################################################################
#
# MIT License
#
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc.
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
Cross-vendor OpenVX benchmark comparison tool.

Usage:
    python3 compare_reports.py report_a.json report_b.json [--output comparison]

Generates a markdown comparison table and CSV from two or more benchmark JSON reports.
"""

import argparse
import json
import os
import sys


def load_report(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_impl_name(report, path='unknown'):
    return report.get('openvx', {}).get('implementation', os.path.basename(path))


def build_result_map(report):
    """Build a dict keyed by (name, mode, resolution) -> result"""
    result_map = {}
    for r in report.get('results', []):
        key = (r['name'], r['mode'], r['resolution'])
        result_map[key] = r
    return result_map


def compare(reports, paths):
    impl_names = []
    result_maps = []

    for i, report in enumerate(reports):
        name = report.get('openvx', {}).get('implementation', os.path.basename(paths[i]))
        impl_names.append(name)
        result_maps.append(build_result_map(report))

    # Collect all unique benchmark keys
    all_keys = set()
    for rm in result_maps:
        all_keys.update(rm.keys())

    all_keys = sorted(all_keys)

    return impl_names, result_maps, all_keys


def write_markdown(impl_names, result_maps, all_keys, output_path):
    with open(output_path + '.md', 'w') as f:
        f.write('# OpenVX Benchmark Comparison\n\n')

        # System info table
        f.write('## Implementations\n\n')
        f.write('| # | Implementation |\n')
        f.write('|---|---|\n')
        for i, name in enumerate(impl_names):
            f.write(f'| {i+1} | {name} |\n')
        f.write('\n')

        # Results table
        header = '| Benchmark | Mode | Resolution |'
        separator = '|:---|:---|:---|'
        for i, name in enumerate(impl_names):
            short = name[:20] if len(name) > 20 else name
            header += f' {short} (ms) | {short} (MP/s) |'
            separator += '---:|---:|'

        header += ' Speedup |'
        separator += '---:|'

        f.write('## Results\n\n')
        f.write(header + '\n')
        f.write(separator + '\n')

        for key in all_keys:
            name, mode, resolution = key
            row = f'| {name} | {mode} | {resolution} |'

            medians = []
            for rm in result_maps:
                r = rm.get(key)
                if r and r.get('supported', False) and r.get('verified', True):
                    wc = r.get('wall_clock', {})
                    median = wc.get('median_ms', 0)
                    mps = r.get('megapixels_per_sec', 0)
                    row += f' {median:.3f} | {mps:.1f} |'
                    medians.append(median)
                else:
                    row += ' N/A | N/A |'
                    medians.append(None)

            # Speedup (first vs second, if both available)
            if len(medians) >= 2 and medians[0] and medians[1] and medians[1] > 0:
                speedup = medians[0] / medians[1]
                row += f' {speedup:.2f}x |'
            else:
                row += ' N/A |'

            f.write(row + '\n')

    print(f'  Comparison markdown: {output_path}.md')


def write_csv(impl_names, result_maps, all_keys, output_path):
    with open(output_path + '.csv', 'w') as f:
        header = 'benchmark,mode,resolution'
        for name in impl_names:
            header += f',{name}_median_ms,{name}_mp_per_sec'
        header += ',speedup'
        f.write(header + '\n')

        for key in all_keys:
            name, mode, resolution = key
            row = f'{name},{mode},{resolution}'

            medians = []
            for rm in result_maps:
                r = rm.get(key)
                if r and r.get('supported', False) and r.get('verified', True):
                    wc = r.get('wall_clock', {})
                    median = wc.get('median_ms', 0)
                    mps = r.get('megapixels_per_sec', 0)
                    row += f',{median:.4f},{mps:.2f}'
                    medians.append(median)
                else:
                    row += ',,'
                    medians.append(None)

            if len(medians) >= 2 and medians[0] and medians[1] and medians[1] > 0:
                row += f',{medians[0]/medians[1]:.4f}'
            else:
                row += ','

            f.write(row + '\n')

    print(f'  Comparison CSV: {output_path}.csv')


def main():
    parser = argparse.ArgumentParser(description='Compare OpenVX benchmark reports')
    parser.add_argument('reports', nargs='+', help='JSON report files to compare')
    parser.add_argument('--output', default='comparison', help='Output file prefix')
    args = parser.parse_args()

    if len(args.reports) < 2:
        print('ERROR: Need at least 2 report files to compare')
        sys.exit(1)

    reports = []
    for path in args.reports:
        if not os.path.exists(path):
            print(f'ERROR: File not found: {path}')
            sys.exit(1)
        reports.append(load_report(path))

    impl_names, result_maps, all_keys = compare(reports, args.reports)

    write_markdown(impl_names, result_maps, all_keys, args.output)
    write_csv(impl_names, result_maps, all_keys, args.output)

    print(f'\nCompared {len(args.reports)} implementations across {len(all_keys)} benchmarks')


if __name__ == '__main__':
    main()
