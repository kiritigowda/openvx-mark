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


def get_system_info(report):
    """Extract system info dict from a report."""
    return report.get('system', {})


def format_ram(ram_value):
    """Format RAM value from report (may be gb float or bytes int)."""
    if isinstance(ram_value, (int,)) and ram_value > 1e9:
        return f'{ram_value / (1024**3):.1f} GB'
    return f'{ram_value} GB'


def compare(reports, paths):
    impl_names = []
    result_maps = []
    system_infos = []

    for i, report in enumerate(reports):
        name = report.get('openvx', {}).get('implementation', os.path.basename(paths[i]))
        impl_names.append(name)
        result_maps.append(build_result_map(report))
        system_infos.append(get_system_info(report))

    all_keys = set()
    for rm in result_maps:
        all_keys.update(rm.keys())

    all_keys = sorted(all_keys)

    return impl_names, result_maps, all_keys, system_infos


def write_markdown(impl_names, result_maps, all_keys, output_path, reports, system_infos=None):
    with open(output_path + '.md', 'w') as f:
        f.write('# OpenVX Benchmark Comparison\n\n')
        f.write(f'**{impl_names[0]}** vs **{impl_names[1]}**\n\n')

        # --- System Info ---
        if system_infos and len(system_infos) >= 2:
            hw_match = (system_infos[0].get('cpu_model') == system_infos[1].get('cpu_model')
                        and system_infos[0].get('cpu_cores') == system_infos[1].get('cpu_cores'))

            f.write('## System Info\n\n')
            if hw_match:
                si = system_infos[0]
                f.write('| Property | Value |\n')
                f.write('|:---|:---|\n')
                f.write(f'| CPU | {si.get("cpu_model", "N/A")} |\n')
                f.write(f'| Cores | {si.get("cpu_cores", "N/A")} |\n')
                f.write(f'| RAM | {format_ram(si.get("ram_gb", si.get("ram_bytes", "N/A")))} |\n')
                f.write(f'| OS | {si.get("os_name", "N/A")} {si.get("os_version", "")} |\n')
                f.write(f'\n> Same hardware — both benchmarks ran on identical hardware.\n\n')
            else:
                f.write(f'| Property |')
                for name in impl_names:
                    f.write(f' {name} |')
                f.write('\n|:---|')
                for _ in impl_names:
                    f.write(':---|')
                f.write('\n')
                for prop, key in [('CPU', 'cpu_model'), ('Cores', 'cpu_cores'),
                                  ('RAM', 'ram_gb'), ('OS', 'os_name')]:
                    f.write(f'| {prop} |')
                    for si in system_infos:
                        val = si.get(key, 'N/A')
                        if key == 'ram_gb' and val != 'N/A':
                            val = format_ram(val)
                        f.write(f' {val} |')
                    f.write('\n')
                f.write(f'\n> **Warning:** Benchmarks ran on different hardware — results may not be directly comparable.\n\n')

        # --- Conformance & Scores ---
        f.write('## Conformance & Scores\n\n')
        f.write(f'| Metric | {impl_names[0]} | {impl_names[1]} |\n')
        f.write('|:---|---:|---:|\n')

        scores = []
        for i, report in enumerate(reports):
            s = report.get('scores', {})
            scores.append(s)

        vision_a = scores[0].get('overall_vision_score', 0) if len(scores) > 0 else 0
        vision_b = scores[1].get('overall_vision_score', 0) if len(scores) > 1 else 0
        f.write(f'| Vision Score (MP/s) | {vision_a:.2f} | {vision_b:.2f} |\n')

        enhanced_a = scores[0].get('enhanced_vision_score', 0) if len(scores) > 0 else 0
        enhanced_b = scores[1].get('enhanced_vision_score', 0) if len(scores) > 1 else 0
        if enhanced_a > 0 or enhanced_b > 0:
            f.write(f'| Enhanced Vision Score (MP/s) | {enhanced_a:.2f} | {enhanced_b:.2f} |\n')

        conformance_info = []
        for report in reports:
            conf_list = report.get('conformance', [])
            if conf_list:
                c = conf_list[0]
                conformance_info.append({
                    'pass': c.get('pass', False),
                    'passed': c.get('passed', 0),
                    'total': c.get('total', 0),
                    'missing': c.get('missing_kernels', [])
                })
            else:
                conformance_info.append({'pass': False, 'passed': 0, 'total': 0, 'missing': []})

        ca, cb = conformance_info[0], conformance_info[1]
        f.write(f'| Conformance | {"PASS" if ca["pass"] else "FAIL"} ({ca["passed"]}/{ca["total"]}) '
                f'| {"PASS" if cb["pass"] else "FAIL"} ({cb["passed"]}/{cb["total"]}) |\n\n')

        # --- Category Sub-Scores ---
        cat_scores_a = scores[0].get('category_scores', {}) if len(scores) > 0 else {}
        cat_scores_b = scores[1].get('category_scores', {}) if len(scores) > 1 else {}

        all_cat_entries = {}
        for fs in set(list(cat_scores_a.keys()) + list(cat_scores_b.keys())):
            cats_a = cat_scores_a.get(fs, {})
            cats_b = cat_scores_b.get(fs, {})
            for cat in set(list(cats_a.keys()) + list(cats_b.keys())):
                key = f'{fs}/{cat}'
                all_cat_entries[key] = (cats_a.get(cat, 0), cats_b.get(cat, 0))

        if all_cat_entries:
            f.write('## Category Sub-Scores\n\n')
            f.write(f'| Category | {impl_names[0]} (MP/s) | {impl_names[1]} (MP/s) | Change % |\n')
            f.write('|:---|---:|---:|---:|\n')
            for key in sorted(all_cat_entries.keys()):
                a_val, b_val = all_cat_entries[key]
                change = ((b_val - a_val) / a_val * 100) if a_val > 0 else 0
                display = key.split('/')[-1] if '/' in key else key
                sign = '+' if change >= 0 else ''
                f.write(f'| {display} | {a_val:.2f} | {b_val:.2f} | {sign}{change:.1f} |\n')
            f.write('\n')

        # --- Summary ---
        regressions = 0
        improvements = 0
        same_count = 0
        cat_regressions = {}
        cat_improvements = {}

        comparison_rows = []
        for key in all_keys:
            name, mode, resolution = key
            r_a = result_maps[0].get(key)
            r_b = result_maps[1].get(key)

            if not (r_a and r_b):
                continue
            if not (r_a.get('supported', False) and r_b.get('supported', False)):
                continue
            if not (r_a.get('verified', True) and r_b.get('verified', True)):
                continue

            wc_a = r_a.get('wall_clock', {})
            wc_b = r_b.get('wall_clock', {})
            median_a = wc_a.get('median_ms', 0)
            median_b = wc_b.get('median_ms', 0)
            if median_a <= 0 or median_b <= 0:
                continue

            mps_a = r_a.get('megapixels_per_sec', 0)
            mps_b = r_b.get('megapixels_per_sec', 0)
            cv_a = wc_a.get('cv_percent', 0)
            cv_b = wc_b.get('cv_percent', 0)
            category = r_a.get('category', '')
            change_pct = ((median_b - median_a) / median_a) * 100

            if change_pct > 5.0:
                status = 'REGRESSION'
                regressions += 1
                cat_regressions[category] = cat_regressions.get(category, 0) + 1
            elif change_pct < -5.0:
                status = 'IMPROVEMENT'
                improvements += 1
                cat_improvements[category] = cat_improvements.get(category, 0) + 1
            else:
                status = 'same'
                same_count += 1

            comparison_rows.append({
                'name': name, 'category': category, 'mode': mode, 'resolution': resolution,
                'median_a': median_a, 'median_b': median_b,
                'mps_a': mps_a, 'mps_b': mps_b,
                'cv_a': cv_a, 'cv_b': cv_b,
                'change_pct': change_pct, 'status': status
            })

        comparison_rows.sort(key=lambda r: r['change_pct'], reverse=True)

        f.write('## Summary\n\n')
        f.write('| Metric | Count |\n')
        f.write('|:---|---:|\n')
        f.write(f'| Total compared | {len(comparison_rows)} |\n')
        f.write(f'| Regressions (>5% slower) | {regressions} |\n')
        f.write(f'| Improvements (>5% faster) | {improvements} |\n')
        f.write(f'| Unchanged | {same_count} |\n\n')

        if cat_regressions or cat_improvements:
            f.write('### By Category\n\n')
            f.write('| Category | Regressions | Improvements |\n')
            f.write('|:---|---:|---:|\n')
            all_summary_cats = sorted(set(list(cat_regressions.keys()) + list(cat_improvements.keys())))
            for cat in all_summary_cats:
                reg = cat_regressions.get(cat, 0)
                imp = cat_improvements.get(cat, 0)
                f.write(f'| {cat} | {reg} | {imp} |\n')
            f.write('\n')

        # --- Detailed Results ---
        f.write('## Detailed Comparison\n\n')
        f.write(f'> Change % is based on median latency. Positive = slower (regression), negative = faster (improvement).\n\n')
        f.write(f'| Benchmark | Mode | Resolution | {impl_names[0]} (ms) | {impl_names[0]} (MP/s) | '
                f'{impl_names[1]} (ms) | {impl_names[1]} (MP/s) | Change % | Status |\n')
        f.write('|:---|:---|:---|---:|---:|---:|---:|---:|:---|\n')

        has_unstable = False
        for row in comparison_rows:
            flag = ''
            if row['cv_a'] > 15 or row['cv_b'] > 15:
                flag = ' *'
                has_unstable = True
            sign = '+' if row['change_pct'] >= 0 else ''
            f.write(f'| {row["name"]} | {row["mode"]} | {row["resolution"]} '
                    f'| {row["median_a"]:.3f} | {row["mps_a"]:.1f} '
                    f'| {row["median_b"]:.3f} | {row["mps_b"]:.1f} '
                    f'| {sign}{row["change_pct"]:.1f} | {row["status"]}{flag} |\n')
        f.write('\n')

        if has_unstable:
            f.write('> \\* High variability (CV% > 15%) — comparison may not be reliable for these benchmarks. '
                    'Consider increasing iterations.\n\n')

        # --- Benchmarks Only In One Report ---
        keys_a = set(result_maps[0].keys())
        keys_b = set(result_maps[1].keys())
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)

        if only_a or only_b:
            f.write('## Benchmarks Only In One Report\n\n')
            if only_a:
                f.write(f'### Only in {impl_names[0]}\n\n')
                f.write('| Benchmark | Mode | Resolution |\n')
                f.write('|:---|:---|:---|\n')
                for name, mode, res in only_a:
                    f.write(f'| {name} | {mode} | {res} |\n')
                f.write('\n')
            if only_b:
                f.write(f'### Only in {impl_names[1]}\n\n')
                f.write('| Benchmark | Mode | Resolution |\n')
                f.write('|:---|:---|:---|\n')
                for name, mode, res in only_b:
                    f.write(f'| {name} | {mode} | {res} |\n')
                f.write('\n')

        # --- Conformance Detail ---
        missing_a = conformance_info[0].get('missing', [])
        missing_b = conformance_info[1].get('missing', [])
        if missing_a or missing_b:
            f.write('## Missing Kernels\n\n')
            f.write(f'| Implementation | Missing Kernels |\n')
            f.write(f'|:---|:---|\n')
            if missing_a:
                f.write(f'| {impl_names[0]} | {", ".join(missing_a)} |\n')
            if missing_b:
                f.write(f'| {impl_names[1]} | {", ".join(missing_b)} |\n')
            f.write('\n')

    print(f'  Comparison markdown: {output_path}.md')


def write_csv(impl_names, result_maps, all_keys, output_path, reports):
    with open(output_path + '.csv', 'w') as f:
        header = f'benchmark,category,mode,resolution'
        header += f',{impl_names[0]}_median_ms,{impl_names[0]}_mp_per_sec'
        header += f',{impl_names[1]}_median_ms,{impl_names[1]}_mp_per_sec'
        header += ',change_percent,status'
        f.write(header + '\n')

        for key in all_keys:
            name, mode, resolution = key
            r_a = result_maps[0].get(key)
            r_b = result_maps[1].get(key)

            if not (r_a and r_b):
                continue
            if not (r_a.get('supported', False) and r_b.get('supported', False)):
                continue
            if not (r_a.get('verified', True) and r_b.get('verified', True)):
                continue

            wc_a = r_a.get('wall_clock', {})
            wc_b = r_b.get('wall_clock', {})
            median_a = wc_a.get('median_ms', 0)
            median_b = wc_b.get('median_ms', 0)
            if median_a <= 0 or median_b <= 0:
                continue

            mps_a = r_a.get('megapixels_per_sec', 0)
            mps_b = r_b.get('megapixels_per_sec', 0)
            category = r_a.get('category', '')
            change_pct = ((median_b - median_a) / median_a) * 100

            if change_pct > 5.0:
                status = 'REGRESSION'
            elif change_pct < -5.0:
                status = 'IMPROVEMENT'
            else:
                status = 'same'

            f.write(f'{name},{category},{mode},{resolution},'
                    f'{median_a:.4f},{mps_a:.2f},'
                    f'{median_b:.4f},{mps_b:.2f},'
                    f'{change_pct:.2f},{status}\n')

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

    impl_names, result_maps, all_keys, system_infos = compare(reports, args.reports)

    write_markdown(impl_names, result_maps, all_keys, args.output, reports, system_infos)
    write_csv(impl_names, result_maps, all_keys, args.output, reports)

    print(f'\nCompared {len(args.reports)} implementations across {len(all_keys)} benchmarks')


if __name__ == '__main__':
    main()
