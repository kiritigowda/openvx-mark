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

        framework_a = scores[0].get('framework_score', 0) if len(scores) > 0 else 0
        framework_b = scores[1].get('framework_score', 0) if len(scores) > 1 else 0
        framework_count_a = scores[0].get('framework_metric_count', 0) if len(scores) > 0 else 0
        framework_count_b = scores[1].get('framework_metric_count', 0) if len(scores) > 1 else 0
        if framework_count_a > 0 or framework_count_b > 0:
            f.write(f'| Framework Score (x, geomean) | {framework_a:.3f} | {framework_b:.3f} |\n')

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

        # --- Framework Metrics Comparison ---
        #
        # Layout intent (uniform & intuitive — designed to remove the
        # per-row inversion that the previous design did):
        #
        #   * One H3 subsection per (benchmark, resolution) scenario, so
        #     30+ mixed metrics aren't mashed into one flat table.
        #   * Within each scenario, metrics are split by direction into
        #     up to three H4 sub-tables: ↑ higher-is-better, ↓ lower-
        #     is-better, · descriptive. Each sub-table is internally
        #     uniform — every row uses the SAME ratio formula and the
        #     SAME "what counts as a win" rule.
        #   * The ratio column is ALWAYS the literal raw `B/A`. No
        #     per-row inversion magic. The H4 heading tells the reader
        #     whether bigger or smaller is the winning direction in
        #     that sub-table:
        #       - ↑ table: bold when ratio > 1.00x (B is bigger = better)
        #       - ↓ table: bold when ratio < 1.00x (B is smaller = better)
        #     Bold cells visually mark winners regardless of direction.
        #   * Descriptive metrics (unit=='count') get a stripped-down
        #     no-ratio sub-table — they're structural, not performance.
        #   * Per-scenario footer summarising total win count for B
        #     across all comparable metrics.
        fw_keys = {}
        fw_metrics_by_key = {}
        per_side_metrics = [{}, {}]
        for side, rmap in enumerate(result_maps):
            for (name, mode, resolution), r in rmap.items():
                fms = r.get('framework_metrics', [])
                if not fms:
                    continue
                key = (name, resolution)
                fw_keys[key] = f'{name} @ {resolution}'
                fw_metrics_by_key.setdefault(key, set())
                per_side_metrics[side].setdefault(key, {})
                for fm in fms:
                    nm = fm.get('name')
                    if not nm:
                        continue
                    fw_metrics_by_key[key].add(nm)
                    per_side_metrics[side][key][nm] = fm

        def _direction(fm):
            """Return 'higher', 'lower', or 'descriptive' for a metric dict."""
            unit = (fm or {}).get('unit', '') or ''
            if unit == 'count':
                return 'descriptive'
            return 'higher' if (fm or {}).get('higher_is_better', True) else 'lower'

        def _write_metric_table(f, rows, direction, name_a, name_b):
            """Render one sub-table for a given direction band.

            rows: list of (metric_name, a_fm, b_fm) tuples sorted alphabetically.
            direction: 'higher', 'lower', or 'descriptive'.
            Returns (wins_for_b, comparable_count) so the per-scenario
            summary can aggregate across directions.
            """
            if direction == 'descriptive':
                f.write(f'| Metric | Unit | {name_a} | {name_b} |\n')
                f.write('|:---|:---|---:|---:|\n')
            else:
                f.write(f'| Metric | Unit | {name_a} | {name_b} | B/A Ratio |\n')
                f.write('|:---|:---|---:|---:|---:|\n')

            wins_b = 0
            comparable = 0
            for nm, a_fm, b_fm in rows:
                unit = (a_fm or b_fm or {}).get('unit', '') or '—'
                a_val = a_fm.get('value') if a_fm else None
                b_val = b_fm.get('value') if b_fm else None
                a_str = f'{a_val:.3f}' if a_val is not None else '—'
                b_str = f'{b_val:.3f}' if b_val is not None else '—'

                if direction == 'descriptive':
                    f.write(f'| `{nm}` | {unit} | {a_str} | {b_str} |\n')
                    continue

                # Always literal B/A — no inversion ever.
                if (a_val is not None and b_val is not None
                        and a_val > 0 and b_val > 0):
                    ratio = b_val / a_val
                    comparable += 1
                    b_wins = (ratio > 1.0) if direction == 'higher' else (ratio < 1.0)
                    if b_wins:
                        wins_b += 1
                        ratio_cell = f'**{ratio:.2f}x**'
                    else:
                        ratio_cell = f'{ratio:.2f}x'
                else:
                    ratio_cell = '—'

                f.write(f'| `{nm}` | {unit} | {a_str} | {b_str} | {ratio_cell} |\n')

            return wins_b, comparable

        if fw_keys:
            f.write('## Framework Metrics Comparison\n\n')
            f.write(
                '> Per-scenario framework metrics — graph orchestration, '
                'scheduling, async streaming, verify cost. Each scenario '
                'is split into up to three sub-tables grouped by metric '
                'direction so every row in a given table follows the same '
                'rule:\n>\n'
                f'>   * **↑ Higher-is-better** — throughput, speedup, fusion ratio. Ratio = {impl_names[1]} / {impl_names[0]}; **{impl_names[1]} wins when ratio > 1.00x**.\n'
                f'>   * **↓ Lower-is-better** — latency, overhead in ms. Ratio = {impl_names[1]} / {impl_names[0]} (raw, no inversion); **{impl_names[1]} wins when ratio < 1.00x** because lower is better here.\n'
                '>   * **· Descriptive** — counts and structural sizes; no ratio shown.\n>\n'
                f'> **Bold** ratios mark cells where {impl_names[1]} wins, regardless of direction.\n\n'
            )

            for key in sorted(fw_keys.keys()):
                display = fw_keys[key]
                f.write(f'### {display}\n\n')

                # Bucket this scenario's metrics by direction.
                buckets = {'higher': [], 'lower': [], 'descriptive': []}
                for nm in sorted(fw_metrics_by_key[key]):
                    a_fm = per_side_metrics[0].get(key, {}).get(nm)
                    b_fm = per_side_metrics[1].get(key, {}).get(nm)
                    buckets[_direction(a_fm or b_fm)].append((nm, a_fm, b_fm))

                total_wins = 0
                total_comparable = 0

                if buckets['higher']:
                    f.write('#### ↑ Higher-is-better metrics\n\n')
                    w, c = _write_metric_table(f, buckets['higher'], 'higher',
                                               impl_names[0], impl_names[1])
                    total_wins += w
                    total_comparable += c
                    f.write(f'\n_{impl_names[1]} wins {w}/{c} in this category '
                            f'(**bold** = {impl_names[1]} better, i.e. ratio > 1.00x)._\n\n')

                if buckets['lower']:
                    f.write('#### ↓ Lower-is-better metrics\n\n')
                    w, c = _write_metric_table(f, buckets['lower'], 'lower',
                                               impl_names[0], impl_names[1])
                    total_wins += w
                    total_comparable += c
                    f.write(f'\n_{impl_names[1]} wins {w}/{c} in this category '
                            f'(**bold** = {impl_names[1]} better, i.e. ratio < 1.00x because '
                            f'{impl_names[1]} is smaller / faster)._\n\n')

                if buckets['descriptive']:
                    f.write('#### · Descriptive metrics\n\n')
                    _write_metric_table(f, buckets['descriptive'], 'descriptive',
                                        impl_names[0], impl_names[1])
                    f.write('\n')

                # Per-scenario summary line.
                if total_comparable > 0:
                    f.write(f'_**Per-scenario summary:** {impl_names[1]} wins '
                            f'**{total_wins}/{total_comparable}** comparable metrics in this scenario._\n\n')
                else:
                    f.write('_No comparable metrics in this scenario._\n\n')

        # --- Build comparison rows (include all results, not just verified) ---
        comparison_rows = []
        for key in all_keys:
            name, mode, resolution = key
            r_a = result_maps[0].get(key)
            r_b = result_maps[1].get(key)

            if not r_a and not r_b:
                continue

            row = {'name': name, 'mode': mode, 'resolution': resolution}

            for side, r in [('a', r_a), ('b', r_b)]:
                if r:
                    row[f'supported_{side}'] = r.get('supported', False)
                    row[f'verified_{side}'] = r.get('verified', True) if r.get('supported', False) else False
                    wc = r.get('wall_clock', {})
                    row[f'median_{side}'] = wc.get('median_ms', 0)
                    row[f'mps_{side}'] = r.get('megapixels_per_sec', 0)
                    row[f'cv_{side}'] = wc.get('cv_percent', 0)
                    row[f'category'] = r.get('category', '')
                else:
                    row[f'supported_{side}'] = False
                    row[f'verified_{side}'] = False
                    row[f'median_{side}'] = 0
                    row[f'mps_{side}'] = 0
                    row[f'cv_{side}'] = 0

            if (row['median_a'] > 0 and row['median_b'] > 0
                    and row['verified_a'] and row['verified_b']):
                row['speedup'] = row['mps_b'] / row['mps_a'] if row['mps_a'] > 0 else 0
            else:
                row['speedup'] = 0

            comparison_rows.append(row)

        comparison_rows.sort(key=lambda r: r.get('speedup', 0))

        # --- Summary ---
        both_verified = sum(1 for r in comparison_rows if r['verified_a'] and r['verified_b'])
        a_only = sum(1 for r in comparison_rows if r['verified_a'] and not r['verified_b'])
        b_only = sum(1 for r in comparison_rows if not r['verified_a'] and r['verified_b'])

        keys_a = set(result_maps[0].keys())
        keys_b = set(result_maps[1].keys())
        only_a_keys = sorted(keys_a - keys_b)
        only_b_keys = sorted(keys_b - keys_a)

        f.write('## Summary\n\n')
        f.write('| Metric | Count |\n')
        f.write('|:---|---:|\n')
        f.write(f'| Total benchmarks compared | {len(comparison_rows)} |\n')
        f.write(f'| Both verified | {both_verified} |\n')
        if a_only > 0:
            f.write(f'| Verified only in {impl_names[0]} | {a_only} |\n')
        if b_only > 0:
            f.write(f'| Verified only in {impl_names[1]} | {b_only} |\n')
        if only_a_keys:
            f.write(f'| Only in {impl_names[0]} | {len(only_a_keys)} |\n')
        if only_b_keys:
            f.write(f'| Only in {impl_names[1]} | {len(only_b_keys)} |\n')
        f.write('\n')

        # --- Detailed Results ---
        f.write('## Detailed Comparison\n\n')
        f.write(f'> Speedup = {impl_names[1]} throughput / {impl_names[0]} throughput. '
                f'Values >1.00 mean {impl_names[1]} is faster.\n\n')
        f.write(f'| Benchmark | Mode | Resolution '
                f'| {impl_names[0]} (ms) | {impl_names[0]} (MP/s) | {impl_names[0]} Verified '
                f'| {impl_names[1]} (ms) | {impl_names[1]} (MP/s) | {impl_names[1]} Verified '
                f'| Speedup |\n')
        f.write('|:---|:---|:---|---:|---:|:---:|---:|---:|:---:|---:|\n')

        has_unstable = False
        for row in comparison_rows:
            flag = ''
            if row['cv_a'] > 15 or row['cv_b'] > 15:
                flag = ' *'
                has_unstable = True

            f.write(f'| {row["name"]} | {row["mode"]} | {row["resolution"]} | ')

            if not row['supported_a']:
                f.write('N/A | N/A | N/A | ')
            else:
                v = 'PASS' if row['verified_a'] else 'FAIL'
                f.write(f'{row["median_a"]:.3f} | {row["mps_a"]:.1f} | {v} | ')

            if not row['supported_b']:
                f.write('N/A | N/A | N/A | ')
            else:
                v = 'PASS' if row['verified_b'] else 'FAIL'
                f.write(f'{row["median_b"]:.3f} | {row["mps_b"]:.1f} | {v} | ')

            if row['speedup'] > 0:
                f.write(f'{row["speedup"]:.2f}x{flag}')
            else:
                f.write('N/A')
            f.write(' |\n')
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
        header += f',{impl_names[0]}_median_ms,{impl_names[0]}_mp_per_sec,{impl_names[0]}_verified'
        header += f',{impl_names[1]}_median_ms,{impl_names[1]}_mp_per_sec,{impl_names[1]}_verified'
        header += ',speedup'
        f.write(header + '\n')

        for key in sorted(all_keys):
            name, mode, resolution = key
            r_a = result_maps[0].get(key)
            r_b = result_maps[1].get(key)

            if not r_a and not r_b:
                continue

            category = ''
            cols_a = ',,'
            cols_b = ',,'
            verified_a = False
            verified_b = False
            mps_a = 0
            mps_b = 0

            if r_a and r_a.get('supported', False):
                category = r_a.get('category', '')
                wc = r_a.get('wall_clock', {})
                median = wc.get('median_ms', 0)
                mps_a = r_a.get('megapixels_per_sec', 0)
                verified_a = r_a.get('verified', True)
                cols_a = f'{median:.4f},{mps_a:.2f},{"PASS" if verified_a else "FAIL"}'

            if r_b and r_b.get('supported', False):
                if not category:
                    category = r_b.get('category', '')
                wc = r_b.get('wall_clock', {})
                median = wc.get('median_ms', 0)
                mps_b = r_b.get('megapixels_per_sec', 0)
                verified_b = r_b.get('verified', True)
                cols_b = f'{median:.4f},{mps_b:.2f},{"PASS" if verified_b else "FAIL"}'

            speedup = ''
            if verified_a and verified_b and mps_a > 0:
                speedup = f'{mps_b / mps_a:.4f}'

            f.write(f'{name},{category},{mode},{resolution},{cols_a},{cols_b},{speedup}\n')

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
