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
ci_pairwise_summary.py — emit an organized GitHub Actions step summary for
N-implementation pairwise benchmark comparisons.

Replaces the inline-Python + do_compare bash block that used to live in
.github/workflows/ci.yml. The old block emitted one heading + headline
table + full per-kernel detail per comparison (6 sections × ~50 lines each
= ~600 line summary). This helper restructures that into three scannable
parts:

  1. TL;DR matrix — every loaded report on both axes, cell = geomean
     speedup of `row impl / column impl`. One glance answers "which
     implementation is fastest overall against which?".
  2. Grouped headline tables — comparisons clustered by the question
     they answer (e.g. "does adopting OpenVX pay off vs cv::?"), each
     row a geomean + median + win/loss + best/worst kernel.
  3. Per-kernel detail — full output of scripts/compare_reports.py per
     comparison, each wrapped in a <details> block collapsed by
     default. One click reveals the same data the old layout dumped
     unconditionally.

Same orientation convention as scripts/compare_reports.py:
    speedup = throughput(candidate) / throughput(baseline)
Values >1.00x mean the candidate is faster than the baseline.

------------------------------------------------------------------------
Config schema (JSON file passed via --config):

    {
      "reports": {
        "<id>": {
          "label": "<display name shown in tables>",
          "path":  "<path/to/benchmark_results.json>"
        },
        ...
      },
      "groups": [
        {
          "title":  "<section title shown as `##`>",
          "intent": "<one-line description rendered as a blockquote>",
          "pairs":  [["candidate_id", "baseline_id"], ...]
        },
        ...
      ],
      "detail_dir": "comparisons"   # where <candidate>-over-<baseline>.md
                                    # files produced by compare_reports.py
                                    # are looked up.
    }

Output goes to stdout — caller redirects to "$GITHUB_STEP_SUMMARY".
------------------------------------------------------------------------

Reports whose `path` does not exist are tolerated: they appear with "—"
cells in the matrix and a "no comparable benchmarks" note in the headline
table rows that reference them. Detail blocks for missing comparisons
render with a brief "_Detail missing_" line instead of a per-kernel
table. This mirrors the existing CI semantics where a single impl-build
crash should not lose the comparison signal for the rest.
"""

import argparse
import json
import math
import os
import sys


# ---------------------------------------------------------------------------
# Data loading & headline-stats computation
# ---------------------------------------------------------------------------

def load_report(path):
    """Return parsed JSON report at `path`, or None if missing/unreadable."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def by_key(report):
    """Index a report's results dict by (name, mode, resolution)."""
    return {(r['name'], r['mode'], r['resolution']): r
            for r in report.get('results', [])}


def compute_pair_stats(cand_report, base_report):
    """Compute headline stats for one (candidate, baseline) pair.

    Returns a dict with keys: geomean, median, count, wins, losses, best,
    worst — or None if either report is missing or no shared verified
    benchmark has positive throughput on both sides. Best/worst are
    {'key': (name, mode, res), 'speedup': float}.

    Matches the orientation of compare_reports.py: speedup is
    candidate/baseline, so values >1.00x mean the candidate is faster.
    """
    if cand_report is None or base_report is None:
        return None

    c = by_key(cand_report)
    b = by_key(base_report)
    shared = sorted(set(c) & set(b))

    speedups = []
    wins = losses = 0
    best = None
    worst = None

    for key in shared:
        rc, rb = c[key], b[key]
        # `verified` defaults to True for reports predating the verify
        # column — preserves backward-compat with older JSON.
        if not (rc.get('verified', True) and rb.get('verified', True)):
            continue
        mc = rc.get('megapixels_per_sec', 0)
        mb = rb.get('megapixels_per_sec', 0)
        if mc <= 0 or mb <= 0:
            continue
        s = mc / mb
        speedups.append(s)
        if s > 1.0:
            wins += 1
        elif s < 1.0:
            losses += 1
        if best is None or s > best[1]:
            best = (key, s)
        if worst is None or s < worst[1]:
            worst = (key, s)

    if not speedups:
        return None

    geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    median = sorted(speedups)[len(speedups) // 2]
    return {
        'geomean': geomean,
        'median': median,
        'count': len(speedups),
        'wins': wins,
        'losses': losses,
        'best': {'key': best[0], 'speedup': best[1]} if best else None,
        'worst': {'key': worst[0], 'speedup': worst[1]} if worst else None,
    }


# ---------------------------------------------------------------------------
# Markdown rendering helpers
# ---------------------------------------------------------------------------

def fmt_key(key):
    """Render a (name, mode, resolution) tuple as a readable label."""
    return f'{key[0]} / {key[1]} / {key[2]}'


def render_tldr_matrix(reports, loaded_ids, out):
    """Emit the N×N geomean speedup matrix over all loaded reports.

    `loaded_ids` is the subset of report ids whose JSON loaded
    successfully — the matrix only shows rows/cols for those, so a
    failed build cleanly drops out instead of producing a row of "—"
    cells.
    """
    out.write('## TL;DR — Geomean speedup matrix\n\n')
    out.write(
        '> Cell value = `row impl megapixels/s ÷ column impl megapixels/s`, '
        'geomean across verified shared benchmarks. **Values >1.00x mean '
        'the row impl is faster than the column impl** (so reading along '
        'a row tells you how much that impl beats each other impl). '
        'Diagonal cells (impl vs itself) and missing cells (no shared '
        'verified benchmarks) render as "—".\n\n'
    )

    header_cells = ['↓ row faster than → ']
    sep_cells = [':---']
    for cid in loaded_ids:
        header_cells.append(reports[cid]['label'])
        sep_cells.append('---:')
    out.write('| ' + ' | '.join(header_cells) + ' |\n')
    out.write('|' + '|'.join(sep_cells) + '|\n')

    for row_id in loaded_ids:
        row_cells = [f'**{reports[row_id]["label"]}**']
        for col_id in loaded_ids:
            if row_id == col_id:
                row_cells.append('—')
                continue
            stats = compute_pair_stats(
                reports[row_id]['report'],
                reports[col_id]['report'],
            )
            if stats is None:
                row_cells.append('—')
            else:
                # Bold the cell when the row impl wins, italic when it
                # loses — gives a quick visual scan even on small
                # screens where the digits blur together.
                if stats['geomean'] >= 1.0:
                    row_cells.append(f'**{stats["geomean"]:.2f}x**')
                else:
                    row_cells.append(f'_{stats["geomean"]:.2f}x_')
        out.write('| ' + ' | '.join(row_cells) + ' |\n')
    out.write('\n')


def render_headline_group(group, reports, pair_stats, out):
    """Render one question-grouped headline table for a list of pairs."""
    out.write(f'## {group["title"]}\n\n')
    if group.get('intent'):
        out.write(f'> {group["intent"]}\n\n')

    out.write('| Candidate | Baseline | Geomean | Median | Compared | '
              'Cand wins | Base wins | Best kernel for candidate | '
              'Worst kernel for candidate |\n')
    out.write('|:---|:---|---:|---:|---:|---:|---:|:---|:---|\n')

    for cand_id, base_id in group['pairs']:
        cand_label = reports[cand_id]['label']
        base_label = reports[base_id]['label']
        stats = pair_stats.get((cand_id, base_id))
        if stats is None:
            cand_ok = '✓' if reports[cand_id].get('report') is not None else '✗'
            base_ok = '✓' if reports[base_id].get('report') is not None else '✗'
            out.write(
                f'| {cand_label} | {base_label} | — | — | 0 | — | — | '
                f'_no comparable benchmarks ({cand_label}: {cand_ok}, '
                f'{base_label}: {base_ok})_ | — |\n'
            )
            continue

        best = stats['best']
        worst = stats['worst']
        best_str = (f'`{fmt_key(best["key"])}` ({best["speedup"]:.2f}x)'
                    if best else '—')
        worst_str = (f'`{fmt_key(worst["key"])}` ({worst["speedup"]:.2f}x)'
                     if worst else '—')
        out.write(
            f'| {cand_label} | {base_label} | '
            f'**{stats["geomean"]:.2f}x** | {stats["median"]:.2f}x | '
            f'{stats["count"]} | {stats["wins"]} | {stats["losses"]} | '
            f'{best_str} | {worst_str} |\n'
        )
    out.write('\n')


# Lines at the top of compare_reports.py output that duplicate what the
# <details><summary> already shows. Skipping them keeps the expanded
# detail focused on the per-kernel tables and category sub-scores.
_DUPLICATE_HEADER_PREFIXES = (
    '# OpenVX Benchmark Comparison',
    '**',  # the "**implA** vs **implB**" subheading
)


def _strip_duplicate_header(md_text):
    """Drop the heading lines from compare_reports.py output that the
    <details> summary already conveys."""
    lines = md_text.splitlines()
    i = 0
    # Skip blank lines and the well-known header lines until we hit
    # actual content (e.g. "## System Info").
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if any(s.startswith(p) for p in _DUPLICATE_HEADER_PREFIXES):
            i += 1
            continue
        break
    return '\n'.join(lines[i:])


def render_details_section(groups, reports, detail_dir, out):
    """Emit a single <details> block per (candidate, baseline) pair
    containing the per-kernel comparison table from compare_reports.py.

    Collapsed by default — this is the bulk of what the old layout
    dumped unconditionally, now one click away instead.
    """
    out.write('## Per-kernel detail (click to expand)\n\n')
    out.write(
        '> The full per-kernel speedup table from '
        '`scripts/compare_reports.py` for each comparison, including '
        'system info, conformance, and category sub-scores. Collapsed '
        'by default to keep the summary scannable; expand a row when '
        'you need to investigate a specific kernel.\n\n'
    )

    for group in groups:
        for cand_id, base_id in group['pairs']:
            cand_label = reports[cand_id]['label']
            base_label = reports[base_id]['label']
            detail_path = os.path.join(
                detail_dir, f'{cand_id}-over-{base_id}.md'
            )
            out.write(
                f'<details><summary><b>{cand_label}</b> over '
                f'<b>{base_label}</b> — per-kernel detail</summary>\n\n'
            )
            if not os.path.isfile(detail_path):
                out.write(
                    f'_Detail file missing (`{detail_path}`) — '
                    f'comparison was skipped because one or both inputs '
                    f'were unavailable._\n\n'
                )
            else:
                with open(detail_path) as f:
                    body = _strip_duplicate_header(f.read())
                out.write(body)
                if not body.endswith('\n'):
                    out.write('\n')
                out.write('\n')
            out.write('</details>\n\n')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Emit an organized pairwise comparison summary for CI.'
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to JSON config file (see script docstring for schema).'
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    reports = config['reports']
    for rinfo in reports.values():
        rinfo['report'] = load_report(rinfo.get('path'))

    groups = config['groups']
    detail_dir = config.get('detail_dir', 'comparisons')

    # Precompute headline stats once per pair that appears in any group;
    # the matrix renderer computes its own (M-1)² off-diagonal stats
    # over loaded reports separately so it can show "everything vs
    # everything" even if a particular pairing isn't in `groups`.
    pair_stats = {}
    for group in groups:
        for cand_id, base_id in group['pairs']:
            if cand_id not in reports or base_id not in reports:
                pair_stats[(cand_id, base_id)] = None
                continue
            pair_stats[(cand_id, base_id)] = compute_pair_stats(
                reports[cand_id]['report'],
                reports[base_id]['report'],
            )

    out = sys.stdout
    out.write('# Cross-implementation pairwise comparisons\n\n')

    # Brief context line so the summary stands alone when read out of
    # the workflow YAML's context (e.g. linked from a PR review).
    out.write(
        '> Generated by `scripts/ci_pairwise_summary.py` from the JSON '
        'reports produced by `openvx-mark` (per OpenVX impl) and '
        '`opencv-mark` (OpenCV baseline). Same orientation as '
        '`scripts/compare_reports.py`: speedup = `candidate / baseline`, '
        '>1.00x means the candidate is faster.\n\n'
    )

    loaded_ids = [rid for rid in reports
                  if reports[rid].get('report') is not None]
    if len(loaded_ids) >= 2:
        render_tldr_matrix(reports, loaded_ids, out)
    else:
        out.write(
            f'_Only {len(loaded_ids)} report(s) loaded successfully — '
            f'no pairwise comparison possible. Check the per-impl build '
            f'jobs for failures._\n\n'
        )

    for group in groups:
        render_headline_group(group, reports, pair_stats, out)

    render_details_section(groups, reports, detail_dir, out)


if __name__ == '__main__':
    main()
