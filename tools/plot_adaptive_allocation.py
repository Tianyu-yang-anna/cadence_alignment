"""Adaptivity-fingerprint figure: content-adaptive information allocation.

Two panels from results/psbits/psbits_<arm>.json (per-window per-scale
teacher-forced bits, n=2048 val windows, 12.8B 2D arm):
  left  — per-scale share of description length, easiest vs hardest quartile
          (the shape rotates: hard content concentrates into q256/q512);
  right — Pearson r between a window's total bits (difficulty) and each
          scale's share: the fingerprint (+0.83 at q512, −0.83 at q32).

Claim this supports (and the wording it forbids): the slot GRID is fixed —
"different token counts for different content" would be false; what adapts
is WHERE the description length lives. Hand-written SVG (no matplotlib).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SRC = "results/psbits/psbits_planner_prefix_owt2_pqsh_b12s2e8.json"


def main():
    d = json.loads(Path(SRC).read_text())
    rows, scales, S = d["rows"], d["scales"], d["segments"]
    n, K = len(rows), len(scales)

    def shares(r):
        tot = [r["per_scale_bits"][i] * scales[i] * S for i in range(K)]
        s = sum(tot)
        return [t / s for t in tot]

    rows.sort(key=lambda r: r["total_bits"])
    q1, q4 = rows[:n // 4], rows[3 * n // 4:]
    sh1 = [sum(shares(r)[k] for r in q1) / len(q1) for k in range(K)]
    sh4 = [sum(shares(r)[k] for r in q4) / len(q4) for k in range(K)]
    tb = [r["total_bits"] for r in rows]
    mt = sum(tb) / n
    rs = []
    for k in range(K):
        ys = [shares(r)[k] for r in rows]
        my = sum(ys) / n
        cov = sum((a - mt) * (b - my) for a, b in zip(tb, ys))
        vx = sum((a - mt) ** 2 for a in tb)
        vy = sum((b - my) ** 2 for b in ys)
        rs.append(cov / math.sqrt(vx * vy))

    W, H, PAD = 940, 400, 60
    PW = (W - 3 * PAD) / 2
    b = ['<svg xmlns="http://www.w3.org/2000/svg" '
         f'width="{W}" height="{H}" font-family="Helvetica,Arial">']

    def xk(x0, k):
        return x0 + k / (K - 1) * PW

    # left panel: share curves (log-y to show all scales)
    x0 = PAD
    b.append(f'<text x="{x0+PW/2:.0f}" y="26" text-anchor="middle" '
             'font-size="14" font-weight="bold">描述长度份额：最易 vs 最难四分位'
             '</text>')
    lo, hi = -4, 0  # log10 share range 0.01%..100%

    def ylog(v):
        return H - PAD - (math.log10(max(v, 1e-4)) - lo) / (hi - lo) \
            * (H - 2 * PAD - 10)
    b.append(f'<rect x="{x0}" y="{PAD-10}" width="{PW:.0f}" '
             f'height="{H-2*PAD+10}" fill="none" stroke="#999"/>')
    for g in (-4, -3, -2, -1, 0):
        b.append(f'<line x1="{x0}" y1="{ylog(10**g):.0f}" x2="{x0+PW:.0f}" '
                 f'y2="{ylog(10**g):.0f}" stroke="#eee"/>')
        b.append(f'<text x="{x0-6}" y="{ylog(10**g)+4:.0f}" text-anchor="end" '
                 f'font-size="10">{100*10**g:g}%</text>')
    for (sh, col, nm) in ((sh1, "#2471a3", "Q1 最易 25%"),
                          (sh4, "#c0392b", "Q4 最难 25%")):
        pts = " ".join(f"{xk(x0,k):.1f},{ylog(sh[k]):.1f}" for k in range(K))
        b.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
                 'stroke-width="2"/>')
        for k in range(K):
            b.append(f'<circle cx="{xk(x0,k):.1f}" cy="{ylog(sh[k]):.1f}" '
                     f'r="3" fill="{col}"/>')
    b.append(f'<text x="{x0+8}" y="{PAD+8}" font-size="11" '
             'fill="#2471a3">Q1 最易 25%</text>')
    b.append(f'<text x="{x0+8}" y="{PAD+22}" font-size="11" '
             'fill="#c0392b">Q4 最难 25%</text>')

    # right panel: correlation fingerprint bars
    x1 = 2 * PAD + PW
    b.append(f'<text x="{x1+PW/2:.0f}" y="26" text-anchor="middle" '
             'font-size="14" font-weight="bold">自适应指纹：r(份额, 难度)，'
             f'n={n}</text>')
    def yr(v):
        return H - PAD - (v + 1) / 2 * (H - 2 * PAD - 10)
    b.append(f'<rect x="{x1}" y="{PAD-10}" width="{PW:.0f}" '
             f'height="{H-2*PAD+10}" fill="none" stroke="#999"/>')
    b.append(f'<line x1="{x1}" y1="{yr(0):.0f}" x2="{x1+PW:.0f}" '
             f'y2="{yr(0):.0f}" stroke="#555"/>')
    for g in (-1, -0.5, 0.5, 1):
        b.append(f'<line x1="{x1}" y1="{yr(g):.0f}" x2="{x1+PW:.0f}" '
                 f'y2="{yr(g):.0f}" stroke="#eee"/>')
        b.append(f'<text x="{x1-6}" y="{yr(g)+4:.0f}" text-anchor="end" '
                 f'font-size="10">{g:+.1f}</text>')
    bw = PW / K * 0.62
    for k in range(K):
        col = "#c0392b" if rs[k] > 0 else "#2471a3"
        y0, y1v = sorted((yr(0), yr(rs[k])))
        b.append(f'<rect x="{xk(x1,k)-bw/2:.1f}" y="{y0:.1f}" '
                 f'width="{bw:.1f}" height="{max(y1v-y0,1):.1f}" '
                 f'fill="{col}" opacity="0.85"/>')
        b.append(f'<text x="{xk(x1,k):.1f}" y="{yr(rs[k])+(-6 if rs[k]>0 else 14):.1f}" '
                 f'text-anchor="middle" font-size="9">{rs[k]:+.2f}</text>')
    for x0p in (PAD, x1):
        for k in range(K):
            b.append(f'<text x="{xk(x0p,k):.1f}" y="{H-PAD+16}" '
                     f'text-anchor="middle" font-size="9">q{scales[k]}</text>')
    b.append("</svg>")
    out = Path("docs/figures/adaptive_allocation.svg")
    out.write_text("\n".join(b))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
