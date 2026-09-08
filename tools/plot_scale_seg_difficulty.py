"""Learned-difficulty figure: per-scale hump (95% CI) + scale x segment heatmap.

Data: results/psbits/psbits_planner_prefix_owt2_pqsh_b12s2e8_seg.json
(2048 val windows, deployed 12.8B 2D arm, teacher-forced bits/slot).
Hand-written SVG (no matplotlib in the venv).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SRC = "results/psbits/psbits_planner_prefix_owt2_pqsh_b12s2e8_seg.json"


def main():
    d = json.loads(Path(SRC).read_text())
    rows, scales, S = d["rows"], d["scales"], d["segments"]
    n, K = len(rows), len(scales)
    means, cis = [], []
    for k in range(K):
        xs = [r["per_scale_bits"][k] for r in rows]
        m = sum(xs) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
        means.append(m)
        cis.append(1.96 * sd / math.sqrt(n))
    segm = [[sum(r["per_scale_seg"][k][s] for r in rows) / n
             for s in range(S)] for k in range(K)]

    W, H, PAD = 940, 420, 62
    PW = (W - 3 * PAD) / 2
    b = ['<svg xmlns="http://www.w3.org/2000/svg" '
         f'width="{W}" height="{H}" font-family="Helvetica,Arial">']

    # left: hump with CI band
    x0 = PAD
    b.append(f'<text x="{x0+PW/2:.0f}" y="24" text-anchor="middle" '
             'font-size="14" font-weight="bold">逐尺度学到的难度'
             f'（bits/slot ± 95% CI，n={n}）</text>')
    ylo, yhi = 0.0, 6.5

    def X(k):
        return x0 + k / (K - 1) * PW

    def Y(v):
        return H - PAD - (v - ylo) / (yhi - ylo) * (H - 2 * PAD - 12)
    b.append(f'<rect x="{x0}" y="{PAD-12}" width="{PW:.0f}" '
             f'height="{H-2*PAD+12}" fill="none" stroke="#999"/>')
    for g in range(0, 7):
        b.append(f'<line x1="{x0}" y1="{Y(g):.0f}" x2="{x0+PW:.0f}" '
                 f'y2="{Y(g):.0f}" stroke="#eee"/>')
        b.append(f'<text x="{x0-6}" y="{Y(g)+4:.0f}" text-anchor="end" '
                 f'font-size="10">{g}</text>')
    band_up = " ".join(f"{X(k):.1f},{Y(means[k]+cis[k]):.1f}" for k in range(K))
    band_dn = " ".join(f"{X(k):.1f},{Y(means[k]-cis[k]):.1f}"
                       for k in reversed(range(K)))
    b.append(f'<polygon points="{band_up} {band_dn}" fill="#c0392b" '
             'opacity="0.25"/>')
    pts = " ".join(f"{X(k):.1f},{Y(means[k]):.1f}" for k in range(K))
    b.append(f'<polyline points="{pts}" fill="none" stroke="#c0392b" '
             'stroke-width="2"/>')
    for k in range(K):
        b.append(f'<circle cx="{X(k):.1f}" cy="{Y(means[k]):.1f}" r="3" '
                 'fill="#c0392b"/>')
        b.append(f'<text x="{X(k):.1f}" y="{H-PAD+16}" text-anchor="middle" '
                 f'font-size="9">q{scales[k]}</text>')
    b.append(f'<text x="{X(5):.1f}" y="{Y(means[5])-10:.1f}" '
             'text-anchor="middle" font-size="10" fill="#c0392b">峰 q32='
             f'{means[5]:.2f}</text>')

    # right: scale x segment heatmap
    x1 = 2 * PAD + PW
    b.append(f'<text x="{x1+PW/2:.0f}" y="24" text-anchor="middle" '
             'font-size="14" font-weight="bold">尺度 × 段 难度矩阵'
             '（bits/slot）</text>')
    ch = (H - 2 * PAD + 12) / K
    cw = PW / S
    lo = min(min(r) for r in segm)
    hi = max(max(r) for r in segm)
    for k in range(K):
        for s in range(S):
            v = segm[k][s]
            t = (v - lo) / (hi - lo)
            rcol = int(255 - 160 * t)
            b.append(f'<rect x="{x1+s*cw:.1f}" y="{PAD-12+k*ch:.1f}" '
                     f'width="{cw:.1f}" height="{ch:.1f}" '
                     f'fill="rgb(255,{rcol},{max(rcol-30,60)})" '
                     'stroke="#fff"/>')
            b.append(f'<text x="{x1+(s+0.5)*cw:.1f}" '
                     f'y="{PAD-12+(k+0.62)*ch:.1f}" text-anchor="middle" '
                     f'font-size="10">{v:.2f}</text>')
        b.append(f'<text x="{x1-6}" y="{PAD-12+(k+0.62)*ch:.1f}" '
                 f'text-anchor="end" font-size="9">q{scales[k]}</text>')
    for s in range(S):
        b.append(f'<text x="{x1+(s+0.5)*cw:.1f}" y="{H-PAD+16}" '
                 f'text-anchor="middle" font-size="10">seg{s}</text>')
    b.append("</svg>")
    out = Path("docs/figures/scale_seg_difficulty.svg")
    out.write_text("\n".join(b))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
