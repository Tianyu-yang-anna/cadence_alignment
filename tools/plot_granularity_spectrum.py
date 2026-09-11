"""Granularity-spectrum figure (mentor request, 2026-09-08): quality vs NFE
and latency vs NFE on the 12.8B 2D arm (b12s2e8), fine-band (C,K) swept with
everything else frozen (same ckpt, p5hot7, paired rows/seeds, batch=1).

Hand-written SVG (the venv has no matplotlib — same precedent as
plot_scale_difficulty.py). Quality numbers are read from the registered
metrics files; latencies come from the job logs' wikisource generation
window (second bench = no model-load inside the window), n=1000 each.
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path("results/benchgen_planner_prefix_owt2_pqsh_b12s2e8")
# (tag, C, K, NFE=64+6CK, latency s/sample from job logs)
CELLS = [("g1x1", 1, 1, 70, 0.50), ("g1x2", 1, 2, 76, 0.51),
         ("g1x4", 1, 4, 88, 0.54), ("g2x2", 2, 2, 88, 0.54),
         ("g2x4", 2, 4, 112, 0.62), ("g4x2", 4, 2, 112, 0.60),
         ("g4x4", 4, 4, 160, 0.78), ("g8x2", 8, 2, 160, 0.75),
         ("g8x4", 8, 4, 256, 1.08), ("g16x4", 16, 4, 448, 1.67),
         # tail cells ran 8-sharded (batch=1 per shard); latency = window/125
         ("g128x2", 128, 2, 1600, 5.69), ("gMx1", 1024, 1, 3648, 12.23),
         ("gMx4", 1024, 4, 14400, 46.0)]


def load():
    rows = []
    for tag, C, K, nfe, lat in CELLS:
        d = {}
        for b in ("wikipedia", "wikisource"):
            r = json.loads((R / f"gens_{b}_{tag}.metrics.json").read_text())
            d[b] = (r["rouge1"] * 100, r["mauve"] * 100)
        rows.append((tag, C, K, nfe, lat, d))
    return rows


def svg(rows, out="docs/figures/granularity_spectrum.svg"):
    W, H, PAD = 900, 380, 55
    PW = (W - 3 * PAD) / 2

    def panel(x0, ys, series, title, ylab, ylo, yhi):
        import math
        parts = [f'<text x="{x0+PW/2:.0f}" y="28" text-anchor="middle" '
                 f'font-size="15" font-weight="bold">{title}</text>']
        xlo, xhi = math.log(60), math.log(16000)
        def X(nfe): return x0 + (math.log(nfe) - xlo) / (xhi - xlo) * PW
        def Y(v): return H - PAD - (v - ylo) / (yhi - ylo) * (H - 2 * PAD - 20)
        parts.append(f'<rect x="{x0}" y="{PAD-15}" width="{PW:.0f}" '
                     f'height="{H-2*PAD+15}" fill="none" stroke="#999"/>')
        for nfe in (70, 256, 1600, 3648, 14400):
            parts.append(f'<text x="{X(nfe):.0f}" y="{H-PAD+16}" '
                         f'text-anchor="middle" font-size="10">{nfe}</text>')
        for gv in range(int(ylo), int(yhi) + 1, max(1, int((yhi-ylo)/5))):
            parts.append(f'<text x="{x0-6}" y="{Y(gv)+4:.0f}" '
                         f'text-anchor="end" font-size="10">{gv}</text>')
            parts.append(f'<line x1="{x0}" y1="{Y(gv):.0f}" x2="{x0+PW:.0f}" '
                         f'y2="{Y(gv):.0f}" stroke="#eee"/>')
        colors = ["#c0392b", "#2471a3", "#1e8449"]
        for (name, vals), col in zip(series, colors):
            pts = " ".join(f"{X(n):.1f},{Y(v):.1f}" for n, v in vals)
            parts.append(f'<polyline points="{pts}" fill="none" '
                         f'stroke="{col}" stroke-width="2"/>')
            for n, v in vals:
                parts.append(f'<circle cx="{X(n):.1f}" cy="{Y(v):.1f}" '
                             f'r="3" fill="{col}"/>')
            parts.append(f'<text x="{x0+8}" y="{PAD+2+14*colors.index(col)}" '
                         f'font-size="11" fill="{col}">{name}</text>')
        parts.append(f'<text x="{x0+PW/2:.0f}" y="{H-14}" '
                     f'text-anchor="middle" font-size="11">NFE（log 轴）</text>')
        parts.append(f'<text x="{x0-38}" y="{PAD-24}" font-size="11">{ylab}</text>')
        return parts

    x = sorted({r[3] for r in rows})
    def best(bench, idx):
        # per-NFE mean when two (C,K) share an NFE
        out = []
        for n in x:
            vs = [r[5][bench][idx] for r in rows if r[3] == n]
            out.append((n, sum(vs) / len(vs)))
        return out
    lat = []
    for n in x:
        vs = [r[4] for r in rows if r[3] == n]
        lat.append((n, sum(vs) / len(vs)))

    body = ['<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{W}" height="{H}" font-family="Helvetica,Arial">']
    body += panel(PAD, None,
                  [("wiki MAUVE@256 (×100)", best("wikipedia", 1)),
                   ("WS MAUVE@256 (×100)", best("wikisource", 1)),
                   ("wiki R1", best("wikipedia", 0))],
                  "质量 vs 粒度（NFE 70→14400=206× 全程平坦）", "分数", 0, 35)
    body += panel(2 * PAD + PW, None, [("时延 s/样本（batch=1）", lat)],
                  "时延 vs 粒度（0.50→46s；线性定律 206× 外推误差 +1%）", "s/样本", 0, 48)
    body.append("</svg>")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(body))
    print(f"wrote {out}")


if __name__ == "__main__":
    svg(load())
