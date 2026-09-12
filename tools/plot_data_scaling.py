"""Data-scaling figures (2D M8 + CFG + HMAR) from results/scaling/.

Three panels, hand-written SVG (no matplotlib):
  A. test/val loss vs training tokens (log-x) — the main scaling curve;
  B. per-scale scissor: coarse (q1-q32) vs fine (q128-q1024) test loss vs
     tokens — the CADENCE-specific signature;
  C. generation quality (wiki R1, wiki MAUVE) vs tokens where full chains exist.
"""
from __future__ import annotations
import json, math
from pathlib import Path

D = json.loads(Path("results/scaling/data_scaling_progress.json").read_text())
pts = sorted(D["points"], key=lambda p: p["tokens_B"])


def val_of(p):
    return p.get("val_seg_bits") or p.get("base_val_seg_bits")


def gen_of(p):
    t = p.get("test")
    if isinstance(t, dict):
        r1, r2, mv = t["wiki"].split("/")
        return float(r1), float(mv)
    return None


W, H, PADL, PADR, PADT, PADB = 340, 300, 52, 20, 34, 46
PW, PH = W - PADL - PADR, H - PADT - PADB
TOK = [p["tokens_B"] for p in pts]
LX = [math.log10(t) for t in TOK]
xlo, xhi = min(LX), max(LX)


def X(t):
    return PADL + (math.log10(t) - xlo) / (xhi - xlo) * PW


def panel(ox, title, ylab, series, ylo, yhi, ticks):
    p = [f'<text x="{ox+PADL+PW/2:.0f}" y="18" text-anchor="middle" '
         f'font-size="12" font-weight="bold">{title}</text>',
         f'<rect x="{ox+PADL}" y="{PADT}" width="{PW}" height="{PH}" '
         'fill="none" stroke="#888"/>']
    def Y(v):
        return PADT + PH - (v - ylo) / (yhi - ylo) * PH
    for gv in ticks:
        p.append(f'<line x1="{ox+PADL}" y1="{Y(gv):.1f}" x2="{ox+PADL+PW}" '
                 f'y2="{Y(gv):.1f}" stroke="#eee"/>')
        p.append(f'<text x="{ox+PADL-5}" y="{Y(gv)+3:.1f}" text-anchor="end" '
                 f'font-size="9">{gv:g}</text>')
    for t in (2, 6.4, 25.6, 102.4, 204.8):
        if TOK[0] <= t <= TOK[-1]:
            p.append(f'<text x="{ox+X(t):.1f}" y="{PADT+PH+13}" '
                     f'text-anchor="middle" font-size="8">{t:g}</text>')
    p.append(f'<text x="{ox+PADL+PW/2:.0f}" y="{H-6}" text-anchor="middle" '
             'font-size="9">training tokens (B, log)</text>')
    p.append(f'<text x="{ox+12}" y="{PADT-6}" font-size="9">{ylab}</text>')
    cols = ["#c0392b", "#2471a3", "#1e8449"]
    for i, (nm, xs, ys) in enumerate(series):
        col = cols[i % 3]
        pl = " ".join(f"{ox+X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))
        p.append(f'<polyline points="{pl}" fill="none" stroke="{col}" '
                 'stroke-width="1.8"/>')
        for x, y in zip(xs, ys):
            p.append(f'<circle cx="{ox+X(x):.1f}" cy="{Y(y):.1f}" r="2.4" '
                     f'fill="{col}"/>')
        p.append(f'<text x="{ox+PADL+8}" y="{PADT+12+13*i}" font-size="9" '
                 f'fill="{col}">{nm}</text>')
    return p


# panel A: test + val overall loss
tv = [(p["tokens_B"], p["test_seg_bits"]) for p in pts if "test_seg_bits" in p]
vv = [(p["tokens_B"], val_of(p)) for p in pts if val_of(p)]
A = panel(0, "A. loss vs tokens", "seg-bits",
          [("test", [x for x, _ in tv], [y for _, y in tv]),
           ("val", [x for x, _ in vv], [y for _, y in vv])],
          3.0, 5.0, [3.0, 3.5, 4.0, 4.5, 5.0])
# panel B: per-scale scissor (test)
co = [(p["tokens_B"], p["test_coarse_q1_32"]) for p in pts if "test_coarse_q1_32" in p]
fi = [(p["tokens_B"], p["test_fine_q128_1024"]) for p in pts if "test_fine_q128_1024" in p]
B = panel(W, "B. per-scale (test)", "seg-bits",
          [("coarse q1-32", [x for x, _ in co], [y for _, y in co]),
           ("fine q128-1024", [x for x, _ in fi], [y for _, y in fi])],
          2.4, 4.8, [2.5, 3.0, 3.5, 4.0, 4.5])
# panel C: generation quality where available
gq = [(p["tokens_B"], gen_of(p)) for p in pts if gen_of(p)]
C = panel(2 * W, "C. gen quality (wiki)", "score",
          [("R1 (/100)", [t for t, _ in gq], [g[0] / 100 for _, g in gq]),
           ("MAUVE", [t for t, _ in gq], [g[1] for _, g in gq])],
          0.0, 0.35, [0.1, 0.2, 0.3])
# panel D: judge Gen-PPL (log10) across 4 benchmarks — the deterministic
# fluency axis; monotone down = no repeated-data overfitting knee.
pp = [p for p in pts if p.get("extra")]


def ppl_series(short):
    return ([p["tokens_B"] for p in pp],
            [math.log10(p["extra"][short]["gen_ppl"]) for p in pp])


D = panel(3 * W, "D. judge Gen-PPL (log)", "log10 PPL",
          [("wiki", *ppl_series("wiki")),
           ("ws", *ppl_series("ws")),
           ("ts", *ppl_series("ts"))],
          1.8, 3.0, [2.0, 2.3, 2.5, 2.7, 3.0])

svg = ['<svg xmlns="http://www.w3.org/2000/svg" '
       f'width="{4*W}" height="{H}" font-family="Helvetica,Arial">']
svg += A + B + C + D + ["</svg>"]
out = Path("docs/figures/data_scaling.svg")
out.write_text("\n".join(svg))
print(f"wrote {out}  ({len(tv)} loss pts, {len(gq)} gen pts, {len(pp)} ppl pts)")
