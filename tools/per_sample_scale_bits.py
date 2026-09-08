"""Per-SAMPLE per-scale teacher-forced bits: the adaptivity evidence.

The ladder gives every window the same slot grid, so "adaptive token counts"
would be a false claim; the true claim is CONTENT-ADAPTIVE INFORMATION
ALLOCATION — where in the scale ladder the description length concentrates
depends on the content's difficulty. This dumps, for N val windows, the
per-scale mean CE (bits/slot) of each individual window under the deployed
planner (plain teacher-forced readout, mask_mode=none semantics), so the
figure can bin windows by total description length and show the per-scale
SHARE curves shift shape across difficulty quartiles.

Loading mirrors generate_prefix.py (same config/ckpt/tokenizer path); the
forward mirrors finetune_prefix_maskgit's plain path: prefix = the window's
own quantized latents, logits over the full ladder, CE per (position, seg).

Usage (inside a job):
  python tools/per_sample_scale_bits.py --config <cfg> --set run_name=<arm> \
      --set planner.tokenizer_run_dir=<tokdir> --bin <val.bin> --n 2048 \
      --out per_sample_scale_bits.json
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.prefix_planner import (PrefixVARPlanner,  # noqa: E402
                                   load_prefix_planner_state, stack_codebooks)
from train_planner import load_frozen_tokenizer  # noqa: E402
from utils.checkpoint import find_resume_ckpt, load_checkpoint  # noqa: E402
from utils.config import load_config, resolved_out_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--bin", required=True, help="uint16 token bin (val)")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, args.sets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = (torch.bfloat16
                      if cfg.train.bf16 and device.type == "cuda" else None)

    def ac():
        return (torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype else nullcontext())

    tokenizer, tok_model_cfg, tok_quant_cfg, _ = load_frozen_tokenizer(
        cfg.planner.tokenizer_run_dir, device)
    scales = tokenizer.msrvq.scales
    seq_len = tok_model_cfg.seq_len
    out_dir = resolved_out_dir(cfg)
    ckpt_path = find_resume_ckpt(out_dir)
    payload = load_checkpoint(ckpt_path, map_location=device)
    use_sampler = cfg.planner.sampler or any(
        k.startswith("sampler.") for k in payload["model"])
    planner = PrefixVARPlanner(
        scales=scales, seq_len=seq_len,
        codebooks=stack_codebooks(tokenizer.msrvq),
        d_model=cfg.planner.d_model, n_layers=cfg.planner.n_layers,
        n_heads=cfg.planner.n_heads, ffn_mult=cfg.planner.ffn_mult,
        rope_theta=cfg.planner.rope_theta,
        upsample_mode=tok_quant_cfg.upsample_mode, sampler=use_sampler,
        sampler_layers=cfg.planner.sampler_layers,
        sampler_width=cfg.planner.sampler_width,
        sampler_heads=cfg.planner.sampler_heads).to(device)
    load_prefix_planner_state(planner, payload["model"])
    planner.eval()

    toks = np.memmap(args.bin, dtype=np.uint16, mode="r")
    n_win = min(args.n, len(toks) // seq_len)
    starts = [0]
    for l in scales:
        starts.append(starts[-1] + l)
    ln2 = float(np.log(2.0))
    rows = []
    with torch.no_grad():
        for s0 in range(0, n_win, args.batch):
            B = min(args.batch, n_win - s0)
            ids = torch.stack([
                torch.from_numpy(np.asarray(
                    toks[(s0 + j) * seq_len:(s0 + j + 1) * seq_len],
                    dtype=np.int64)) for j in range(B)]).to(device)
            mask = torch.ones_like(ids, dtype=torch.bool)
            with ac():
                z = tokenizer.encode(ids, mask.long())
                ms = tokenizer.msrvq(z, update=False, mask=mask)
            codes = torch.cat(ms.codes, dim=1)  # list of [B,l_k,S] -> flat
            prefix_e = ms.z_q.float()
            with ac():
                logits = planner(codes, prefix_e, prefix_mask=mask)
            V = logits.shape[-1]
            ce = F.cross_entropy(
                logits.float().reshape(-1, V), codes.reshape(-1),
                reduction="none").reshape(B, -1, codes.shape[-1]) / ln2
            for j in range(B):
                per_scale = [float(ce[j, starts[k]:starts[k + 1]].mean())
                             for k in range(len(scales))]
                # per-scale x per-segment matrix: is difficulty segment-uniform?
                per_seg = [[float(ce[j, starts[k]:starts[k + 1], s].mean())
                            for s in range(ce.shape[-1])]
                           for k in range(len(scales))]
                total = float(sum(p * l * ce.shape[-1]
                                  for p, l in zip(per_scale, scales)))
                rows.append({"window": s0 + j, "per_scale_bits": per_scale,
                             "per_scale_seg": per_seg, "total_bits": total})
            if (s0 // args.batch) % 10 == 0:
                print(f"[psbits] {s0 + B}/{n_win}", flush=True)
    Path(args.out).write_text(json.dumps(
        {"arm": cfg.run_name, "ckpt": str(ckpt_path), "n": len(rows),
         "scales": list(scales), "segments": ce.shape[-1],
         "rows": rows}))
    print(f"[psbits] wrote {args.out} ({len(rows)} windows)", flush=True)


if __name__ == "__main__":
    main()
