"""Post-hoc re-scoring of existing gens files: MAUVE@1024, Gen-PPL, entropy.

Protocol change (2026-09-08, user direction): MAUVE's featurization length
moves from 256 to the featurizer's full 1024 context, and Gen-PPL / unigram
entropy join as primary metrics (ROUGE demoted to auxiliary). This tool
re-scores WITHOUT regenerating, so every number stays tied to the exact
registered gens files. Disclosed as a post-hoc metric change everywhere the
numbers appear.

Metrics per gens JSONL ({prompt, reference, generated} rows):
  mauve_1024      MAUVE with max_text_length=1024 (featurizer/buckets
                  otherwise identical to eval_generation.py, so rows stay
                  comparable to the 256 tables within the same n).
  gen_ppl         exp(corpus NLL / corpus tokens) of the GENERATED
                  continuation under gpt2-large, conditioned on the prompt
                  (prompt left-truncated to fit the 1024 context; loss only
                  on generated tokens). gpt2-large is judge-only — the same
                  external model MAUVE's featurizer already uses.
  gen_ppl_seqmean mean of per-sequence PPLs (the length-insensitive variant;
                  reported alongside until "GNPPL" is pinned to a citation).
  unigram_entropy corpus unigram entropy (bits, gpt2 tokens) of generations.
  ref_gen_ppl / ref_unigram_entropy: same numbers for the references —
                  the anchor that makes "low PPL = degenerate?" readable
                  (SSD-LM lesson: word-soup can score LOW on judge PPL).

Usage:
  python tools/eval_extra.py --gen <gens.jsonl> --out <out.json> [--device 0]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def unigram_entropy(texts, tok, batch=256):
    counts = Counter()
    for i in range(0, len(texts), batch):
        for ids in tok(texts[i:i + batch])["input_ids"]:
            counts.update(ids)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


@torch.no_grad()
def gen_ppl(rows, key, tok, model, device, batch=8, ctx=1024):
    """NLL of rows[key] tokens given the prompt as context."""
    nll_sum, tok_sum, seq_ppls = 0.0, 0, []
    items = []
    for r in rows:
        gen_ids = tok(r[key])["input_ids"][:ctx - 1]
        if not gen_ids:
            continue
        prom_ids = tok(r["prompt"])["input_ids"]
        keep = ctx - len(gen_ids)
        prom_ids = prom_ids[-keep:] if keep > 0 else []
        items.append((prom_ids, gen_ids))
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        L = max(len(p) + len(g) for p, g in chunk)
        ids = torch.zeros(len(chunk), L, dtype=torch.long)
        att = torch.zeros(len(chunk), L, dtype=torch.long)
        lmask = torch.zeros(len(chunk), L, dtype=torch.bool)
        for j, (p, g) in enumerate(chunk):
            seq = p + g
            ids[j, :len(seq)] = torch.tensor(seq)
            att[j, :len(seq)] = 1
            lmask[j, len(p):len(seq)] = True
        ids, att, lmask = ids.to(device), att.to(device), lmask.to(device)
        logits = model(input_ids=ids, attention_mask=att).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        m = lmask[:, 1:]
        # the first generated token is predicted from the last prompt
        # position, which sits at index len(p)-1 of the shifted grid — the
        # shifted mask above covers exactly positions len(p)..len(seq)-1
        per = (nll * m).sum(1)
        cnt = m.sum(1)
        nll_sum += float(per.sum())
        tok_sum += int(cnt.sum())
        for a, b in zip(per.tolist(), cnt.tolist()):
            if b > 0:
                seq_ppls.append(math.exp(a / b))
    return (math.exp(nll_sum / max(tok_sum, 1)),
            sum(seq_ppls) / max(len(seq_ppls), 1), tok_sum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--judge", default="gpt2-large",
                    help="judge LM (smoke tests use plain gpt2)")
    ap.add_argument("--skip_mauve", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.gen).read_text().splitlines()]
    gens = [r["generated"] for r in rows]
    refs = [r["reference"] for r in rows]
    device = torch.device(f"cuda:{args.device}"
                          if torch.cuda.is_available() else "cpu")

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(args.judge)
    # bf16 only where it's hardware-native; CPU emulation is ~100x slower
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = GPT2LMHeadModel.from_pretrained(
        args.judge, torch_dtype=dtype).to(device).eval()

    report = {"gen_file": str(args.gen), "n": len(rows),
              "judge": args.judge, "protocol": "extra-metrics v1"}
    ppl, ppl_sm, ntok = gen_ppl(rows, "generated", tok, model, device)
    report.update(gen_ppl=ppl, gen_ppl_seqmean=ppl_sm, gen_tokens=ntok)
    rppl, rppl_sm, _ = gen_ppl(rows, "reference", tok, model, device)
    report.update(ref_gen_ppl=rppl, ref_gen_ppl_seqmean=rppl_sm)
    report["unigram_entropy"] = unigram_entropy(gens, tok)
    report["ref_unigram_entropy"] = unigram_entropy(refs, tok)

    if not args.skip_mauve:
        del model
        torch.cuda.empty_cache()
        import mauve
        report["mauve_1024"] = float(mauve.compute_mauve(
            p_text=refs, q_text=gens, device_id=args.device,
            max_text_length=1024, verbose=False).mauve)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
