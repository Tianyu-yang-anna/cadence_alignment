"""Pretrain a t5-small-GEOMETRY encoder on OUR corpus under OUR side budget.

The fair-resource point of the ELF attribution triple: ELF's random-encoder
arm sits on the MAUVE floor and its C4-pretrained arm (~1T tokens of external
data) tops the table, so the open question is whether PRETRAINING AT OUR OWN
RESOURCE SCALE is enough. This trains the same 35M T5EncoderModel geometry on
the same OWT2 corpus (T5-tokenized bins), with the token budget byte-matched
to our frozen tokenizer's side budget: 150k steps x 256 x 1024 = 39.3B tokens.

Objective: BERT-style MLM (15% positions; 80% -> the <extra_id_0> sentinel,
10% random, 10% kept), CE through a head TIED to the token embedding. This is
a DISCLOSED deviation from T5's span-corruption pretraining (which needs a
decoder we would immediately discard); what ELF consumes is contextual
embeddings, which MLM provides.

Self-contained like train_elf.py (no repo utils/ imports).

Usage:
  torchrun --nproc_per_node=8 pretrain_t5enc.py --data_dir <owt2_t5> \
      --run_dir runs/t5enc_owt2 --steps 150000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

SEQ_LEN = 1024


class WindowDataset(Dataset):
    def __init__(self, bin_path: Path):
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.n = len(self.tokens) // SEQ_LEN

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        w = np.asarray(self.tokens[i * SEQ_LEN:(i + 1) * SEQ_LEN],
                       dtype=np.int64)
        return torch.from_numpy(w)


def log0(msg, rank):
    if rank == 0:
        print(f"[t5enc] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--global_batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = (torch.device(f"cuda:{local_rank}")
              if torch.cuda.is_available() else torch.device("cpu"))

    meta = json.loads((Path(args.data_dir) / "meta.json").read_text())
    assert "t5" in meta["tokenizer"]
    from transformers import AutoTokenizer, T5Config, T5EncoderModel
    tok = AutoTokenizer.from_pretrained(meta["tokenizer"], use_fast=True)
    vocab = len(tok)
    sentinel = tok.convert_tokens_to_ids("<extra_id_0>")

    torch.manual_seed(args.seed)  # identical init everywhere
    cfg = T5Config.from_pretrained("t5-small")
    model = T5EncoderModel(cfg).to(device)
    d_model = cfg.d_model
    n_params = sum(p.numel() for p in model.parameters())

    ds = WindowDataset(Path(args.data_dir) / "train.bin")
    assert args.global_batch % world == 0
    micro = args.global_batch // world
    sampler = (DistributedSampler(ds, num_replicas=world, rank=rank,
                                  shuffle=True, drop_last=True)
               if world > 1 else None)
    loader = DataLoader(ds, batch_size=micro, sampler=sampler,
                        shuffle=sampler is None, num_workers=args.num_workers,
                        drop_last=True, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    if world > 1:
        model = DDP(model, device_ids=[local_rank])
    inner = model.module if hasattr(model, "module") else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(args.warmup, 1)) * 0.5 *
        (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    run_dir = Path(args.run_dir)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps({
            "geometry": "t5-small", "n_params": n_params, "vocab": vocab,
            "steps": args.steps, "global_batch": args.global_batch,
            "tokens_budget": args.steps * args.global_batch * SEQ_LEN,
            "objective": "mlm-15 (80 sentinel / 10 random / 10 keep), "
                         "tied-embedding head; disclosed deviation from T5 "
                         "span corruption"}, indent=2))
    step = 0
    latest = run_dir / "latest.txt"
    if latest.exists():  # resubmit-to-resume
        ck = run_dir / latest.read_text().strip()
        if ck.exists():
            pl = torch.load(ck, map_location=device, weights_only=False)
            inner.load_state_dict(pl["model"])
            opt.load_state_dict(pl["opt"])
            sched.load_state_dict(pl["sched"])
            step = int(pl["step"])
            log0(f"resumed at step {step}", rank)

    def save(tag):
        if rank != 0:
            return
        torch.save({"model": inner.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step},
                   run_dir / f"ckpt_{tag}.pt")
        latest.write_text(f"ckpt_{tag}.pt\n")

    log0(f"params {n_params/1e6:.1f}M vocab {vocab} windows {len(ds)} "
         f"micro {micro}x{world} budget "
         f"{args.steps*args.global_batch*SEQ_LEN/1e9:.1f}B", rank)
    g = torch.Generator(device="cpu").manual_seed(args.seed + rank)
    mf = open(run_dir / "metrics.jsonl", "a") if rank == 0 else None
    win, t_last, epoch = {"loss": 0.0, "acc": 0.0, "n": 0}, time.time(), 0
    it = iter(loader)
    model.train()
    while step < args.steps:
        try:
            ids = next(it)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            it = iter(loader)
            ids = next(it)
        ids = ids.to(device, non_blocking=True)
        B = ids.shape[0]
        r = torch.rand(B, SEQ_LEN, generator=g).to(device)
        masked = r < args.mask_prob
        # ensure at least one masked position per sample (loss well-defined)
        masked[:, 0] |= ~masked.any(dim=1)
        corrupt = ids.clone()
        u = torch.rand(B, SEQ_LEN, generator=g).to(device)
        corrupt[masked & (u < 0.8)] = sentinel
        rnd = torch.randint(0, vocab, (B, SEQ_LEN), generator=g).to(device)
        corrupt[masked & (u >= 0.8) & (u < 0.9)] = \
            rnd[masked & (u >= 0.8) & (u < 0.9)]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
            h = model(input_ids=corrupt,
                      attention_mask=torch.ones_like(ids)).last_hidden_state
        # tied head in fp32, with T5's own tied-head convention: the hidden
        # is scaled by d_model**-0.5 before the embedding product (without it
        # the logits explode and CE starts ~200 instead of ~ln(V)=10.4)
        logits = (h.float() * (d_model ** -0.5)) \
            @ inner.shared.weight.float().t()
        ce = F.cross_entropy(logits[masked], ids[masked])
        opt.zero_grad(set_to_none=True)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        with torch.no_grad():
            acc = (logits[masked].argmax(-1) == ids[masked]).float().mean()
        win["loss"] += float(ce)
        win["acc"] += float(acc)
        win["n"] += 1
        if rank == 0 and step % args.log_every == 0:
            n = max(win["n"], 1)
            rec = {"step": step, "loss": win["loss"] / n,
                   "mask_acc": win["acc"] / n,
                   "lr": opt.param_groups[0]["lr"],
                   "sec_per_step": (time.time() - t_last) / n}
            print(f"[t5enc] {json.dumps(rec)}", flush=True)
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            win, t_last = {"loss": 0.0, "acc": 0.0, "n": 0}, time.time()
        if step % args.save_every == 0:
            save(f"step{step}")
    save(f"step{args.steps}")
    log0("done", rank)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
