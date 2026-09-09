"""Gates for the throughput benchmark path (generate.run_benchmark_batched).

The registered quality rows never touch this path (gen_batch=1 is the
untouched row-by-row run_benchmark), so the gates cover the protocol pieces
the batched path re-implements: suffix truncation of prompts, word-truncation
to the reference length, row order/indices across batch boundaries, and the
chained-row rejection.
"""
import json

import torch

from generate import run_benchmark, run_benchmark_batched


class _Detok:
    """Whitespace tokenizer stub: token id = word length (content-free)."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [len(w) for w in text.split()]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join("w" * max(int(i), 1) for i in ids)


def _rows(n):
    return [{"prompt": f"p{i} " * 4, "reference": "r " * 6} for i in range(n)]


def test_batched_matches_rowwise_protocol(tmp_path):
    detok = _Detok()
    def echo_batch(cur_list):
        return torch.stack([torch.nn.functional.pad(c, (0, 8 - c.numel()),
                                                    value=2) for c in cur_list])
    def echo_single(cur, generator=None):
        return torch.nn.functional.pad(cur, (0, 8 - cur.shape[1]), value=2)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    rows = _rows(7)
    run_benchmark(rows, detok, echo_single, 64, a, max_prompt_tokens=16)
    run_benchmark_batched(rows, detok, echo_batch, 64, b,
                          max_prompt_tokens=16, gen_batch=3)
    ra = [json.loads(l) for l in a.read_text().splitlines()]
    rb = [json.loads(l) for l in b.read_text().splitlines()]
    assert [r["index"] for r in rb] == list(range(7))  # order kept over batches
    for x, y in zip(ra, rb):
        assert x["prompt"] == y["prompt"] and x["reference"] == y["reference"]
        assert x["generated"] == y["generated"]  # same stub -> same protocol
        assert len(y["generated"].split()) <= len(y["reference"].split())


def test_batched_rejects_chained_rows(tmp_path):
    detok = _Detok()
    rows = [{"prompt": "p", "reference": "r " * 5000}]  # needs >1 window
    try:
        run_benchmark_batched(rows, detok, lambda cl: torch.zeros(1, 4), 64,
                              tmp_path / "o.jsonl", gen_batch=2)
    except AssertionError as e:
        assert "chained" in str(e)
    else:
        raise AssertionError("chained row must be rejected")


def test_no_cfg_training_keeps_null_prefix_in_graph():
    """cond_drop_p=0 + training mode: null_prefix must still receive a (zero)
    gradient, or multi-GPU DDP dies on step 2 (measured: b12ncf/b12ncfa25,
    2026-09-09)."""
    import torch
    from models.prefix_planner import PrefixVARPlanner

    scales = [1, 2, 4]
    S, N, d_seg = 4, 8, 4          # PQ: 4 segments x 8-entry books, d_code 16
    books = torch.randn(len(scales), S, N, d_seg)
    p = PrefixVARPlanner(scales=scales, seq_len=4, codebooks=books,
                         d_model=32, n_layers=1, n_heads=2, ffn_mult=2,
                         cond_drop_p=0.0)
    p.train()
    B, L = 2, sum(scales)
    codes = torch.zeros(B, L, S, dtype=torch.long)
    prefix_e = torch.randn(B, 4, S * d_seg)
    logits = p(codes, prefix_e,
               prefix_mask=torch.ones(B, 4, dtype=torch.bool))
    logits.float().sum().backward()
    g = p.null_prefix.grad
    assert g is not None, "null_prefix missing from the autograd graph"
    assert float(g.abs().max()) == 0.0, "all-False drop must not perturb"
