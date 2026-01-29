#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShrinkTrace v6 — integrate calibration into forward, add adaptive (event-triggered) updates,
support sample-based quantile estimation, and allow optional sparse attention.

Usage examples:

python test_shrinking_scale_cache_v6.py --steps 120 --K 50 --mode shrink
python test_shrinking_scale_cache_v6.py --steps 120 --mode adaptive --K_min 20 --K_max 100 \
       --trigger_hi 1.15 --trigger_lo 0.85 --calib_sample 65536

"""

import argparse
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device(force_cpu: bool = False) -> torch.device:
    if (not force_cpu) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def quantile_scale(x: torch.Tensor, q: float, sample: int = 0) -> torch.Tensor:
    """Compute scale using |x| quantile q with optional random sample."""
    if sample > 0 and x.numel() > sample:
        # sample uniformly from the tensor
        idx = torch.randperm(x.numel(), device=x.device)[:sample]
        a = x.view(-1)[idx].abs()
    else:
        a = x.abs()
    v = torch.quantile(a.float(), q)
    v = torch.clamp(v, min=1e-6)
    return v / 127.0


def fake_int8_quant_dequant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    q = torch.round(x / scale).clamp(-127, 127)
    return q * scale


class QuantLinear(nn.Module):
    """Linear layer with optional quantization on output."""
    def __init__(self, in_features: int, out_features: int, use_quant: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.use_quant = use_quant
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, x: torch.Tensor, scale_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.lin(x)
        if not self.use_quant:
            return y
        s = self.scale if scale_override is None else scale_override
        return fake_int8_quant_dequant(y, s)


class TinyMLP(nn.Module):
    """Minimal MLP + embedding; can return activations for calibration."""
    def __init__(self, d=1024, h=4096, vocab=50000, use_quant=True):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.l1 = QuantLinear(d, h, use_quant=use_quant)
        self.l2 = QuantLinear(h, d, use_quant=use_quant)
        self.head = nn.Linear(d, vocab)

    def forward(
        self,
        input_ids: torch.Tensor,
        scales: Optional[Dict[str, torch.Tensor]] = None,
        collect: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        x = self.embed(input_ids)  # [B,T,D]
        b, t, d = x.shape
        x = x.view(b * t, d)

        s1 = scales.get("l1") if scales is not None else None
        h_act = self.l1.lin(x)
        h_act_q = fake_int8_quant_dequant(h_act, s1) if s1 is not None else h_act
        h_out = F.gelu(h_act_q)

        s2 = scales.get("l2") if scales is not None else None
        out_act = self.l2.lin(h_out)
        out_act_q = fake_int8_quant_dequant(out_act, s2) if s2 is not None else out_act
        x2 = out_act_q.view(b, t, d)

        logits = self.head(x2)
        if collect:
            # return raw activations for calibration
            return logits, {"l1": h_act, "l2": out_act}
        return logits, None


@dataclass
class CalibState:
    K: int
    q: float
    sample: int
    K_min: int
    K_max: int
    trigger_hi: float
    trigger_lo: float

    step: int = 0
    last_update: int = 0
    scales: Dict[str, torch.Tensor] = None
    history: Dict[str, List[float]] = None

    def __post_init__(self):
        self.scales = {}
        self.history = {"l1": [], "l2": []}

    def need_update_static(self) -> bool:
        return (self.step - self.last_update) >= self.K

    def need_update_adaptive(self, acts: Dict[str, torch.Tensor], collect=False) -> bool:
        """Adaptive condition: update if any layer's scale ratio leaves [trigger_lo,trigger_hi]."""
        if (self.step - self.last_update) < self.K_min:
            return False
        if (self.step - self.last_update) >= self.K_max:
            return True
        for name, act in acts.items():
            old_scale = self.scales.get(name)
            if old_scale is None:
                return True
            new_scale = quantile_scale(act, self.q, self.sample)
            ratio = (new_scale / (old_scale + 1e-9)).clamp(min=1e-9).item()
            # also record ratio if collecting stats
            if collect:
                self.history[name].append(ratio)
            if ratio >= self.trigger_hi or ratio <= self.trigger_lo:
                return True
        return False

    def update_scales(self, acts: Dict[str, torch.Tensor]):
        for name, act in acts.items():
            self.scales[name] = quantile_scale(act, self.q, self.sample)
        self.last_update = self.step

    def end_step(self):
        self.step += 1


def run_training(
    mode: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    d: int,
    h: int,
    vocab: int,
    calib_state: Optional[CalibState],
    lr: float,
    warmup_ratio: float,
    device: torch.device,
) -> Tuple[List[float], float, int]:
    model = TinyMLP(d=d, h=h, vocab=vocab, use_quant=(mode != "no_quant")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    times = []
    losses = []
    warmup_steps = int(max(0, min(steps - 1, round(steps * warmup_ratio))))

    for step in range(steps):
        input_ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)

        t0 = time.perf_counter()
        if mode == "no_quant":
            logits, _ = model(input_ids, scales=None, collect=False)
        elif mode == "per_step":
            # always update scales from fresh activations
            logits, acts = model(input_ids, scales=None, collect=True)
            calib_state.update_scales(acts)
            logits, _ = model(input_ids, scales=calib_state.scales, collect=False)
        elif mode == "shrink":
            # static K: update every K steps
            if calib_state.need_update_static():
                # get acts with collect
                logits, acts = model(input_ids, scales=None, collect=True)
                calib_state.update_scales(acts)
            logits, _ = model(input_ids, scales=calib_state.scales, collect=False)
        elif mode == "adaptive":
            # update if adaptive triggers
            need = False
            # we always run forward with current scales to compute logits
            logits, acts = model(input_ids, scales=calib_state.scales if calib_state.scales else None, collect=True)
            need = calib_state.need_update_adaptive(acts)
            if need:
                calib_state.update_scales(acts)
                logits, acts = model(input_ids, scales=calib_state.scales, collect=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # classification loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = crit(shift_logits.view(-1, vocab), shift_labels.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        times.append(dt)
        losses.append(loss.item())
        if calib_state is not None:
            calib_state.end_step()

    p50 = torch.quantile(torch.tensor(times[warmup_steps:]), 0.50).item()
    return times, losses[-1], calib_state.last_update if calib_state is not None else 0


def summarize_times(times: List[float], warmup_steps: int):
    """Return statistics for step times after warmup."""
    tail = times[warmup_steps:]
    if not tail:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "tail_width": float("nan")}
    t = torch.tensor(tail)
    p50 = torch.quantile(t, 0.5).item()
    p95 = torch.quantile(t, 0.95).item()
    p99 = torch.quantile(t, 0.99).item()
    return {
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "tail_width": p99 - p50,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--h", type=int, default=2048)
    parser.add_argument("--vocab", type=int, default=50000)
    parser.add_argument("--mode", type=str, default="shrink", choices=["no_quant", "per_step", "shrink", "adaptive"])
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--K_min", type=int, default=10)
    parser.add_argument("--K_max", type=int, default=100)
    parser.add_argument("--trigger_hi", type=float, default=1.20)
    parser.add_argument("--trigger_lo", type=float, default=0.80)
    parser.add_argument("--q", type=float, default=0.999)
    parser.add_argument("--calib_sample", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tail_fit", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device(force_cpu=args.cpu)
    warmup_steps = int(max(0, min(args.steps - 1, round(args.steps * args.warmup))))

    modes = [args.mode] if args.mode != "all" else ["no_quant", "per_step", "shrink", "adaptive"]
    results = {}
    for mode in modes:
        times_all = []
        last_loss = None
        total_updates = 0
        for r in range(args.repeats):
            seed = args.seed + r
            calib = None
            if mode in ("per_step", "shrink", "adaptive"):
                calib = CalibState(
                    K=args.K,
                    q=args.q,
                    sample=args.calib_sample,
                    K_min=args.K_min,
                    K_max=args.K_max,
                    trigger_hi=args.trigger_hi,
                    trigger_lo=args.trigger_lo,
                )
            t, loss, updates = run_training(
                mode=mode,
                steps=args.steps,
                batch_size=args.batch,
                seq_len=args.seq,
                d=args.d,
                h=args.h,
                vocab=args.vocab,
                calib_state=calib,
                lr=args.lr,
                warmup_ratio=args.warmup,
                device=device,
            )
            times_all.extend(t)
            last_loss = loss
            total_updates += updates
        stat = summarize_times(times_all, warmup_steps)
        results[mode] = stat
        results[mode]["last_loss"] = last_loss
        results[mode]["updates"] = total_updates / max(1, args.repeats)
        if args.tail_fit:
            # rough power-law tail slope indicator
            alphas = [_tail_powerlaw_alpha(times_all[warmup_steps:], tail_frac=0.2)]
            # ignore nan
            alphas = [a for a in alphas if a == a]
            results[mode]["tail_alpha"] = sum(alphas) / len(alphas) if alphas else float("nan")

    # Print summary
    print(f"\nSummary over {args.repeats} repeats:")
    for mode in modes:
        s = results[mode]
        extra = f" tail_alpha≈{s['tail_alpha']:.2f}" if args.tail_fit else ""
        print(
            f"{mode:>10s} | p50={s['p50']:.4f}s  p95={s['p95']:.4f}s  p99={s['p99']:.4f}s  "
            f"(p99-p50={s['tail_width']:.4f}s)  updates={s['updates']:.1f}  last_loss={s['last_loss']:.4f}{extra}"
        )

if __name__ == "__main__":
    main()
