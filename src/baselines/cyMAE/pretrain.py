"""Per-cohort MAE pretraining for cyMAE.

Run once per cohort. Reads `splits.json[in_panel][{cohort}][train]`, selects channels
of the included kinds (data.INCLUDED_KINDS), subsamples cells per parquet, fits per-channel
z-score statistics, and trains the cyMAE MAE recipe (mask ratio 0.25) for a fixed
epoch budget.

Outputs in `<output-dir>/{cohort}/`:
    encoder.pth        -- encoder state dict (no decoder)
    marker_order.json  -- frozen channel order + display names + kinds
    channel_stats.npz  -- per-channel mean/std fitted on the capped pretrain set
    pretrain_args.json -- CLI args + commit info for reproducibility

Example:
    uv run python -m src.baselines.cyMAE.pretrain \\
        --cohort Acute2020 \\
        --benchmark benchmark/ --data-dir data/ \\
        --output-dir artifacts/cymae/ \\
        --epochs 100 --max-cells-per-sample 10000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data as data_mod
from . import models as models_mod
from .vendor.masking_generator import RandomMaskingGenerator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-cohort cyMAE MAE pretraining")
    p.add_argument("--cohort", required=True, help="Cohort name (e.g. Acute2020)")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, required=True,
                   help="Used only to locate splits.json (benchmark/splits.json)")
    p.add_argument("--splits", type=Path, default=None,
                   help="Override path to splits.json (defaults to <benchmark>/splits.json)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--mask-ratio", type=float, default=0.25)
    p.add_argument("--max-cells-per-sample", type=int, default=10_000)
    p.add_argument("--num-workers", type=int, default=0,
                   help="Unused; cells live in memory. Kept for back-compat.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=50, help="Iters per stdout log line")
    p.add_argument("--amp-dtype", type=str, default="bf16",
                   choices=["bf16", "fp16", "off"],
                   help="Autocast dtype for forward/backward. bf16 needs Ampere+ GPU.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model (PyTorch 2+).")
    p.add_argument("--cells-on-gpu", action="store_true", default=True,
                   help="Keep the full cell tensor on GPU and shuffle indices "
                        "(skips DataLoader). Disable with --no-cells-on-gpu if OOM.")
    p.add_argument("--no-cells-on-gpu", dest="cells_on_gpu", action="store_false")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stop after this many epochs without val-loss improvement. "
                        "Set 0 to disable early stop (still capped by --epochs).")
    p.add_argument("--min-epochs", type=int, default=10,
                   help="Never early-stop before this many epochs (lets warmup finish).")
    p.add_argument("--val-max-cells-per-sample", type=int, default=10_000,
                   help="Cap val cells per sample (same default as train).")
    return p.parse_args()


def make_mask_batch(batch_size: int, num_features: int, num_mask: int,
                    device: torch.device,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Per-row random mask of shape (batch_size, num_features); each row has exactly
    `num_mask` True entries (masked-out positions). Fully vectorized on `device`
    via a top-k of uniform noise — no Python loop, no host↔device copy."""
    rand = torch.rand(batch_size, num_features, device=device, generator=generator)
    idx = rand.topk(num_mask, dim=1).indices
    mask = torch.zeros(batch_size, num_features, dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return mask


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    # Mask RNG lives on the same device as compute so we can vectorize it.
    mask_gen = torch.Generator(device=device)
    mask_gen.manual_seed(args.seed)

    splits_path = args.splits or (args.benchmark / "splits.json")
    out_dir = args.output_dir / args.cohort
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load cohort metadata -----
    meta_path = args.data_dir / args.cohort / "parquet" / "meta.json"
    meta = json.loads(meta_path.read_text())
    channel_marker_map = {ch: info[0] for ch, info in meta["channels"].items()}
    marker_kinds = {ch: (info[1] if len(info) > 1 else "unknown")
                    for ch, info in meta["channels"].items()}

    channel_order = data_mod.select_channels(meta["channels"])
    if not channel_order:
        raise RuntimeError(f"No usable channels in {meta_path}")
    print(f"[{args.cohort}] selected {len(channel_order)} channels: "
          f"{[channel_marker_map[c] for c in channel_order[:8]]}{' ...' if len(channel_order) > 8 else ''}")

    # ----- Load train cells -----
    train_samples = data_mod.cohort_train_samples(splits_path, args.cohort, "train")
    if not train_samples:
        raise RuntimeError(f"No train samples for {args.cohort} in {splits_path}")
    print(f"[{args.cohort}] {len(train_samples)} train samples; "
          f"capping {args.max_cells_per_sample} cells/sample")
    cells = data_mod.load_concatenated_cells(
        args.data_dir, args.cohort, train_samples, channel_order,
        max_cells_per_sample=args.max_cells_per_sample, seed=args.seed,
    )
    print(f"[{args.cohort}] concatenated cells: shape={cells.shape}")

    # ----- Fit & apply z-score -----
    mean, std = data_mod.fit_zscore(cells)
    cells = data_mod.apply_zscore(cells, mean, std)
    data_mod.save_channel_artifacts(
        out_dir, args.cohort, channel_order, mean, std, channel_marker_map, marker_kinds,
    )

    # ----- Build model -----
    num_features = len(channel_order)
    num_mask = max(1, int(round(args.mask_ratio * num_features)))
    if num_mask >= num_features:
        raise ValueError(f"mask_ratio={args.mask_ratio} masks every token (num_features={num_features})")
    model = models_mod.build_mae(num_features=num_features).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                              betas=(0.9, 0.95))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.cohort}] model params: {n_params/1e6:.2f}M, num_features={num_features}, "
          f"num_mask={num_mask} (ratio={args.mask_ratio})")

    if args.compile:
        model = torch.compile(model)

    # AMP setup. bf16 needs no GradScaler; fp16 does.
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp_dtype]
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype is torch.float16 and device.type == "cuda"))

    # ----- Cells stay on GPU; we shuffle indices instead of going through DataLoader.
    cells_t = torch.from_numpy(cells)
    if args.cells_on_gpu and device.type == "cuda":
        cells_t = cells_t.to(device)
    else:
        cells_t = cells_t.pin_memory() if device.type == "cuda" else cells_t
    N = cells_t.shape[0]
    iters_per_epoch = N // args.batch_size  # drop_last
    print(f"[{args.cohort}] batches/epoch: {iters_per_epoch}, "
          f"cells on {'GPU' if cells_t.is_cuda else 'host'}, "
          f"AMP={args.amp_dtype}, compile={args.compile}")

    # ----- Load val cells (same channel order, train-fit z-score). -----
    val_samples = data_mod.cohort_train_samples(splits_path, args.cohort, "val")
    val_cells_t: torch.Tensor | None = None
    if val_samples:
        val_arr = data_mod.load_concatenated_cells(
            args.data_dir, args.cohort, val_samples, channel_order,
            max_cells_per_sample=args.val_max_cells_per_sample, seed=args.seed + 1,
        )
        val_arr = data_mod.apply_zscore(val_arr, mean, std)
        val_cells_t = torch.from_numpy(val_arr)
        if args.cells_on_gpu and device.type == "cuda":
            val_cells_t = val_cells_t.to(device)
        print(f"[{args.cohort}] val: {len(val_samples)} samples, {val_cells_t.shape[0]} cells")
    else:
        print(f"[{args.cohort}] no val samples — early stop disabled, running full --epochs")

    # ----- Cosine LR with linear warmup -----
    warmup_epochs = min(5, max(1, args.epochs // 20))
    total_steps = args.epochs * iters_per_epoch
    warmup_steps = warmup_epochs * iters_per_epoch

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return args.lr * (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # ----- Val eval (deterministic mask each call so val-loss is comparable). -----
    @torch.no_grad()
    def eval_val() -> float | None:
        if val_cells_t is None:
            return None
        model.eval()
        eval_gen = torch.Generator(device=device)
        eval_gen.manual_seed(args.seed + 1234)
        Nv = val_cells_t.shape[0]
        running_v = torch.zeros((), device=device)
        seen_v = 0
        for s in range(0, Nv, bs):
            xv = val_cells_t[s : s + bs]
            if xv.shape[0] < num_features:  # need at least one full row
                continue
            if not xv.is_cuda and device.type == "cuda":
                xv = xv.to(device, non_blocking=True)
            mv = make_mask_batch(xv.shape[0], num_features, num_mask, device, eval_gen)
            tgt = xv[mv].view(xv.shape[0], num_mask)
            if use_amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    pv = model(xv, mv).squeeze(-1)
                    lv = F.mse_loss(pv, tgt)
            else:
                pv = model(xv, mv).squeeze(-1)
                lv = F.mse_loss(pv, tgt)
            running_v += lv.detach() * xv.shape[0]
            seen_v += xv.shape[0]
        model.train()
        return float(running_v.item()) / max(1, seen_v)

    # ----- Train loop -----
    model.train()
    step = 0
    t0 = time.time()
    losses_per_epoch: list[float] = []
    val_losses_per_epoch: list[float | None] = []
    best_val: float = float("inf")
    best_epoch: int = -1
    best_state: dict | None = None
    bad_epochs: int = 0
    early_stop = args.patience > 0 and val_cells_t is not None
    bs = args.batch_size
    last_epoch_done = 0
    for epoch in range(args.epochs):
        last_epoch_done = epoch + 1
        # Shuffle index tensor; on-device randperm avoids host RNG.
        perm = torch.randperm(N, device=cells_t.device)
        running = torch.zeros((), device=device)
        n_seen = 0
        for it in range(iters_per_epoch):
            for g in optim.param_groups:
                g["lr"] = lr_at(step)
            batch_idx = perm[it * bs : (it + 1) * bs]
            x = cells_t.index_select(0, batch_idx)
            if not x.is_cuda and device.type == "cuda":
                x = x.to(device, non_blocking=True)
            mask = make_mask_batch(x.shape[0], num_features, num_mask, device, mask_gen)
            target = x[mask].view(x.shape[0], num_mask)

            if use_amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    pred = model(x, mask).squeeze(-1)
                    loss = F.mse_loss(pred, target)
            else:
                pred = model(x, mask).squeeze(-1)
                loss = F.mse_loss(pred, target)

            optim.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            # Accumulate on-device; no per-step .item() sync.
            running += loss.detach() * x.shape[0]
            n_seen += x.shape[0]
            if step % args.log_every == 0:
                # Single sync per log line.
                print(f"[{args.cohort}] epoch {epoch+1}/{args.epochs} step {step} "
                      f"loss {loss.item():.4f} lr {lr_at(step):.2e}")
            step += 1
        epoch_loss = float(running.item()) / max(1, n_seen)
        losses_per_epoch.append(epoch_loss)
        val_loss = eval_val()
        val_losses_per_epoch.append(val_loss)
        elapsed = time.time() - t0
        v_str = f"val_loss={val_loss:.4f}" if val_loss is not None else "val_loss=NA"
        print(f"[{args.cohort}] epoch {epoch+1}/{args.epochs} train_loss={epoch_loss:.4f} "
              f"{v_str} elapsed={elapsed:.0f}s")

        if val_loss is not None:
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_epoch = epoch + 1
                raw = getattr(model, "_orig_mod", model)
                best_state = {k: v.detach().cpu().clone()
                              for k, v in raw.encoder.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

        if early_stop and (epoch + 1) >= args.min_epochs and bad_epochs >= args.patience:
            print(f"[{args.cohort}] early stop at epoch {epoch+1}: "
                  f"no val-loss improvement for {bad_epochs} epochs "
                  f"(best epoch={best_epoch}, best val={best_val:.4f})")
            break

    # ----- Save encoder + metadata -----
    raw_model = getattr(model, "_orig_mod", model)  # unwrap torch.compile if used
    if best_state is not None:
        save_state = best_state
        save_epoch = best_epoch
        print(f"[{args.cohort}] saving best-val checkpoint from epoch {best_epoch} "
              f"(val_loss={best_val:.4f})")
    else:
        save_state = raw_model.encoder.state_dict()
        save_epoch = last_epoch_done
        print(f"[{args.cohort}] saving final encoder from epoch {save_epoch} "
              f"(no val available)")
    torch.save({
        "encoder": save_state,
        "encoder_embed_dim": models_mod.ENCODER_EMBED_DIM,
        "encoder_depth": models_mod.ENCODER_DEPTH,
        "encoder_num_heads": models_mod.ENCODER_HEADS,
        "num_features": num_features,
        "epoch": save_epoch,
        "best_val_loss": best_val if best_state is not None else None,
        "ran_epochs": last_epoch_done,
    }, out_dir / "encoder.pth")
    (out_dir / "pretrain_args.json").write_text(json.dumps({
        "cohort": args.cohort,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_train_samples": len(train_samples),
        "n_train_cells": int(cells.shape[0]),
        "n_val_samples": len(val_samples) if val_samples else 0,
        "n_val_cells": int(val_cells_t.shape[0]) if val_cells_t is not None else 0,
        "num_features": num_features,
        "num_mask": num_mask,
        "ran_epochs": last_epoch_done,
        "best_epoch": best_epoch if best_state is not None else None,
        "best_val_loss": best_val if best_state is not None else None,
        "losses_per_epoch": losses_per_epoch,
        "val_losses_per_epoch": val_losses_per_epoch,
    }, indent=2))
    print(f"[{args.cohort}] saved encoder + artifacts to {out_dir}/")


if __name__ == "__main__":
    main()
