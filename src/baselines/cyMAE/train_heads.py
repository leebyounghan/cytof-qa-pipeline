"""Per-(cohort, internal-step) classification head training on top of a frozen cyMAE encoder.

For each step in the cohort's gating tree, gather parent-pop cells from every train-split
task (capped at `--max-cells-per-sample` per task), embed them with the frozen encoder,
and fit a small MLP head with cross-entropy + inverse-frequency class weights. Use the
val split for early stopping on macro F1. Save one head checkpoint per step.

Run after pretrain.py. Typical layout:

    artifacts/cymae/{cohort}/encoder.pth         (from pretrain.py)
    artifacts/cymae/{cohort}/marker_order.json   (from pretrain.py)
    artifacts/cymae/{cohort}/channel_stats.npz   (from pretrain.py)
    artifacts/cymae/{cohort}/heads/step_{NN}.pth (this script writes)

Example:
    uv run python -m src.baselines.cyMAE.train_heads \\
        --cohort Acute2020 \\
        --benchmark benchmark/ --data-dir data/ \\
        --encoder-dir artifacts/cymae/Acute2020/ \\
        --output-dir  artifacts/cymae/Acute2020/heads/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from src.bench import BenchmarkLoader

from . import data as data_mod
from . import models as models_mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cyMAE per-step head training")
    p.add_argument("--cohort", required=True)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--splits", type=Path, default=None,
                   help="Path to splits.json (default: <benchmark>/splits.json)")
    p.add_argument("--encoder-dir", type=Path, required=True,
                   help="Cohort directory written by pretrain.py "
                        "(contains encoder.pth, marker_order.json, channel_stats.npz)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, nargs="*", default=None,
                   help="Restrict to these step numbers (default: all)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-cells-per-sample", type=int, default=10_000,
                   help="Cap parent-pop cells per (sample, step) task (random subsample)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-train heads whose checkpoint already exists")
    return p.parse_args()


def gather_step_data(
    loader: BenchmarkLoader,
    cohort: str,
    splits_path: Path,
    split_set: str,
    steps: list[int] | None,
    channel_order: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    encoder: nn.Module,
    max_cells_per_sample: int,
    seed: int,
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, list[str]]]:
    """Return {step_num: (embeddings, label_indices, categories)}.

    Embeddings are mean-pooled per-cell vectors (kept as GPU tensors so the head
    trainer doesn't have to re-upload them every batch). Label indices reference
    the `categories` list (order taken from the first-seen task at this step).
    """
    rng = np.random.default_rng(seed)
    by_step: dict[int, dict] = defaultdict(lambda: {"emb": [], "y": [], "cats": None})

    for ti in loader.iter_tasks(
        datasets=[cohort], split=str(splits_path), split_set=split_set, steps=steps
    ):
        cats = list(ti.categories)
        slot = by_step[ti.step]
        if slot["cats"] is None:
            slot["cats"] = cats
        elif slot["cats"] != cats:
            raise RuntimeError(
                f"Inconsistent categories across samples for step {ti.step}: "
                f"{slot['cats']} vs {cats}"
            )

        missing = [c for c in channel_order if c not in ti.expression.columns]
        if missing:
            raise RuntimeError(
                f"Missing expected channels in {ti.task_id}: {missing[:5]}{' ...' if len(missing) > 5 else ''}"
            )
        x_raw = ti.expression[channel_order].to_numpy(dtype=np.float32, na_value=0.0)

        n = len(x_raw)
        if n == 0:
            continue
        if n > max_cells_per_sample:
            idx = rng.choice(n, size=max_cells_per_sample, replace=False)
            x_raw = x_raw[idx]
            y_str = np.asarray(ti.y)[idx]
        else:
            y_str = np.asarray(ti.y)

        x_norm = data_mod.apply_zscore(x_raw, mean, std)
        # Keep emb on GPU; head trainer indexes it directly.
        emb = models_mod.embed_cells(encoder, torch.from_numpy(x_norm), return_device=device)

        cat_to_idx = {c: i for i, c in enumerate(cats)}
        y_idx = np.full(len(y_str), -1, dtype=np.int64)
        for i, lbl in enumerate(y_str):
            if lbl in cat_to_idx:
                y_idx[i] = cat_to_idx[lbl]
            elif "Unassigned" in cat_to_idx:
                y_idx[i] = cat_to_idx["Unassigned"]
        keep_np = y_idx >= 0
        if not keep_np.any():
            continue
        keep_t = torch.from_numpy(keep_np).to(device)
        slot["emb"].append(emb[keep_t])
        slot["y"].append(torch.from_numpy(y_idx[keep_np]).to(device))

    out: dict[int, tuple[torch.Tensor, torch.Tensor, list[str]]] = {}
    for step, slot in by_step.items():
        if not slot["emb"]:
            continue
        out[step] = (
            torch.cat(slot["emb"], dim=0),
            torch.cat(slot["y"], dim=0),
            slot["cats"],
        )
    return out


def class_weights_from_tensor(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=num_classes).to(torch.float64)
    n = float(y.numel())
    weights = torch.where(counts > 0, n / (num_classes * counts), torch.zeros_like(counts))
    return weights.to(torch.float32)


def train_head(
    train_emb: torch.Tensor,   # (N, D) on device
    train_y: torch.Tensor,     # (N,)   on device, int64
    val_emb: torch.Tensor | None,
    val_y: torch.Tensor | None,
    num_classes: int,
    embed_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[models_mod.StepHead, dict]:
    head = models_mod.StepHead(embed_dim=embed_dim, num_classes=num_classes).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = class_weights_from_tensor(train_y.cpu(), num_classes).to(device)

    perm_gen = torch.Generator(device=device)
    perm_gen.manual_seed(args.seed)

    best_state = None
    best_metric = -1.0
    best_epoch = -1
    bad = 0
    metrics_log = []
    N = train_emb.shape[0]

    for epoch in range(args.epochs):
        head.train()
        order = torch.randperm(N, device=device, generator=perm_gen)
        running = torch.zeros((), device=device)
        seen = 0
        for start in range(0, N, args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            xb = train_emb.index_select(0, batch_idx)
            yb = train_y.index_select(0, batch_idx)
            logits = head(xb)
            loss = F.cross_entropy(logits, yb, weight=weights)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += loss.detach() * batch_idx.numel()
            seen += batch_idx.numel()
        train_loss = float(running.item()) / max(1, seen)

        if val_emb is None or val_y is None or val_emb.numel() == 0:
            metrics_log.append({"epoch": epoch + 1, "train_loss": train_loss})
            continue

        head.eval()
        with torch.no_grad():
            val_pred_chunks = []
            val_loss_sum = torch.zeros((), device=device)
            val_seen = 0
            for start in range(0, val_emb.shape[0], args.batch_size):
                xb = val_emb[start : start + args.batch_size]
                yb = val_y[start : start + args.batch_size]
                logits = head(xb)
                val_loss_sum += F.cross_entropy(logits, yb, weight=weights, reduction="sum")
                val_pred_chunks.append(logits.argmax(dim=1))
                val_seen += yb.numel()
            val_pred = torch.cat(val_pred_chunks, dim=0).cpu().numpy()
        val_loss = float(val_loss_sum.item()) / max(1, val_seen)
        val_f1 = f1_score(val_y.cpu().numpy(), val_pred, average="macro", zero_division=0)
        metrics_log.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1_macro": float(val_f1),
        })

        if val_f1 > best_metric:
            best_metric = val_f1
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head, {"best_metric": best_metric, "best_epoch": best_epoch, "log": metrics_log}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    splits_path = args.splits or (args.benchmark / "splits.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load encoder + channel artifacts ---
    channel_order, mean, std = data_mod.load_channel_artifacts(args.encoder_dir)
    ckpt = torch.load(args.encoder_dir / "encoder.pth", map_location="cpu")
    encoder = models_mod.build_encoder(num_features=ckpt["num_features"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    embed_dim = models_mod.ENCODER_EMBED_DIM

    # --- Loader (no perturbation; raw in-panel benchmark) ---
    loader = BenchmarkLoader(args.benchmark, args.data_dir)

    print(f"[{args.cohort}] gathering train/val embeddings ...")
    train_data = gather_step_data(
        loader, args.cohort, splits_path, "train", args.steps,
        channel_order, mean, std, encoder, args.max_cells_per_sample, args.seed, device,
    )
    val_data = gather_step_data(
        loader, args.cohort, splits_path, "val", args.steps,
        channel_order, mean, std, encoder, args.max_cells_per_sample, args.seed + 1, device,
    )
    if not train_data:
        raise RuntimeError(f"No train tasks found for cohort={args.cohort}")
    print(f"[{args.cohort}] training heads for {len(train_data)} step(s)")

    summary: dict[str, dict] = {}
    for step, (tr_emb, tr_y, cats) in sorted(train_data.items()):
        out_path = args.output_dir / f"step_{step:02d}.pth"
        if out_path.exists() and not args.overwrite:
            print(f"[{args.cohort}] step {step} exists, skip (use --overwrite to retrain)")
            continue
        v_emb, v_y = (None, None)
        if step in val_data:
            v_emb, v_y, v_cats = val_data[step]
            if v_cats != cats:
                raise RuntimeError(f"train/val categories diverge at step {step}")

        n_val = int(v_y.numel()) if v_y is not None else 0
        print(f"[{args.cohort}] step {step:02d}: train={int(tr_y.numel())} cells, "
              f"val={n_val}, classes={len(cats)} ({cats})")
        head, info = train_head(
            tr_emb, tr_y, v_emb, v_y,
            num_classes=len(cats), embed_dim=embed_dim,
            args=args, device=device,
        )
        torch.save({
            "head": head.state_dict(),
            "categories": cats,
            "embed_dim": embed_dim,
            "step": step,
            "cohort": args.cohort,
            "best_val_f1_macro": info["best_metric"],
            "best_epoch": info["best_epoch"],
            "metrics_log": info["log"],
        }, out_path)
        summary[f"step_{step:02d}"] = {
            "n_train": int(tr_y.numel()),
            "n_val": n_val,
            "categories": cats,
            "best_val_f1_macro": info["best_metric"],
            "best_epoch": info["best_epoch"],
            "ran_epochs": len(info["log"]),
            "metrics_log": info["log"],
        }
        print(f"[{args.cohort}] step {step:02d} done; "
              f"best val F1={info['best_metric']:.4f} @ epoch {info['best_epoch']}")

    (args.output_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{args.cohort}] heads saved to {args.output_dir}")


if __name__ == "__main__":
    main()
