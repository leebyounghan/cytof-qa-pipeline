"""cyMAE inference: load per-cohort encoder + per-step head, predict on test tasks.

Mirrors the canonical baseline CLI shape (see src/baselines/flowsom_baseline.py):
uses BenchmarkLoader's CLI helpers and writes prediction.json via
TaskInput.save_prediction. No subsampling — every cell of every parent population
gets a label.

Example:
    uv run python -m src.baselines.cyMAE.inference \\
        --benchmark benchmark/ --data-dir data/ \\
        --encoder-root artifacts/cymae/ \\
        --output-dir results/cymae/ \\
        --split benchmark/splits.json --split-set test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.bench import BenchmarkLoader, TaskInput

from . import data as data_mod
from . import models as models_mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cyMAE inference")
    p.add_argument("--encoder-root", type=Path, required=True,
                   help="Root dir written by pretrain.py / train_heads.py "
                        "(contains <cohort>/encoder.pth + heads/step_NN.pth)")
    p.add_argument("--output-dir", type=Path, default=Path("results/cymae/"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    BenchmarkLoader.add_cli_args(p)
    return p.parse_args()


class CohortBundle:
    """Encoder + channel artifacts for one cohort, plus a lazy cache of step heads."""

    def __init__(self, encoder_root: Path, cohort: str, device: torch.device):
        self.cohort = cohort
        self.device = device
        cohort_dir = encoder_root / cohort
        self.cohort_dir = cohort_dir
        if not cohort_dir.exists():
            raise FileNotFoundError(
                f"No cyMAE artifacts for cohort '{cohort}' under {encoder_root}/"
            )

        self.channel_order, self.mean, self.std = data_mod.load_channel_artifacts(cohort_dir)
        ckpt = torch.load(cohort_dir / "encoder.pth", map_location=device)
        self.num_features = ckpt["num_features"]
        if self.num_features != len(self.channel_order):
            raise RuntimeError(
                f"encoder.pth num_features={self.num_features} != "
                f"len(marker_order)={len(self.channel_order)} for {cohort}"
            )
        self.encoder = models_mod.build_encoder(num_features=self.num_features).to(device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self._head_cache: dict[int, dict] = {}

    def get_head(self, step: int) -> dict | None:
        if step in self._head_cache:
            return self._head_cache[step]
        path = self.cohort_dir / "heads" / f"step_{step:02d}.pth"
        if not path.exists():
            self._head_cache[step] = None
            return None
        ckpt = torch.load(path, map_location=self.device)
        head = models_mod.StepHead(
            embed_dim=ckpt["embed_dim"], num_classes=len(ckpt["categories"]),
        ).to(self.device)
        head.load_state_dict(ckpt["head"])
        head.eval()
        ckpt["head_module"] = head
        self._head_cache[step] = ckpt
        return ckpt


def predict_task(ti: TaskInput, bundle: CohortBundle) -> tuple[list, dict] | None:
    head_ckpt = bundle.get_head(ti.step)
    if head_ckpt is None:
        return None

    missing = [c for c in bundle.channel_order if c not in ti.expression.columns]
    if missing:
        raise RuntimeError(
            f"{ti.task_id}: channels missing from expression: "
            f"{missing[:5]}{' ...' if len(missing) > 5 else ''}"
        )
    x = ti.expression[bundle.channel_order].to_numpy(dtype=np.float32, na_value=0.0)
    if len(x) == 0:
        return [], {"n_cells": 0, "skipped": "empty parent population"}

    x = data_mod.apply_zscore(x, bundle.mean, bundle.std)
    with torch.no_grad():
        emb = models_mod.embed_cells(
            bundle.encoder, torch.from_numpy(x), return_device=bundle.device,
        )
        logits = head_ckpt["head_module"](emb)
        pred_idx = logits.argmax(dim=1).cpu().numpy()

    cats = head_ckpt["categories"]
    pred_labels = [cats[i] for i in pred_idx]
    info = {
        "cohort": bundle.cohort,
        "step": ti.step,
        "n_cells": int(len(pred_labels)),
        "n_markers": int(bundle.num_features),
        "categories": cats,
        "best_val_f1_macro": head_ckpt.get("best_val_f1_macro"),
    }
    return pred_labels, info


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    loader = BenchmarkLoader.from_cli(args)
    print(f"cyMAE inference → {args.output_dir}")

    bundles: dict[str, CohortBundle] = {}
    n_tasks = 0
    n_skipped = 0
    current_ds = current_sample = None
    for ti in loader.iter_tasks_from_cli(args):
        if ti.dataset != current_ds:
            current_ds, current_sample = ti.dataset, None
            print(f"\n{'='*60}\nDataset: {ti.dataset}\n{'='*60}")
            if ti.dataset not in bundles:
                bundles[ti.dataset] = CohortBundle(args.encoder_root, ti.dataset, device)
        if ti.sample != current_sample:
            current_sample = ti.sample
            print(f"  {ti.sample}")

        bundle = bundles[ti.dataset]
        result = predict_task(ti, bundle)
        if result is None:
            n_skipped += 1
            print(f"    step {ti.step:02d}: SKIP (no head trained for this step)")
            continue
        pred_labels, info = result
        ti.save_prediction(pred_labels, output_dir=args.output_dir, method="cymae", info=info)
        n_tasks += 1

    print(f"\nDone. Predicted: {n_tasks} tasks, skipped: {n_skipped}")


if __name__ == "__main__":
    main()
