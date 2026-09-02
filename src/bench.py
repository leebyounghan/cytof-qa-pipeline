"""
Benchmark I/O for CytoGate-Bench.

Provides a clean input interface for method implementations so each baseline
does not have to re-implement parquet loading, channel→marker resolution,
parent-population masking, and label extraction.

Typical usage:

    from src.bench import BenchmarkLoader

    loader = BenchmarkLoader(benchmark_dir="benchmark/", data_dir="data/")
    for ti in loader.iter_tasks(datasets=["Lyoplate_tcell"], split="test"):
        # ti.expression : pd.DataFrame (N, M) parent-pop cells, marker-named columns
        # ti.markers    : list[str] all available markers
        # ti.y          : np.ndarray (N,) GT labels for the task's annotation column
        # ti.categories : list[str] valid category names (incl. "Unassigned")
        # ti.X_2d       : np.ndarray (N, 2) shortcut for [x_marker, y_marker]
        pred = my_method(ti.expression[ti.markers].values, ti.categories)
        ti.save_prediction(pred, output_dir="results/my_method/", method="my_method")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Perturbations (Setting 2: In-Panel Hard)
# ---------------------------------------------------------------------------


class Perturbation:
    """Sample-level manipulation applied to a sample's cell dataframe.

    Subclasses transform the *post-rename* sample df (Step* columns already
    category-mapped; channel columns already renamed to marker names) and
    return a new df. Row-filtering perturbations return a subset; value-
    transforming perturbations return an equally-sized df with modified
    columns.
    """

    def apply(self, df: pd.DataFrame, context: dict | None = None) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> dict:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Deplete(Perturbation):
    """Remove cells whose GT label at `column` is in `values`.

    `fraction` ∈ [0, 1] is the share of matching cells to remove
    (1.0 = full depletion, 0.5 = remove a random half, 0.0 = no-op).
    Subsampling is reproducible via `seed`.
    """

    column: str
    values: list[str]
    fraction: float = 1.0
    seed: int = 42

    def apply(self, df: pd.DataFrame, context: dict | None = None) -> pd.DataFrame:
        if self.column not in df.columns or self.fraction <= 0.0:
            return df
        match = np.isin(df[self.column].values, self.values)
        if self.fraction >= 1.0:
            return df.loc[~match].reset_index(drop=True)
        match_idx = np.where(match)[0]
        if match_idx.size == 0:
            return df
        n_drop = int(round(match_idx.size * self.fraction))
        if n_drop <= 0:
            return df
        rng = np.random.default_rng(self.seed)
        drop_idx = rng.choice(match_idx, size=n_drop, replace=False)
        keep = np.ones(len(df), dtype=bool)
        keep[drop_idx] = False
        return df.loc[keep].reset_index(drop=True)

    def describe(self) -> dict:
        return {
            "type": "deplete",
            "column": self.column,
            "values": list(self.values),
            "fraction": float(self.fraction),
            "seed": int(self.seed),
        }


@dataclass
class RatioShift(Perturbation):
    """Downsample cells per-category at `column` to target ratios.

    `ratios` is a mapping {category: weight}. We compute the common scale
    `unit = min(count[cat] / weight[cat])` across target categories and
    subsample each to `unit * weight`. Cells whose label is not in `ratios`
    are left untouched. Never upsamples.
    """

    column: str
    ratios: dict
    seed: int = 42

    def apply(self, df: pd.DataFrame, context: dict | None = None) -> pd.DataFrame:
        if self.column not in df.columns or not self.ratios:
            return df
        col = df[self.column].values
        counts = {cat: int(np.sum(col == cat)) for cat in self.ratios}
        scales = [counts[cat] / w for cat, w in self.ratios.items() if w > 0 and counts[cat] > 0]
        if not scales:
            return df
        unit = min(scales)

        rng = np.random.default_rng(self.seed)
        keep = np.ones(len(df), dtype=bool)
        for cat, weight in self.ratios.items():
            target = int(round(unit * weight))
            idx = np.where(col == cat)[0]
            if target >= len(idx):
                continue
            drop_idx = rng.choice(idx, size=len(idx) - target, replace=False)
            keep[drop_idx] = False
        return df.loc[keep].reset_index(drop=True)

    def describe(self) -> dict:
        return {
            "type": "ratio_shift",
            "column": self.column,
            "ratios": dict(self.ratios),
            "seed": self.seed,
        }


@dataclass
class ChannelShift(Perturbation):
    """Add a constant offset to specific channel columns.

    `shifts` is a mapping {name: delta} where `name` may be either a channel
    name (direct column key in the channel-space sample df) or a marker display
    name. If a marker is given, the shift is broadcast to every channel that
    maps to that marker (handles collisions like `140Ce_Bead` / `142Ce_Bead`
    → marker `Bead`).

    Names that match nothing in the df are silently ignored.
    """

    shifts: dict

    def apply(self, df: pd.DataFrame, context: dict | None = None) -> pd.DataFrame:
        if not self.shifts:
            return df
        ch_map: dict = (context or {}).get("channel_marker_map", {})
        out = df.copy()
        for name, delta in self.shifts.items():
            d = float(delta)
            targets: list[str] = []
            if name in out.columns:
                targets.append(name)
            else:
                # marker → channel fallback (may match multiple channels)
                targets.extend(ch for ch, mk in ch_map.items() if mk == name and ch in out.columns)
            for ch in targets:
                out[ch] = out[ch].astype(np.float32) + d
        return out

    def describe(self) -> dict:
        return {"type": "channel_shift", "shifts": {k: float(v) for k, v in self.shifts.items()}}


@dataclass
class ChannelScale(Perturbation):
    """Multiplicative scale on raw (pre-arcsinh) signal.

    Models lot / antibody / instrument calibration drift in a physically
    realistic way: ``raw' = raw * factor``. Compared to ChannelShift (additive
    in arcsinh space), this preserves dim populations near zero — additive
    arcsinh shifts drive dim cells into unphysical negative raw counts,
    whereas a multiplicative factor leaves raw≈0 cells essentially unchanged
    and rescales bright populations proportionally.

    Implementation note: the sample df is already arcsinh-transformed; the
    equivalent in-place operation is

        arcsinh' = arcsinh(sinh(arcsinh) * factor)

    The cofactor cancels:
        raw      = sinh(x) * cofactor
        raw'     = raw * factor
        arcsinh' = arcsinh(raw' / cofactor) = arcsinh(sinh(x) * factor)

    `scales` is a mapping {name: factor}. `factor=1.0` is a no-op; <1 dims,
    >1 brightens. Marker names broadcast to every channel mapping to the
    marker (same convention as ChannelShift).

    Names that match nothing in the df are silently ignored.
    """

    scales: dict

    def apply(self, df: pd.DataFrame, context: dict | None = None) -> pd.DataFrame:
        if not self.scales:
            return df
        ch_map: dict = (context or {}).get("channel_marker_map", {})
        out = df.copy()
        for name, factor in self.scales.items():
            f = float(factor)
            if f == 1.0:
                continue
            targets: list[str] = []
            if name in out.columns:
                targets.append(name)
            else:
                targets.extend(ch for ch, mk in ch_map.items() if mk == name and ch in out.columns)
            for ch in targets:
                v = out[ch].astype(np.float64).values
                out[ch] = np.arcsinh(np.sinh(v) * f).astype(np.float32)
        return out

    def describe(self) -> dict:
        return {"type": "channel_scale", "scales": {k: float(v) for k, v in self.scales.items()}}


@dataclass
class Compose(Perturbation):
    """Chain multiple perturbations — apply in order."""

    steps: list

    def apply(self, df: pd.DataFrame, context: dict | None = None) -> pd.DataFrame:
        for p in self.steps:
            df = p.apply(df, context=context)
        return df

    def describe(self) -> dict:
        return {"type": "compose", "steps": [p.describe() for p in self.steps]}


# ---------------------------------------------------------------------------
# Inline --perturbation syntax
# ---------------------------------------------------------------------------
#
# Each --perturbation takes a space-separated spec:
#
#   deplete       <column> <value> [<value> ...]
#       e.g. 'deplete Step01_CD3|SSC-A CD3pos'
#            'deplete Step02_CD4|CD8 DPT DNT'
#
#   ratio_shift   <column> <cat>=<weight> [<cat>=<weight> ...] [seed=<int>]
#       e.g. 'ratio_shift Step02_CD4|CD8 CD4=10 CD8=1'
#            'ratio_shift Step02_CD4|CD8 CD4=20 CD8=1 seed=42'
#
#   channel_shift <marker>=<delta> [<marker>=<delta> ...]
#       e.g. 'channel_shift CD3=0.5'
#            'channel_shift CD3=0.5 CD4=-0.3'
#
#   channel_scale <marker>=<factor> [<marker>=<factor> ...]
#       e.g. 'channel_scale CD3=1.5'      # 1.5× brighter on raw
#            'channel_scale CD3=0.7 CD4=1.3'
#
# Multiple --perturbation flags are composed (applied left-to-right).


def parse_perturbation_spec(spec: str) -> Perturbation:
    import shlex
    tokens = shlex.split(spec.strip())
    if not tokens:
        raise ValueError("empty perturbation spec")
    ptype, rest = tokens[0], tokens[1:]

    if ptype == "deplete":
        if len(rest) < 2:
            raise ValueError(f"deplete expects: deplete <column> <value> [<value> ...] [frac=F] [seed=N] (got {spec!r})")
        column = rest[0]
        values: list[str] = []
        fraction = 1.0
        seed = 42
        for tok in rest[1:]:
            if tok.startswith("frac="):
                fraction = float(tok.split("=", 1)[1])
            elif tok.startswith("seed="):
                seed = int(tok.split("=", 1)[1])
            else:
                values.append(tok)
        if not values:
            raise ValueError(f"deplete needs at least one value (got {spec!r})")
        return Deplete(column=column, values=values, fraction=fraction, seed=seed)

    if ptype == "ratio_shift":
        if len(rest) < 2:
            raise ValueError(f"ratio_shift expects: ratio_shift <column> <cat>=<w> ... (got {spec!r})")
        column = rest[0]
        ratios: dict = {}
        seed = 42
        for tok in rest[1:]:
            if "=" not in tok:
                raise ValueError(f"ratio_shift: expected cat=weight or seed=N, got {tok!r}")
            k, v = tok.split("=", 1)
            if k == "seed":
                seed = int(v)
            else:
                ratios[k] = float(v)
        if not ratios:
            raise ValueError(f"ratio_shift needs at least one cat=weight pair (got {spec!r})")
        return RatioShift(column=column, ratios=ratios, seed=seed)

    if ptype == "channel_shift":
        if not rest:
            raise ValueError(f"channel_shift expects: channel_shift <marker>=<delta> ... (got {spec!r})")
        shifts: dict = {}
        for tok in rest:
            if "=" not in tok:
                raise ValueError(f"channel_shift: expected marker=delta, got {tok!r}")
            m, d = tok.split("=", 1)
            shifts[m] = float(d)
        return ChannelShift(shifts=shifts)

    if ptype == "channel_scale":
        if not rest:
            raise ValueError(f"channel_scale expects: channel_scale <marker>=<factor> ... (got {spec!r})")
        scales: dict = {}
        for tok in rest:
            if "=" not in tok:
                raise ValueError(f"channel_scale: expected marker=factor, got {tok!r}")
            m, f = tok.split("=", 1)
            scales[m] = float(f)
        return ChannelScale(scales=scales)

    raise ValueError(f"unknown perturbation type: {ptype!r}")


def _slug_one(p: Perturbation) -> str:
    import re
    d = p.describe()
    t = d.get("type", "perturb")
    parts = [t]
    if t == "deplete":
        parts.append(d["column"])
        parts.extend(d["values"])
        frac = d.get("fraction", 1.0)
        if frac != 1.0:
            parts.append(f"frac{int(round(frac * 100))}")
        if d.get("seed", 42) != 42:
            parts.append(f"seed{d['seed']}")
    elif t == "ratio_shift":
        parts.append(d["column"])
        parts.extend(f"{k}{v}" for k, v in d["ratios"].items())
        if d.get("seed", 42) != 42:
            parts.append(f"seed{d['seed']}")
    elif t == "channel_shift":
        parts.extend(f"{k}{v:+g}" for k, v in d["shifts"].items())
    elif t == "channel_scale":
        parts.extend(f"{k}x{v:g}" for k, v in d["scales"].items())
    else:
        parts.append(str(d))
    raw = "-".join(parts)
    return re.sub(r"[^A-Za-z0-9._+-]", "_", raw)


def parse_perturbations(
    specs: list[str] | None, name_override: str | None = None
) -> tuple[str | None, Perturbation | None]:
    """Parse one or more inline --perturbation specs.

    Returns (slug_name, Perturbation). Multiple specs compose left-to-right.
    Returns (None, None) if `specs` is empty or None.
    """
    if not specs:
        return None, None
    perts = [parse_perturbation_spec(s) for s in specs]
    p = perts[0] if len(perts) == 1 else Compose(steps=perts)
    name = name_override or "+".join(_slug_one(pp) for pp in perts)
    return name, p


# ---------------------------------------------------------------------------
# TaskInput
# ---------------------------------------------------------------------------


@dataclass
class TaskInput:
    """Clean, method-agnostic input for a single gating-step task.

    Columns are in **channel-space** — i.e., the exact channel keys from the
    parquet file and from `meta.json["channels"]`. Use `channel_marker_map`
    (or `display_name()`) to get a marker display name when you need it for
    rendering. We do not rename columns because, in CyTOF cohorts, multiple
    channels can collapse to the same marker (e.g., several bead channels →
    marker `"Bead"`), and renaming would destroy that identity.

    Attributes:
        task: Raw task.json dict.
        expression: Parent-population cells × channels. Columns are channel
            names (same as parquet). Includes scatter, tech, and protein.
        markers: List of channel names available in `expression` (ordered).
            Kept as the attribute name for back-compat with callers; these
            are channel keys, not display markers.
        y: GT labels for `task["annotation_column"]` over parent-pop rows.
            NaN/empty entries are replaced with "Unassigned".
        channel_marker_map: `{channel: marker_display_name}` from meta.json.
        marker_kinds: `{channel: kind}` (e.g. "protein", "tech", "scatter").
        categories: `task["categories"]`.
        x_marker / y_marker: channel-space column keys for the 2D gating plane.
        x_marker_display / y_marker_display: human-readable marker names.
    """

    task: dict
    expression: pd.DataFrame
    markers: list[str]
    y: np.ndarray
    _benchmark_dir: Path
    _step_dir: Path
    channel_marker_map: dict = None  # type: ignore[assignment]
    marker_kinds: dict = None  # type: ignore[assignment]
    _x_channel: str | None = None
    _y_channel: str | None = None
    perturbation_name: str | None = None
    original_indices: np.ndarray | None = None
    cofactor: float | None = None

    @property
    def task_id(self) -> str:
        return self.task["task_id"]

    @property
    def dataset(self) -> str:
        return self.task["dataset"]

    @property
    def sample(self) -> str:
        return self.task["sample"]

    @property
    def step(self) -> int:
        return self.task["step"]

    @property
    def parent(self) -> str:
        return self.task["parent"]

    @property
    def step_dir_name(self) -> str:
        """Name of the step directory in the benchmark (e.g. 'step_01')."""
        return self._step_dir.name

    @property
    def x_marker(self) -> str:
        """Channel-space column key for the x axis (resolved from task.json)."""
        return self._x_channel or self.task["x_marker"]

    @property
    def y_marker(self) -> str:
        """Channel-space column key for the y axis (resolved from task.json)."""
        return self._y_channel or self.task["y_marker"]

    @property
    def x_marker_display(self) -> str:
        """Human-readable marker name for the x axis (for plots / sentences)."""
        return self.display_name(self.x_marker)

    @property
    def y_marker_display(self) -> str:
        return self.display_name(self.y_marker)

    def display_name(self, channel: str) -> str:
        """Marker display name for a channel key; falls back to the channel name."""
        if self.channel_marker_map and channel in self.channel_marker_map:
            return self.channel_marker_map[channel]
        return channel

    @property
    def categories(self) -> list[str]:
        return list(self.task["categories"])

    @property
    def X_2d(self) -> np.ndarray:
        """Shortcut: (N, 2) array for the task's 2D gating plane."""
        return self.expression[[self.x_marker, self.y_marker]].values.astype(np.float32)

    def save_prediction(
        self,
        predicted_labels,
        output_dir: str | Path,
        method: str,
        info: dict | None = None,
    ) -> Path:
        """Write prediction.json mirroring the benchmark directory layout.

        Output path: {output_dir}/{dataset}/{sample}/step_{NN}/prediction.json
        """
        if isinstance(predicted_labels, np.ndarray):
            predicted_labels = predicted_labels.tolist()

        step_name = self._step_dir.name
        out_dir = Path(output_dir) / self.dataset / self.sample / step_name
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "task_id": self.task_id,
            "method": method,
            "predicted_labels": predicted_labels,
        }
        if self.perturbation_name is not None:
            payload["perturbation"] = self.perturbation_name
        if info is not None:
            payload["info"] = info

        out_path = out_dir / "prediction.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return out_path


# ---------------------------------------------------------------------------
# BenchmarkLoader
# ---------------------------------------------------------------------------


class BenchmarkLoader:
    """Iterate over CytoGate-Bench tasks as clean TaskInput objects.

    Caches per-dataset config and per-sample parquet loads within a single
    loader instance, so iterating all steps of a sample is cheap.

    For new methods, the recommended entry point is the CLI helpers so that
    users do not have to re-implement perturbation / split / dataset filtering
    plumbing:

        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, required=True)
        BenchmarkLoader.add_cli_args(parser)
        args = parser.parse_args()

        loader = BenchmarkLoader.from_cli(args)
        for ti in loader.iter_tasks_from_cli(args):
            pred = my_method(ti.X_2d, ti.categories)
            ti.save_prediction(pred, args.output_dir, method="mine")
    """

    # -- CLI helpers -------------------------------------------------------

    @staticmethod
    def add_cli_args(parser) -> None:
        """Attach loader-related flags to an argparse parser.

        Adds: --benchmark, --data-dir, --datasets, --samples, --split,
        --split-set, --perturbation (repeatable), --perturbation-name,
        --hard-depletion + --hard-depletion-yaml + --hard-depletion-fraction,
        --hard-shift + --hard-shift-yaml + --hard-shift-scale.
        """
        parser.add_argument("--benchmark", type=Path, default=Path("benchmark/"),
                            help="Benchmark directory (default: benchmark/)")
        parser.add_argument("--data-dir", type=Path, default=Path("data/"),
                            help="Raw data directory (default: data/)")
        parser.add_argument("--datasets", nargs="*", default=None,
                            help="Restrict to these dataset names (default: all)")
        parser.add_argument("--samples", nargs="*", default=None,
                            help="Restrict to these sample names")
        parser.add_argument("--split", type=str, default=None,
                            help="Path to splits.json (optional)")
        parser.add_argument("--split-set", type=str, default="test",
                            choices=["dev", "test", "train", "val"],
                            help="Which split to use (default: test)")
        parser.add_argument(
            "--perturbation", action="append", default=[],
            help="Inline perturbation spec (repeatable, composed left-to-right). "
                 "Syntax: 'deplete <col> <val> ...' | "
                 "'ratio_shift <col> <cat>=<w> ... [seed=N]' | "
                 "'channel_shift <marker>=<delta> ...' | "
                 "'channel_scale <marker>=<factor> ...'.",
        )
        parser.add_argument("--perturbation-name", default=None,
                            help="Override auto-derived perturbation slug")
        # ── Hard depletion ──
        parser.add_argument(
            "--hard-depletion", default=None,
            help="Hard depletion id from hard_depletion.yaml (e.g. hiv_acute2020_step29). "
                 "Auto-resolves --datasets, --perturbation, --perturbation-name, and a "
                 "single-step filter. Mutually exclusive with --perturbation / --datasets / --hard-shift.",
        )
        parser.add_argument(
            "--hard-depletion-yaml", type=Path, default=Path("benchmark/hard_depletion.yaml"),
            help="Path to hard_depletion.yaml (default: benchmark/hard_depletion.yaml)",
        )
        parser.add_argument(
            "--hard-depletion-fraction", type=float, default=1.0,
            dest="hard_depletion_fraction",
            help="Fraction of matching cells to deplete (default 1.0 = full). "
                 "Slug suffixed with _frac{NN} when != 1.0.",
        )
        # ── Hard shift ──
        parser.add_argument(
            "--hard-shift", default=None,
            help="Hard shift id from hard_shift.yaml (e.g. lineage_dim_acute2020_step29). "
                 "Auto-resolves --datasets, --perturbation, --perturbation-name, and a "
                 "single-step filter. Mutually exclusive with --perturbation / --datasets / --hard-depletion.",
        )
        parser.add_argument(
            "--hard-shift-yaml", type=Path, default=Path("benchmark/hard_shift.yaml"),
            help="Path to hard_shift.yaml (default: benchmark/hard_shift.yaml)",
        )
        parser.add_argument(
            "--hard-shift-scale", type=float, default=1.0,
            dest="hard_shift_scale",
            help="Scale multiplier for the calibrated shift magnitude (default 1.0). "
                 "Slug suffixed with _scale{NN} when != 1.0.",
        )

    @classmethod
    def from_cli(cls, args, verbose: bool = True) -> "BenchmarkLoader":
        """Construct a BenchmarkLoader from a parsed argparse Namespace.

        Expects the flags added by `add_cli_args`. Prints the perturbation
        spec when present (disable with verbose=False). When --hard-depletion
        or --hard-shift is set, resolves it to its underlying perturbation +
        slug and overrides --datasets / --perturbation / --perturbation-name
        (raises if those flags were also passed explicitly).
        """
        hard_dep = getattr(args, "hard_depletion", None)
        hard_shift = getattr(args, "hard_shift", None)
        if hard_dep and hard_shift:
            raise SystemExit(
                "[ERROR] --hard-depletion and --hard-shift are mutually exclusive."
            )

        if hard_dep:
            from src.hard_depletions import get_hard_depletion

            user_pert = getattr(args, "perturbation", None) or []
            user_pname = getattr(args, "perturbation_name", None)
            user_datasets = getattr(args, "datasets", None)
            if user_pert or user_pname or user_datasets:
                raise SystemExit(
                    "[ERROR] --hard-depletion is mutually exclusive with "
                    "--perturbation / --perturbation-name / --datasets. "
                    f"hard_depletion={hard_dep!r} already determines those."
                )
            hd = get_hard_depletion(hard_dep, yaml_path=args.hard_depletion_yaml)
            fraction = float(getattr(args, "hard_depletion_fraction", 1.0))
            p, name = hd.to_perturbation(fraction=fraction)
            args.datasets = [hd.dataset]
            args.hard_step = hd.step  # consumed by iter_tasks_from_cli
            if verbose:
                frac_msg = "" if fraction == 1.0 else f", fraction={fraction}"
                print(f"Hard depletion: {hd.hard_id} (scenario={hd.scenario}, "
                      f"dataset={hd.dataset}, step={hd.step}{frac_msg}) "
                      f"→ Deplete({hd.step_column!r}, {list(hd.deplete)})")
            return cls(
                benchmark_dir=args.benchmark,
                data_dir=args.data_dir,
                perturbation=p,
                perturbation_name=name,
            )

        if hard_shift:
            import json
            from src.hard_shifts import get_hard_shift

            user_pert = getattr(args, "perturbation", None) or []
            user_pname = getattr(args, "perturbation_name", None)
            user_datasets = getattr(args, "datasets", None)
            if user_pert or user_pname or user_datasets:
                raise SystemExit(
                    "[ERROR] --hard-shift is mutually exclusive with "
                    "--perturbation / --perturbation-name / --datasets. "
                    f"hard_shift={hard_shift!r} already determines those."
                )
            hs = get_hard_shift(hard_shift, yaml_path=args.hard_shift_yaml)
            scale = float(getattr(args, "hard_shift_scale", 1.0))
            meta_path = Path(args.data_dir) / hs.dataset / "parquet" / "meta.json"
            meta_channels = json.loads(meta_path.read_text())["channels"] \
                if meta_path.exists() else None
            p, name = hs.to_perturbation(meta_channels=meta_channels, scale=scale)
            args.datasets = [hs.dataset]
            args.hard_step = hs.step
            if verbose:
                scale_msg = "" if scale == 1.0 else f", scale={scale}"
                print(f"Hard shift: {hs.hard_id} (scenario={hs.scenario}, "
                      f"dataset={hs.dataset}, step={hs.step}{scale_msg}) "
                      f"→ ChannelScale({len(p.scales)} markers)")
            return cls(
                benchmark_dir=args.benchmark,
                data_dir=args.data_dir,
                perturbation=p,
                perturbation_name=name,
            )

        name, p = parse_perturbations(
            getattr(args, "perturbation", None) or [],
            name_override=getattr(args, "perturbation_name", None),
        )
        if verbose and p is not None:
            print(f"Perturbation: {name} ({p.describe()})")
        return cls(
            benchmark_dir=args.benchmark,
            data_dir=args.data_dir,
            perturbation=p,
            perturbation_name=name,
        )

    def iter_tasks_from_cli(self, args, **overrides) -> Iterator["TaskInput"]:
        """Iterate tasks using filters from a parsed argparse Namespace.

        Reads --datasets / --samples / --split / --split-set from `args`.
        Any keyword overrides (e.g. `min_cells=50`) are forwarded to
        `iter_tasks`. When --hard-task was given, also restricts to the
        target step number (and forces --split-set=test for that hard task).
        """
        steps = overrides.pop("steps", None)
        # If --hard-task was used, force step filter + test split
        hard_step = getattr(args, "hard_step", None)
        if hard_step is not None:
            steps = [int(hard_step)]
            # Default to test split unless user explicitly overrode --split-set
            if getattr(args, "split", None) is None:
                args.split = str(self.benchmark_dir / "splits.json")

        return self.iter_tasks(
            datasets=getattr(args, "datasets", None),
            samples=getattr(args, "samples", None),
            split=getattr(args, "split", None),
            split_set=getattr(args, "split_set", "test"),
            steps=steps,
            **overrides,
        )

    # -- init --------------------------------------------------------------

    def __init__(
        self,
        benchmark_dir: str | Path,
        data_dir: str | Path,
        perturbation: Perturbation | None = None,
        perturbation_name: str | None = None,
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.data_dir = Path(data_dir)
        self.perturbation = perturbation
        self.perturbation_name = perturbation_name
        self._config_cache: dict[str, dict] = {}
        self._sample_cache: dict[tuple[str, str], tuple[pd.DataFrame, list[str]]] = {}

    # -- config & sample loading -------------------------------------------

    def _load_config(self, dataset: str) -> dict:
        if dataset in self._config_cache:
            return self._config_cache[dataset]
        with open(self.data_dir / dataset / "parquet" / "meta.json") as f:
            meta = json.load(f)
        # channel -> marker display name
        channel_marker_map = {ch: info[0] for ch, info in meta["channels"].items()}
        # channel -> kind (protein / tech / scatter / bead / livedead / ...)
        marker_kinds = {
            ch: (info[1] if len(info) > 1 else "unknown")
            for ch, info in meta["channels"].items()
        }
        cfg = {
            "channel_marker_map": channel_marker_map,
            "marker_kinds": marker_kinds,
            "category_map": meta.get("category_map", {}) or {},
            "meta_channels": meta["channels"],
            "cofactor": meta.get("cofactor"),
        }
        self._config_cache[dataset] = cfg
        return cfg

    def _load_sample(self, dataset: str, sample: str) -> tuple[pd.DataFrame, list[str]]:
        """Load a sample's parquet in **channel-space** (no column rename).

        Returns (df, channels) where `channels` is the ordered list of channel
        keys present in df (non-Step columns intersected with meta.channels).
        """
        key = (dataset, sample)
        if key in self._sample_cache:
            return self._sample_cache[key]

        cfg = self._load_config(dataset)
        parquet_path = self.data_dir / dataset / "parquet" / f"{sample}.parquet"
        df = pd.read_parquet(parquet_path)

        # Normalize Step* label columns: "" → NaN, apply category_map.
        label_cols = [c for c in df.columns if c.startswith("Step")]
        for col in label_cols:
            s = df[col].astype(object)
            s.replace("", np.nan, inplace=True)
            if cfg["category_map"]:
                s.replace(cfg["category_map"], inplace=True)
            df[col] = s

        # Channel-ordered feature list (meta order, restricted to df columns).
        channels = [ch for ch in cfg["meta_channels"] if ch in df.columns]

        # Apply perturbation in channel-space. Pass channel_marker_map via
        # context so ChannelShift can resolve marker tokens → channels.
        if self.perturbation is not None:
            df = self.perturbation.apply(
                df, context={"channel_marker_map": cfg["channel_marker_map"]}
            ).reset_index(drop=True)

        self._sample_cache[key] = (df, channels)
        return df, channels

    # -- parent mask -------------------------------------------------------

    @staticmethod
    def _parent_mask(df: pd.DataFrame, parent: str) -> np.ndarray:
        n = len(df)
        if parent == "ALL":
            return np.ones(n, dtype=bool)
        mask = np.ones(n, dtype=bool)
        for part in parent.split("&"):
            part = part.strip()
            if "==" in part:
                col, val = [x.strip() for x in part.split("==")]
                mask &= (df[col].values == val) if col in df.columns else np.zeros(n, dtype=bool)
            elif "!=" in part:
                col, val = [x.strip() for x in part.split("!=")]
                mask &= (df[col].values != val) if col in df.columns else np.zeros(n, dtype=bool)
            elif " in " in part:
                col, vals_str = [x.strip() for x in part.split(" in ")]
                vals = [v.strip() for v in vals_str.split(",")]
                mask &= np.isin(df[col].values, vals) if col in df.columns else np.zeros(n, dtype=bool)
        return mask

    # -- public API --------------------------------------------------------

    def list_datasets(self) -> list[str]:
        return sorted(
            d.name for d in self.benchmark_dir.iterdir()
            if d.is_dir() and (d / "panel_meta.yaml").exists()
        )

    def _split_samples(self, split_path: Path, split_set: str) -> dict[str, list[str]]:
        with open(split_path) as f:
            splits = json.load(f)
        key_candidates = [split_set, f"{split_set}_samples"]
        if split_set == "dev":
            key_candidates.append("val")
        out: dict[str, list[str]] = {}
        for ds, info in splits.get("in_panel", {}).items():
            for k in key_candidates:
                if k in info:
                    out[ds] = list(info[k])
                    break
            else:
                out[ds] = []
        return out

    def _resolve_channel(self, dataset: str, name: str, df_cols: list[str]) -> str | None:
        """Resolve a name from task.json into a channel key present in `df_cols`.

        - If `name` is already a channel column, return it.
        - Otherwise, treat `name` as a marker display name and return the first
          channel in meta order that maps to it. Returns None if unresolved.
        """
        if name in df_cols:
            return name
        cfg = self._load_config(dataset)
        for ch in cfg["meta_channels"]:
            if cfg["channel_marker_map"].get(ch) == name and ch in df_cols:
                return ch
        return None

    def load_task(self, task_path: str | Path) -> TaskInput | None:
        """Load a single task by path to its task.json (or step dir)."""
        task_path = Path(task_path)
        if task_path.is_dir():
            task_path = task_path / "task.json"
        with open(task_path) as f:
            task = json.load(f)

        df, channels = self._load_sample(task["dataset"], task["sample"])
        cfg = self._load_config(task["dataset"])

        x_ch = self._resolve_channel(task["dataset"], task["x_marker"], list(df.columns))
        y_ch = self._resolve_channel(task["dataset"], task["y_marker"], list(df.columns))
        if x_ch is None or y_ch is None:
            return None

        mask = self._parent_mask(df, task["parent"])
        original_indices = np.where(mask)[0].astype(np.int32)
        n_parent = int(original_indices.size)

        # Avoid ``df.loc[mask]``: that slices every column — including the
        # string-dtype ``Step*`` label columns — which drags in pandas
        # ``_take_nd_object`` + ``_isna_string_dtype`` passes for data we
        # don't need. Slice channel and annotation columns directly.
        if channels:
            # ``df[channels]`` is a column-subset view; ``.to_numpy()`` on a
            # numeric-only block is a single fast copy, then fancy-index.
            expr_values = df[channels].to_numpy(copy=False)[original_indices]
        else:
            expr_values = np.empty((n_parent, 0), dtype=np.float32)
        expression = pd.DataFrame(expr_values, columns=channels, copy=False)

        col = task["annotation_column"]
        if col in df.columns:
            y_raw = df[col].to_numpy()[original_indices]
            # Vectorised NaN → "Unassigned" replacement (original code did
            # this in a Python per-cell loop).
            y = np.where(pd.isna(y_raw), "Unassigned", y_raw).astype(object)
        else:
            y = np.full(n_parent, "Unassigned", dtype=object)

        return TaskInput(
            task=task,
            expression=expression,
            markers=channels,
            y=y,
            _benchmark_dir=self.benchmark_dir,
            _step_dir=task_path.parent,
            channel_marker_map=dict(cfg["channel_marker_map"]),
            marker_kinds=dict(cfg["marker_kinds"]),
            _x_channel=x_ch,
            _y_channel=y_ch,
            perturbation_name=self.perturbation_name,
            original_indices=original_indices,
            cofactor=cfg.get("cofactor"),
        )

    def iter_tasks(
        self,
        datasets: list[str] | None = None,
        samples: list[str] | None = None,
        split: str | Path | None = None,
        split_set: str = "test",
        min_cells: int = 1,
        steps: list[int] | None = None,
    ) -> Iterator[TaskInput]:
        """Yield TaskInput for every valid step task matching the filters.

        Args:
            datasets: Restrict to these dataset names (default: all).
            samples: Restrict to these sample names (applied after split).
            split: Path to splits.json; if given, restricts samples per-dataset.
            split_set: "dev" or "test" (used with `split`).
            min_cells: Skip tasks whose parent population has fewer cells.
            steps: Restrict to these step numbers (e.g. [29] for step_29 only).
        """
        ds_names = datasets if datasets else self.list_datasets()

        split_map: dict[str, list[str]] = {}
        if split:
            split_map = self._split_samples(Path(split), split_set)

        step_filter: set[int] | None = set(int(s) for s in steps) if steps else None

        for ds in ds_names:
            ds_dir = self.benchmark_dir / ds
            if not ds_dir.is_dir():
                continue
            sample_dirs = sorted([d for d in ds_dir.iterdir() if d.is_dir()])
            allowed = set(split_map.get(ds, [])) if split else None
            if allowed is not None:
                sample_dirs = [d for d in sample_dirs if d.name in allowed]
            if samples:
                sample_dirs = [d for d in sample_dirs if d.name in set(samples)]

            for sdir in sample_dirs:
                parquet_path = self.data_dir / ds / "parquet" / f"{sdir.name}.parquet"
                if not parquet_path.exists():
                    continue

                step_dirs = sorted(
                    d for d in sdir.iterdir() if d.is_dir() and d.name.startswith("step_")
                )
                for stepd in step_dirs:
                    if step_filter is not None:
                        try:
                            stepnum = int(stepd.name.split("_", 1)[1])
                        except (IndexError, ValueError):
                            continue
                        if stepnum not in step_filter:
                            continue
                    tp = stepd / "task.json"
                    if not tp.exists():
                        continue
                    ti = self.load_task(tp)
                    if ti is None:
                        continue
                    if len(ti.expression) < min_cells:
                        continue
                    yield ti
