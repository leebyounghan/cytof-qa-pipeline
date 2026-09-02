"""
Hard shift tasks loader for In-Panel Hard (Setting 2) channel-scale benchmark.

Each task is a single (dataset, step, channel_scale) triple defined in
benchmark/hard_shift.yaml. At runtime, a HardShift materializes into a
ChannelScale perturbation whose slug equals the hard_id, mirroring the
src.hard_depletions infrastructure.

Magnitudes (multiplicative factors on raw, pre-arcsinh signal) are empirically
calibrated per (dataset, step) — see hard_shift.yaml header for details.
factor=1.0 is no perturbation; <1 dims, >1 brightens.

Why multiplicative? Additive shifts in arcsinh space drive dim cells (raw≈0)
into unphysical negative raw counts. Multiplicative on raw preserves dim
populations near zero and rescales bright populations proportionally — which
is how lot/antibody/instrument calibration drift actually behaves.

Special sentinel `__all_protein_markers__` in `scales` is expanded at
materialize time to every channel with kind == "protein" in meta.json.

Typical usage:

    from src.hard_shifts import load_hard_shifts, get_hard_shift

    scenarios, tasks = load_hard_shifts()
    hs = get_hard_shift("lineage_dim_acute2020_step29")
    cs, slug = hs.to_perturbation(meta_channels)  # → (ChannelScale, "lineage_dim_acute2020_step29")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.bench import ChannelScale


DEFAULT_HARD_SHIFT_YAML = Path("benchmark/hard_shift.yaml")
ALL_PROTEIN_SENTINEL = "__all_protein_markers__"


@dataclass(frozen=True)
class HardShift:
    """A single (dataset, step, channel_scale) benchmark task."""

    hard_id: str
    scenario: str
    dataset: str
    step: int
    step_column: str
    scales: tuple[tuple[str, float], ...]  # frozen mapping marker→factor

    def scales_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.scales}

    def to_perturbation(
        self,
        meta_channels: dict[str, list] | None = None,
        scale: float = 1.0,
    ) -> tuple[ChannelScale, str]:
        """Return (ChannelScale, slug).

        `scale` is the magnitude knob — linearly interpolates each calibrated
        factor toward 1.0 (no-op):

            effective_factor = 1 + (calibrated_factor - 1) * scale

        So scale=0.0 → effective_factor = 1.0 for every marker (baseline);
        scale=1.0 → calibrated factor as-is; scale=0.5 → halfway. Slug =
        hard_id, with `_scale{NN}` appended only when scale != 1.0.

        If `scales` contains the sentinel `__all_protein_markers__`, expand
        it to every protein channel in `meta_channels` (the
        meta.json["channels"] dict). Without `meta_channels`, the sentinel is
        preserved verbatim and will silently no-op at apply time.
        """
        out: dict[str, float] = {}
        for k, v in self.scales:
            calibrated = float(v)
            effective = 1.0 + (calibrated - 1.0) * float(scale)
            if k == ALL_PROTEIN_SENTINEL:
                if meta_channels is None:
                    out[k] = effective  # preserve; will no-op downstream
                    continue
                for ch, info in meta_channels.items():
                    kind = info[1] if len(info) > 1 else "unknown"
                    if kind == "protein":
                        marker = info[0]
                        # Use marker name so ChannelScale.apply broadcasts via channel_marker_map
                        out[marker] = effective
            else:
                out[k] = effective
        slug = self.hard_id
        if scale != 1.0:
            slug += f"_scale{int(round(scale * 100))}"
        return ChannelScale(scales=out), slug


def load_hard_shifts(
    yaml_path: str | Path = DEFAULT_HARD_SHIFT_YAML,
) -> tuple[dict[str, dict], list[HardShift]]:
    """Load benchmark/hard_shift.yaml. Returns (scenario_info, tasks)."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        spec = yaml.safe_load(f)

    scenarios = spec.get("scenarios", {}) or {}

    tasks: list[HardShift] = []
    seen: set[str] = set()
    for t in spec["tasks"]:
        hid = t["hard_id"]
        if hid in seen:
            raise ValueError(f"Duplicate hard_id in {yaml_path}: {hid!r}")
        seen.add(hid)
        scenario = t["scenario"]
        if scenario not in scenarios:
            raise ValueError(
                f"Hard shift {hid!r} references unknown scenario {scenario!r}. "
                f"Known: {sorted(scenarios)}"
            )
        scales_d = t["scales"]
        if not scales_d:
            raise ValueError(f"Hard shift {hid!r} has empty `scales`")
        scales = tuple(sorted((str(k), float(v)) for k, v in scales_d.items()))
        tasks.append(
            HardShift(
                hard_id=hid,
                scenario=scenario,
                dataset=t["dataset"],
                step=int(t["step"]),
                step_column=t["step_column"],
                scales=scales,
            )
        )
    return scenarios, tasks


def get_hard_shift(
    hard_id: str,
    yaml_path: str | Path = DEFAULT_HARD_SHIFT_YAML,
) -> HardShift:
    """Look up one hard shift by ID. Raises KeyError if not found."""
    _, tasks = load_hard_shifts(yaml_path)
    for hs in tasks:
        if hs.hard_id == hard_id:
            return hs
    available = ", ".join(sorted(t.hard_id for t in tasks))
    raise KeyError(f"hard_id {hard_id!r} not found. Available: {available}")


def add_hard_shift_args(parser) -> None:
    """Attach --hard-shift / --hard-shift-yaml / --hard-shift-scale flags."""
    parser.add_argument(
        "--hard-shift", default=None,
        help="Hard shift id from hard_shift.yaml (e.g. lineage_dim_acute2020_step29). "
             "Overrides --perturbation / --perturbation-name.",
    )
    parser.add_argument(
        "--hard-shift-yaml", type=Path,
        default=DEFAULT_HARD_SHIFT_YAML,
        help=f"Path to hard_shift.yaml (default: {DEFAULT_HARD_SHIFT_YAML})",
    )
    parser.add_argument(
        "--hard-shift-scale", type=float, default=1.0,
        dest="hard_shift_scale",
        help="Magnitude knob: linearly interpolates each calibrated factor "
             "toward 1.0. 0.0 = no shift (baseline, every factor=1), "
             "0.5 = half-magnitude, 1.0 (default) = calibrated. Slug "
             "is suffixed with _scale{NN} only when != 1.0.",
    )


def resolve_hard_shift_args(args, meta_channels_lookup=None) -> HardShift | None:
    """If --hard-shift is set on `args`, stash the resolved ChannelScale
    object on args and return the HardShift.

    Sets:
        args.perturbation_name      — slug
        args._resolved_perturbation — ChannelScale object (callers should
                                      prefer this over re-parsing
                                      args.perturbation, since marker names
                                      may contain whitespace which the inline
                                      spec format cannot round-trip).
        args.perturbation           — left as []; callers branch on
                                      _resolved_perturbation.

    Mutually exclusive with --perturbation / --perturbation-name / --hard-depletion.
    `meta_channels_lookup(dataset) -> meta_channels_dict` is required if any
    task uses the `__all_protein_markers__` sentinel.
    """
    hid = getattr(args, "hard_shift", None)
    if not hid:
        return None
    if getattr(args, "perturbation", None) or getattr(args, "perturbation_name", None):
        raise SystemExit("[ERROR] --hard-shift is mutually exclusive with --perturbation / --perturbation-name.")
    if getattr(args, "hard_depletion", None):
        raise SystemExit("[ERROR] --hard-shift is mutually exclusive with --hard-depletion.")

    yaml_path = getattr(args, "hard_shift_yaml", DEFAULT_HARD_SHIFT_YAML)
    hs = get_hard_shift(hid, yaml_path=yaml_path)

    meta_channels = None
    if meta_channels_lookup is not None:
        meta_channels = meta_channels_lookup(hs.dataset)

    scale = float(getattr(args, "hard_shift_scale", 1.0))
    cs, slug = hs.to_perturbation(meta_channels=meta_channels, scale=scale)
    args.perturbation = []
    args.perturbation_name = slug
    args._resolved_perturbation = cs
    return hs


def hard_shifts_for(
    dataset: str | None = None,
    scenario: str | None = None,
    yaml_path: str | Path = DEFAULT_HARD_SHIFT_YAML,
) -> list[HardShift]:
    """Filter hard shifts by dataset and/or scenario."""
    _, tasks = load_hard_shifts(yaml_path)
    if dataset is not None:
        tasks = [t for t in tasks if t.dataset == dataset]
    if scenario is not None:
        tasks = [t for t in tasks if t.scenario == scenario]
    return tasks


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """Run as `uv run python -m src.hard_shifts` to validate the YAML."""
    import json

    scenarios, tasks = load_hard_shifts()

    # Materialize each task; sentinel-only tasks need meta_channels
    for hs in tasks:
        meta_path = Path("data") / hs.dataset / "parquet" / "meta.json"
        meta_channels = None
        if meta_path.exists():
            meta_channels = json.loads(meta_path.read_text())["channels"]
        cs, slug = hs.to_perturbation(meta_channels=meta_channels)
        assert isinstance(cs, ChannelScale), f"{hs.hard_id}: not a ChannelScale"
        assert slug == hs.hard_id, f"{hs.hard_id}: slug != hard_id"
        assert cs.scales, f"{hs.hard_id}: empty scales after expansion"
        # Sentinel must not survive expansion when meta_channels is provided
        if meta_channels is not None and ALL_PROTEIN_SENTINEL in cs.scales:
            raise AssertionError(
                f"{hs.hard_id}: sentinel {ALL_PROTEIN_SENTINEL!r} not expanded "
                f"despite meta_channels being provided"
            )
        # describe round-trip
        d = cs.describe()
        assert d["type"] == "channel_scale"

    # Sanity: every referenced dataset directory exists
    data_root = Path("data")
    bench_root = Path("benchmark")
    missing_data = {hs.dataset for hs in tasks if not (data_root / hs.dataset).exists()}
    missing_bench = {hs.dataset for hs in tasks if not (bench_root / hs.dataset).exists()}
    if missing_data:
        print(f"  WARN: data dirs missing for: {sorted(missing_data)}")
    if missing_bench:
        print(f"  WARN: benchmark dirs missing for: {sorted(missing_bench)}")

    from collections import Counter
    by_scenario = Counter(t.scenario for t in tasks)
    by_dataset = Counter(t.dataset for t in tasks)

    print(f"OK — {len(tasks)} hard shift tasks, {len(scenarios)} scenarios")
    print("By scenario:")
    for k, v in by_scenario.most_common():
        print(f"  {k:25s} {v}")
    print("By dataset:")
    for k, v in sorted(by_dataset.items()):
        print(f"  {k:25s} {v}")


if __name__ == "__main__":
    _selftest()
