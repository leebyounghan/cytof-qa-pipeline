"""
Hard depletions loader for In-Panel Hard (Setting 2) — Deplete branch.

Each hard depletion is a single (dataset, step, depletion) triple defined in
benchmark/hard_depletion.yaml. At runtime, a HardDepletion materializes into
a standard Deplete perturbation whose slug equals the hard_id, so the entire
existing perturbation infrastructure (file naming, slug round-trip, eval
cross-checks) carries through unchanged.

Sister module: ``src.hard_shifts`` for the channel-shift branch.

Typical usage:

    from src.hard_depletions import load_hard_depletions, get_hard_depletion

    scenarios, tasks = load_hard_depletions()
    hd = get_hard_depletion("hiv_acute2020_step29")
    deplete, slug = hd.to_perturbation()         # → (Deplete, "hiv_acute2020_step29")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.bench import Deplete


DEFAULT_HARD_DEPLETION_YAML = Path("benchmark/hard_depletion.yaml")


@dataclass(frozen=True)
class HardDepletion:
    """A single (dataset, step, depletion) benchmark task."""

    hard_id: str
    scenario: str
    dataset: str
    step: int
    step_column: str
    deplete: tuple[str, ...]  # frozen: tuple instead of list

    def to_perturbation(
        self, fraction: float = 1.0, seed: int = 42,
    ) -> tuple[Deplete, str]:
        """Return (Deplete object, slug).

        Slug = hard_id, with `_frac{NN}` appended only when `fraction != 1.0`
        so existing 100%-depletion artefacts remain bit-identical.
        """
        d = Deplete(
            column=self.step_column, values=list(self.deplete),
            fraction=fraction, seed=seed,
        )
        slug = self.hard_id
        if fraction != 1.0:
            slug += f"_frac{int(round(fraction * 100))}"
        return d, slug


def load_hard_depletions(
    yaml_path: str | Path = DEFAULT_HARD_DEPLETION_YAML,
) -> tuple[dict[str, str], list[HardDepletion]]:
    """Load benchmark/hard_depletion.yaml. Returns (scenario_descriptions, tasks)."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        spec = yaml.safe_load(f)

    scenarios = {k: v["description"] for k, v in spec.get("scenarios", {}).items()}

    tasks: list[HardDepletion] = []
    seen_ids: set[str] = set()
    for t in spec["tasks"]:
        hard_id = t["hard_id"]
        if hard_id in seen_ids:
            raise ValueError(f"Duplicate hard_id in {yaml_path}: {hard_id!r}")
        seen_ids.add(hard_id)
        scenario = t["scenario"]
        if scenario not in scenarios:
            raise ValueError(
                f"Hard depletion {hard_id!r} references unknown scenario {scenario!r}. "
                f"Known scenarios: {sorted(scenarios)}"
            )
        tasks.append(
            HardDepletion(
                hard_id=hard_id,
                scenario=scenario,
                dataset=t["dataset"],
                step=int(t["step"]),
                step_column=t["step_column"],
                deplete=tuple(t["deplete"]),
            )
        )

    return scenarios, tasks


def get_hard_depletion(
    hard_id: str,
    yaml_path: str | Path = DEFAULT_HARD_DEPLETION_YAML,
) -> HardDepletion:
    """Look up a single hard depletion by ID. Raises KeyError if not found."""
    _, tasks = load_hard_depletions(yaml_path)
    for hd in tasks:
        if hd.hard_id == hard_id:
            return hd
    available = ", ".join(sorted(t.hard_id for t in tasks))
    raise KeyError(f"hard_id {hard_id!r} not found. Available: {available}")


def add_hard_depletion_args(parser) -> None:
    """Attach --hard-depletion / --hard-depletion-yaml flags to a parser.

    Use this for stages that do NOT call BenchmarkLoader.add_cli_args() but
    accept their own --perturbation / --perturbation-name flags. Pair with
    resolve_hard_depletion_args(args) right after parse_args() to translate
    the hard_id into args.perturbation / args.perturbation_name.
    """
    parser.add_argument(
        "--hard-depletion", default=None,
        help="Hard depletion id from hard_depletion.yaml (e.g. hiv_acute2020_step29). "
             "Overrides --perturbation / --perturbation-name.",
    )
    parser.add_argument(
        "--hard-depletion-yaml", type=Path,
        default=DEFAULT_HARD_DEPLETION_YAML,
        help=f"Path to hard_depletion.yaml (default: {DEFAULT_HARD_DEPLETION_YAML})",
    )
    parser.add_argument(
        "--hard-depletion-fraction", type=float, default=1.0,
        dest="hard_depletion_fraction",
        help="Fraction of matching cells to deplete (default 1.0 = full). "
             "When != 1.0, slug is suffixed with _frac{NN}.",
    )


def resolve_hard_depletion_args(args) -> HardDepletion | None:
    """If --hard-depletion is set on `args`, mutate args to inject the
    corresponding --perturbation / --perturbation-name spec, and return the
    HardDepletion.

    Mutually exclusive with --perturbation / --perturbation-name / --hard-shift.
    Returns None when --hard-depletion is not set.
    """
    import shlex

    hard_id = getattr(args, "hard_depletion", None)
    if not hard_id:
        return None

    user_pert = getattr(args, "perturbation", None) or []
    user_pname = getattr(args, "perturbation_name", None)
    if user_pert or user_pname:
        raise SystemExit(
            "[ERROR] --hard-depletion is mutually exclusive with "
            "--perturbation / --perturbation-name."
        )
    if getattr(args, "hard_shift", None):
        raise SystemExit(
            "[ERROR] --hard-depletion is mutually exclusive with --hard-shift."
        )

    yaml_path = getattr(args, "hard_depletion_yaml", DEFAULT_HARD_DEPLETION_YAML)
    hd = get_hard_depletion(hard_id, yaml_path=yaml_path)

    fraction = float(getattr(args, "hard_depletion_fraction", 1.0))
    spec = "deplete " + shlex.quote(hd.step_column)
    spec += " " + " ".join(shlex.quote(v) for v in hd.deplete)
    if fraction != 1.0:
        spec += f" frac={fraction}"
    args.perturbation = [spec]
    args.perturbation_name = hd.hard_id + (
        f"_frac{int(round(fraction * 100))}" if fraction != 1.0 else ""
    )
    return hd


def hard_depletions_for(
    dataset: str | None = None,
    scenario: str | None = None,
    yaml_path: str | Path = DEFAULT_HARD_DEPLETION_YAML,
) -> list[HardDepletion]:
    """Filter hard depletions by dataset and/or scenario."""
    _, tasks = load_hard_depletions(yaml_path)
    if dataset is not None:
        tasks = [t for t in tasks if t.dataset == dataset]
    if scenario is not None:
        tasks = [t for t in tasks if t.scenario == scenario]
    return tasks


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _selftest() -> None:
    """Run as `uv run python -m src.hard_depletions` to validate the YAML."""
    scenarios, tasks = load_hard_depletions()

    # Materialize each task into Deplete + slug
    for hd in tasks:
        d, slug = hd.to_perturbation()
        assert isinstance(d, Deplete), f"{hd.hard_id}: not a Deplete"
        assert slug == hd.hard_id, f"{hd.hard_id}: slug != hard_id"
        assert d.column == hd.step_column
        assert list(d.values) == list(hd.deplete)
        # describe() must round-trip
        desc = d.describe()
        assert desc["type"] == "deplete"
        assert desc["column"] == hd.step_column

    # Sanity: every referenced dataset directory exists
    data_root = Path("data")
    bench_root = Path("benchmark")
    missing_data = {hd.dataset for hd in tasks if not (data_root / hd.dataset).exists()}
    missing_bench = {hd.dataset for hd in tasks if not (bench_root / hd.dataset).exists()}
    if missing_data:
        print(f"  WARN: data dirs missing for: {sorted(missing_data)}")
    if missing_bench:
        print(f"  WARN: benchmark dirs missing for: {sorted(missing_bench)}")

    # Per-scenario / per-dataset counts
    from collections import Counter

    by_scenario = Counter(t.scenario for t in tasks)
    by_dataset = Counter(t.dataset for t in tasks)

    print(f"OK — {len(tasks)} hard depletions, {len(scenarios)} scenarios")
    print("By scenario:")
    for k, v in by_scenario.most_common():
        print(f"  {k:28s} {v}")
    print("By dataset:")
    for k, v in sorted(by_dataset.items()):
        print(f"  {k:25s} {v}")


if __name__ == "__main__":
    _selftest()
