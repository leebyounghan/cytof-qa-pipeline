#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt builder for ``gate_task.json`` — one prompt per (sample, step).

Each prompt asks the model to produce, for every named category at that
step, an axis-aligned rectangular gate ``{x_min, x_max, y_min, y_max}``
on the ``(x_marker, y_marker)`` plane. Cells outside every gate are
``Unassigned``; cells inside multiple gates are resolved at propagate
time (smallest-area gate wins by default).

Layout — prompts go into an ablation-named sibling directory of
``gate_task.json``::

    step_09/
      gate_task.json
      gate_full/
        cells.md

The ``[Axis distribution]`` section is currently a placeholder — the
representation is being decided separately. ``_render_axis_summary``
returns a TODO marker for now.

CLI::

    python -m src.vlm_gate.prompt benchmark/.../step_09/gate_task.json --save_md
    python -m src.vlm_gate.prompt --batch benchmark/Acute2020/994567_Normalized --save_md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from src.bench import parse_perturbations
from src.vlm_gate.utils.axis_distribution import render_axis_summary
from src.vlm_gate.utils.joint_distribution import render_joint_summary


# ── Note YAML override ──────────────────────────────────────────────────────

_NOTES_PATH = Path(__file__).resolve().parents[2] / 'benchmark' / 'notes.yaml'


def _load_notes() -> dict:
    """Load per-(cohort, step) note overrides from benchmark/notes.yaml.

    The YAML's per-step note replaces ``data['note']`` in
    :func:`format_prompt`. Returns an empty dict if the file is missing
    or empty.
    """
    if not _NOTES_PATH.exists():
        return {}
    with open(_NOTES_PATH) as f:
        return yaml.safe_load(f) or {}


_NOTES = _load_notes()


def lookup_note_override(cohort: str | None, step: int | str | None) -> str | None:
    """Return the YAML-defined note for ``(cohort, step)`` or None."""
    if not cohort or step is None:
        return None
    cohort_notes = _NOTES.get(cohort) or {}
    try:
        step_int = int(step)
    except (TypeError, ValueError):
        return None
    val = cohort_notes.get(step_int)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


# ── Ablation config ─────────────────────────────────────────────────────────

def ablation_slug() -> str:
    """Single ablation slug for now — the only variable here is the axis
    representation, which is being decided separately. Returns ``full``.
    """
    return 'full'


def ablation_dir(
    json_path: Path, slug: str, perturbation_name: str | None = None,
) -> Path:
    name = f'gate_{slug}'
    if perturbation_name:
        name = f'{name}__{perturbation_name}'
    return json_path.parent / name


# ── Helpers ─────────────────────────────────────────────────────────────────

def _breadcrumb(x_marker: str, y_marker: str, parent: str) -> str:
    if parent and parent.strip().upper() != 'ALL':
        return f'{parent} → [{x_marker} vs {y_marker}]'
    return f'[{x_marker} vs {y_marker}]'


def _render_axis_summary(data: dict, use_joint_2d: bool = False) -> str:
    """Render the [Axis distribution] section.

    Reads bin-center / count lists from ``data['axis_distribution']``
    (populated by ``src.vlm_gate.task``) and emits one block per axis.
    Falls back to a TBD stub for old ``gate_task.json`` files where the
    field is null — re-running ``src.vlm_gate.task`` upgrades them.

    The 2-D joint-distribution heatmap is opt-in via ``use_joint_2d``
    (default off — only per-axis 1-D bins are shown).
    """
    ad = data.get('axis_distribution')
    x_marker = data.get('x_marker', 'x')
    y_marker = data.get('y_marker', 'y')
    header = '[Axis distribution]\n'
    if not ad or 'x' not in ad or 'y' not in ad:
        return (
            f'{header}<<TBD: distribution of {x_marker} (x) and '
            f'{y_marker} (y) — re-run src.vlm_gate.task to populate>>\n'
        )
    x_block = render_axis_summary(x_marker, ad['x'], role='x')
    y_block = render_axis_summary(y_marker, ad['y'], role='y')
    pieces = [f'{header}{x_block}', y_block]
    if use_joint_2d:
        jd = data.get('joint_distribution')
        if jd:
            pieces.append(render_joint_summary(x_marker, y_marker, jd))
    return '\n\n'.join(pieces) + '\n'


def _output_format_block(categories: list[str]) -> str:
    """Per-category gate JSON example, with multi-rectangle option.

    Per category the LLM emits **either** a single rectangle dict
    (default) **or** a list of rectangle dicts (only when the
    multi-rectangle criteria in the system prompt are met). The example
    block shows single-rectangle by default and a separate multi-rect
    example to keep the LLM biased toward single — see
    ``inference/system_prompts.py`` for the guardrails.
    """
    cats = [c for c in categories if c != 'Unassigned']
    if not cats:
        return ''
    example_lines = []
    for c in cats:
        example_lines.append(
            f'  "{c}": {{"x_min": <float>, "x_max": <float>, '
            f'"y_min": <float>, "y_max": <float>, '
            f'"rationale": "<brief reason>"}}'
        )
    body = ',\n'.join(example_lines)
    multi_example = (
        '\n\nMulti-rectangle form (only when the multi-rectangle criteria '
        'in the system prompt are met) — replace the chosen category\'s '
        'value with a list of rectangle dicts:\n'
        '```json\n'
        '{\n'
        f'  "{cats[0]}": [\n'
        '    {"x_min": <float>, "x_max": <float>, '
        '"y_min": <float>, "y_max": <float>, "rationale": "<sub-cluster 1>"},\n'
        '    {"x_min": <float>, "x_max": <float>, '
        '"y_min": <float>, "y_max": <float>, "rationale": "<sub-cluster 2>"}\n'
        '  ]\n'
        '}\n'
        '```'
    )
    absent_example = (
        '\n\nAbsent form (only when the criteria in *Declaring a category '
        'absent* in the system prompt are met) — replace the chosen '
        "category's value with an absent dict instead of a gate:\n"
        '```json\n'
        '{\n'
        f'  "{cats[0]}": {{"absent": true, "rationale": "<why this category '
        'has no mass in the parent population>"}}\n'
        '}\n'
        '```'
    )
    return (
        '## Output Format\n'
        'Return a JSON object keyed by category name. For each named '
        'category, output **either** a single axis-aligned rectangular '
        'gate dict (default — see example below) **or** a list of '
        'rectangle dicts (only when the multi-rectangle criteria in the '
        'system prompt are met). Coordinates are in the same numeric '
        'units as the axis distribution shown above. Do **not** output '
        'a gate for `Unassigned` — it is defined as the set of cells '
        "outside every named category's gate(s). Cells falling inside "
        'multiple rectangles (across or within categories) are resolved '
        'at evaluation time (smallest-area rectangle wins).\n\n'
        'Single-rectangle form (default):\n'
        '```json\n'
        '{\n'
        f'{body}\n'
        '}\n'
        '```'
        f'{multi_example}'
        f'{absent_example}'
    )


# ── Prompt formatter ────────────────────────────────────────────────────────

def format_prompt(
    data: dict,
    use_note: bool = True,
    use_joint_2d: bool = False,
    use_notes_yaml: bool = True,
) -> str:
    x_marker = data['x_marker']
    y_marker = data['y_marker']
    parent = data.get('parent', 'ALL')
    categories = data['options']
    breadcrumb = _breadcrumb(x_marker, y_marker, parent)

    desc_lines: list[str] = []
    if use_note:
        # YAML override (benchmark/notes.yaml) takes precedence
        # over the task.json `note` field. Look up by (cohort, step) —
        # cohort is read from data['cohort'] (set by src.vlm_gate.task)
        # or data['_cohort'] (injected by runners that bypass task.json).
        override = None
        if use_notes_yaml:
            cohort = data.get('cohort') or data.get('_cohort')
            override = lookup_note_override(cohort, data.get('step'))
        note = override if override is not None else (data.get('note') or '').strip()
        if note:
            desc_lines.append(note)
    if 'Unassigned' in categories:
        desc_lines.append(
            "Unassigned: cells whose (x, y) position falls outside every "
            "named category's gate at this step."
        )
    desc_section = ('\n[Description]\n' + '\n'.join(desc_lines) + '\n'
                    if desc_lines else '')

    modality_raw = (data.get('modality') or '').strip().lower()
    modality_label = {'cytof': 'CyTOF', 'flow': 'Flow Cytometry'}.get(
        modality_raw, modality_raw or 'unknown',
    )
    cofactor = data.get('cofactor')
    cofactor_str = (f'{cofactor:g}' if isinstance(cofactor, (int, float))
                    else 'unknown')
    modality_line = (f'Modality: {modality_label} '
                     f'(arcsinh cofactor: {cofactor_str})')

    axis_summary = _render_axis_summary(data, use_joint_2d=use_joint_2d)
    output_format = _output_format_block(categories)

    return f"""[Context]
{modality_line}
Path: {breadcrumb}
Step: {data['step']}
X-axis marker: {x_marker}
Y-axis marker: {y_marker}
Parent cells: {data['n_parent_cells']:,}

[Categories]
{', '.join(categories)}
{desc_section}
---

{axis_summary}
---

{output_format}
"""


# ── Single / batch processing ───────────────────────────────────────────────

def _write_prompt(
    data: dict,
    json_path: Path,
    save_md: bool,
    perturbation_name: str | None = None,
    use_note: bool = True,
    use_joint_2d: bool = False,
    use_notes_yaml: bool = True,
) -> None:
    slug = ablation_slug()
    out_dir = ablation_dir(json_path, slug, perturbation_name=perturbation_name)
    prompt = format_prompt(
        data, use_note=use_note, use_joint_2d=use_joint_2d,
        use_notes_yaml=use_notes_yaml,
    )
    if save_md:
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / 'cells.md'
        with open(md_path, 'w') as f:
            f.write(prompt)
        print(f'  {md_path}')
    else:
        print(f'----- [{slug}] -----')
        print(prompt)
        print()


def process_json(
    json_path: str,
    save_md: bool = False,
    perturbation_name: str | None = None,
    use_note: bool = True,
    use_joint_2d: bool = False,
    use_notes_yaml: bool = True,
) -> None:
    json_path = Path(json_path)
    if not json_path.exists():
        print(f'[ERROR] {json_path} not found', file=sys.stderr)
        return

    with open(json_path) as f:
        data = json.load(f)

    # Inject cohort from path layout benchmark/{cohort}/{sample}/step_NN/gate_task.json
    # so notes.yaml override lookup works (task.json doesn't carry cohort itself).
    if '_cohort' not in data and 'cohort' not in data:
        try:
            data['_cohort'] = json_path.parents[2].name
        except IndexError:
            pass

    effective_pname = data.get('perturbation') or perturbation_name
    _write_prompt(data, json_path, save_md,
                  perturbation_name=effective_pname,
                  use_note=use_note,
                  use_joint_2d=use_joint_2d,
                  use_notes_yaml=use_notes_yaml)


def process_batch(
    sample_dir: str,
    save_md: bool = False,
    perturbation_name: str | None = None,
    use_note: bool = True,
    use_joint_2d: bool = False,
    use_notes_yaml: bool = True,
) -> None:
    sample_dir = Path(sample_dir)
    pattern = (f'step_*/gate_task__{perturbation_name}.json'
               if perturbation_name else 'step_*/gate_task.json')
    step_files = sorted(sample_dir.glob(pattern))
    if not step_files:
        print(f'[ERROR] no {pattern} found under {sample_dir}',
              file=sys.stderr)
        return
    for p in step_files:
        process_json(
            str(p), save_md=save_md,
            perturbation_name=perturbation_name,
            use_note=use_note,
            use_joint_2d=use_joint_2d,
            use_notes_yaml=use_notes_yaml,
        )


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build LLM prompts from gate_task.json (one cells.md '
                    'per step). Prompts are written into a sibling '
                    'directory named gate_{ablation_slug}/.',
    )
    parser.add_argument('json_file', nargs='?',
                        help='Path to gate_task.json (single mode)')
    parser.add_argument('--batch', help='Sample directory for batch mode')
    parser.add_argument('--save_md', action='store_true',
                        help='Save prompts to disk (default: print to stdout)')
    parser.add_argument('--without-note', dest='use_note',
                        action='store_false', default=True,
                        help='Exclude the task.json `note` field from the '
                             '[Description] section. Default: include note.')
    parser.add_argument('--without-notes-yaml', dest='use_notes_yaml',
                        action='store_false', default=True,
                        help='Disable the (cohort, step) override from '
                             'benchmark/notes.yaml. Default: yaml '
                             'overrides task.json note when present.')
    parser.add_argument('--with-joint-2d', dest='use_joint_2d',
                        action='store_true', default=False,
                        help='Append the 2-D joint-distribution heatmap '
                             'block to [Axis distribution]. Default: '
                             'per-axis 1-D bins only.')
    parser.add_argument(
        '--perturbation', action='append', default=[],
        help='Inline perturbation spec (repeatable; must match the spec '
             'passed to src.vlm_gate.task). Reads gate_task__{slug}.json '
             'and writes prompts to gate_{ablation}__{slug}/.',
    )
    parser.add_argument('--perturbation-name', default=None,
                        help='Override auto-derived perturbation slug.')
    import json
    from src.hard_depletions import add_hard_depletion_args, resolve_hard_depletion_args
    from src.hard_shifts import add_hard_shift_args, resolve_hard_shift_args
    add_hard_depletion_args(parser)
    add_hard_shift_args(parser)
    args = parser.parse_args()

    resolve_hard_depletion_args(args)
    resolve_hard_shift_args(
        args,
        meta_channels_lookup=lambda ds: json.loads(
            (Path('data') / ds / 'parquet' / 'meta.json').read_text()
        )['channels'],
    )

    if getattr(args, '_resolved_perturbation', None) is not None:
        perturbation_name = args.perturbation_name
    else:
        perturbation_name, _p = parse_perturbations(
            args.perturbation or [], name_override=args.perturbation_name,
        )

    common = dict(
        save_md=args.save_md,
        perturbation_name=perturbation_name,
        use_note=args.use_note,
        use_joint_2d=args.use_joint_2d,
        use_notes_yaml=args.use_notes_yaml,
    )

    if args.batch:
        process_batch(args.batch, **common)
    elif args.json_file:
        process_json(args.json_file, **common)
    else:
        parser.error('Provide a json path or --batch <dir>')


if __name__ == '__main__':
    main()
