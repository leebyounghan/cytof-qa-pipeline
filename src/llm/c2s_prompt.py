#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt builder for ``c2s.json`` — one LLM prompt per (sample, step).

Each prompt asks the model to annotate a batch of individual anchor cells
selected by axis-weighted farthest-point sampling. Prompts are
**cell-level only** (no cluster narrative, no neighbor connectivity, no
global distribution summary).

Layout — prompts go into an ablation-named sibling directory of
``c2s.json``, one ``cells.md`` file per step::

    step_09/
      c2s.json
      c2s_full/           # default (all sections on)
        cells.md
      c2s_nohvp/          # --no-top-hvp
        cells.md
      c2s_hvp10/          # --top-hvp-n 10
        cells.md

``--n-cells N`` truncates the anchor set to the first N cells of
``c2s.json``'s anchor list. Because FPS is coarse-to-refined, the
prefix is itself a valid N-point covering.

CLI::

    # Default (full) ablation, single step
    python -m src.c2s_prompt benchmark/.../step_09/c2s.json --save_md

    # Batch every step in a sample
    python -m src.c2s_prompt --batch benchmark/Acute2020/994567_Normalized --save_md

    # Specific ablation
    python -m src.c2s_prompt --batch <sample_dir> --save_md --no-top-hvp
    python -m src.c2s_prompt --batch <sample_dir> --save_md --top-hvp-n 10

    # Cap at 20 anchors
    python -m src.c2s_prompt --batch <sample_dir> --save_md --n-cells 20

    # Every ablation variant at once
    python -m src.c2s_prompt --batch <sample_dir> --ablation-all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from src.bench import parse_perturbations
from src.llm_gate.utils.axis_distribution import render_axis_summary
from src.llm_gate.utils.joint_distribution import render_joint_summary


# ── Note YAML override (optional, mirrors llm_gate) ─────────────────────────

_DEFAULT_NOTES_PATH = Path(__file__).resolve().parents[2] / 'benchmark' / 'notes.yaml'


def _load_notes(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _lookup_note_override(
    notes: dict, cohort: str | None, step: int | str | None,
) -> str | None:
    if not cohort or step is None:
        return None
    cohort_notes = notes.get(cohort) or {}
    try:
        step_int = int(step)
    except (TypeError, ValueError):
        return None
    val = cohort_notes.get(step_int)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


# ── Ablation config ─────────────────────────────────────────────────────────

def ablation_slug(
    use_top_hvp: bool = True,
    top_hvp_n: int | None = None,
    use_notes_yaml: bool = False,
    use_axis_distribution: bool = False,
    use_joint_2d: bool = False,
) -> str:
    """Return the directory name slug for an ablation config.

    Base: ``full`` (HVP on, no truncation), ``nohvp``, or ``hvp{N}``.
    Suffixes (in order) reflect optional gate-style sections that have
    been opted into: ``_notes``, ``_dist``, ``_joint`` (joint implies
    dist). All three are off by default — cell-sentence only.
    """
    if not use_top_hvp:
        base = 'nohvp'
    elif top_hvp_n is not None:
        base = f'hvp{top_hvp_n}'
    else:
        base = 'full'
    suffix = ''
    if use_notes_yaml:
        suffix += '_notes'
    if use_axis_distribution:
        suffix += '_dist'
        if use_joint_2d:
            suffix += '_joint'
    return f'{base}{suffix}'


def ablation_dir(
    json_path: Path, slug: str, perturbation_name: str | None = None,
) -> Path:
    """Prompt directory for a given c2s.json path and ablation slug.

    Under a perturbation, the directory is suffixed with ``__{pslug}`` so
    perturbed and unperturbed prompts can coexist for the same ablation.
    """
    name = f'c2s_{slug}'
    if perturbation_name:
        name = f'{name}__{perturbation_name}'
    return json_path.parent / name


# ── Helpers ─────────────────────────────────────────────────────────────────

def _breadcrumb(x_marker: str, y_marker: str, parent: str) -> str:
    if parent and parent.strip().upper() != 'ALL':
        return f'{parent} → [{x_marker} vs {y_marker}]'
    return f'[{x_marker} vs {y_marker}]'


def _render_cell_sentence(
    cell: dict,
    x_marker: str,
    y_marker: str,
    hvp_filter: set[str] | None = None,
) -> str:
    """``[axis] x(val), y(val) | [hvp] ...`` — matches v2's layout.

    When ``hvp_filter`` is passed, only HVP tokens whose marker name is
    in the set are kept (order preserved). ``None`` means no filter;
    an empty set would drop the HVP portion entirely.
    """
    axis = f'[axis] {x_marker}({cell["x"]:.2f}), {y_marker}({cell["y"]:.2f})'
    hvp = cell.get('cell_sentence', '')
    if hvp and hvp_filter is not None:
        tokens = [t.strip() for t in hvp.split('>')]
        kept = [t for t in tokens
                if t and t.split('(', 1)[0].strip() in hvp_filter]
        hvp = ' > '.join(kept)
    return f'{axis} | [hvp] {hvp}' if hvp else axis


def _render_axis_summary_block(data: dict, use_joint_2d: bool = False) -> str:
    """Render the optional ``[Axis distribution]`` section.

    Mirrors llm_gate.prompt._render_axis_summary so the cell-sentence
    pipeline can opt into the same distribution context as gate
    prompts when comparing the two modalities. Returns '' if c2s.json
    does not carry distribution fields (older runs).
    """
    ad = data.get('axis_distribution')
    if not ad or 'x' not in ad or 'y' not in ad:
        return ''
    x_marker = data.get('x_marker', 'x')
    y_marker = data.get('y_marker', 'y')
    pieces = [
        '[Axis distribution]\n' + render_axis_summary(x_marker, ad['x'], role='x'),
        render_axis_summary(y_marker, ad['y'], role='y'),
    ]
    if use_joint_2d:
        jd = data.get('joint_distribution')
        if jd:
            pieces.append(render_joint_summary(x_marker, y_marker, jd))
    return '\n\n'.join(pieces) + '\n'


# ── Prompt formatter ────────────────────────────────────────────────────────

def format_prompt(
    data: dict,
    use_top_hvp: bool = True,
    top_hvp_n: int | None = None,
    n_cells: int | None = None,
    use_tips: bool = False,
    use_notes_yaml: bool = False,
    use_axis_distribution: bool = False,
    use_joint_2d: bool = False,
    notes: dict | None = None,
) -> str:
    """Build one prompt for the step's anchor batch.

    Ablation flags:
      use_top_hvp : when False, drop the ``[Top HVP markers]`` section
                    AND drop every HVP token from each cell sentence.
      top_hvp_n   : when set, truncate the HVP list (and per-cell HVP
                    tokens) to the first N markers. Ignored if
                    ``use_top_hvp=False``.
      n_cells     : when set, use only the first N anchors from c2s.json.
                    Default (None) uses every anchor.
      use_tips    : when True, append the c2s.json ``tips`` field to the
                    [Description] section. Default False — tips are off.
    """
    x_marker = data['x_marker']
    y_marker = data['y_marker']
    parent = data.get('parent', 'ALL')
    categories = data['options']

    top_hvp = [m for m in data.get('top_hvp_markers', [])
               if m not in (x_marker, y_marker)]
    if top_hvp_n is not None and use_top_hvp:
        top_hvp = top_hvp[:top_hvp_n]

    if not use_top_hvp:
        hvp_filter: set[str] | None = set()  # drop all HVP tokens
    elif top_hvp_n is not None:
        hvp_filter = set(top_hvp)            # keep only truncated set
    else:
        hvp_filter = None                    # no filter — keep all

    cells_dict = data['cells']
    cell_ids_sorted = sorted(cells_dict.keys(), key=int)
    if n_cells is not None:
        cell_ids_sorted = cell_ids_sorted[:n_cells]
    n_used = len(cell_ids_sorted)
    breadcrumb = _breadcrumb(x_marker, y_marker, parent)

    n_parent = int(data.get('n_parent_cells', 0))
    cell_lines = []
    for cid in cell_ids_sorted:
        cell = cells_dict[cid]
        sentence = _render_cell_sentence(
            cell, x_marker, y_marker, hvp_filter=hvp_filter,
        )
        n = int(cell.get('n_cells', 0))
        if n and n_parent:
            header_extra = f' (n={n:,}, {n / n_parent * 100:.1f}%)'
        elif n:
            header_extra = f' (n={n:,})'
        else:
            header_extra = ''
        cell_lines.append(f'## cell_{cid}{header_extra}\n```\n{sentence}\n```')
    cells_section = '\n\n'.join(cell_lines)

    hvp_section = (f'\n[Top HVP markers (excl. axis)]\n{", ".join(top_hvp)}\n'
                   if use_top_hvp and top_hvp else '')

    override = None
    if use_notes_yaml and notes is not None:
        cohort = data.get('cohort') or data.get('_cohort')
        override = _lookup_note_override(notes, cohort, data.get('step'))
    note = (override if override is not None
            else (data.get('note') or '').strip())
    desc_lines: list[str] = []
    if note:
        desc_lines.append(note)
    if use_tips:
        tips = (data.get('tips') or '').strip()
        if tips:
            desc_lines.append(f'Tips: {tips}')
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

    axis_block = (_render_axis_summary_block(data, use_joint_2d=use_joint_2d)
                  if use_axis_distribution else '')
    axis_section = f'\n{axis_block}---\n\n' if axis_block else ''

    return f"""[Context]
{modality_line}
Path: {breadcrumb}
Step: {data['step']}
Parent cells: {data['n_parent_cells']:,}
Representative cells in this batch: {n_used}

[Categories]
{', '.join(categories)}
{desc_section}{hvp_section}
---

{axis_section}{cells_section}

---

## Output Format
Return a JSON object keyed by cell ID (``cell_1``, ``cell_2``, ...).
You MUST annotate ALL {n_used} cells listed above — do not skip any.
```json
{{
  "cell_1": {{"annotation": "CATEGORY", "rationale": "brief reason"}},
  "cell_2": {{"annotation": "CATEGORY", "rationale": "brief reason"}},
  ...
}}
```"""


# ── Single / batch processing ───────────────────────────────────────────────

def _write_variant(
    data: dict,
    json_path: Path,
    use_top_hvp: bool,
    top_hvp_n: int | None,
    save_md: bool,
    n_cells: int | None = None,
    perturbation_name: str | None = None,
    use_tips: bool = False,
    use_notes_yaml: bool = False,
    use_axis_distribution: bool = False,
    use_joint_2d: bool = False,
    notes: dict | None = None,
    prompt_subdir: str | None = None,
) -> None:
    slug = ablation_slug(
        use_top_hvp=use_top_hvp, top_hvp_n=top_hvp_n,
        use_notes_yaml=use_notes_yaml,
        use_axis_distribution=use_axis_distribution,
        use_joint_2d=use_joint_2d,
    )
    if prompt_subdir:
        # Explicit directory name (e.g. c2s_den_hvp10) — bypasses the
        # auto-derived c2s_{slug} so paradigms with their own JSON source
        # (density-grid) never collide with the clustering c2s_{slug}/ dir.
        out_dir = json_path.parent / prompt_subdir
    else:
        out_dir = ablation_dir(json_path, slug,
                               perturbation_name=perturbation_name)
    prompt = format_prompt(
        data,
        use_top_hvp=use_top_hvp, top_hvp_n=top_hvp_n,
        n_cells=n_cells,
        use_tips=use_tips,
        use_notes_yaml=use_notes_yaml,
        use_axis_distribution=use_axis_distribution,
        use_joint_2d=use_joint_2d,
        notes=notes,
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
    use_top_hvp: bool = True,
    top_hvp_n: int | None = None,
    ablation_all: bool = False,
    n_cells: int | None = None,
    perturbation_name: str | None = None,
    use_tips: bool = False,
    use_notes_yaml: bool = False,
    use_axis_distribution: bool = False,
    use_joint_2d: bool = False,
    notes: dict | None = None,
    prompt_subdir: str | None = None,
) -> None:
    json_path = Path(json_path)
    if not json_path.exists():
        print(f'[ERROR] {json_path} not found', file=sys.stderr)
        return

    with open(json_path) as f:
        data = json.load(f)
    if not data.get('cells'):
        print(f'[ERROR] no cells in {json_path}', file=sys.stderr)
        return

    # Inject cohort from path layout benchmark/{cohort}/{sample}/step_NN/c2s.json
    # so notes.yaml override lookup works (c2s.json may not carry cohort).
    if '_cohort' not in data and 'cohort' not in data:
        try:
            data['_cohort'] = json_path.parents[2].name
        except IndexError:
            pass

    # If the c2s.json itself records a perturbation, prefer that over the
    # CLI arg so prompts always land in the directory matching their source.
    effective_pname = data.get('perturbation') or perturbation_name

    common = dict(
        save_md=save_md, n_cells=n_cells,
        perturbation_name=effective_pname, use_tips=use_tips,
        use_notes_yaml=use_notes_yaml,
        use_axis_distribution=use_axis_distribution,
        use_joint_2d=use_joint_2d,
        notes=notes,
        prompt_subdir=prompt_subdir,
    )
    if ablation_all:
        variants = [(True, None), (False, None)]
        for uth, n in variants:
            _write_variant(data, json_path, uth, n, **common)
    else:
        _write_variant(data, json_path, use_top_hvp, top_hvp_n, **common)


def process_batch(
    sample_dir: str,
    save_md: bool = False,
    use_top_hvp: bool = True,
    top_hvp_n: int | None = None,
    ablation_all: bool = False,
    n_cells: int | None = None,
    perturbation_name: str | None = None,
    use_tips: bool = False,
    use_notes_yaml: bool = False,
    use_axis_distribution: bool = False,
    use_joint_2d: bool = False,
    notes: dict | None = None,
    prompt_subdir: str | None = None,
) -> None:
    sample_dir = Path(sample_dir)
    pattern = (f'step_*/c2s__{perturbation_name}.json'
               if perturbation_name else 'step_*/c2s.json')
    step_files = sorted(sample_dir.glob(pattern))
    if not step_files:
        print(f'[ERROR] no {pattern} found under {sample_dir}',
              file=sys.stderr)
        if perturbation_name:
            available = sorted({
                p.stem.removeprefix('c2s__')
                for p in sample_dir.glob('step_*/c2s__*.json')
            })
            if available:
                print(f'[hint]  available perturbation slugs in this sample: '
                      f'{available}', file=sys.stderr)
                print(f'[hint]  expected slug was {perturbation_name!r} — '
                      f'likely a shlex-quoting issue in --perturbation spec '
                      f'(wrap multi-word values in quotes, e.g. '
                      f'\'deplete "col|name" "value with spaces"\').',
                      file=sys.stderr)
        return
    for p in step_files:
        process_json(
            str(p), save_md=save_md,
            use_top_hvp=use_top_hvp, top_hvp_n=top_hvp_n,
            ablation_all=ablation_all,
            n_cells=n_cells,
            perturbation_name=perturbation_name,
            use_tips=use_tips,
            use_notes_yaml=use_notes_yaml,
            use_axis_distribution=use_axis_distribution,
            use_joint_2d=use_joint_2d,
            notes=notes,
            prompt_subdir=prompt_subdir,
        )


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build LLM prompts from c2s.json (one cells.md per step). '
                    'Prompts are written into a sibling directory named '
                    'c2s_{ablation_slug}/',
    )
    parser.add_argument('json_file', nargs='?',
                        help='Path to c2s.json (single mode)')
    parser.add_argument('--batch', help='Sample directory for batch mode')
    parser.add_argument('--save_md', action='store_true',
                        help='Save prompts to disk (default: print to stdout)')
    parser.add_argument('--n-cells', dest='n_cells', type=int, default=None,
                        help='Use only the first N anchors from c2s.json. '
                             'Default: use every anchor.')

    # Ablation flags
    parser.add_argument('--no-top-hvp', dest='no_top_hvp',
                        action='store_true',
                        help='Drop [Top HVP] section AND strip HVP tokens '
                             'from cell sentences. Writes to c2s_nohvp/.')
    parser.add_argument('--top-hvp-n', dest='top_hvp_n', type=int, default=None,
                        help='Truncate HVP list + per-cell HVP tokens to '
                             'first N markers. Writes to c2s_hvp{N}/. '
                             'Mutually exclusive with --no-top-hvp.')
    parser.add_argument('--ablation-all', dest='ablation_all',
                        action='store_true',
                        help='Generate every ablation variant (currently: '
                             'full + nohvp) — one dir each.')
    parser.add_argument('--with-tips', dest='with_tips',
                        action='store_true',
                        help='Include the task.json `tips` field (carried '
                             'through c2s.json) in the [Description] section. '
                             'Default: omit tips.')
    # Optional gate-style sections — off by default for cell-sentence runs.
    parser.add_argument('--with-notes-yaml', dest='with_notes_yaml',
                        action='store_true',
                        help='Override task.json `note` with the per-(cohort, '
                             'step) entry from --notes-yaml-path. Default: '
                             'off (use task.json note as-is).')
    parser.add_argument('--notes-yaml-path', default=None,
                        help='Path to notes.yaml. Default: benchmark/notes.yaml '
                             '(may not exist — no-op then).')
    parser.add_argument('--with-axis-distribution', dest='with_axis_distribution',
                        action='store_true',
                        help='Append the [Axis distribution] (1-D bins) '
                             'section, mirroring llm_gate prompts. Default: '
                             'off — cell-sentence only.')
    parser.add_argument('--with-joint-2d', dest='with_joint_2d',
                        action='store_true',
                        help='Append the 2-D joint-distribution heatmap '
                             '(requires --with-axis-distribution). Default: '
                             'off.')
    parser.add_argument(
        '--perturbation', action='append', default=[],
        help='Inline perturbation spec (repeatable; must match the spec '
             'passed to src.llm.c2s). Reads c2s__{slug}.json and writes '
             'prompts to c2s_{ablation}__{slug}/.',
    )
    parser.add_argument('--perturbation-name', default=None,
                        help='Override auto-derived perturbation slug.')
    parser.add_argument('--prompt-subdir', dest='prompt_subdir', default=None,
                        help='Explicit prompt directory name (e.g. '
                             'c2s_den_hvp10), written as a sibling of the '
                             'source JSON. Overrides the auto-derived '
                             'c2s_{ablation_slug} name — use this for the '
                             'density-grid paradigm so its prompts never '
                             'collide with the clustering c2s_{slug}/ dir.')
    from src.hard_depletions import add_hard_depletion_args, resolve_hard_depletion_args
    add_hard_depletion_args(parser)
    args = parser.parse_args()

    if args.no_top_hvp and args.top_hvp_n is not None:
        parser.error('--no-top-hvp and --top-hvp-n are mutually exclusive')
    if args.with_joint_2d and not args.with_axis_distribution:
        parser.error('--with-joint-2d requires --with-axis-distribution')

    use_top_hvp = not args.no_top_hvp

    resolve_hard_depletion_args(args)

    perturbation_name, _p = parse_perturbations(
        args.perturbation or [], name_override=args.perturbation_name,
    )

    notes_path = (Path(args.notes_yaml_path) if args.notes_yaml_path
                  else _DEFAULT_NOTES_PATH)
    notes = _load_notes(notes_path) if args.with_notes_yaml else None

    common = dict(
        save_md=args.save_md,
        use_top_hvp=use_top_hvp,
        top_hvp_n=args.top_hvp_n,
        ablation_all=args.ablation_all,
        n_cells=args.n_cells,
        perturbation_name=perturbation_name,
        use_tips=args.with_tips,
        use_notes_yaml=args.with_notes_yaml,
        use_axis_distribution=args.with_axis_distribution,
        use_joint_2d=args.with_joint_2d,
        notes=notes,
        prompt_subdir=args.prompt_subdir,
    )

    if args.batch:
        process_batch(args.batch, **common)
    elif args.json_file:
        process_json(args.json_file, **common)
    else:
        parser.error('Provide a json path or --batch <dir>')


if __name__ == '__main__':
    main()
