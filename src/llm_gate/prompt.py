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

CLI::

    python -m src.llm_gate.prompt benchmark/.../step_09/gate_task.json --save_md
    python -m src.llm_gate.prompt --batch benchmark/Acute2020/994567_Normalized --save_md
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

ABLATION_SLUGS = ('full', 'no_desc', 'no_hist', 'no_pv', 'anon')


def ablation_slug(mode: str = 'full') -> str:
    """Return the ablation slug. Supported modes: full, no_desc, no_hist, anon."""
    if mode not in ABLATION_SLUGS:
        raise ValueError(f'Unknown ablation mode {mode!r}; choose from {ABLATION_SLUGS}')
    return mode


def ablation_dir(
    json_path: Path, slug: str, perturbation_name: str | None = None,
) -> Path:
    name = f'gate_{slug}'
    if perturbation_name:
        name = f'{name}__{perturbation_name}'
    return json_path.parent / name


# ── Helpers ─────────────────────────────────────────────────────────────────

def _build_anon_map(data: dict) -> dict[str, str]:
    """Build a replacement map for anonymizing marker names only.

    Category names are kept as-is — only axis markers and markers
    referenced in the breadcrumb path are anonymized. Also scans the
    note text for isotope_marker patterns (e.g. ``140Ce_Bead``) and
    maps them to the same anonymized name so that the full channel
    reference is replaced cleanly.

    Returns a dict {original: replacement} sorted longest-first so that
    longer tokens are replaced before shorter substrings.
    """
    import re

    rmap: dict[str, str] = {}
    counter = 1

    # Axis markers
    for m in (data['x_marker'], data['y_marker']):
        if m not in rmap:
            rmap[m] = f'Marker_{counter}'
            counter += 1

    # Markers referenced in breadcrumb path (e.g. "Step09_172Yb_CD66b|89Y_CD45")
    parent = data.get('parent', '') or ''
    for part in re.findall(r'Step\d+_(.+?)(?:\s*==|\s*$)', parent):
        for ch in part.split('|'):
            ch = ch.strip()
            if ch and ch not in rmap:
                rmap[ch] = f'Marker_{counter}'
                counter += 1

    # Scan note text for isotope_marker patterns (e.g. "140Ce_Bead",
    # "172Yb_CD66b") and map the full string to the marker's anon name.
    note = (data.get('note') or '')
    for marker, anon_name in list(rmap.items()):
        # Match patterns like "140Ce_Bead" — digits + element + _ + marker
        for full_ch in re.findall(r'\d+[A-Za-z]+_' + re.escape(marker), note):
            if full_ch not in rmap:
                rmap[full_ch] = anon_name

    # Also map marker substrings found in note text. E.g. y_marker is
    # "CD56_NCAM" but note uses "CD56" (short form). Scan note for
    # words that are a prefix/substring of a known marker and map them
    # to the same anon name.
    for marker, anon_name in list(rmap.items()):
        # If marker has underscore parts (e.g. CD56_NCAM), check if
        # note uses the prefix (CD56)
        if '_' in marker:
            prefix = marker.split('_')[0]
            if prefix and len(prefix) >= 3 and prefix not in rmap:
                # Verify it actually appears in note
                if re.search(r'\b' + re.escape(prefix) + r'(?:[+\-\s]|$)', note):
                    rmap[prefix] = anon_name

    return dict(sorted(rmap.items(), key=lambda x: -len(x[0])))


def _apply_anon(text: str, anon_map: dict[str, str],
                categories: list[str]) -> str:
    """Apply anonymization replacements (longest-first).

    Marker names are anonymized everywhere EXCEPT in [Categories] and
    ## Output Format sections (where category names must stay intact for
    JSON key matching). Category names that overlap with marker names
    (e.g. 'Bead') are protected via temporary placeholders.
    """
    # Sections where category names must be preserved verbatim:
    # [Categories] line, and ## Output Format block.
    # Strategy: protect category names → replace markers → restore.
    cat_placeholders = {}
    for i, cat in enumerate(categories):
        ph = f'\x00CAT{i}\x00'
        cat_placeholders[ph] = cat
        text = text.replace(cat, ph)

    # Apply marker replacements globally
    for orig, repl in anon_map.items():
        text = text.replace(orig, repl)

    # Restore category names
    for ph, cat in cat_placeholders.items():
        text = text.replace(ph, cat)

    return text


def _breadcrumb(x_marker: str, y_marker: str, parent: str) -> str:
    if parent and parent.strip().upper() != 'ALL':
        return f'{parent} → [{x_marker} vs {y_marker}]'
    return f'[{x_marker} vs {y_marker}]'


def _render_axis_summary(data: dict, use_joint_2d: bool = False) -> str:
    """Render the [Axis distribution] section.

    Reads bin-center / count lists from ``data['axis_distribution']``
    (populated by ``src.llm_gate.task``) and emits one block per axis.
    Falls back to a TBD stub for old ``gate_task.json`` files where the
    field is null — re-running ``src.llm_gate.task`` upgrades them.

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
            f'{y_marker} (y) — re-run src.llm_gate.task to populate>>\n'
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
    ablation: str = 'full',
) -> str:
    x_marker = data['x_marker']
    y_marker = data['y_marker']
    parent = data.get('parent', 'ALL')
    categories = data['options']

    # --- For anon: build marker map and apply to marker names,
    # breadcrumb, description text — but NOT category names ---
    anon_map: dict[str, str] = {}
    cat_anon_map: dict[str, str] = {}  # original → A, B, C, ...
    if ablation == 'anon':
        anon_map = _build_anon_map(data)
        # Build category anonymization map: A, B, C, ...
        named_cats = [c for c in categories if c != 'Unassigned']
        for i, c in enumerate(named_cats):
            cat_anon_map[c] = chr(ord('A') + i)

    def _anon(s: str) -> str:
        """Replace marker names in a string (anon mode only)."""
        if not anon_map:
            return s
        for orig, repl in anon_map.items():
            s = s.replace(orig, repl)
        return s

    def _anon_cat(s: str) -> str:
        """Replace category names in a string (anon mode only).
        Longer names replaced first to avoid partial matches."""
        if not cat_anon_map:
            return s
        for orig, repl in sorted(cat_anon_map.items(), key=lambda x: -len(x[0])):
            s = s.replace(orig, repl)
        return s

    # Anonymize marker names used in Context / Axis headers
    display_x = _anon(x_marker)
    display_y = _anon(y_marker)
    display_parent = _anon(parent) if parent else parent
    breadcrumb = _breadcrumb(display_x, display_y, display_parent)
    # Anonymize category names in breadcrumb too (e.g. "== Mononuclear →")
    breadcrumb = _anon_cat(breadcrumb)

    # Display categories: anonymize named ones, keep Unassigned
    display_categories = [cat_anon_map.get(c, c) for c in categories]

    # --- Description section (skipped for no_desc) ---
    desc_section = ''
    if ablation != 'no_desc':
        desc_lines: list[str] = []
        if use_note:
            # YAML override (benchmark/notes.yaml) takes precedence
            override = None
            if use_notes_yaml:
                cohort = data.get('cohort') or data.get('_cohort')
                override = lookup_note_override(cohort, data.get('step'))
            note = override if override is not None else (data.get('note') or '').strip()
            if note:
                if anon_map or cat_anon_map:
                    import re as _re
                    for orig, repl in anon_map.items():
                        if _re.match(r'\d+[A-Za-z]+_', orig):
                            note = note.replace(orig, repl)
                    note = _anon_cat(note)
                    note = _anon(note)
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

    # --- Axis distribution (skipped for no_hist) ---
    if ablation == 'no_hist':
        axis_summary = ''
    else:
        axis_summary = _render_axis_summary(data, use_joint_2d=use_joint_2d)
        if ablation == 'no_pv':
            # Strip peak/valley label lines and P/V numeric lines,
            # keep header + histogram bar + range line only.
            import re
            axis_summary = re.sub(
                r'^[ ]*[PV]\d.*\n', '', axis_summary, flags=re.MULTILINE,
            )
        if anon_map:
            axis_summary = _anon(axis_summary)

    output_format = _output_format_block(display_categories)

    prompt = f"""[Context]
{modality_line}
Path: {breadcrumb}
Step: {data['step']}
X-axis marker: {display_x}
Y-axis marker: {display_y}
Parent cells: {data['n_parent_cells']:,}

[Categories]
{', '.join(display_categories)}
{desc_section}
---

{axis_summary}
---

{output_format}
"""

    return prompt


# ── Single / batch processing ───────────────────────────────────────────────

def _write_prompt(
    data: dict,
    json_path: Path,
    save_md: bool,
    perturbation_name: str | None = None,
    use_note: bool = True,
    use_joint_2d: bool = False,
    use_notes_yaml: bool = True,
    ablation_mode: str = 'full',
) -> None:
    slug = ablation_slug(ablation_mode)
    out_dir = ablation_dir(json_path, slug, perturbation_name=perturbation_name)
    prompt = format_prompt(
        data, use_note=use_note, use_joint_2d=use_joint_2d,
        use_notes_yaml=use_notes_yaml, ablation=ablation_mode,
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
    ablation_mode: str = 'full',
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
                  use_notes_yaml=use_notes_yaml,
                  ablation_mode=ablation_mode)


def process_batch(
    sample_dir: str,
    save_md: bool = False,
    perturbation_name: str | None = None,
    use_note: bool = True,
    use_joint_2d: bool = False,
    use_notes_yaml: bool = True,
    ablation_mode: str = 'full',
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
            ablation_mode=ablation_mode,
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
    parser.add_argument('--ablation', default='full',
                        choices=ABLATION_SLUGS,
                        help='Ablation mode: full (baseline), no_desc '
                             '(remove [Description]), no_hist (remove '
                             '[Axis distribution]), anon (anonymize '
                             'marker/category names). Default: full.')
    parser.add_argument(
        '--perturbation', action='append', default=[],
        help='Inline perturbation spec (repeatable; must match the spec '
             'passed to src.llm_gate.task). Reads gate_task__{slug}.json '
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
        ablation_mode=args.ablation,
    )

    if args.batch:
        process_batch(args.batch, **common)
    elif args.json_file:
        process_json(args.json_file, **common)
    else:
        parser.error('Provide a json path or --batch <dir>')


if __name__ == '__main__':
    main()
