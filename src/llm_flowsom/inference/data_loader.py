"""Walk a benchmark directory and assemble per-(sample, step) prompt
groups for the LLM eval runners.

One group = one rendered prompt (``cells.md`` from a ``c2s_{slug}/``
ablation dir) plus the per-cell GT labels needed to score the response.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_c2s(
    benchmark_dir: str,
    ablation_slug: str,
    sample_filter: set | None = None,
    perturbation_name: str | None = None,
) -> list[dict]:
    """Walk ``benchmark_dir/{sample}/step_NN/`` and collect one group per
    (sample, step). Each group holds the rendered prompt text plus the
    per-cell GT labels needed to score the LLM's response.

    Returns list of groups::

        {
          'prompt_text': str,
          'cells':       list[dict],    # [{cell_id, cell_index, gt_label, x, y}, ...]
          'options':     list[str],
          'sample':      str,
          'step':        int,
          'ablation':    str,            # slug
        }

    ``cells`` only includes the anchors whose ``cell_N`` markdown header
    actually appears in ``cells.md`` — c2s_prompt may have truncated via
    ``--n-cells``.
    """
    base = Path(benchmark_dir)
    all_samples = sorted(d for d in base.iterdir() if d.is_dir())
    if sample_filter:
        missing = sample_filter - {d.name for d in all_samples}
        if missing:
            print(f'  [WARN] samples not found: {missing}')
        selected = [d for d in all_samples if d.name in sample_filter]
    else:
        selected = all_samples

    c2s_filename = (f'c2s__{perturbation_name}.json'
                    if perturbation_name else 'c2s.json')
    prompt_dir_name = (f'c2s_{ablation_slug}__{perturbation_name}'
                       if perturbation_name else f'c2s_{ablation_slug}')

    print(f'  Samples: {len(selected)} / {len(all_samples)}  ({benchmark_dir})')
    print(f'  Ablation: {prompt_dir_name}/cells.md')
    if perturbation_name:
        print(f'  Perturbation: {perturbation_name}')

    groups: list[dict] = []
    missing_prompt_dirs = 0
    for sample_dir in selected:
        for step_dir in sorted(sample_dir.iterdir()):
            if not step_dir.is_dir():
                continue
            json_path = step_dir / c2s_filename
            prompt_dir = step_dir / prompt_dir_name
            md_path = prompt_dir / 'cells.md'
            if not json_path.exists():
                continue
            if not md_path.exists():
                if not prompt_dir.is_dir():
                    missing_prompt_dirs += 1
                continue

            data = json.loads(json_path.read_text())
            cells_dict = data.get('cells') or {}
            gt_answers = data.get('gt_answers') or {}
            options = data.get('options', [])

            prompt_text = md_path.read_text()
            # Truncation via --n-cells leaves only the first K cell_N
            # headers in the md; mirror that subset here so the groups
            # list matches what the LLM was actually asked to annotate.
            used_ids = _cell_ids_from_prompt(prompt_text)
            if used_ids is None:
                ordered_cids = sorted(cells_dict.keys(), key=int)
            else:
                ordered_cids = [cid for cid in used_ids if cid in cells_dict]

            cells = []
            for cid in ordered_cids:
                c = cells_dict[cid]
                gt = gt_answers.get(str(int(c['cell_index'])))
                cells.append({
                    'cell_id':     f'cell_{cid}',
                    'cell_index':  int(c['cell_index']),
                    'parquet_row': int(c.get('parquet_row', c['cell_index'])),
                    'gt_label':    gt,
                    'x':           float(c['x']),
                    'y':           float(c['y']),
                })
            if not cells:
                continue

            groups.append({
                'prompt_text': prompt_text,
                'cells':       cells,
                'options':     options,
                'sample':      sample_dir.name,
                'step':        int(data['step']),
                'ablation':    ablation_slug,
                'x_marker':    data.get('x_marker'),
                'y_marker':    data.get('y_marker'),
            })

    if missing_prompt_dirs:
        print(f'  [WARN] {missing_prompt_dirs} step_dir(s) missing '
              f'{prompt_dir_name}/ — run src.llm.c2s_prompt first.')
    print(f'  Found {len(groups)} (sample, step) prompts')
    return groups


def _cell_ids_from_prompt(prompt_text: str) -> list[str] | None:
    """Extract ordered ``cell_N`` ids from ``## cell_N`` headers in the md.

    Returns ``None`` if no headers are found — callers fall back to the
    full c2s.json cell set (covers non-standard prompt layouts).
    """
    import re
    ids = re.findall(r'^## cell_(\d+)\b', prompt_text, flags=re.MULTILINE)
    return ids if ids else None
