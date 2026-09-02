#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI async agentic runner for agent-gate — emits the same hierarchical
``pred.json`` ``src.agent_gate.postprocess.propagate`` consumes, but each
(sample, step) is produced by an iterative tool-call loop instead of a
single chat turn.

Per (sample, step) the loop is:

  1. system + user(cells.md) + tools=[render_gate_overlay].
  2. Turn 1: ``tool_choice="required"`` — model MUST emit a first
     ``render_gate_overlay`` call (clean experiment: every step gets at
     least one visual revision pass).
  3. Mid turns: ``tool_choice="auto"`` — model decides whether to revise
     or to commit to a final JSON answer.
  4. For each tool call we render the overlay (in a worker thread so
     matplotlib does not block the event loop), append a ``tool`` role
     message with the parsed-coords text read-back, then a follow-up
     ``user`` role message carrying the rendered PNG as an
     ``image_url`` data URI. (The OpenAI ``tool`` role cannot carry
     images on the Chat Completions API.)
  5. Final turn: ``tool_choice="none"`` — the model must emit a plain
     text JSON answer. That text is the ``raw`` for ``extract_gates``,
     so the rest of the pipeline (assemble → propagate → eval) sees the
     same shape it sees from ``src.agent_gate`` non-agentic runs.

Usage::

    export OPENAI_API_KEY=sk-...

    python -m src.agent_gate.inference.run_agent \\
        --dataset_path benchmark/Acute2020 \\
        --model        gpt-5.4 \\
        --max-turns    4 \\
        --concurrency  10 \\
        --output_path  results/agent_gpt-5.4_full/Acute2020/pred.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.bench import parse_perturbations
from src.agent_gate.inference.data_loader import load_gate_tasks
from src.agent_gate.inference.output_schema import (
    assemble_eval_output, extract_gates, iter_step_records,
)
from src.agent_gate.inference.system_prompts import get_system_prompt
from src.agent_gate.tools.registry import (
    OVERLAY_TOOL_NAME, dispatch_tool_call, get_tool_schemas,
)
from src.agent_gate.tools.render_overlay import (
    render_bare_scatter, render_gate_trajectory,
)

try:
    from openai import AsyncOpenAI
except ImportError:
    print('[ERROR] openai package not found. Run: pip install openai',
          file=sys.stderr)
    sys.exit(1)

load_dotenv()


# ── per-step agentic loop ───────────────────────────────────────────────────

def _assistant_msg_dict(msg) -> dict:
    """Re-serialize an OpenAI assistant message that carries tool_calls.

    The OpenAI SDK accepts pydantic message objects in ``messages``, but
    constructing the dict explicitly avoids surprises with vendor-extra
    fields (e.g. ``reasoning_content`` from open-source thinking
    parsers, function-style legacy fields).
    """
    out: dict = {
        'role': 'assistant',
        # Always include ``content`` (None is fine in JSON). Native OpenAI
        # accepts both omit and ``content: null`` for assistant-with-
        # tool_calls, but some compat proxies want the field present.
        'content': msg.content,
    }
    if getattr(msg, 'tool_calls', None):
        out['tool_calls'] = [
            {
                'id': tc.id,
                'type': 'function',
                'function': {
                    'name': tc.function.name,
                    'arguments': tc.function.arguments or '{}',
                },
            }
            for tc in msg.tool_calls
        ]
    return out


def _tool_choice_for_turn(turn: int, max_turns: int) -> str:
    """``required`` on turn 1, ``none`` on the last, ``auto`` between.

    The model receives the decorated scatter in the initial user
    message, so the forced first call is now an honest **verification**
    of its initial proposal (not a peek-at-the-canvas hack). Without
    the force, GPT-5.4 confidently skips the tool entirely and the
    agentic loop never fires — empirically observed.
    """
    if turn == max_turns - 1:
        return 'none'
    if turn == 0:
        return 'required'
    return 'auto'


def _dump_overlay(
    dump_dir: Path, sample: str, step: int, turn: int, call_idx: int,
    image_uri: str,
) -> str:
    """Persist one tool-call overlay PNG. Returns the path written."""
    out_dir = dump_dir / sample / f'step_{step:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'turn{turn:02d}_call{call_idx:02d}.png'
    raw = base64.b64decode(image_uri.split(',', 1)[1])
    path.write_bytes(raw)
    return str(path)


async def _run_agent_step(
    client: AsyncOpenAI,
    group: dict,
    *,
    model: str,
    system_prompt: str,
    tool_schemas: list[dict],
    max_turns: int,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    presence_penalty: float | None,
    frequency_penalty: float | None,
    semaphore: asyncio.Semaphore,
    dump_dir: Path | None = None,
) -> tuple[str, dict]:
    """Run the tool-call loop for one (sample, step). Returns
    ``(final_text, trace)`` where ``final_text`` is the model's last
    plain-text content (the JSON answer ``extract_gates`` will parse)
    and ``trace`` is a small dict for ``--save-raw``.
    """
    ctx = {
        'scatter_bg_path': group.get('scatter_bg_path'),
        'scatter_extent':  group['scatter_extent'],
        'x_marker':        group.get('x_marker', 'x'),
        'y_marker':        group.get('y_marker', 'y'),
        'options':         group.get('options', []),
        'step_label':      f'{group["sample"]} step_{group["step"]:02d}',
    }
    # Render the decorated initial scatter (axes + ticks + grid). The
    # raw scatter_bg.png is decoration-free — anything the LLM SEES has
    # to carry tick labels so it can read coordinates. Same wrapper the
    # tool uses when called with empty gates.
    initial_scatter_uri = await asyncio.to_thread(
        render_bare_scatter,
        scatter_bg_path=ctx['scatter_bg_path'],
        scatter_extent=ctx['scatter_extent'],
        x_marker=ctx['x_marker'],
        y_marker=ctx['y_marker'],
        step_label=ctx['step_label'],
    )
    messages: list[dict] = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user',   'content': [
            {'type': 'text',      'text': group['prompt_text']},
            {'type': 'image_url',
             'image_url': {'url': initial_scatter_uri}},
        ]},
    ]

    final_text = ''
    trace = {
        'n_turns':      0,
        'n_tool_calls': 0,
        'tool_diagnostics': [],   # the read-back text for each call
        'tool_args':      [],     # raw `gates` arg per call (None on parse fail)
        'overlay_paths':  [],     # populated when --dump-overlays is set
        'trajectory_path': None,  # populated when --dump-overlays is set
        'finish_reason':  None,
    }

    async with semaphore:
        for turn in range(max_turns):
            tool_choice = _tool_choice_for_turn(turn, max_turns)
            kwargs = dict(
                model=model,
                messages=messages,
                tools=tool_schemas,
                tool_choice=tool_choice,
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            if top_p is not None:
                kwargs['top_p'] = top_p
            if presence_penalty is not None:
                kwargs['presence_penalty'] = presence_penalty
            if frequency_penalty is not None:
                kwargs['frequency_penalty'] = frequency_penalty

            resp = await client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            msg = choice.message
            trace['n_turns'] += 1
            trace['finish_reason'] = choice.finish_reason

            if msg.tool_calls:
                # Append the assistant turn (must precede every tool reply).
                messages.append(_assistant_msg_dict(msg))
                # Dispatch each tool call. matplotlib is blocking + not
                # thread-safe, so we hop to a worker thread per call so
                # concurrent steps do not serialise on the event loop.
                for call_idx, tc in enumerate(msg.tool_calls):
                    trace['n_tool_calls'] += 1
                    if tc.function.name != OVERLAY_TOOL_NAME:
                        text, image_uri = (
                            f'[error] unknown tool {tc.function.name!r}',
                            None,
                        )
                    else:
                        text, image_uri = await asyncio.to_thread(
                            dispatch_tool_call, tc, ctx,
                        )
                    trace['tool_diagnostics'].append(text)
                    # Capture the raw `gates` arg per call so a trajectory
                    # composite can be drawn after the loop. Parse failures
                    # land as None.
                    try:
                        _args = json.loads(tc.function.arguments or '{}')
                        trace['tool_args'].append(
                            _args.get('gates')
                            if isinstance(_args, dict) else None
                        )
                    except (json.JSONDecodeError, TypeError, ValueError):
                        trace['tool_args'].append(None)
                    if image_uri and dump_dir is not None:
                        # disk write also blocks — keep it off the loop
                        path = await asyncio.to_thread(
                            _dump_overlay, dump_dir,
                            group['sample'], group['step'],
                            turn, call_idx, image_uri,
                        )
                        trace['overlay_paths'].append(path)
                    messages.append({
                        'role':         'tool',
                        'tool_call_id': tc.id,
                        'content':      text,
                    })
                    if image_uri:
                        messages.append({
                            'role': 'user',
                            'content': [
                                {'type': 'text',
                                 'text': f'[overlay for tool_call_id={tc.id}]'},
                                {'type': 'image_url',
                                 'image_url': {'url': image_uri}},
                            ],
                        })
                continue

            # No tool call → final content turn. Even on a forced
            # ``tool_choice="none"`` last turn, ``content`` may be empty
            # (e.g., refusal); empty string falls through and the
            # downstream extractor records ``None`` per category.
            final_text = msg.content or ''
            break

    # Trajectory composite: one PNG per step showing the chronological
    # progression of REAL gate proposals through the loop, capped by
    # the committed final answer. Empty/null peek calls are filtered
    # inside ``render_gate_trajectory`` so the legend reflects only
    # actual revisions.
    if dump_dir is not None and trace['tool_args']:
        # Parse the committed final answer into a renderable gate dict
        # so the trajectory ends on the actual evaluated geometry —
        # which can differ from the last tool call when the model
        # revised between the loop and the final text turn.
        final_gates_parsed = extract_gates(final_text, ctx['options']) \
            if final_text else {}
        # extract_gates emits None for missing/invalid; drop those so
        # the trajectory shows committed boxes only.
        final_gates_clean = {c: g for c, g in final_gates_parsed.items()
                             if g is not None}

        traj_path = dump_dir / group['sample'] \
            / f'step_{group["step"]:02d}' / 'trajectory.png'
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            render_gate_trajectory,
            trace['tool_args'],
            str(traj_path),
            scatter_bg_path=ctx['scatter_bg_path'],
            scatter_extent=ctx['scatter_extent'],
            x_marker=ctx['x_marker'],
            y_marker=ctx['y_marker'],
            options=ctx['options'],
            step_label=ctx['step_label'],
            final_gates=final_gates_clean or None,
        )
        trace['trajectory_path'] = str(traj_path)

    return final_text, trace


# ── orchestration ───────────────────────────────────────────────────────────

async def run_online(
    groups: list, model: str, ablation: str, dataset_path: str,
    output_path: str, concurrency: int, max_tokens: int, max_turns: int,
    system_prompt: str, perturbation: str | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    save_raw: bool = False,
    dump_dir: Path | None = None,
) -> None:
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    tool_schemas = get_tool_schemas()

    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        print(f'Dumping overlay PNGs → {dump_dir}/'
              '<sample>/step_NN/turnTT_callCC.png')
    print(f'Sending {len(groups)} agentic loops '
          f'(concurrency={concurrency}, max_turns={max_turns}) ...')
    outputs = await asyncio.gather(
        *[_run_agent_step(
            client, g,
            model=model,
            system_prompt=system_prompt,
            tool_schemas=tool_schemas,
            max_turns=max_turns,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            semaphore=semaphore,
            dump_dir=dump_dir,
          ) for g in groups],
        return_exceptions=True,
    )

    raw_texts: list[str] = []
    traces: list[dict] = []
    for group, output in zip(groups, outputs):
        if isinstance(output, Exception):
            print(f'  [WARN] {group["sample"]} step{group["step"]}: {output}')
            raw_texts.append('')
            traces.append({
                'n_turns': 0, 'n_tool_calls': 0,
                'tool_diagnostics': [],
                'tool_args':      [],
                'overlay_paths':  [],
                'trajectory_path': None,
                'finish_reason': None,
                'error': f'{type(output).__name__}: {output}',
            })
        else:
            text, trace = output
            raw_texts.append(text)
            traces.append(trace)

    eval_data = assemble_eval_output(
        groups, raw_texts,
        model=model, ablation=ablation, dataset_path=dataset_path,
        perturbation=perturbation,
        save_raw=save_raw,
        reasoning_texts=None,
        strip_think=False,
    )

    if save_raw:
        # Inject the agent trace alongside ``raw`` per step. Kept opt-in
        # because ``tool_diagnostics`` adds a few KB per step (and the
        # base64 image bytes are intentionally NOT persisted — the
        # diagnostics text is the agent's actual readable feedback).
        for r, trace in zip(iter_step_records(eval_data), traces):
            sample_step = (eval_data['samples'][r['sample']]
                           ['steps'][str(r['step'])])
            sample_step['agent_trace'] = trace

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    n_steps = sum(len(sample['steps'])
                  for sample in eval_data['samples'].values())
    n_gates = sum(
        sum(1 for g in step['gates'].values() if g is not None)
        for sample in eval_data['samples'].values()
        for step in sample['steps'].values()
    )
    n_calls = sum(t['n_tool_calls'] for t in traces)
    print(f'Wrote {n_steps:,} steps / {n_gates:,} gates / '
          f'{n_calls:,} overlay calls → {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='OpenAI async agentic runner (agent-gate, '
                    'tool-call loop per step)',
    )
    parser.add_argument('--dataset_path', required=True,
                        help='benchmark/{dataset} directory')
    parser.add_argument('--model', default='gpt-5.4')
    parser.add_argument('--output_path', required=True,
                        help='Results JSON path')
    parser.add_argument('--ablation_slug', default='full',
                        help='Prompt directory slug. Default: full')
    parser.add_argument('--sample', nargs='+', default=None,
                        help='Sample names to include')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Random subsample of groups (0 = all)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--concurrency', type=int, default=10,
                        help='Concurrent agentic loops. Default lower than '
                             'run_openai (10 vs 20) — each loop fires '
                             'multiple chat turns + uploads images.')
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument(
        '--max-turns', type=int, default=4,
        help='Max chat turns per (sample, step). Turn 1 always renders '
             '(tool_choice=required); the last turn forces a text JSON '
             'answer (tool_choice=none). Min 2. Default 4 = 1 initial '
             'render + up to 2 revisions + 1 final text turn.',
    )
    parser.add_argument(
        '--temperature', type=float, default=0.0,
        help='Sampling temperature. Default 0.0 (greedy).',
    )
    parser.add_argument('--top-p', type=float, default=None,
                        help='Nucleus sampling. Default: server side.')
    parser.add_argument('--presence-penalty', type=float, default=None)
    parser.add_argument('--frequency-penalty', type=float, default=None)
    parser.add_argument(
        '--save-raw', action='store_true',
        help='Save final raw text + agent_trace (n_turns, n_tool_calls, '
             'per-call diagnostics text) per step. Image bytes are NOT '
             'persisted — only the text read-backs. Use --dump-overlays '
             'to also keep the PNGs.',
    )
    parser.add_argument(
        '--dump-overlays', type=Path, default=None,
        help='Optional directory to save every overlay PNG sent to the '
             'model: <DIR>/<sample>/step_NN/turnTT_callCC.png. The same '
             'paths are recorded under agent_trace.overlay_paths when '
             '--save-raw is set. Useful for debugging / paper figures; '
             'a few hundred KB per call so disk usage adds up quickly.',
    )
    parser.add_argument(
        '--perturbation', action='append', default=[],
        help='Inline perturbation spec (repeatable; must match the spec '
             'passed to src.agent_gate.task). Used to locate '
             'gate_task__{slug}.json + gate_{ablation}__{slug}/ prompts; '
             'recorded in pred.json meta.',
    )
    parser.add_argument('--perturbation-name', default=None,
                        help='Override auto-derived perturbation slug.')
    from src.hard_depletions import (
        add_hard_depletion_args, resolve_hard_depletion_args,
    )
    from src.hard_shifts import (
        add_hard_shift_args, resolve_hard_shift_args,
    )
    add_hard_depletion_args(parser)
    add_hard_shift_args(parser)
    args = parser.parse_args()

    if args.max_turns < 2:
        parser.error('--max-turns must be >= 2 (turn 1 forced to call the '
                     'tool, last turn forced to text — collapses otherwise)')

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

    sample_filter = set(args.sample) if args.sample else None
    groups = load_gate_tasks(
        args.dataset_path,
        ablation_slug=args.ablation_slug,
        sample_filter=sample_filter,
        perturbation_name=perturbation_name,
    )
    if not groups:
        print('No groups found. Exiting.')
        return

    if args.max_samples > 0:
        random.seed(args.seed)
        groups = random.sample(groups, min(args.max_samples, len(groups)))

    print(f'Groups: {len(groups):,}')
    missing_bg = sum(1 for g in groups if not g.get('scatter_bg_path'))
    if missing_bg:
        print(f'  [WARN] {missing_bg} group(s) have no scatter_bg.png — '
              f're-run src.agent_gate.task to populate the background; '
              f'overlays for these will render boxes on a blank plane.')

    asyncio.run(run_online(
        groups,
        model=args.model,
        ablation=args.ablation_slug,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        max_turns=args.max_turns,
        system_prompt=get_system_prompt(),
        perturbation=perturbation_name,
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        save_raw=args.save_raw,
        dump_dir=args.dump_overlays,
    ))


if __name__ == '__main__':
    main()
