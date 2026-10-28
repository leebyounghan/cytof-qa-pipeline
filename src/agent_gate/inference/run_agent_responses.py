"""Responses API agent-gate runner for GPT-6 Luna with reasoning and tools."""

import argparse
import asyncio
import json
import random
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.agent_gate.inference.output_schema import assemble_eval_output, iter_step_records
from src.agent_gate.inference.system_prompts import get_system_prompt
from src.agent_gate.tools.registry import dispatch_tool_call, get_tool_schemas
from src.agent_gate.tools.render_overlay import render_bare_scatter
from src.visual_variants import describe_plot, load_visual_groups


def _response_tools(style: str) -> list[dict]:
    return [dict(type='function', name=t['function']['name'],
                 description=describe_plot(t['function']['description'], style),
                 parameters=t['function']['parameters'], strict=False)
            for t in get_tool_schemas()]


async def _one(client, group, semaphore, *, model, effort, max_tokens, max_turns,
               temperature, tools, system_prompt, style):
    ctx = {
        'scatter_bg_path': group['scatter_bg_path'],
        'scatter_extent': group['scatter_extent'],
        'x_marker': group['x_marker'],
        'y_marker': group['y_marker'],
        'options': group['options'],
        'step_label': f'step_{group["step"]:02d}',
    }
    initial_uri = await asyncio.to_thread(
        render_bare_scatter, scatter_bg_path=ctx['scatter_bg_path'],
        scatter_extent=ctx['scatter_extent'], x_marker=ctx['x_marker'],
        y_marker=ctx['y_marker'], step_label=ctx['step_label'],
    )
    turn_input = [{
        'role': 'user',
        'content': [
            {'type': 'input_text', 'text': group['prompt_text']},
            {'type': 'input_image', 'image_url': initial_uri},
        ],
    }]
    trace = {'n_turns': 0, 'n_tool_calls': 0, 'tool_diagnostics': [],
             'tool_args': [], 'finish_reason': None, 'api': 'responses'}
    previous_id = None
    final_text = ''
    async with semaphore:
        for turn in range(max_turns):
            tool_choice = ('required' if turn == 0 else
                           'none' if turn == max_turns - 1 else 'auto')
            kwargs = {
                'model': model,
                'instructions': system_prompt,
                'input': turn_input,
                'tools': tools,
                'tool_choice': tool_choice,
                'reasoning': {'effort': effort},
                'max_output_tokens': max_tokens,
                'temperature': temperature,
                'store': True,
            }
            if previous_id is not None:
                kwargs['previous_response_id'] = previous_id
            response = await client.responses.create(**kwargs)
            if response.status != 'completed':
                raise RuntimeError(f'Response status {response.status}: '
                                   f'{response.incomplete_details}')
            trace['n_turns'] += 1
            previous_id = response.id
            calls = [item for item in response.output
                     if item.type == 'function_call']
            if not calls:
                final_text = response.output_text or ''
                trace['finish_reason'] = 'final_text'
                if not final_text:
                    raise RuntimeError('Empty final response')
                break
            if tool_choice == 'none':
                raise RuntimeError('Tool call returned despite tool_choice=none')
            turn_input = []
            for call in calls:
                trace['n_tool_calls'] += 1
                fake_call = SimpleNamespace(
                    function=SimpleNamespace(name=call.name,
                                             arguments=call.arguments),
                    id=call.call_id,
                )
                text, image_uri = await asyncio.to_thread(
                    dispatch_tool_call, fake_call, ctx,
                )
                if text.startswith('[error]'):
                    raise RuntimeError(f'Tool error: {text}')
                text = text.replace(
                    'The overlay image follows in the next message.',
                    'The overlay image is included in this tool result.',
                )
                text = describe_plot(text, style)
                trace['tool_diagnostics'].append(text)
                args = json.loads(call.arguments or '{}')
                trace['tool_args'].append(args.get('gates'))
                output = [{'type': 'input_text', 'text': text}]
                if image_uri:
                    output.append({'type': 'input_image', 'image_url': image_uri})
                turn_input.append({'type': 'function_call_output',
                                   'call_id': call.call_id, 'output': output})
        else:
            raise RuntimeError('Turn budget exhausted without final JSON')
    return final_text, trace


async def run(args):
    groups = load_visual_groups(args.dataset_path, args.input_root,
                                args.plot_style,
                                set(args.sample) if args.sample else None)
    if args.max_samples:
        random.seed(args.seed)
        groups = random.sample(groups, min(args.max_samples, len(groups)))
    output_path = Path(args.output_path)
    checkpoint_path = output_path.with_name(output_path.stem + '.checkpoint.json')
    checkpoint_meta = {'model': args.model, 'dataset_path': args.dataset_path,
                       'input_root': args.input_root, 'plot_style': args.plot_style,
                       'effort': args.api_reasoning_effort,
                       'temperature': args.temperature, 'max_turns': args.max_turns}
    key = lambda group: f'{group["sample"]}\t{group["step"]}'
    expected_keys = {key(group) for group in groups}
    if len(expected_keys) != len(groups):
        raise ValueError('Duplicate sample/step in prepared inputs')
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint['meta'] != checkpoint_meta:
            raise ValueError(f'Checkpoint configuration mismatch: {checkpoint_path}')
        records = checkpoint['records']
        if set(records) - expected_keys:
            raise ValueError(f'Checkpoint contains unexpected steps: {checkpoint_path}')
    else:
        records = {}
    remaining = [group for group in groups if key(group) not in records]
    print(f'{len(groups)} total steps; {len(records)} checkpointed; '
          f'{len(remaining)} to request', flush=True)

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)
    tools = _response_tools(args.plot_style)
    system_prompt = describe_plot(get_system_prompt(), args.plot_style)
    for start in range(0, len(remaining), args.batch_size):
        batch = remaining[start:start + args.batch_size]
        outputs = await asyncio.gather(*[
            _one(client, group, semaphore, model=args.model,
                 effort=args.api_reasoning_effort, max_tokens=args.max_tokens,
                 max_turns=args.max_turns, temperature=args.temperature,
                 tools=tools, system_prompt=system_prompt,
                 style=args.plot_style)
            for group in batch
        ], return_exceptions=True)
        failures = []
        for group, outcome in zip(batch, outputs):
            if isinstance(outcome, Exception):
                failures.append((group['sample'], group['step'], outcome))
            else:
                records[key(group)] = {'raw_text': outcome[0], 'trace': outcome[1]}
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_name(checkpoint_path.name + '.tmp')
        tmp_path.write_text(json.dumps({'meta': checkpoint_meta, 'records': records},
                                       ensure_ascii=False))
        tmp_path.replace(checkpoint_path)
        print(f'Checkpointed {len(records)}/{len(groups)} steps', flush=True)
        if failures:
            sample, step, error = failures[0]
            raise RuntimeError(f'{len(failures)} agent steps failed; '
                               f'first: {sample} step {step}: {error}') from error

    if set(records) != expected_keys:
        raise RuntimeError('Checkpoint is missing expected steps')
    raw_texts = [records[key(group)]['raw_text'] for group in groups]
    traces = [records[key(group)]['trace'] for group in groups]
    result = assemble_eval_output(
        groups, raw_texts, model=args.model, ablation='full',
        dataset_path=args.dataset_path, save_raw=True,
    )
    for record, trace in zip(iter_step_records(result), traces):
        result['samples'][record['sample']]['steps'][str(record['step'])]['agent_trace'] = trace
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_name(output_path.name + '.tmp')
    tmp_output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    tmp_output.replace(output_path)
    checkpoint_path.unlink()
    print(f'Wrote {len(groups)} steps, {sum(t["n_tool_calls"] for t in traces)} '
          f'tool calls → {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--plot-style', required=True,
                        choices=['scatter', 'contour', 'density'])
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--api-reasoning-effort', default='medium')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=32768)
    parser.add_argument('--max-turns', type=int, default=4)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--sample', nargs='+', default=None)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()
    if args.max_turns < 2:
        parser.error('--max-turns must be >= 2')
    if args.batch_size < 1:
        parser.error('--batch-size must be >= 1')
    load_dotenv()
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
