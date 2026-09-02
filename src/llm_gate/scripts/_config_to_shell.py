#!/usr/bin/env python3
"""YAML model config → shell variable assignments.

Reads a per-model YAML and prints a block of shell statements that the
unified runner script (run_vllm_gate.sh) eval's to populate:

    MODEL                — vllm serve positional
    SERVED_NAME          — vllm --served-model-name (also used in output paths)
    VLLM_ARGS  (array)   — flags appended to `vllm serve "$MODEL"`
    OPENAI_ARGS (array)  — flags appended to `python -m ...run_openai`
    export KEY=VALUE     — for any `server_env:` entries

Null / missing values become "don't pass the flag" — the runner relies on
this so e.g. an unset `temperature` falls through to run_openai's own
default instead of being passed as the literal string "None".
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import yaml


# (yaml_key, cli_flag, kind)
# kind = "value"  → emit [flag, str(val)] when val is not None
# kind = "switch" → emit [flag] when val is truthy; omit otherwise
SERVER_FLAGS = [
    ("data_parallel_size",     "--data-parallel-size",     "value"),
    ("tensor_parallel_size",   "--tensor-parallel-size",   "value"),
    ("max_model_len",          "--max-model-len",          "value"),
    ("gpu_memory_utilization", "--gpu-memory-utilization", "value"),
    ("enable_prefix_caching",  "--enable-prefix-caching",  "switch"),
    ("trust_remote_code",      "--trust-remote-code",      "switch"),
    ("enable_expert_parallel", "--enable-expert-parallel", "switch"),
    ("enable_auto_tool_choice","--enable-auto-tool-choice","switch"),
    ("reasoning_parser",       "--reasoning-parser",       "value"),
    ("reasoning_config",       "--reasoning-config",       "value"),
    ("tool_call_parser",       "--tool-call-parser",       "value"),
    ("kv_cache_dtype",         "--kv-cache-dtype",         "value"),
    ("block_size",             "--block-size",             "value"),
    ("tokenizer_mode",         "--tokenizer-mode",         "value"),
    ("mm_encoder_tp_mode",     "--mm-encoder-tp-mode",     "value"),
    ("compilation_config",     "--compilation-config",     "value"),
]

# run_openai flag names mix snake_case and kebab-case — match what the
# existing scripts already pass to that script.
GENERATION_FLAGS = [
    ("max_tokens",         "--max_tokens",         "value"),
    ("temperature",        "--temperature",        "value"),
    ("top_p",              "--top-p",              "value"),
    ("top_k",              "--top-k",              "value"),
    ("min_p",              "--min-p",              "value"),
    ("presence_penalty",   "--presence-penalty",   "value"),
    ("frequency_penalty",  "--frequency-penalty",  "value"),
    ("repetition_penalty", "--repetition-penalty", "value"),
    ("thinking",              "--thinking",              "value"),
    ("reasoning_effort",      "--reasoning-effort",      "value"),
    ("thinking_token_budget", "--thinking-token-budget", "value"),
    ("strip_think",           "--strip-think",           "switch"),
]


def _stringify(val):
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (dict, list)):
        return json.dumps(val, separators=(",", ":"))
    return str(val)


def collect_args(section: dict, flags) -> list[str]:
    out: list[str] = []
    for key, flag, kind in flags:
        val = section.get(key)
        if val is None:
            continue
        if kind == "switch":
            if val:
                out.append(flag)
        else:
            out.extend([flag, _stringify(val)])
    return out


def emit_array(name: str, items: list[str]) -> str:
    quoted = " ".join(shlex.quote(s) for s in items)
    # `-g` keeps the array global even when this string is eval'd inside a
    # function — without it bash treats `declare` as `local`, which silently
    # drops args before they reach the caller.
    return f"declare -ga {name}=({quoted})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    try:
        cfg = yaml.safe_load(args.config.read_text()) or {}
    except yaml.YAMLError as e:
        print(f"[config_to_shell] {args.config}: YAML parse error: {e}",
              file=sys.stderr)
        return 2

    model = cfg.get("model") or {}
    server = cfg.get("server") or {}
    server_env = cfg.get("server_env") or {}
    generation = cfg.get("generation") or {}

    hf_id = model.get("hf_id")
    openai_id = model.get("openai_id")
    if hf_id and openai_id:
        print(f"[config_to_shell] {args.config}: model.hf_id and model.openai_id "
              "are mutually exclusive", file=sys.stderr)
        return 2
    if not hf_id and not openai_id:
        print(f"[config_to_shell] {args.config}: model.hf_id (vllm-served) "
              "or model.openai_id (OpenAI API) is required", file=sys.stderr)
        return 2

    openai_args = collect_args(generation, GENERATION_FLAGS)

    if openai_id:
        # OpenAI API mode — no vllm serve, runner skips Phase 1.
        served = model.get("served_name") or str(openai_id)
        print(f"MODEL={shlex.quote(str(openai_id))}")
        print(f"SERVED_NAME={shlex.quote(served)}")
        print("OPENAI_MODE=1")
        print(emit_array("VLLM_ARGS", []))
        print(emit_array("OPENAI_ARGS", openai_args))
        # server_env is ignored in OpenAI mode (no vllm process).
        return 0

    served = model.get("served_name") or Path(str(hf_id)).name
    vllm_args = collect_args(server, SERVER_FLAGS)
    vllm_args += ["--served-model-name", served]

    print(f"MODEL={shlex.quote(str(hf_id))}")
    print(f"SERVED_NAME={shlex.quote(served)}")
    print(emit_array("VLLM_ARGS", vllm_args))
    print(emit_array("OPENAI_ARGS", openai_args))
    for k, v in server_env.items():
        print(f"export {k}={shlex.quote(_stringify(v))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
