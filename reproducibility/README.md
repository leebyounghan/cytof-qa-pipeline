# Reproduction reference environment

This directory provides two installation-tested environments based on the GPU environment captured on 6 November 2026: one for inference and analysis, and one for raw FCS conversion. Current versions are retained where compatible. This record does not assert that every historical result used these versions.

The separate cluster R environment was verified on 9 November 2026. Its R,
Bioconductor, and CRAN package versions and installation paths are recorded in
[r_flowdensity.md](r_flowdensity.md). This check confirms the installed R
packages and `flowDensity::deGate`; it does not constitute a clean R install or
an end-to-end flowDensity benchmark run.

## Software and hardware

| Component | Reference value |
|---|---|
| Platform | Linux x86_64 |
| Python | 3.12.3 |
| vLLM | 0.32.0 |
| PyTorch runtime | 2.13.0+cu130 |
| PyTorch CUDA build | 13.0 |
| Transformers | 5.19.0 |
| Tokenizers | 0.23.3 |
| OpenAI Python client | 3.35.1 |
| NumPy | 2.3.5 for inference; 1.26.4 for FCS conversion |
| pandas | 3.0.6 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.1 |
| PyArrow | 26.0.0 for inference; 21.0.0 for FCS conversion |
| Matplotlib | 3.11.2 |
| MiniSom | 2.3.6 |
| GPUs | 8 NVIDIA H200 |
| NVIDIA driver | 580.126.16 |

[inference.lock.txt](inference.lock.txt) and [preprocess.lock.txt](preprocess.lock.txt) pin the validated environments, including artifact hashes. [installation-validation.json](installation-validation.json) records the clean-install results and installed versions. [environment.json](environment.json) and [requirements.snapshot.txt](requirements.snapshot.txt) preserve the original 218-package inventory for provenance; do not use that original inventory as an installation lockfile. PyTorch distribution metadata reports `2.13.0`, while its runtime version file reports `2.13.0+cu130`; both values are retained.

The container digest identifies the base image. Packages installed in the separate Python environment are recorded independently, so the image alone does not reconstruct the complete environment.

## Install the validated environments

Validation used Python 3.12.3 on Linux x86_64 in a fresh `python:3.12.3-slim-bookworm` Kubernetes pod, with uv 0.11.14. Run these commands from the repository root:

```bash
python3.12 -m pip install uv==0.11.14
uv venv --python 3.12.3 .venv-inference
uv pip sync --python .venv-inference/bin/python --require-hashes reproducibility/inference.lock.txt
uv pip check --python .venv-inference/bin/python
.venv-inference/bin/python reproducibility/check_environment.py --profile inference
```

For raw FCS conversion, create a separate environment:

```bash
uv venv --python 3.12.3 .venv-preprocess
uv pip sync --python .venv-preprocess/bin/python --require-hashes reproducibility/preprocess.lock.txt
uv pip check --python .venv-preprocess/bin/python
.venv-preprocess/bin/python reproducibility/check_environment.py --profile preprocess
```

The inference environment includes the scientific Python stack, OpenAI client, and vLLM. It has 217 packages and uses NumPy 2.3.5, which satisfies the installed mistral-common and numba requirements. The seven-package conversion environment uses fcsparser 0.2.8, NumPy 1.26.4, and PyArrow 21.0.0. FCS conversion produces Parquet files for the inference environment.

The original inventory failed dependency resolution because fcsparser requires NumPy below 2, whereas the installed OpenCV used by vLLM requires NumPy 2 or newer. PyArrow 26 also failed its import check with NumPy 1.26.4, so only the conversion environment uses PyArrow 21.

Both hash-pinned environments installed successfully and passed `uv pip check`. Checks also passed for FCS parsing using a bundled fixture, Parquet roundtrips, KDE, KMeans, MiniSom, plot rendering, CPU tensor operations, and imports including vLLM, torchvision, and torchaudio. The public checkout's evaluator and LLM-Gate, VLM-Gate, and Agent-Gate command-line help checks passed in the inference environment.

The validation pod requested no GPUs. GPU model inference, hosted API calls,
R/Bioconductor/flowDensity, and complete benchmark reproduction were not tested
in that pod. The separate LSF R package check is recorded above.

## Revised Qwen visual experiments

Preserved server logs confirm vLLM 0.32.0 for the revised Qwen3.6-27B visual experiments. Their run manifest records execution dates from 1–4 November 2026 and generation seeds 42, 123, and 456. The reference settings are:

- Model: `Qwen/Qwen3.6-27B`.
- Generation: temperature 1.0, top-p 0.95, top-k 20, presence penalty 0, thinking-token budget 8096, and maximum completion budget 16384 tokens.
- Serving: data parallel size 2, tensor parallel size 2, and maximum model length 65536. These values describe each serving lane, rather than a single eight-GPU tensor-parallel process.
- Parsing: `qwen3` reasoning parser, `qwen3_coder` tool parser, and chat-template content format `string`.
- Agent-Gate: at most four turns, final-turn tool-call arguments accepted as the final gate, and context-overflow requests scored as empty answers.
- Runner options for these settings: `--thinking-token-budget 8096`, `--request-seed <seed>`, and, for `src/agent_gate/inference/run_agent.py`, `--max-turns 4 --final-tool-call-as-answer --context-overflow-as-empty`. The cascade runner `src/llm_gate_flat/runner.py` takes `--temperature 1.0 --top-p 0.95 --top-k 20 --thinking-token-budget 8096 --seed <seed>`.

The recorded combined MD5 of the Agent-Gate and VLM-Gate runners is `e88cf8e0755d5b401269298c7b09ca63`. The preserved working files reproduce this fingerprint. Individual SHA-256 hashes are included in `environment.json`.

## Code provenance

The preserved visual run manifest identifies source commit `6137debaf3006fefc3b4804c5824dd812e146dea` **with local changes** in five execution files. Those changes are now in the repository. Four files are byte-identical to the preserved files and match the SHA-256 values in `environment.json`: `src/agent_gate/inference/run_agent.py`, `src/vlm_gate/inference/run_openai.py`, `src/llm_gate/inference/run_openai.py`, and `src/llm/inference/run_openai.py`. `src/llm_gate_flat/runner.py` combines the preserved sampling options with the later GPT-6 latency and cost logging, so its hash differs; models without an entry in `pricing.json` are logged with a null cost.

This repository contains the code used to produce all experimental results
reported in the manuscript. The five execution-file changes that were local at
the time of the revised Qwen visual runs are now committed here. The historical
file hashes identify the exact files preserved from those runs; the later
latency and cost logging in `src/llm_gate_flat/runner.py` explains why its
current hash differs. The environment inventory records software versions and
is separate from the code provenance.

GPT-6 Luna/Sol API runtime records and execution dates remain unverified in this audit. Hosted model identifiers and inference settings should accompany their run records. Code and dependency pinning cannot guarantee identical future responses from a hosted API model.
