# GPT-6 API run record

The GPT-6 experiments reported in the manuscript were run from the project
checkout through LSF. The model IDs below are the IDs sent in API requests;
they are not names inferred from result folders. Dates are cluster-local
calendar dates (America/New_York) transcribed from the retained LSF output and
experiment records in the working project. Date ranges include completed
retries. Local job logs and per-sample API records are kept outside Git because
they contain paths, sample metadata, and response identifiers.

| Reported experiment | API model ID and reasoning effort | Execution dates in 2026 | Request settings and retained evidence |
| --- | --- | --- | --- |
| LLM-Gate full test and definition updates | `gpt-6-luna` (default, none, low, medium, high); `gpt-6-sol` (medium) | Oct 20–27 | Temperature 1.0, 32,768 maximum completion tokens. The original Luna full-test LSF job `49740394` finished Oct 21; the definition-update job `49857928` finished Oct 27. Results were merged and audited under `results/_definition_rerun_20261027/`. |
| Density-grid and FlowSOM C2S medium runs | `gpt-6-luna`, `gpt-6-sol` (medium) | Oct 27–28 | Temperature 1.0, 32,768 maximum completion tokens. LSF array `49893497` finished Oct 28; per-cohort outputs were validated before the combined C2S summary. |
| Visual VLM-Gate and Agent-Gate | `gpt-6-luna` (medium) | Oct 27–28 | Temperature 1.0. VLM-Gate used Chat Completions and a 32,768-token ceiling; Agent-Gate used Responses, at most four turns, and a 65,536-token ceiling after recovery. Full-run array `49892404` and Agent resume `49963397` supplied the completed comparison. |
| Visual scatter comparison | `gpt-6-sol` (medium) | Oct 28 | Temperature 1.0. VLM-Gate used Chat Completions at 32,768 tokens; Agent-Gate used Responses at 65,536 tokens. The final reported scope is scatter input; Agent scatter completed in array `50041564` and summary job `50041624`. |
| Hard depletion and shift | `gpt-6-luna` (medium) | Oct 29 | Temperature 1.0, 32,768 maximum completion tokens. Completed evaluation matched 1,677 depletion and 1,883 shift predictions; the zero-magnitude baseline reused the full-test run. |
| Flat Cascade and Flat C2S | `gpt-6-luna` (medium) | Oct 29–Nov 1 | Temperature 1.0, 32,768 maximum completion tokens. Both methods covered 17 cohort/plan settings and 646 test sample/plan pairs. The Nov 2 audit and summary validated the completed predictions. |
| Flat Cascade and Flat C2S | `gpt-6-sol` (medium) | Nov 3–4 | Temperature 1.0, 32,768 maximum completion tokens. Inference arrays `50073444` and `50073446`, with documented retries, covered the same 17 settings and 646 pairs. The Nov 4 summary validated all case summaries. |

The request model IDs and generation defaults are in
`src/llm_gate/configs/models/gpt_6_{luna,sol}.yaml` and
`src/llm/configs/models/gpt_6_{luna,sol}.yaml`. The flat runner records UTC
request starts, API response IDs, token usage, model ID, and reasoning effort
in local `inference_calls.jsonl` files. The API model IDs are hosted aliases;
the provider may change their backing implementation, so these records do not
guarantee identical future responses.
