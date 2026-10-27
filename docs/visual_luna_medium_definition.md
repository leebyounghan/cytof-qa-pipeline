# Visual gate experiment: definition only

Six conditions: VLM-Gate and Agent-Gate, each with scatter, contour, and
density images. All use `gpt-6-luna`, API reasoning effort `medium`, model
temperature 1.0, the benchmark test split, and `definition` from
`benchmark/notes.yaml`. A snapshot is in `notes.yaml.used`.

The same `cells.md` text and parent-cell coordinate extent are reused for
the three image styles at each step. The scatter is the established
60,000-point rendering. Density is a log-scaled two-dimensional histogram;
contour is a Gaussian-smoothed version of the same histogram. The bin count
is `clip(1.5 * sqrt(min(n_cells, 60000)), 65, 180)`, contour smoothing
sigma is 1.8 bins, and images are 640 by 640 pixels before the shared
axis/tick/grid wrapper is applied. Background images contain no labels or
ground truth. Their rendered model-visible examples are in `preview/`.

The VLM runner makes one Chat Completions request per step. Agent-Gate uses
the Responses API to retain function tools with medium reasoning. Turn 1
requires `render_gate_overlay` with a proposed gate; later turns may call it
again or finish, with a maximum of four turns. Tool output contains the
same plot style with gate outlines and the existing coordinate, overlap,
and range diagnostics. The Agent trace stores turn count, tool-call count,
tool proposals, and diagnostic text.

The plot name in the system prompt and tool description is adjusted to match
the image style. All other prompt text is held fixed within each method.
The VLM and Agent methods retain their respective original system prompts.

LSF jobs: input generation `49879565`; six-condition inference `49883257`
(depends on input generation); propagation/evaluation `49883288` (depends
on inference). A queued or running job is not a completed result.
