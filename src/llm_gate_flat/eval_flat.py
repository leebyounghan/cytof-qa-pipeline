"""Compatibility shim: flat-annotation eval moved to ``src.flat_eval``.

The implementation now lives at ``src/flat_eval.py`` so it can be shared
with ``src/flowsom_c2s/``. This module re-exports the public API so legacy
imports continue to work, and ``python -m src.llm_gate_flat.eval_flat``
remains a valid CLI.
"""

from src.flat_eval import (  # noqa: F401
    DISCARD_LABEL,
    derive_leaf_set,
    evaluate_sample,
    main,
)


if __name__ == "__main__":
    main()
