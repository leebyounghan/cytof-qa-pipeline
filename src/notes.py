"""Per-(cohort, step) prompt notes — the single source of [Description] text.

``benchmark/notes.yaml`` holds, per cohort and step, a structured entry with
``definition`` (category meanings), ``rule`` (region mapping for rule-based
systems) and ``details`` (curated context). Every prompt builder reads its
note from here; task.json / gating_plan.json carry no note text.

Note modes:
  definition          — ``definition`` only
  definition_details  — ``definition`` followed by ``details``

A missing cohort, step or field is an error.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

NOTES_PATH = Path(__file__).resolve().parents[1] / 'benchmark' / 'notes.yaml'

NOTE_MODES = ('definition', 'definition_details')
DEFAULT_NOTE_MODE = 'definition'


@lru_cache(maxsize=None)
def load_notes(path: Path = NOTES_PATH) -> dict:
    with open(path) as f:
        notes = yaml.safe_load(f)
    if not isinstance(notes, dict) or not notes:
        raise ValueError(f'{path}: expected a non-empty cohort → step mapping')
    return notes


def _field(entry: dict, field: str, where: str) -> str:
    text = entry.get(field)
    if not isinstance(text, str) or not text.strip():
        raise KeyError(f'{where}: missing non-empty {field!r}')
    return text.strip()


def get_note(
    cohort: str, step: int | str, mode: str = DEFAULT_NOTE_MODE,
    path: Path = NOTES_PATH,
) -> str:
    """Return the note text for ``(cohort, step)`` under ``mode``."""
    if mode not in NOTE_MODES:
        raise ValueError(f'Unknown note mode {mode!r}; choose from {NOTE_MODES}')
    if not cohort:
        raise ValueError(f'cohort is required to look up a note (got {cohort!r})')
    notes = load_notes(Path(path))
    if cohort not in notes:
        raise KeyError(f'{path}: no entry for cohort {cohort!r}')
    step_int = int(step)
    entry = (notes[cohort] or {}).get(step_int)
    where = f'{path}: ({cohort}, step {step_int})'
    if not isinstance(entry, dict):
        raise KeyError(f'{where}: expected a definition/rule/details mapping, '
                       f'got {type(entry).__name__}')
    parts = [_field(entry, 'definition', where)]
    if mode == 'definition_details':
        parts.append(_field(entry, 'details', where))
    return '\n'.join(parts)


def add_note_mode_arg(parser) -> None:
    parser.add_argument('--note-mode', default=DEFAULT_NOTE_MODE,
                        choices=NOTE_MODES,
                        help='Which benchmark/notes.yaml fields form the '
                             '[Description] note. Default: definition.')
