"""Prototype phrases for the similarity channel, from the default target profile
(targets/a2179_screen.json, section prototypes). Hand-written, kept separate from
eval/labeled.jsonl (no eval title is copied there)."""
from .target import default_target

POSITIVE = list(default_target().positive)
NEGATIVE = list(default_target().negative)
