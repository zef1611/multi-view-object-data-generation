"""Linear-time JSON-array extraction with truncation salvage.

Extracted from `gemini_labeler.py` so all VLM clients (Gemini, Qwen3-VL,
filter, verifier) can share a single parser.

Replaces an earlier regex pattern that suffered catastrophic backtracking
(observed >5 min CPU on a 600-char malformed input). Also salvages
truncated arrays from small-model repetition loops by truncating to the
last complete `{...}` at array depth 1 and synthesizing `]`.
"""

from __future__ import annotations

from typing import Optional


def find_json_array(text: str) -> Optional[str]:
    """Find the first JSON array in `text` using bracket-counting (linear).

    Two failure modes worth handling:

      1. Catastrophic regex backtracking — the previous regex-based
         implementation hung for >5 min on malformed responses. The
         scanner here is O(n) so unbounded input is safe.

      2. Repetition loops at temperature=0 — small models (e.g. 8B)
         sometimes generate well-formed `{...}, {...}, ...` entries but
         loop forever and run out the token budget before emitting `]`.
         When the outer `[` never closes, we **salvage** by truncating to
         the last complete `{...}` at array level and synthesizing `]`.
         Better to keep 16 valid labels than fail the whole frame.
    """
    start = text.find("[")
    if start < 0:
        return None
    bracket_depth = 0
    brace_depth = 0
    in_string = False
    escape = False
    last_obj_end_at_array_level = -1
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c == "[":
            bracket_depth += 1
        elif c == "]":
            bracket_depth -= 1
            if bracket_depth == 0:
                return text[start:i + 1]
        elif c == "{":
            brace_depth += 1
        elif c == "}":
            brace_depth -= 1
            if bracket_depth == 1 and brace_depth == 0:
                last_obj_end_at_array_level = i
    if last_obj_end_at_array_level > start:
        return text[start:last_obj_end_at_array_level + 1] + "]"
    return None


# Back-compat alias — older callers may still import the underscore name.
_find_json_array = find_json_array
