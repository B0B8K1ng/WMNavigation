"""
Cascading response parser for VLM outputs.

Handles both standard JSON and Qwen's single-quote Python dict format.
"""

import ast
import json
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ParseResult:
    success: bool
    data: Any = None
    method: Optional[str] = None
    error: Optional[str] = None
    raw: str = ""


def _try_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _try_ast(text: str) -> Optional[dict]:
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    return None


def _try_regex(text: str) -> Optional[dict]:
    """Extract key-value pairs via regex as a last resort."""
    pattern = r"""['"]?(\w+)['"]?\s*:\s*(['"]?)(.*?)\2(?=[,}]|$)"""
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    result = {}
    for key, _, value in matches:
        value = value.strip().strip("'\"")
        result[key] = value
    return result if result else None


def _extract_bracketed(text: str) -> str:
    """Pull out the first {...} block from a larger string."""
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def parse_vlm_response(raw: str) -> ParseResult:
    """
    Parse a VLM response string using a cascading strategy:
      1. Direct JSON
      2. ast.literal_eval (handles single-quote Python dicts from Qwen)
      3. Regex extraction

    Returns a ParseResult with success flag, parsed data, and method used.
    """
    if not raw or not isinstance(raw, str):
        return ParseResult(success=False, error="Empty or non-string input", raw=str(raw))

    candidate = _extract_bracketed(raw)

    # Stage 1: JSON
    data = _try_json(candidate)
    if data is not None:
        return ParseResult(success=True, data=data, method="json", raw=raw)

    # Stage 2: ast.literal_eval
    data = _try_ast(candidate)
    if data is not None:
        return ParseResult(success=True, data=data, method="ast", raw=raw)

    # Stage 3: regex
    data = _try_regex(candidate)
    if data is not None:
        logging.warning("Fell back to regex parsing for VLM response.")
        return ParseResult(success=True, data=data, method="regex", raw=raw)

    return ParseResult(
        success=False,
        error="All parsing strategies failed",
        raw=raw,
    )
