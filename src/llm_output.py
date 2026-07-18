"""Parsing of raw LLM output into usable dicts/strings.

LLMs return garbage in many shapes. These functions funnel all of it into
either a clean string (captions) or a dict (keywords). The repair ladder in
clean_json/clean_tags is ordered cheapest-first so well-behaved responses
(direct JSON, especially with grammar enabled) never pay for the fallbacks.
"""
import json
import re

from json_repair import repair_json as rj

from .llmii_utils import first_json

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_TAGS = re.compile(r"</?think>")
_NEWLINES = re.compile(r"\n")
_SMART_QUOTES = re.compile(r'["""]')
_DOUBLE_BACKSLASH = re.compile(r"\\{2}")
_MD_JSON_BLOCK = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
_MD_LIST_ITEM = re.compile(r"(?:^\s*[-*+]|\d+\.)\s*(.+)$", re.MULTILINE)
_KEYWORDS_ARRAY = re.compile(r'"Keywords"\s*:\s*\[(.*?)\]', re.DOTALL)


def clean_string(data):
    """Make a string safe for addition to metadata."""
    if isinstance(data, dict):
        data = json.dumps(data)

    if not isinstance(data, str):
        return ""

    # Strip <think> content: matched pairs first, then orphaned tags
    data = _THINK_BLOCK.sub("", data)
    data = _THINK_TAGS.sub("", data)

    # Normalize
    data = _NEWLINES.sub("", data)
    data = _SMART_QUOTES.sub('"', data)
    data = _DOUBLE_BACKSLASH.sub("", data)

    # Truncate at the last complete sentence
    last_period = data.rfind(".")
    if last_period != -1:
        data = data[:last_period + 1]

    return data


def markdown_list_to_dict(text):
    """Convert a markdown list found in text to {"Keywords": [...]}, else None."""
    items = _MD_LIST_ITEM.findall(text)
    return {"Keywords": items} if items else None


def _unwrap(result):
    """[{...}] -> {...}; everything else passes through."""
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return result[0]
    return result


def _try_loads(text):
    """json.loads that returns None instead of raising."""
    try:
        return _unwrap(json.loads(text))
    except (ValueError, TypeError):
        return None


def clean_json(data):
    """Pull anything dict-like out of arbitrary LLM output.

    Handles direct dicts, list-wrapped dicts, markdown-fenced JSON, and
    malformed JSON via progressively more aggressive repair.
    """
    if data is None:
        return None

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        return _unwrap(data) if data else None

    if not isinstance(data, str):
        return None

    # 1. Direct parse (fast path; always works with JSON grammar)
    result = _try_loads(data)
    if result is not None:
        return result

    # 2. Markdown-fenced JSON
    match = _MD_JSON_BLOCK.search(data)
    if match:
        result = _try_loads(match.group(1).strip())
        if result is not None:
            return result

    # 3. repair_json
    try:
        result = _try_loads(rj(data))
        if result is not None:
            return result
    except Exception:
        pass

    # 4. first_json + repair_json
    try:
        result = _try_loads(rj(first_json(data)))
        if result is not None:
            return result
    except Exception:
        pass

    # 5. Nuclear option: wrap in braces and repair
    try:
        result = _try_loads(first_json(rj("{" + data + "}")))
        if isinstance(result, dict) and result.get("Keywords"):
            return result
    except Exception:
        pass

    # 6. Strangelove option: markdown list
    try:
        return markdown_list_to_dict(data)
    except Exception:
        return None


def clean_tags(data):
    """Extract and combine every Keywords entry found in LLM output.

    When the EOS token is banned the model may emit several JSON objects,
    each with its own Keywords array; collect them all.

    Returns {"Keywords": [...]} or None.
    """
    if data is None:
        return None

    if isinstance(data, dict):
        keywords = data.get("Keywords") or []
        return {"Keywords": list(keywords)} if keywords else None

    if isinstance(data, list):
        all_keywords = []
        for item in data:
            if isinstance(item, dict):
                all_keywords.extend(item.get("Keywords") or [])
        return {"Keywords": all_keywords} if all_keywords else None

    if not isinstance(data, str):
        return None

    # Single JSON document
    result = _try_loads(data)
    if result is not None:
        return clean_tags(result)

    # Markdown-fenced JSON
    match = _MD_JSON_BLOCK.search(data)
    if match:
        result = _try_loads(match.group(1))
        if result is not None:
            return clean_tags(result)

    # Multiple {"Keywords": [...]} fragments in one string
    all_keywords = []
    for fragment in _KEYWORDS_ARRAY.findall(data):
        array_str = "[" + fragment + "]"
        keywords = _try_loads(array_str)
        if keywords is None:
            try:
                keywords = _try_loads(rj(array_str))
            except Exception:
                keywords = None
        if keywords:
            all_keywords.extend(keywords)
    if all_keywords:
        return {"Keywords": all_keywords}

    # Last resort: repair the whole string
    try:
        result = _try_loads(rj(data))
        if result is not None:
            return clean_tags(result)
    except Exception:
        pass

    return None
