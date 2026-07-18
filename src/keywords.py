"""Keyword normalization.
"""
import re

from .llmii_utils import de_pluralize, AND_EXCEPTIONS

_NON_ASCII = re.compile(r"[^\x00-\x7F]")
_NON_WORD = re.compile(r"[^\w\s-]")
_UNDERSCORES = re.compile(r"_")
_MULTI_SPACE = re.compile(r"\s+")
_MULTI_HYPHEN = re.compile(r"-+")
_HYPHEN_TOKEN = re.compile(r"^\w+-\w+$")
_LEADING_DIGITS = re.compile(r"^\d{3,}")

_SHORT_WORD_ALLOWED = frozenset(("x", "u"))
_CONJUNCTIONS = frozenset(("and", "or"))


class _DefaultRules:
    """Fallback rules when no config is supplied."""
    normalize_keywords = True
    depluralize_keywords = False
    limit_word_count = True
    max_words_per_keyword = 2
    split_and_entries = True
    ban_prompt_words = True
    no_digits_start = True
    min_word_length = True
    latin_only = True


_DEFAULT_RULES = _DefaultRules()


def split_on_internal_capital(word):
    """Split a word on a capital letter after the 4th position.
    BlueSky -> Blue Sky, microService -> micro Service
    """
    if len(word) <= 4:
        return word
    for i in range(4, len(word)):
        if word[i].isupper():
            return word[:i] + " " + word[i:]
    return word


def normalize_keyword(keyword, banned_words, config=None):
    """Normalize a keyword per config rules.

    Returns:
        str  - normalized keyword
        list - two keywords, when an 'and'/'or' entry is split
        None - keyword rejected
    """
    if config is None:
        config = _DEFAULT_RULES

    if not isinstance(keyword, str):
        keyword = str(keyword)

    if not config.normalize_keywords:
        return keyword.strip()

    # Handle internal capitalization before lowercasing
    split_words = []
    for word in keyword.strip().split():
        split_words.extend(split_on_internal_capital(word).split())
    keyword = " ".join(split_words).lower().strip()

    if config.latin_only:
        keyword = _NON_ASCII.sub("", keyword)

    # Remove non-alphanumeric, fix multiple spaces and hyphens
    keyword = _NON_WORD.sub("", keyword)
    keyword = _UNDERSCORES.sub(" ", keyword)
    keyword = _MULTI_SPACE.sub(" ", keyword).strip()
    keyword = _MULTI_HYPHEN.sub("-", keyword)

    if not keyword:
        return None

    if config.ban_prompt_words:
        for word in keyword.strip().split():
            if word in banned_words:
                return None

    # Account for hyphenated words by splitting and recombining later
    tokens = keyword.split()
    words = []
    is_hyphenated = False

    for token in tokens:
        if "-" in token:
            if not _HYPHEN_TOKEN.match(token):
                return None
            is_hyphenated = True
            words.extend(token.split("-"))
        else:
            words.append(token)

    if config.limit_word_count:
        max_words = config.max_words_per_keyword
        middle_conjunction = len(words) == 3 and words[1] in _CONJUNCTIONS
        limit = max_words + 1 if middle_conjunction else max_words
        if len(words) > limit:
            return None

    if config.no_digits_start and words and _LEADING_DIGITS.match(words[0]):
        return None

    if config.min_word_length:
        for word in words:
            if len(word) < 2 and word not in _SHORT_WORD_ALLOWED:
                return None

    # Split "X and Y" / "X or Y" into two keywords
    if (config.split_and_entries
            and len(words) == 3
            and words[1] in _CONJUNCTIONS
            and not is_hyphenated
            and " ".join(words) not in AND_EXCEPTIONS):
        if config.depluralize_keywords:
            return [de_pluralize(words[0]), de_pluralize(words[2])]
        return [words[0], words[2]]

    if config.depluralize_keywords:
        tokens[-1] = de_pluralize(tokens[-1])

    return " ".join(tokens)
