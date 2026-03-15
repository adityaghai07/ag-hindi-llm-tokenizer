"""
Hindi-aware pre-tokenization using grapheme cluster-based regex.

Key design decisions:
- Devanagari grapheme clusters (consonant + virama + consonant etc.) are ATOMIC
- Matras, anusvara, chandrabindu, nukta stay attached to their base consonant
- English words handled for code-mixed text
- Both Devanagari and Arabic numerals supported
"""

import regex
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Core grapheme cluster pattern for Devanagari
# A "syllable unit" = base consonant/vowel + all combining marks
# Combining marks: matras (U+093E–U+094C), virama (U+094D),
#                  anusvara (U+0902), visarga (U+0903),
#                  chandrabindu (U+0901), nukta (U+093C),
#                  vedic extensions (U+0900), etc.
# ---------------------------------------------------------------------------

# Devanagari syllable: one or more base chars + all combining marks
# Using \X (extended grapheme cluster) from the `regex` library
GRAPHEME_CLUSTER = r'\X'

# Raw pattern string — needed by tokenizers.Regex (HuggingFace tokenizers library)
HINDI_PRETOK_PATTERN_STR = (
    r"[\u0964\u0965\u0970]"
    r"|[\u0900-\u0963\u0966-\u096F\u0971-\u097F]"
    r"[\u0900-\u0903\u093C\u093E-\u094D\u0951-\u0954\u0962\u0963]*"
    r"(?:[\u0900-\u0963\u0966-\u096F\u0971-\u097F]"
    r"[\u0900-\u0903\u093C\u093E-\u094D\u0951-\u0954\u0962\u0963]*)*"
    r"|[a-zA-Z]+(?:'[a-zA-Z]+)*"
    r"|[\u0966-\u096F]{1,3}"
    r"|[0-9]{1,3}"
    r"|\s+"
    r"|[^\s\u0900-\u097Fa-zA-Z0-9]"
)

HINDI_PRETOK_PATTERN = regex.compile(
    r"""
    # Devanagari punctuation FIRST (before word pattern, since danda U+0964/U+0965
    # falls inside the Devanagari block and would otherwise be consumed by the word rule)
    [\u0964\u0965\u0970]
    |
    # Devanagari word: sequence of Devanagari base chars + combining marks only.
    # Excludes: danda (U+0964), double danda (U+0965), abbreviation sign (U+0970),
    #           and Devanagari digits (U+0966-U+096F) which have their own rule.
    # Combining marks kept attached:
    #   U+0900-U+0903  : chandrabindu, anusvara, visarga
    #   U+093C         : nukta
    #   U+093E-U+094C  : matras (vowel signs)
    #   U+094D         : virama (halant) — joins consonants into conjuncts
    #   U+0951-U+0954  : vedic tone marks
    #   U+0962-U+0963  : vowel signs for vocalic r/l
    [\u0900-\u0963\u0966-\u096F\u0971-\u097F]
    [\u0900-\u0903\u093C\u093E-\u094D\u0951-\u0954\u0962\u0963]*
    (?:[\u0900-\u0963\u0966-\u096F\u0971-\u097F]
    [\u0900-\u0903\u093C\u093E-\u094D\u0951-\u0954\u0962\u0963]*)*
    |
    # English word (code-mixed), with optional apostrophe contractions
    [a-zA-Z]+(?:'[a-zA-Z]+)*
    |
    # Devanagari digits (1–3 digit group)
    [\u0966-\u096F]{1,3}
    |
    # Arabic digits (1–3 digit group)
    [0-9]{1,3}
    |
    # Whitespace
    \s+
    |
    # Other punctuation / symbols (one at a time)
    [^\s\u0900-\u097Fa-zA-Z0-9]
    """,
    regex.VERBOSE | regex.UNICODE
)


def pretokenize(text: str) -> List[str]:
    """Split text into pre-tokens using the Hindi-aware pattern."""
    return [m.group() for m in HINDI_PRETOK_PATTERN.finditer(text)]


def pretokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Return (token, start, end) tuples for debugging."""
    return [(m.group(), m.start(), m.end()) for m in HINDI_PRETOK_PATTERN.finditer(text)]


def grapheme_clusters(text: str) -> List[str]:
    """
    Split a string into Unicode extended grapheme clusters.
    This is the minimum atomic unit — never split below this.
    """
    return regex.findall(GRAPHEME_CLUSTER, text)


def validate_no_virama_split(tokens: List[str]) -> List[str]:
    """
    Sanity check: ensure no token ends with a bare virama (U+094D).
    A trailing virama means a conjunct was split — that's a bug.
    Returns list of offending tokens (empty = all good).
    """
    VIRAMA = '\u094D'
    return [t for t in tokens if t.endswith(VIRAMA)]


if __name__ == "__main__":
    test_cases = [
        "भारत एक महान देश है।",
        "विद्यार्थी कार्यक्रम में गए।",
        "मैं office जा रहा हूँ",
        "ज़िंदगी फ़ैसला क़िस्मत",
        "२०२५ और 2025 दोनों सही हैं।",
        "क्ष त्र ज्ञ श्र",  # conjuncts
        "हिंदी भाषा",       # anusvara
        "हँसना",             # chandrabindu
    ]

    for text in test_cases:
        tokens = pretokenize(text)
        bad = validate_no_virama_split(tokens)
        status = "✓" if not bad else f"✗ BAD: {bad}"
        print(f"{status} | {text!r}")
        print(f"    → {tokens}\n")
