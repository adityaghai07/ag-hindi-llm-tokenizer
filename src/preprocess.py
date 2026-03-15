"""
Corpus cleaning pipeline for Hindi tokenizer training.
Handles NFC normalization, noise removal, deduplication, and filtering.
"""

import unicodedata
import regex
import os
from pathlib import Path

# Devanagari Unicode block: U+0900–U+097F
DEVANAGARI_RE = regex.compile(r'[\u0900-\u097F]')

# Allowed characters: Devanagari, basic Latin, digits, common punctuation, whitespace
ALLOWED_RE = regex.compile(
    r'[^\u0900-\u097F'          # Devanagari block
    r'a-zA-Z0-9'                # Basic Latin + digits
    r'\u0966-\u096F'            # Devanagari digits
    r'\s'                       # Whitespace
    r'।॥॰'                     # Devanagari punctuation
    r'.,!?;:\-\(\)\[\]\"\''    # Western punctuation
    r']'
)


def nfc_normalize(text: str) -> str:
    """Apply Unicode NFC normalization — critical for Devanagari consistency."""
    return unicodedata.normalize("NFC", text)


def remove_noise(text: str) -> str:
    """Strip characters outside the allowed set."""
    return ALLOWED_RE.sub(' ', text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single space."""
    text = regex.sub(r'\s+', ' ', text)
    return text.strip()


def hindi_ratio(text: str) -> float:
    """Fraction of characters that are Devanagari."""
    if not text:
        return 0.0
    devanagari_chars = len(DEVANAGARI_RE.findall(text))
    return devanagari_chars / len(text)


def clean_line(line: str) -> str | None:
    """
    Full cleaning pipeline for a single line.
    Returns None if the line should be discarded.
    """
    line = nfc_normalize(line)
    line = remove_noise(line)
    line = normalize_whitespace(line)

    # Filter: too short
    if len(line) < 50:
        return None

    # Filter: less than 30% Hindi characters
    if hindi_ratio(line) < 0.30:
        return None

    return line


def process_file(input_path: str, output_path: str) -> dict:
    """
    Clean a raw text file and write the result.
    Returns stats dict.
    """
    seen = set()
    stats = {"total": 0, "kept": 0, "duplicates": 0, "filtered": 0}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8", errors="ignore") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            stats["total"] += 1
            cleaned = clean_line(line)

            if cleaned is None:
                stats["filtered"] += 1
                continue

            # Document-level deduplication (line-level for simplicity)
            if cleaned in seen:
                stats["duplicates"] += 1
                continue

            seen.add(cleaned)
            fout.write(cleaned + "\n")
            stats["kept"] += 1

    return stats


def process_directory(input_dir: str, output_dir: str) -> None:
    """Process all .txt files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    for txt_file in input_path.glob("*.txt"):
        out_file = output_path / txt_file.name
        print(f"Processing {txt_file.name}...")
        stats = process_file(str(txt_file), str(out_file))
        print(f"  total={stats['total']}, kept={stats['kept']}, "
              f"filtered={stats['filtered']}, duplicates={stats['duplicates']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        stats = process_file(sys.argv[1], sys.argv[2])
        print(stats)
    else:
        print("Usage: python preprocess.py <input.txt> <output.txt>")
