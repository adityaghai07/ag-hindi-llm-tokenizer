"""
Download and prepare Hindi training corpus from HuggingFace datasets.
Uses Hindi Wikipedia (clean, high quality) as the primary source.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm


def download_hindi_wikipedia(output_path: str, max_docs: int = None) -> int:
    """
    Download Hindi Wikipedia via HuggingFace datasets.
    Returns number of documents written.
    """
    from datasets import load_dataset

    print("Loading Hindi Wikipedia dataset...")
    # New HuggingFace datasets format — no loading scripts, uses Parquet directly
    dataset = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(tqdm(dataset, desc="Writing Wikipedia")):
            if max_docs and i >= max_docs:
                break
            text = doc.get("text", "").strip()
            if text:
                f.write(text + "\n")
                count += 1

    print(f"Written {count} documents to {output_path}")
    return count


def download_oscar_hindi(output_path: str, max_docs: int = 500_000) -> int:
    """
    Download OSCAR Hindi split (web-crawled, noisier than Wikipedia).
    Requires HuggingFace login for some versions.
    """
    from datasets import load_dataset

    print("Loading OSCAR Hindi dataset (this may take a while)...")
    try:
        # Try OSCAR 2301 first
        dataset = load_dataset("oscar-corpus/OSCAR-2301", "hi", split="train",
                               streaming=True)
    except Exception:
        # Fallback to OSCAR 2109
        dataset = load_dataset("oscar-corpus/OSCAR-2109", "unshuffled_deduplicated_hi",
                               split="train", streaming=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(tqdm(dataset, desc="Writing OSCAR", total=max_docs)):
            if i >= max_docs:
                break
            text = doc.get("text", "").strip()
            if text:
                f.write(text + "\n")
                count += 1

    print(f"Written {count} documents to {output_path}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Hindi corpus")
    parser.add_argument("--source", choices=["wikipedia", "oscar", "both"], default="wikipedia")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()

    if args.source in ("wikipedia", "both"):
        download_hindi_wikipedia(
            os.path.join(args.output_dir, "hindi_wikipedia.txt"),
            max_docs=args.max_docs
        )

    if args.source in ("oscar", "both"):
        download_oscar_hindi(
            os.path.join(args.output_dir, "hindi_oscar.txt"),
            max_docs=args.max_docs or 500_000
        )
