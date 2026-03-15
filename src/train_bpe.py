"""
Train a character-level BPE tokenizer on Hindi corpus.
Outputs a HuggingFace-compatible tokenizer.json.
"""

import os
import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders, Regex
from tokenizers.pre_tokenizers import Split, Sequence as PreTokenizerSequence

# Import our Hindi pattern
import sys
sys.path.insert(0, str(Path(__file__).parent))
from pretokenize import HINDI_PRETOK_PATTERN, HINDI_PRETOK_PATTERN_STR

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|sep|>",
    "<|mask|>",
    "<|user|>",
    "<|assistant|>",
]


def build_tokenizer() -> Tokenizer:
    """Construct the tokenizer with normalizer and pre-tokenizer."""
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # NFC normalization — critical for Devanagari
    tokenizer.normalizer = normalizers.NFC()

    # Hindi-aware pre-tokenizer using Metaspace (like SentencePiece)
    # Replaces spaces with ▁ and attaches them to words
    # This prevents spaces from being separate tokens
    from tokenizers.pre_tokenizers import Metaspace
    tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")

    # Add Metaspace decoder to properly convert ▁ back to spaces
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")

    return tokenizer


def train(
    corpus_files: list[str],
    output_dir: str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train BPE tokenizer and save to output_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        # No initial_alphabet — we want character-level, not byte-level
    )

    print(f"Training BPE tokenizer (vocab_size={vocab_size}, min_freq={min_frequency})")
    print(f"Corpus files: {corpus_files}")

    tokenizer.train(files=corpus_files, trainer=trainer)

    out_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(out_path)
    print(f"Saved tokenizer to {out_path}")

    return tokenizer


def wrap_for_huggingface(tokenizer_json_path: str, output_dir: str) -> None:
    """Wrap the trained tokenizer.json as a HuggingFace PreTrainedTokenizerFast."""
    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json_path,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        sep_token="<|sep|>",
        mask_token="<|mask|>",
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    hf_tokenizer.save_pretrained(output_dir)
    print(f"HuggingFace tokenizer saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hindi BPE tokenizer")
    parser.add_argument("--corpus", nargs="+", required=True, help="Path(s) to clean corpus .txt files")
    parser.add_argument("--output", default="models/bpe_32k", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--hf-output", default="models/bpe_32k_hf", help="HuggingFace output dir")
    args = parser.parse_args()

    tokenizer = train(args.corpus, args.output, args.vocab_size, args.min_freq)
    wrap_for_huggingface(
        os.path.join(args.output, "tokenizer.json"),
        args.hf_output
    )
