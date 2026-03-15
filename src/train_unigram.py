"""
Train a Unigram (SentencePiece) tokenizer on Hindi corpus — for comparison with BPE.
"""

import argparse
import os
from pathlib import Path


def train(
    corpus_file: str,
    output_dir: str,
    vocab_size: int = 32000,
    num_threads: int = 4,
    input_sentence_size: int = 5_000_000,
) -> None:
    """Train SentencePiece Unigram model."""
    import sentencepiece as spm

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_prefix = os.path.join(output_dir, "hindi_unigram")

    print(f"Training Unigram tokenizer (vocab_size={vocab_size})")

    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        # High character coverage to capture all Devanagari characters
        character_coverage=0.9995,
        # NFC is already applied by our preprocess.py pipeline.
        # SentencePiece's built-in "nfc" charsmap is not available on all platforms.
        # Use "nmt_nfkc" (safe cross-platform) or identity (no-op) since input is pre-normalized.
        normalization_rule_name="nmt_nfkc",
        num_threads=num_threads,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        # Treat whitespace as part of token (like SentencePiece default)
        add_dummy_prefix=True,
        # Special tokens
        user_defined_symbols=[
            "<|pad|>", "<|bos|>", "<|eos|>",
            "<|sep|>", "<|mask|>", "<|user|>", "<|assistant|>"
        ],
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
    )

    print(f"Saved Unigram model to {model_prefix}.model and {model_prefix}.vocab")


def wrap_for_huggingface(model_path: str, output_dir: str) -> None:
    """Wrap SentencePiece model as HuggingFace tokenizer."""
    from transformers import XLNetTokenizer

    # Use XLNetTokenizer as it wraps SentencePiece natively
    # Alternatively use T5Tokenizer for unigram SP models
    try:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer(vocab_file=model_path)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
        print(f"HuggingFace Unigram tokenizer saved to {output_dir}")
    except Exception as e:
        print(f"HF wrapping failed: {e}")
        print("The raw .model file can still be used directly with sentencepiece.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hindi Unigram tokenizer")
    parser.add_argument("--corpus", required=True, help="Path to clean corpus .txt file")
    parser.add_argument("--output", default="models/ag_hindi_uni_tokenizer_32k")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    train(args.corpus, args.output, args.vocab_size, args.threads)
