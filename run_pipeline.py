"""
Full training pipeline for Hindi tokenizer.
Runs: clean → train BPE → train Unigram → evaluate both

Usage (from inside hindi-tokenizer/):
    python run_pipeline.py
    python run_pipeline.py --skip-clean        # if already cleaned
    python run_pipeline.py --skip-unigram      # BPE only
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to this script, which lives in hindi-tokenizer/)
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
DATA_RAW      = BASE / "../data/raw/hindi_wikipedia.txt"
DATA_CLEAN    = BASE / "../data/clean/hindi_clean.txt"
BPE_DIR       = BASE / "../models/bpe_32k"
BPE_HF_DIR    = BASE / "../models/ag_hindi_bpe_tokenizer_32k"
UNIGRAM_DIR   = BASE / "../models/ag_hindi_uni_tokenizer_32k"

VOCAB_SIZE    = 32000
MIN_FREQ      = 2

sys.path.insert(0, str(BASE / "src"))


def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def step_clean(skip: bool) -> None:
    banner("STEP 1: Clean corpus")
    if skip:
        print("Skipping (--skip-clean passed)")
        if not DATA_CLEAN.exists():
            print(f"ERROR: {DATA_CLEAN} does not exist. Run without --skip-clean first.")
            sys.exit(1)
        return

    if not DATA_RAW.exists():
        print(f"ERROR: Raw corpus not found at {DATA_RAW}")
        print("Run data_download.py first:")
        print("  python src/data_download.py --source wikipedia --output-dir ../data/raw")
        sys.exit(1)

    from preprocess import process_file
    DATA_CLEAN.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    stats = process_file(str(DATA_RAW), str(DATA_CLEAN))
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.1f}s")
    print(f"  Total lines : {stats['total']:,}")
    print(f"  Kept        : {stats['kept']:,}")
    print(f"  Filtered    : {stats['filtered']:,}")
    print(f"  Duplicates  : {stats['duplicates']:,}")
    print(f"  Output      : {DATA_CLEAN}")


def step_train_bpe() -> None:
    banner("STEP 2: Train BPE tokenizer (32K vocab)")
    from train_bpe import train, wrap_for_huggingface

    t0 = time.time()
    train(
        corpus_files=[str(DATA_CLEAN)],
        output_dir=str(BPE_DIR),
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
    )
    wrap_for_huggingface(
        tokenizer_json_path=str(BPE_DIR / "tokenizer.json"),
        output_dir=str(BPE_HF_DIR),
    )
    print(f"Done in {time.time() - t0:.1f}s")
    print(f"  Raw tokenizer : {BPE_DIR}/tokenizer.json")
    print(f"  HF tokenizer  : {BPE_HF_DIR}  (name: ag_hindi_bpe_tokenizer_32k)")


def step_train_unigram(skip: bool) -> None:
    banner("STEP 3: Train Unigram tokenizer (32K vocab)")
    if skip:
        print("Skipping (--skip-unigram passed)")
        return

    from train_unigram import train
    t0 = time.time()
    train(
        corpus_file=str(DATA_CLEAN),
        output_dir=str(UNIGRAM_DIR),
        vocab_size=VOCAB_SIZE,
    )
    print(f"Done in {time.time() - t0:.1f}s")
    print(f"  Model : {UNIGRAM_DIR}/hindi_unigram.model")


def step_evaluate(skip_unigram: bool) -> None:
    banner("STEP 4: Evaluate")
    from evaluate import run_full_evaluation, compare_tokenizers
    from transformers import PreTrainedTokenizerFast

    TEST_TEXTS = [
        "भारत एक महान देश है जहाँ अनेक भाषाएँ बोली जाती हैं।",
        "विद्यार्थी विश्वविद्यालय में पढ़ाई करते हैं।",
        "सरकार ने नई नीति की घोषणा की।",
        "हिंदी भारत की राजभाषा है।",
        "मैं office जा रहा हूँ, market से सामान लाऊँगा।",
        "ज़िंदगी में फ़ैसले लेना ज़रूरी है।",
        "२०२५ में भारत की जनसंख्या बहुत अधिक है।",
        "माता-पिता का सम्मान करना चाहिए।",
        "क्षमा करना एक महान गुण है।",
        "रचना और हरकत दोनों महत्वपूर्ण हैं।",
    ]

    results = []

    # BPE
    bpe_tok = PreTrainedTokenizerFast.from_pretrained(str(BPE_HF_DIR))
    results.append(run_full_evaluation(bpe_tok, TEST_TEXTS, "ag_hindi_bpe_tokenizer_32k"))

    # Unigram (via sentencepiece directly)
    if not skip_unigram:
        unigram_model = UNIGRAM_DIR / "hindi_unigram.model"
        if unigram_model.exists():
            try:
                import sentencepiece as spm

                class SPTokenizer:
                    """Thin wrapper so evaluate.py works with SentencePiece."""
                    def __init__(self, model_path):
                        self.sp = spm.SentencePieceProcessor()
                        self.sp.Load(str(model_path))
                    def encode(self, text):
                        return self.sp.EncodeAsIds(text)
                    def convert_ids_to_tokens(self, ids):
                        return [self.sp.IdToPiece(i) for i in ids]

                sp_tok = SPTokenizer(unigram_model)
                results.append(run_full_evaluation(sp_tok, TEST_TEXTS, "hindi_unigram_32k"))
            except Exception as e:
                print(f"Could not evaluate Unigram: {e}")

    # GPT-2 baseline
    try:
        from transformers import AutoTokenizer
        gpt2 = AutoTokenizer.from_pretrained("gpt2")
        results.append(run_full_evaluation(gpt2, TEST_TEXTS, "GPT-2 (baseline)"))
    except Exception:
        pass

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Tokenizer':<35} {'Fertility':>10} {'Compression':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<35} {r['fertility']:>10.3f} {r['compression_ratio']:>12.3f}")
    print()
    print("Fertility  = tokens/word  (lower is better, target < 2.0)")
    print("Compression = chars/token (higher is better)")


def main():
    parser = argparse.ArgumentParser(description="Hindi tokenizer training pipeline")
    parser.add_argument("--skip-clean",   action="store_true", help="Skip corpus cleaning step")
    parser.add_argument("--skip-unigram", action="store_true", help="Skip Unigram training")
    args = parser.parse_args()

    print("\nHindi Tokenizer Training Pipeline")
    print(f"Raw corpus  : {DATA_RAW}")
    print(f"Clean corpus: {DATA_CLEAN}")
    print(f"BPE output  : {BPE_HF_DIR}")
    print(f"Unigram out : {UNIGRAM_DIR}")
    print(f"Vocab size  : {VOCAB_SIZE}")

    step_clean(args.skip_clean)
    step_train_bpe()
    step_train_unigram(args.skip_unigram)
    step_evaluate(args.skip_unigram)

    banner("DONE")
    print(f"HF tokenizer ready at: {BPE_HF_DIR.resolve()}")
    print("Load it with:")
    print("  from transformers import PreTrainedTokenizerFast")
    print(f"  tok = PreTrainedTokenizerFast.from_pretrained('{BPE_HF_DIR.resolve()}')")


if __name__ == "__main__":
    main()
