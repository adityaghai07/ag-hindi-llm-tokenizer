"""
Demo: Using the trained Hindi tokenizers on sample sentences.
Shows both BPE (HuggingFace) and Unigram (SentencePiece) usage.
"""

from transformers import PreTrainedTokenizerFast
import sentencepiece as spm
from pathlib import Path


SENTENCES = [
    "भारत एक महान देश है।",
    "मैं office जा रहा हूँ।",
    "विद्यार्थी विश्वविद्यालय में पढ़ाई करते हैं।",
    "ॐ नमः शिवाय।",
    "ज़िंदगी में फ़ैसले लेना ज़रूरी है।",
    "क्षमा करना एक महान गुण है।",
    "मेरा नाम आदित्य घई है।"
]

BPE_PATH = "models/ag_hindi_bpe_tokenizer_32k"
UNI_PATH = "models/ag_hindi_uni_tokenizer_32k/hindi_unigram.model"


def demo_bpe():
    print("\n" + "="*70)
    print("  BPE Tokenizer (Metaspace)")
    print("="*70)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(BPE_PATH)

    for sent in SENTENCES:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        decoded = tokenizer.decode(ids)

        print(f"\nInput:   {sent}")
        print(f"Tokens:  {tokens}")
        print(f"Count:   {len(ids)} tokens")
        print(f"Decoded: {decoded}")

    print("\n" + "-"*70)
    batch = tokenizer(SENTENCES, padding=True, return_tensors="pt")
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")


def demo_unigram():
    print("\n" + "="*70)
    print("  Unigram Tokenizer (SentencePiece)")
    print("="*70)

    sp = spm.SentencePieceProcessor()
    sp.Load(UNI_PATH)

    for sent in SENTENCES:
        ids = sp.EncodeAsIds(sent)
        tokens = sp.EncodeAsPieces(sent)
        decoded = sp.DecodeIds(ids)

        print(f"\nInput:   {sent}")
        print(f"Tokens:  {tokens}")
        print(f"Count:   {len(ids)} tokens")
        print(f"Decoded: {decoded}")

    print("\n" + "-"*70)
    print(f"Vocab size: {sp.GetPieceSize()}")
    print(f"BOS: {sp.IdToPiece(sp.bos_id())} (id={sp.bos_id()})")
    print(f"EOS: {sp.IdToPiece(sp.eos_id())} (id={sp.eos_id()})")


def compare_side_by_side():
    print("\n" + "="*70)
    print("  BPE vs Unigram Comparison")
    print("="*70)

    bpe = PreTrainedTokenizerFast.from_pretrained(BPE_PATH)
    sp = spm.SentencePieceProcessor()
    sp.Load(UNI_PATH)

    test_sent = "मेरा नाम आदित्य घई है।"
    print(f"\nSentence: {test_sent}\n")

    bpe_ids = bpe.encode(test_sent, add_special_tokens=False)
    bpe_tokens = bpe.convert_ids_to_tokens(bpe_ids)
    print(f"BPE:     {bpe_tokens}")
    print(f"         {len(bpe_ids)} tokens\n")

    uni_ids = sp.EncodeAsIds(test_sent)
    uni_tokens = sp.EncodeAsPieces(test_sent)
    print(f"Unigram: {uni_tokens}")
    print(f"         {len(uni_ids)} tokens")


if __name__ == "__main__":
    print("\nHindi Tokenizer Demo\n")

    if not Path(BPE_PATH).exists():
        print(f"BPE model not found at {BPE_PATH}")
        exit(1)

    if not Path(UNI_PATH).exists():
        print(f"Unigram model not found at {UNI_PATH}")
        exit(1)

    demo_bpe()
    demo_unigram()
    compare_side_by_side()

    print("\n" + "="*70)
    print("Demo complete.")
    print("="*70)
