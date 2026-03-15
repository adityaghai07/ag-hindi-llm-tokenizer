"""
Evaluation framework for Hindi tokenizer.
Measures fertility, compression ratio, and qualitative token inspection.
"""

import regex
from typing import List, Dict, Any
from pathlib import Path


# ---------------------------------------------------------------------------
# Word splitter for fertility calculation
# Splits on whitespace, counts Devanagari + Latin word tokens
# ---------------------------------------------------------------------------
WORD_RE = regex.compile(r'[\u0900-\u097F]+|[a-zA-Z]+')


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def fertility(tokenizer, texts: List[str]) -> float:
    """
    Fertility = total tokens / total words.
    Lower is better. Target: < 2.0 for Hindi.
    """
    total_tokens = 0
    total_words = 0
    for text in texts:
        encoded = tokenizer.encode(text)
        # Handle both HF tokenizer (has .ids) and raw tokenizers lib
        if hasattr(encoded, 'ids'):
            n_tokens = len(encoded.ids)
        else:
            n_tokens = len(encoded)
        total_tokens += n_tokens
        total_words += count_words(text)

    if total_words == 0:
        return float('inf')
    return total_tokens / total_words


def compression_ratio(tokenizer, texts: List[str]) -> float:
    """
    Compression ratio = total characters / total tokens.
    Higher is better — more characters packed per token.
    """
    total_chars = sum(len(t) for t in texts)
    total_tokens = 0
    for text in texts:
        encoded = tokenizer.encode(text)
        if hasattr(encoded, 'ids'):
            total_tokens += len(encoded.ids)
        else:
            total_tokens += len(encoded)
    if total_tokens == 0:
        return 0.0
    return total_chars / total_tokens


def continued_word_ratio(tokenizer, texts: List[str]) -> float:
    """
    Proportion of tokens that are continuations (not word-initial).
    High ratio = heavy fragmentation.
    """
    total_tokens = 0
    continued = 0
    for text in texts:
        encoded = tokenizer.encode(text)
        if hasattr(encoded, 'tokens'):
            tokens = encoded.tokens
        else:
            # HF tokenizer
            tokens = tokenizer.convert_ids_to_tokens(encoded)

        for tok in tokens:
            total_tokens += 1
            # Tokens starting with '##' (BERT-style) or '▁' absence (SP-style)
            # For our BPE, a continuation token won't start with a space
            if tok and tok[0] not in (' ', '▁') and not tok.startswith('<|'):
                continued += 1

    if total_tokens == 0:
        return 0.0
    return continued / total_tokens


def qualitative_report(tokenizer, words: List[str]) -> List[Dict[str, Any]]:
    """
    Tokenize individual words and return detailed breakdown.
    """
    results = []
    for word in words:
        encoded = tokenizer.encode(word)
        if hasattr(encoded, 'tokens'):
            tokens = encoded.tokens
            ids = encoded.ids
        else:
            ids = encoded
            tokens = tokenizer.convert_ids_to_tokens(ids)

        results.append({
            "word": word,
            "tokens": tokens,
            "ids": ids,
            "n_tokens": len(tokens),
        })
    return results


def print_qualitative_report(tokenizer) -> None:
    """Print qualitative inspection of key Hindi words."""
    test_words = [
        # Common words
        "भारत", "सरकार", "विश्वविद्यालय",
        # Conjunct-heavy
        "विद्यार्थी", "कार्यक्रम", "सत्याग्रह",
        # Schwa deletion
        "रचना", "हरकत", "सरकना",
        # Compound
        "लाभ-हानि", "माता-पिता",
        # Code-mixed
        "office", "market",
        # Nukta
        "ज़िंदगी", "फ़ैसला", "क़िस्मत",
        # Numbers
        "२०२५", "2025", "१,२३,४५६",
        # Anusvara / chandrabindu
        "हिंदी", "हँसना",
        # Conjuncts
        "क्ष", "त्र", "ज्ञ", "श्र",
        # Morphological suffixes
        "जाना", "जाता", "जाती", "जाएगा", "जाकर",
        # Postpositions
        "में", "पर", "से", "को",
    ]

    print("\n=== Qualitative Token Inspection ===\n")
    report = qualitative_report(tokenizer, test_words)
    for r in report:
        flag = "WARN" if r["n_tokens"] > 3 else "OK"
        print(f"[{flag}] {r['word']!r:20s} -> {r['tokens']}  ({r['n_tokens']} tokens)")


def run_full_evaluation(tokenizer, test_texts: List[str], tokenizer_name: str = "Hindi BPE") -> Dict:
    """Run all metrics and return results dict."""
    f = fertility(tokenizer, test_texts)
    cr = compression_ratio(tokenizer, test_texts)

    print(f"\n=== {tokenizer_name} Evaluation ===")
    print(f"  Fertility (tokens/word):  {f:.3f}  (target < 2.0)")
    print(f"  Compression ratio (chars/token): {cr:.3f}")

    print_qualitative_report(tokenizer)

    return {"fertility": f, "compression_ratio": cr, "name": tokenizer_name}


# ---------------------------------------------------------------------------
# Comparison against other tokenizers
# ---------------------------------------------------------------------------

def compare_tokenizers(test_texts: List[str], our_tokenizer_path: str) -> None:
    """Compare our tokenizer against GPT-2, LLaMA-style, etc."""
    from transformers import PreTrainedTokenizerFast, AutoTokenizer

    results = []

    # Our tokenizer
    try:
        our_tok = PreTrainedTokenizerFast.from_pretrained(our_tokenizer_path)
        results.append(run_full_evaluation(our_tok, test_texts, "Hindi BPE (ours)"))
    except Exception as e:
        print(f"Could not load our tokenizer: {e}")

    # GPT-2 (as baseline)
    try:
        gpt2 = AutoTokenizer.from_pretrained("gpt2")
        results.append(run_full_evaluation(gpt2, test_texts, "GPT-2"))
    except Exception as e:
        print(f"Could not load GPT-2: {e}")

    # Summary table
    if results:
        print("\n=== Summary ===")
        print(f"{'Tokenizer':<30} {'Fertility':>10} {'Compression':>12}")
        print("-" * 55)
        for r in results:
            print(f"{r['name']:<30} {r['fertility']:>10.3f} {r['compression_ratio']:>12.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="Path to HF tokenizer dir")
    parser.add_argument("--test-file", help="Optional .txt file with test sentences")
    parser.add_argument("--compare", action="store_true", help="Compare against GPT-2")
    args = parser.parse_args()

    # Default test sentences if no file provided
    DEFAULT_TEXTS = [
        "भारत एक महान देश है जहाँ अनेक भाषाएँ बोली जाती हैं।",
        "विद्यार्थी विश्वविद्यालय में पढ़ाई करते हैं।",
        "सरकार ने नई नीति की घोषणा की।",
        "हिंदी भारत की राजभाषा है।",
        "मैं office जा रहा हूँ, market से सामान लाऊँगा।",
        "ज़िंदगी में फ़ैसले लेना ज़रूरी है।",
        "२०२५ में भारत की जनसंख्या बहुत अधिक है।",
        "माता-पिता का सम्मान करना चाहिए।",
        "रचना और हरकत दोनों महत्वपूर्ण हैं।",
        "क्षमा करना एक महान गुण है।",
    ]

    if args.test_file:
        with open(args.test_file, encoding="utf-8") as f:
            test_texts = [l.strip() for l in f if l.strip()]
    else:
        test_texts = DEFAULT_TEXTS

    if args.compare:
        compare_tokenizers(test_texts, args.tokenizer)
    else:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
        run_full_evaluation(tok, test_texts)
