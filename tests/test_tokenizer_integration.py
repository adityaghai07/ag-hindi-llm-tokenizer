"""
Integration tests: train a tiny BPE tokenizer on sample data,
then verify HuggingFace compatibility and Hindi-specific properties.
"""

import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


# ---------------------------------------------------------------------------
# Fixture: tiny sample corpus + trained tokenizer
# ---------------------------------------------------------------------------

SAMPLE_CORPUS = """भारत एक महान देश है जहाँ अनेक भाषाएँ बोली जाती हैं।
हिंदी भारत की राजभाषा है और करोड़ों लोग इसे बोलते हैं।
विद्यार्थी विश्वविद्यालय में पढ़ाई करते हैं।
सरकार ने नई नीति की घोषणा की।
माता-पिता का सम्मान करना चाहिए।
ज़िंदगी में फ़ैसले लेना ज़रूरी है।
क्षमा करना एक महान गुण है।
रचना और हरकत दोनों महत्वपूर्ण हैं।
हँसना स्वास्थ्य के लिए अच्छा है।
मैं office जा रहा हूँ, market से सामान लाऊँगा।
२०२५ में भारत की जनसंख्या बहुत अधिक है।
विद्यार्थी कार्यक्रम में भाग लेते हैं।
सत्याग्रह एक महान आंदोलन था।
भारतीय संस्कृति बहुत समृद्ध है।
हिंदी साहित्य में अनेक महान कवि हुए हैं।
""" * 50  # Repeat to give BPE enough data to learn merges


@pytest.fixture(scope="module")
def trained_tokenizer():
    """Train a small BPE tokenizer on sample data and return it."""
    from train_bpe import train, wrap_for_huggingface
    from transformers import PreTrainedTokenizerFast

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "corpus.txt")
        bpe_dir = os.path.join(tmpdir, "bpe")
        hf_dir = os.path.join(tmpdir, "hf")

        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CORPUS)

        # Train with small vocab for speed
        train(
            corpus_files=[corpus_path],
            output_dir=bpe_dir,
            vocab_size=1000,
            min_frequency=2,
        )

        wrap_for_huggingface(
            os.path.join(bpe_dir, "tokenizer.json"),
            hf_dir
        )

        tokenizer = PreTrainedTokenizerFast.from_pretrained(hf_dir)
        yield tokenizer


class TestHuggingFaceCompatibility:
    def test_encode_returns_list(self, trained_tokenizer):
        ids = trained_tokenizer.encode("भारत")
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_decode_roundtrip(self, trained_tokenizer):
        """Encode then decode should recover the original text (approximately)."""
        text = "भारत एक देश है।"
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids, skip_special_tokens=True)
        # Should contain the key words
        assert "भारत" in decoded

    def test_special_tokens_present(self, trained_tokenizer):
        """All special tokens must be in the vocabulary."""
        vocab = trained_tokenizer.get_vocab()
        for tok in ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]:
            assert tok in vocab, f"Special token {tok!r} missing from vocab"

    def test_bos_eos_tokens(self, trained_tokenizer):
        """BOS and EOS tokens should be accessible."""
        assert trained_tokenizer.bos_token == "<|bos|>"
        assert trained_tokenizer.eos_token == "<|eos|>"
        assert trained_tokenizer.pad_token == "<|pad|>"
        assert trained_tokenizer.unk_token == "<|unk|>"

    def test_batch_encoding(self, trained_tokenizer):
        """Batch encoding should work."""
        texts = ["भारत", "हिंदी भाषा", "विद्यार्थी"]
        result = trained_tokenizer(texts, padding=True, return_tensors=None)
        assert "input_ids" in result
        assert len(result["input_ids"]) == 3

    def test_save_and_reload(self, trained_tokenizer):
        """Tokenizer should survive save/reload cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_tokenizer.save_pretrained(tmpdir)
            from transformers import PreTrainedTokenizerFast
            reloaded = PreTrainedTokenizerFast.from_pretrained(tmpdir)
            ids1 = trained_tokenizer.encode("भारत")
            ids2 = reloaded.encode("भारत")
            assert ids1 == ids2


class TestHindiProperties:
    def test_no_unk_on_devanagari(self, trained_tokenizer):
        """Common Hindi words should not produce UNK tokens."""
        unk_id = trained_tokenizer.unk_token_id
        for word in ["भारत", "हिंदी", "सरकार"]:
            ids = trained_tokenizer.encode(word, add_special_tokens=False)
            assert unk_id not in ids, f"UNK found for {word!r}: {ids}"

    def test_fertility_reasonable(self, trained_tokenizer):
        """Fertility should be reasonable even for a tiny 1K vocab tokenizer."""
        from evaluate import fertility
        texts = [
            "भारत एक महान देश है।",
            "हिंदी भारत की राजभाषा है।",
            "विद्यार्थी पढ़ाई करते हैं।",
        ]
        f = fertility(trained_tokenizer, texts)
        # Even with 1K vocab, fertility should be < 10 (not byte-level fragmentation)
        assert f < 10.0, f"Fertility too high: {f}"

    def test_conjunct_not_split_across_tokens(self, trained_tokenizer):
        """
        Verify that conjuncts like क्ष are not split into
        क + ् + ष (which would be semantically broken).
        """
        # Encode क्ष and check that virama is not a standalone token
        ids = trained_tokenizer.encode("क्ष", add_special_tokens=False)
        tokens = trained_tokenizer.convert_ids_to_tokens(ids)
        VIRAMA = '\u094D'
        assert VIRAMA not in tokens, f"Bare virama token found: {tokens}"

    def test_encode_decode_hindi_sentence(self, trained_tokenizer):
        """Full sentence encode/decode roundtrip."""
        sentence = "भारत एक महान देश है।"
        ids = trained_tokenizer.encode(sentence, add_special_tokens=False)
        decoded = trained_tokenizer.decode(ids)
        # Key words should survive roundtrip
        assert "भारत" in decoded


class TestEvaluationMetrics:
    def test_fertility_function(self, trained_tokenizer):
        from evaluate import fertility
        texts = ["भारत एक देश है।"]
        f = fertility(trained_tokenizer, texts)
        assert isinstance(f, float)
        assert f > 0

    def test_compression_ratio(self, trained_tokenizer):
        from evaluate import compression_ratio
        texts = ["भारत एक देश है।"]
        cr = compression_ratio(trained_tokenizer, texts)
        assert isinstance(cr, float)
        assert cr > 0

    def test_qualitative_report(self, trained_tokenizer):
        from evaluate import qualitative_report
        words = ["भारत", "हिंदी"]
        report = qualitative_report(trained_tokenizer, words)
        assert len(report) == 2
        for r in report:
            assert "word" in r
            assert "tokens" in r
            assert "n_tokens" in r
            assert r["n_tokens"] > 0
