"""
Tests for Hindi pre-tokenization.
Covers: conjuncts, matras, anusvara, chandrabindu, nukta, code-mixing, numbers.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from pretokenize import pretokenize, grapheme_clusters, validate_no_virama_split

VIRAMA = '\u094D'


class TestConjuncts:
    """Virama-joined consonant clusters must never be split."""

    def test_ksha_conjunct_intact(self):
        """क्ष must be a single pre-token."""
        tokens = pretokenize("क्ष")
        assert len(tokens) == 1, f"क्ष split into {tokens}"
        assert tokens[0] == "क्ष"

    def test_tra_conjunct_intact(self):
        """त्र must be a single pre-token."""
        tokens = pretokenize("त्र")
        assert len(tokens) == 1, f"त्र split into {tokens}"

    def test_gya_conjunct_intact(self):
        """ज्ञ must be a single pre-token."""
        tokens = pretokenize("ज्ञ")
        assert len(tokens) == 1, f"ज्ञ split into {tokens}"

    def test_three_consonant_cluster(self):
        """स्त्र (stra) — three-consonant cluster must stay together."""
        tokens = pretokenize("स्त्र")
        assert len(tokens) == 1, f"स्त्र split into {tokens}"

    def test_conjunct_in_word(self):
        """विद्यार्थी — contains conjunct, must not split it."""
        tokens = pretokenize("विद्यार्थी")
        # Should be a single word token
        assert len(tokens) == 1
        # No token should end with bare virama
        bad = validate_no_virama_split(tokens)
        assert not bad, f"Virama split detected: {bad}"

    def test_no_bare_virama_tokens(self):
        """No token should ever be a standalone virama."""
        text = "कार्यक्रम सत्याग्रह विद्यार्थी"
        tokens = pretokenize(text)
        assert VIRAMA not in tokens, f"Bare virama found in tokens: {tokens}"

    def test_no_token_ends_with_virama(self):
        """No token should end with virama (would indicate a split conjunct)."""
        text = "क्षमा त्रिकोण ज्ञान श्रम"
        tokens = pretokenize(text)
        bad = validate_no_virama_split(tokens)
        assert not bad, f"Tokens ending with virama: {bad}"


class TestMatras:
    """Vowel signs must stay attached to their base consonant."""

    def test_aa_matra(self):
        """का — consonant + ā matra must be one token."""
        tokens = pretokenize("का")
        assert len(tokens) == 1

    def test_i_matra(self):
        """कि — consonant + i matra."""
        tokens = pretokenize("कि")
        assert len(tokens) == 1

    def test_u_matra(self):
        """कु — consonant + u matra."""
        tokens = pretokenize("कु")
        assert len(tokens) == 1

    def test_e_matra(self):
        """के — consonant + e matra."""
        tokens = pretokenize("के")
        assert len(tokens) == 1

    def test_o_matra(self):
        """को — consonant + o matra."""
        tokens = pretokenize("को")
        assert len(tokens) == 1

    def test_word_with_multiple_matras(self):
        """भारत — multiple matras, should be one word token."""
        tokens = pretokenize("भारत")
        assert len(tokens) == 1

    def test_hindi_word_intact(self):
        """हिंदी — consonant + i matra + anusvara, must be one token."""
        tokens = pretokenize("हिंदी")
        assert len(tokens) == 1


class TestAnusvaraAndChandrabindu:
    """Anusvara (ं) and chandrabindu (ँ) must stay with their base."""

    def test_anusvara_stays_attached(self):
        """हिंदी — anusvara on इ must not be a separate token."""
        tokens = pretokenize("हिंदी")
        # Should be one word, anusvara not standalone
        assert '\u0902' not in tokens  # U+0902 = anusvara

    def test_chandrabindu_stays_attached(self):
        """हँसना — chandrabindu must not be a separate token."""
        tokens = pretokenize("हँसना")
        assert '\u0901' not in tokens  # U+0901 = chandrabindu
        assert len(tokens) == 1

    def test_anusvara_in_sentence(self):
        """Sentence with anusvara words."""
        tokens = pretokenize("हिंदी भाषा")
        # Two word tokens + possible space handling
        word_tokens = [t for t in tokens if t.strip()]
        assert "हिंदी" in word_tokens
        assert "भाषा" in word_tokens


class TestNukta:
    """Nukta (़) modifies consonants for Persian/Arabic sounds."""

    def test_za_nukta(self):
        """ज़ (za) — nukta must stay with ज."""
        tokens = pretokenize("ज़")
        assert len(tokens) == 1
        assert '\u093C' not in tokens  # U+093C = nukta not standalone

    def test_fa_nukta(self):
        """फ़ (fa) — nukta must stay with फ."""
        tokens = pretokenize("फ़")
        assert len(tokens) == 1

    def test_nukta_word(self):
        """ज़िंदगी — nukta word must be one token."""
        tokens = pretokenize("ज़िंदगी")
        assert len(tokens) == 1

    def test_nukta_not_standalone(self):
        """Nukta character must never appear as a standalone token."""
        text = "ज़िंदगी फ़ैसला क़िस्मत"
        tokens = pretokenize(text)
        assert '\u093C' not in tokens


class TestCodeMixing:
    """English words in Hindi text (Hinglish)."""

    def test_english_word_tokenized(self):
        """English word in Hindi sentence should be its own token."""
        tokens = pretokenize("मैं office जा रहा हूँ")
        assert "office" in tokens

    def test_hindi_english_separation(self):
        """Hindi and English words should be separate tokens."""
        tokens = pretokenize("मैं market जा रहा हूँ")
        assert "market" in tokens
        assert "मैं" in tokens

    def test_pure_english(self):
        """Pure English text should tokenize normally."""
        tokens = pretokenize("Hello World")
        assert "Hello" in tokens
        assert "World" in tokens


class TestNumbers:
    """Both Devanagari and Arabic numerals."""

    def test_devanagari_digits(self):
        """Devanagari digits should tokenize."""
        tokens = pretokenize("२०२५")
        assert len(tokens) >= 1
        # All chars should be Devanagari digits
        combined = "".join(tokens)
        assert all('\u0966' <= c <= '\u096F' for c in combined)

    def test_arabic_digits(self):
        """Arabic digits should tokenize."""
        tokens = pretokenize("2025")
        assert len(tokens) >= 1

    def test_mixed_numbers_in_sentence(self):
        """Both number types in same sentence."""
        tokens = pretokenize("२०२५ और 2025 दोनों")
        # Should contain both number representations
        all_text = " ".join(tokens)
        assert "२०२५" in all_text or any(t.startswith("२") for t in tokens)
        assert "2025" in all_text or any(t.isdigit() for t in tokens)


class TestPunctuation:
    """Devanagari and Western punctuation."""

    def test_danda_is_token(self):
        """Danda (।) should be its own token."""
        tokens = pretokenize("भारत एक देश है।")
        assert "।" in tokens

    def test_double_danda(self):
        """Double danda (॥) should be its own token."""
        tokens = pretokenize("राम राम।। सीता राम॥")
        assert "॥" in tokens

    def test_sentence_boundary(self):
        """Sentence with danda should split correctly."""
        tokens = pretokenize("यह वाक्य है।")
        assert "।" in tokens
        assert "यह" in tokens


class TestGraphemeClusters:
    """Test the grapheme cluster splitter directly."""

    def test_conjunct_is_single_cluster(self):
        """क्ष should be one grapheme cluster."""
        clusters = grapheme_clusters("क्ष")
        assert len(clusters) == 1, f"Expected 1 cluster, got {clusters}"

    def test_matra_is_part_of_cluster(self):
        """का should be one grapheme cluster."""
        clusters = grapheme_clusters("का")
        assert len(clusters) == 1

    def test_anusvara_is_part_of_cluster(self):
        """हिं should be one grapheme cluster (ह + ि + ं)."""
        # Actually हि is one cluster, ं attaches to it
        clusters = grapheme_clusters("हिं")
        # Should be 1 or 2 clusters depending on Unicode rules
        # Key: anusvara must NOT be a standalone cluster
        assert '\u0902' not in clusters  # anusvara not standalone


class TestEdgeCases:
    """Edge cases and tricky inputs."""

    def test_empty_string(self):
        tokens = pretokenize("")
        assert tokens == []

    def test_only_spaces(self):
        tokens = pretokenize("   ")
        # Spaces may or may not produce tokens depending on pattern
        # Key: no crash
        assert isinstance(tokens, list)

    def test_only_punctuation(self):
        tokens = pretokenize("।॥")
        assert "।" in tokens
        assert "॥" in tokens

    def test_long_compound_word(self):
        """विश्वविद्यालय — long word with multiple conjuncts."""
        tokens = pretokenize("विश्वविद्यालय")
        assert len(tokens) == 1
        bad = validate_no_virama_split(tokens)
        assert not bad

    def test_hyphenated_compound(self):
        """माता-पिता — hyphenated compound."""
        tokens = pretokenize("माता-पिता")
        # Should split on hyphen: माता, -, पिता
        assert "माता" in tokens
        assert "पिता" in tokens

    def test_mixed_script_sentence(self):
        """Full mixed-script sentence."""
        text = "मैं office जा रहा हूँ, market से सामान लाऊँगा।"
        tokens = pretokenize(text)
        bad = validate_no_virama_split(tokens)
        assert not bad
        assert "office" in tokens
        assert "market" in tokens
        assert "।" in tokens

    def test_satra_three_consonant_cluster(self):
        """स्त्र — three-consonant cluster."""
        tokens = pretokenize("स्त्र")
        assert len(tokens) == 1
        bad = validate_no_virama_split(tokens)
        assert not bad
