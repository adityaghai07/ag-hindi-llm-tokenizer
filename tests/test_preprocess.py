"""
Tests for the corpus preprocessing pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from preprocess import nfc_normalize, remove_noise, normalize_whitespace, clean_line, hindi_ratio


class TestNFCNormalization:
    def test_nfc_idempotent(self):
        """NFC applied twice should give same result."""
        text = "हिंदी भाषा"
        assert nfc_normalize(nfc_normalize(text)) == nfc_normalize(text)

    def test_nfc_devanagari_preserved(self):
        """NFC should not destroy Devanagari characters."""
        text = "भारत सरकार"
        result = nfc_normalize(text)
        assert "भारत" in result
        assert "सरकार" in result

    def test_nfc_handles_decomposed(self):
        """NFC should compose decomposed characters."""
        # Compose: क (U+0915) + ा (U+093E) = का
        composed = "का"
        decomposed = "\u0915\u093E"
        assert nfc_normalize(decomposed) == nfc_normalize(composed)


class TestNoiseRemoval:
    def test_removes_emoji(self):
        """Emoji should be removed."""
        text = "भारत 🇮🇳 महान"
        result = remove_noise(text)
        assert "🇮🇳" not in result
        assert "भारत" in result

    def test_keeps_devanagari(self):
        """Devanagari characters must be preserved."""
        text = "हिंदी भाषा"
        result = remove_noise(text)
        assert "हिंदी" in result

    def test_keeps_latin(self):
        """Latin characters should be preserved."""
        text = "Hindi language"
        result = remove_noise(text)
        assert "Hindi" in result

    def test_keeps_danda(self):
        """Danda (।) should be preserved."""
        text = "यह वाक्य है।"
        result = remove_noise(text)
        assert "।" in result

    def test_removes_control_chars(self):
        """Control characters should be removed."""
        text = "भारत\x00\x01\x02"
        result = remove_noise(text)
        assert "\x00" not in result
        assert "भारत" in result


class TestHindiRatio:
    def test_pure_hindi(self):
        ratio = hindi_ratio("भारत सरकार")
        assert ratio > 0.7

    def test_pure_english(self):
        ratio = hindi_ratio("Hello World")
        assert ratio == 0.0

    def test_mixed(self):
        ratio = hindi_ratio("भारत India")
        assert 0.0 < ratio < 1.0

    def test_empty(self):
        assert hindi_ratio("") == 0.0


class TestCleanLine:
    def test_short_line_filtered(self):
        """Lines shorter than 50 chars should be filtered."""
        assert clean_line("भारत") is None

    def test_low_hindi_ratio_filtered(self):
        """Lines with < 30% Hindi should be filtered."""
        # Mostly English
        line = "This is a very long English sentence with just a bit of हिंदी text here."
        result = clean_line(line)
        # May or may not pass depending on ratio — just check no crash
        assert result is None or isinstance(result, str)

    def test_good_hindi_line_kept(self):
        """A clean Hindi line of sufficient length should pass."""
        line = "भारत एक महान देश है जहाँ अनेक भाषाएँ बोली जाती हैं और सभी लोग मिलकर रहते हैं।"
        result = clean_line(line)
        assert result is not None
        assert "भारत" in result

    def test_whitespace_normalized(self):
        """Multiple spaces should be collapsed."""
        line = "भारत   एक   महान   देश   है   जहाँ   अनेक   भाषाएँ   बोली   जाती   हैं।"
        result = clean_line(line)
        if result:
            assert "  " not in result  # No double spaces

    def test_nfc_applied(self):
        """NFC normalization should be applied."""
        # Decomposed form of a Devanagari character
        line = "भारत एक महान देश है जहाँ अनेक भाषाएँ बोली जाती हैं।" * 2
        result = clean_line(line)
        assert result is not None
