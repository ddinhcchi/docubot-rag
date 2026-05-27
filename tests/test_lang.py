import unicodedata

from src.lang import detect_language


def test_nfc_vietnamese_is_detected():
    assert detect_language("Tài liệu này nói về điều gì?") == "vi"


def test_nfd_vietnamese_is_detected():
    """NFD = decomposed form (base letter + combining mark). Some macOS
    paste paths produce this. Without normalization the regex misses it."""
    nfd = unicodedata.normalize("NFD", "Tài liệu nói gì?")
    # Sanity: NFD really is different from the original NFC string
    assert nfd != "Tài liệu nói gì?"
    assert detect_language(nfd) == "vi"


def test_diacritic_free_vietnamese_via_keywords():
    assert detect_language("Mo hinh nay la gi?") == "vi"
    assert detect_language("YOLOv8 la gi") == "vi"


def test_plain_english():
    assert detect_language("What is YOLOv8?") == "en"
    assert detect_language("How many parameters does the model have?") == "en"


def test_whitespace_around_diacritics():
    assert detect_language("   Mô hình   ") == "vi"


def test_empty_or_whitespace_defaults_to_english():
    assert detect_language("") == "en"
    assert detect_language("   ") == "en"
    assert detect_language("\n\t") == "en"


def test_mixed_language_with_diacritic_wins():
    # Question mixes EN scaffolding with VN content — VN wins because the
    # accent signal is unambiguous.
    assert detect_language("Please summarize this in tiếng Việt") == "vi"


def test_punctuation_only():
    assert detect_language("???") == "en"
    assert detect_language("...") == "en"
