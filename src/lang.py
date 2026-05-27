"""Detect whether a query is Vietnamese or English by checking for
diacritic characters specific to Vietnamese — fast, dependency-free,
and unambiguous compared to char-frequency language detectors."""
import re
import unicodedata

_VIETNAMESE_DIACRITICS = re.compile(
    r"[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]"
)

# Vietnamese-specific base letters (with diacritics stripped) and short
# function words. The diacritic-free fallback catches users who type without
# accents — common on QWERTY keyboards without VN input method.
_VN_KEYWORDS = {
    "la", "gi", "nao", "co", "khong", "duoc", "voi",
    "trong", "cua", "thi", "ban", "toi", "bao", "nhieu",
}


def _normalize(text: str) -> str:
    """Trim and normalize to NFC so composed/decomposed forms of the same
    glyph match the diacritics regex consistently.

    macOS pastes and some browser inputs deliver Vietnamese in NFD
    (e.g. `a` + combining grave) instead of NFC (`à` as one codepoint).
    Without normalization the regex would silently miss those characters.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFC", text).strip()


def detect_language(text: str) -> str:
    """Return 'vi' or 'en'.

    Accent characters are a hard signal; for accent-stripped Vietnamese
    (often typed on English keyboards) we fall back to keyword presence.
    Empty or whitespace-only input defaults to 'en'.
    """
    text = _normalize(text)
    if not text:
        return "en"
    if _VIETNAMESE_DIACRITICS.search(text):
        return "vi"
    tokens = {t.lower() for t in re.findall(r"[a-zA-Z]+", text)}
    if tokens & _VN_KEYWORDS:
        return "vi"
    return "en"


LANGUAGE_NAMES = {"vi": "Vietnamese (Tiếng Việt)", "en": "English"}
