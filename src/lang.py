"""Detect whether a query is Vietnamese or English by checking for
diacritic characters specific to Vietnamese — fast, dependency-free,
and unambiguous compared to char-frequency language detectors."""
import re

_VIETNAMESE_DIACRITICS = re.compile(
    r"[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]"
)

_VN_KEYWORDS = {
    "la", "gi", "nao", "co", "khong", "duoc", "voi",
    "trong", "cua", "thi", "ban", "toi", "bao", "nhieu",
}


def detect_language(text: str) -> str:
    """Return 'vi' or 'en'.

    Accent characters are a hard signal; for accent-stripped Vietnamese
    (often typed on English keyboards) we fall back to keyword presence.
    """
    if _VIETNAMESE_DIACRITICS.search(text):
        return "vi"
    tokens = {t.lower() for t in re.findall(r"[a-zA-Z]+", text)}
    if tokens & _VN_KEYWORDS:
        return "vi"
    return "en"


LANGUAGE_NAMES = {"vi": "Vietnamese (Tiếng Việt)", "en": "English"}
