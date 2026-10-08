"""
Kashmiri Text Normalizer and Script Canonicalizer.
Designed for neural TTS models to ensure authentic pronunciation,
preservation of central vowels, and palatalization (Plat Ye).
"""

import re
import unicodedata
from typing import List, Dict

# Kashmiri Number to Spoken Words Mapping (0 to 20, tens, hundreds)
KASHMIRI_NUMBERS = {
    0: "صفر",
    1: "اکھ",
    2: "زٕ",
    3: "ترےٚ",
    4: "ژور",
    5: "پانژھ",
    6: "شےٚ",
    7: "ستھ",
    8: "ٲٹھ",
    9: "نَو",
    10: "دَہ",
    11: "کاە",
    12: "باە",
    13: "ترواە",
    14: "ژوداە",
    15: "پنداہ",
    16: "شوراە",
    17: "سداہ",
    18: "ارداہ",
    19: "کُنہٕ وُہ",
    20: "وُہ",
    30: "ترِہ",
    40: "ژتجیِہ",
    50: "پژاہ",
    60: "شےٚٹھ",
    70: "ستتھ",
    80: "شیٖتھ",
    90: "نَوے",
    100: "ہَتھ",
}

# Arabic/Indic Digit to ASCII Digit mapping
DIGIT_MAP = {
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}

# Unicode character canonicalization map
CANONICAL_CHAR_MAP = {
    '\u064A': '\u06CC',  # Arabic Yeh -> Farsi/Urdu/Kashmiri Yeh
    '\u0649': '\u06CC',  # Alef Maksura -> Yeh
    '\u0643': '\u06A9',  # Arabic Kaf -> Keheh
    '\u0629': '\u06C1',  # Teh Marbuta -> Heh Goal
    '\u06BE': '\u06BE',  # Doachashmee Heh preserved
    '\u06C0': '\u06C1\u0654',  # Heh with Yeh above decomposed
}

# Comprehensive symbol inventory for Kashmiri character-level TTS
# Includes: Standard Perso-Arabic alphabet, Kashmiri extensions (ٲ, ٳ, ۄ, ۆ, ؠ),
# Diacritics (َ, ِ, ُ, ٕ, ٖ, ٗ, ٰ, ٔ), and standard punctuation for prosodic pauses.
KASHMIRI_ALPHABET = [
    # Control tokens
    "<pad>", "<unk>", "<s>", "</s>", " ",
    # Punctuation (critical for prosodic pauses)
    "،", "؛", "؟", ".", "!", "-", ":", "«", "»", "\"", "'",
    # Basic Consonants & Vowels
    "ا", "آ", "ب", "پ", "ت", "ٹ", "ث", "ج", "چ", "ح", "خ",
    "د", "ڈ", "ذ", "ر", "ڑ", "ز", "ژ", "س", "ش", "ص", "ض",
    "ط", "ظ", "ع", "غ", "ف", "ق", "ک", "گ", "ل", "م", "ن",
    "ں", "و", "ہ", "ھ", "ء", "ی", "ے",
    # Kashmiri-Specific Characters
    "ٲ",  # Alef with wavy hamza / circle (high central vowel)
    "ٳ",  # Alef with bottom hamza
    "ۄ",  # Waw with ring (open-back vowel)
    "ۆ",  # Waw with v above
    "ؠ",  # Kashmiri Plat Ye (Palatalization marker)
    "ۍ",  # Yeh with tail
    "ۅ",  # Waw with inverted v
    # Kashmiri & Arabic Diacritics
    "\u064E",  # Zabar (Fatha)
    "\u0650",  # Zer (Kasra)
    "\u064F",  # Pesh (Damma)
    "\u0651",  # Tashdeed (Gamination)
    "\u0652",  # Sukun (Jazm)
    "\u0654",  # Hamza Above
    "\u0655",  # Hamza Below
    "\u0656",  # Subscript Alef
    "\u0657",  # Inverted Damma
    "\u0670",  # Superscript Alef (Khari Zabar)
    "\u065F",  # Kashmiri wavy diacritic
]

# Vocabulary lookup tables
CHAR_TO_ID: Dict[str, int] = {char: idx for idx, char in enumerate(KASHMIRI_ALPHABET)}
ID_TO_CHAR: Dict[int, str] = {idx: char for idx, char in enumerate(KASHMIRI_ALPHABET)}
VOCAB_SIZE = len(KASHMIRI_ALPHABET)


def expand_number_to_kashmiri(num: int) -> str:
    """Recursively converts an integer to spoken Kashmiri words."""
    if num in KASHMIRI_NUMBERS:
        return KASHMIRI_NUMBERS[num]
    if num < 30:
        return f"{KASHMIRI_NUMBERS.get(num % 10, '')} تہٕ {KASHMIRI_NUMBERS[20]}"
    if num < 100:
        tens = (num // 10) * 10
        remainder = num % 10
        if remainder == 0:
            return KASHMIRI_NUMBERS.get(tens, str(num))
        return f"{KASHMIRI_NUMBERS.get(remainder, '')} تہٕ {KASHMIRI_NUMBERS.get(tens, '')}"
    if num < 1000:
        hundreds = num // 100
        remainder = num % 100
        prefix = f"{KASHMIRI_NUMBERS.get(hundreds, '')} {KASHMIRI_NUMBERS[100]}"
        if remainder == 0:
            return prefix
        return f"{prefix} {expand_number_to_kashmiri(remainder)}"
    return str(num)


def normalize_kashmiri_text(text: str) -> str:
    """
    Complete normalization pipeline for Kashmiri Perso-Arabic text:
    1. Unicode NFC normalization
    2. Conversion of Indic/Eastern Arabic digits to ASCII, then expansion to Kashmiri words
    3. Canonicalization of character glyphs (Yeh, Kaf, Teh Marbuta)
    4. Preservation of critical Kashmiri vowels (ٲ, ۄ, ۆ) and palatalization (ؠ)
    5. Clean punctuation spacing for TTS prosodic phrasing
    """
    if not text:
        return ""

    # Step 1: Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Step 2: Convert Eastern Arabic digits to standard digits
    for arabic_digit, ascii_digit in DIGIT_MAP.items():
        text = text.replace(arabic_digit, ascii_digit)

    # Step 3: Expand digit sequences to spoken Kashmiri words
    def _replace_num(match):
        val = int(match.group(0))
        return expand_number_to_kashmiri(val)

    text = re.sub(r'\b\d+\b', _replace_num, text)

    # Step 4: Canonicalize character representations
    for src, tgt in CANONICAL_CHAR_MAP.items():
        text = text.replace(src, tgt)

    # Step 5: Clean superfluous whitespaces while retaining prosodic pauses
    text = re.sub(r'[\t\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Clean non-standard characters while preserving Kashmiri inventory
    allowed_chars = set(KASHMIRI_ALPHABET)
    cleaned_chars = [ch if ch in allowed_chars else ' ' for ch in text]
    text = "".join(cleaned_chars)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def text_to_sequence(text: str) -> List[int]:
    """Converts normalized Kashmiri text to a sequence of token IDs."""
    normalized = normalize_kashmiri_text(text)
    unk_id = CHAR_TO_ID.get("<unk>", 1)
    sequence = [CHAR_TO_ID.get(ch, unk_id) for ch in normalized]
    return sequence


def sequence_to_text(sequence: List[int]) -> str:
    """Converts a sequence of token IDs back into readable Kashmiri text."""
    chars = [ID_TO_CHAR.get(idx, "") for idx in sequence if idx not in [0, 1, 2, 3]]
    return "".join(chars)
