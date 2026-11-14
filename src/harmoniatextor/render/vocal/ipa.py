# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""IPA phoneme tables and tokenisation for the built-in singer.

The tables are deliberately broad: any language can be sung as long as its
pronunciation is written in the International Phonetic Alphabet.  Unknown
symbols fall back to a neutral schwa, so the singer never crashes on exotic
input and never renders a syllable as digital silence.

Real-world IPA coming from a language model is often imperfect.  Tone letters
(``˥˦˧˨˩``), ASCII tone digits, stress marks and precomposed tone-bearing Latin
letters (``ǐ``, ``ǎ``, ``ǒ``, ...) are therefore normalised before
classification so that pinyin-style input still carries a vowel.
"""

from __future__ import annotations

import unicodedata

__all__ = ["VOWELS", "base_symbol", "consonant_kind", "tokenize"]

#: Approximate adult formants ``(F1, F2, F3)`` in Hz for IPA vowels.
VOWELS: dict[str, tuple[float, float, float]] = {
    "i": (270.0, 2290.0, 3010.0),
    "y": (280.0, 1820.0, 2200.0),
    "ɨ": (300.0, 1400.0, 2400.0),
    "ʉ": (300.0, 1400.0, 2400.0),
    "ɯ": (300.0, 1300.0, 2200.0),
    "u": (300.0, 870.0, 2240.0),
    "ɪ": (390.0, 1990.0, 2550.0),
    "ʏ": (400.0, 1700.0, 2200.0),
    "ʊ": (440.0, 1020.0, 2240.0),
    "e": (400.0, 2100.0, 2700.0),
    "ø": (400.0, 1650.0, 2200.0),
    "ɘ": (400.0, 1600.0, 2500.0),
    "ɵ": (430.0, 1400.0, 2300.0),
    "ɤ": (430.0, 1200.0, 2300.0),
    "o": (420.0, 830.0, 2400.0),
    "ə": (500.0, 1500.0, 2500.0),
    "ɚ": (500.0, 1400.0, 1600.0),
    "ɝ": (500.0, 1350.0, 1600.0),
    "ɛ": (550.0, 1770.0, 2490.0),
    "œ": (550.0, 1450.0, 2200.0),
    "ɜ": (550.0, 1350.0, 2500.0),
    "ɞ": (550.0, 1200.0, 2400.0),
    "ʌ": (640.0, 1190.0, 2390.0),
    "ɔ": (570.0, 840.0, 2410.0),
    "æ": (660.0, 1720.0, 2410.0),
    "ɐ": (700.0, 1300.0, 2400.0),
    "a": (730.0, 1090.0, 2440.0),
    "ɶ": (750.0, 1650.0, 2400.0),
    "ɑ": (750.0, 940.0, 2400.0),
    "ɒ": (700.0, 880.0, 2400.0),
}

_STOPS = frozenset("pbtdkɡgʔʈɖɟɢʡ")
_FRICATIVES = frozenset("fvszʃʒθðxɣhçʝʂʐɕʑɸβχʁħʕɬ")
_NASALS = frozenset("mnŋɲɳɴɱ")
_LIQUIDS = frozenset("lrɹɾɬɮʎjwɥʋɻɰɭɺ")
_AFFRICATES = frozenset({"tʃ", "dʒ", "ts", "dz", "ʈʂ", "ɖʐ", "tɕ", "dʑ", "tɬ"})
#: Length marks stay attached to their base symbol.
_MODIFIER_CHARS = "ːˑ̯"
_MODIFIERS = frozenset(_MODIFIER_CHARS)
#: Tone letters and ASCII tone digits are prosody, not phonemes.
_TONE_LETTERS = frozenset("˥˦˧˨˩")
_DIGITS = frozenset("0123456789")
_IGNORED = frozenset("ˈˌ.-–—") | _TONE_LETTERS | _DIGITS


def tokenize(ipa: str) -> list[str]:
    """Split an IPA string into phoneme tokens.

    Combining marks and length marks stay attached to their base symbol, so
    ``t͡ʃ`` and ``uː`` remain single units.  Tone letters and tone digits are
    dropped because they carry no pronunciation.

    Args:
        ipa: IPA pronunciation.

    Returns:
        The phoneme tokens in order.
    """
    tokens: list[str] = []
    current = ""
    for char in unicodedata.normalize("NFD", ipa):
        if char.isspace() or char in _IGNORED:
            if current:
                tokens.append(current)
                current = ""
            continue
        if unicodedata.combining(char) or char in _MODIFIERS:
            if current:
                current += char
            continue
        if current:
            tokens.append(current)
        current = char
    if current:
        tokens.append(current)
    return tokens


def base_symbol(token: str) -> str:
    """Return the bare phoneme symbol of a token.

    Ties, length marks and any other combining diacritic are removed, so
    ``ǐ``, ``iː`` and ``ǐ`` all reduce to ``i``.

    Args:
        token: A token from :func:`tokenize`.

    Returns:
        The base phoneme symbol, possibly empty.
    """
    decomposed = unicodedata.normalize("NFD", token)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return stripped.strip(_MODIFIER_CHARS)


def consonant_kind(token: str) -> str:
    """Classify one phoneme token.

    Combining diacritics are ignored for classification, so a tone-marked
    vowel such as ``ǐ`` is still recognised as a vowel.

    Args:
        token: A token from :func:`tokenize`.

    Returns:
        ``vowel``, ``stop``, ``fricative``, ``nasal``, ``liquid``,
        ``affricate`` or ``unknown``.
    """
    base = base_symbol(token)
    if base in VOWELS:
        return "vowel"
    if base in _AFFRICATES:
        return "affricate"
    first = base[0] if base else ""
    if first in _STOPS:
        return "stop"
    if first in _FRICATIVES:
        return "fricative"
    if first in _NASALS:
        return "nasal"
    if first in _LIQUIDS:
        return "liquid"
    return "unknown"
