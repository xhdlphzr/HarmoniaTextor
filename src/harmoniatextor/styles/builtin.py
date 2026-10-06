# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""The built-in style kits and the full library preset."""

from __future__ import annotations

from harmoniatextor.checker.rules import BUILTIN_RULES
from harmoniatextor.styles.base import StyleKit
from harmoniatextor.techniques.registry import build_default_registry

__all__ = [
    "ALL_RULE_IDS",
    "ALL_TECHNIQUE_IDS",
    "BUILTIN_KITS",
    "DEFAULT_STYLE_ID",
    "FULL_STYLE_ID",
    "STRUCTURAL_TECHNIQUES",
]

ALL_RULE_IDS: frozenset[str] = frozenset(rule.rule_id for rule in BUILTIN_RULES)
ALL_TECHNIQUE_IDS: frozenset[str] = frozenset(build_default_registry().ids())

DEFAULT_STYLE_ID = "baroque"
FULL_STYLE_ID = "full"

_MELODIC = frozenset(
    {
        "imitation",
        "inversion",
        "retrograde",
        "augmentation",
        "diminution",
        "transposition",
        "sequence",
        "voice_exchange",
    }
)
_MELODIC_NO_RETRO = _MELODIC - {"retrograde"}
_HARMONIC = frozenset(
    {
        "functional_cycle",
        "dominant_seventh",
        "diminished_seventh",
        "harmonic_sequence",
        "chromatic_harmony",
        "modulation_bridge",
    }
)
_HARMONIC_NO_FUNCTIONAL = _HARMONIC - {"functional_cycle"}
_RHYTHMIC = frozenset(
    {"syncopation", "rhythmic_independence", "counter_rhythm", "voice_motion"}
)
_NEW = frozenset(
    {
        "alberti_bass",
        "broken_chord",
        "chromatic_modulation",
        "extended_harmony",
        "rubato",
        "whole_tone",
        "parallel_chords",
        "modal_harmony",
        "color_chord",
        "planing",
    }
)

#: Structural techniques follow the genre rather than the style, so every
#: built-in kit keeps them all.
STRUCTURAL_TECHNIQUES: frozenset[str] = frozenset(
    {
        "exposition",
        "development",
        "recapitulation",
        "rondo",
        "stretto",
        "pedal_point",
        "pedal_tone",
    }
)

_BAROQUE_TECHNIQUES = (
    _MELODIC
    | _HARMONIC
    | _RHYTHMIC
    | STRUCTURAL_TECHNIQUES
    | {
        "broken_chord",
        "chromatic_modulation",
        "rubato",
        "modal_harmony",
        "color_chord",
    }
)
_CLASSICAL_TECHNIQUES = (
    _MELODIC_NO_RETRO
    | _HARMONIC
    | _RHYTHMIC
    | STRUCTURAL_TECHNIQUES
    | {
        "alberti_bass",
        "broken_chord",
        "chromatic_modulation",
        "extended_harmony",
        "rubato",
        "modal_harmony",
        "color_chord",
    }
)
_ROMANTIC_TECHNIQUES = (
    _MELODIC_NO_RETRO
    | _HARMONIC
    | _RHYTHMIC
    | STRUCTURAL_TECHNIQUES
    | {
        "alberti_bass",
        "broken_chord",
        "chromatic_modulation",
        "extended_harmony",
        "rubato",
        "whole_tone",
        "parallel_chords",
        "modal_harmony",
        "color_chord",
        "free_voice_leading",
    }
)
_IMPRESSION_TECHNIQUES = (
    _MELODIC_NO_RETRO | _HARMONIC_NO_FUNCTIONAL | _RHYTHMIC | STRUCTURAL_TECHNIQUES
) | {
    "broken_chord",
    "chromatic_modulation",
    "extended_harmony",
    "rubato",
    "whole_tone",
    "parallel_chords",
    "modal_harmony",
    "color_chord",
    "planing",
    "free_voice_leading",
}

_CLASSICAL_RULES = ALL_RULE_IDS - {"hf5th", "ho8ve", "omission"}
_ROMANTIC_RULES = frozenset(
    {"empty", "pf5th", "po8ve", "dom7res", "tonality", "voices"}
)
_IMPRESSION_RULES = frozenset({"empty", "voices"})

BUILTIN_KITS: dict[str, StyleKit] = {
    "full": StyleKit(
        id="full",
        name="全集",
        brief=(
            "The complete technique set with all strict rules; no stylistic "
            "lean; the strictest setting."
        ),
        rules=ALL_RULE_IDS,
        techniques=ALL_TECHNIQUE_IDS,
        builtin=True,
    ),
    "baroque": StyleKit(
        id="baroque",
        name="巴洛克",
        brief=(
            "Strict four-part counterpoint at its core: highly unified motivic "
            "material, with imitation, sequence, inversion, augmentation/"
            "diminution and stretto throughout; clear harmonic function and "
            "cadences, and strict observance of parallel fifth/octave, hidden "
            "and voice-leading rules."
        ),
        rules=ALL_RULE_IDS,
        techniques=frozenset(_BAROQUE_TECHNIQUES),
        builtin=True,
    ),
    "classical": StyleKit(
        id="classical",
        name="古典主义",
        brief=(
            "Clear, balanced phrases and forms (sonata, rondo, ...); elegant, "
            "symmetrical melodies and mainly tonic-dominant harmony; freer inner "
            "voices than Baroque, but with clear tonality and cadences."
        ),
        rules=_CLASSICAL_RULES,
        techniques=frozenset(_CLASSICAL_TECHNIQUES),
        builtin=True,
    ),
    "romantic": StyleKit(
        id="romantic",
        name="浪漫主义",
        brief=(
            "Expression and emotional tension first: broad singing melodies, "
            "colourful harmony, frequent modulation and thick textures; parallel "
            "fifths/octaves, free voice leading, extended harmony, whole-tone and "
            "chromatic writing are allowed, and strict counterpoint is relaxed."
        ),
        rules=_ROMANTIC_RULES,
        techniques=frozenset(_ROMANTIC_TECHNIQUES),
        builtin=True,
    ),
    "impressionist": StyleKit(
        id="impressionist",
        name="印象派",
        brief=(
            "Timbre, light and colour lead: parallel chords, planing, whole-tone, "
            "extended and colour harmony; avoids traditional functional "
            "progressions and fixed cadences, seeking blurred tonality and "
            "flowing textures."
        ),
        rules=_IMPRESSION_RULES,
        techniques=frozenset(_IMPRESSION_TECHNIQUES),
        builtin=True,
    ),
}
