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
_RHYTHMIC = frozenset({"syncopation", "rhythmic_independence", "counter_rhythm", "voice_motion"})
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
_ROMANTIC_RULES = frozenset({"empty", "pf5th", "po8ve", "dom7res", "tonality", "voices"})
_IMPRESSION_RULES = frozenset({"empty", "voices"})

BUILTIN_KITS: dict[str, StyleKit] = {
    "full": StyleKit(
        id="full",
        name="全集",
        brief="完整技法与全部严格规则;不预设任何风格倾向,严格程度最高。",
        rules=ALL_RULE_IDS,
        techniques=ALL_TECHNIQUE_IDS,
        builtin=True,
    ),
    "baroque": StyleKit(
        id="baroque",
        name="巴洛克",
        brief=(
            "以严格四部对位为核心:主题动机高度统一,模仿、模进、倒影、扩缩、密接和应贯穿全曲;"
            "和声功能清晰、终止明确,纵向严格遵守平行五/八度、隐伏与声部进行等规则。"
        ),
        rules=ALL_RULE_IDS,
        techniques=frozenset(_BAROQUE_TECHNIQUES),
        builtin=True,
    ),
    "classical": StyleKit(
        id="classical",
        name="古典主义",
        brief=(
            "强调清晰均衡的乐句与曲式(呈示—展开—再现、回旋等);旋律优雅对称,"
            "和声以主属功能为主;内声部比巴洛克更自由,但保持明确调性与终止。"
        ),
        rules=_CLASSICAL_RULES,
        techniques=frozenset(_CLASSICAL_TECHNIQUES),
        builtin=True,
    ),
    "romantic": StyleKit(
        id="romantic",
        name="浪漫主义",
        brief=(
            "以强烈表情与情感张力为先;旋律宽广歌唱,和声色彩化、频繁转调、织体厚重;"
            "允许平行五/八度、自由声部进行、扩展和声、全音阶与半音化,弱化严格对位。"
        ),
        rules=_ROMANTIC_RULES,
        techniques=frozenset(_ROMANTIC_TECHNIQUES),
        builtin=True,
    ),
    "impressionist": StyleKit(
        id="impressionist",
        name="印象派",
        brief=(
            "以音色、光影与色彩为主导;强调平行和弦、平行进行、全音阶、扩展与色彩和声,"
            "回避传统功能进行与固定终止,追求模糊调性与流动织体。"
        ),
        rules=_IMPRESSION_RULES,
        techniques=frozenset(_IMPRESSION_TECHNIQUES),
        builtin=True,
    ),
}
