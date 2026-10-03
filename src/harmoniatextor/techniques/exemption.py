# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""The free-voice-leading rule exemption technique.

This is a callable technique, but it transforms no music: invoking it records a
scoped waiver so that the symbolic checker ignores the strict voice-leading
rules for one voice over a measure range.  It exists so that expressive styles
can relax counterpoint deliberately, rather than globally disabling rules.
"""

from __future__ import annotations

from harmoniatextor.domain.params import FreeVoiceLeadingParams
from harmoniatextor.techniques.base import (
    Technique,
    TechniqueCategory,
    TechniqueContext,
    TechniqueResult,
)

__all__ = ["FREE_VOICE_LEADING_EXEMPT", "FreeVoiceLeadingTechnique"]

#: Rules waived by :class:`FreeVoiceLeadingTechnique` within its scope.
FREE_VOICE_LEADING_EXEMPT: frozenset[str] = frozenset(
    {
        "hf5th",
        "ho8ve",
        "crossing",
        "spacing",
        "final_outer",
        "leading",
        "omission",
        "diminterval",
    }
)


class FreeVoiceLeadingTechnique(Technique[FreeVoiceLeadingParams]):
    """Waive the strict voice-leading rules for one voice over a measure range."""

    id = "free_voice_leading"
    name = "自由声部进行"
    category = TechniqueCategory.RHYTHMIC
    summary = (
        "Exempt one voice's measures from the strict voice-leading rules "
        "(hidden fifth/octave, crossing, spacing, final outer interval, leading tone, "
        "leading-tone doubling and diminished leaps). Only call this when free voice "
        "leading genuinely improves the music, never to avoid meeting the rules."
    )
    params_model = FreeVoiceLeadingParams
    exempts = FREE_VOICE_LEADING_EXEMPT

    def apply(
        self, ctx: TechniqueContext, params: FreeVoiceLeadingParams
    ) -> TechniqueResult:
        """Return the score unchanged, noting the exempted scope."""
        note = (
            f"已豁免 {params.voice} 第 {params.measure_range.start}-"
            f"{params.measure_range.end} 小节的声部进行规则。理由:{params.reason}"
        )
        return TechniqueResult(ctx.score, [note])
