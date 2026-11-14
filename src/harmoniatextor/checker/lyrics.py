# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Checks for sung lyrics: text, IPA and language tags."""

from __future__ import annotations

from collections.abc import Mapping

from music21 import stream

from harmoniatextor.domain.enums import Severity
from harmoniatextor.domain.models import CheckViolation
from harmoniatextor.score.lyrics import lyric_issues

__all__ = ["lyric_violations"]

_MESSAGES: dict[str, str] = {
    "missing_lyric": "this sung note has no syllable",
    "missing_ipa": "this syllable has no IPA pronunciation",
    "invalid_ipa": "the IPA contains unsupported characters",
    "missing_language": "this vocal part has no language tag",
}


def lyric_violations(
    score: stream.Score, vocal_voices: Mapping[str, str]
) -> list[CheckViolation]:
    """Report lyric problems for every sung voice.

    A sung note without a syllable (or a syllable without IPA/language) is an
    error, not a warning: the vocal renderer would otherwise have to guess a
    vowel, and a movement that is accepted with missing words sounds like a
    wordless vocalise.  The composer loop therefore has to complete the lyrics
    before it can finish a vocal movement.

    Args:
        score: Score to inspect.
        vocal_voices: Voices that are sung, mapped to language tags.

    Returns:
        Error-severity lyric violations.
    """
    result: list[CheckViolation] = []
    for voice in vocal_voices:
        part = next(
            (item for item in score.parts if str(item.id or item.partName) == voice),
            None,
        )
        if part is None:
            continue
        for measure, kind, detail in lyric_issues(part):
            message = f"m. {measure}: {_MESSAGES[kind]}"
            if kind == "invalid_ipa" and detail:
                message += f" ({detail})"
            result.append(
                CheckViolation(
                    rule_id="lyrics",
                    severity=Severity.ERROR,
                    measure=measure,
                    voice_a=voice,
                    voice_b=None,
                    kind=kind,
                    message=message + ".",
                    snippet=detail,
                )
            )
    return result
