# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""music21 bridge: import/export, editing and flattening helpers."""

from __future__ import annotations

from harmoniatextor.score.analysis import (
    VoiceEvent,
    measure_of,
    onsets,
    voice_events,
    voice_order,
)
from harmoniatextor.score.io import (
    clone_score,
    from_musicxml,
    new_part,
    new_score,
    to_musicxml,
)
from harmoniatextor.score.streamops import ScoreEditor

__all__ = [
    "ScoreEditor",
    "VoiceEvent",
    "clone_score",
    "from_musicxml",
    "measure_of",
    "new_part",
    "new_score",
    "onsets",
    "to_musicxml",
    "voice_events",
    "voice_order",
]
