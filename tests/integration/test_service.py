# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for the composition service."""

from __future__ import annotations

import pytest

from harmoniatextor.domain.enums import Severity, WorkStatus
from harmoniatextor.domain.models import CheckReport, CheckViolation, ThemeNote
from harmoniatextor.score.io import new_score, to_musicxml
from harmoniatextor.score.streamops import ScoreEditor
from harmoniatextor.service.service import (
    DEFAULT_TITLE,
    CompositionService,
    _filter_exemptions,
    _in_exempt_scope,
)

_SYMPHONY_MOVEMENTS = 4
_FAST_TEMPO = 100
_SLOW_TEMPO = 72
_INTERRUPTED_WORKS = 2


def melody_xml(notes: list[tuple[str, float]], voice: str = "soprano") -> str:
    """Build a single-voice melody as MusicXML."""
    score = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=[voice])
    ScoreEditor(score).write_line(voice, 1, [ThemeNote(pitch, length) for pitch, length in notes])
    return to_musicxml(score)


def cadence_xml() -> str:
    """Build a four-voice authentic cadence that passes every rule."""
    score = new_score(
        key="C",
        time_signature="4/4",
        tempo_bpm=84,
        voices=["soprano", "alto", "tenor", "bass"],
    )
    editor = ScoreEditor(score)
    editor.write_line("soprano", 1, [ThemeNote("D5", 4.0), ThemeNote("C5", 4.0)])
    editor.write_line("alto", 1, [ThemeNote("B4", 4.0), ThemeNote("C5", 4.0)])
    editor.write_line("tenor", 1, [ThemeNote("G4", 4.0), ThemeNote("E4", 4.0)])
    editor.write_line("bass", 1, [ThemeNote("G3", 4.0), ThemeNote("C4", 4.0)])
    return to_musicxml(score)


class TestCreation:
    """Work creation and lookup."""

    def test_create_work(self, service: CompositionService) -> None:
        """Creating a work yields movements and an initial revision."""
        work = service.create_work("Demo", "plain", "C")
        assert work.id in service.list_works()
        assert service.get_work(work.id).title == "Demo"
        assert service.current_score(work.id, work.movements[0].id) is not None

    def test_get_movement_missing(self, service: CompositionService) -> None:
        """Unknown movements raise KeyError."""
        work = service.create_work("Demo", "plain", "C")
        try:
            service.get_movement(work, "m99")
        except KeyError:
            return
        raise AssertionError("expected KeyError")

    def test_set_title(self, service: CompositionService) -> None:
        """The title tool updates the work."""
        work = service.create_work("Demo", "plain", "C")
        result = service.set_title(work.id, "新标题")
        assert result.ok
        assert service.get_work(work.id).title == "新标题"

    def test_ensure_title_keeps_existing(self, service: CompositionService) -> None:
        """An existing title is preserved."""
        work = service.create_work("Demo", "plain", "C")
        service.set_title(work.id, "已命名")
        assert service.ensure_title(work.id) == "已命名"

    def test_ensure_title_defaults_to_genre(self, service: CompositionService) -> None:
        """A default title falls back to the genre name."""
        work = service.create_work(DEFAULT_TITLE, "plain", "C")
        expected = service.genres.get("plain").display_name
        assert service.ensure_title(work.id) == expected
        assert service.get_work(work.id).title == expected


class TestSubmitTheme:
    """Theme submission."""

    def test_success(self, service: CompositionService) -> None:
        """A clean theme is registered with an increasing id."""
        work = service.create_work("Demo", "plain", "C")
        movement = work.movements[0]
        result = service.submit_theme(
            work.id, movement.id, melody_xml([("C5", 1.0), ("D5", 1.0), ("E5", 1.0), ("F5", 1.0)])
        )
        assert result.ok
        assert result.theme_id == 1
        assert result.full_musicxml is not None

    def test_first_theme_starts_at_first_measure(self, service: CompositionService) -> None:
        """The first theme fills measure 1 instead of leaving an empty opening."""
        work = service.create_work("Demo", "plain", "C")
        movement = work.movements[0]
        result = service.submit_theme(work.id, movement.id, melody_xml([("C5", 1.0)]))
        assert result.ok
        theme = service.store.load_themes(work.id, movement.id)[1]
        assert theme.start_measure == 1

    def test_bad_xml(self, service: CompositionService) -> None:
        """Malformed MusicXML is rejected."""
        work = service.create_work("Demo", "plain", "C")
        result = service.submit_theme(work.id, work.movements[0].id, "not xml <<<")
        assert not result.ok
        assert result.error_code == "BAD_PARAM"

    def test_empty_theme(self, service: CompositionService) -> None:
        """A theme without notes is rejected."""
        work = service.create_work("Demo", "plain", "C")
        empty = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=["soprano"])
        result = service.submit_theme(work.id, work.movements[0].id, to_musicxml(empty))
        assert not result.ok
        assert result.error_code == "BAD_PARAM"

    def test_check_failure(self, service: CompositionService) -> None:
        """A non-compliant theme fails the symbolic check."""
        work = service.create_work("Demo", "plain", "C")
        result = service.submit_theme(
            work.id,
            work.movements[0].id,
            melody_xml([("B4", 1.0), ("G4", 1.0)]),
        )
        assert not result.ok
        assert result.report is not None

    def test_deferred_check(self, service: CompositionService) -> None:
        """Deferring the check skips the report but still records the theme."""
        work = service.create_work("Demo", "plain", "C")
        result = service.submit_theme(
            work.id,
            work.movements[0].id,
            melody_xml([("B4", 1.0), ("G4", 1.0)]),
            check=False,
        )
        assert result.ok
        assert result.report is None
        assert result.theme_id == 1

    def test_choose_major_key(self, service: CompositionService) -> None:
        """The composer sets the major key when submitting the first theme."""
        work = service.create_work("Demo", "plain")
        result = service.submit_theme(
            work.id,
            work.movements[0].id,
            melody_xml([("C5", 1.0), ("D5", 1.0), ("E5", 1.0), ("F5", 1.0)]),
            key="G",
        )
        assert result.ok
        reloaded = service.get_work(work.id)
        assert reloaded.tonic.raw == "G"
        assert reloaded.movements[0].key.raw == "G"
        assert reloaded.movements[0].key.is_major

    def test_choose_minor_key(self, service: CompositionService) -> None:
        """A lower-case key selects the minor mode."""
        work = service.create_work("Demo", "plain")
        result = service.submit_theme(
            work.id,
            work.movements[0].id,
            melody_xml([("C5", 1.0), ("D5", 1.0), ("E5", 1.0), ("F5", 1.0)]),
            key="a",
        )
        assert result.ok
        reloaded = service.get_work(work.id)
        assert reloaded.movements[0].key.raw == "a"
        assert not reloaded.movements[0].key.is_major

    def test_bad_key(self, service: CompositionService) -> None:
        """An invalid key is rejected before anything is written."""
        work = service.create_work("Demo", "plain")
        result = service.submit_theme(
            work.id, work.movements[0].id, melody_xml([("C5", 1.0)]), key="H"
        )
        assert not result.ok
        assert result.error_code == "BAD_PARAM"

    def test_change_key_retunes_existing_parts(self, service: CompositionService) -> None:
        """A later theme can change the key and retune the existing parts."""
        work = service.create_work("Demo", "plain", "C")
        movement = work.movements[0]
        service.submit_theme(work.id, movement.id, melody_xml([("C5", 1.0)]), key="C", check=False)
        service.submit_theme(work.id, movement.id, melody_xml([("D5", 1.0)]), key="G", check=False)
        score = service.current_score(work.id, movement.id)
        keys = list(score.recurse().getElementsByClass("Key"))
        assert keys
        assert all(item.tonic.name == "G" for item in keys)


class TestApplyTechnique:
    """Technique application."""

    def test_success(self, service: CompositionService) -> None:
        """A valid technique produces a full score."""
        work = service.create_work("Demo", "plain", "C")
        movement = work.movements[0]
        theme = service.submit_theme(work.id, movement.id, melody_xml([("C5", 1.0), ("D5", 1.0)]))
        result = service.apply_technique(
            work.id,
            movement.id,
            "imitation",
            {
                "theme_id": theme.theme_id,
                "target_voice": "bass",
                "delay_measures": 1,
                "interval": 5,
            },
        )
        assert result.ok
        assert result.full_musicxml is not None

    def test_check_failure(self, service: CompositionService) -> None:
        """A technique whose result violates the rules is rejected."""
        work = service.create_work("Demo", "plain", "C")
        movement = work.movements[0]
        theme = service.submit_theme(
            work.id, movement.id, melody_xml([("C5", 1.0), ("D5", 1.0), ("E5", 1.0), ("F5", 1.0)])
        )
        result = service.apply_technique(
            work.id,
            movement.id,
            "imitation",
            {
                "theme_id": theme.theme_id,
                "target_voice": "alto",
                "delay_measures": 0,
                "interval": 5,
            },
        )
        assert not result.ok
        assert result.report is not None

    def test_unknown_technique(self, service: CompositionService) -> None:
        """Unknown techniques are rejected."""
        work = service.create_work("Demo", "plain", "C")
        result = service.apply_technique(work.id, work.movements[0].id, "nope", {})
        assert result.error_code == "BAD_PARAM"

    def test_invalid_params(self, service: CompositionService) -> None:
        """Invalid parameters are rejected."""
        work = service.create_work("Demo", "plain", "C")
        result = service.apply_technique(work.id, work.movements[0].id, "imitation", {})
        assert result.error_code == "BAD_PARAM"

    def test_missing_theme(self, service: CompositionService) -> None:
        """A missing theme raises a technique error."""
        work = service.create_work("Demo", "plain", "C")
        result = service.apply_technique(
            work.id,
            work.movements[0].id,
            "imitation",
            {"theme_id": 99, "target_voice": "bass"},
        )
        assert result.error_code == "THEME_NOT_FOUND"

    def _two_movements(self, service: CompositionService) -> tuple[str, str, str]:
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.add_movement(work.id)
        first, second = service.get_work(work.id).movements
        return work.id, first.id, second.id

    def test_themes_visible_across_movements(self, service: CompositionService) -> None:
        """Later movements can reference earlier movements' themes."""
        work_id, first_id, second_id = self._two_movements(service)
        first = service.submit_theme(work_id, first_id, melody_xml([("C5", 1.0)]))
        second = service.submit_theme(work_id, first_id, melody_xml([("D5", 1.0)]))
        own = service.submit_theme(work_id, second_id, melody_xml([("E5", 1.0)]))
        assert [first.theme_id, second.theme_id, own.theme_id] == [1, 2, 3]
        visible = service._themes_for(service.get_work(work_id), second_id)
        assert set(visible) == {1, 2, 3}
        assert visible[2].start_measure == 1
        assert service.store.load_themes(work_id, first_id)[2].start_measure > 1

    def test_cross_movement_imitation(self, service: CompositionService) -> None:
        """A movement can apply a technique to an earlier movement's theme."""
        work_id, first_id, second_id = self._two_movements(service)
        theme = service.submit_theme(work_id, first_id, melody_xml([("C5", 1.0), ("D5", 1.0)]))
        result = service.apply_technique(
            work_id,
            second_id,
            "imitation",
            {
                "theme_id": theme.theme_id,
                "target_voice": "alto",
                "delay_measures": 0,
                "interval": 5,
            },
            check=False,
        )
        assert result.ok
        assert result.full_musicxml is not None


class TestCheckAndFinalize:
    """Checking, finalisation and rollback."""

    def test_check(self, service: CompositionService) -> None:
        """An empty score is rejected by the symbolic layer."""
        work = service.create_work("Demo", "plain", "C")
        report = service.check(work.id, work.movements[0].id)
        assert not report.ok
        assert "empty" in [item.rule_id for item in report.errors]

    def test_finalize_failure(self, service: CompositionService) -> None:
        """A draft without a cadence cannot be finalised."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        service.edit_measure(
            work.id, movement_id, 1, "soprano", melody_xml([("C5", 1.0), ("D5", 1.0)]), check=False
        )
        assert not service.finalize(work.id, movement_id).ok

    def test_finalize_success(self, service: CompositionService) -> None:
        """A proper cadence can be finalised."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        lines = {
            "soprano": [("D5", 4.0), ("C5", 4.0)],
            "alto": [("B4", 4.0), ("C5", 4.0)],
            "tenor": [("G4", 4.0), ("E4", 4.0)],
            "bass": [("G3", 4.0), ("C4", 4.0)],
        }
        for voice, notes in lines.items():
            service.edit_measure(
                work.id, movement_id, 1, voice, melody_xml(notes, voice), check=False
            )
        result = service.finalize(work.id, movement_id)
        assert result.ok
        assert service.get_work(work.id).status is WorkStatus.FINAL

    def test_rollback(self, service: CompositionService) -> None:
        """Rolling back restores an earlier revision."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        service.edit_measure(
            work.id, movement_id, 1, "soprano", melody_xml([("C5", 1.0)]), check=False
        )
        result = service.rollback(work.id, movement_id, 0)
        assert result.ok
        assert result.full_musicxml is not None

    def test_rollback_changes_current_score(self, service: CompositionService) -> None:
        """Rolling back makes the earlier revision the authoritative score."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        service.edit_measure(
            work.id, movement_id, 1, "soprano", melody_xml([("C5", 1.0)]), check=False
        )
        service.edit_measure(
            work.id, movement_id, 1, "soprano", melody_xml([("G5", 1.0)]), check=False
        )
        service.rollback(work.id, movement_id, 0)
        current = service.current_score(work.id, movement_id)
        assert [note.nameWithOctave for note in current.recurse().notes] == ["C5"]

    def test_rollback_missing(self, service: CompositionService) -> None:
        """Rolling back to a missing revision fails."""
        work = service.create_work("Demo", "plain", "C")
        assert service.rollback(work.id, work.movements[0].id, 99).error_code == "BAD_PARAM"

    def test_record_audit(self, service: CompositionService) -> None:
        """Audition notes are journalled and move the work to revising."""
        work = service.create_work("Demo", "plain", "C")
        service.record_audit(work.id, work.movements[0].id, "需要更多模仿")
        assert service.get_work(work.id).status is WorkStatus.REVISING
        assert any(item["event"] == "audit_note" for item in service.store.load_journal(work.id))

    def test_create_multi_movement(self, service: CompositionService) -> None:
        """Multi-movement genres create several movements."""
        work = service.create_work("Symphony", "symphony", "C")
        assert len(work.movements) == _SYMPHONY_MOVEMENTS
        assert service.current_score(work.id, work.movements[1].id) is not None


class TestInstruments:
    """Instrument-aware parts."""

    def test_submit_theme_with_instrument(self, service: CompositionService) -> None:
        """A theme can target a voice slot and instrument."""
        work = service.create_work("Demo", "symphony", "C")
        movement = work.movements[0]
        result = service.submit_theme(
            work.id,
            movement.id,
            melody_xml([("C5", 1.0), ("D5", 1.0)]),
            voice="violin1",
            instrument="Violin",
        )
        assert result.ok
        assert result.full_musicxml is not None
        assert "Violin" in result.full_musicxml
        themes = service.store.load_themes(work.id, movement.id)
        assert result.theme_id is not None
        assert themes[result.theme_id].voice == "violin1"

    def test_add_part_and_duplicate(self, service: CompositionService) -> None:
        """A new instrument part can be added once."""
        work = service.create_work("Demo", "symphony", "C")
        movement = work.movements[0]
        first = service.add_part(work.id, movement.id, "flute", "Flute", check=False)
        assert first.ok
        assert first.full_musicxml is not None
        duplicate = service.add_part(work.id, movement.id, "flute", "Flute")
        assert not duplicate.ok
        assert duplicate.error_code == "BAD_PARAM"

    def test_add_part_check_failure(
        self, service: CompositionService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A part that fails the symbolic check is rejected."""
        work = service.create_work("Demo", "symphony", "C")
        failing = CheckReport([CheckViolation("r", Severity.ERROR, 1, None, None, "k", "m", "s")])
        monkeypatch.setattr(service, "_check", lambda *_args, **_kwargs: failing)
        result = service.add_part(work.id, work.movements[0].id, "flute", "Flute")
        assert not result.ok
        assert result.report is failing

    def test_remove_part(self, service: CompositionService) -> None:
        """A voice can be removed, and a missing voice is rejected."""
        work = service.create_work("Demo", "symphony", "C")
        movement_id = work.movements[0].id
        service.add_part(work.id, movement_id, "flute", "Flute", check=False)
        assert service.remove_part(work.id, movement_id, "flute", check=False).ok
        assert (
            service.remove_part(work.id, movement_id, "flute", check=False).error_code
            == "BAD_PARAM"
        )

    def test_remove_part_check_failure(
        self, service: CompositionService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A removal that fails the symbolic check returns the report."""
        work = service.create_work("Demo", "symphony", "C")
        movement_id = work.movements[0].id
        service.add_part(work.id, movement_id, "flute", "Flute", check=False)
        failing = CheckReport([CheckViolation("r", Severity.ERROR, 1, None, None, "k", "m", "s")])
        monkeypatch.setattr(service, "_check", lambda *_args, **_kwargs: failing)
        result = service.remove_part(work.id, movement_id, "flute")
        assert not result.ok
        assert result.report is failing

    def test_set_tempo(self, service: CompositionService) -> None:
        """The movement tempo is updated and reaches the score."""
        work = service.create_work("Demo", "symphony", "C")
        movement_id = work.movements[0].id
        service.add_part(work.id, movement_id, "flute", "Flute", check=False)
        assert service.set_tempo(work.id, movement_id, _FAST_TEMPO, check=False).ok
        assert service.get_work(work.id).movements[0].tempo == _FAST_TEMPO
        assert f"<per-minute>{_FAST_TEMPO}</per-minute>" in service.current_musicxml(
            work.id, movement_id
        )

    def test_set_tempo_partless(self, service: CompositionService) -> None:
        """Tempo can be set before any voice exists."""
        work = service.create_work("Demo", "symphony", "C")
        movement_id = work.movements[0].id
        assert service.set_tempo(work.id, movement_id, _SLOW_TEMPO, check=False).ok
        assert service.get_work(work.id).movements[0].tempo == _SLOW_TEMPO

    def test_annotate_marks(self, service: CompositionService) -> None:
        """Every supported expressive mark can be added."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        service.submit_theme(
            work.id, movement_id, melody_xml([("C5", 1.0), ("D5", 1.0)]), check=False
        )
        marks = [
            ("dynamic", "f"),
            ("text", "dolce"),
            ("crescendo", ""),
            ("diminuendo", ""),
            ("accent", ""),
            ("tenuto", ""),
            ("staccato", ""),
            ("slur", ""),
            ("pedal", ""),
            ("tempo", "90"),
        ]
        for mark, value in marks:
            result = service.annotate(work.id, movement_id, 1, "soprano", mark, value, check=False)
            assert result.ok, mark
        xml = service.current_musicxml(work.id, movement_id)
        assert "dolce" in xml
        assert "<wedge" in xml
        assert "<slur" in xml
        assert "<pedal" in xml
        assert "accent" in xml
        assert "tenuto" in xml
        assert "staccato" in xml

    def test_annotate_errors(self, service: CompositionService) -> None:
        """Unknown voices, measures, marks and values are rejected."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        service.submit_theme(work.id, movement_id, melody_xml([("C5", 1.0)]), check=False)
        assert (
            service.annotate(
                work.id, movement_id, 1, "ghost", "dynamic", "f", check=False
            ).error_code
            == "BAD_PARAM"
        )
        assert (
            service.annotate(
                work.id, movement_id, 99, "soprano", "dynamic", "f", check=False
            ).error_code
            == "BAD_PARAM"
        )
        assert (
            service.annotate(work.id, movement_id, 1, "soprano", "nope", "", check=False).error_code
            == "BAD_PARAM"
        )
        assert (
            service.annotate(
                work.id, movement_id, 1, "soprano", "tempo", "abc", check=False
            ).error_code
            == "BAD_PARAM"
        )
        assert (
            service.annotate(work.id, movement_id, 1, "soprano", "slur", "", check=False).error_code
            == "BAD_PARAM"
        )

    def test_annotate_check_failure(
        self, service: CompositionService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mark that fails the symbolic check returns the report."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        service.submit_theme(work.id, movement_id, melody_xml([("C5", 1.0)]), check=False)
        failing = CheckReport([CheckViolation("r", Severity.ERROR, 1, None, None, "k", "m", "s")])
        monkeypatch.setattr(service, "_check", lambda *_args, **_kwargs: failing)
        result = service.annotate(work.id, movement_id, 1, "soprano", "dynamic", "f")
        assert not result.ok
        assert result.report is failing

    def test_set_tempo_check_failure(
        self, service: CompositionService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tempo change that fails the symbolic check returns the report."""
        work = service.create_work("Demo", "symphony", "C")
        movement_id = work.movements[0].id
        failing = CheckReport([CheckViolation("r", Severity.ERROR, 1, None, None, "k", "m", "s")])
        monkeypatch.setattr(service, "_check", lambda *_args, **_kwargs: failing)
        result = service.set_tempo(work.id, movement_id, _SLOW_TEMPO)
        assert not result.ok
        assert result.report is failing


class TestReviewJournal:
    """Reviewer verdict persistence."""

    def test_record_and_latest_review(self, service: CompositionService) -> None:
        """Reviewer verdicts are journaled and the latest can be read."""
        work = service.create_work("Demo", "plain")
        assert service.latest_review(work.id) is None
        service.record_review(work.id, work.movements[0].id, False, "问题")
        latest = service.latest_review(work.id)
        assert latest is not None
        assert latest["passed"] is False
        assert latest["suggestions"] == "问题"


class TestArchitecture:
    """Per-movement planning and merging."""

    def test_add_movement(self, service: CompositionService) -> None:
        """A movement is appended and numbered without any template."""
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        result = service.add_movement(work.id)
        assert result.ok
        assert result.movement_id == "m01"
        assert service.get_work(work.id).movements[0].name

    def test_edit_measure(self, service: CompositionService) -> None:
        """A measure is replaced for one voice and the full score is returned."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        service.submit_theme(work.id, movement_id, melody_xml([("C5", 1.0)]), check=False)
        result = service.edit_measure(
            work.id, movement_id, 1, "soprano", melody_xml([("G5", 1.0)]), check=False
        )
        assert result.ok
        assert result.full_musicxml is not None
        notes = [
            note.nameWithOctave
            for note in service.current_score(work.id, movement_id).recurse().notes
        ]
        assert "G5" in notes

    def test_edit_measure_errors(self, service: CompositionService) -> None:
        """Bad fragments are rejected; empty fragments clear; voices are created on demand."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        assert (
            service.edit_measure(work.id, movement_id, 1, "soprano", "nope").error_code
            == "BAD_PARAM"
        )
        created = service.edit_measure(
            work.id, movement_id, 1, "ghost", melody_xml([("C5", 1.0)]), check=False
        )
        assert created.ok
        cleared = service.edit_measure(
            work.id, movement_id, 1, "ghost", melody_xml([]), check=False
        )
        assert cleared.ok
        assert not list(service.current_score(work.id, movement_id).recurse().notes)
        assert (
            service.edit_measure(
                work.id, movement_id, 1, "nowhere", melody_xml([]), check=False
            ).error_code
            == "BAD_PARAM"
        )

    def test_edit_measure_failing_check(self, service: CompositionService) -> None:
        """An edit that introduces a violation returns the report."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        soprano = melody_xml([("C5", 1.0), ("D5", 1.0)])
        alto = melody_xml([("F4", 1.0), ("G4", 1.0)])
        service.edit_measure(work.id, movement_id, 1, "soprano", soprano, check=False)
        service.edit_measure(work.id, movement_id, 1, "alto", alto, check=False)
        result = service.edit_measure(work.id, movement_id, 1, "soprano", soprano)
        assert not result.ok
        assert result.report is not None

    def test_insert_measure(self, service: CompositionService) -> None:
        """A measure can be inserted in every voice."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        service.submit_theme(work.id, movement_id, melody_xml([("C5", 1.0)]), check=False)
        before = len(
            list(service.current_score(work.id, movement_id).parts[0].getElementsByClass("Measure"))
        )
        result = service.insert_measure(work.id, movement_id, 1, check=False)
        assert result.ok
        after = len(
            list(service.current_score(work.id, movement_id).parts[0].getElementsByClass("Measure"))
        )
        assert after == before + 1

    def test_insert_measure_filled(self, service: CompositionService) -> None:
        """An inserted measure can be filled in one voice."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        service.submit_theme(work.id, movement_id, melody_xml([("C5", 1.0)]), check=False)
        result = service.insert_measure(
            work.id,
            movement_id,
            1,
            voice="soprano",
            musicxml=melody_xml([("D5", 1.0)]),
            check=False,
        )
        assert result.ok
        score = service.current_score(work.id, movement_id)
        assert ScoreEditor(score).read_line("soprano", 1, 1) == [ThemeNote("D5", 1.0)]

    def test_insert_measure_errors(self, service: CompositionService) -> None:
        """Insertion validates the voice and the fragment."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        assert service.insert_measure(work.id, movement_id, 1).error_code == "BAD_PARAM"
        service.submit_theme(work.id, movement_id, melody_xml([("C5", 1.0)]), check=False)
        assert (
            service.insert_measure(
                work.id, movement_id, 1, musicxml=melody_xml([("D5", 1.0)])
            ).error_code
            == "BAD_PARAM"
        )
        assert (
            service.insert_measure(
                work.id, movement_id, 1, voice="soprano", musicxml="nope"
            ).error_code
            == "BAD_PARAM"
        )
        empty = service.insert_measure(
            work.id, movement_id, 1, voice="soprano", musicxml=melody_xml([]), check=False
        )
        assert empty.ok

    def test_delete_measure(self, service: CompositionService) -> None:
        """A measure can be deleted from every voice."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        service.submit_theme(
            work.id, movement_id, melody_xml([("C5", 1.0), ("D5", 1.0)]), check=False
        )
        assert service.delete_measure(work.id, movement_id, 1, check=False).ok
        assert (
            service.delete_measure(work.id, movement_id, 99, check=False).error_code == "BAD_PARAM"
        )

    def test_insert_measure_failing_check(
        self, service: CompositionService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An inserted measure that fails the checker returns the report."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        service.submit_theme(work.id, movement_id, melody_xml([("C5", 1.0)]), check=False)
        failing = CheckReport([CheckViolation("r", Severity.ERROR, 1, None, None, "k", "m", "s")])
        monkeypatch.setattr(service, "_check", lambda *_args, **_kwargs: failing)
        result = service.insert_measure(work.id, movement_id, 1)
        assert not result.ok
        assert result.report is not None

    def test_delete_measure_failing_check(
        self, service: CompositionService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deleted measure that fails the checker returns the report."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        service.submit_theme(
            work.id, movement_id, melody_xml([("C5", 1.0), ("D5", 1.0)]), check=False
        )
        failing = CheckReport([CheckViolation("r", Severity.ERROR, 1, None, None, "k", "m", "s")])
        monkeypatch.setattr(service, "_check", lambda *_args, **_kwargs: failing)
        result = service.delete_measure(work.id, movement_id, 1)
        assert not result.ok
        assert result.report is not None

    def test_prompts_and_missing(self, service: CompositionService) -> None:
        """Coverage tracks movements without a prompt."""
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        assert len(service.missing_movement_prompts(work.id)) == 1
        assert service.set_movement_prompt(work.id, "m01", "写一乐章").ok
        assert service.missing_movement_prompts(work.id) == []
        assert not service.set_movement_prompt(work.id, "nope", "x").ok

    def test_movement_score_and_merge(self, service: CompositionService) -> None:
        """Composed movements merge into one work."""
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.add_movement(work.id)
        service.submit_theme(work.id, "m01", cadence_xml(), check=False)
        service.submit_theme(work.id, "m02", cadence_xml(), check=False)
        assert service._has_notes(work.id, "m01")
        assert service.merged_musicxml(work.id)

    def test_merged_empty(self, service: CompositionService) -> None:
        """A work without composed movements merges to nothing."""
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        assert service.merged_musicxml(work.id) == ""
        service.add_movement(work.id)
        assert service.merged_musicxml(work.id) == ""

    def test_movement_target_composition(self, service: CompositionService) -> None:
        """Movements are first-class composition targets with an empty start."""
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        assert service.current_musicxml(work.id, "m01") == ""
        assert not service.current_score(work.id, "m01").parts
        assert not service.check(work.id, "m01").ok
        submitted = service.submit_theme(work.id, "m01", cadence_xml(), check=False)
        assert submitted.ok
        score = service.current_score(work.id, "m01")
        assert service.check_score(work.id, "m01", score).ok
        with pytest.raises(KeyError):
            service.current_musicxml(work.id, "nope")

    def test_history_is_newest_first(self, service: CompositionService) -> None:
        """Works are listed in strict reverse-chronological order."""
        service.create_work("First", "plain", "C")
        second = service.create_work("Second", "plain", "C")
        assert service.list_works()[0] == second.id

    def test_generation_lifecycle(self, service: CompositionService) -> None:
        """A generation run is recorded as started then finished or failed."""
        work = service.create_work("Demo", "plain", "C")
        assert service.latest_generation_state(work.id) is None
        service.start_generation(work.id, "写一段")
        assert service.latest_generation_state(work.id) == "generation_started"
        service.finish_generation(work.id, True)
        assert service.latest_generation_state(work.id) == "generation_finished"
        service.start_generation(work.id, "再来")
        service.finish_generation(work.id, False)
        assert service.latest_generation_state(work.id) == "generation_failed"

    def test_interrupt_stale_generations(self, service: CompositionService) -> None:
        """Runs without a completion tag (killed or failed) are interrupted."""
        stale = service.create_work("Stale", "plain", "C")
        crashed = service.create_work("Crashed", "plain", "C")
        done = service.create_work("Done", "plain", "C")
        service.start_generation(stale.id, "goal")
        service.start_generation(crashed.id, "goal")
        service.finish_generation(crashed.id, False)
        service.start_generation(done.id, "goal")
        service.finish_generation(done.id, True)
        assert service.interrupt_stale_generations() == _INTERRUPTED_WORKS
        assert service.latest_generation_state(stale.id) == "generation_interrupted"
        assert service.latest_generation_state(crashed.id) == "generation_interrupted"
        assert service.latest_generation_state(done.id) == "generation_finished"
        assert service.interrupt_stale_generations() == 0


class TestStyleAndExemption:
    """Style kits and the free-voice-leading exemption."""

    def test_style_snapshot(self, service: CompositionService) -> None:
        """A created work stores a style snapshot."""
        work = service.create_work("Demo", "plain", "C", style="impressionist")
        assert work.style is not None
        assert work.style.id == "impressionist"
        assert work.style.brief

    def test_legacy_style_defaults(self, service: CompositionService) -> None:
        """A work without a snapshot falls back to the default kit."""
        work = service.create_work("Demo", "plain", "C")
        work.style = None
        assert service.style_for(work).id == "baroque"

    def test_effective_rules(self, service: CompositionService) -> None:
        """The effective rules come from the style."""
        work = service.create_work("Demo", "plain", "C", style="impressionist")
        assert service.effective_rules(work) == frozenset({"empty", "voices"})

    def test_techniques_for(self, service: CompositionService) -> None:
        """The technique registry is limited to the style."""
        work = service.create_work("Demo", "plain", "C", style="impressionist")
        ids = service.techniques_for(work).ids()
        assert "planing" in ids
        assert "functional_cycle" not in ids

    def test_free_voice_leading_scope(self, service: CompositionService) -> None:
        """Invoking the exemption records its scope."""
        work = service.create_work("Demo", "plain", "C", style="full")
        movement = work.movements[0]
        service.set_tempo(work.id, movement.id, 100, check=False)
        service.apply_technique(
            work.id,
            movement.id,
            "free_voice_leading",
            {
                "voice": "soprano",
                "measure_range": {"start": 1, "end": 2},
                "reason": "为了音乐表现需要自由进行",
            },
        )
        loaded = service.get_work(work.id)
        assert service._exempt_scopes(loaded, loaded.movements[0]) == [("soprano", 1, 2)]

    def test_filter_exemptions(self) -> None:
        """Waived violations are dropped from a report."""
        violation = CheckViolation("crossing", Severity.ERROR, 1, "soprano", "alto", "k", "m", "s")
        report = CheckReport(violations=[violation])
        assert _filter_exemptions(report, [("soprano", 1, 1)]).violations == []
        assert _filter_exemptions(report, []).violations == [violation]
        assert _filter_exemptions(report, [("bass", 1, 1)]).violations == [violation]

    def test_exempt_scope_bounds(self) -> None:
        """A violation outside the scope is not waived."""
        violation = CheckViolation("crossing", Severity.ERROR, 5, "soprano", None, "k", "m", "s")
        assert not _in_exempt_scope(violation, [("soprano", 1, 2)])
