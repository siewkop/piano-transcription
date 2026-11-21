import mido
import pytest

from score_follower.domain import NoteEvent
from score_follower.transcription.midi_replay import (
    STEP_SEC,
    MidiReadError,
    MidiReplay,
    read_midi_onsets,
)

TICKS_PER_BEAT = 480


def write_midi(path, notes, *, tempo=500000, tempo_changes=(), note_off_as_velocity_0=False):
    events = [(tick, mido.MetaMessage("set_tempo", tempo=value)) for tick, value in tempo_changes]
    events.append((0, mido.MetaMessage("set_tempo", tempo=tempo)))
    for pitch, start, length in notes:
        events.append((start, mido.Message("note_on", note=pitch, velocity=80)))
        off = (mido.Message("note_on", note=pitch, velocity=0) if note_off_as_velocity_0
               else mido.Message("note_off", note=pitch, velocity=0))
        events.append((start + length, off))

    events.sort(key=lambda e: (e[0], e[1].type == "note_on" and e[1].velocity > 0))

    track = mido.MidiTrack()
    previous = 0
    for tick, message in events:
        track.append(message.copy(time=tick - previous))
        previous = tick
    midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    midi.tracks.append(track)
    midi.save(path)
    return path


def onsets(notes):
    return [(n.pitch, pytest.approx(n.onset_sec)) for n in notes]


def test_reads_onsets_in_seconds(tmp_path):
    path = write_midi(tmp_path / "a.mid", [(60, 0, 480), (64, 0, 480), (67, 480, 480)])

    assert onsets(read_midi_onsets(path)) == [(60, 0.0), (64, 0.0), (67, 0.5)]


def test_velocity_zero_is_note_off(tmp_path):
    path = write_midi(tmp_path / "a.mid", [(60, 0, 240), (60, 480, 240)], note_off_as_velocity_0=True)

    assert onsets(read_midi_onsets(path)) == [(60, 0.0), (60, 0.5)]


def test_tempo_change(tmp_path):
    path = write_midi(
        tmp_path / "a.mid", [(60, 0, 240), (62, 480, 240), (64, 960, 240)],
        tempo_changes=[(480, 1000000)],
    )

    assert onsets(read_midi_onsets(path)) == [(60, 0.0), (62, 0.5), (64, 1.5)]


def test_tracks_are_merged(tmp_path):
    right = mido.MidiTrack([mido.Message("note_on", note=72, velocity=80, time=480),
                            mido.Message("note_off", note=72, time=480)])
    left = mido.MidiTrack([mido.Message("note_on", note=48, velocity=80, time=0),
                           mido.Message("note_off", note=48, time=960)])
    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    midi.tracks.extend([right, left])
    path = tmp_path / "a.mid"
    midi.save(path)

    assert onsets(read_midi_onsets(path)) == [(48, 0.0), (72, 0.5)]


def test_missing_file(tmp_path):
    with pytest.raises(MidiReadError):
        read_midi_onsets(tmp_path / "nope.mid")


def test_not_midi(tmp_path):
    path = tmp_path / "a.mid"
    path.write_text("not a midi file")
    with pytest.raises(MidiReadError):
        read_midi_onsets(path)


def test_empty_midi_cannot_be_replayed(tmp_path):
    path = write_midi(tmp_path / "a.mid", [])
    with pytest.raises(MidiReadError):
        MidiReplay.from_file(path)


def test_every_note_appears_exactly_once_in_order():
    notes = [NoteEvent(60, 0.0), NoteEvent(64, 0.0), NoteEvent(67, 0.1), NoteEvent(72, 1.25)]
    steps = list(MidiReplay(notes).steps())

    assert [n for s in steps for n in s.notes] == notes


def test_note_is_in_the_step_that_covers_its_onset():
    steps = list(MidiReplay([NoteEvent(60, 0.0), NoteEvent(62, 0.1)]).steps())

    previous = 0.0
    for step in steps:
        for note in step.notes:
            assert previous <= note.onset_sec < step.processed_until_sec
        previous = step.processed_until_sec


def test_steps_advance_by_step_sec_and_continue_after_last_note():
    steps = list(MidiReplay([NoteEvent(60, 0.5)], tail_sec=1.0).steps())

    times = [s.processed_until_sec for s in steps]
    assert times[0] == pytest.approx(STEP_SEC)
    assert all(b - a == pytest.approx(STEP_SEC) for a, b in zip(times, times[1:]))
    assert times[-1] >= 1.5
    assert times[-2] < 1.5
    assert not any(s.notes for s in steps if s.processed_until_sec > 0.5 + STEP_SEC)


def test_from_file_matches_read_midi_onsets(tmp_path):
    path = write_midi(tmp_path / "a.mid", [(60, 0, 480), (67, 480, 480)])
    replay = MidiReplay.from_file(path, tail_sec=0.0)

    assert replay.notes == tuple(read_midi_onsets(path))
    assert [n for s in replay.steps() for n in s.notes] == list(replay.notes)
