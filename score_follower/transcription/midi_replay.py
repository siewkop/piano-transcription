from collections.abc import Iterator, Sequence
from pathlib import Path

import mido

from ..domain import NoteEvent, TranscriptionStep

STEP_SEC = 512 / 16000
MIN_PITCH, MAX_PITCH = 21, 108


class MidiReadError(Exception):
    pass


def read_midi_onsets(path: str | Path) -> list[NoteEvent]:
    path = Path(path)
    if not path.is_file():
        raise MidiReadError(f"File does not exist: {path}")
    try:
        midi = mido.MidiFile(path)
    except Exception as error:
        raise MidiReadError(f"Invalid MIDI file: {error}") from error

    notes = []
    now = 0.0
    for message in midi:
        now += message.time
        if message.type == "note_on" and message.velocity > 0:
            if MIN_PITCH <= message.note <= MAX_PITCH:
                notes.append(NoteEvent(pitch=message.note, onset_sec=now))
    notes.sort(key=lambda note: (note.onset_sec, note.pitch))
    return notes


class MidiReplay:
    def __init__(self, notes: Sequence[NoteEvent], step_sec: float = STEP_SEC, tail_sec: float = 1.0):
        if step_sec <= 0:
            raise ValueError("step_sec must be positive.")
        self.notes = tuple(sorted(notes, key=lambda note: (note.onset_sec, note.pitch)))
        self.step_sec = step_sec
        self.tail_sec = tail_sec

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "MidiReplay":
        notes = read_midi_onsets(path)
        if not notes:
            raise MidiReadError(f"The MIDI file does not contain any notes: {path}")
        return cls(notes, **kwargs)

    def steps(self) -> Iterator[TranscriptionStep]:
        end_sec = (self.notes[-1].onset_sec if self.notes else 0.0) + self.tail_sec
        next_note = 0
        step = 0
        while True:
            step += 1
            until = step * self.step_sec
            batch = []
            while next_note < len(self.notes) and self.notes[next_note].onset_sec < until:
                batch.append(self.notes[next_note])
                next_note += 1
            yield TranscriptionStep(notes=tuple(batch), processed_until_sec=until)
            if until >= end_sec and next_note == len(self.notes):
                return
