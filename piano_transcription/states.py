"""Kolejność klas jest wspólna dla etykiet, wyjścia sieci i plików wynikowych."""

from dataclasses import dataclass
from enum import IntEnum

MIN_MIDI = 21
NUM_KEYS = 88


class NoteState(IntEnum):
    OFF = 0
    ONSET = 1
    SUSTAIN = 2
    REONSET = 3
    OFFSET = 4


STATE_NAMES = ("off", "onset", "sustain", "re-onset", "offset")
NUM_STATES = len(STATE_NAMES)


def note_name(pitch: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[pitch % 12]}{pitch // 12 - 1}"


@dataclass(frozen=True)
class NoteEvent:
    """Początek pojedynczej nuty; czas w sekundach od początku pliku audio."""

    pitch: int
    onset_sec: float
    frame_index: int
    state: NoteState
    probabilities: tuple[float, ...]
