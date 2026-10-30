"""MIDI z pedałem sustain → jedna z pięciu klas dla każdej ramki i wysokości."""

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import pretty_midi
import torch

from .config import AudioConfig
from .states import MIN_MIDI, NUM_KEYS, NoteState


@dataclass(frozen=True)
class MidiNote:
    pitch: int
    start: float
    end: float  # Koniec z uwzględnieniem sustain, przed skróceniem przez kolejny atak.


def read_midi_notes(path: str | Path) -> list[MidiNote]:
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum or not 0 <= instrument.program <= 7:
            if instrument.notes:
                raise ValueError(f"Oczekiwano MIDI wyłącznie z partią fortepianu: {path}")
            continue
        pedal = sorted(
            (cc for cc in instrument.control_changes if cc.number == 64), key=lambda cc: cc.time
        )
        pedal_times = [cc.time for cc in pedal]
        # Dla każdej pozycji zapamiętujemy najbliższe przyszłe zwolnienie pedału.
        release_times = [midi.get_end_time()] * len(pedal)
        next_release = midi.get_end_time()
        for index in range(len(pedal) - 1, -1, -1):
            if pedal[index].value < 64:
                next_release = pedal[index].time
            release_times[index] = next_release
        for note in instrument.notes:
            if not MIN_MIDI <= note.pitch < MIN_MIDI + NUM_KEYS:
                raise ValueError(f"Nuta MIDI {note.pitch} poza zakresem 21–108 w {path}")
            end = note.end
            index = bisect_right(pedal_times, end) - 1
            if index >= 0 and pedal[index].value >= 64:
                end = max(end, release_times[index])
            notes.append(MidiNote(note.pitch, note.start, end))
    return sorted(notes, key=lambda note: (note.pitch, note.start))


def make_labels(notes: list[MidiNote], frame_count: int, audio: AudioConfig) -> torch.Tensor:
    """Onset i offset zajmują po jednej ramce (32 ms w konfiguracji domyślnej).

    Granice zaokrąglamy do najbliższej ramki, z połówką w górę. Nowy atak ma
    pierwszeństwo przed offsetem poprzedniej nuty. Tę samą wysokość obsługujemy
    chronologicznie, aby poprawnie oznaczyć re-onset i nie nadpisać go sustainem.
    """
    labels = torch.full((frame_count, NUM_KEYS), NoteState.OFF, dtype=torch.uint8)
    for pitch in range(MIN_MIDI, MIN_MIDI + NUM_KEYS):
        pitch_notes = sorted(
            (note for note in notes if note.pitch == pitch), key=lambda note: note.start
        )
        sounding_until = -1.0
        for index, note in enumerate(pitch_notes):
            start = int(note.start / audio.frame_seconds + 0.5)
            end = max(start + 1, int(note.end / audio.frame_seconds + 0.5))
            is_reonset = note.start < sounding_until
            sounding_until = note.end
            if index + 1 < len(pitch_notes):
                next_start = int(pitch_notes[index + 1].start / audio.frame_seconds + 0.5)
                if next_start == start:
                    raise ValueError(
                        f"Dwa ataki MIDI {pitch} w ramce {start}; zwiększ rozdzielczość."
                    )
                end = min(end, next_start)
            key = pitch - MIN_MIDI
            if start >= frame_count:
                continue
            labels[start : min(end, frame_count), key] = NoteState.SUSTAIN
            labels[start, key] = NoteState.REONSET if is_reonset else NoteState.ONSET
            if end < frame_count:
                labels[end, key] = NoteState.OFFSET
    return labels
