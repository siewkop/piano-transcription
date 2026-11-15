"""Dekoder tworzy zdarzenie przy wejściu w onset lub re-onset."""

import torch

from .states import MIN_MIDI, NUM_KEYS, NoteEvent, NoteState


def attack_starts(states: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """Pozycje [ramka, klawisz] wejścia w onset lub re-onset."""
    preceding = torch.cat((previous.unsqueeze(0), states[:-1]))
    attacks = (states == NoteState.ONSET) | (states == NoteState.REONSET)
    return torch.nonzero(attacks & (states != preceding))


class NoteDecoder:
    def __init__(self, frame_seconds: float):
        self.frame_seconds = frame_seconds
        self.previous = torch.zeros(NUM_KEYS, dtype=torch.long)
        self.frame_index = 0

    def process(self, probabilities: torch.Tensor) -> list[NoteEvent]:
        """Przyjmuje [T, 88, 5] na CPU. Zachowuje stan na następne wywołanie."""
        states = probabilities.argmax(dim=-1)
        events = []
        if not len(states):
            return events
        for frame, key in attack_starts(states, self.previous).tolist():
            absolute_frame = self.frame_index + frame
            events.append(
                NoteEvent(
                    pitch=MIN_MIDI + key,
                    onset_sec=absolute_frame * self.frame_seconds,
                    frame_index=absolute_frame,
                    state=NoteState(int(states[frame, key])),
                    probabilities=tuple(probabilities[frame, key].tolist()),
                )
            )
        self.previous = states[-1].clone()
        self.frame_index += len(states)
        return events
