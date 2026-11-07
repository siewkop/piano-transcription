"""Dekoder tworzy zdarzenie przy wejściu w onset lub re-onset."""

import torch

from .states import MIN_MIDI, NUM_KEYS, NoteEvent, NoteState


def attack_starts(states: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """Pozycje [ramka, klawisz] wejścia w onset lub re-onset."""
    preceding = torch.cat((previous.unsqueeze(0), states[:-1]))
    attacks = (states == NoteState.ONSET) | (states == NoteState.REONSET)
    return torch.nonzero(attacks & (states != preceding))
