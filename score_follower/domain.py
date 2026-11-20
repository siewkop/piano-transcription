from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class ScoreEvent:
    index: int
    pitches: frozenset[int]
    onset_q: Fraction
    measure_index: int
    offset_in_measure_q: Fraction

@dataclass(frozen=True)
class ScoreDocument:
    score_id: str
    content_hash: str
    music_xml: str
    part_id: str
    events: tuple[ScoreEvent, ...]

@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    onset_sec: float

@dataclass(frozen=True)
class TranscriptionStep:
    notes: tuple[NoteEvent, ...]
    processed_until_sec: float
