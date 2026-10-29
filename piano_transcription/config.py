"""Parametry zapisujemy razem z wagami, żeby odtworzyć ten sam model."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 229
    f_min: float = 30.0
    f_max: float = 8000.0

    @property
    def frame_seconds(self) -> float:
        return self.hop_length / self.sample_rate
