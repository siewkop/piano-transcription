from collections.abc import Iterator
from typing import Protocol

from ..domain import TranscriptionStep


class StepSource(Protocol):
    
    def steps(self) -> Iterator[TranscriptionStep]:
        """Yield consecutive transcription steps, processed_until_sec grows with every step."""
        ...
