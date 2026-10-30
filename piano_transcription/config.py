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


@dataclass(frozen=True)
class ModelConfig:
    cnn_channels: tuple[int, int, int] = (48, 48, 96)
    acoustic_size: int = 768
    hidden_size: int = 256
    embedding_size: int = 2
    acoustic_layer_norm: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    sequence_frames: int = 313
    segments_per_recording: int = 4
    learning_rate: float = 6e-4
    lr_step_size: int = 10000
    lr_gamma: float = 0.98
    gradient_clip: float = 3.0
    seed: int = 42
    class_balance_power: float = 0.0
    checkpoint_metric: str = "cross_entropy"
    class_weights: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        model = dict(data["model"])
        model["cnn_channels"] = tuple(model["cnn_channels"])
        training = dict(data["training"])
        if training.get("class_weights") is not None:
            training["class_weights"] = tuple(training["class_weights"])
        return cls(
            AudioConfig(**data["audio"]), ModelConfig(**model), TrainingConfig(**training)
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
