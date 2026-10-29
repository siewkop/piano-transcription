"""Ten sam tor przygotowania audio jest używany w treningu i transkrypcji."""

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import nn
from torchaudio.transforms import MelSpectrogram

from .config import AudioConfig


def load_audio(path: str | Path, sample_rate: int) -> tuple[torch.Tensor, float]:
    samples, original_rate = sf.read(path, dtype="float32", always_2d=True)
    if len(samples) == 0:
        raise ValueError(f"Plik audio jest pusty: {path}")
    if not np.isfinite(samples).all():
        raise ValueError(f"Audio zawiera NaN lub nieskończoności: {path}")
    duration = len(samples) / original_rate
    mono = samples.mean(axis=1)
    if original_rate != sample_rate:
        divisor = gcd(original_rate, sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, original_rate // divisor)
    return torch.from_numpy(mono), duration


class LogMelSpectrogram(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.mel = MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=1.0,
            norm="slaney",
            mel_scale="htk",
            center=True,
            pad_mode="constant",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """[próbki] → [ramki, pasma]. Ramka t ma czas t * hop / sample_rate."""
        frame_count = (waveform.shape[-1] + self.config.hop_length - 1) // self.config.hop_length
        mel = self.mel(waveform)[..., :frame_count]
        return mel.clamp_min(1e-5).log().transpose(-1, -2).contiguous()
