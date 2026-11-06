"""CNN opisuje brzmienie, LSTM łączy je z historią stanów 88 klawiszy."""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn

from .config import Config
from .states import NUM_KEYS, NUM_STATES


@dataclass
class RecurrentState:
    hidden: tuple[torch.Tensor, torch.Tensor]
    previous_notes: torch.Tensor


class AcousticEncoder(nn.Module):
    context_frames = 3  # Trzy konwolucje 3×3 dają 7 ramek kontekstu.

    def __init__(self, config: Config):
        super().__init__()
        channels = (1, *config.model.cnn_channels)
        layers = []
        for index in range(3):
            layers.extend(
                [
                    nn.Conv2d(channels[index], channels[index + 1], 3, padding=1),
                    nn.BatchNorm2d(channels[index + 1]),
                    nn.ReLU(),
                ]
            )
            if index in (1, 2):
                layers.extend([nn.MaxPool2d((1, 2)), nn.Dropout(0.25)])
        self.convolutions = nn.Sequential(*layers)
        flattened_size = channels[-1] * (config.audio.n_mels // 4)
        projection = [nn.Linear(flattened_size, config.model.acoustic_size)]
        if config.model.acoustic_layer_norm:
            # Stabilna skala cech przed bramkami LSTM, także przy małym batchu.
            projection.append(nn.LayerNorm(config.model.acoustic_size))
        projection.append(nn.Dropout(0.5))
        self.projection = nn.Sequential(*projection)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # [B, T, mel] → [B, kanały, T, częstotliwość] → [B, T, cechy]
        features = self.convolutions(mel.unsqueeze(1))
        features = features.transpose(1, 2).flatten(2)
        return self.projection(features)


class PianoModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = AcousticEncoder(config)
        self.embedding = nn.Embedding(NUM_STATES, config.model.embedding_size)
        self.lstm = nn.LSTM(
            config.model.acoustic_size + NUM_KEYS * config.model.embedding_size,
            config.model.hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.classifier = nn.Linear(config.model.hidden_size, NUM_KEYS * NUM_STATES)

    def initial_state(self, batch_size: int) -> RecurrentState:
        parameter = next(self.parameters())
        shape = (2, batch_size, self.config.model.hidden_size)
        return RecurrentState(
            (parameter.new_zeros(shape), parameter.new_zeros(shape)),
            torch.zeros(batch_size, NUM_KEYS, dtype=torch.long, device=parameter.device),
        )

    def forward(self, mel: torch.Tensor, previous_states: torch.Tensor) -> torch.Tensor:
        """Trening: poprzednie etykiety są znane, więc liczymy całą sekwencję naraz.

        mel: [B, T, 229], previous_states: [B, T, 88], wynik: [B, T, 88, 5].
        Zwracamy logity. CrossEntropyLoss wykonuje log-softmax wewnętrznie.
        """
        acoustic = self.encoder(mel)
        previous = self.embedding(previous_states).flatten(2)
        sequence, _ = self.lstm(torch.cat((acoustic, previous), dim=-1))
        return self.classifier(sequence).reshape(*sequence.shape[:2], NUM_KEYS, NUM_STATES)

    def step(self, acoustic: torch.Tensor, state: RecurrentState):
        """Jeden krok inferencji. Do następnego kroku trafia argmax, a nie etykieta MIDI."""
        previous = self.embedding(state.previous_notes).flatten(1)
        inputs = torch.cat((acoustic, previous), dim=-1).unsqueeze(1)
        output, hidden = self.lstm(inputs, state.hidden)
        logits = self.classifier(output[:, 0]).reshape(-1, NUM_KEYS, NUM_STATES)
        return logits, RecurrentState(hidden, logits.argmax(dim=-1))

    @torch.inference_mode()
    def iter_logits(self, mel: torch.Tensor, chunk_frames: int = 512) -> Iterator[torch.Tensor]:
        """Inferencja całego nagrania. Zwraca kolejne [T, 88, 5] na CPU.

        CNN dostaje nakładające się fragmenty z kontekstem. Pamięć LSTM i poprzednie
        predykcje przechodzą między fragmentami, więc granice nie resetują utworu.
        """
        if self.training:
            raise RuntimeError("Przed inferencją wywołaj model.eval().")
        if chunk_frames <= 0:
            raise ValueError("chunk_frames musi być dodatnie.")
        device = next(self.parameters()).device
        state = self.initial_state(1)
        context = self.encoder.context_frames
        for start in range(0, len(mel), chunk_frames):
            end = min(start + chunk_frames, len(mel))
            left, right = max(0, start - context), min(len(mel), end + context)
            acoustic = self.encoder(mel[left:right].unsqueeze(0).to(device))
            acoustic = acoustic[:, start - left : end - left]
            outputs = []
            for frame in range(end - start):
                logits, state = self.step(acoustic[:, frame], state)
                outputs.append(logits[0])
            yield torch.stack(outputs).cpu()
