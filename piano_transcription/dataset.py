"""Trening losuje fragmenty nagrań; walidacja korzysta z całych utworów."""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import Config
from .states import NUM_KEYS


def load_index(directory: Path, config: Config) -> list[dict]:
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    if index["format_version"] != 1 or index["audio"] != asdict(config.audio):
        raise ValueError("Cache ma inną konfigurację audio. Przygotuj dane w nowym katalogu.")
    return index["records"]


class TrainingDataset(Dataset):
    def __init__(self, directory: Path, records: list[dict], config: Config):
        self.directory = directory
        self.records = records
        self.length = config.training.sequence_frames
        self.repeats = config.training.segments_per_recording

    def __len__(self):
        return len(self.records) * self.repeats

    def __getitem__(self, index):
        record = self.records[index % len(self.records)]
        # mmap odczytuje potrzebny fragment bez kopiowania całego utworu do RAM.
        data = torch.load(self.directory / record["file"], weights_only=True, mmap=True)
        total = len(data["mel"])
        start = int(torch.randint(max(1, total - self.length + 1), ()).item())
        count = min(self.length, total - start)
        mel = torch.full((self.length, data["mel"].shape[1]), -11.512925)
        target = torch.full((self.length, NUM_KEYS), -100, dtype=torch.long)
        previous = torch.zeros((self.length, NUM_KEYS), dtype=torch.long)
        mel[:count] = data["mel"][start : start + count]
        target[:count] = data["labels"][start : start + count].long()
        # To przesunięcie jest istotą teacher forcing: y[t-1], nigdy y[t].
        if start > 0:
            previous[0] = data["labels"][start - 1]
        previous[1:count] = target[: count - 1]
        return mel, previous, target
