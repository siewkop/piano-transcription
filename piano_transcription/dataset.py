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
