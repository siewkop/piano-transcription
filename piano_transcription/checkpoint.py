"""Plik .pt zawiera wagi, konfigurację oraz stan potrzebny do wznowienia uczenia."""

from pathlib import Path

import torch

from .config import Config
from .model import PianoModel
from .states import STATE_NAMES


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def read_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1 or tuple(checkpoint["states"]) != STATE_NAMES:
        raise ValueError("Nieobsługiwany format checkpointu lub inna kolejność klas.")
    return checkpoint


def load_model(path: Path, device: torch.device) -> PianoModel:
    checkpoint = read_checkpoint(path)
    model = PianoModel(Config.from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()
