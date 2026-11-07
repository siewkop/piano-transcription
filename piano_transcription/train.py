"""Uruchom: python -m piano_transcription.train --help."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .checkpoint import choose_device
from .config import Config
from .dataset import TrainingDataset, load_index
from .evaluate import evaluate
from .model import PianoModel
from .states import NUM_STATES, STATE_NAMES, NoteState


def training_class_weights(data_dir: Path, records: list[dict], power: float) -> torch.Tensor:
    """Wagi odwrotnie proporcjonalne do częstości klas, liczone na train."""
    counts = torch.zeros(NUM_STATES, dtype=torch.long)
    for record in records:
        labels = torch.load(data_dir / record["file"], weights_only=True, mmap=True)["labels"]
        counts += torch.bincount(labels.flatten().long(), minlength=NUM_STATES)
    weights = (counts[NoteState.OFF].clamp_min(1) / counts.clamp_min(1)).pow(power)
    print("Wagi klas:", ", ".join(f"{name}={w:.2f}" for name, w in zip(STATE_NAMES, weights)))
    return weights


def train(
    data_dir: Path,
    run_dir: Path,
    config: Config,
    device: torch.device,
    epochs: int,
):
    records = load_index(data_dir, config)
    training = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    if not training or not validation:
        raise ValueError("Uczenie wymaga osobnych nagrań train i validation w manifeście.")
    power = config.training.class_balance_power
    if not 0 <= power <= 1:
        raise ValueError("class_balance_power musi być pomiędzy 0 a 1.")
    explicit_weights = config.training.class_weights
    if explicit_weights is not None:
        weights = torch.tensor(explicit_weights, dtype=torch.float32)
        if (
            weights.shape != (NUM_STATES,)
            or not torch.isfinite(weights).all()
            or (weights <= 0).any()
        ):
            raise ValueError("class_weights wymaga pięciu dodatnich, skończonych wag.")
        if power:
            raise ValueError("Wybierz class_weights albo class_balance_power, nie oba.")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.training.seed)
    model = PianoModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.training.lr_step_size, gamma=config.training.lr_gamma
    )
    dataset = TrainingDataset(data_dir, training, config)
    class_weights = training_class_weights(data_dir, training, power).to(device) if power else None
    if explicit_weights is not None:
        class_weights = weights.to(device)
        print("Jawne wagi [off, onset, sustain, re-onset, offset]:", explicit_weights)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    print(f"Urządzenie: {device}; parametry: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trening: {len(training)} nagrań; walidacja: {len(validation)}; zapis: {run_dir}")
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, weight_sum = 0.0, 0.0
        for batch_index, (mel, previous, target) in enumerate(loader, 1):
            mel, previous, target = mel.to(device), previous.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(mel, previous)
            loss = F.cross_entropy(
                logits.reshape(-1, NUM_STATES),
                target.flatten(),
                weight=class_weights,
                ignore_index=-100,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Niefinitywna strata. Sprawdź dane i learning_rate.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()  # Harmonogram liczy kroki optymalizatora, a nie epoki.
            valid = target != -100
            weight = float(
                class_weights[target[valid]].sum() if class_weights is not None else valid.sum()
            )
            loss_sum += loss.item() * weight
            weight_sum += weight
            if batch_index == 1 or batch_index % 20 == 0:
                print(
                    f"Epoka {epoch}/{epochs}, batch {batch_index}/{len(loader)}, CE={loss.item():.4f}",
                    flush=True,
                )
        print("Walidacja autoregresyjna całych nagrań...", flush=True)
        metrics = evaluate(model, data_dir, validation)
        validation_loss = metrics["cross_entropy"]
        attack_f1 = metrics["attack_frames"]["f1"]
        onset_f1 = metrics["note_onsets"]["macro_f1"]
        print(
            f"Epoka {epoch}: train CE={loss_sum / weight_sum:.4f}, "
            f"validation CE={validation_loss:.4f}, F1 ramek ataku={attack_f1:.4f}, "
            f"F1 nut (macro, 50 ms)={onset_f1:.4f}",
            flush=True,
        )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Ucz autoregresyjny model transkrypcji fortepianu."
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("configs/small_events.json"))
    parser.add_argument(
        "--run", type=Path, help="Folder eksperymentu (domyślnie models/piano_small)"
    )
    parser.add_argument("--epochs", type=int, help="Docelowa łączna liczba epok")
    parser.add_argument("--device", default="auto", help="auto, cpu lub cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = Config.load(args.config)
    run_dir = args.run or Path("models/piano_small")
    epochs = args.epochs if args.epochs is not None else config.training.epochs
    train(
        args.data, run_dir, config, choose_device(args.device), epochs
    )


if __name__ == "__main__":
    main()
