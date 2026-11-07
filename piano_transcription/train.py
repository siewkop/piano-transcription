"""Uruchom: python -m piano_transcription.train --help."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .checkpoint import choose_device, read_checkpoint, save_checkpoint
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
    resume: Path | None = None,
    initialize: Path | None = None,
):
    if resume and initialize:
        raise ValueError("Wybierz wznowienie albo inicjalizację wag.")
    records = load_index(data_dir, config)
    training = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    if not training or not validation:
        raise ValueError("Uczenie wymaga osobnych nagrań train i validation w manifeście.")
    power = config.training.class_balance_power
    if not 0 <= power <= 1:
        raise ValueError("class_balance_power musi być pomiędzy 0 a 1.")
    checkpoint_metric = config.training.checkpoint_metric
    if checkpoint_metric not in ("cross_entropy", "attack_frame_f1", "note_onset_f1"):
        raise ValueError("Nieznane checkpoint_metric.")
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
    if resume is None and (run_dir / "last.pt").exists():
        raise FileExistsError(
            f"Eksperyment już istnieje: {run_dir}. Użyj --resume lub innej nazwy."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.training.seed)
    model = PianoModel(config).to(device)
    if initialize:
        initial = read_checkpoint(initialize)
        source_config = Config.from_dict(initial["config"])
        if source_config.audio != config.audio or source_config.model != config.model:
            raise ValueError("Inicjalizacja wymaga tej samej architektury i konfiguracji audio.")
        model.load_state_dict(initial["model"])
        (run_dir / "initialization.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(initialize.resolve()),
                    "source_epoch": initial["epoch"],
                    "source_step": initial["global_step"],
                    "optimizer_reset": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Początkowe wagi: {initialize}; nowy optymalizator i nowe kryterium wyboru.")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.training.lr_step_size, gamma=config.training.lr_gamma
    )
    first_epoch, global_step, best_loss = 1, 0, float("inf")
    if resume is not None:
        saved = read_checkpoint(resume)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        first_epoch, global_step = saved["epoch"] + 1, saved["global_step"]
        best_loss = saved["best_loss"]
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"] and device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if first_epoch > epochs:
        raise ValueError(f"Zapisano już {first_epoch - 1} epok. Podaj większe --epochs.")
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
    for epoch in range(first_epoch, epochs + 1):
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
            global_step += 1
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
        # Oba kryteria przechowujemy jako wartość do minimalizacji w checkpointcie.
        onset_f1 = metrics["note_onsets"]["macro_f1"]
        selection_loss = {
            "cross_entropy": validation_loss,
            "attack_frame_f1": 1 - attack_f1,
            "note_onset_f1": 1 - onset_f1,
        }[checkpoint_metric]
        improved = selection_loss < best_loss
        best_loss = min(best_loss, selection_loss)
        save_checkpoint(
            run_dir / "last.pt", model, optimizer, scheduler, epoch, global_step, best_loss
        )
        if improved:
            save_checkpoint(
                run_dir / "best.pt", model, optimizer, scheduler, epoch, global_step, best_loss
            )
            (run_dir / "best_validation.json").write_text(
                json.dumps({"epoch": epoch, "global_step": global_step, **metrics}, indent=2),
                encoding="utf-8",
            )
        history_path = run_dir / "history.csv"
        write_header = not history_path.exists()
        # Nie zmieniamy liczby kolumn przy wznowieniu starszego eksperymentu.
        extended_history = (
            write_header
            or "note_onset_macro_f1" in (history_path.read_text(encoding="utf-8").splitlines()[0])
        )
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(
                    [
                        "epoch",
                        "step",
                        "train_ce",
                        "validation_ce",
                        "learning_rate",
                        "attack_frame_f1",
                        "note_onset_macro_f1",
                    ]
                )
            row = [
                epoch,
                global_step,
                loss_sum / weight_sum,
                validation_loss,
                optimizer.param_groups[0]["lr"],
            ]
            if extended_history:
                row.extend([attack_f1, onset_f1])
            writer.writerow(row)
        (run_dir / "validation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
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
    start = parser.add_mutually_exclusive_group()
    start.add_argument(
        "--resume", type=Path, help="Wznów last.pt; konfiguracja pochodzi z checkpointu"
    )
    start.add_argument(
        "--initialize", type=Path, help="Nowy eksperyment z podanych wag, bez stanu optymalizatora"
    )
    parser.add_argument("--epochs", type=int, help="Docelowa łączna liczba epok")
    parser.add_argument("--device", default="auto", help="auto, cpu lub cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = (
        Config.from_dict(read_checkpoint(args.resume)["config"])
        if args.resume
        else Config.load(args.config)
    )
    run_dir = args.run or (args.resume.parent if args.resume else Path("models/piano_small"))
    if args.resume and run_dir.resolve() != args.resume.parent.resolve():
        parser.error("Wznowienie zapisuje wyniki w tym samym folderze co checkpoint.")
    epochs = args.epochs if args.epochs is not None else config.training.epochs
    train(
        args.data, run_dir, config, choose_device(args.device), epochs, args.resume, args.initialize
    )


if __name__ == "__main__":
    main()
