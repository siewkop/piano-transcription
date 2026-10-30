"""Uruchom: python -m piano_transcription.prepare --help."""

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch

from .audio import LogMelSpectrogram, load_audio
from .config import Config
from .dataset import load_index
from .labels import make_labels, read_midi_notes


def read_manifest(path: Path, maestro: bool = False) -> list[dict]:
    """Ścieżki względne liczymy od CSV; zachowujemy oficjalny podział MAESTRO."""
    audio_column, midi_column = (
        ("audio_filename", "midi_filename") if maestro else ("audio", "midi")
    )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest nie zawiera nagrań.")
    records, seen_audio, seen_midi = [], set(), set()
    for row in rows:
        split = row["split"]
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Nieznany split: {split}. Użyj train, validation lub test.")
        audio = (path.parent / row[audio_column]).resolve()
        midi = (path.parent / row[midi_column]).resolve()
        if audio in seen_audio or midi in seen_midi:
            raise ValueError(f"Powtórzona para audio/MIDI w manifeście: {audio}")
        seen_audio.add(audio)
        seen_midi.add(midi)
        records.append({"audio": audio, "midi": midi, "split": split})
    return records


def prepare_dataset(
    manifest: Path,
    output: Path,
    config: Config,
    maestro: bool = False,
    reuse_cache: Path | None = None,
):
    records = read_manifest(manifest, maestro)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "index.json").exists():
        raise FileExistsError(f"Zbiór już istnieje: {output}. Wskaż nowy katalog cache.")
    frontend = LogMelSpectrogram(config.audio)
    cached = (
        {(r["audio"], r["midi"]): r for r in load_index(reuse_cache, config)} if reuse_cache else {}
    )
    index = []
    for number, record in enumerate(records, 1):
        previous = cached.get((str(record["audio"]), str(record["midi"])))
        if previous:
            shutil.copy2(reuse_cache / previous["file"], output / previous["file"])
            index.append({**previous, "split": record["split"]})
            print(f"[{number}/{len(records)}] Kopia cache: {record['audio'].name}", flush=True)
            continue
        waveform, duration = load_audio(record["audio"], config.audio.sample_rate)
        with torch.inference_mode():
            mel = frontend(waveform)
        notes = read_midi_notes(record["midi"])
        if notes and max(note.start for note in notes) >= duration:
            raise ValueError(f"MIDI zawiera początek nuty za końcem audio: {record['midi']}")
        labels = make_labels(notes, len(mel), config.audio)
        key = hashlib.sha256(str(record["audio"]).encode()).hexdigest()[:16]
        filename = f"{key}.pt"
        torch.save({"mel": mel, "labels": labels}, output / filename)
        index.append(
            {
                "file": filename,
                "split": record["split"],
                "frames": len(mel),
                "audio": str(record["audio"]),
                "midi": str(record["midi"]),
                "duration_sec": duration,
            }
        )
        print(f"[{number}/{len(records)}] {record['audio'].name}: {len(mel)} ramek", flush=True)
    metadata = {"format_version": 1, "audio": asdict(config.audio), "records": index}
    (output / "index.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="Przygotuj log-mel i etykiety z par audio/MIDI.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="CSV z kolumnami audio,midi,split")
    source.add_argument("--maestro", type=Path, help="Oryginalny CSV wewnątrz katalogu MAESTRO")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("configs/small.json"))
    parser.add_argument("--reuse-cache", type=Path, help="Cache tych samych, niezmienionych źródeł")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    prepare_dataset(
        args.maestro or args.manifest,
        args.output,
        Config.load(args.config),
        maestro=args.maestro is not None,
        reuse_cache=args.reuse_cache,
    )


if __name__ == "__main__":
    main()
