"""Evaluate the frozen MAESTRO model on all 36 Batik Disklavier recordings."""

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pretty_midi
import torch

from piano_transcription.audio import LogMelSpectrogram, load_audio
from piano_transcription.checkpoint import choose_device, load_model
from piano_transcription.decoder import attack_starts
from piano_transcription.labels import read_midi_notes
from piano_transcription.metrics import detection_metrics, match_onsets
from piano_transcription.states import NoteState

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
OUT = ROOT / "results/batik_600"
CHECKPOINT = ROOT / "models/maestro_600_events/best.pt"


def write_json(path, data):
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


@torch.inference_mode()
def run():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model(CHECKPOINT, choose_device("auto"))
    frontend = LogMelSpectrogram(model.config.audio)
    step = model.config.audio.frame_seconds
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    rows = list(csv.DictReader((RAW / "batik/metadata-batik.csv").open()))
    results = []
    for index, row in enumerate(rows, 1):
        name = Path(row["midi"]).stem
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        result_path = folder / "metrics.json"
        if result_path.exists():
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if saved["checkpoint_sha256"] != digest:
                raise ValueError("Checkpoint changed, use a new output folder")
            results.append(saved)
            continue
        audio, midi = RAW / row["audio"], RAW / row["midi"]
        waveform, duration = load_audio(audio, model.config.audio.sample_rate)
        mel = frontend(waveform)
        predictions = []
        notes = []
        active = {}
        previous = torch.zeros(88, dtype=torch.long)
        offset = 0
        for logits in model.iter_logits(mel):
            states = logits.argmax(-1)
            attacks = set(map(tuple, attack_starts(states, previous).tolist()))
            for frame, values in enumerate(states.tolist()):
                timestamp = (offset + frame) * step
                for key, state in enumerate(values):
                    onset = (frame, key) in attacks
                    if onset or state in (NoteState.OFF, NoteState.OFFSET):
                        start = active.pop(key, None)
                        if start is not None:
                            notes.append(
                                pretty_midi.Note(80, key + 21, start, max(start + step, timestamp))
                            )
                    if onset:
                        predictions.append((key + 21, timestamp))
                        active[key] = timestamp
            previous = states[-1]
            offset += len(states)
        for key, start in active.items():
            notes.append(pretty_midi.Note(80, key + 21, start, max(start + step, duration)))
        transcription = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
        piano.notes = sorted(notes, key=lambda n: (n.start, n.pitch))
        transcription.instruments.append(piano)
        transcription.write(str(folder / "transcription.mid"))
        reference = [(n.pitch, n.start) for n in read_midi_notes(midi)]
        metrics = detection_metrics(
            match_onsets(reference, predictions), len(predictions), len(reference)
        )
        result = dict(
            name=name,
            audio=Path(os.path.relpath(audio, OUT)).as_posix(),
            reference_midi=Path(os.path.relpath(midi, OUT)).as_posix(),
            transcription_midi=(folder / "transcription.mid").relative_to(OUT).as_posix(),
            duration_sec=duration,
            checkpoint_sha256=digest,
            **metrics,
        )
        result["false_positive_count"] = len(predictions) - metrics["matched_count"]
        result["missed_count"] = len(reference) - metrics["matched_count"]
        result["false_discovery_rate"] = 1 - metrics["precision"]
        result["f1_100ms"] = detection_metrics(
            match_onsets(reference, predictions, 0.1), len(predictions), len(reference)
        )["f1"]
        write_json(result_path, result)
        write_json(folder / "onsets.json", {"reference": reference, "predicted": predictions})
        results.append(result)
        print(
            f"[{index}/{len(rows)}] {name}: P={metrics['precision']:.3%} R={metrics['recall']:.3%} F1={metrics['f1']:.3%}",
            flush=True,
        )
    aggregate = detection_metrics(
        sum(r["matched_count"] for r in results),
        sum(r["predicted_count"] for r in results),
        sum(r["reference_count"] for r in results),
    )
    aggregate.update(
        macro_f1=float(np.mean([r["f1"] for r in results])),
        false_positive_count=sum(r["false_positive_count"] for r in results),
        missed_count=sum(r["missed_count"] for r in results),
        false_discovery_rate=1 - aggregate["precision"],
    )
    summary = {
        "checkpoint": Path(os.path.relpath(CHECKPOINT, OUT)).as_posix(),
        "checkpoint_sha256": digest,
        "tolerance_sec": 0.05,
        "audio_source": "Batik performance MIDI replayed on JKU Yamaha Disklavier",
        "dataset_url": "https://github.com/pymatchmaker/matchmaker-benchmark-data",
        "recordings": len(results),
        "aggregate": aggregate,
        "per_recording": results,
    }
    write_json(OUT / "summary.json", summary)
    with (OUT / "per_recording.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(aggregate), flush=True)


if __name__ == "__main__":
    run()
