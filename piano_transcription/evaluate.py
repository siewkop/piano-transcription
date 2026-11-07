"""Walidacja bez teacher forcing: model korzysta tylko z własnej historii."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .checkpoint import choose_device, load_model
from .dataset import load_index
from .decoder import attack_starts
from .labels import read_midi_notes
from .metrics import detection_metrics, match_onsets
from .model import PianoModel
from .states import MIN_MIDI, NUM_KEYS, NUM_STATES, STATE_NAMES, NoteState


@torch.inference_mode()
def evaluate(model: PianoModel, directory: Path, records: list[dict]) -> dict:
    model.eval()
    loss_sum, count = 0.0, 0
    confusion = torch.zeros(NUM_STATES, NUM_STATES, dtype=torch.long)
    pieces = []
    for record in records:
        data = torch.load(directory / record["file"], weights_only=True, mmap=True)
        start = 0
        predicted_notes = []
        previous = torch.zeros(NUM_KEYS, dtype=torch.long)
        for logits in model.iter_logits(data["mel"]):
            targets = data["labels"][start : start + len(logits)].long()
            loss_sum += F.cross_entropy(
                logits.reshape(-1, NUM_STATES), targets.flatten(), reduction="sum"
            ).item()
            count += targets.numel()
            states = logits.argmax(-1)
            predicted_notes.extend(
                (key + MIN_MIDI, (start + frame) * model.config.audio.frame_seconds)
                for frame, key in attack_starts(states, previous).tolist()
            )
            previous = states[-1]
            pairs = targets.flatten() * NUM_STATES + states.flatten()
            confusion += torch.bincount(pairs, minlength=NUM_STATES**2).reshape(
                NUM_STATES, NUM_STATES
            )
            start += len(logits)
        if "midi" in record:
            reference = [(n.pitch, n.start) for n in read_midi_notes(record["midi"])]
        else:
            # Małe syntetyczne cache testów nie zawierają oryginalnego MIDI.
            reference = [
                (key + MIN_MIDI, frame * model.config.audio.frame_seconds)
                for frame, key in attack_starts(
                    data["labels"].long(), torch.zeros(NUM_KEYS, dtype=torch.long)
                ).tolist()
            ]
        piece = detection_metrics(
            match_onsets(reference, predicted_notes), len(predicted_notes), len(reference)
        )
        pieces.append({"file": record["file"], **piece})
    per_state = {}
    for index, name in enumerate(STATE_NAMES):
        hits = int(confusion[index, index])
        predicted, reference = int(confusion[:, index].sum()), int(confusion[index].sum())
        per_state[name] = detection_metrics(hits, predicted, reference)
    attack_states = [NoteState.ONSET, NoteState.REONSET]
    attack_frames = detection_metrics(
        int(confusion[attack_states][:, attack_states].sum()),
        int(confusion[:, attack_states].sum()),
        int(confusion[attack_states, :].sum()),
    )
    return {
        "cross_entropy": loss_sum / count,
        "frame_accuracy": confusion.diag().sum().item() / count,
        "per_state": per_state,
        "attack_frames": attack_frames,
        "note_onsets": {
            **detection_metrics(
                sum(p["matched_count"] for p in pieces),
                sum(p["predicted_count"] for p in pieces),
                sum(p["reference_count"] for p in pieces),
            ),
            "macro_f1": sum(p["f1"] for p in pieces) / len(pieces),
            "onset_tolerance_sec": 0.05,
            "reference": "midi" if all("midi" in r for r in records) else "mixed_or_labels",
        },
        "per_recording": pieces,
        "confusion_matrix": confusion.tolist(),
        "recordings": len(records),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Oceń stany ramek i początki nut na wydzielonym zbiorze."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model = load_model(args.checkpoint, choose_device(args.device))
    records = [r for r in load_index(args.data, model.config) if r["split"] == args.split]
    if not records:
        parser.error(f"Brak nagrań w podzbiorze {args.split}.")
    metrics = evaluate(model, args.data, records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"CE: {metrics['cross_entropy']:.4f}; F1 ramek ataku: "
        f"{metrics['attack_frames']['f1']:.4f}; wyniki: {args.output}"
    )
    print(
        f"Nutowe F1 (50 ms): {metrics['note_onsets']['f1']:.4f}; "
        f"macro F1: {metrics['note_onsets']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
