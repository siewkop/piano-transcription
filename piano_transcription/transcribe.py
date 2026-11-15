"""Prosty interfejs plikowy oraz klasa Transcriber do późniejszej integracji z M5."""

import argparse
import csv
import json
from pathlib import Path

import torch

from .audio import LogMelSpectrogram, load_audio
from .checkpoint import choose_device, load_model
from .decoder import NoteDecoder
from .states import MIN_MIDI, NUM_KEYS, STATE_NAMES, NoteEvent, note_name


class Transcriber:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        self.checkpoint = Path(checkpoint)
        self.model = load_model(self.checkpoint, choose_device(device))
        self.frontend = LogMelSpectrogram(self.model.config.audio)

    @torch.inference_mode()
    def transcribe_file(
        self, audio_path: str | Path, output_dir: str | Path, chunk_frames: int = 512
    ) -> list[NoteEvent]:
        """Zwraca początki nut; pełne rozkłady zapisuje stopniowo do states.txt."""
        audio_path, output_dir = Path(audio_path), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_config = self.model.config.audio
        waveform, duration = load_audio(audio_path, audio_config.sample_rate)
        mel = self.frontend(waveform)
        decoder = NoteDecoder(audio_config.frame_seconds)
        events = []
        probability_columns = [f"p_{state}" for state in STATE_NAMES]
        with (
            (output_dir / "states.txt").open("w", encoding="utf-8", newline="") as state_file,
            (output_dir / "notes.txt").open("w", encoding="utf-8", newline="") as note_file,
        ):
            state_writer = csv.writer(state_file, delimiter="\t")
            note_writer = csv.writer(note_file, delimiter="\t")
            columns = ["frame", "time_sec", "midi", "note", "state", *probability_columns]
            state_writer.writerow(columns)
            note_writer.writerow(columns)
            start = 0
            for logits in self.model.iter_logits(mel, chunk_frames):
                probabilities = logits.softmax(dim=-1)
                batch_events = decoder.process(probabilities)
                events.extend(batch_events)
                states = probabilities.argmax(-1).tolist()
                rows = probabilities.tolist()
                for local_frame, frame_probabilities in enumerate(rows):
                    frame = start + local_frame
                    for key in range(NUM_KEYS):
                        pitch = key + MIN_MIDI
                        state_writer.writerow(
                            [
                                frame,
                                f"{frame * audio_config.frame_seconds:.6f}",
                                pitch,
                                note_name(pitch),
                                STATE_NAMES[states[local_frame][key]],
                                *(f"{value:.8f}" for value in frame_probabilities[key]),
                            ]
                        )
                for event in batch_events:
                    note_writer.writerow(
                        [
                            event.frame_index,
                            f"{event.onset_sec:.6f}",
                            event.pitch,
                            note_name(event.pitch),
                            STATE_NAMES[event.state],
                            *(f"{value:.8f}" for value in event.probabilities),
                        ]
                    )
                start += len(logits)
        metadata = {
            "audio": str(audio_path.resolve()),
            "checkpoint": str(self.checkpoint.resolve()),
            "duration_sec": duration,
            "frames": len(mel),
            "note_events": len(events),
            "frame_seconds": audio_config.frame_seconds,
            "state_order": list(STATE_NAMES),
            "time_reference": "frame center; seconds from start of audio",
            "config": self.model.config.to_dict(),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return events


def main():
    parser = argparse.ArgumentParser(
        description="Rozpoznaj nuty z audio i zapisz stany oraz NoteEvent."
    )
    parser.add_argument("audio", type=Path, help="Plik WAV lub FLAC, np. input/utwor.wav")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/piano_small/best.pt"))
    parser.add_argument("--output", type=Path, help="Domyślnie results/<nazwa nagrania>")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-frames", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = args.output or Path("results") / args.audio.stem
    transcriber = Transcriber(args.checkpoint, args.device)
    events = transcriber.transcribe_file(args.audio, output, args.chunk_frames)
    print(f"Rozpoznane początki nut: {len(events)}")
    print(f"NoteEvent: {output / 'notes.txt'}")
    print(f"Wszystkie ramki, 88 klawiszy i 5 prawdopodobieństw: {output / 'states.txt'}")


if __name__ == "__main__":
    main()
