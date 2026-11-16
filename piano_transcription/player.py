"""Transkrypcja audio i odtwarzanie rozpoznanych nut na klawiaturze w Windows."""

import argparse
import csv
import math
import tempfile
import time
import tkinter as tk
import wave
import winsound
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .states import MIN_MIDI, NUM_KEYS, note_name

DEFAULT_DURATION = 0.25
SAMPLE_RATE = 44100
BLACK_KEYS = {1, 3, 6, 8, 10}
HIGHLIGHT = "#238D86"


@dataclass
class PlaybackNote:
    pitch: int
    start: float
    end: float


def load_notes(path: Path) -> list[PlaybackNote]:
    """Wczytuje początki nut i uzupełnia ich końce z sąsiedniego states.txt."""
    notes = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file, delimiter="\t")
        if not {"time_sec", "midi"}.issubset(rows.fieldnames or []):
            raise ValueError("Wybierz notes.txt z kolumnami time_sec i midi.")
        for row in rows:
            start, pitch = float(row["time_sec"]), int(row["midi"])
            if not math.isfinite(start) or start < 0 or not MIN_MIDI <= pitch < MIN_MIDI + NUM_KEYS:
                raise ValueError(f"Błędny czas lub numer MIDI w wierszu {rows.line_num}.")
            notes.append(PlaybackNote(pitch, start, start + DEFAULT_DURATION))

    notes.sort(key=lambda note: note.start)
    states_path = path.with_name("states.txt")
    if notes and path.name == "notes.txt" and states_path.is_file():
        starts = {(note.start, note.pitch): note for note in notes}
        active = {}
        with states_path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                timestamp, pitch = float(row["time_sec"]), int(row["midi"])
                new_note = starts.get((timestamp, pitch))
                if new_note is not None or row["state"] in ("off", "offset"):
                    previous = active.pop(pitch, None)
                    if previous is not None and timestamp > previous.start:
                        previous.end = timestamp
                if new_note is not None:
                    active[pitch] = new_note
        for note in active.values():
            if timestamp > note.start:
                note.end = timestamp

    # Kolejny atak tego samego klawisza kończy poprzednią nutę.
    next_start = {}
    for note in reversed(notes):
        if note.pitch in next_start and next_start[note.pitch] > note.start:
            note.end = min(note.end, next_start[note.pitch])
        next_start[note.pitch] = note.start
    return notes


def synthesize(notes: list[PlaybackNote], destination: Path) -> None:
    """Zapisuje nagranie syntezowanych nut, z zachowaniem akordów i przerw."""
    signal = np.zeros(math.ceil(max(note.end for note in notes) * SAMPLE_RATE), dtype=np.float32)
    for note in notes:
        start = round(note.start * SAMPLE_RATE)
        end = round(note.end * SAMPLE_RATE)
        seconds = np.arange(end - start) / SAMPLE_RATE
        frequency = 440 * 2 ** ((note.pitch - 69) / 12)
        phase = 2 * np.pi * frequency * seconds
        tone = np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.10 * np.sin(3 * phase)
        attack = np.minimum(1, seconds / 0.005)
        release = np.minimum(1, (seconds.size / SAMPLE_RATE - seconds) / 0.03)
        signal[start:end] += tone * attack * release * np.exp(-2.5 * seconds)

    peak = float(np.max(np.abs(signal)))
    samples = (signal * (28000 / max(1.0, peak))).astype("<i2")
    with wave.open(str(destination), "wb") as file:
        file.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        file.writeframes(samples.tobytes())


class PianoPlayer:
    def __init__(self, window: tk.Tk, checkpoint: Path | None = None):
        self.window = window
        self.checkpoint = checkpoint
        self.worker = ThreadPoolExecutor(max_workers=1) if checkpoint else None
        self.notes: list[PlaybackNote] = []
        self.active: dict[int, float] = {}
        self.cursor = 0
        self.duration = 0.0
        self.started = 0.0
        self.timer: str | None = None
        self.temporary = tempfile.TemporaryDirectory(prefix="piano-player-")
        self.audio = Path(self.temporary.name) / "playback.wav"

        window.title("Odtwarzacz transkrypcji")
        window.geometry("1120x330")
        window.minsize(850, 315)
        window.configure(background="#F3F5F7")
        window.protocol("WM_DELETE_WINDOW", self.close)

        panel = ttk.Frame(window, padding=20)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="Odtwarzacz transkrypcji", font=("Segoe UI", 20, "bold")).pack(
            anchor="w"
        )
        controls = ttk.Frame(panel)
        controls.pack(fill="x", pady=12)
        if self.worker is not None:
            self.audio_button = ttk.Button(
                controls, text="Wybierz nagranie…", command=self.choose_audio
            )
            self.audio_button.pack(side="left", padx=(0, 8))
        self.notes_button = ttk.Button(controls, text="Otwórz notes.txt", command=self.choose_file)
        self.notes_button.pack(side="left")
        self.play_button = ttk.Button(controls, text="Odtwórz od początku", command=self.play)
        self.play_button.pack(side="left", padx=8)
        self.play_button.state(["disabled"])
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop)
        self.stop_button.pack(side="left")
        self.stop_button.state(["disabled"])
        self.clock = ttk.Label(controls, text="0.00 / 0.00 s", font=("Segoe UI", 11))
        self.clock.pack(side="right")
        prompt = (
            f"Model: {checkpoint.parent.name} · wybierz plik WAV lub FLAC."
            if checkpoint else "Wybierz plik z rozpoznanymi nutami."
        )
        self.status = ttk.Label(panel, text=prompt, wraplength=785)
        self.status.pack(anchor="w")
        self.canvas = tk.Canvas(panel, background="#273142", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=(12, 0))
        self.canvas.bind("<Configure>", lambda _: self.draw_keys())

    def draw_keys(self):
        canvas = self.canvas
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        pitches = range(MIN_MIDI, MIN_MIDI + NUM_KEYS)
        key_width = width / 52
        whites = 0
        for pitch in pitches:
            if pitch % 12 not in BLACK_KEYS:
                left = whites * key_width
                canvas.create_rectangle(
                    left,
                    0,
                    left + key_width,
                    height,
                    fill=HIGHLIGHT if pitch in self.active else "#FFFFFF",
                    outline="#CBD1DA",
                    tags=f"key{pitch}",
                )
                if pitch % 12 == 0:
                    canvas.create_text(
                        left + key_width / 2,
                        height - 12,
                        text=note_name(pitch),
                        font=("Segoe UI", 8),
                        fill="#364152",
                    )
                whites += 1
        whites = 0
        for pitch in pitches:
            if pitch % 12 in BLACK_KEYS:
                center = whites * key_width
                canvas.create_rectangle(
                    center - key_width * 0.30,
                    0,
                    center + key_width * 0.30,
                    height * 0.61,
                    fill=HIGHLIGHT if pitch in self.active else "#273142",
                    outline="#17202D",
                    tags=f"key{pitch}",
                )
            else:
                whites += 1

    def choose_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Rozpoznane nuty", "*.txt")])
        if filename:
            self.open_file(Path(filename))

    def choose_audio(self):
        filename = filedialog.askopenfilename(
            parent=self.window, filetypes=[("Nagrania WAV i FLAC", ("*.wav", "*.flac"))]
        )
        if filename:
            self.transcribe_audio(Path(filename))

    def transcribe_audio(self, audio: Path):
        if self.worker is None or self.checkpoint is None:
            return
        self.audio_button.state(["disabled"])
        self.notes_button.state(["disabled"])
        self.status.configure(text=f"Rozpoznawanie nut z {audio.name}…")
        future = self.worker.submit(self.export_audio, audio, self.checkpoint)

        def poll():
            if not future.done():
                self.window.after(100, poll)
                return
            self.audio_button.state(["!disabled"])
            self.notes_button.state(["!disabled"])
            try:
                notes_path = future.result()
            except Exception as error:
                self.status.configure(text=f"Transkrypcja pliku {audio.name} nie powiodła się.")
                messagebox.showerror("Błąd transkrypcji", str(error), parent=self.window)
            else:
                self.open_file(notes_path)

        self.window.after(100, poll)

    @staticmethod
    def export_audio(audio: Path, checkpoint: Path) -> Path:
        import torch

        from .transcribe import Transcriber

        torch.set_num_threads(4)
        model = Transcriber(checkpoint)
        root = Path("results") / checkpoint.parent.name
        root.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=f"{audio.stem[:48]}-", dir=root))
        model.transcribe_file(audio, output)
        return output / "notes.txt"

    def open_file(self, path: Path):
        self.stop()
        self.notes = []
        self.duration = 0.0
        self.play_button.state(["disabled"])
        self.clock.configure(text="0.00 / 0.00 s")
        try:
            notes = load_notes(path)
            if notes:
                synthesize(notes, self.audio)
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
            self.status.configure(text="Nie udało się wczytać pliku.")
            messagebox.showerror("Błąd odczytu", str(error), parent=self.window)
            return
        self.notes = notes
        self.duration = max((note.end for note in notes), default=0.0)
        self.clock.configure(text=f"0.00 / {self.duration:.2f} s")
        self.status.configure(text=f"{path.parent.name}/{path.name} · {len(notes)} nut")
        if notes:
            self.play_button.state(["!disabled"])

    def play(self):
        self.stop()
        try:
            winsound.PlaySound(
                str(self.audio), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT
            )
        except RuntimeError as error:
            messagebox.showerror("Błąd odtwarzania", str(error), parent=self.window)
            return
        self.started = time.monotonic()
        self.cursor = 0
        self.stop_button.state(["!disabled"])
        self.tick()

    def tick(self):
        elapsed = min(time.monotonic() - self.started, self.duration)
        previous_keys = set(self.active)
        while self.cursor < len(self.notes) and self.notes[self.cursor].start <= elapsed:
            note = self.notes[self.cursor]
            self.active[note.pitch] = note.end
            self.cursor += 1
        self.active = {pitch: end for pitch, end in self.active.items() if end > elapsed}
        for pitch in previous_keys ^ self.active.keys():
            color = (
                HIGHLIGHT
                if pitch in self.active
                else ("#273142" if pitch % 12 in BLACK_KEYS else "#FFFFFF")
            )
            self.canvas.itemconfigure(f"key{pitch}", fill=color)
        self.clock.configure(text=f"{elapsed:.2f} / {self.duration:.2f} s")
        if elapsed < self.duration:
            self.timer = self.window.after(15, self.tick)
        else:
            self.timer = None
            self.stop_button.state(["disabled"])

    def stop(self):
        if self.timer is not None:
            self.window.after_cancel(self.timer)
            self.timer = None
        winsound.PlaySound(None, 0)
        self.active.clear()
        self.draw_keys()
        self.clock.configure(text=f"0.00 / {self.duration:.2f} s")
        self.stop_button.state(["disabled"])

    def close(self):
        self.stop()
        if self.worker is not None:
            self.worker.shutdown(wait=False, cancel_futures=True)
        self.temporary.cleanup()
        self.window.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notes", nargs="?", type=Path, help="Ścieżka do notes.txt")
    parser.add_argument("--checkpoint", type=Path, help="Włącz transkrypcję WAV i FLAC tym modelem")
    args = parser.parse_args()
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"Nie znaleziono checkpointu: {args.checkpoint}")
    window = tk.Tk()
    player = PianoPlayer(window, args.checkpoint)
    if args.notes is not None:
        player.open_file(args.notes)
    window.mainloop()


if __name__ == "__main__":
    main()
