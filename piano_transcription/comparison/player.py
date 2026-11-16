"""Window for auditioning Batik recordings and piano MIDI transcriptions."""

import argparse
import json
import os
import tempfile
import time
import tkinter as tk
import winsound
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from tkinter import messagebox, ttk

import mido
import soundfile as sf

from .midi_output import Song, output_ports, play_song


def midi_song(path):
    events = []
    timestamp = 0.0
    for channel in range(16):
        events.append(
            (0.0, bytes(mido.Message("program_change", channel=channel, program=0).bin()))
        )
    for message in mido.MidiFile(path):
        timestamp += message.time
        if message.is_meta or message.type == "sysex":
            continue
        if message.type == "program_change":
            message.program = 0
        events.append((timestamp, bytes(message.bin())))
    return Song(Path(path).read_bytes(), tuple(events), timestamp)


class ComparisonPlayer:
    def __init__(self, window, summary):
        self.window = window
        self.summary = summary.resolve()
        self.data = json.loads(summary.read_text(encoding="utf-8"))
        self.rows = self.data["per_recording"]
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.stop_event = Event()
        self.future = None
        self.started = None
        self.closing = False
        self.temporary = tempfile.TemporaryDirectory(prefix="batik-listen-")
        self.ports = output_ports()
        window.title("Batik · oryginał i transkrypcja fortepianowa")
        window.geometry("850x430")
        window.minsize(760, 410)
        window.protocol("WM_DELETE_WINDOW", self.close)
        panel = ttk.Frame(window, padding=24)
        panel.pack(fill="both", expand=True)
        ttk.Label(
            panel, text="Mozart · test modelu na nagraniach Batika", font=("Segoe UI", 18, "bold")
        ).pack(anchor="w")
        ttk.Label(
            panel,
            text="Nagrania JKU Disklavier · model MAESTRO 600 · tolerancja początku nuty 50 ms",
        ).pack(anchor="w", pady=(5, 18))
        self.selection = ttk.Combobox(
            panel, state="readonly", values=[r["name"] for r in self.rows]
        )
        self.selection.pack(fill="x")
        self.selection.current(0)
        self.selection.bind("<<ComboboxSelected>>", self.select)
        self.metrics = ttk.Label(panel, wraplength=790, font=("Segoe UI", 11))
        self.metrics.pack(anchor="w", pady=15)
        self.port_box = ttk.Combobox(panel, state="readonly", values=[p.name for p in self.ports])
        self.port_box.pack(fill="x")
        if self.ports:
            self.port_box.current(next((i for i, p in enumerate(self.ports) if not p.hardware), 0))
        controls = ttk.Frame(panel)
        controls.pack(fill="x", pady=15)
        self.buttons = []
        for label, kind in [
            ("Oryginał audio", "original"),
            ("Transkrypcja · piano", "transcription"),
            ("Wzorcowe MIDI · piano", "reference"),
        ]:
            button = ttk.Button(controls, text=label, command=lambda k=kind: self.play(k))
            button.pack(side="left", padx=(0, 8))
            self.buttons.append(button)
        ttk.Button(controls, text="Stop", command=self.stop).pack(side="left")
        self.status = ttk.Label(panel, text="Wybierz utwór i wersję do odsłuchu.")
        self.status.pack(anchor="w")
        self.progress = ttk.Progressbar(panel)
        self.progress.pack(fill="x", pady=10)
        ttk.Button(
            panel,
            text="Otwórz folder wyników i MIDI",
            command=lambda: os.startfile(str(self.summary.parent)),
        ).pack(anchor="w")
        self.select()

    def select(self, event=None):
        row = self.rows[self.selection.current()]
        self.metrics.configure(
            text=(
                f"Wykryte nuty wzorcowe: {row['recall']:.1%} · Fałszywe wśród wykrytych: {row['false_discovery_rate']:.1%} · F1: {row['f1']:.1%}\n"
                f"Trafione {row['matched_count']} · Fałszywe {row['false_positive_count']} · Pominięte {row['missed_count']}"
            )
        )

    def play(self, kind):
        row = self.rows[self.selection.current()]
        if kind != "original" and self.port_box.current() < 0:
            messagebox.showerror(
                "Wyjście MIDI", "Brak dostępnego syntezatora MIDI.", parent=self.window
            )
            return
        self.stop_event.clear()
        self.started = None
        self.duration = row["duration_sec"]
        self.status.configure(text="Przygotowuję odtwarzanie…")
        for button in self.buttons:
            button.state(["disabled"])
        self.selection.configure(state="disabled")
        self.port_box.configure(state="disabled")
        port = self.ports[self.port_box.current()] if self.ports else None
        self.future = self.worker.submit(self.playback, row, kind, port)
        self.window.after(100, self.poll)

    def playback(self, row, kind, port):
        if kind == "original":
            wav = Path(self.temporary.name) / "original.wav"
            samples, rate = sf.read(self.summary.parent / row["audio"], dtype="float32")
            sf.write(wav, samples, rate, subtype="PCM_16")
            if self.stop_event.is_set():
                return
            self.started = time.monotonic()
            winsound.PlaySound(
                str(wav), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT
            )
            self.stop_event.wait(len(samples) / rate)
            winsound.PlaySound(None, 0)
        else:
            path = row["transcription_midi"] if kind == "transcription" else row["reference_midi"]
            song = midi_song(self.summary.parent / path)
            self.started = time.monotonic()
            play_song(song, port, self.stop_event)

    def poll(self):
        if self.future.done():
            error = self.future.exception()
            if self.closing:
                self.finish_close()
                return
            self.selection.configure(state="readonly")
            self.port_box.configure(state="readonly")
            for button in self.buttons:
                button.state(["!disabled"])
            self.status.configure(
                text="Zatrzymano." if self.stop_event.is_set() else "Odtwarzanie zakończone."
            )
            if error:
                messagebox.showerror("Odtwarzanie", str(error), parent=self.window)
            return
        if self.started is not None:
            elapsed = min(time.monotonic() - self.started, self.duration)
            self.status.configure(text=f"{elapsed:.1f} / {self.duration:.1f} s")
            self.progress["value"] = 100 * elapsed / max(self.duration, 1)
        self.window.after(100, self.poll)

    def stop(self):
        self.stop_event.set()

    def close(self):
        self.closing = True
        self.stop()
        if self.future is not None and not self.future.done():
            self.window.withdraw()
        else:
            self.finish_close()

    def finish_close(self):
        self.worker.shutdown(wait=False)
        self.temporary.cleanup()
        self.window.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/batik_600/summary.json"))
    args = parser.parse_args()
    window = tk.Tk()
    ComparisonPlayer(window, args.summary)
    window.mainloop()


if __name__ == "__main__":
    main()
