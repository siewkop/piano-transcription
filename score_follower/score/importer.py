import hashlib
import itertools
import warnings
import zipfile
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import partitura
from partitura.score import GraceNote, Part

from ..domain import ScoreDocument, ScoreEvent


class ScoreImportError(Exception):
    ## TO DO: some kind of log message like: Coudn't import the score
    pass


@dataclass(frozen=True)
class ImportResult:
    document: ScoreDocument
    warnings: tuple[str, ...]


class ScoreImporter:
    def __init__(self):
        self._ids = itertools.count(1)

    def load(self, path: str | Path) -> ImportResult:
        path = Path(path)
        if not path.is_file():
            raise ScoreImportError(f"File does not exist: {path}")
        raw = path.read_bytes()
        music_xml = _read_music_xml(path, raw)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                score = partitura.load_musicxml(path)
            except Exception as error:
                raise ScoreImportError(f"Invalid MusicXML file: {error}") from error
        messages = [str(w.message) for w in caught]

        if len(score.parts) == 0:
            raise ScoreImportError("The score does not contain any part.")
        if len(score.parts) > 1:
            messages.append(
                f"The file contains {len(score.parts)} parts; using the first one ({score.parts[0].id})."
            )
        part = score.parts[0]

        events, skipped_grace = build_events(part)
        if skipped_grace:
            messages.append(f"Skipped grace notes: {skipped_grace}.")
        if not events:
            raise ScoreImportError("The score does not contain any playable notes.")

        document = ScoreDocument(
            score_id=f"score-{next(self._ids)}",
            content_hash=hashlib.sha256(raw).hexdigest(),
            music_xml=music_xml,
            part_id=part.id,
            events=tuple(events),
        )
        return ImportResult(document, tuple(messages))


def build_events(part: Part) -> tuple[list[ScoreEvent], int]:
    to_quarters = _quarter_converter(part)
    measure_starts = [m.start.t for m in part.measures]
    if not measure_starts:
        raise ScoreImportError("The score does not contain any measures.")

    notes = [n for n in part.notes_tied if not isinstance(n, GraceNote)]
    skipped_grace = len(part.notes_tied) - len(notes)

    pitches_by_time: dict[int, set[int]] = defaultdict(set)
    for note in notes:
        pitches_by_time[note.start.t].add(note.midi_pitch)

    events = []
    for index, t in enumerate(sorted(pitches_by_time)):
        measure_index = max(bisect_right(measure_starts, t) - 1, 0)
        onset_q = to_quarters(t)
        events.append(
            ScoreEvent(
                index=index,
                pitches=frozenset(pitches_by_time[t]),
                onset_q=onset_q,
                measure_index=measure_index,
                offset_in_measure_q=onset_q - to_quarters(measure_starts[measure_index]),
            )
        )
    return events, skipped_grace


def _quarter_converter(part: Part):
    segments = [(int(t), int(d)) for t, d in part.quarter_durations()]
    starts = [t for t, _ in segments]
    cumulative = [Fraction(0)]
    for (t0, d0), (t1, _) in zip(segments, segments[1:]):
        cumulative.append(cumulative[-1] + Fraction(t1 - t0, d0))

    def to_quarters(t: int) -> Fraction:
        i = max(bisect_right(starts, t) - 1, 0)
        seg_t, divs = segments[i]
        return cumulative[i] + Fraction(t - seg_t, divs)

    return to_quarters


def _read_music_xml(path: Path, raw: bytes) -> str:
    if zipfile.is_zipfile(path):
        from xml.etree import ElementTree

        try:
            with zipfile.ZipFile(path) as archive:
                container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
                rootfile = container.find(".//{*}rootfile")
                if rootfile is None:
                    rootfile = container.find(".//rootfile")
                raw = archive.read(rootfile.get("full-path"))
        except (KeyError, AttributeError, zipfile.BadZipFile, ElementTree.ParseError) as error:
            raise ScoreImportError(f"Corrupted .mxl file: {error}") from error
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("utf-16")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Print score events from a MusicXML file.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    result = ScoreImporter().load(args.path)
    doc = result.document
    print(f"{doc.score_id}  part={doc.part_id}  events={len(doc.events)}  sha256={doc.content_hash[:12]}")
    for message in result.warnings:
        print(f"  warning: {message}")
    for event in doc.events[: args.limit]:
        pitches = " ".join(str(p) for p in sorted(event.pitches))
        print(
            f"{event.index:4d}  measure {event.measure_index:3d} + {str(event.offset_in_measure_q):>5}"
            f"  onsetQ={str(event.onset_q):>7}  [{pitches}]"
        )


if __name__ == "__main__":
    main()
