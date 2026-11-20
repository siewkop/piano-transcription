import zipfile
from fractions import Fraction
from pathlib import Path

import pytest

from score_follower.score.score_importer import ScoreImporter, ScoreImportError

RESOURCES = Path(__file__).resolve().parents[1] / "resources"
TYPES = {1: "eighth", 2: "quarter", 4: "half", 8: "whole"}


def note(step, octave, duration, *, voice=1, staff=1, chord=False, tie=None, grace=False,
         type_=None, extra=""):
    parts = ["<note>"]
    if grace:
        parts.append("<grace/>")
    if chord:
        parts.append("<chord/>")
    parts.append(f"<pitch><step>{step}</step><octave>{octave}</octave></pitch>")
    if not grace:
        parts.append(f"<duration>{duration}</duration>")
    if tie:
        parts.append(f'<tie type="{tie}"/>')
    parts.append(f"<voice>{voice}</voice><type>{type_ or TYPES[duration]}</type>{extra}")
    parts.append(f"<staff>{staff}</staff>")
    if tie:
        parts.append(f'<notations><tied type="{tie}"/></notations>')
    parts.append("</note>")
    return "".join(parts)


def rest(duration, *, voice=1, staff=1, type_=None):
    return (f"<note><rest/><duration>{duration}</duration><voice>{voice}</voice>"
            f"<type>{type_ or TYPES[duration]}</type><staff>{staff}</staff></note>")


def backup(duration):
    return f"<backup><duration>{duration}</duration></backup>"


def musicxml(*measures, divisions=2):
    attributes = (
        f"<attributes><divisions>{divisions}</divisions><key><fifths>0</fifths></key>"
        "<time><beats>4</beats><beat-type>4</beat-type></time><staves>2</staves>"
        '<clef number="1"><sign>G</sign><line>2</line></clef>'
        '<clef number="2"><sign>F</sign><line>4</line></clef></attributes>'
    )
    body = "".join(
        f'<measure number="{i + 1}">{attributes if i == 0 else ""}{m}</measure>'
        for i, m in enumerate(measures)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<score-partwise version="3.1">\n'
        '<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>\n'
        f'<part id="P1">{body}</part>\n'
        "</score-partwise>\n"
    )


@pytest.fixture
def write(tmp_path):
    def _write(text, name="score.musicxml"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    return _write


def load(path):
    return ScoreImporter().load(path)


def test_chord_and_both_hands_form_one_event(write):
    path = write(musicxml(
        note("C", 4, 2) + note("E", 4, 2, chord=True) + note("G", 4, 2, chord=True)
        + note("D", 5, 2) + note("E", 5, 4)
        + backup(8) + note("C", 3, 8, voice=5, staff=2)
    ))
    events = load(path).document.events

    assert [e.pitches for e in events] == [
        frozenset({60, 64, 67, 48}), frozenset({74}), frozenset({76})
    ]
    assert [e.onset_q for e in events] == [0, 1, 2]
    assert [e.index for e in events] == [0, 1, 2]
    assert [(e.measure_index, e.offset_in_measure_q) for e in events] == [(0, 0), (0, 1), (0, 2)]


def test_tied_note_does_not_create_event(write):
    path = write(musicxml(
        note("C", 4, 8, tie="start"),
        note("C", 4, 4, tie="stop") + note("D", 4, 4),
    ))
    events = load(path).document.events

    assert len(events) == 2
    second = events[1]
    assert second.pitches == frozenset({62})
    assert second.onset_q == 6
    assert (second.measure_index, second.offset_in_measure_q) == (1, 2)


def test_triplets_are_exact_fractions(write):
    triplet = "<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>"
    path = write(musicxml(
        note("C", 4, 1, extra=triplet) + note("D", 4, 1, extra=triplet)
        + note("E", 4, 1, extra=triplet) + rest(9, type_="half"),
        divisions=3,
    ))
    events = load(path).document.events

    assert [e.onset_q for e in events] == [0, Fraction(1, 3), Fraction(2, 3)]
    assert all(isinstance(e.onset_q, Fraction) for e in events)


def test_grace_notes_are_skipped_with_warning(write):
    path = write(musicxml(note("D", 4, 0, grace=True, type_="eighth") + note("C", 4, 8)))
    result = load(path)

    assert [e.pitches for e in result.document.events] == [frozenset({60})]
    assert any("grace notes" in w for w in result.warnings)


def test_document_metadata(write):
    path = write(musicxml(note("C", 4, 8)))
    importer = ScoreImporter()
    first, second = importer.load(path).document, importer.load(path).document

    assert (first.score_id, second.score_id) == ("score-1", "score-2")
    assert first.content_hash == second.content_hash
    assert first.part_id == "P1"
    assert first.music_xml == path.read_bytes().decode("utf-8")


def test_compressed_mxl(tmp_path):
    xml = musicxml(note("C", 4, 8))
    path = tmp_path / "score.mxl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0" encoding="UTF-8"?><container><rootfiles>'
            '<rootfile full-path="score.xml"/></rootfiles></container>',
        )
        archive.writestr("score.xml", xml)
    document = load(path).document

    assert document.music_xml == xml
    assert [e.pitches for e in document.events] == [frozenset({60})]


def test_missing_file(tmp_path):
    with pytest.raises(ScoreImportError):
        load(tmp_path / "nope.musicxml")


def test_not_musicxml(write):
    with pytest.raises(ScoreImportError):
        load(write("to nie jest xml"))


def test_only_rests(write):
    with pytest.raises(ScoreImportError):
        load(write(musicxml(rest(8))))


@pytest.mark.parametrize("name", ["mozart_fantasie_475.musicxml", "Canon_in_D.mxl"])
def test_resource_scores_load(name):
    document = load(RESOURCES / name).document

    assert document.events
    assert [e.index for e in document.events] == list(range(len(document.events)))
    onsets = [e.onset_q for e in document.events]
    assert onsets == sorted(onsets) and len(set(onsets)) == len(onsets)
    assert all(0 <= e.offset_in_measure_q for e in document.events)
    assert all(21 <= p <= 108 for e in document.events for p in e.pitches)
