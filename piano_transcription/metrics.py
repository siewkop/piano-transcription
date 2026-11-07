"""Dopasowanie początków nut o tej samej wysokości z tolerancją czasu."""

from collections import defaultdict


def detection_metrics(hits: int, predicted: int, reference: int) -> dict:
    return {
        "precision": hits / max(predicted, 1),
        "recall": hits / max(reference, 1),
        "f1": 2 * hits / max(predicted + reference, 1),
        "matched_count": hits,
        "predicted_count": predicted,
        "reference_count": reference,
    }


def match_onsets(
    reference: list[tuple[int, float]],
    predicted: list[tuple[int, float]],
    tolerance: float = 0.05,
) -> int:
    """Maksymalna liczba par 1:1 dla MIDI pitch i początku w sekundach.

    Dla jednej wysokości i stałej tolerancji wystarczy przejście po czasach
    w kolejności. Najwcześniejsza możliwa para zostawia miejsce kolejnym nutom.
    Jedna długa aktywacja ani kilka predykcji nie zalicza wielokrotnie jednej nuty.
    """
    ref_by_pitch, pred_by_pitch = defaultdict(list), defaultdict(list)
    for pitch, time in reference:
        ref_by_pitch[pitch].append(time)
    for pitch, time in predicted:
        pred_by_pitch[pitch].append(time)
    hits = 0
    for pitch, times in ref_by_pitch.items():
        refs, preds = sorted(times), sorted(pred_by_pitch[pitch])
        i, j = 0, 0
        while i < len(refs) and j < len(preds):
            difference = preds[j] - refs[i]
            if difference < -tolerance - 1e-9:
                j += 1
            elif difference > tolerance + 1e-9:
                i += 1
            else:
                hits += 1
                i += 1
                j += 1
    return hits
