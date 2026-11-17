"""Choose which mass-difference connectors to draw, and which vertical tier each uses."""

from __future__ import annotations

from dataclasses import dataclass

from kmz2viewer.differences import PairAssignment


@dataclass(frozen=True)
class PlacedAnnotation:
    pair: PairAssignment
    tier: int


def annotation_priority(pair: PairAssignment) -> tuple:
    """Lower sorts first: precursor relationships, then taller peaks, then tighter ppm."""
    best = pair.best
    error = abs(best.ppm_error) if best is not None else 1e9
    height = max(pair.left_intensity, pair.right_intensity)
    return (
        0 if pair.involves_precursor else 1,
        -height,
        error,
        pair.left_mz,
        pair.right_mz,
    )


def place_annotations(
    pairs: list[PairAssignment],
    xmin: float,
    xmax: float,
    view_width_px: float,
    *,
    max_tiers: int = 6,
    gap_px: float = 10.0,
) -> list[PlacedAnnotation]:
    """Keep both endpoints inside the window. Hide a connector when every tier collides."""
    if xmax <= xmin or view_width_px < 30:
        return []
    span = float(xmax - xmin)

    def pixel(mz: float) -> float:
        return (mz - xmin) / span * view_width_px

    visible = [
        pair
        for pair in pairs
        if pair.best is not None and xmin <= pair.left_mz and pair.right_mz <= xmax
    ]
    visible.sort(key=annotation_priority)
    tiers: list[list[tuple[float, float]]] = []
    placed: list[PlacedAnnotation] = []
    for pair in visible:
        left = pixel(pair.left_mz)
        right = pixel(pair.right_mz)
        if right < left:
            left, right = right, left
        chosen = None
        for tier, occupied in enumerate(tiers):
            if all(right < start - gap_px or left > end + gap_px for start, end in occupied):
                occupied.append((left, right))
                chosen = tier
                break
        if chosen is None and len(tiers) < max_tiers:
            tiers.append([(left, right)])
            chosen = len(tiers) - 1
        if chosen is None:
            continue
        placed.append(PlacedAnnotation(pair=pair, tier=chosen))
    return placed
