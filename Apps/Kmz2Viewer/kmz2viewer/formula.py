"""CHNOS exact-mass formula table and Δmass lookup.

Δm/z is treated as a neutral mass. That is the singly charged small-molecule
assumption: one m/z unit equals one dalton. O and divalent S do not change DBE.

DBE = C − H/2 + N/2 + 1

The cached table is rebuilt only when element limits or the DBE window change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MASS_C = 12.0
MASS_H = 1.00782503224
MASS_N = 14.00307400443
MASS_O = 15.99491461957
MASS_S = 31.9720711744

@dataclass(frozen=True)
class FormulaSettings:
    ppm: float = 50.0
    c_max: int = 20
    h_max: int = 50
    n_max: int = 10
    o_max: int = 15
    s_max: int = 2
    dbe_min: int = 0
    dbe_max: int = 15

    def table_key(self) -> tuple[int, ...]:
        return (
            self.c_max,
            self.h_max,
            self.n_max,
            self.o_max,
            self.s_max,
            self.dbe_min,
            self.dbe_max,
        )


@dataclass(frozen=True)
class FormulaHit:
    formula: str
    theoretical: float
    ppm_error: float
    counts: tuple[int, int, int, int, int]


@dataclass
class FormulaTable:
    masses: np.ndarray
    counts: np.ndarray  # columns C H N O S
    dbe: np.ndarray

    def __len__(self) -> int:
        return int(self.masses.shape[0])


def dbe_value(c: int, h: int, n: int, o: int = 0, s: int = 0) -> float:
    return c - h / 2.0 + n / 2.0 + 1.0


def format_formula(counts: tuple[int, int, int, int, int]) -> str:
    carbon, hydrogen, nitrogen, oxygen, sulfur = counts
    if carbon > 0:
        sequence = (("C", carbon), ("H", hydrogen), ("N", nitrogen), ("O", oxygen), ("S", sulfur))
    else:
        # Nitrogen hydrides are written NH3, while H2O stays hydrogen-first.
        sequence = (("N", nitrogen), ("H", hydrogen), ("O", oxygen), ("S", sulfur))
    parts: list[str] = []
    for symbol, count in sequence:
        if count <= 0:
            continue
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def build_formula_table(settings: FormulaSettings) -> FormulaTable:
    if settings.dbe_max <= settings.dbe_min:
        raise ValueError("DBE max must be greater than DBE min.")
    if min(settings.c_max, settings.h_max, settings.n_max, settings.o_max, settings.s_max) < 0:
        raise ValueError("Element limits cannot be negative.")
    c_axis = np.arange(settings.c_max + 1, dtype=np.int16)
    h_axis = np.arange(settings.h_max + 1, dtype=np.int16)
    n_axis = np.arange(settings.n_max + 1, dtype=np.int16)
    o_axis = np.arange(settings.o_max + 1, dtype=np.int16)
    s_axis = np.arange(settings.s_max + 1, dtype=np.int16)
    carbon, hydrogen, nitrogen, oxygen, sulfur = np.meshgrid(
        c_axis, h_axis, n_axis, o_axis, s_axis, indexing="ij"
    )
    carbon = carbon.ravel().astype(np.int32)
    hydrogen = hydrogen.ravel().astype(np.int32)
    nitrogen = nitrogen.ravel().astype(np.int32)
    oxygen = oxygen.ravel().astype(np.int32)
    sulfur = sulfur.ravel().astype(np.int32)
    # 2*DBE = 2C - H + N + 2. Even values are integer DBE.
    twice = 2 * carbon - hydrogen + nitrogen + 2
    keep = (
        (twice % 2 == 0)
        & (twice >= 2 * settings.dbe_min)
        & (twice < 2 * settings.dbe_max)
        & ((carbon + hydrogen + nitrogen + oxygen + sulfur) > 0)
    )
    carbon = carbon[keep]
    hydrogen = hydrogen[keep]
    nitrogen = nitrogen[keep]
    oxygen = oxygen[keep]
    sulfur = sulfur[keep]
    twice = twice[keep]
    masses = (
        carbon * MASS_C
        + hydrogen * MASS_H
        + nitrogen * MASS_N
        + oxygen * MASS_O
        + sulfur * MASS_S
    ).astype(np.float64)
    order = np.argsort(masses, kind="mergesort")
    counts = np.column_stack((carbon, hydrogen, nitrogen, oxygen, sulfur)).astype(np.int16)
    return FormulaTable(masses=masses[order], counts=counts[order], dbe=(twice[order] / 2).astype(np.int16))


class FormulaIndex:
    """Holds one sorted table. Rebuilds only when the table key changes."""

    def __init__(self) -> None:
        self._key: tuple[int, ...] | None = None
        self._table: FormulaTable | None = None

    def table(self, settings: FormulaSettings) -> FormulaTable:
        key = settings.table_key()
        if self._table is None or key != self._key:
            self._table = build_formula_table(settings)
            self._key = key
        return self._table

    def lookup(self, delta: float, settings: FormulaSettings, limit: int = 8) -> list[FormulaHit]:
        if not np.isfinite(delta) or delta <= 0 or settings.ppm < 0 or limit <= 0:
            return []
        table = self.table(settings)
        low = delta / (1.0 + settings.ppm * 1e-6)
        high = delta / (1.0 - settings.ppm * 1e-6) if settings.ppm * 1e-6 < 1 else np.inf
        left = int(np.searchsorted(table.masses, low, side="left"))
        right = int(np.searchsorted(table.masses, high, side="right"))
        if left >= right:
            return []
        masses = table.masses[left:right]
        ppm_error = (delta - masses) / masses * 1e6
        order = np.argsort(np.abs(ppm_error), kind="mergesort")
        hits: list[FormulaHit] = []
        for index in order[:limit]:
            row = left + int(index)
            counts = tuple(int(value) for value in table.counts[row])
            hits.append(
                FormulaHit(
                    formula=format_formula(counts),
                    theoretical=float(table.masses[row]),
                    ppm_error=float(ppm_error[index]),
                    counts=counts,  # type: ignore[arg-type]
                )
            )
        return hits
