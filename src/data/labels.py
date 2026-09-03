from __future__ import annotations

"""Central MIT-BIH to AAMI-style label mapping.

N: N, L, R, e, j
S: A, a, J, S
V: V, E
F: F
Q: /, f, Q, |, ~, !, +, [, ], ", x
"""

from dataclasses import dataclass


AAMI_LABEL_MAP: dict[str, str] = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
    "F": "F",
    "/": "Q",
    "f": "Q",
    "Q": "Q",
    "|": "Q",
    "~": "Q",
    "!": "Q",
    "+": "Q",
    "[": "Q",
    "]": "Q",
    '"': "Q",
    "x": "Q",
}


@dataclass(frozen=True)
class LabelInfo:
    original_symbol: str
    mapped_class: str


def map_mitdb_symbol(symbol: str, unknown_class: str = "Q") -> LabelInfo:
    """Map a detailed MIT-BIH annotation symbol into an AAMI-style class."""
    return LabelInfo(original_symbol=symbol, mapped_class=AAMI_LABEL_MAP.get(symbol, unknown_class))


def label_mapping_table() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for symbol, mapped in AAMI_LABEL_MAP.items():
        grouped.setdefault(mapped, []).append(symbol)
    return dict(sorted(grouped.items()))
