from __future__ import annotations

import math


def pct(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{decimals}f}%"


def ratio(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{decimals}f}%"


def seconds(value: float | int | None) -> str:
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{float(value):,.4f} s"


def bytes_to_gb(value: float | int | None) -> str:
    return "—" if value is None else f"{float(value) / 1_000_000_000:.3f} GB"
