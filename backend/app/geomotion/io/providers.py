from __future__ import annotations

import csv
import io
from typing import Iterable


def _rows(text: str) -> list[dict[str, str]]:
    return [
        {str(key).strip().lower(): str(value).strip() for key, value in row.items()}
        for row in csv.DictReader(io.StringIO(text))
        if row and any(str(value or "").strip() for value in row.values())
    ]


def _number(row: dict[str, str], names: Iterable[str], required: bool = True) -> float | None:
    for name in names:
        value = row.get(name.lower())
        if value not in (None, ""):
            parsed = float(value)
            if parsed != parsed:
                break
            return parsed
    if required:
        raise ValueError(f"Missing numeric field; expected one of {', '.join(names)}")
    return None


def parse_block_model_csv(text: str) -> list[dict]:
    output = []
    for index, row in enumerate(_rows(text), start=1):
        output.append(
            {
                "id": row.get("block id") or row.get("id") or f"B{index:07d}",
                "x": _number(row, ("x", "easting")),
                "y": _number(row, ("y", "northing")),
                "z": _number(row, ("z", "rl", "elevation")),
                "size_x_m": _number(row, ("size x", "dx", "block size x"), required=False) or 1.0,
                "size_y_m": _number(row, ("size y", "dy", "block size y"), required=False) or 1.0,
                "size_z_m": _number(row, ("size z", "dz", "block size z"), required=False) or 1.0,
                "density_t_m3": _number(row, ("density", "density_t_m3")),
                "grade_cpht": _number(row, ("grade", "grade_cpht", "cpht"), required=False) or 0.0,
                "facies": row.get("facies") or row.get("material") or "UNKNOWN",
                "provenance": "measured",
            }
        )
    return output


def parse_surface_csv(text: str) -> list[dict]:
    return [
        {
            "x": _number(row, ("x", "easting")),
            "y": _number(row, ("y", "northing")),
            "z": _number(row, ("z", "rl", "elevation")),
            "provenance": "measured",
        }
        for row in _rows(text)
    ]


def parse_movement_monitors_csv(text: str) -> list[dict]:
    output = []
    for index, row in enumerate(_rows(text), start=1):
        output.append(
            {
                "id": row.get("monitor id") or row.get("id") or f"M{index:04d}",
                "x": _number(row, ("x", "easting")),
                "y": _number(row, ("y", "northing")),
                "z": _number(row, ("z", "rl", "elevation")),
                "dx": _number(row, ("dx", "movement x")),
                "dy": _number(row, ("dy", "movement y")),
                "dz": _number(row, ("dz", "movement z")),
                "provenance": "measured",
            }
        )
    return output


def parse_dig_limits_csv(text: str) -> list[dict]:
    points = []
    for index, row in enumerate(_rows(text), start=1):
        points.append(
            {
                "polygon_id": row.get("polygon id") or row.get("polygon") or "DIG-1",
                "sequence": int(_number(row, ("sequence", "vertex", "order"), required=False) or index),
                "x": _number(row, ("x", "easting")),
                "y": _number(row, ("y", "northing")),
                "destination": row.get("destination") or "ORE",
                "provenance": "measured",
            }
        )
    return points
