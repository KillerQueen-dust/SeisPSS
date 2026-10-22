"""Event and station metadata helpers for pretraining batches."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

MISSING_UID = np.int64(-1)


def stable_uid(*parts: Any) -> np.int64:
    joined = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return np.int64(int(digest[:15], 16))


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(np.isnan(value)):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def clean_location_code(value: Any) -> str:
    location = clean_string(value)
    if location.isdigit() and len(location) == 1:
        return location.zfill(2)
    return location


def extract_event_id(row: Any, *, dataset_name: str) -> str:
    dataset_key = dataset_name.lower()
    if dataset_key == "pnw":
        event_id = clean_string(_get(row, "event_id"))
        if event_id:
            return event_id

    for column in ("source_id", "event_id"):
        event_id = clean_string(_get(row, column))
        if event_id:
            return event_id
    return ""


def build_station_id(row: Any) -> str:
    network = clean_string(_get(row, "station_network_code"))
    station = clean_string(_get(row, "station_code"))
    location = clean_location_code(_get(row, "station_location_code"))
    return f"{network}.{station}.{location}"


def event_station_uids(row: Any, *, dataset_name: str) -> tuple[np.int64, np.int64]:
    event_id = extract_event_id(row, dataset_name=dataset_name)
    station_id = build_station_id(row)
    event_uid = stable_uid("event", dataset_name, event_id) if event_id else MISSING_UID
    station_uid = stable_uid("station", station_id) if station_id != ".." else MISSING_UID
    return event_uid, station_uid


def _get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    if hasattr(row, "get"):
        return row.get(key)
    return getattr(row, key, None)
