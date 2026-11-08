"""Weather verification: a hail or flood claim must match an event at that place and date."""

from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd

from .catalog import WEATHER_CAUSES


def km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p = math.pi / 180
    a = (
        0.5
        - math.cos((lat2 - lat1) * p) / 2
        + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def as_events(events) -> list[dict]:
    """Event records with real dates, from a DataFrame (or a list already in that shape)."""
    rows = events.to_dict("records") if isinstance(events, pd.DataFrame) else list(events)
    out = []
    for e in rows:
        e = dict(e)
        for k in ("start", "end"):
            if isinstance(e[k], str):
                e[k] = date.fromisoformat(e[k][:10])
            elif isinstance(e[k], datetime):  # includes pandas Timestamp
                e[k] = e[k].date()
        out.append(e)
    return out


def verify(
    cause: str, loss_date: date, lat: float, lon: float, events, window_days: int = 2, radius_margin_km: float = 10.0
) -> dict:
    """Return status verified, no_event, or not_applicable, with the matching event or the nearest miss."""
    if cause not in WEATHER_CAUSES:
        return {"status": "not_applicable", "event_id": None, "detail": "Cause is not weather related."}
    if isinstance(loss_date, str):
        loss_date = date.fromisoformat(loss_date[:10])
    if isinstance(events, pd.DataFrame):
        events = as_events(events)
    best = None
    for ev in events:
        if ev["kind"] != cause:
            continue
        dist = km(lat, lon, ev["lat"], ev["lon"])
        s, e = ev["start"], ev["end"]
        days_off = 0 if s <= loss_date <= e else min(abs((loss_date - s).days), abs((loss_date - e).days))
        if days_off <= window_days and dist <= ev["radius_km"] + radius_margin_km:
            return {
                "status": "verified",
                "event_id": ev["event_id"],
                "detail": f"{ev['event_id']}: {cause} on {s} to {e}, loss {dist:.0f} km from the center "
                f"(radius {ev['radius_km']:.0f} km).",
            }
        score = days_off + dist / 10
        if best is None or score < best[0]:
            best = (score, ev, dist, days_off)
    if best is None:
        return {"status": "no_event", "event_id": None, "detail": f"No {cause} events on record."}
    _, ev, dist, days_off = best
    return {
        "status": "no_event",
        "event_id": None,
        "detail": f"No {cause} event within {window_days} days and range. Nearest: {ev['event_id']}, {dist:.0f} km "
        f"away and {days_off} days off.",
    }
