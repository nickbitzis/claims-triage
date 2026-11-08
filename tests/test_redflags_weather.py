from datetime import date, datetime

import pandas as pd

from claimstriage.redflags import fired, red_flags
from claimstriage.weather import verify

from .conftest import make_claim

NET = {"ring_flag": False, "net_claimants": 1, "net_phone_claimants": 1}
WX_NA = {"status": "not_applicable", "detail": ""}

EVENTS = pd.DataFrame(
    [
        {
            "event_id": "WX-H",
            "kind": "hail",
            "region": "Plains",
            "start": date(2025, 5, 14),
            "end": date(2025, 5, 15),
            "lat": 39.1,
            "lon": -96.6,
            "radius_km": 80.0,
            "severity": "catastrophe",
        },
        {
            "event_id": "WX-F",
            "kind": "flood",
            "region": "Gulf",
            "start": date(2025, 9, 8),
            "end": date(2025, 9, 10),
            "lat": 29.8,
            "lon": -95.4,
            "radius_km": 60.0,
            "severity": "catastrophe",
        },
    ]
)


def flags(c, cfg, wx=WX_NA, net=NET):
    return fired(red_flags(c, wx, net, cfg["red_flags"]))


def test_clean_claim_has_no_flags(cfg):
    assert flags(make_claim(), cfg) == []


def test_new_policy(cfg):
    assert "RF-NEW-POLICY" in flags(make_claim(policy_inception=date(2026, 4, 25)), cfg)


def test_late_report(cfg):
    assert "RF-LATE-REPORT" in flags(make_claim(report_date=date(2026, 7, 1)), cfg)


def test_photos_before_loss(cfg):
    assert "RF-PHOTO-BEFORE-LOSS" in flags(make_claim(photo_exif_min=datetime(2026, 3, 1, 12)), cfg)


def test_photos_same_evening_are_fine(cfg):
    assert "RF-PHOTO-BEFORE-LOSS" not in flags(make_claim(photo_exif_min=datetime(2026, 5, 10, 22)), cfg)


def test_far_from_home(cfg):
    assert "RF-FAR-FROM-HOME" in flags(make_claim(distance_km=420.0), cfg)


def test_injury_without_police(cfg):
    assert "RF-INJURY-NO-POLICE" in flags(make_claim(injury_claimed=True, police_report=False), cfg)


def test_ring_and_weather_flags(cfg):
    out = flags(
        make_claim(),
        cfg,
        wx={"status": "no_event", "detail": "x"},
        net={"ring_flag": True, "net_claimants": 5, "net_phone_claimants": 3},
    )
    assert {"RF-RING", "RF-WEATHER-MISMATCH"} <= set(out)


def test_weather_verified_inside_event():
    r = verify("hail", date(2025, 5, 15), 39.3, -96.4, EVENTS)
    assert r["status"] == "verified" and r["event_id"] == "WX-H"


def test_weather_window_allows_two_days():
    assert verify("hail", date(2025, 5, 17), 39.1, -96.6, EVENTS)["status"] == "verified"
    assert verify("hail", date(2025, 5, 20), 39.1, -96.6, EVENTS)["status"] == "no_event"


def test_weather_wrong_place():
    r = verify("hail", date(2025, 5, 14), 44.9, -93.2, EVENTS)
    assert r["status"] == "no_event" and "WX-H" in r["detail"]


def test_weather_kind_must_match():
    assert verify("flood", date(2025, 5, 14), 39.1, -96.6, EVENTS)["status"] == "no_event"


def test_weather_not_applicable_for_collision():
    assert verify("rear_end", date(2025, 5, 14), 39.1, -96.6, EVENTS)["status"] == "not_applicable"
