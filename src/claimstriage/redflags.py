"""Red-flag rules. Each one is a readable condition with its threshold in config/routing.yaml."""

from __future__ import annotations

from datetime import date, datetime

HARD_FLAGS = ("RF-PHOTO-BEFORE-LOSS", "RF-WEATHER-MISMATCH", "RF-RING")


def _date(v) -> date:
    if isinstance(v, str):
        return date.fromisoformat(v[:10])
    if isinstance(v, datetime):
        return v.date()
    return v


def _dt(v) -> datetime | None:
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, str):
        return datetime.fromisoformat(v)
    if hasattr(v, "to_pydatetime"):
        return None if str(v) == "NaT" else v.to_pydatetime()
    return v


def red_flags(c: dict, weather: dict, net: dict, cfg: dict) -> list[dict]:
    loss, report, inception = _date(c["loss_date"]), _date(c["report_date"]), _date(c["policy_inception"])
    out = []

    def flag(fid, fired, detail):
        out.append({"id": fid, "fired": bool(fired), "detail": detail})

    age = (loss - inception).days
    flag(
        "RF-NEW-POLICY",
        0 <= age <= cfg["new_policy_days"],
        f"Loss {age} days after policy inception (flag at {cfg['new_policy_days']} or fewer).",
    )
    lag = (report - loss).days
    flag(
        "RF-LATE-REPORT",
        lag > cfg["late_report_days"],
        f"Reported {lag} days after the loss (flag above {cfg['late_report_days']}).",
    )
    exif = _dt(c.get("photo_exif_min"))
    if exif is not None:
        before = (datetime.combine(loss, datetime.min.time()) - exif).total_seconds() / 86400
        flag(
            "RF-PHOTO-BEFORE-LOSS",
            before > cfg["photo_before_loss_days"],
            f"Earliest photo taken {exif:%Y-%m-%d}, {before:.0f} days before the loss date."
            if before > 0
            else "Photos taken after the loss.",
        )
        gps = c.get("photo_gps_km")
        flag(
            "RF-PHOTO-LOCATION",
            gps is not None and gps > cfg["photo_gps_km"],
            f"Photo GPS {gps:.0f} km from the stated loss location (flag above {cfg['photo_gps_km']})."
            if gps is not None
            else "No photo GPS.",
        )
    else:
        flag("RF-PHOTO-BEFORE-LOSS", False, "No photos with timestamps.")
        flag("RF-PHOTO-LOCATION", False, "No photos.")
    flag(
        "RF-FAR-FROM-HOME",
        c["distance_km"] > cfg["far_from_home_km"],
        f"Loss {c['distance_km']:.0f} km from the insured address (flag above {cfg['far_from_home_km']}).",
    )
    flag("RF-WEATHER-MISMATCH", weather["status"] == "no_event", weather["detail"])
    flag(
        "RF-RING",
        net["ring_flag"],
        f"Linked to {int(net['net_claimants'])} claimants through shared phones or addresses; a phone "
        f"shared by {int(net['net_phone_claimants'])}.",
    )
    flag(
        "RF-INJURY-NO-POLICE",
        c["third_party"] and c["injury_claimed"] and not c["police_report"],
        "Injury claimed with a third party and no police report." if c["injury_claimed"] else "No injury claimed.",
    )
    flag(
        "RF-PRIOR-CLAIMS",
        c["prior_claims_3y"] >= cfg["prior_claims_3y"],
        f"{c['prior_claims_3y']} prior claims on this policy in 3 years (flag at {cfg['prior_claims_3y']}).",
    )
    return out


def fired(flags: list[dict]) -> list[str]:
    return [f["id"] for f in flags if f["fired"]]
