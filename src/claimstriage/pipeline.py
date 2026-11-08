"""Claim processing: coverage, estimate review, weather, network, red flags, and model features.

`ClaimProcessor` is the single code path for both the batch job (history in report order) and a live FNOL
submitted to the API, so training features and serving features cannot drift apart.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .catalog import AUTO_PERILS, HOME_PERILS, REGIONS
from .coverage import check_coverage, jewelry_amount, settle
from .network import EntityIndex
from .pricebook import review_estimate
from .redflags import fired, red_flags
from .weather import as_events, verify

ROOT = Path(__file__).resolve().parents[2]
PERIL_CODES = {p: i for i, p in enumerate([*AUTO_PERILS, *HOME_PERILS, "hail", "flood"])}
REGION_CODES = {r: i for i, r in enumerate(REGIONS)}

FEATURES = [
    "amount_log",
    "line_auto",
    "peril_code",
    "region_code",
    "policy_age_days",
    "report_lag_days",
    "prior_claims_3y",
    "third_party",
    "passengers",
    "injury_claimed",
    "police_report",
    "photo_count",
    "photo_before_loss_days",
    "photo_gps_km",
    "distance_km",
    "est_variance",
    "est_flagged_lines",
    "est_unexpected_ops",
    "weather_applicable",
    "weather_mismatch",
    "net_claimants",
    "net_prior_claims",
    "net_phone_claimants",
    "net_address_claimants",
    "net_top_shop_share",
    "shop_prior_claims",
    "shop_linked_share",
    "shop_flagged_share",
    "vehicle_age",
    "vehicle_value",
    "roof_age",
    "claimant_age",
    "coverage_question",
]
CATEGORICAL = ["peril_code", "region_code"]
NETWORK_KEYS = [
    "net_claimants",
    "net_prior_claims",
    "net_phone_claimants",
    "net_address_claimants",
    "net_top_shop_share",
    "shop_prior_claims",
    "shop_linked_share",
    "shop_flagged_share",
    "ring_flag",
]
FEATURE_LABELS = {
    "amount_log": "Claimed amount",
    "line_auto": "Auto line",
    "peril_code": "Peril",
    "region_code": "Region",
    "policy_age_days": "Policy age (days)",
    "report_lag_days": "Days to report",
    "prior_claims_3y": "Prior claims (3y)",
    "third_party": "Third party involved",
    "passengers": "Passengers",
    "injury_claimed": "Injury claimed",
    "police_report": "Police report",
    "photo_count": "Photo count",
    "photo_before_loss_days": "Photos before loss (days)",
    "photo_gps_km": "Photo GPS distance (km)",
    "distance_km": "Loss distance from home (km)",
    "est_variance": "Estimate above book",
    "est_flagged_lines": "Flagged estimate lines",
    "est_unexpected_ops": "Unexpected operations",
    "weather_applicable": "Weather cause",
    "weather_mismatch": "No weather event",
    "net_claimants": "Linked claimants",
    "net_prior_claims": "Linked prior claims",
    "net_phone_claimants": "Claimants on phone",
    "net_address_claimants": "Claimants at address",
    "net_top_shop_share": "Network shop concentration",
    "shop_prior_claims": "Shop prior claims",
    "shop_linked_share": "Shop share of linked claims",
    "shop_flagged_share": "Shop share of flagged estimates",
    "vehicle_age": "Vehicle age",
    "vehicle_value": "Vehicle value",
    "roof_age": "Roof age",
    "claimant_age": "Claimant age",
    "coverage_question": "Coverage question",
}


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or os.environ.get("CT_CONFIG", ROOT / "config" / "routing.yaml"))
    return yaml.safe_load(path.read_text())


@dataclass
class Paths:
    data: Path

    @classmethod
    def from_env(cls, data_dir: str | Path | None = None) -> Paths:
        return cls(Path(data_dir or os.environ.get("CT_DATA_DIR", ROOT / "data")))

    @property
    def claims(self) -> Path:
        return self.data / "claims.pkl"

    @property
    def weather(self) -> Path:
        return self.data / "weather.csv"

    @property
    def processed(self) -> Path:
        return self.data / "processed.pkl"

    @property
    def model(self) -> Path:
        return self.data / "model.pkl"

    @property
    def scored(self) -> Path:
        return self.data / "scored.pkl"

    @property
    def evaluation(self) -> Path:
        return self.data / "evaluation.json"

    @property
    def index(self) -> Path:
        return self.data / "entity_index.pkl"

    @property
    def db(self) -> Path:
        return self.data / "claims.db"


def _date(v) -> date:
    if isinstance(v, str):
        return date.fromisoformat(v[:10])
    if isinstance(v, datetime):
        return v.date()
    return v


def _num(v, default=np.nan) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return float(v)


class ClaimProcessor:
    def __init__(self, cfg: dict, weather: pd.DataFrame, index: EntityIndex | None = None):
        self.cfg = cfg
        self.weather = as_events(weather)
        self.index = index or EntityIndex()

    def process(self, c: dict, add: bool = True, net: dict | None = None) -> dict:
        cfg = self.cfg
        cov = check_coverage(c)
        est = review_estimate(
            c["estimate_lines"], c["region"], c["line"], c["peril"], c["loss_cause"], cfg["estimate_tolerance"]
        )
        wx = verify(
            c["loss_cause"],
            _date(c["loss_date"]),
            c["loss_lat"],
            c["loss_lon"],
            self.weather,
            cfg["weather"]["window_days"],
            cfg["weather"]["radius_margin_km"],
        )
        net = net if net is not None else self.index.features(c)
        flags = red_flags(c, wx, net, cfg["red_flags"])
        if add:
            self.index.add(c, net, est["flagged_lines"] > 0)
        loss, report, inception = _date(c["loss_date"]), _date(c["report_date"]), _date(c["policy_inception"])
        exif = c.get("photo_exif_min")
        before = np.nan
        if exif is not None and not (isinstance(exif, float) and math.isnan(exif)) and str(exif) != "NaT":
            exif = datetime.fromisoformat(exif) if isinstance(exif, str) else exif
            before = max(0.0, (datetime.combine(loss, datetime.min.time()) - exif).total_seconds() / 86400)
        feats = {
            "amount_log": math.log1p(est["claimed_total"]),
            "line_auto": int(c["line"] == "auto"),
            "peril_code": PERIL_CODES.get(c["peril"], -1),
            "region_code": REGION_CODES.get(c["region"], -1),
            "policy_age_days": (loss - inception).days,
            "report_lag_days": (report - loss).days,
            "prior_claims_3y": int(c.get("prior_claims_3y", 0)),
            "third_party": int(bool(c["third_party"])),
            "passengers": int(c["passengers"]),
            "injury_claimed": int(bool(c["injury_claimed"])),
            "police_report": int(bool(c["police_report"])),
            "photo_count": int(c["photo_count"]),
            "photo_before_loss_days": before,
            "photo_gps_km": _num(c.get("photo_gps_km")),
            "distance_km": float(c["distance_km"]),
            "est_variance": est["variance"],
            "est_flagged_lines": est["flagged_lines"],
            "est_unexpected_ops": est["unexpected_ops"],
            "weather_applicable": int(wx["status"] != "not_applicable"),
            "weather_mismatch": int(wx["status"] == "no_event"),
            **{k: float(v) for k, v in net.items() if k != "ring_flag"},
            "vehicle_age": (loss.year - int(c["vehicle_year"])) if c["line"] == "auto" else np.nan,
            "vehicle_value": float(c["vehicle_value"]) if c["line"] == "auto" else np.nan,
            "roof_age": float(c["roof_age"]) if c["line"] == "home" else np.nan,
            "claimant_age": float(c["claimant_age"]),
            "coverage_question": int(cov["status"] == "question"),
        }
        jewelry = jewelry_amount(c["estimate_lines"])
        return {
            "claim_id": c["claim_id"],
            "coverage": cov,
            "estimate": est,
            "weather": wx,
            "network": net,
            "red_flags": flags,
            "flags_fired": fired(flags),
            "features": feats,
            "payable_claimed": settle(est["claimed_total"], cov, jewelry),
            "payable_adjusted": settle(est["adjusted_total"], cov, est["adjusted_jewelry"]),
        }


def process_history(claims: pd.DataFrame, weather: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, EntityIndex]:
    """Process every claim in report order. Returns one row per claim with features and decision inputs."""
    proc = ClaimProcessor(cfg, weather)
    rows = []
    for c in claims.to_dict("records"):
        r = proc.process(c)
        truth_cov = check_coverage(c, cause=c["truth_cause"])
        fair = 0.0 if c["truth_label"] == "fraud" else settle(c["damage_cost"], truth_cov, c["damage_jewelry"])
        # what each queue would pay if it does not deny the claim
        pay_desk = settle(r["estimate"]["adjusted_total"], truth_cov, r["estimate"]["adjusted_jewelry"])
        pay_field = settle(c["damage_cost"], truth_cov, c["damage_jewelry"])
        rows.append(
            {
                "claim_id": c["claim_id"],
                "report_date": c["report_date"],
                "coverage_status": r["coverage"]["status"],
                "claimed_total": r["estimate"]["claimed_total"],
                "adjusted_total": r["estimate"]["adjusted_total"],
                "variance": r["estimate"]["variance"],
                "flagged_lines": r["estimate"]["flagged_lines"],
                "weather_status": r["weather"]["status"],
                "ring_flag": r["network"]["ring_flag"],
                "flags_fired": r["flags_fired"],
                "n_flags": len(r["flags_fired"]),
                "pay_stp": r["payable_claimed"],
                "pay_desk": pay_desk,
                "pay_field": pay_field,
                "fair_settlement": fair,
                "is_fraud": int(c["truth_label"] == "fraud"),
                "truth_label": c["truth_label"],
                "review_draw": c["review_draw"],
                **r["features"],
            }
        )
    return pd.DataFrame(rows), proc.index


def split_of(report_date, cfg: dict) -> str:
    d = _date(report_date)
    if d <= date.fromisoformat(cfg["model"]["train_end"]):
        return "train"
    if d <= date.fromisoformat(cfg["model"]["calibration_end"]):
        return "calibration"
    return "test"
