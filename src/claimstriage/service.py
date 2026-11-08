"""Triage service: loads the batch artifacts and answers claim detail, queue, and FNOL requests.

Ground truth (labels, fair settlement, review draws) stays in the batch artifacts. Nothing here returns it.
"""

from __future__ import annotations

import json
import math
import pickle
import threading
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from .model import FraudModel
from .network import ego_graph
from .pipeline import FEATURES, NETWORK_KEYS, ClaimProcessor, Paths, load_config
from .routing import route_model
from .store import Store
from .weather import km

PRIVATE_COLUMNS = {
    "truth_label",
    "truth_fraud_type",
    "truth_cause",
    "truth_ring_id",
    "fair_settlement",
    "review_draw",
    "damage_cost",
    "damage_jewelry",
}
POLICY_COLUMNS = [
    "policy_id",
    "line",
    "region",
    "policy_inception",
    "policy_cancel",
    "cov_collision",
    "cov_comprehensive",
    "ded_collision",
    "ded_comprehensive",
    "vehicle_year",
    "vehicle_value",
    "dwelling_limit",
    "ded_home",
    "flood_endorsement",
    "roof_age",
    "claimant_id",
    "claimant_name",
    "phone",
    "address_id",
    "claimant_age",
    "home_lat",
    "home_lon",
]


class NotFound(Exception):
    pass


class BadRequest(Exception):
    pass


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if v is pd.NaT:
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def band(score: float) -> str:
    return "high" if score >= 0.3 else "elevated" if score >= 0.03 else "low"


class Triage:
    def __init__(self, paths: Paths, cfg: dict | None = None, store: Store | None = None):
        self.paths = paths
        self.cfg = cfg or load_config()
        self.claims = pd.read_pickle(paths.claims)
        self.scored = pd.read_pickle(paths.scored)
        self.model = FraudModel.load(paths.model)
        self.evaluation = json.loads(paths.evaluation.read_text())
        weather = pd.read_csv(paths.weather)
        index = pickle.loads(paths.index.read_bytes())
        self.proc = ClaimProcessor(self.cfg, weather, index)
        self.weather = self.proc.weather
        self.store = store or Store(paths.db)
        self.pos = {cid: i for i, cid in enumerate(self.claims["claim_id"])}
        self.policies = self.claims.drop_duplicates("policy_id", keep="last").set_index("policy_id")[POLICY_COLUMNS[1:]]
        self.lock = threading.Lock()
        test = self.scored[self.scored["split"] == "test"]
        self.queue_frames = {q: g for q, g in test.groupby("route")}

    # ---------- reads ----------
    def public_claim(self, c: dict) -> dict:
        return {k: _clean(v) for k, v in c.items() if k not in PRIVATE_COLUMNS}

    def detail(self, claim_id: str) -> dict:
        sub = self.store.get_fnol(claim_id)
        if sub is not None:
            return sub
        if claim_id not in self.pos:
            raise NotFound(claim_id)
        i = self.pos[claim_id]
        c = self.claims.iloc[i].to_dict()
        s = self.scored.iloc[i].to_dict()
        net = {k: s[k] for k in NETWORK_KEYS}
        r = self.proc.process(c, add=False, net=net)
        score = float(s["score"])
        queue, reasons = route_model(s, score, self.cfg)
        return self._result(c, r, score, {f: s[f] for f in FEATURES}, queue, reasons, split=s["split"])

    def _result(self, c, r, score, feats, queue, reasons, split) -> dict:
        return _clean(
            {
                "claim_id": c["claim_id"],
                "split": split,
                "claim": self.public_claim(c),
                "coverage": r["coverage"],
                "estimate": r["estimate"],
                "weather": r["weather"],
                "network": r["network"],
                "red_flags": r["red_flags"],
                "fraud": {"score": round(score, 4), "band": band(score), "reasons": self.model.explain(feats)},
                "route": {"queue": queue, "reasons": reasons},
                "payable_claimed": r["payable_claimed"],
                "payable_adjusted": r["payable_adjusted"],
                "graph": ego_graph(self.claims, c),
            }
        )

    def queue_counts(self) -> dict:
        counts = {q: int(len(self.queue_frames.get(q, []))) for q in ("stp", "desk", "field", "siu")}
        for row in self.store.list_fnol():
            counts[row["route"]] = counts.get(row["route"], 0) + 1
        return counts

    def queue(self, route: str, limit: int = 50, offset: int = 0) -> dict:
        if route not in ("stp", "desk", "field", "siu"):
            raise NotFound(route)
        g = self.queue_frames.get(route, self.scored.iloc[0:0])
        g = (
            g.sort_values(["score", "report_date"], ascending=False)
            if route in ("siu", "field")
            else g.sort_values("report_date", ascending=False)
        )
        items = []
        for row in self.store.list_fnol(route) if offset == 0 else []:
            res = row["result"]
            items.append(
                {
                    "claim_id": row["claim_id"],
                    "report_date": res["claim"]["report_date"],
                    "line": res["claim"]["line"],
                    "peril": res["claim"]["peril"],
                    "region": res["claim"]["region"],
                    "claimed_total": res["estimate"]["claimed_total"],
                    "adjusted_total": res["estimate"]["adjusted_total"],
                    "score": res["fraud"]["score"],
                    "flags": len([f for f in res["red_flags"] if f["fired"]]),
                    "coverage_status": res["coverage"]["status"],
                    "submitted": True,
                }
            )
        page = g.iloc[offset : offset + limit]
        meta = self.claims.iloc[[self.pos[cid] for cid in page["claim_id"]]]
        for s, (_, m) in zip(page.to_dict("records"), meta.iterrows(), strict=True):
            items.append(
                {
                    "claim_id": s["claim_id"],
                    "report_date": _clean(s["report_date"]),
                    "line": m["line"],
                    "peril": m["peril"],
                    "region": m["region"],
                    "claimed_total": s["claimed_total"],
                    "adjusted_total": s["adjusted_total"],
                    "score": round(float(s["score"]), 4),
                    "flags": int(s["n_flags"]),
                    "coverage_status": s["coverage_status"],
                    "submitted": False,
                }
            )
        return {"route": route, "total": int(len(g)), "offset": offset, "items": items}

    # ---------- FNOL ----------
    def build_claim(self, p: dict, claim_id: str) -> dict:
        pid = p["policy_id"]
        if pid not in self.policies.index:
            raise BadRequest(f"unknown policy {pid}")
        pol = self.policies.loc[pid].to_dict()
        pol["policy_id"] = pid
        if p.get("peril") is None or p.get("loss_cause") is None:
            raise BadRequest("peril and loss_cause are required")
        loss = date.fromisoformat(str(p["loss_date"])[:10])
        report = date.fromisoformat(str(p.get("report_date") or p["loss_date"])[:10])
        lat = p.get("loss_lat") if p.get("loss_lat") is not None else pol["home_lat"]
        lon = p.get("loss_lon") if p.get("loss_lon") is not None else pol["home_lon"]
        hist = self.claims[self.claims["policy_id"] == pid]["report_date"]
        prior = int(sum(1 for d in hist if 0 <= (report - d).days <= 1095))
        lines = []
        for i, li in enumerate(p.get("estimate_lines") or [], 1):
            qty = float(li["qty"])
            lines.append(
                {
                    "line_no": i,
                    "code": li["code"],
                    "desc": li.get("desc") or li["code"],
                    "kind": li["kind"],
                    "units": int(li.get("units", 1)),
                    "qty": qty,
                    "unit_price": float(li["unit_price"]),
                    "category": "jewelry" if li["code"] == "H-JEWELRY" else "repair",
                    "amount": round(qty * float(li["unit_price"]), 2),
                }
            )
        if not lines:
            raise BadRequest("estimate_lines must not be empty")
        photos = p.get("photos") or {}
        exif = photos.get("exif_min")
        return {
            **pol,
            "claim_id": claim_id,
            "peril": p["peril"],
            "loss_cause": p["loss_cause"],
            "loss_date": loss,
            "report_date": report,
            "description": p.get("description", ""),
            "loss_lat": float(lat),
            "loss_lon": float(lon),
            "distance_km": round(km(pol["home_lat"], pol["home_lon"], float(lat), float(lon)), 1),
            "third_party": bool(p.get("third_party", False)),
            "third_party_phone": p.get("third_party_phone"),
            "passengers": int(p.get("passengers", 0)),
            "injury_claimed": bool(p.get("injury_claimed", False)),
            "police_report": bool(p.get("police_report", False)),
            "photo_count": int(photos.get("count", 0)),
            "photo_exif_min": datetime.fromisoformat(exif) if exif else None,
            "photo_exif_max": None,
            "photo_gps_km": photos.get("gps_km"),
            "shop_id": p.get("shop_id") or "SH-UNKNOWN",
            "estimate_lines": lines,
            "estimate_total": round(sum(x["amount"] for x in lines), 2),
            "prior_claims_3y": prior,
        }

    def submit(self, payload: dict, actor: str = "fnol-api") -> dict:
        with self.lock:
            claim_id = self.store.next_id()
            c = self.build_claim(payload, claim_id)
            r = self.proc.process(c, add=True)
        feats = r["features"]
        score = float(self.model.predict(pd.DataFrame([feats]))[0])
        row = {
            "coverage_status": r["coverage"]["status"],
            "claimed_total": r["estimate"]["claimed_total"],
            "variance": r["estimate"]["variance"],
            "flags_fired": r["flags_fired"],
            "ring_flag": r["network"]["ring_flag"],
        }
        queue, reasons = route_model(row, score, self.cfg)
        result = self._result(c, r, score, feats, queue, reasons, split="live")
        result["submitted_at"] = datetime.now().isoformat(timespec="seconds")
        self.store.save_fnol(claim_id, queue, score, payload, result)
        self.store.audit(
            actor,
            "fnol_submitted",
            claim_id,
            {
                "route": queue,
                "score": round(score, 4),
                "coverage": r["coverage"]["status"],
                "adjusted_total": r["estimate"]["adjusted_total"],
            },
        )
        return result


def sample_fnol(t: Triage) -> dict:
    """A ready-to-submit FNOL on a real synthetic policy, used by the UI and the README."""
    pol = t.claims[(t.claims["line"] == "auto") & t.claims["cov_collision"]].iloc[-1]
    loss = date(2026, 11, 2)
    return {
        "policy_id": pol["policy_id"],
        "peril": "collision",
        "loss_cause": "rear_end",
        "loss_date": loss.isoformat(),
        "report_date": (loss + timedelta(days=1)).isoformat(),
        "description": "Traffic slowed on Cedar Rd and the other driver rear ended me.",
        "third_party": True,
        "passengers": 0,
        "injury_claimed": False,
        "police_report": True,
        "photos": {"count": 6, "exif_min": f"{loss.isoformat()}T17:20:00", "gps_km": 0.3},
        "shop_id": pol["shop_id"],
        "estimate_lines": [
            {
                "code": "A-BUMPER-R",
                "kind": "part",
                "units": 1,
                "qty": 1,
                "unit_price": 520.0,
                "desc": "Rear bumper cover",
            },
            {
                "code": "A-BUMPER-R",
                "kind": "labor",
                "units": 1,
                "qty": 2.6,
                "unit_price": 64.0,
                "desc": "Rear bumper labor",
            },
            {
                "code": "A-REFINISH",
                "kind": "part",
                "units": 2,
                "qty": 2,
                "unit_price": 98.0,
                "desc": "Refinish materials",
            },
            {
                "code": "A-REFINISH",
                "kind": "labor",
                "units": 2,
                "qty": 4.0,
                "unit_price": 64.0,
                "desc": "Refinish labor",
            },
        ],
    }
