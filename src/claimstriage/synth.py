"""Seeded synthetic claims book for Bramblewood Mutual.

About 60,000 first notice of loss (FNOL) records over two years for personal auto and home, with policies,
claimants, photo metadata, line item estimates, weather events, and ground truth:

- legitimate: the estimate matches the real damage at fair regional prices (with normal shop variation).
- inflated: a real loss with padding (prices or hours above book, duplicate or extra operations).
- fraudulent: staged accidents by rings that share phones, addresses, and repair shops; prior damage on a
  new policy; or a hail or flood claim with no matching weather event. The fair settlement is zero.

Truth columns start with `truth_` (plus `fair_settlement`) and are used only for training and evaluation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from .catalog import (
    AUTO_PERILS,
    CAUSES,
    FIRST_NAMES,
    HOME_PERILS,
    LAST_NAMES,
    OPERATIONS,
    REGIONS,
    STREETS,
)

START = date(2024, 11, 1)
MONTHS = 24
REFERENCE_CLAIMS = 60_000

TRUTH_COLUMNS = ["truth_label", "truth_fraud_type", "truth_cause", "truth_ring_id", "fair_settlement"]

TEMPLATES = {
    "rear_end": [
        "I was stopped at the light on {street} Ave when the car behind me hit my rear bumper.",
        "Traffic slowed on {street} Rd around {hour}:00 and the other driver rear ended me.",
        "Another vehicle struck the back of my car while I waited to turn onto {street} St.",
    ],
    "parking_lot": [
        "Someone backed into my car in the {street} Plaza parking lot.",
        "My car was hit while parked outside the grocery store on {street} St.",
    ],
    "intersection": [
        "The other driver ran the stop sign at {street} and 4th and hit my front end.",
        "We collided in the intersection at {street} Blvd when the other car turned left in front of me.",
    ],
    "single_vehicle": [
        "I slid on ice on {street} Rd and hit a guardrail.",
        "I swerved to avoid debris on {street} Ave and struck a pole.",
    ],
    "animal": ["A deer ran into the road on {street} Rd around {hour}:00 and I hit it."],
    "road_debris": [
        "A rock from a truck cracked my windshield on the {street} highway.",
        "Gravel chipped and cracked the glass.",
    ],
    "burglary": [
        "Someone broke in through the back door on {street} St while we were away.",
        "My car was broken into overnight on {street} Ave and parts were taken.",
    ],
    "vandalism": ["My car was keyed and a window was smashed overnight on {street} St."],
    "pipe_burst": ["A pipe burst in the upstairs bathroom and water came through the ceiling."],
    "appliance": ["The washing machine supply line failed and flooded the laundry room and hallway."],
    "gradual_leak": ["We found water damage and mold under the kitchen sink from a slow leak."],
    "unknown": ["We came home to water on the basement floor. Not sure where it came from."],
    "kitchen": ["A grease fire started on the stove and spread to the cabinets."],
    "electrical": ["An outlet in the living room sparked and caused a small fire."],
    "wind": ["High winds tore shingles off the roof and damaged the gutters."],
    "hail": [
        "Hail damaged the {obj} during the storm on {day}.",
        "Large hail hit the {obj} on {day}. Lots of dents and cracks.",
        "The storm on {day} dropped hail that damaged the {obj}.",
    ],
    "flood": [
        "Flood water from the storm on {day} got into the {obj}.",
        "The street flooded on {day} and water reached the {obj}.",
    ],
}
STAGED_TEMPLATES = [
    "The other driver stopped suddenly in front of me on {street} Ave and I could not avoid hitting them.",
    "A car pulled out in front of me on {street} Rd and braked hard. My passengers were hurt.",
    "We were hit at the {street} and 2nd intersection. The other driver was very cooperative and admitted fault.",
]


@dataclass
class Dataset:
    claims: pd.DataFrame
    weather: pd.DataFrame
    shops: pd.DataFrame


class FastRng:
    """The subset of numpy's Generator API the generator uses, on Python's random (much faster per scalar draw)."""

    def __init__(self, seed: int):
        self.r = random.Random(seed)

    def random(self) -> float:
        return self.r.random()

    def integers(self, lo: int, hi: int | None = None) -> int:
        if hi is None:
            lo, hi = 0, lo
        return self.r.randrange(int(lo), int(hi))

    def uniform(self, a: float, b: float) -> float:
        return self.r.uniform(a, b)

    def normal(self, mu: float = 0.0, sd: float = 1.0) -> float:
        return self.r.gauss(mu, sd)

    def exponential(self, scale: float) -> float:
        return self.r.expovariate(1.0 / scale)

    def lognormal(self, mu: float, sigma: float) -> float:
        return self.r.lognormvariate(mu, sigma)

    def poisson(self, lam: float) -> int:
        limit, k, p = math.exp(-lam), 0, self.r.random()
        while p > limit:
            k += 1
            p *= self.r.random()
        return k

    def choice(self, a, size=None, replace=True, p=None):
        pool = range(a) if isinstance(a, int) else list(a)
        if size is not None:
            return self.r.sample(pool, size) if not replace else self.r.choices(pool, k=size)
        if p is not None:
            return self.r.choices(pool, weights=list(p))[0]
        return pool[self.r.randrange(len(pool))]


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _km(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180
    a = (
        0.5
        - math.cos((lat2 - lat1) * p) / 2
        + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _offset(lat, lon, km, bearing):
    return lat + km / 111.0 * math.cos(bearing), lon + km / (111.0 * math.cos(lat * math.pi / 180)) * math.sin(bearing)


def make_weather(rng: FastRng) -> pd.DataFrame:
    """Two catastrophe events (a hail storm and a flood) plus minor local events."""
    rows = [
        dict(
            event_id="WX-HAIL-2505",
            kind="hail",
            region="Plains",
            start=date(2025, 5, 14),
            end=date(2025, 5, 15),
            lat=39.1,
            lon=-96.6,
            radius_km=80.0,
            severity="catastrophe",
        ),
        dict(
            event_id="WX-FLOOD-2509",
            kind="flood",
            region="Gulf",
            start=date(2025, 9, 8),
            end=date(2025, 9, 10),
            lat=29.8,
            lon=-95.4,
            radius_km=60.0,
            severity="catastrophe",
        ),
    ]
    minors = [
        ("hail", "Mountain"),
        ("hail", "North"),
        ("hail", "Plains"),
        ("hail", "Lakes"),
        ("hail", "Mountain"),
        ("flood", "Lakes"),
        ("flood", "Gulf"),
        ("hail", "Plains"),
        ("flood", "North"),
        ("hail", "Lakes"),
    ]
    for i, (kind, region) in enumerate(minors):
        r = REGIONS[region]
        year = 2025 if i % 2 == 0 else 2026
        d = date(year, int(rng.integers(4, 10)), int(rng.integers(1, 28)))
        lat, lon = _offset(r.lat, r.lon, float(rng.uniform(10, 90)), float(rng.uniform(0, 2 * math.pi)))
        rows.append(
            dict(
                event_id=f"WX-{kind.upper()}-{d:%y%m%d}-{region[:3].upper()}",
                kind=kind,
                region=region,
                start=d,
                end=d + timedelta(days=int(rng.integers(0, 2))),
                lat=round(lat, 4),
                lon=round(lon, 4),
                radius_km=float(rng.uniform(25, 45)),
                severity="local",
            )
        )
    return pd.DataFrame(rows)


class _Gen:
    def __init__(self, n_claims: int, seed: int):
        self.rng = FastRng(seed)
        self.scale = n_claims / REFERENCE_CLAIMS
        self.n_claims = n_claims
        self.end = START.replace(year=START.year + MONTHS // 12) - timedelta(days=1)
        self.claims: list[dict] = []
        self.claimants: dict[str, dict] = {}
        self.by_region_claimants: dict[str, list[str]] = {r: [] for r in REGIONS}
        self.policies: dict[str, dict] = {}
        self.region_policies: dict[tuple[str, str], list[str]] = {}
        self.n_addr = 0
        self.addr_count: dict[str, int] = {}
        self.n_phone = 0
        self.weather = make_weather(self.rng)
        self._make_shops()

    # ---------- entities ----------
    def _make_shops(self):
        rows = []
        for region in REGIONS:
            for line, n in (("auto", 28), ("home", 18)):
                for i in range(n):
                    rows.append(
                        dict(
                            shop_id=f"SH-{region[:3].upper()}-{line[0].upper()}{i:02d}",
                            region=region,
                            line=line,
                            complicit=False,
                            weight=1.0 / (i + 1) ** 0.8,
                        )
                    )
        # one complicit body shop per region and one complicit home contractor in three regions
        for r in rows:
            if r["shop_id"].endswith("A05") or (
                r["shop_id"].endswith("H03") and r["region"] in ("Plains", "Gulf", "Coast")
            ):
                r["complicit"] = True
        self.shops = pd.DataFrame(rows)
        self.shop_lists = {}
        for (region, line), g in self.shops.groupby(["region", "line"]):
            w = g["weight"].to_numpy()
            self.shop_lists[(region, line)] = (
                g["shop_id"].tolist(),
                list(np.cumsum(w / w.sum())),
                g.loc[g["complicit"], "shop_id"].tolist(),
            )

    def pick_shop(self, region, line, complicit=False):
        ids, cum, bad = self.shop_lists[(region, line)]
        if complicit and bad:
            return bad[0]
        return self.rng.r.choices(ids, cum_weights=cum)[0]

    def new_phone(self):
        self.n_phone += 1
        return f"555-{(self.n_phone * 7919) % 9000 + 1000:04d}-{self.n_phone % 97:02d}"

    def new_claimant(self, region, lat=None, lon=None, household=True):
        rng = self.rng
        cid = f"C{len(self.claimants) + 1:06d}"
        pool = self.by_region_claimants[region]
        mate = (
            self.claimants[pool[int(rng.integers(len(pool)))]] if (household and pool and rng.random() < 0.14) else None
        )
        if mate is not None and self.addr_count.get(mate["address_id"], 0) < 3:
            addr, hlat, hlon = mate["address_id"], mate["home_lat"], mate["home_lon"]
            phone = mate["phone"] if rng.random() < 0.05 else self.new_phone()
            last = mate["last_name"]
        else:
            self.n_addr += 1
            addr = f"A{self.n_addr:06d}"
            if lat is None:
                r = REGIONS[region]
                lat, lon = _offset(r.lat, r.lon, float(abs(rng.normal(0, 60))), float(rng.uniform(0, 2 * math.pi)))
            hlat, hlon = lat, lon
            phone = self.new_phone()
            last = LAST_NAMES[int(rng.integers(len(LAST_NAMES)))]
        self.addr_count[addr] = self.addr_count.get(addr, 0) + 1
        c = dict(
            claimant_id=cid,
            first_name=FIRST_NAMES[int(rng.integers(len(FIRST_NAMES)))],
            last_name=last,
            phone=phone,
            address_id=addr,
            home_lat=round(hlat, 4),
            home_lon=round(hlon, 4),
            region=region,
            claimant_age=int(rng.integers(19, 84)),
        )
        self.claimants[cid] = c
        pool.append(cid)
        return c

    def new_policy(self, line, region, loss_date, age_days=None, claimant=None):
        rng = self.rng
        claimant = claimant or self.new_claimant(region)
        if age_days is None:
            age_days = int(min(4000, rng.exponential(900))) + 1
        pid = f"P{len(self.policies) + 1:06d}"
        p = dict(
            policy_id=pid,
            line=line,
            region=region,
            claimant_id=claimant["claimant_id"],
            policy_inception=loss_date - timedelta(days=age_days),
            policy_cancel=None,
        )
        if line == "auto":
            vyear = int(rng.integers(2008, 2027))
            p.update(
                cov_collision=bool(rng.random() < 0.88),
                cov_comprehensive=True,
                ded_collision=float(rng.choice([500, 1000])),
                ded_comprehensive=float(rng.choice([250, 500, 1000])),
                vehicle_year=vyear,
                vehicle_value=float(max(2500, round(34000 * 0.88 ** (2026 - vyear) * rng.uniform(0.8, 1.2), -2))),
                dwelling_limit=0.0,
                ded_home=0.0,
                flood_endorsement=False,
                roof_age=0,
            )
        else:
            gulf = region == "Gulf"
            p.update(
                cov_collision=False,
                cov_comprehensive=False,
                ded_collision=0.0,
                ded_comprehensive=0.0,
                vehicle_year=0,
                vehicle_value=0.0,
                dwelling_limit=float(round(rng.uniform(180, 650)) * 1000),
                ded_home=float(rng.choice([1000, 2500])),
                flood_endorsement=bool(rng.random() < (0.3 if gulf else 0.1)),
                roof_age=int(rng.integers(1, 26)),
            )
        self.policies[pid] = p
        self.region_policies.setdefault((line, region), []).append(pid)
        return p

    def policy_for(self, line, region, loss_date, lat=None, lon=None):
        pool = self.region_policies.get((line, region), [])
        if pool and self.rng.random() < 0.15:
            p = self.policies[pool[int(self.rng.integers(len(pool)))]]
            if p["policy_inception"] < loss_date and (p["policy_cancel"] is None or p["policy_cancel"] > loss_date):
                return p
        c = self.new_claimant(region, lat, lon)
        p = self.new_policy(line, region, loss_date, claimant=c)
        if self.rng.random() < 0.025:  # lapsed for non-payment before this loss
            gap = int(self.rng.integers(5, 120))
            p["policy_inception"] = loss_date - timedelta(days=gap + int(self.rng.integers(30, 700)))
            p["policy_cancel"] = loss_date - timedelta(days=gap)
        return p

    # ---------- estimates ----------
    def estimate(self, line, peril, region, n_ops=None, big=False):
        rng = self.rng
        reg = REGIONS[region]
        ops = [o for o in OPERATIONS.values() if o.line == line and peril in o.perils]
        if peril == "glass":
            n = 1
        else:
            n = n_ops or int(rng.integers(2, min(6, len(ops)) + 1))
            if big:
                n = min(len(ops), n + 2)
        chosen = [ops[i] for i in rng.choice(len(ops), size=min(n, len(ops)), replace=False)]
        lines = []
        rate = (reg.labor_auto if line == "auto" else reg.labor_home) * float(_clip(rng.normal(1, 0.03), 0.93, 1.07))
        for op in chosen:
            units = int(rng.integers(op.qty[0], op.qty[1] + 1))
            if op.part_price > 0:
                unit = op.part_price * reg.parts_mult * float(_clip(rng.normal(1, 0.035), 0.92, 1.08))
                lines.append(
                    dict(
                        code=op.code,
                        desc=op.desc,
                        kind="part",
                        units=units,
                        qty=float(units),
                        unit_price=round(unit, 2),
                        category=op.category,
                    )
                )
            if op.std_hours > 0:
                hours = op.std_hours * units * float(_clip(rng.normal(1, 0.05), 0.9, 1.1))
                lines.append(
                    dict(
                        code=op.code,
                        desc=op.desc + ", labor",
                        kind="labor",
                        units=units,
                        qty=round(hours, 1),
                        unit_price=round(rate, 2),
                        category="repair",
                    )
                )
        for ln in lines:
            ln["amount"] = round(ln["qty"] * ln["unit_price"], 2)
        return lines

    def pad(self, lines, line, peril):
        """Inflate an estimate. Returns the padded copy; the originals stay the true damage."""
        rng = self.rng
        out = [dict(x) for x in lines]
        k = int(rng.integers(1, min(3, len(out)) + 1))
        for i in rng.choice(len(out), size=k, replace=False):
            ln = out[int(i)]
            f = float(rng.uniform(1.06, 1.11)) if rng.random() < 0.3 else float(rng.uniform(1.25, 1.9))
            if ln["kind"] == "labor" and rng.random() < 0.5:
                ln["qty"] = round(ln["qty"] * f, 1)
            else:
                ln["unit_price"] = round(ln["unit_price"] * f, 2)
            ln["amount"] = round(ln["qty"] * ln["unit_price"], 2)
        if rng.random() < 0.45:
            present = {x["code"] for x in out}
            same = [o for o in OPERATIONS.values() if o.line == line and peril in o.perils and o.code not in present]
            other = [o for o in OPERATIONS.values() if o.line == line and peril not in o.perils]
            pool = same if (same and rng.random() < 0.6) else other
            op = pool[int(rng.integers(len(pool)))]
            reg_lines = self.estimate(line, op.perils[0], self.cur_region, n_ops=1)
            extra = [x for x in reg_lines if x["code"] == op.code] or [
                dict(
                    code=op.code,
                    desc=op.desc,
                    kind="part",
                    units=1,
                    qty=1.0,
                    unit_price=op.part_price,
                    category=op.category,
                    amount=op.part_price,
                )
            ]
            out.extend(extra)
        return out

    # ---------- claims ----------
    def describe(self, cause, staged=False, day=None, obj="roof"):
        rng = self.rng
        pool = STAGED_TEMPLATES if staged else TEMPLATES[cause]
        t = pool[int(rng.integers(len(pool)))]
        return t.format(
            street=STREETS[int(rng.integers(len(STREETS)))],
            hour=int(rng.integers(6, 22)),
            day=(day or date.today()).strftime("%B %-d"),
            obj=obj,
        )

    def add_claim(
        self,
        *,
        line,
        peril,
        cause,
        region,
        loss_date,
        policy,
        label="legit",
        fraud_type=None,
        ring_id=None,
        third_party_claimant=None,
        loss_lat=None,
        loss_lon=None,
        staged=False,
        wrong_date=False,
    ):
        rng = self.rng
        self.cur_region = region
        holder = self.claimants[policy["claimant_id"]]
        hl, ho = holder["home_lat"], holder["home_lon"]
        fraud = label == "fraud"
        # loss location
        if loss_lat is None:
            if line == "home":
                loss_lat, loss_lon = hl, ho
            else:
                far = rng.random() < (0.35 if staged else 0.2 if fraud else 0.06)
                km = float(rng.uniform(150, 700)) if far else float(rng.exponential(8))
                loss_lat, loss_lon = _offset(hl, ho, km, float(rng.uniform(0, 2 * math.pi)))
        distance = _km(hl, ho, loss_lat, loss_lon)
        # reporting
        if fraud and fraud_type == "prior_damage":
            lag = int(rng.lognormal(1.6, 1.0)) + (int(rng.integers(30, 120)) if rng.random() < 0.3 else 0)
        elif staged:
            lag = int(rng.lognormal(0.5, 0.8))
        else:
            lag = int(rng.lognormal(0.7, 1.0)) + (int(rng.integers(30, 90)) if rng.random() < 0.04 else 0)
        report_date = loss_date + timedelta(days=lag)
        stated_loss = (
            loss_date + timedelta(days=int(rng.choice([-1, 1]) * rng.integers(3, 9))) if wrong_date else loss_date
        )
        # estimate
        est_peril = "wind_hail" if (line == "home" and cause == "hail") else peril
        true_lines = self.estimate(line, est_peril, region, big=(fraud and fraud_type == "prior_damage"))
        complicit_shop = staged or (fraud and rng.random() < 0.3)
        shop = self.pick_shop(region, line, complicit=complicit_shop)
        shop_bad = shop in self._bad_shops
        if label == "legit" and rng.random() < (0.35 if shop_bad else 0.08 * self.inflate_boost):
            label = "inflated"
        padded = label == "inflated" or (fraud and rng.random() < 0.5)
        lines = self.pad(true_lines, line, est_peril) if padded else [dict(x) for x in true_lines]
        for i, ln in enumerate(lines, 1):
            ln["line_no"] = i
        damage_cost = round(sum(x["amount"] for x in true_lines), 2)
        # photos
        n_photos = 0 if rng.random() < 0.04 else int(rng.poisson(5)) + 1
        loss_dt = datetime.combine(stated_loss, datetime.min.time()) + timedelta(hours=float(rng.uniform(7, 21)))
        exif_min = exif_max = None
        photo_km = None
        if n_photos:
            taken = loss_dt + timedelta(hours=float(rng.uniform(0.2, 24 * max(lag, 1))))
            if fraud and fraud_type == "prior_damage" and rng.random() < 0.55:
                taken = loss_dt - timedelta(days=float(rng.uniform(15, 300)))
            elif fraud and rng.random() < 0.15:
                taken = loss_dt - timedelta(days=float(rng.uniform(5, 60)))
            elif rng.random() < 0.015:  # camera clock set wrong
                taken = loss_dt - timedelta(days=float(rng.uniform(2, 400)))
            exif_min = taken
            exif_max = taken + timedelta(minutes=float(rng.uniform(1, 90)))
            far_photo = rng.random() < (0.3 if (staged or fraud) else 0.02)
            photo_km = float(rng.uniform(40, 200)) if far_photo else float(rng.exponential(0.4))
        # parties
        if line == "auto" and peril == "collision":
            tp_prob = {"rear_end": 0.9, "intersection": 0.9, "parking_lot": 0.4}.get(cause, 0.0)
            third_party = staged or rng.random() < tp_prob
            police_p = 0.25 if staged else 0.8 if cause == "intersection" else 0.5
            passengers = int(rng.integers(1, 5)) if staged else int(rng.poisson(0.4))
            injury = bool(third_party and rng.random() < (0.7 if staged else 0.12))
        else:
            third_party, passengers, injury = False, 0, False
            police_p = 0.85 if peril == "theft" else 0.5 if peril == "vandalism" else 0.0
        police = bool(rng.random() < police_p)
        tp_phone = (
            self.claimants[third_party_claimant]["phone"]
            if third_party_claimant
            else (self.new_phone() if third_party else None)
        )
        obj = "car" if line == "auto" else ("roof" if cause == "hail" else "basement")
        desc = self.describe(cause, staged=staged, day=stated_loss, obj=obj)
        if injury and "hurt" not in desc:
            desc += " I have neck and back pain."
        true_cause = cause
        if cause == "unknown":
            true_cause = "pipe_burst" if rng.random() < 0.6 else "gradual_leak"
        self.claims.append(
            dict(
                policy_id=policy["policy_id"],
                line=line,
                region=region,
                peril=peril,
                loss_cause=cause,
                loss_date=stated_loss,
                report_date=report_date,
                description=desc,
                claimant_id=holder["claimant_id"],
                claimant_name=f"{holder['first_name']} {holder['last_name']}",
                phone=holder["phone"],
                address_id=holder["address_id"],
                claimant_age=holder["claimant_age"],
                home_lat=hl,
                home_lon=ho,
                loss_lat=round(loss_lat, 4),
                loss_lon=round(loss_lon, 4),
                distance_km=round(distance, 1),
                third_party=bool(third_party),
                third_party_phone=tp_phone,
                passengers=passengers,
                injury_claimed=injury,
                police_report=police,
                photo_count=n_photos,
                photo_exif_min=exif_min,
                photo_exif_max=exif_max,
                photo_gps_km=None if photo_km is None else round(photo_km, 2),
                shop_id=shop,
                estimate_lines=lines,
                estimate_total=round(sum(x["amount"] for x in lines), 2),
                damage_cost=damage_cost,
                damage_jewelry=round(sum(x["amount"] for x in true_lines if x["category"] == "jewelry"), 2),
                **{
                    k: policy[k]
                    for k in (
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
                    )
                },
                truth_label=label,
                truth_fraud_type=fraud_type,
                truth_cause=true_cause,
                truth_ring_id=ring_id,
                review_draw=float(rng.random()),
            )
        )

    def rand_date(self):
        d = START + timedelta(days=int(self.rng.integers(0, (self.end - START).days)))
        return d

    def run(self):
        rng = self.rng
        self._bad_shops = set(self.shops.loc[self.shops["complicit"], "shop_id"])
        self.inflate_boost = 1.0
        n_event = self.events()
        n_rings = self.rings()
        n_opp = self.opportunistic()
        n_base = max(0, self.n_claims - n_event - n_rings - n_opp)
        regions = list(REGIONS)
        rw = np.array([REGIONS[r].weight for r in regions])
        rw = rw / rw.sum()
        for _ in range(n_base):
            region = regions[int(rng.choice(len(regions), p=rw))]
            line = "auto" if rng.random() < 0.62 else "home"
            perils = AUTO_PERILS if line == "auto" else HOME_PERILS
            peril = str(rng.choice(list(perils), p=np.array(list(perils.values()))))
            d = self.rand_date()
            cause = CAUSES[peril][int(rng.integers(len(CAUSES[peril])))]
            if peril == "water":
                cause = str(rng.choice(CAUSES["water"], p=[0.5, 0.25, 0.13, 0.12]))
            if rng.random() < 0.012 and peril in ("collision", "vandalism", "wind_hail", "water", "theft"):
                c = self.new_claimant(region)
                age = int(rng.integers(3, 45)) if rng.random() < 0.75 else None
                policy = self.new_policy(line, region, d, age_days=age, claimant=c)
                self.add_claim(
                    line=line,
                    peril=peril,
                    cause=cause,
                    region=region,
                    loss_date=d,
                    policy=policy,
                    label="fraud",
                    fraud_type="prior_damage",
                )
                continue
            policy = self.policy_for(line, region, d)
            self.add_claim(line=line, peril=peril, cause=cause, region=region, loss_date=d, policy=policy)

    def events(self):
        rng = self.rng
        total = 0
        self.inflate_boost = 1.5  # storm chasers
        for ev in self.weather.itertuples():
            n = int(
                (
                    2400
                    if ev.event_id == "WX-HAIL-2505"
                    else 900
                    if ev.event_id == "WX-FLOOD-2509"
                    else rng.integers(40, 120)
                )
                * self.scale
            )
            for _ in range(n):
                lat, lon = _offset(
                    ev.lat,
                    ev.lon,
                    float(ev.radius_km * math.sqrt(rng.random()) * 0.95),
                    float(rng.uniform(0, 2 * math.pi)),
                )
                d = ev.start + timedelta(days=int(rng.integers(0, (ev.end - ev.start).days + 1)))
                line = "home" if rng.random() < (0.55 if ev.kind == "hail" else 0.6) else "auto"
                peril = ("wind_hail" if line == "home" else "hail") if ev.kind == "hail" else "flood"
                policy = self.policy_for(line, ev.region, d, lat, lon)
                if rng.random() < 0.01:
                    c = self.new_claimant(ev.region, lat, lon)
                    policy = self.new_policy(line, ev.region, d, age_days=int(rng.integers(2, 30)), claimant=c)
                    self.add_claim(
                        line=line,
                        peril=peril,
                        cause=ev.kind,
                        region=ev.region,
                        loss_date=d,
                        policy=policy,
                        label="fraud",
                        fraud_type="prior_damage",
                        loss_lat=lat if line == "auto" else None,
                        loss_lon=lon if line == "auto" else None,
                    )
                else:
                    self.add_claim(
                        line=line,
                        peril=peril,
                        cause=ev.kind,
                        region=ev.region,
                        loss_date=d,
                        policy=policy,
                        loss_lat=lat if line == "auto" else None,
                        loss_lon=lon if line == "auto" else None,
                        wrong_date=rng.random() < 0.03,
                    )
                total += 1
        self.inflate_boost = 1.0
        return total

    def rings(self):
        rng = self.rng
        n_rings = max(2, int(round(36 * self.scale)))
        total = 0
        regions = list(REGIONS)
        for k in range(n_rings):
            region = regions[k % len(regions)]
            ring_id = f"R{k + 1:03d}"
            phones = [self.new_phone() for _ in range(int(rng.integers(1, 3)))]
            base = self.new_claimant(region, household=False)
            members = [base]
            for _ in range(int(rng.integers(3, 7))):
                m = self.new_claimant(region, household=False)
                if rng.random() < 0.6:
                    m["phone"] = phones[int(rng.integers(len(phones)))]
                if rng.random() < 0.45:
                    m.update(address_id=base["address_id"], home_lat=base["home_lat"], home_lon=base["home_lon"])
                members.append(m)
            start = self.rand_date()
            window = int(rng.integers(90, 240))
            pols = {}
            for _ in range(int(rng.integers(6, 15))):
                d = min(self.end - timedelta(days=1), start + timedelta(days=int(rng.integers(0, window))))
                m, other = rng.choice(len(members), size=2, replace=False)
                m = members[int(m)]
                if m["claimant_id"] not in pols:
                    pols[m["claimant_id"]] = self.new_policy(
                        "auto", region, d, age_days=int(rng.integers(40, 400)), claimant=m
                    )
                    pols[m["claimant_id"]]["cov_collision"] = True
                cause = "rear_end" if rng.random() < 0.6 else "intersection"
                self.add_claim(
                    line="auto",
                    peril="collision",
                    cause=cause,
                    region=region,
                    loss_date=d,
                    policy=pols[m["claimant_id"]],
                    label="fraud",
                    fraud_type="staged_ring",
                    ring_id=ring_id,
                    third_party_claimant=members[int(other)]["claimant_id"],
                    staged=True,
                )
                total += 1
        return total

    def opportunistic(self):
        """Hail or flood claims with no matching event: wrong place, or a month or more off the storm date."""
        rng = self.rng
        n = int(380 * self.scale)
        regions = list(REGIONS)
        for _ in range(n):
            kind = "hail" if rng.random() < 0.7 else "flood"
            if rng.random() < 0.4:
                ev = self.weather.iloc[int(rng.integers(len(self.weather)))]
                region = ev["region"]
                d = ev["start"] + timedelta(days=int(rng.choice([-1, 1]) * rng.integers(20, 70)))
                d = min(max(d, START), self.end - timedelta(days=1))
            else:
                region = regions[int(rng.integers(len(regions)))]
                d = self.rand_date()
            line = "home" if rng.random() < 0.5 else "auto"
            peril = ("wind_hail" if line == "home" else "hail") if kind == "hail" else "flood"
            policy = self.policy_for(line, region, d)
            self.add_claim(
                line=line,
                peril=peril,
                cause=kind,
                region=region,
                loss_date=d,
                policy=policy,
                label="fraud",
                fraud_type="weather_opportunistic",
            )
        return n


def generate(n_claims: int = REFERENCE_CLAIMS, seed: int = 42) -> Dataset:
    g = _Gen(n_claims, seed)
    g.run()
    df = pd.DataFrame(g.claims)
    # Snapshot at the end of the window: losses not yet reported (incurred but not reported) are not in the book.
    df = df[df["report_date"] <= g.end]
    df = df.sort_values(["report_date", "loss_date", "policy_id"], kind="stable").reset_index(drop=True)
    df.insert(0, "claim_id", [f"BWM-{i + 1:06d}" for i in range(len(df))])
    df["report_month"] = pd.to_datetime(df["report_date"]).dt.strftime("%Y-%m")
    # prior claims on the same policy in the 3 years before this report (history only)
    prior = np.zeros(len(df), dtype=int)
    seen: dict[str, list[date]] = {}
    for i, (pid, rd) in enumerate(zip(df["policy_id"], df["report_date"], strict=True)):
        hist = seen.setdefault(pid, [])
        prior[i] = sum(1 for d in hist if (rd - d).days <= 1095)
        hist.append(rd)
    df["prior_claims_3y"] = prior
    df["fair_settlement"] = 0.0  # filled by the pipeline from coverage rules on the true cause and damage
    return Dataset(claims=df, weather=g.weather, shops=g.shops)
