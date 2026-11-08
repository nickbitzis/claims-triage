from __future__ import annotations

from datetime import date, datetime

import pytest

from claimstriage.batch import run_all
from claimstriage.pipeline import Paths, load_config

N_CLAIMS = 8000


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def built(tmp_path_factory, cfg):
    paths = Paths(tmp_path_factory.mktemp("data"))
    report = run_all(paths, N_CLAIMS, 7, cfg, log=lambda *_: None)
    return paths, report


def make_claim(**kw) -> dict:
    """A small, clean auto collision claim with its policy fields. Override anything with keyword arguments."""
    c = dict(
        claim_id="T-1",
        policy_id="P1",
        line="auto",
        region="North",
        peril="collision",
        loss_cause="rear_end",
        loss_date=date(2026, 5, 10),
        report_date=date(2026, 5, 11),
        description="Rear ended at a light.",
        claimant_id="C1",
        claimant_name="Test Person",
        phone="555-0000-01",
        address_id="A1",
        claimant_age=40,
        home_lat=44.9,
        home_lon=-93.2,
        loss_lat=44.91,
        loss_lon=-93.21,
        distance_km=1.2,
        third_party=True,
        third_party_phone=None,
        passengers=0,
        injury_claimed=False,
        police_report=True,
        photo_count=4,
        photo_exif_min=datetime(2026, 5, 10, 18, 0),
        photo_exif_max=None,
        photo_gps_km=0.2,
        shop_id="SH-NOR-A01",
        estimate_lines=[],
        estimate_total=0.0,
        policy_inception=date(2024, 1, 1),
        policy_cancel=None,
        cov_collision=True,
        cov_comprehensive=True,
        ded_collision=500.0,
        ded_comprehensive=250.0,
        vehicle_year=2020,
        vehicle_value=18000.0,
        dwelling_limit=0.0,
        ded_home=0.0,
        flood_endorsement=False,
        roof_age=0,
        prior_claims_3y=0,
    )
    c.update(kw)
    return c


def make_home(**kw) -> dict:
    base = dict(
        line="home",
        peril="water",
        loss_cause="pipe_burst",
        cov_collision=False,
        cov_comprehensive=False,
        ded_collision=0.0,
        ded_comprehensive=0.0,
        vehicle_year=0,
        vehicle_value=0.0,
        dwelling_limit=300000.0,
        ded_home=1000.0,
        flood_endorsement=False,
        roof_age=10,
        third_party=False,
        police_report=False,
    )
    base.update(kw)
    return make_claim(**base)
