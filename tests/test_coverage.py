from datetime import date

import pytest

from claimstriage.coverage import check_coverage, settle

from .conftest import make_claim, make_home


def status(c):
    return check_coverage(c)["status"]


def failed(c):
    return [k["id"] for k in check_coverage(c)["checks"] if not k["passed"]]


def test_clean_collision_is_covered():
    assert status(make_claim()) == "covered"


def test_loss_before_inception_denied():
    c = make_claim(policy_inception=date(2026, 6, 1))
    assert status(c) == "denied" and "COV-IN-FORCE" in failed(c)


def test_loss_after_cancellation_denied():
    c = make_claim(policy_cancel=date(2026, 5, 1))
    assert status(c) == "denied" and "COV-IN-FORCE" in failed(c)


def test_cancellation_after_loss_still_in_force():
    assert status(make_claim(policy_cancel=date(2026, 6, 1))) == "covered"


def test_collision_without_collision_coverage_denied():
    c = make_claim(cov_collision=False)
    assert "COV-PERIL" in failed(c)


def test_hail_on_auto_is_comprehensive():
    assert status(make_claim(peril="hail", loss_cause="hail", cov_collision=False)) == "covered"


def test_home_flood_needs_endorsement():
    assert status(make_home(peril="flood", loss_cause="flood")) == "denied"
    assert status(make_home(peril="flood", loss_cause="flood", flood_endorsement=True)) == "covered"


def test_gradual_leak_excluded():
    c = make_home(loss_cause="gradual_leak")
    assert status(c) == "denied" and "COV-EXCLUSION" in failed(c)


def test_unknown_water_source_is_a_question():
    c = make_home(loss_cause="unknown")
    assert status(c) == "question"
    assert check_coverage(c, cause="pipe_burst")["status"] == "covered"


def test_report_before_loss_fails():
    assert "COV-REPORT-DATE" in failed(make_claim(report_date=date(2026, 5, 9)))


def test_every_check_has_a_readable_detail():
    for ch in check_coverage(make_home(peril="flood", loss_cause="flood"))["checks"]:
        assert ch["detail"] and ch["id"].startswith("COV-")


# ---- deductible and limit math ----


def test_collision_deductible_applied():
    cov = check_coverage(make_claim(ded_collision=1000.0))
    assert cov["deductible"] == 1000.0 and settle(3500.0, cov) == 2500.0


def test_comprehensive_deductible_for_glass():
    cov = check_coverage(make_claim(peril="glass", loss_cause="road_debris", ded_comprehensive=250.0))
    assert settle(600.0, cov) == 350.0


def test_amount_below_deductible_pays_zero():
    assert settle(400.0, check_coverage(make_claim())) == 0.0


def test_auto_limited_to_actual_cash_value():
    cov = check_coverage(make_claim(vehicle_value=5000.0))
    assert settle(12000.0, cov) == 5000.0


def test_percent_wind_deductible_in_plains():
    cov = check_coverage(make_home(region="Plains", peril="wind_hail", loss_cause="hail", dwelling_limit=400000.0))
    assert cov["deductible"] == pytest.approx(4000.0)
    assert settle(10000.0, cov) == pytest.approx(6000.0)


def test_flat_deductible_where_no_percent_rule():
    cov = check_coverage(make_home(region="North", peril="wind_hail", loss_cause="hail", dwelling_limit=400000.0))
    assert cov["deductible"] == 1000.0


def test_jewelry_sublimit_then_deductible():
    cov = check_coverage(make_home(peril="theft", loss_cause="burglary", police_report=True))
    # 4,000 of jewelry capped at 1,500, plus 1,000 of electronics, minus the 1,000 deductible
    assert settle(5000.0, cov, jewelry=4000.0) == 1500.0


def test_denied_pays_zero():
    assert settle(9000.0, check_coverage(make_claim(cov_collision=False))) == 0.0
