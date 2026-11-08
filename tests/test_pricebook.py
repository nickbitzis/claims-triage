import pytest

from claimstriage.catalog import OPERATIONS, REGIONS
from claimstriage.pricebook import book_line, review_estimate

TOL = {"part": 0.12, "labor": 0.15}


def part(code, units, unit_price):
    return {
        "code": code,
        "desc": code,
        "kind": "part",
        "units": units,
        "qty": float(units),
        "unit_price": unit_price,
        "amount": round(units * unit_price, 2),
        "category": OPERATIONS[code].category,
    }


def labor(code, units, hours, rate):
    return {
        "code": code,
        "desc": code,
        "kind": "labor",
        "units": units,
        "qty": hours,
        "unit_price": rate,
        "amount": round(hours * rate, 2),
        "category": "repair",
    }


def test_book_uses_regional_parts_multiplier_and_labor_rate():
    b = book_line(part("A-HOOD", 1, 0), "Coast", "auto")
    assert b["book_unit"] == pytest.approx(520 * REGIONS["Coast"].parts_mult)
    b = book_line(labor("A-HOOD", 1, 2.0, 0), "Plains", "auto")
    assert b["book_amount"] == pytest.approx(2.0 * REGIONS["Plains"].labor_auto)


def test_fair_estimate_has_no_flags():
    lines = [part("A-HOOD", 1, 530.0), labor("A-HOOD", 1, 2.1, 63.0)]
    r = review_estimate(lines, "North", "auto", "collision", "rear_end", TOL)
    assert r["flagged_lines"] == 0 and r["adjusted_total"] == r["claimed_total"] and r["variance"] == 0


def test_padded_part_price_flagged_and_adjusted_to_book():
    r = review_estimate([part("A-HOOD", 1, 700.0)], "North", "auto", "collision", "rear_end", TOL)
    line = r["lines"][0]
    assert line["flagged"] and "above book" in line["reason"]
    assert line["adjusted_amount"] == pytest.approx(520.0)
    assert r["variance"] == pytest.approx(700 / 520 - 1, abs=1e-3)


def test_price_within_tolerance_not_flagged():
    r = review_estimate([part("A-HOOD", 1, 520 * 1.10)], "North", "auto", "collision", "rear_end", TOL)
    assert r["flagged_lines"] == 0


def test_labor_rate_above_region_flagged():
    r = review_estimate([labor("A-FENDER", 1, 2.2, 95.0)], "North", "auto", "collision", "rear_end", TOL)
    assert r["lines"][0]["flagged"] and "rate" in r["lines"][0]["reason"]


def test_excess_hours_flagged():
    r = review_estimate([labor("A-FENDER", 1, 4.0, 62.0)], "North", "auto", "collision", "rear_end", TOL)
    assert "standard hours" in r["lines"][0]["reason"]


def test_operation_not_expected_for_peril_removed():
    r = review_estimate(
        [part("A-WINDSHIELD", 1, 420.0), part("A-AIRBAG", 1, 950.0)], "North", "auto", "glass", "road_debris", TOL
    )
    flagged = [x for x in r["lines"] if x["flagged"]]
    assert len(flagged) == 1 and flagged[0]["code"] == "A-AIRBAG" and flagged[0]["adjusted_amount"] == 0
    assert r["unexpected_ops"] == 1


def test_duplicate_line_removed():
    r = review_estimate(
        [part("A-HOOD", 1, 520.0), part("A-HOOD", 1, 520.0)], "North", "auto", "collision", "rear_end", TOL
    )
    assert r["adjusted_total"] == pytest.approx(520.0) and "Duplicate" in r["lines"][1]["reason"]


def test_home_hail_uses_wind_hail_operations():
    r = review_estimate([part("H-SHINGLE", 20, 140.0)], "Plains", "home", "wind_hail", "hail", TOL)
    assert r["flagged_lines"] == 0
