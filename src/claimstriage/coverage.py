"""Coverage checks as explainable rules, plus deductible and limit math.

Each check returns an id, pass or fail, and a sentence an adjuster can read. The result status is:
- covered: every check passed and the cause is known.
- denied: a hard check failed (not in force, peril not covered, excluded cause).
- question: nothing failed, but coverage depends on a fact only an adjuster can establish (unknown water source).
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from .catalog import (
    AUTO_COMPREHENSIVE_PERILS,
    CONTENTS_SHARE,
    EXCLUDED_CAUSES,
    HOME_COVERED_PERILS,
    JEWELRY_SUBLIMIT,
    REGIONS,
)


def _d(v) -> date | None:
    if v is None or v is pd.NaT or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, str):
        return date.fromisoformat(v[:10])
    if isinstance(v, datetime):
        return v.date()
    return v


def _check(cid: str, passed: bool, detail: str) -> dict:
    return {"id": cid, "passed": bool(passed), "detail": detail}


def check_coverage(c: dict, cause: str | None = None) -> dict:
    """Run coverage rules on one claim (policy fields are on the claim). `cause` overrides the stated cause."""
    cause = cause or c["loss_cause"]
    loss, report = _d(c["loss_date"]), _d(c["report_date"])
    inception, cancel = _d(c["policy_inception"]), _d(c.get("policy_cancel"))
    checks = []
    in_force = inception <= loss and (cancel is None or loss < cancel)
    if in_force:
        detail = f"Policy in force on {loss}: incepted {inception}" + (
            f", cancelled {cancel}." if cancel else ", active."
        )
    elif loss < inception:
        detail = f"Loss date {loss} is before policy inception {inception}."
    else:
        detail = f"Policy cancelled {cancel}, before the loss date {loss}."
    checks.append(_check("COV-IN-FORCE", in_force, detail))
    checks.append(
        _check("COV-REPORT-DATE", report >= loss, f"Reported {report}, {(report - loss).days} days after the loss.")
    )

    peril, line = c["peril"], c["line"]
    if line == "auto":
        if peril == "collision":
            ok = bool(c["cov_collision"])
            checks.append(
                _check("COV-PERIL", ok, "Collision coverage on the policy." if ok else "No collision coverage.")
            )
        else:
            ok = bool(c["cov_comprehensive"]) and peril in AUTO_COMPREHENSIVE_PERILS
            checks.append(
                _check(
                    "COV-PERIL",
                    ok,
                    f"{peril.replace('_', ' ').title()} falls under comprehensive."
                    if ok
                    else f"{peril} is not covered.",
                )
            )
    else:
        if peril == "flood":
            ok = bool(c["flood_endorsement"])
            checks.append(
                _check(
                    "COV-PERIL",
                    ok,
                    "Flood endorsement on the policy."
                    if ok
                    else "Flood is excluded from the homeowners form and there is no flood endorsement.",
                )
            )
        else:
            ok = peril in HOME_COVERED_PERILS
            checks.append(
                _check(
                    "COV-PERIL", ok, f"{peril.replace('_', ' ')} is a covered peril." if ok else f"{peril} not covered."
                )
            )

    excl = EXCLUDED_CAUSES.get(cause)
    checks.append(_check("COV-EXCLUSION", excl is None, excl or "No exclusion applies to the stated cause."))

    deductible, ded_detail = deductible_for(c)
    limit, limit_detail = limit_for(c)
    checks.append(_check("COV-DEDUCTIBLE", True, ded_detail))
    checks.append(_check("COV-LIMIT", True, limit_detail))

    hard_fail = any(not ch["passed"] for ch in checks)
    if hard_fail:
        status = "denied"
    elif cause == "unknown":
        status = "question"
        checks.append(
            _check(
                "COV-CAUSE-KNOWN", False, "Water source unknown: sudden discharge is covered, gradual leakage is not."
            )
        )
    else:
        status = "covered"
    return {
        "status": status,
        "checks": checks,
        "deductible": deductible,
        "limit": limit,
        "jewelry_sublimit": JEWELRY_SUBLIMIT if (line == "home" and peril == "theft") else None,
    }


def deductible_for(c: dict) -> tuple[float, str]:
    if c["line"] == "auto":
        if c["peril"] == "collision":
            return float(c["ded_collision"]), f"Collision deductible ${c['ded_collision']:,.0f}."
        return float(c["ded_comprehensive"]), f"Comprehensive deductible ${c['ded_comprehensive']:,.0f}."
    flat = float(c["ded_home"])
    pct = REGIONS[c["region"]].pct_wind_deductible
    if c["peril"] == "wind_hail" and pct > 0:
        pct_amt = pct * float(c["dwelling_limit"])
        if pct_amt > flat:
            return (
                pct_amt,
                f"Wind and hail deductible {pct:.0%} of the ${c['dwelling_limit']:,.0f} dwelling limit: "
                f"${pct_amt:,.0f}.",
            )
    return flat, f"All-peril deductible ${flat:,.0f}."


def limit_for(c: dict) -> tuple[float, str]:
    if c["line"] == "auto":
        v = float(c["vehicle_value"])
        return v, f"Physical damage limited to actual cash value ${v:,.0f} ({c['vehicle_year']} vehicle)."
    if c["peril"] == "theft":
        v = CONTENTS_SHARE * float(c["dwelling_limit"])
        return (
            v,
            f"Contents limit ${v:,.0f} ({CONTENTS_SHARE:.0%} of dwelling), jewelry sublimit ${JEWELRY_SUBLIMIT:,.0f}.",
        )
    v = float(c["dwelling_limit"])
    return v, f"Dwelling limit ${v:,.0f}."


def settle(amount: float, cov: dict, jewelry: float = 0.0) -> float:
    """Payable amount: apply the jewelry sublimit, then the deductible, then the limit. Zero when not covered."""
    if cov["status"] == "denied":
        return 0.0
    if cov.get("jewelry_sublimit") is not None and jewelry > cov["jewelry_sublimit"]:
        amount -= jewelry - cov["jewelry_sublimit"]
    return round(min(max(amount - cov["deductible"], 0.0), cov["limit"]), 2)


def jewelry_amount(lines: list[dict]) -> float:
    return sum(x["amount"] for x in lines if x.get("category") == "jewelry")
