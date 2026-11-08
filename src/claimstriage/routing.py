"""Routing policies: straight-through payment (stp), desk adjuster, field adjuster, or special investigations (siu).

Every policy returns a queue and the reasons, so a routing decision is as explainable as a coverage check.
"""

from __future__ import annotations

from .redflags import HARD_FLAGS

QUEUES = ("stp", "desk", "field", "siu")


def route_model(c: dict, score: float, cfg: dict) -> tuple[str, list[str]]:
    r = cfg["routing"]
    if c["coverage_status"] == "denied":
        return "desk", ["Coverage denied: an adjuster confirms and sends the denial letter."]
    hard = [f for f in c["flags_fired"] if f in HARD_FLAGS]
    if score >= r["siu_min_score"]:
        return "siu", [f"Fraud score {score:.2f} at or above {r['siu_min_score']:.2f}."]
    if c["ring_flag"] and score >= r["siu_ring_min_score"]:
        return "siu", [f"Linked ring and fraud score {score:.2f} at or above {r['siu_ring_min_score']:.2f}."]
    reasons = []
    if c["claimed_total"] >= r["field_min_amount"]:
        reasons.append(f"Claimed ${c['claimed_total']:,.0f} at or above ${r['field_min_amount']:,.0f}.")
    if score >= r["field_min_score"]:
        reasons.append(f"Fraud score {score:.2f} at or above {r['field_min_score']:.2f}.")
    if c["variance"] > r["field_min_variance"]:
        reasons.append(
            f"Estimate {c['variance']:.0%} above book, over {r['field_min_variance']:.0%}: inspect the damage."
        )
    if hard:
        reasons.append("Red flags that need an inspection: " + ", ".join(hard) + ".")
    if reasons:
        return "field", reasons
    blockers = []
    if c["claimed_total"] > r["stp_max_amount"]:
        blockers.append(
            f"Claimed ${c['claimed_total']:,.0f} over the ${r['stp_max_amount']:,.0f} straight-through limit."
        )
    if c["coverage_status"] != "covered":
        blockers.append("Coverage needs an adjuster to confirm the cause.")
    if score >= r["stp_max_score"]:
        blockers.append(f"Fraud score {score:.2f} at or above {r['stp_max_score']:.2f}.")
    if c["variance"] > r["stp_max_variance"]:
        blockers.append(f"Estimate {c['variance']:.0%} above book, over {r['stp_max_variance']:.0%}.")
    if not blockers:
        return "stp", ["Low amount, coverage clear, low fraud score, estimate within tolerance."]
    return "desk", blockers


def route_rules(c: dict, cfg: dict) -> tuple[str, list[str]]:
    r = cfg["rules_only"]
    if c["coverage_status"] == "denied":
        return "desk", ["Coverage denied."]
    hard = [f for f in c["flags_fired"] if f in HARD_FLAGS]
    if hard or len(c["flags_fired"]) >= r["siu_min_flags"]:
        return "siu", ["Red flags: " + ", ".join(c["flags_fired"]) + "."]
    if c["claimed_total"] >= r["field_min_amount"]:
        return "field", ["High amount."]
    if (
        c["claimed_total"] <= r["stp_max_amount"]
        and c["coverage_status"] == "covered"
        and not c["flags_fired"]
        and c["variance"] <= r["stp_max_variance"]
    ):
        return "stp", ["Low amount, no red flags, estimate within tolerance."]
    return "desk", ["Default."]


def route_all_adjuster(c: dict, cfg: dict) -> tuple[str, list[str]]:
    if c["claimed_total"] >= cfg["all_adjuster"]["field_min_amount"]:
        return "field", ["High amount."]
    return "desk", ["Every claim goes to an adjuster."]


POLICIES = {
    "all_adjuster": lambda c, s, cfg: route_all_adjuster(c, cfg),
    "rules_only": lambda c, s, cfg: route_rules(c, cfg),
    "model": route_model,
}
