import copy

import numpy as np
import pandas as pd
import pytest

from claimstriage.evaluate import account, assign, evaluate_policies, gate
from claimstriage.model import FraudModel, splits
from claimstriage.pipeline import process_history
from claimstriage.routing import route_all_adjuster, route_model, route_rules
from claimstriage.synth import generate

ROW = {"coverage_status": "covered", "claimed_total": 2500.0, "variance": 0.02, "flags_fired": [], "ring_flag": False}


def row(**kw):
    r = dict(ROW)
    r.update(kw)
    return r


# ---- routing thresholds ----


def test_clean_low_claim_goes_straight_through(cfg):
    assert route_model(row(), 0.001, cfg)[0] == "stp"


def test_stp_amount_limit(cfg):
    lim = cfg["routing"]["stp_max_amount"]
    assert route_model(row(claimed_total=lim), 0.001, cfg)[0] == "stp"
    assert route_model(row(claimed_total=lim + 1), 0.001, cfg)[0] == "desk"


def test_stp_score_limit(cfg):
    s = cfg["routing"]["stp_max_score"]
    assert route_model(row(), s - 1e-6, cfg)[0] == "stp"
    assert route_model(row(), s, cfg)[0] == "desk"


def test_variance_blocks_stp_then_sends_to_field(cfg):
    r = cfg["routing"]
    assert route_model(row(variance=r["stp_max_variance"] + 0.01), 0.001, cfg)[0] == "desk"
    assert route_model(row(variance=r["field_min_variance"] + 0.01), 0.001, cfg)[0] == "field"


def test_siu_threshold(cfg):
    s = cfg["routing"]["siu_min_score"]
    assert route_model(row(), s, cfg)[0] == "siu"
    assert route_model(row(), s - 0.01, cfg)[0] == "field"


def test_ring_lowers_siu_threshold(cfg):
    s = cfg["routing"]["siu_ring_min_score"]
    assert route_model(row(ring_flag=True, flags_fired=["RF-RING"]), s, cfg)[0] == "siu"


def test_high_amount_goes_to_field(cfg):
    assert route_model(row(claimed_total=cfg["routing"]["field_min_amount"]), 0.001, cfg)[0] == "field"


def test_coverage_question_never_stp(cfg):
    assert route_model(row(coverage_status="question"), 0.0, cfg)[0] == "desk"


def test_denied_goes_to_desk_even_with_high_score(cfg):
    assert route_model(row(coverage_status="denied"), 0.99, cfg)[0] == "desk"


def test_routes_come_with_reasons(cfg):
    for s in (0.0, 0.05, 0.5):
        q, reasons = route_model(row(), s, cfg)
        assert reasons and all(isinstance(x, str) for x in reasons)


def test_thresholds_are_config_driven(cfg):
    c = copy.deepcopy(cfg)
    c["routing"]["stp_max_amount"] = 1000
    assert route_model(row(), 0.0, c)[0] == "desk"


def test_rules_only_policy(cfg):
    assert route_rules(row(), cfg)[0] == "stp"
    assert route_rules(row(flags_fired=["RF-LATE-REPORT", "RF-FAR-FROM-HOME"]), cfg)[0] == "siu"
    assert route_rules(row(flags_fired=["RF-PHOTO-BEFORE-LOSS"]), cfg)[0] == "siu"
    assert route_rules(row(flags_fired=["RF-LATE-REPORT"]), cfg)[0] == "desk"


def test_all_adjuster_never_stp_or_siu(cfg):
    assert route_all_adjuster(row(), cfg)[0] == "desk"
    assert route_all_adjuster(row(claimed_total=50000), cfg)[0] == "field"


# ---- evaluation accounting ----


def acct_frame():
    # four claims: legit, inflated, fraud (caught at siu), fraud (missed at stp)
    return pd.DataFrame(
        {
            "claim_id": ["a", "b", "c", "d"],
            "is_fraud": [0, 0, 1, 1],
            "coverage_status": ["covered"] * 4,
            "pay_stp": [1000.0, 1500.0, 2000.0, 800.0],
            "pay_desk": [1000.0, 1300.0, 1900.0, 800.0],
            "pay_field": [1000.0, 1000.0, 1800.0, 700.0],
            "fair_settlement": [1000.0, 1000.0, 0.0, 0.0],
            "review_draw": [0.5, 0.5, 0.1, 0.1],
        }
    )


def test_accounting_adds_up(cfg):
    df = acct_frame()
    a = account(df, pd.Series(["stp", "desk", "siu", "stp"]), cfg)
    assert a["leakage"] == 300.0  # inflated claim paid 1,300 at desk vs 1,000 fair
    assert a["missed_fraud"] == 800.0  # stp never catches fraud
    assert a["fraud_caught_claims"] == 1 and a["fraud_caught_dollars"] == 2000.0
    assert a["leakage_plus_missed_fraud"] == 1100.0
    assert a["paid"] == 1000 + 1300 + 0 + 800
    assert a["siu_precision"] == 1.0 and a["stp_share"] == 0.5
    q = cfg["queues"]
    assert a["avg_cycle_days"] == pytest.approx(
        (2 * q["stp"]["handling_days"] + q["desk"]["handling_days"] + q["siu"]["handling_days"]) / 4
    )
    assert a["handling_cost"] == 2 * q["stp"]["cost"] + q["desk"]["cost"] + q["siu"]["cost"]


def test_denied_claims_pay_nothing(cfg):
    df = acct_frame()
    df["coverage_status"] = "denied"
    a = account(df, pd.Series(["stp", "desk", "field", "siu"]), cfg)
    assert a["paid"] == 0.0 and a["leakage"] == 0.0


def test_field_pays_inspected_damage(cfg):
    a = account(acct_frame(), pd.Series(["field"] * 4), cfg)
    assert a["leakage"] == 0.0


def test_gate_compares_to_rules_only():
    ok, _ = gate({"model": {"leakage_plus_missed_fraud": 10}, "rules_only": {"leakage_plus_missed_fraud": 11}})
    bad, msg = gate({"model": {"leakage_plus_missed_fraud": 12}, "rules_only": {"leakage_plus_missed_fraud": 11}})
    assert ok and not bad and ">" in msg


# ---- model: time split, no leakage, calibration ----


def test_splits_are_ordered_in_time(built, cfg):
    paths, _ = built
    df = pd.read_pickle(paths.scored)
    s = df.groupby("split")["report_date"].agg(["min", "max"])
    assert s.loc["train", "max"] < s.loc["calibration", "min"]
    assert s.loc["calibration", "max"] < s.loc["test", "min"]
    assert (splits(df, cfg) == df["split"]).all()


def test_network_features_do_not_see_the_future(cfg):
    ds = generate(3000, 11)
    full, _ = process_history(ds.claims, ds.weather, cfg)
    cutoff = ds.claims["report_date"].iloc[1500]
    early = ds.claims[ds.claims["report_date"] < cutoff].reset_index(drop=True)
    part, _ = process_history(early, ds.weather, cfg)
    cols = [c for c in full.columns if c.startswith(("net_", "shop_")) or c in ("prior_claims_3y", "ring_flag")]
    pd.testing.assert_frame_equal(full.loc[: len(part) - 1, cols].reset_index(drop=True), part[cols])


def test_model_trained_only_on_training_months(built):
    paths, report = built
    df = pd.read_pickle(paths.scored)
    assert report["model"]["train_rows"] == int((df["split"] == "train").sum())


def test_model_ranks_fraud_on_held_out_months(built):
    _, report = built
    assert report["model"]["test"]["roc_auc"] > 0.85


def test_calibrated_scores_are_probabilities(built):
    paths, _ = built
    df = pd.read_pickle(paths.scored)
    assert df["score"].between(0, 1).all()
    test = df[df["split"] == "test"]
    assert abs(test["score"].mean() - test["is_fraud"].mean()) < 0.03


def test_explanations_name_features(built):
    paths, _ = built
    m = FraudModel.load(paths.model)
    df = pd.read_pickle(paths.scored)
    top = df.sort_values("score").iloc[-1]
    reasons = m.explain({f: top[f] for f in m.features})
    assert len(reasons) == 5 and all(r["label"] for r in reasons)
    assert reasons[0]["contribution"] > 0


def test_model_policy_beats_baseline_on_small_data(built, cfg):
    # The 8,000-claim test set has about 60 held-out fraud claims, too few for the rules-only gate to be stable.
    # The gate itself runs on the full 60,000 claims in CI; here we check the direction of each metric.
    _, report = built
    p = report["policies"]
    assert p["model"]["leakage_plus_missed_fraud"] < 0.8 * p["all_adjuster"]["leakage_plus_missed_fraud"]
    assert p["model"]["leakage_plus_missed_fraud"] < 1.15 * p["rules_only"]["leakage_plus_missed_fraud"]
    assert p["model"]["siu_precision"] > p["rules_only"]["siu_precision"]
    assert p["model"]["stp_share"] > p["rules_only"]["stp_share"]
    assert report["gate"]["message"].startswith("model policy leakage plus missed fraud")


def test_evaluation_is_paired_and_deterministic(built, cfg):
    paths, _ = built
    df = pd.read_pickle(paths.scored)
    test = df[df["split"] == "test"]
    a = evaluate_policies(test, test["score"].to_numpy(), cfg)
    b = evaluate_policies(test, test["score"].to_numpy(), cfg)
    assert a == b
    assert len(set(assign(test, np.zeros(len(test)), cfg, "all_adjuster")["queue"]) - {"desk", "field"}) == 0
