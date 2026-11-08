"""Batch job: generate, process history, train, score, route, evaluate. Writes artifacts to the data dir."""

from __future__ import annotations

import copy
import itertools
import json
import pickle
import time

import pandas as pd

from .evaluate import account, assign, evaluate_policies, gate
from .model import FraudModel, score_metrics, train
from .pipeline import Paths, load_config, process_history
from .routing import route_model
from .synth import generate


def run_all(paths: Paths, n_claims: int = 60_000, seed: int = 42, cfg: dict | None = None, log=print) -> dict:
    cfg = cfg or load_config()
    paths.data.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ds = generate(n_claims, seed)
    log(f"generated {len(ds.claims):,} claims in {time.time() - t0:.1f}s")
    t = time.time()
    df, index = process_history(ds.claims, ds.weather, cfg)
    ds.claims["fair_settlement"] = df["fair_settlement"].to_numpy()
    log(f"processed coverage, estimates, weather, network, red flags in {time.time() - t:.1f}s")
    t = time.time()
    model, split = train(df, cfg)
    df["split"] = split.to_numpy()
    df["score"] = model.predict(df)
    log(
        f"trained on {model.metrics['train_rows']:,} claims, calibrated on {model.metrics['calibration_rows']:,} "
        f"in {time.time() - t:.1f}s"
    )
    routes = [route_model(c, s, cfg) for c, s in zip(df.to_dict("records"), df["score"], strict=True)]
    df["route"] = [r[0] for r in routes]
    df["route_reasons"] = [r[1] for r in routes]
    test = df[df["split"] == "test"]
    results = evaluate_policies(test, test["score"].to_numpy(), cfg)
    ok, msg = gate(results)
    by_type = {}
    for ft, g in ds.claims.loc[test.index].groupby("truth_fraud_type"):
        sc = test.loc[g.index, "score"]
        by_type[ft] = {
            "claims": int(len(g)),
            "mean_score": round(float(sc.mean()), 3),
            "to_siu_or_field": round(float(test.loc[g.index, "route"].isin(["siu", "field"]).mean()), 3),
        }
    report = {
        "dataset": {
            "claims": int(len(df)),
            "seed": seed,
            "test_claims": int(len(test)),
            "test_months": f"{pd.Timestamp(test['report_date'].min()):%Y-%m} to "
            f"{pd.Timestamp(test['report_date'].max()):%Y-%m}",
            "labels": ds.claims["truth_label"].value_counts().to_dict(),
        },
        "config_version": cfg["version"],
        "model": {**model.metrics, "test": score_metrics(test["is_fraud"].to_numpy(), test["score"].to_numpy())},
        "fraud_by_type_test": by_type,
        "policies": results,
        "gate": {"passed": ok, "message": msg},
        "queues": cfg["queues"],
    }
    ds.claims.to_pickle(paths.claims)
    ds.weather.to_csv(paths.weather, index=False)
    df.to_pickle(paths.scored)
    model.save(paths.model)
    paths.index.write_bytes(pickle.dumps(index))
    paths.evaluation.write_text(json.dumps(report, indent=2, default=str))
    log(f"gate: {msg} ({'pass' if ok else 'FAIL'}); total {time.time() - t0:.1f}s")
    return report


def tune(paths: Paths, cfg: dict | None = None) -> list[dict]:
    """Grid search the model policy's score thresholds on the calibration months only."""
    cfg = cfg or load_config()
    df = pd.read_pickle(paths.scored)
    model = FraudModel.load(paths.model)
    cal = df[df["split"] == "calibration"]
    p = model.predict(cal)
    out = []
    for siu, fld, stp, fv in itertools.product(
        [0.2, 0.3, 0.4, 0.5], [0.03, 0.06, 0.1, 0.15], [0.01, 0.02, 0.05], [0.15, 0.2, 0.3]
    ):
        c = copy.deepcopy(cfg)
        c["routing"].update(siu_min_score=siu, field_min_score=fld, stp_max_score=stp, field_min_variance=fv)
        a = account(cal, assign(cal, p, c, "model")["queue"], c)
        out.append(
            {
                "siu_min_score": siu,
                "field_min_score": fld,
                "stp_max_score": stp,
                "field_min_variance": fv,
                "leakage_plus_missed_fraud": a["leakage_plus_missed_fraud"],
                "handling_cost": a["handling_cost"],
                "stp_share": a["stp_share"],
            }
        )
    out.sort(key=lambda r: (r["leakage_plus_missed_fraud"], r["handling_cost"]))
    return out
