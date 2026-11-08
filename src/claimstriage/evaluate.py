"""Policy evaluation on held-out claims: what each routing policy pays, catches, and costs.

Payment model per queue (assumptions in config/routing.yaml):
- Coverage denied pays zero in every queue.
- stp pays the claimed estimate after deductible and limits.
- desk pays the price-book adjusted estimate (padding the book can see is removed).
- field and siu inspect the damage and pay the true damage cost (all padding removed).
- A fraudulent claim is caught with the queue's catch rate and then pays zero. The same random draw per claim
  is used for every policy, so the comparison is paired.

Leakage is overpayment above the fair settlement on non-fraud claims. Missed fraud is money paid on fraud.
"""

from __future__ import annotations

import pandas as pd

from .routing import POLICIES, QUEUES


def assign(df: pd.DataFrame, scores, cfg: dict, policy: str) -> pd.DataFrame:
    fn = POLICIES[policy]
    rows = df.to_dict("records")
    q, why = [], []
    for c, s in zip(rows, scores, strict=True):
        queue, reasons = fn(c, float(s), cfg)
        q.append(queue)
        why.append(reasons)
    return pd.DataFrame({"claim_id": df["claim_id"].to_numpy(), "queue": q, "reasons": why}, index=df.index)


def account(df: pd.DataFrame, queues: pd.Series, cfg: dict) -> dict:
    qcfg = cfg["queues"]
    catch = queues.map(lambda q: qcfg[q]["fraud_catch_rate"])
    caught = (df["is_fraud"] == 1) & (df["review_draw"] < catch)
    denied = df["coverage_status"] == "denied"
    basis = pd.Series(0.0, index=df.index)
    basis[queues == "stp"] = df.loc[queues == "stp", "pay_stp"]
    basis[queues == "desk"] = df.loc[queues == "desk", "pay_desk"]
    fs = queues.isin(["field", "siu"])
    basis[fs] = df.loc[fs, "pay_field"]
    paid = basis.where(~(denied | caught), 0.0)
    fraud = df["is_fraud"] == 1
    over = (paid - df["fair_settlement"]).clip(lower=0)
    under = (df["fair_settlement"] - paid).clip(lower=0)
    siu = queues == "siu"
    n = len(df)
    by_queue = {q: int((queues == q).sum()) for q in QUEUES}
    leakage = float(over[~fraud].sum())
    missed = float(paid[fraud].sum())
    return {
        "claims": n,
        "queues": by_queue,
        "stp_share": round(by_queue["stp"] / n, 4) if n else 0.0,
        "paid": round(float(paid.sum()), 2),
        "fair": round(float(df["fair_settlement"].sum()), 2),
        "leakage": round(leakage, 2),
        "underpaid": round(float(under[~fraud].sum()), 2),
        "fraud_claims": int(fraud.sum()),
        "fraud_caught_claims": int(caught.sum()),
        "fraud_caught_dollars": round(float(df.loc[caught, "pay_stp"].where(~denied[caught], 0.0).sum()), 2),
        "missed_fraud": round(missed, 2),
        "leakage_plus_missed_fraud": round(leakage + missed, 2),
        "siu_referrals": by_queue["siu"],
        "siu_precision": round(float(fraud[siu].mean()), 4) if siu.any() else 0.0,
        "avg_cycle_days": round(float(queues.map(lambda q: qcfg[q]["handling_days"]).mean()), 2) if n else 0.0,
        "handling_cost": round(float(queues.map(lambda q: qcfg[q]["cost"]).sum()), 2),
    }


def evaluate_policies(df: pd.DataFrame, scores, cfg: dict) -> dict:
    out = {}
    for name in POLICIES:
        a = assign(df, scores, cfg, name)
        out[name] = account(df, a["queue"], cfg)
    return out


def gate(results: dict) -> tuple[bool, str]:
    m, r = results["model"]["leakage_plus_missed_fraud"], results["rules_only"]["leakage_plus_missed_fraud"]
    ok = m <= r
    return ok, (f"model policy leakage plus missed fraud ${m:,.0f} {'<=' if ok else '>'} rules-only ${r:,.0f}")
