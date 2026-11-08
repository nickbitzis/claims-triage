"""Fraud model: gradient boosting on claim and network features, time split, Platt calibration, explanations."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .pipeline import CATEGORICAL, FEATURE_LABELS, FEATURES, split_of


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class FraudModel:
    clf: HistGradientBoostingClassifier
    calibrator: LogisticRegression
    baseline: dict
    metrics: dict = field(default_factory=dict)
    features: list[str] = field(default_factory=lambda: list(FEATURES))

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X[self.features])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict_proba(_logit(self.raw(X)).reshape(-1, 1))[:, 1]

    def explain(self, feats: dict, top: int = 5) -> list[dict]:
        """Occlusion: the change in log-odds of the calibrated score when each feature is set to its training median.

        Log-odds, not probability, so reasons stay readable when the score is near 0 or 1.
        """
        x = pd.DataFrame([{f: feats.get(f, np.nan) for f in self.features}])
        base = float(_logit(self.predict(x))[0])
        probes = pd.concat([x] * len(self.features), ignore_index=True)
        for i, f in enumerate(self.features):
            probes.loc[i, f] = self.baseline[f]
        deltas = base - _logit(self.predict(probes))
        out = []
        for f, d in zip(self.features, deltas, strict=True):
            v = feats.get(f)
            if f == "amount_log" and v is not None:
                shown = f"${math.expm1(v):,.0f}"
            elif v is None or (isinstance(v, float) and math.isnan(v)):
                shown = "none"
            else:
                shown = f"{v:,.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
            out.append(
                {"feature": f, "label": FEATURE_LABELS.get(f, f), "value": shown, "contribution": round(float(d), 4)}
            )
        out.sort(key=lambda r: -abs(r["contribution"]))
        return out[:top]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> FraudModel:
        return pickle.loads(path.read_bytes())


def splits(df: pd.DataFrame, cfg: dict) -> pd.Series:
    return df["report_date"].map(lambda d: split_of(d, cfg))


def train(df: pd.DataFrame, cfg: dict) -> tuple[FraudModel, pd.Series]:
    m = cfg["model"]
    split = splits(df, cfg)
    tr, ca = df[split == "train"], df[split == "calibration"]
    clf = HistGradientBoostingClassifier(
        max_iter=m["max_iter"],
        learning_rate=m["learning_rate"],
        max_leaf_nodes=m["max_leaf_nodes"],
        l2_regularization=m["l2_regularization"],
        random_state=m["seed"],
        categorical_features=[f in CATEGORICAL for f in FEATURES],
        early_stopping=False,
    )
    clf.fit(tr[FEATURES], tr["is_fraud"])
    cal = LogisticRegression(C=100.0)
    cal.fit(_logit(clf.predict_proba(ca[FEATURES])[:, 1]).reshape(-1, 1), ca["is_fraud"])
    baseline = {f: (float(tr[f].median()) if tr[f].notna().any() else np.nan) for f in FEATURES}
    model = FraudModel(clf=clf, calibrator=cal, baseline=baseline)
    model.metrics = {
        "train_rows": int(len(tr)),
        "calibration_rows": int(len(ca)),
        "test_rows": int((split == "test").sum()),
        "train_fraud_rate": round(float(tr["is_fraud"].mean()), 4),
        "train_end": m["train_end"],
        "calibration_end": m["calibration_end"],
    }
    return model, split


def score_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if len(set(y)) < 2:
        return {"n": int(len(y))}
    bins = np.clip((p * 10).astype(int), 0, 9)
    calib = []
    for b in range(10):
        m = bins == b
        if m.sum() >= 20:
            calib.append(
                {
                    "bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}",
                    "n": int(m.sum()),
                    "mean_score": round(float(p[m].mean()), 4),
                    "fraud_rate": round(float(y[m].mean()), 4),
                }
            )
    return {
        "n": int(len(y)),
        "fraud_rate": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "calibration": calib,
    }
