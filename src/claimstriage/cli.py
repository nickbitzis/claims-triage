"""Command line: all (batch job), evaluate, gate, tune, serve."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .pipeline import Paths

POLICY_NAMES = {"all_adjuster": "Everything to an adjuster", "rules_only": "Rules only", "model": "Model policy"}


def table(report: dict) -> str:
    p = report["policies"]
    names = list(POLICY_NAMES)
    rows = [
        ("Straight-through share", lambda r: f"{r['stp_share']:.1%}"),
        (
            "Desk / field / SIU",
            lambda r: f"{r['queues']['desk']:,} / {r['queues']['field']:,} / {r['queues']['siu']:,}",
        ),
        ("Leakage (overpaid vs fair)", lambda r: f"${r['leakage']:,.0f}"),
        ("Missed fraud (paid on fraud)", lambda r: f"${r['missed_fraud']:,.0f}"),
        ("Leakage + missed fraud", lambda r: f"${r['leakage_plus_missed_fraud']:,.0f}"),
        ("Fraud dollars caught", lambda r: f"${r['fraud_caught_dollars']:,.0f}"),
        ("Fraud claims caught", lambda r: f"{r['fraud_caught_claims']} of {r['fraud_claims']}"),
        ("SIU referral precision", lambda r: f"{r['siu_precision']:.1%}" if r["siu_referrals"] else "n/a"),
        ("Average cycle time (days)", lambda r: f"{r['avg_cycle_days']:.2f}"),
        ("Handling cost", lambda r: f"${r['handling_cost']:,.0f}"),
    ]
    out = ["| | " + " | ".join(POLICY_NAMES[n] for n in names) + " |", "|---|" + "---:|" * len(names)]
    for label, fn in rows:
        out.append(f"| {label} | " + " | ".join(fn(p[n]) for n in names) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="claimstriage")
    ap.add_argument("--data-dir", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("all", help="generate, process, train, score, route, evaluate")
    a.add_argument("--claims", type=int, default=60_000)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--gate", action="store_true", help="exit 1 if the model policy loses to rules-only")
    sub.add_parser("evaluate", help="print the policy comparison from the last run")
    sub.add_parser("gate", help="exit 1 if leakage plus missed fraud exceeds the rules-only baseline")
    sub.add_parser("tune", help="grid search routing thresholds on the calibration months")
    s = sub.add_parser("serve")
    s.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8834")))
    args = ap.parse_args(argv)
    paths = Paths.from_env(args.data_dir)

    if args.cmd == "all":
        from .batch import run_all

        report = run_all(paths, args.claims, args.seed)
        print(table(report))
        if args.gate and not report["gate"]["passed"]:
            sys.exit(1)
    elif args.cmd in ("evaluate", "gate"):
        report = json.loads(paths.evaluation.read_text())
        if args.cmd == "evaluate":
            print(table(report))
            print(json.dumps(report["model"]["test"] | {"calibration": "see evaluation.json"}, indent=2))
        print(("PASS: " if report["gate"]["passed"] else "FAIL: ") + report["gate"]["message"])
        if not report["gate"]["passed"]:
            sys.exit(1)
    elif args.cmd == "tune":
        from .batch import tune

        for r in tune(paths)[:10]:
            print(r)
    elif args.cmd == "serve":
        import uvicorn

        from .app import create_app

        uvicorn.run(create_app(paths), host=args.host, port=args.port, log_level="warning")
