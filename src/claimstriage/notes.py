"""Claim notes: an adjuster summary and a customer status message, written from computed facts only.

The model never sees raw claimant data or the ground truth. It gets the triage result as facts, and the
customer message must never mention fraud, scores, or investigations: a test checks the offline template,
and the Claude prompt says so. The default provider is an offline template so the demo needs no key.
"""

from __future__ import annotations

import json
import os

DEFAULT_MODEL = "claude-opus-5-5"
QUEUE_TEXT = {
    "stp": "approved for straight-through payment",
    "desk": "assigned to a desk adjuster",
    "field": "assigned to a field adjuster for an inspection",
    "siu": "referred to the special investigations unit",
}
CUSTOMER_NEXT = {
    "stp": "Your payment has been approved and should arrive within 2 business days.",
    "desk": "An adjuster is reviewing your claim and will contact you within 3 business days.",
    "field": "An adjuster will contact you within 3 business days to schedule an inspection.",
    "siu": "Your claim needs some additional review. A claims specialist will contact you within 5 business days.",
}
SYSTEM = (
    "You write claim notes for Bramblewood Mutual, a personal auto and home insurer. The triage engine has "
    "already decided coverage, the adjusted estimate, and the routing. Use only the facts given. Do not invent "
    "amounts, dates, or causes. Plain English, short sentences. In any message to the customer, never mention "
    "fraud, fraud scores, red flags, investigations, or the special investigations unit."
)
BANNED_CUSTOMER_WORDS = ("fraud", "investigat", "siu", "red flag", "score", "suspic")


def claim_facts(detail: dict) -> dict:
    est, cov = detail["estimate"], detail["coverage"]
    return {
        "claim_id": detail["claim_id"],
        "line": detail["claim"]["line"],
        "peril": detail["claim"]["peril"],
        "loss_date": str(detail["claim"]["loss_date"]),
        "coverage_status": cov["status"],
        "failed_checks": [c["detail"] for c in cov["checks"] if not c["passed"]],
        "deductible": cov["deductible"],
        "claimed_total": est["claimed_total"],
        "adjusted_total": est["adjusted_total"],
        "flagged_lines": [f"{x['desc']}: {x['reason']}" for x in est["lines"] if x["flagged"]][:6],
        "payable_on_adjusted": detail["payable_adjusted"],
        "weather": detail["weather"]["detail"] if detail["weather"]["status"] != "not_applicable" else None,
        "fraud_score": detail["fraud"]["score"],
        "fraud_reasons": [f"{r['label']} = {r['value']}" for r in detail["fraud"]["reasons"] if r["contribution"] > 0][
            :4
        ],
        "red_flags": [f["detail"] for f in detail["red_flags"] if f["fired"]],
        "route": detail["route"]["queue"],
        "route_reasons": detail["route"]["reasons"],
    }


class MockProvider:
    name = "mock"
    model = "template-v1"

    def write(self, facts: dict) -> dict:
        parts = [
            f"{facts['claim_id']}: {facts['line']} {facts['peril'].replace('_', ' ')} loss on {facts['loss_date']}, "
            f"{QUEUE_TEXT[facts['route']]}."
        ]
        if facts["coverage_status"] == "denied":
            parts.append("Coverage denied: " + " ".join(facts["failed_checks"]))
        elif facts["coverage_status"] == "question":
            parts.append("Coverage open: " + " ".join(facts["failed_checks"]))
        else:
            parts.append(f"Coverage confirmed, deductible ${facts['deductible']:,.0f}.")
        parts.append(
            f"Estimate ${facts['claimed_total']:,.2f}, adjusted to ${facts['adjusted_total']:,.2f} "
            "against the price book"
            + (f" ({len(facts['flagged_lines'])} lines flagged)." if facts["flagged_lines"] else ".")
        )
        if facts["weather"]:
            parts.append(f"Weather: {facts['weather']}")
        parts.append(
            f"Fraud score {facts['fraud_score']:.2f}."
            + (" Main drivers: " + "; ".join(facts["fraud_reasons"]) + "." if facts["fraud_reasons"] else "")
        )
        if facts["red_flags"]:
            parts.append("Red flags: " + " ".join(facts["red_flags"]))
        parts.append("Routing: " + " ".join(facts["route_reasons"]))
        if facts["coverage_status"] == "denied":
            customer = (
                "We reviewed your claim against your policy and an adjuster will contact you within 3 business "
                "days to explain the coverage decision."
            )
        else:
            customer = f"Thank you for reporting your claim {facts['claim_id']}. {CUSTOMER_NEXT[facts['route']]}"
        return {
            "adjuster_summary": " ".join(parts),
            "customer_message": customer,
            "input_tokens": 0,
            "output_tokens": 0,
        }


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, client=None, model: str = DEFAULT_MODEL):
        if client is None:
            import anthropic  # lazy: only needed with the claude extra

            client = anthropic.Anthropic()
        self.client = client
        self.model = model

    def write(self, facts: dict) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=700,
            system=SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": "Return JSON with two keys: adjuster_summary (under 120 words, for the adjuster) and "
                    "customer_message (under 60 words, a status update for the policyholder).\n\nFacts (JSON):\n"
                    + json.dumps(facts, default=str),
                }
            ],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        try:
            data = json.loads(text)
            summary, customer = str(data["adjuster_summary"]), str(data["customer_message"])
        except (ValueError, KeyError, TypeError):
            fallback = MockProvider().write(facts)
            summary, customer = text or fallback["adjuster_summary"], fallback["customer_message"]
        if any(w in customer.lower() for w in BANNED_CUSTOMER_WORDS):
            customer = MockProvider().write(facts)["customer_message"]  # never send investigation details out
        return {
            "adjuster_summary": summary,
            "customer_message": customer,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }


def get_provider(name: str | None = None, client=None):
    name = name or os.environ.get("CT_NOTES_PROVIDER", "mock")
    if name in ("anthropic", "claude"):
        return AnthropicProvider(client=client, model=os.environ.get("CT_NOTES_MODEL", DEFAULT_MODEL))
    return MockProvider()


def write_notes(provider, detail: dict) -> dict:
    facts = claim_facts(detail)
    out = provider.write(facts)
    return {
        "advisory": True,
        "note": "Drafted from the triage engine's computed facts. The engine's decision is the source of truth.",
        "provider": provider.name,
        "model": provider.model,
        "adjuster_summary": out["adjuster_summary"],
        "customer_message": out["customer_message"],
        "usage": {"input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"]},
    }
