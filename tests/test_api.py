import sqlite3
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from claimstriage.app import create_app
from claimstriage.notes import BANNED_CUSTOMER_WORDS, AnthropicProvider, MockProvider, get_provider, write_notes
from claimstriage.store import Store


@pytest.fixture(scope="module")
def client(built):
    paths, _ = built
    return TestClient(create_app(paths, store=Store(":memory:")))


def top_of(client, route):
    return client.get(f"/api/queues/{route}?limit=1").json()["items"][0]["claim_id"]


def test_health_and_metrics(client):
    assert client.get("/healthz").json()["status"] == "ok"
    client.get("/api/queues")
    m = client.get("/metrics").text
    assert "ct_http_requests_total" in m and "ct_gate_passed" in m and "ct_http_request_duration_seconds_bucket" in m


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Claims Triage" in r.text


def test_queues_sorted_and_counted(client):
    counts = client.get("/api/queues").json()
    assert set(counts) == {"stp", "desk", "field", "siu"} and counts["stp"] > 0
    siu = client.get("/api/queues/siu?limit=20").json()
    scores = [x["score"] for x in siu["items"]]
    assert scores == sorted(scores, reverse=True) and siu["total"] == counts["siu"]


def test_unknown_queue_and_claim_404(client):
    assert client.get("/api/queues/nope").status_code == 404
    assert client.get("/api/claims/BWM-999999").status_code == 404


def test_claim_detail_has_everything_and_no_truth(client):
    d = client.get(f"/api/claims/{top_of(client, 'siu')}").json()
    for k in ("coverage", "estimate", "weather", "network", "red_flags", "fraud", "route", "graph"):
        assert k in d
    assert d["route"]["queue"] == "siu" and d["fraud"]["reasons"]
    text = str(d)
    for private in ("truth_label", "fair_settlement", "review_draw", "damage_cost", "is_fraud"):
        assert private not in text


def test_submit_fnol_end_to_end(client):
    sample = client.get("/api/fnol/sample").json()
    r = client.post("/api/fnol", json=sample, headers={"x-adjuster": "j.alvarez"})
    assert r.status_code == 200
    d = r.json()
    assert d["claim_id"].startswith("FNOL-") and d["coverage"]["status"] == "covered"
    assert d["route"]["queue"] in ("stp", "desk", "field", "siu") and 0 <= d["fraud"]["score"] <= 1
    assert d["estimate"]["adjusted_total"] <= d["estimate"]["claimed_total"]
    again = client.get(f"/api/claims/{d['claim_id']}").json()
    assert again["claim_id"] == d["claim_id"]
    audit = client.get(f"/api/audit?claim_id={d['claim_id']}").json()["items"]
    assert audit[0]["action"] == "fnol_submitted" and audit[0]["actor"] == "j.alvarez"
    q = client.get(f"/api/queues/{d['route']['queue']}").json()["items"]
    assert any(x["claim_id"] == d["claim_id"] and x["submitted"] for x in q)


def test_padded_fnol_with_old_photos_is_not_stp(client):
    s = client.get("/api/fnol/sample").json()
    s["estimate_lines"][0]["unit_price"] = 1400.0
    s["photos"]["exif_min"] = "2026-06-01T10:00:00"
    d = client.post("/api/fnol", json=s).json()
    assert d["route"]["queue"] in ("field", "siu")
    assert any(f["id"] == "RF-PHOTO-BEFORE-LOSS" and f["fired"] for f in d["red_flags"])
    assert d["estimate"]["flagged_lines"] >= 1


def test_fnol_validation(client):
    s = client.get("/api/fnol/sample").json()
    assert client.post("/api/fnol", json={**s, "policy_id": "P999999"}).status_code == 400
    assert client.post("/api/fnol", json={**s, "estimate_lines": []}).status_code == 400
    bad = {**s, "estimate_lines": [{**s["estimate_lines"][0], "kind": "magic"}]}
    assert client.post("/api/fnol", json=bad).status_code == 422


def test_evaluation_endpoint(client):
    ev = client.get("/api/evaluation").json()
    assert set(ev["policies"]) == {"all_adjuster", "rules_only", "model"} and "passed" in ev["gate"]


def test_notes_endpoint_and_audit(client):
    cid = top_of(client, "siu")
    n = client.post(f"/api/claims/{cid}/notes").json()
    assert n["provider"] == "mock" and cid in n["adjuster_summary"]
    assert not any(w in n["customer_message"].lower() for w in BANNED_CUSTOMER_WORDS)
    assert client.get(f"/api/audit?claim_id={cid}").json()["items"][0]["action"] == "notes_written"


def test_audit_is_append_only():
    s = Store(":memory:")
    s.audit("a", "x", "c1", {})
    with pytest.raises(sqlite3.DatabaseError):
        s.con.execute("UPDATE audit SET actor = 'b'")
    with pytest.raises(sqlite3.DatabaseError):
        s.con.execute("DELETE FROM audit")


# ---- notes providers ----


class FakeClient:
    def __init__(self, text):
        self.text, self.calls = text, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            usage=SimpleNamespace(input_tokens=321, output_tokens=54),
        )


def detail_for(client, route):
    return client.get(f"/api/claims/{top_of(client, route)}").json()


def test_claude_provider_uses_facts_only(client):
    fake = FakeClient('{"adjuster_summary": "Summary.", "customer_message": "We are reviewing your claim."}')
    out = write_notes(AnthropicProvider(client=fake), detail_for(client, "desk"))
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5" and "never mention" in call["system"]
    assert "truth" not in call["messages"][0]["content"] and "phone" not in call["messages"][0]["content"]
    assert out["adjuster_summary"] == "Summary." and out["usage"]["input_tokens"] == 321


def test_claude_customer_message_cannot_leak_investigation(client):
    fake = FakeClient(
        '{"adjuster_summary": "S", "customer_message": "Your claim was sent to our fraud investigators."}'
    )
    out = write_notes(AnthropicProvider(client=fake), detail_for(client, "siu"))
    assert "fraud" not in out["customer_message"].lower()


def test_claude_malformed_reply_falls_back(client):
    out = write_notes(AnthropicProvider(client=FakeClient("not json")), detail_for(client, "stp"))
    assert out["adjuster_summary"] == "not json" and out["customer_message"]


def test_mock_customer_message_never_mentions_siu(client):
    out = write_notes(MockProvider(), detail_for(client, "siu"))
    assert not any(w in out["customer_message"].lower() for w in BANNED_CUSTOMER_WORDS)


def test_default_provider_is_offline(monkeypatch):
    monkeypatch.delenv("CT_NOTES_PROVIDER", raising=False)
    sys.modules.pop("anthropic", None)
    assert get_provider().name == "mock" and "anthropic" not in sys.modules
