import pandas as pd

from claimstriage.synth import generate


def test_same_seed_same_data():
    a, b = generate(1500, 3), generate(1500, 3)
    pd.testing.assert_frame_equal(a.claims.drop(columns=["estimate_lines"]), b.claims.drop(columns=["estimate_lines"]))


def test_size_and_labels(built):
    paths, report = built
    claims = pd.read_pickle(paths.claims)
    assert 0.97 * 8000 < len(claims) <= 8000  # a few late reports fall after the snapshot date
    assert claims["report_date"].max() <= pd.Timestamp("2026-10-31").date()
    labels = claims["truth_label"].value_counts(normalize=True)
    assert 0.01 < labels["fraud"] < 0.05 and 0.05 < labels["inflated"] < 0.15
    assert set(claims["truth_fraud_type"].dropna()) == {"staged_ring", "prior_damage", "weather_opportunistic"}


def test_fraud_has_zero_fair_settlement(built):
    paths, _ = built
    claims = pd.read_pickle(paths.claims)
    assert (claims.loc[claims["truth_label"] == "fraud", "fair_settlement"] == 0).all()


def test_inflated_estimates_exceed_true_damage(built):
    paths, _ = built
    c = pd.read_pickle(paths.claims)
    inf = c[c["truth_label"] == "inflated"]
    assert (inf["estimate_total"] > inf["damage_cost"]).mean() > 0.95


def test_ring_members_share_phones_or_addresses(built):
    paths, _ = built
    c = pd.read_pickle(paths.claims)
    ring = c[c["truth_ring_id"].notna()]
    for _, g in ring.groupby("truth_ring_id"):
        assert (
            g["phone"].duplicated().any() or g["address_id"].duplicated().any() or g["third_party_phone"].notna().all()
        )


def test_catastrophe_events_produce_claims(built):
    paths, _ = built
    c = pd.read_pickle(paths.claims)
    hail = c[(c["loss_cause"] == "hail") & (c["region"] == "Plains")]
    assert (pd.to_datetime(hail["loss_date"]).dt.strftime("%Y-%m") == "2025-05").sum() > 100
