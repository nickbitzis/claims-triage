from claimstriage.network import EntityIndex, ego_graph, ring_components

from .conftest import make_claim


def claim(i, claimant, phone, addr, tp=None, shop="S1"):
    return make_claim(
        claim_id=f"T-{i}", claimant_id=claimant, phone=phone, address_id=addr, third_party_phone=tp, shop_id=shop
    )


def feed(idx, claims):
    out = []
    for c in claims:
        f = idx.features(c)
        idx.add(c, f, False)
        out.append(f)
    return out


def test_ring_sharing_phones_is_flagged():
    claims = [
        claim(i, f"R{i}", "555-RING" if i % 2 == 0 else f"555-{i}", "A-RING" if i < 3 else f"A{i}") for i in range(6)
    ]
    f = feed(EntityIndex(), claims)
    assert f[4]["ring_flag"] and f[4]["net_claimants"] >= 4
    assert not f[0]["ring_flag"]


def test_household_is_not_a_ring():
    claims = [claim(1, "H1", "555-1", "A-HOME"), claim(2, "H2", "555-2", "A-HOME"), claim(3, "H1", "555-1", "A-HOME")]
    f = feed(EntityIndex(), claims)
    assert not any(x["ring_flag"] for x in f)
    assert f[-1]["net_address_claimants"] == 2


def test_third_party_phone_links_claimants():
    claims = [claim(1, "X1", "555-A", "A1"), claim(2, "X2", "555-B", "A2", tp="555-A")]
    f = feed(EntityIndex(), claims)
    assert f[1]["net_claimants"] == 2 and f[1]["net_prior_claims"] == 1


def test_missing_third_party_phone_never_links():
    claims = [claim(i, f"N{i}", f"555-{i}", f"A{i}", tp=float("nan")) for i in range(10)]
    f = feed(EntityIndex(), claims)
    assert max(x["net_claimants"] for x in f) == 1


def test_features_use_history_only():
    idx = EntityIndex()
    c1, c2 = claim(1, "Z1", "555-Z", "AZ"), claim(2, "Z2", "555-Z", "AZ2")
    before = idx.features(c2)
    feed(idx, [c1])
    after = idx.features(c2)
    assert before["net_prior_claims"] == 0 and after["net_prior_claims"] == 1


def test_shop_concentration_counts_prior_claims():
    idx = EntityIndex()
    f = feed(idx, [claim(i, f"S{i}", f"555-{i}", f"A{i}", shop="SHOP-X") for i in range(4)])
    assert f[-1]["shop_prior_claims"] == 3


def test_batch_ring_components():
    claims = [claim(i, f"R{i}", "555-RING", f"A{i}") for i in range(4)] + [claim(9, "Solo", "555-9", "A9")]
    rings = ring_components(claims)
    assert len(rings) == 1 and rings[0]["size"] == 4


def test_ego_graph_contains_linked_claims():
    import pandas as pd

    claims = [claim(1, "E1", "555-E", "AE1"), claim(2, "E2", "555-E", "AE2"), claim(3, "E3", "555-3", "AE3")]
    g = ego_graph(pd.DataFrame(claims), claims[0])
    ids = {n["id"] for n in g["nodes"]}
    assert {"T-1", "T-2"} <= ids and "T-3" not in ids
