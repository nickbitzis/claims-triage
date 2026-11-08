"""Shared entity network: claimants linked by phone numbers, addresses, and third-party phones.

`EntityIndex` processes claims in report order and returns each claim's network features from history only
(state before the claim is added), so the same code serves training and live FNOL without leakage.
Repair shops are not graph links (popular shops would join everything); they enter as concentration features.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import networkx as nx


def tp_phone(c: dict) -> str | None:
    """Third-party phone, or None (pandas turns a missing value into NaN, which must not become a node)."""
    v = c.get("third_party_phone")
    return v if isinstance(v, str) and v else None


class EntityIndex:
    def __init__(self, ring_min_claimants: int = 4, ring_min_phone_share: int = 3):
        self.parent: dict[str, str] = {}
        self.n_claimants: dict[str, int] = {}
        self.n_claims: dict[str, int] = {}
        self.shops: dict[str, Counter] = {}
        self.phone_claimants: dict[str, set] = defaultdict(set)
        self.addr_claimants: dict[str, set] = defaultdict(set)
        self.shop_claims: Counter = Counter()
        self.shop_linked: Counter = Counter()
        self.shop_flagged: Counter = Counter()
        self.ring_min_claimants = ring_min_claimants
        self.ring_min_phone_share = ring_min_phone_share

    def _find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def _node(self, x: str, claimant: bool = False) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.n_claimants[x] = 1 if claimant else 0
            self.n_claims[x] = 0
            self.shops[x] = Counter()
        return self._find(x)

    def _union(self, a: str, b: str) -> str:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return ra
        if self.n_claims[ra] + self.n_claimants[ra] < self.n_claims[rb] + self.n_claimants[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.n_claimants[ra] += self.n_claimants.pop(rb)
        self.n_claims[ra] += self.n_claims.pop(rb)
        self.shops[ra].update(self.shops.pop(rb))
        return ra

    @staticmethod
    def entities(c: dict) -> list[str]:
        ents = [f"C:{c['claimant_id']}", f"P:{c['phone']}", f"A:{c['address_id']}"]
        if tp_phone(c):
            ents.append(f"P:{tp_phone(c)}")
        return ents

    def features(self, c: dict) -> dict:
        roots = {self._find(e) for e in self.entities(c) if e in self.parent}
        cid = f"C:{c['claimant_id']}"
        claimants = sum(self.n_claimants[r] for r in roots) + (0 if cid in self.parent else 1)
        claims = sum(self.n_claims[r] for r in roots)
        shops = Counter()
        for r in roots:
            shops.update(self.shops[r])
        shops[c["shop_id"]] += 1
        phone_share = len(self.phone_claimants[c["phone"]] | {c["claimant_id"]})
        if tp_phone(c):
            phone_share = max(phone_share, len(self.phone_claimants[tp_phone(c)] | {c["claimant_id"]}))
        addr_share = len(self.addr_claimants[c["address_id"]] | {c["claimant_id"]})
        sc = self.shop_claims[c["shop_id"]]
        ring = claimants >= self.ring_min_claimants or phone_share >= self.ring_min_phone_share
        return {
            "net_claimants": claimants,
            "net_prior_claims": claims,
            "net_phone_claimants": phone_share,
            "net_address_claimants": addr_share,
            "net_top_shop_share": round(max(shops.values()) / (claims + 1), 4),
            "shop_prior_claims": sc,
            "shop_linked_share": round(self.shop_linked[c["shop_id"]] / sc, 4) if sc else 0.0,
            "shop_flagged_share": round(self.shop_flagged[c["shop_id"]] / sc, 4) if sc else 0.0,
            "ring_flag": bool(ring),
        }

    def add(self, c: dict, feats: dict, estimate_flagged: bool) -> None:
        ents = self.entities(c)
        self._node(ents[0], claimant=True)
        for e in ents[1:]:
            self._node(e)
        root = ents[0]
        for e in ents[1:]:
            root = self._union(root, e)
        root = self._find(root)
        self.n_claims[root] += 1
        self.shops[root][c["shop_id"]] += 1
        self.phone_claimants[c["phone"]].add(c["claimant_id"])
        if tp_phone(c):
            self.phone_claimants[tp_phone(c)].add(c["claimant_id"])
        self.addr_claimants[c["address_id"]].add(c["claimant_id"])
        self.shop_claims[c["shop_id"]] += 1
        self.shop_linked[c["shop_id"]] += int(feats["net_claimants"] >= 3)
        self.shop_flagged[c["shop_id"]] += int(estimate_flagged)


def ring_components(claims: list[dict], min_claimants: int = 4, min_phone_share: int = 3) -> list[dict]:
    """Batch ring detection over a claim set: connected components of claimants via phones and addresses."""
    g = nx.Graph()
    for c in claims:
        for e in EntityIndex.entities(c)[1:]:
            g.add_edge(f"C:{c['claimant_id']}", e)
    rings = []
    for comp in nx.connected_components(g):
        claimants = {n for n in comp if n.startswith("C:")}
        phones = [n for n in comp if n.startswith("P:")]
        max_phone = max((sum(1 for nb in g.neighbors(p) if nb.startswith("C:")) for p in phones), default=0)
        if len(claimants) >= min_claimants or max_phone >= min_phone_share:
            rings.append(
                {"claimants": sorted(c[2:] for c in claimants), "size": len(claimants), "max_phone_share": max_phone}
            )
    return sorted(rings, key=lambda r: -r["size"])


def ego_graph(df, claim: dict, max_claims: int = 24) -> dict:
    """Nodes and edges around one claim for the UI: claims, claimants, phones, addresses, and shops."""
    phone_cols = [claim["phone"]] + ([tp_phone(claim)] if tp_phone(claim) else [])
    frontier_claimants = {claim["claimant_id"]}
    phones, addrs = set(phone_cols), {claim["address_id"]}
    for _ in range(2):  # two hops through shared phones and addresses
        m = (
            df["claimant_id"].isin(frontier_claimants)
            | df["phone"].isin(phones)
            | df["address_id"].isin(addrs)
            | df["third_party_phone"].isin(phones)
        )
        sub = df[m]
        frontier_claimants |= set(sub["claimant_id"])
        phones |= set(sub["phone"]) | set(sub["third_party_phone"].dropna())
        addrs |= set(sub["address_id"])
    sub = df[df["claimant_id"].isin(frontier_claimants)].sort_values("report_date").tail(max_claims)
    if claim["claim_id"] not in set(sub["claim_id"]):
        sub = sub.iloc[1:] if len(sub) >= max_claims else sub
    nodes, edges = {}, []

    def node(nid, kind, label):
        nodes.setdefault(nid, {"id": nid, "kind": kind, "label": label})

    rows = list(sub.to_dict("records"))
    if claim["claim_id"] not in {r["claim_id"] for r in rows}:
        rows.append(claim)
    for r in rows:
        node(r["claim_id"], "claim", r["claim_id"])
        node(r["claimant_id"], "claimant", r.get("claimant_name") or r["claimant_id"])
        node("P:" + r["phone"], "phone", r["phone"])
        node("A:" + r["address_id"], "address", r["address_id"])
        node("S:" + r["shop_id"], "shop", r["shop_id"])
        edges += [
            (r["claim_id"], r["claimant_id"], "filed"),
            (r["claimant_id"], "P:" + r["phone"], "phone"),
            (r["claimant_id"], "A:" + r["address_id"], "address"),
            (r["claim_id"], "S:" + r["shop_id"], "shop"),
        ]
        tp = r.get("third_party_phone")
        if isinstance(tp, str) and tp:
            node("P:" + tp, "phone", tp)
            edges.append((r["claim_id"], "P:" + tp, "third party"))
    uniq = sorted({(a, b, k) for a, b, k in edges})
    return {
        "center": claim["claim_id"],
        "nodes": list(nodes.values()),
        "edges": [{"source": a, "target": b, "kind": k} for a, b, k in uniq],
    }
