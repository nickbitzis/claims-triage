"""Estimate review against the regional price book.

Each line item is compared to the book: parts to list price times the regional parts multiplier, labor to
standard hours for the operation times the regional hourly rate. A line is flagged when it is more than the
tolerance above book, when the operation is not one this peril causes, or when it duplicates another line.
Flagged lines are adjusted down to book (or removed), which gives the adjusted estimate.
"""

from __future__ import annotations

from .catalog import OPERATIONS, REGIONS


def book_line(line_item: dict, region: str, line: str) -> dict:
    op = OPERATIONS[line_item["code"]]
    reg = REGIONS[region]
    if line_item["kind"] == "part":
        unit = op.part_price * reg.parts_mult
        return {
            "book_qty": float(line_item["units"]),
            "book_unit": round(unit, 2),
            "book_amount": round(unit * line_item["units"], 2),
        }
    rate = reg.labor_auto if line == "auto" else reg.labor_home
    hours = op.std_hours * line_item["units"]
    return {"book_qty": round(hours, 2), "book_unit": rate, "book_amount": round(hours * rate, 2)}


def review_estimate(lines: list[dict], region: str, line: str, peril: str, cause: str, tol: dict) -> dict:
    part_tol = float(tol.get("part", 0.12))
    labor_tol = float(tol.get("labor", 0.15))
    est_peril = "wind_hail" if (line == "home" and cause == "hail") else peril
    out = []
    seen: set[tuple[str, str]] = set()
    for li in lines:
        op = OPERATIONS.get(li["code"])
        row = dict(li)
        if op is None or op.line != line:
            row.update(
                book_amount=None,
                book_unit=None,
                book_qty=None,
                flagged=True,
                reason="Operation not in the price book.",
                adjusted_amount=0.0,
            )
            out.append(row)
            continue
        b = book_line(li, region, line)
        row.update(b)
        key = (li["code"], li["kind"])
        reasons = []
        adjusted = li["amount"]
        if key in seen:
            reasons.append("Duplicate of an earlier line.")
            adjusted = 0.0
        elif est_peril not in op.perils:
            reasons.append(f"Operation not expected for a {est_peril.replace('_', ' ')} loss.")
            adjusted = 0.0
        else:
            t = part_tol if li["kind"] == "part" else labor_tol
            if li["amount"] > b["book_amount"] * (1 + t):
                if li["kind"] == "part":
                    reasons.append(
                        f"Unit price ${li['unit_price']:,.2f} is {li['unit_price'] / b['book_unit'] - 1:.0%} "
                        f"above book ${b['book_unit']:,.2f}."
                    )
                else:
                    if li["unit_price"] > b["book_unit"] * (1 + t):
                        reasons.append(
                            f"Labor rate ${li['unit_price']:,.2f}/h above the {region} rate ${b['book_unit']:,.2f}/h."
                        )
                    if li["qty"] > b["book_qty"] * (1 + t):
                        reasons.append(f"{li['qty']:.1f} hours vs {b['book_qty']:.1f} standard hours.")
                    if not reasons:
                        reasons.append("Labor total above book.")
                adjusted = b["book_amount"]
        seen.add(key)
        row.update(flagged=bool(reasons), reason=" ".join(reasons), adjusted_amount=round(adjusted, 2))
        out.append(row)
    claimed = round(sum(x["amount"] for x in lines), 2)
    adjusted_total = round(sum(x["adjusted_amount"] for x in out), 2)
    n_flag = sum(1 for x in out if x["flagged"])
    return {
        "lines": out,
        "claimed_total": claimed,
        "adjusted_total": adjusted_total,
        "reduction": round(claimed - adjusted_total, 2),
        "variance": round(claimed / adjusted_total - 1, 4) if adjusted_total > 0 else (0.0 if claimed == 0 else 9.99),
        "flagged_lines": n_flag,
        "unexpected_ops": sum(1 for x in out if "not expected" in x["reason"] or "Duplicate" in x["reason"]),
        "adjusted_jewelry": round(sum(x["adjusted_amount"] for x in out if x.get("category") == "jewelry"), 2),
    }
