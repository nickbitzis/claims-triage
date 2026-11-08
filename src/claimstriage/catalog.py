"""Reference data for Bramblewood Mutual: regions, the price book, perils, and coverage terms.

The price book is synthetic. Part prices are national list prices scaled by a regional parts multiplier, and
labor is standard hours per unit of work times the regional hourly rate for the trade (auto body or home repair).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    name: str
    lat: float
    lon: float
    labor_auto: float  # $/hour body shop
    labor_home: float  # $/hour home trades
    parts_mult: float
    weight: float
    pct_wind_deductible: float  # home wind/hail deductible as a share of the dwelling limit (0 = flat deductible)


REGIONS: dict[str, Region] = {
    r.name: r
    for r in [
        Region("North", 44.9, -93.2, 62.0, 68.0, 1.00, 0.17, 0.0),
        Region("Lakes", 41.9, -87.7, 66.0, 72.0, 1.03, 0.20, 0.0),
        Region("Plains", 39.1, -96.6, 54.0, 58.0, 0.96, 0.16, 0.01),
        Region("Gulf", 29.8, -95.4, 56.0, 60.0, 0.98, 0.17, 0.0),
        Region("Mountain", 39.7, -105.0, 64.0, 70.0, 1.02, 0.12, 0.01),
        Region("Coast", 37.6, -122.2, 82.0, 90.0, 1.10, 0.18, 0.0),
    ]
}


@dataclass(frozen=True)
class Operation:
    code: str
    line: str  # auto or home
    desc: str
    part_price: float  # national list price per unit, 0 if labor only
    std_hours: float  # standard labor hours per unit, 0 if no labor (contents)
    qty: tuple[int, int]  # typical quantity range on one claim
    perils: tuple[str, ...]
    category: str = "repair"  # repair or jewelry (sublimit)


OPERATIONS: dict[str, Operation] = {
    o.code: o
    for o in [
        Operation("A-BUMPER-F", "auto", "Front bumper cover", 450, 2.5, (1, 1), ("collision", "vandalism")),
        Operation("A-BUMPER-R", "auto", "Rear bumper cover", 420, 2.5, (1, 1), ("collision",)),
        Operation("A-HEADLAMP", "auto", "Headlamp assembly", 380, 0.8, (1, 2), ("collision",)),
        Operation("A-HOOD", "auto", "Hood panel", 520, 2.0, (1, 1), ("collision", "hail")),
        Operation("A-FENDER", "auto", "Fender", 310, 2.2, (1, 2), ("collision",)),
        Operation("A-DOOR", "auto", "Door shell", 640, 3.0, (1, 2), ("collision", "vandalism")),
        Operation("A-QPANEL", "auto", "Quarter panel repair", 0, 6.0, (1, 1), ("collision",)),
        Operation(
            "A-REFINISH",
            "auto",
            "Refinish per panel, paint and materials",
            95,
            2.0,
            (1, 4),
            ("collision", "hail", "vandalism"),
        ),
        Operation("A-FRAME", "auto", "Frame straightening", 0, 5.0, (1, 1), ("collision",)),
        Operation("A-AIRBAG", "auto", "Airbag module", 950, 1.5, (1, 2), ("collision",)),
        Operation("A-PDR", "auto", "Paintless dent repair per panel", 0, 2.5, (2, 7), ("hail",)),
        Operation("A-WINDSHIELD", "auto", "Windshield", 420, 1.5, (1, 1), ("glass", "hail", "collision", "vandalism")),
        Operation("A-SIDEGLASS", "auto", "Door glass", 210, 1.0, (1, 2), ("glass", "vandalism", "theft")),
        Operation("A-CATCONV", "auto", "Catalytic converter", 1350, 2.0, (1, 1), ("theft",)),
        Operation("A-LOCKSET", "auto", "Ignition and lock set", 340, 1.5, (1, 1), ("theft",)),
        Operation("A-ROOF", "auto", "Roof panel replace", 780, 8.0, (1, 1), ("hail",)),
        Operation("A-INTERIOR", "auto", "Interior carpet and seats", 1600, 6.0, (1, 1), ("flood",)),
        Operation("A-ECU", "auto", "Engine control module", 1100, 1.5, (1, 1), ("flood", "collision")),
        Operation("H-SHINGLE", "home", "Roof shingles per square", 145, 1.3, (12, 32), ("wind_hail",)),
        Operation("H-UNDERLAY", "home", "Underlayment per square", 38, 0.3, (12, 32), ("wind_hail",)),
        Operation("H-GUTTER", "home", "Gutter per 10 ft", 65, 0.6, (3, 14), ("wind_hail",)),
        Operation("H-SIDING", "home", "Siding per 100 sq ft", 260, 2.5, (2, 8), ("wind_hail", "fire")),
        Operation("H-WINDOW", "home", "Window unit", 480, 1.8, (1, 4), ("wind_hail", "theft", "fire")),
        Operation(
            "H-DRYWALL", "home", "Drywall per sheet, hung and finished", 22, 0.9, (8, 40), ("water", "flood", "fire")
        ),
        Operation("H-FLOOR", "home", "Flooring per 100 sq ft", 420, 3.0, (1, 6), ("water", "flood", "fire")),
        Operation("H-MITIGATION", "home", "Water mitigation per equipment day", 180, 0.5, (3, 12), ("water", "flood")),
        Operation("H-CABINET", "home", "Base cabinet per linear ft", 230, 0.8, (4, 16), ("water", "flood", "fire")),
        Operation("H-WATERHEATER", "home", "Water heater", 1150, 3.0, (1, 1), ("flood", "water")),
        Operation("H-HVAC", "home", "Furnace or air handler", 3900, 6.0, (1, 1), ("flood", "fire")),
        Operation("H-ELECTRICAL", "home", "Electrical circuit repair", 140, 2.0, (1, 4), ("fire", "flood")),
        Operation("H-SMOKE", "home", "Smoke and soot cleaning per room", 320, 4.0, (2, 6), ("fire",)),
        Operation("H-DOOR-EXT", "home", "Exterior door and lock", 620, 2.5, (1, 1), ("theft", "wind_hail")),
        Operation("H-TV", "home", "Television (contents)", 900, 0, (1, 2), ("theft", "fire")),
        Operation("H-LAPTOP", "home", "Laptop (contents)", 1100, 0, (1, 2), ("theft", "fire")),
        Operation("H-JEWELRY", "home", "Jewelry (contents)", 1200, 0, (1, 3), ("theft",), "jewelry"),
    ]
}

# peril -> (cause, weight) for base (non-event) claims
AUTO_PERILS = {"collision": 0.62, "glass": 0.15, "theft": 0.12, "vandalism": 0.11}
HOME_PERILS = {"water": 0.45, "fire": 0.10, "theft": 0.15, "wind_hail": 0.30}
CAUSES = {
    "collision": ["rear_end", "parking_lot", "intersection", "single_vehicle", "animal"],
    "glass": ["road_debris"],
    "theft": ["burglary"],
    "vandalism": ["vandalism"],
    "water": ["pipe_burst", "appliance", "gradual_leak", "unknown"],
    "fire": ["kitchen", "electrical"],
    "wind_hail": ["wind"],
    "hail": ["hail"],
    "flood": ["flood"],
}
WEATHER_CAUSES = ("hail", "flood")

# Coverage terms. Auto physical damage: collision needs collision coverage; everything else is comprehensive.
AUTO_COMPREHENSIVE_PERILS = ("glass", "theft", "vandalism", "hail", "flood")
HOME_COVERED_PERILS = ("water", "fire", "theft", "wind_hail")  # flood needs the flood endorsement
EXCLUDED_CAUSES = {"gradual_leak": "Wear, tear, and gradual leakage are excluded (HO form section I exclusions)."}
JEWELRY_SUBLIMIT = 1500.0
CONTENTS_SHARE = 0.5  # contents limit as a share of the dwelling limit

FIRST_NAMES = (
    "Ava Ben Cara Dev Eli Faye Gus Hana Ian Jada Kai Lena Milo Nia Omar Pia Quinn Rosa Sam Tess Umar Vera Wes "
    "Xena Yusuf Zoe Ada Bo Cyrus Dina Emil Flor Gil Hugo Iris Joel Kira Lars Maya Noel Otto Priya Reid Sana Theo"
).split()
LAST_NAMES = (
    "Abbott Barros Chen Dalton Ellis Fischer Grant Hale Ito Jensen Kaur Lowe Mendez Nakamura Okoro Patel Quist "
    "Rhodes Silva Tran Underwood Vance Walsh Xu Young Zeller Moreau Lindqvist Haddad Brennan Castillo Doyle"
).split()
STREETS = "Maple Cedar Ridge Lakeview Mill Station Harbor Orchard Union Willow Granite Prairie Elm Summit".split()
