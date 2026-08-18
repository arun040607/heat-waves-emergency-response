import pandas as pd

from ml.predict import predict_zones


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/zones.csv"

# Available emergency resources
RESOURCES = {
    "cooling_centre": 2,
    "water_tanker": 2,
    "medical_team": 2,
}


# ============================================================
# ALLOCATION CONFIGURATION
# ============================================================

# Maximum benefit obtained from matching a resource to the
# specific stress it is designed to address.
RESOURCE_NEED_WEIGHT = 0.20

# Encourage coverage of different zones before repeatedly
# allocating to the same zone.
ZONE_DIVERSITY_WEIGHT = 0.12

# Stronger priority for severe ML classifications.
CRITICAL_BONUS = 10.0
HIGH_BONUS = 4.0

# Diminishing-return factor for multiple interventions in
# the same zone.
#
# First intervention: 100% of candidate value
# Second:           55%
# Third:            30%
# Fourth+:          progressively smaller
REPEAT_VALUE_FACTORS = {
    0: 1.00,
    1: 0.55,
    2: 0.30,
    3: 0.15,
}

# A repeated intervention is allowed when it is substantially
# more useful than the best uncovered-zone alternative.
REPEAT_ZONE_ADVANTAGE = 6.0

# Minimum resource-specific need below which an intervention
# receives a substantial reduction in priority.
LOW_NEED_THRESHOLD = 25.0

# Prevent LOW-risk zones from receiving resources while
# meaningful HIGH/CRITICAL alternatives remain.
LOW_RISK_BLOCK_SCORE = 75.0

# Number of top alternatives retained for audit information.
TOP_ALTERNATIVES_TO_STORE = 3


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n======================================")
print("       HEATWAVE AI RESPONSE ENGINE")
print("======================================\n")

print(f"Loaded {len(df)} urban zones.")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "zone",
    "population",
    "temperature",
    "vegetation",
    "building_density",
    "vulnerability",
    "cooling_access",
    "water_access",
    "hospital_capacity",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns in zones.csv: "
        + ", ".join(missing_columns)
    )

if df["zone"].duplicated().any():
    duplicated_zones = (
        df.loc[df["zone"].duplicated(), "zone"]
        .astype(str)
        .tolist()
    )

    raise ValueError(
        "Duplicate zone IDs found: "
        + ", ".join(duplicated_zones)
    )


# ============================================================
# 2. POPULATION EXPOSURE
# ============================================================

min_population = df["population"].min()
max_population = df["population"].max()

if max_population == min_population:

    df["exposure_score"] = 50.0

else:

    df["exposure_score"] = (
        (df["population"] - min_population)
        / (max_population - min_population)
    ) * 100


# ============================================================
# 3. NORMALIZE ENVIRONMENTAL FACTORS
# ============================================================

# ------------------------------------------------------------
# Temperature
# ------------------------------------------------------------

min_temp = df["temperature"].min()
max_temp = df["temperature"].max()

if max_temp == min_temp:

    df["temperature_score"] = 50.0

else:

    df["temperature_score"] = (
        (df["temperature"] - min_temp)
        / (max_temp - min_temp)
    ) * 100


# ------------------------------------------------------------
# Vegetation
# ------------------------------------------------------------
# Low vegetation = higher heat contribution
# ------------------------------------------------------------

min_vegetation = df["vegetation"].min()
max_vegetation = df["vegetation"].max()

if max_vegetation == min_vegetation:

    df["vegetation_heat_score"] = 50.0

else:

    df["vegetation_heat_score"] = (
        (max_vegetation - df["vegetation"])
        / (max_vegetation - min_vegetation)
    ) * 100


# ------------------------------------------------------------
# Building density
# ------------------------------------------------------------
# High density = higher heat contribution
# ------------------------------------------------------------

min_density = df["building_density"].min()
max_density = df["building_density"].max()

if max_density == min_density:

    df["building_heat_score"] = 50.0

else:

    df["building_heat_score"] = (
        (df["building_density"] - min_density)
        / (max_density - min_density)
    ) * 100


# ============================================================
# 4. ENVIRONMENTAL HEAT DRIVER
# ============================================================

df["heat_driver_score"] = (
    0.50 * df["temperature_score"]
    + 0.25 * df["vegetation_heat_score"]
    + 0.25 * df["building_heat_score"]
)


# ============================================================
# 5. RANDOM FOREST AI RISK
# ============================================================

print("\nRunning Random Forest AI risk assessment...")

ml_results = predict_zones(df)


# ============================================================
# 6. STORE AI RESULTS
# ============================================================

df["risk_class"] = [
    result["risk_class"]
    for result in ml_results
]

df["risk_label"] = [
    result["risk_label"]
    for result in ml_results
]

df["ml_confidence"] = [
    result["confidence"]
    for result in ml_results
]


# ============================================================
# 7. CONTINUOUS AI RISK
# ============================================================

df["ml_risk"] = [
    100 * (
        result["probabilities"].get("HIGH", 0)
        + result["probabilities"].get("CRITICAL", 0)
    )
    for result in ml_results
]


# ============================================================
# 8. SERVICE STRESS
# ============================================================

df["cooling_stress"] = (
    100 - df["cooling_access"]
)

df["water_stress"] = (
    100 - df["water_access"]
)

df["healthcare_stress"] = (
    100 - df["hospital_capacity"]
)


# ============================================================
# 9. COMPOUND RISK
# ============================================================
#
# AI risk          = 40%
# Heat drivers     = 20%
# Vulnerability    = 20%
# Cooling stress   = 10%
# Water stress     = 5%
# Healthcare       = 5%
# ============================================================

df["compound_risk"] = (
    0.40 * df["ml_risk"]
    + 0.20 * df["heat_driver_score"]
    + 0.20 * df["vulnerability"]
    + 0.10 * df["cooling_stress"]
    + 0.05 * df["water_stress"]
    + 0.05 * df["healthcare_stress"]
)


# ============================================================
# 10. INTERVENTION IMPACT
# ============================================================

# ------------------------------------------------------------
# Cooling centre
# ------------------------------------------------------------

df["cooling_impact"] = (
    0.50 * df["compound_risk"]
    + 0.30 * df["cooling_stress"]
    + 0.20 * df["vulnerability"]
)


# ------------------------------------------------------------
# Water tanker
# ------------------------------------------------------------

df["water_impact"] = (
    0.50 * df["compound_risk"]
    + 0.30 * df["water_stress"]
    + 0.20 * df["exposure_score"]
)


# ------------------------------------------------------------
# Medical team
# ------------------------------------------------------------

df["medical_impact"] = (
    0.50 * df["compound_risk"]
    + 0.30 * df["healthcare_stress"]
    + 0.20 * df["vulnerability"]
)


# ============================================================
# 11. EQUITY SCORE
# ============================================================

df["equity_score"] = (
    0.50 * df["vulnerability"]
    + 0.25 * df["cooling_stress"]
    + 0.25 * df["water_stress"]
)


# ============================================================
# 12. CREATE INTERVENTION CANDIDATES
# ============================================================

candidates = []

for _, row in df.iterrows():

    # --------------------------------------------------------
    # Cooling centre
    # --------------------------------------------------------

    candidates.append({
        "zone": row["zone"],
        "intervention": "cooling_centre",
        "impact": float(row["cooling_impact"]),
        "equity": float(row["equity_score"]),
        "risk_class": int(row["risk_class"]),
        "risk_label": str(row["risk_label"]),
        "confidence": float(row["ml_confidence"]),
        "ml_risk": float(row["ml_risk"]),
        "compound_risk": float(row["compound_risk"]),
        "cooling_stress": float(row["cooling_stress"]),
        "water_stress": float(row["water_stress"]),
        "healthcare_stress": float(row["healthcare_stress"]),
        "exposure_score": float(row["exposure_score"]),
        "vulnerability": float(row["vulnerability"]),
        "heat_driver_score": float(row["heat_driver_score"]),
    })


    # --------------------------------------------------------
    # Water tanker
    # --------------------------------------------------------

    candidates.append({
        "zone": row["zone"],
        "intervention": "water_tanker",
        "impact": float(row["water_impact"]),
        "equity": float(row["equity_score"]),
        "risk_class": int(row["risk_class"]),
        "risk_label": str(row["risk_label"]),
        "confidence": float(row["ml_confidence"]),
        "ml_risk": float(row["ml_risk"]),
        "compound_risk": float(row["compound_risk"]),
        "cooling_stress": float(row["cooling_stress"]),
        "water_stress": float(row["water_stress"]),
        "healthcare_stress": float(row["healthcare_stress"]),
        "exposure_score": float(row["exposure_score"]),
        "vulnerability": float(row["vulnerability"]),
        "heat_driver_score": float(row["heat_driver_score"]),
    })


    # --------------------------------------------------------
    # Medical team
    # --------------------------------------------------------

    candidates.append({
        "zone": row["zone"],
        "intervention": "medical_team",
        "impact": float(row["medical_impact"]),
        "equity": float(row["equity_score"]),
        "risk_class": int(row["risk_class"]),
        "risk_label": str(row["risk_label"]),
        "confidence": float(row["ml_confidence"]),
        "ml_risk": float(row["ml_risk"]),
        "compound_risk": float(row["compound_risk"]),
        "cooling_stress": float(row["cooling_stress"]),
        "water_stress": float(row["water_stress"]),
        "healthcare_stress": float(row["healthcare_stress"]),
        "exposure_score": float(row["exposure_score"]),
        "vulnerability": float(row["vulnerability"]),
        "heat_driver_score": float(row["heat_driver_score"]),
    })


# ============================================================
# 13. BASE DECISION SCORE
# ============================================================
#
# Intervention impact = 70%
# Equity              = 30%
# ============================================================

for candidate in candidates:

    candidate["base_score"] = (
        0.70 * candidate["impact"]
        + 0.30 * candidate["equity"]
    )


# ============================================================
# 14. RESOURCE-AWARE ALLOCATION
# ============================================================
#
# Improvements over a simple global sort:
#
# 1. RESOURCE-TO-NEED MATCHING
#
#    cooling_centre -> cooling stress
#    water_tanker   -> water stress
#    medical_team   -> healthcare stress
#
# 2. DYNAMIC ZONE COVERAGE
#
#    A zone receiving one intervention has diminishing
#    priority for another intervention.
#
# 3. SEVERE-RISK PRIORITY
#
#    CRITICAL and HIGH zones receive additional priority.
#
# 4. REPEATED INTERVENTIONS ARE NOT FORBIDDEN
#
#    A CRITICAL zone can still receive multiple resources
#    when the additional intervention provides substantial
#    value.
#
# 5. RESOURCE BALANCING
#
#    The algorithm considers each resource independently
#    while allocating globally.
# ============================================================


remaining_resources = RESOURCES.copy()

allocated = []

zone_allocations = {}

resource_allocations = {
    resource: 0
    for resource in RESOURCES
}


# ------------------------------------------------------------
# Resource-specific need
# ------------------------------------------------------------

def resource_need(candidate):
    """
    Return how strongly the selected intervention addresses
    the dominant stress of this zone.
    """

    intervention = candidate["intervention"]

    if intervention == "cooling_centre":

        # Cooling centres are especially valuable when:
        # - cooling access is poor
        # - temperature/heat exposure is high
        return (
            0.60 * candidate["cooling_stress"]
            + 0.40 * candidate["heat_driver_score"]
        )

    if intervention == "water_tanker":

        # Water tankers are especially valuable when:
        # - water access is poor
        # - exposed population is large
        return (
            0.70 * candidate["water_stress"]
            + 0.30 * candidate["exposure_score"]
        )

    if intervention == "medical_team":

        # Medical teams are especially valuable when:
        # - hospital capacity is constrained
        # - vulnerability is high
        return (
            0.70 * candidate["healthcare_stress"]
            + 0.30 * candidate["vulnerability"]
        )

    return 0.0


# ------------------------------------------------------------
# Risk bonus
# ------------------------------------------------------------

def risk_bonus(candidate):
    """
    Additional priority based on ML risk class.
    """

    label = str(
        candidate["risk_label"]
    ).upper()

    if label == "CRITICAL":
        return CRITICAL_BONUS

    if label == "HIGH":
        return HIGH_BONUS

    return 0.0


# ------------------------------------------------------------
# Repeat-value factor
# ------------------------------------------------------------

def repeat_value_factor(zone):
    """
    Diminishing return for repeated interventions in a zone.
    """

    count = zone_allocations.get(zone, 0)

    if count in REPEAT_VALUE_FACTORS:
        return REPEAT_VALUE_FACTORS[count]

    return 0.08


# ------------------------------------------------------------
# Zone coverage bonus
# ------------------------------------------------------------

def coverage_bonus(zone):
    """
    Prefer zones that have not received resources yet.
    """

    count = zone_allocations.get(zone, 0)

    if count == 0:
        return 8.0

    if count == 1:
        return 0.0

    return -4.0 * (count - 1)


# ------------------------------------------------------------
# Need mismatch penalty
# ------------------------------------------------------------

def need_adjustment(candidate):
    """
    Reward an intervention when it directly addresses the
    zone's dominant service stress.

    This prevents a resource from being allocated merely
    because the overall risk is high.
    """

    need = resource_need(candidate)

    if need >= 75:
        return 8.0

    if need >= 60:
        return 5.0

    if need >= 45:
        return 2.0

    if need >= LOW_NEED_THRESHOLD:
        return 0.0

    return -8.0


# ------------------------------------------------------------
# Dynamic effective score
# ------------------------------------------------------------

def effective_score(candidate):
    """
    Calculate the complete dynamic allocation score.
    """

    zone = candidate["zone"]

    base_score = candidate["base_score"]

    need = resource_need(candidate)

    repeat_factor = repeat_value_factor(zone)

    # Diminishing return is applied primarily to the
    # intervention's impact/equity value.
    adjusted_base = (
        base_score * repeat_factor
    )

    # Resource-to-need matching.
    resource_match = (
        RESOURCE_NEED_WEIGHT * need
    )

    # Severe risk remains important even when a zone already
    # received another resource.
    severe_risk_bonus = risk_bonus(candidate)

    # Encourage geographic/zone coverage.
    diversity = (
        ZONE_DIVERSITY_WEIGHT
        * coverage_bonus(zone)
    )

    # Additional intervention-specific need adjustment.
    need_bonus = need_adjustment(candidate)

    score = (
        adjusted_base
        + resource_match
        + severe_risk_bonus
        + diversity
        + need_bonus
    )

    return float(score)


# ------------------------------------------------------------
# Create candidates that are currently feasible
# ------------------------------------------------------------

def build_available_candidates():
    """
    Return all candidates whose resource is still available.
    """

    available = []

    for candidate in candidates:

        intervention = candidate["intervention"]

        if remaining_resources.get(
            intervention,
            0
        ) <= 0:
            continue

        candidate_copy = candidate.copy()

        candidate_copy["resource_need"] = (
            resource_need(candidate)
        )

        candidate_copy["effective_score"] = (
            effective_score(candidate)
        )

        candidate_copy["repeat_factor"] = (
            repeat_value_factor(
                candidate["zone"]
            )
        )

        candidate_copy["coverage_bonus"] = (
            coverage_bonus(
                candidate["zone"]
            )
        )

        available.append(candidate_copy)

    return available


# ------------------------------------------------------------
# Check whether repeated allocation is justified
# ------------------------------------------------------------

def should_allow_repeat(
    selected,
    best_uncovered
):
    """
    Decide whether a candidate that already has resources
    should beat an uncovered zone.

    Critical zones get more flexibility.

    Otherwise, repeated allocation needs to beat the best
    uncovered-zone candidate by REPEAT_ZONE_ADVANTAGE.
    """

    zone = selected["zone"]

    current_count = zone_allocations.get(
        zone,
        0
    )

    if current_count == 0:
        return True

    if (
        str(selected["risk_label"]).upper()
        == "CRITICAL"
    ):

        # Critical zones can receive a second intervention
        # when it directly addresses a strong need.
        if (
            selected["resource_need"]
            >= 60
        ):
            return True

    if best_uncovered is None:
        return True

    return (
        selected["effective_score"]
        >= (
            best_uncovered["effective_score"]
            + REPEAT_ZONE_ADVANTAGE
        )
    )


# ------------------------------------------------------------
# Select the next allocation
# ------------------------------------------------------------

def select_next_candidate(available):
    """
    Select the best feasible candidate while explicitly
    balancing zone coverage against repeated intervention.
    """

    if not available:
        return None

    # --------------------------------------------------------
    # Best candidate overall
    # --------------------------------------------------------

    available_sorted = sorted(
        available,
        key=lambda x: (
            x["effective_score"],
            x["compound_risk"],
            x["resource_need"],
            x["equity"],
        ),
        reverse=True,
    )

    best_overall = available_sorted[0]

    # --------------------------------------------------------
    # Best candidate from a previously uncovered zone
    # --------------------------------------------------------

    uncovered = [
        candidate
        for candidate in available
        if zone_allocations.get(
            candidate["zone"],
            0
        ) == 0
    ]

    if not uncovered:
        return best_overall

    uncovered.sort(
        key=lambda x: (
            x["effective_score"],
            x["compound_risk"],
            x["resource_need"],
            x["equity"],
        ),
        reverse=True,
    )

    best_uncovered = uncovered[0]

    # --------------------------------------------------------
    # If the best overall candidate is already covered,
    # require meaningful additional value before repeating.
    # --------------------------------------------------------

    if (
        zone_allocations.get(
            best_overall["zone"],
            0
        ) > 0
    ):

        if should_allow_repeat(
            best_overall,
            best_uncovered,
        ):
            return best_overall

        return best_uncovered

    return best_overall


# ------------------------------------------------------------
# Greedy allocation
# ------------------------------------------------------------

total_resources = sum(
    RESOURCES.values()
)

for allocation_number in range(
    total_resources
):

    available_candidates = (
        build_available_candidates()
    )

    if not available_candidates:
        break

    selected = select_next_candidate(
        available_candidates
    )

    if selected is None:
        break

    zone = selected["zone"]
    intervention = selected["intervention"]

    # --------------------------------------------------------
    # Store audit information
    # --------------------------------------------------------

    alternatives = sorted(
        available_candidates,
        key=lambda x: (
            x["effective_score"],
            x["compound_risk"],
            x["resource_need"],
        ),
        reverse=True,
    )

    selected["allocation_number"] = (
        allocation_number + 1
    )

    selected["zone_previous_allocations"] = (
        zone_allocations.get(zone, 0)
    )

    selected["allocation_reason"] = (
        f"{intervention} selected because "
        f"resource need={selected['resource_need']:.2f}, "
        f"compound risk={selected['compound_risk']:.2f}, "
        f"equity={selected['equity']:.2f}, "
        f"and dynamic score="
        f"{selected['effective_score']:.2f}"
    )

    selected["top_alternatives"] = [
        (
            item["zone"],
            item["intervention"],
            round(
                item["effective_score"],
                2
            ),
        )
        for item in alternatives[
            :TOP_ALTERNATIVES_TO_STORE
        ]
    ]

    # --------------------------------------------------------
    # Allocate
    # --------------------------------------------------------

    allocated.append(selected)

    remaining_resources[
        intervention
    ] -= 1

    zone_allocations[zone] = (
        zone_allocations.get(zone, 0)
        + 1
    )

    resource_allocations[
        intervention
    ] += 1


# ============================================================
# 15. ZONE RISK ASSESSMENT
# ============================================================

print("\n======================================")
print("          ZONE RISK ASSESSMENT")
print("======================================\n")

risk_columns = [
    "zone",
    "temperature",
    "vegetation",
    "building_density",
    "risk_label",
    "ml_confidence",
    "ml_risk",
    "vulnerability",
    "exposure_score",
    "cooling_stress",
    "water_stress",
    "healthcare_stress",
    "compound_risk",
]

display_df = df[
    risk_columns
].copy()

display_df["ml_confidence"] = (
    display_df["ml_confidence"] * 100
)

print(
    display_df
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 16. RISK DISTRIBUTION
# ============================================================

print("\n======================================")
print("          RISK DISTRIBUTION")
print("======================================\n")

risk_distribution = (
    df["risk_label"]
    .value_counts()
)

total_zones = len(df)

for label in [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]:

    count = int(
        risk_distribution.get(
            label,
            0
        )
    )

    percentage = (
        count / total_zones * 100
        if total_zones > 0
        else 0
    )

    print(
        f"{label:<10}: "
        f"{count:>4} zones "
        f"({percentage:>6.2f}%)"
    )


# ============================================================
# 17. INTERVENTION IMPACT
# ============================================================

print("\n======================================")
print("       INTERVENTION IMPACT")
print("======================================\n")

impact_columns = [
    "zone",
    "risk_label",
    "compound_risk",
    "cooling_impact",
    "water_impact",
    "medical_impact",
    "equity_score",
]

print(
    df[
        impact_columns
    ]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 18. AI RISK EXPLANATIONS
# ============================================================

print("\n======================================")
print("          AI RISK EXPLANATIONS")
print("======================================\n")

for _, row in df.iterrows():

    factors = []

    if row["temperature"] >= 42:

        factors.append(
            "very high temperature"
        )

    if row["vegetation"] <= 20:

        factors.append(
            "low vegetation"
        )

    if row["building_density"] >= 80:

        factors.append(
            "high building density"
        )

    if row["vulnerability"] >= 80:

        factors.append(
            "high vulnerability"
        )

    if row["cooling_access"] <= 30:

        factors.append(
            "limited cooling access"
        )

    if row["water_access"] <= 30:

        factors.append(
            "limited water access"
        )

    if row["hospital_capacity"] <= 50:

        factors.append(
            "limited hospital capacity"
        )

    if not factors:

        factors.append(
            "no dominant threshold-based stress factor"
        )

    print(
        f"Zone {row['zone']} → "
        f"{row['risk_label']} "
        f"({row['ml_confidence']:.2%} confidence)"
    )

    print(
        "    Factors: "
        + ", ".join(factors)
    )


# ============================================================
# 19. EMERGENCY RESPONSE PLAN
# ============================================================

print("\n======================================")
print("       HEATWAVE RESPONSE PLAN")
print("======================================\n")

if not allocated:

    print(
        "No interventions could be allocated."
    )

else:

    for priority, item in enumerate(
        allocated,
        start=1,
    ):

        print(
            f"Priority {priority}: "
            f"Zone {item['zone']} → "
            f"{item['intervention']}"
        )

        print(
            f"    Risk: "
            f"{item['risk_label']}"
        )

        print(
            f"    AI Confidence: "
            f"{item['confidence']:.2%}"
        )

        print(
            f"    ML Risk: "
            f"{item['ml_risk']:.2f}"
        )

        print(
            f"    Compound Risk: "
            f"{item['compound_risk']:.2f}"
        )

        print(
            f"    Resource Need: "
            f"{item['resource_need']:.2f}"
        )

        print(
            f"    Impact: "
            f"{item['impact']:.2f}"
        )

        print(
            f"    Equity: "
            f"{item['equity']:.2f}"
        )

        print(
            f"    Base Score: "
            f"{item['base_score']:.2f}"
        )

        print(
            f"    Repeat Value Factor: "
            f"{item['repeat_factor']:.2f}"
        )

        print(
            f"    Effective Score: "
            f"{item['effective_score']:.2f}"
        )

        print(
            f"    Previous Zone Allocations: "
            f"{item['zone_previous_allocations']}"
        )

        print(
            f"    Reason: "
            f"{item['allocation_reason']}"
        )

        print()


# ============================================================
# 20. ZONE COVERAGE SUMMARY
# ============================================================

print("======================================")
print("          ZONE COVERAGE")
print("======================================\n")

zones_with_resources = len(
    zone_allocations
)

print(
    f"Zones receiving at least one "
    f"intervention: "
    f"{zones_with_resources}"
)

print(
    f"Total interventions allocated: "
    f"{len(allocated)}"
)

print()

for zone, count in sorted(
    zone_allocations.items(),
    key=lambda x: (-x[1], x[0]),
):

    print(
        f"{zone}: "
        f"{count} intervention(s)"
    )


# ============================================================
# 21. RESOURCE UTILIZATION
# ============================================================

print("\n======================================")
print("       RESOURCE UTILIZATION")
print("======================================\n")

for resource, original_quantity in (
    RESOURCES.items()
):

    used = resource_allocations[
        resource
    ]

    remaining = remaining_resources[
        resource
    ]

    utilization = (
        used / original_quantity * 100
        if original_quantity > 0
        else 0
    )

    print(
        f"{resource}: "
        f"{used}/{original_quantity} used "
        f"({utilization:.1f}%)"
    )

    print(
        f"    Remaining: {remaining}"
    )


# ============================================================
# 22. REMAINING RESOURCES
# ============================================================

print("\n======================================")
print("       REMAINING RESOURCES")
print("======================================\n")

for resource, quantity in (
    remaining_resources.items()
):

    print(
        f"{resource}: {quantity}"
    )


# ============================================================
# 23. ALLOCATION BY RESOURCE
# ============================================================

print("\n======================================")
print("       ALLOCATION BY RESOURCE")
print("======================================\n")

for resource in RESOURCES:

    resource_items = [
        item
        for item in allocated
        if item["intervention"]
        == resource
    ]

    print(
        f"{resource}:"
    )

    if not resource_items:

        print(
            "    No allocation"
        )

        continue

    for item in resource_items:

        print(
            f"    Zone {item['zone']} "
            f"→ score "
            f"{item['effective_score']:.2f}, "
            f"need "
            f"{item['resource_need']:.2f}"
        )


# ============================================================
# 24. TOP PRIORITY ZONES
# ============================================================

print("\n======================================")
print("        TOP PRIORITY ZONES")
print("======================================\n")

top_zones = (
    df[
        [
            "zone",
            "risk_label",
            "ml_confidence",
            "ml_risk",
            "compound_risk",
            "equity_score",
        ]
    ]
    .sort_values(
        by=[
            "compound_risk",
            "equity_score",
        ],
        ascending=False,
    )
    .head(10)
)

top_zones = top_zones.copy()

top_zones["ml_confidence"] = (
    top_zones["ml_confidence"] * 100
)

print(
    top_zones
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 25. ALLOCATION SUMMARY TABLE
# ============================================================

print("\n======================================")
print("       FINAL ALLOCATION TABLE")
print("======================================\n")

if allocated:

    allocation_table = pd.DataFrame(
        [
            {
                "priority": index + 1,
                "zone": item["zone"],
                "intervention": item[
                    "intervention"
                ],
                "risk": item[
                    "risk_label"
                ],
                "confidence": (
                    item["confidence"] * 100
                ),
                "ml_risk": item[
                    "ml_risk"
                ],
                "compound_risk": item[
                    "compound_risk"
                ],
                "resource_need": item[
                    "resource_need"
                ],
                "impact": item[
                    "impact"
                ],
                "equity": item[
                    "equity"
                ],
                "base_score": item[
                    "base_score"
                ],
                "effective_score": item[
                    "effective_score"
                ],
            }
            for index, item
            in enumerate(allocated)
        ]
    )

    print(
        allocation_table
        .round(2)
        .to_string(index=False)
    )

else:

    print(
        "No allocation records available."
    )


# ============================================================
# 26. HUMAN APPROVAL NOTICE
# ============================================================

print("\n======================================")
print("        HUMAN APPROVAL REQUIRED")
print("======================================\n")

print(
    "The generated response plan is AI-assisted."
)

print(
    "Officials must review and approve "
    "interventions before deployment."
)

print(
    "AI recommendations must not be "
    "automatically deployed without "
    "official validation."
)


# ============================================================
# 27. COMPLETE
# ============================================================

print("\n======================================")
print("              COMPLETE")
print("======================================\n")