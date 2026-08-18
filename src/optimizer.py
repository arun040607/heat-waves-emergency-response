import pandas as pd

from ml.predict import predict_zone


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/zones.csv"

# Available emergency resources for this scenario
RESOURCES = {
    "cooling_centre": 2,
    "water_tanker": 2,
    "medical_team": 2
}


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n======================================")
print("       HEATWAVE AI RESPONSE ENGINE")
print("======================================\n")

print(f"Loaded {len(df)} urban zones.")


# ============================================================
# 2. POPULATION EXPOSURE
# ============================================================

min_population = df["population"].min()
max_population = df["population"].max()

if max_population == min_population:
    df["exposure_score"] = 50
else:
    df["exposure_score"] = (
        (df["population"] - min_population)
        / (max_population - min_population)
    ) * 100


# ============================================================
# 3. NORMALIZE ENVIRONMENTAL FACTORS
# ============================================================

# -----------------------------
# Temperature
# -----------------------------

min_temp = df["temperature"].min()
max_temp = df["temperature"].max()

if max_temp == min_temp:
    df["temperature_score"] = 50
else:
    df["temperature_score"] = (
        (df["temperature"] - min_temp)
        / (max_temp - min_temp)
    ) * 100


# -----------------------------
# Vegetation
# -----------------------------
# Low vegetation = higher heat contribution
# -----------------------------

min_vegetation = df["vegetation"].min()
max_vegetation = df["vegetation"].max()

if max_vegetation == min_vegetation:
    df["vegetation_heat_score"] = 50
else:
    df["vegetation_heat_score"] = (
        (max_vegetation - df["vegetation"])
        / (max_vegetation - min_vegetation)
    ) * 100


# -----------------------------
# Building density
# -----------------------------
# High density = higher heat contribution
# -----------------------------

min_density = df["building_density"].min()
max_density = df["building_density"].max()

if max_density == min_density:
    df["building_heat_score"] = 50
else:
    df["building_heat_score"] = (
        (df["building_density"] - min_density)
        / (max_density - min_density)
    ) * 100


# ============================================================
# 4. ENVIRONMENTAL HEAT DRIVER
# ============================================================
#
# Temperature          = 50%
# Vegetation deficit   = 25%
# Building density     = 25%
#
# This is an interpretable factor.
# It is NOT the ML prediction.
# ============================================================

df["heat_driver_score"] = (
    0.50 * df["temperature_score"]
    + 0.25 * df["vegetation_heat_score"]
    + 0.25 * df["building_heat_score"]
)


# ============================================================
# 5. RANDOM FOREST AI RISK
# ============================================================
#
# Connects directly to Member 2's trained model.
#
# The model returns:
# - risk_class
# - risk_label
# - confidence
# - probabilities
# ============================================================

ml_results = []

for _, row in df.iterrows():

    zone_data = {
        "temperature": row["temperature"],
        "vegetation": row["vegetation"],
        "building_density": row["building_density"],
        "population": row["population"],
        "vulnerability": row["vulnerability"],
        "cooling_access": row["cooling_access"],
        "water_access": row["water_access"],
        "hospital_capacity": row["hospital_capacity"]
    }

    result = predict_zone(zone_data)

    ml_results.append(result)


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
#
# Instead of using:
#
# LOW = 0
# MEDIUM = 1
# HIGH = 2
# CRITICAL = 3
#
# use the probability of HIGH + CRITICAL.
#
# Example:
#
# HIGH = 15.8%
# CRITICAL = 81.49%
#
# ML risk = 97.29
#
# This gives the optimizer a continuous AI signal.
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
#
# This is the AI-INFLUENCED decision layer.
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

# -----------------------------
# Cooling centre
# -----------------------------

df["cooling_impact"] = (
    0.50 * df["compound_risk"]
    + 0.30 * df["cooling_stress"]
    + 0.20 * df["vulnerability"]
)


# -----------------------------
# Water tanker
# -----------------------------

df["water_impact"] = (
    0.50 * df["compound_risk"]
    + 0.30 * df["water_stress"]
    + 0.20 * df["exposure_score"]
)


# -----------------------------
# Medical team
# -----------------------------

df["medical_impact"] = (
    0.50 * df["compound_risk"]
    + 0.30 * df["healthcare_stress"]
    + 0.20 * df["vulnerability"]
)


# ============================================================
# 11. EQUITY SCORE
# ============================================================
#
# High vulnerability + poor access
# means higher equity priority.
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

    # -------------------------
    # Cooling centre
    # -------------------------

    candidates.append({
        "zone": row["zone"],
        "intervention": "cooling_centre",
        "impact": row["cooling_impact"],
        "equity": row["equity_score"],
        "risk_class": int(row["risk_class"]),
        "risk_label": row["risk_label"],
        "confidence": row["ml_confidence"],
        "compound_risk": row["compound_risk"]
    })


    # -------------------------
    # Water tanker
    # -------------------------

    candidates.append({
        "zone": row["zone"],
        "intervention": "water_tanker",
        "impact": row["water_impact"],
        "equity": row["equity_score"],
        "risk_class": int(row["risk_class"]),
        "risk_label": row["risk_label"],
        "confidence": row["ml_confidence"],
        "compound_risk": row["compound_risk"]
    })


    # -------------------------
    # Medical team
    # -------------------------

    candidates.append({
        "zone": row["zone"],
        "intervention": "medical_team",
        "impact": row["medical_impact"],
        "equity": row["equity_score"],
        "risk_class": int(row["risk_class"]),
        "risk_label": row["risk_label"],
        "confidence": row["ml_confidence"],
        "compound_risk": row["compound_risk"]
    })


# ============================================================
# 13. DECISION SCORE
# ============================================================
#
# Intervention impact = 70%
# Equity              = 30%
#
# This ensures that resource allocation
# considers both effectiveness and fairness.
# ============================================================

for candidate in candidates:

    candidate["decision_score"] = (
        0.70 * candidate["impact"]
        + 0.30 * candidate["equity"]
    )


# ============================================================
# 14. SORT CANDIDATES
# ============================================================

candidates = sorted(
    candidates,
    key=lambda x: x["decision_score"],
    reverse=True
)


# ============================================================
# 15. RESOURCE ALLOCATION
# ============================================================
#
# Each resource has a limited capacity.
#
# A zone CAN receive multiple intervention types.
#
# Example:
#
# Zone A:
#   cooling centre
#   water tanker
#   medical team
#
# if those resources are available.
# ============================================================

remaining_resources = RESOURCES.copy()

allocated = []

for candidate in candidates:

    intervention = candidate["intervention"]

    if remaining_resources[intervention] <= 0:
        continue

    allocated.append(candidate)

    remaining_resources[intervention] -= 1


# ============================================================
# 16. ZONE RISK ASSESSMENT
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
    "compound_risk"
]

display_df = df[risk_columns].copy()

display_df["ml_confidence"] = (
    display_df["ml_confidence"] * 100
)

print(
    display_df
    .round(2)
    .to_string(index=False)
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
    "equity_score"
]

print(
    df[impact_columns]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 18. AI EXPLANATION
# ============================================================

print("\n======================================")
print("          AI RISK EXPLANATIONS")
print("======================================\n")

for _, row in df.iterrows():

    factors = []

    if row["temperature"] >= 42:
        factors.append("very high temperature")

    if row["vegetation"] <= 20:
        factors.append("low vegetation")

    if row["building_density"] >= 80:
        factors.append("high building density")

    if row["vulnerability"] >= 80:
        factors.append("high vulnerability")

    if row["cooling_access"] <= 30:
        factors.append("limited cooling access")

    if row["water_access"] <= 30:
        factors.append("limited water access")

    if row["hospital_capacity"] <= 50:
        factors.append("limited hospital capacity")

    if not factors:
        factors.append("no dominant threshold-based stress factor")

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

    print("No interventions could be allocated.")

else:

    for priority, item in enumerate(
        allocated,
        start=1
    ):

        print(
            f"Priority {priority}: "
            f"Zone {item['zone']} → "
            f"{item['intervention']}"
        )

        print(
            f"    Risk: {item['risk_label']}"
        )

        print(
            f"    AI Confidence: "
            f"{item['confidence']:.2%}"
        )

        print(
            f"    Compound Risk: "
            f"{item['compound_risk']:.2f}"
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
            f"    Decision Score: "
            f"{item['decision_score']:.2f}"
        )

        print()


# ============================================================
# 20. REMAINING RESOURCES
# ============================================================

print("======================================")
print("       REMAINING RESOURCES")
print("======================================\n")

for resource, quantity in remaining_resources.items():

    print(
        f"{resource}: {quantity}"
    )


# ============================================================
# 21. HUMAN APPROVAL NOTICE
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