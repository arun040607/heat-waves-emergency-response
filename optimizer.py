import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/zones.csv")


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

# Temperature
min_temp = df["temperature"].min()
max_temp = df["temperature"].max()

if max_temp == min_temp:
    df["temperature_score"] = 50
else:
    df["temperature_score"] = (
        (df["temperature"] - min_temp)
        / (max_temp - min_temp)
    ) * 100


# Vegetation
#
# Low vegetation = higher heat contribution.
#
min_vegetation = df["vegetation"].min()
max_vegetation = df["vegetation"].max()

if max_vegetation == min_vegetation:
    df["vegetation_heat_score"] = 50
else:
    df["vegetation_heat_score"] = (
        (max_vegetation - df["vegetation"])
        / (max_vegetation - min_vegetation)
    ) * 100


# Building density
#
# High building density = higher heat contribution.
#
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
# 4. ENVIRONMENTAL HEAT DRIVER SCORE
# ============================================================
#
# 50% temperature
# 25% vegetation deficit
# 25% building density
#
# This is an interpretable factor score.
# It is NOT the ML prediction.
# ============================================================

df["heat_driver_score"] = (
    0.50 * df["temperature_score"]
    + 0.25 * df["vegetation_heat_score"]
    + 0.25 * df["building_heat_score"]
)


# ============================================================
# 5. TEMPORARY ML RISK
# ============================================================
#
# IMPORTANT:
# This is temporary until Member 2's
# Random Forest + XGBoost model is connected.
#
# For now we use temperature as a placeholder.
# ============================================================

df["ml_risk"] = df["temperature_score"]


# ============================================================
# 6. SERVICE STRESS
# ============================================================

df["cooling_stress"] = 100 - df["cooling_access"]

df["water_stress"] = 100 - df["water_access"]

df["healthcare_stress"] = 100 - df["hospital_capacity"]


# ============================================================
# 7. COMPOUND RISK
# ============================================================
#
# ML risk          = 40%
# Heat drivers     = 20%
# Vulnerability    = 20%
# Cooling stress   = 10%
# Water stress     = 5%
# Healthcare       = 5%
#
# This combines AI influence + interpretable factors.
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
# 8. INTERVENTION IMPACT
# ============================================================

# Cooling centre
df["cooling_impact"] = (
    0.50 * df["compound_risk"]
    + 0.30 * df["cooling_stress"]
    + 0.20 * df["vulnerability"]
)


# Water tanker
df["water_impact"] = (
    0.50 * df["compound_risk"]
    + 0.30 * df["water_stress"]
    + 0.20 * df["exposure_score"]
)


# Medical team
df["medical_impact"] = (
    0.50 * df["compound_risk"]
    + 0.30 * df["healthcare_stress"]
    + 0.20 * df["vulnerability"]
)


# ============================================================
# 9. EQUITY SCORE
# ============================================================

df["equity_score"] = (
    0.50 * df["vulnerability"]
    + 0.25 * df["cooling_stress"]
    + 0.25 * df["water_stress"]
)


# ============================================================
# 10. AVAILABLE RESOURCES
# ============================================================

resources = {
    "cooling_centre": 2,
    "water_tanker": 2,
    "medical_team": 2
}


# ============================================================
# 11. CREATE INTERVENTION CANDIDATES
# ============================================================

candidates = []

for _, row in df.iterrows():

    candidates.append({
        "zone": row["zone"],
        "intervention": "cooling_centre",
        "impact": row["cooling_impact"],
        "equity": row["equity_score"]
    })

    candidates.append({
        "zone": row["zone"],
        "intervention": "water_tanker",
        "impact": row["water_impact"],
        "equity": row["equity_score"]
    })

    candidates.append({
        "zone": row["zone"],
        "intervention": "medical_team",
        "impact": row["medical_impact"],
        "equity": row["equity_score"]
    })


# ============================================================
# 12. DECISION SCORE
# ============================================================
#
# 70% intervention impact
# 30% equity
# ============================================================

for candidate in candidates:

    candidate["decision_score"] = (
        0.70 * candidate["impact"]
        + 0.30 * candidate["equity"]
    )


# ============================================================
# 13. SORT BY DECISION SCORE
# ============================================================

candidates = sorted(
    candidates,
    key=lambda x: x["decision_score"],
    reverse=True
)


# ============================================================
# 14. RESOURCE ALLOCATION
# ============================================================

allocated = []

for candidate in candidates:

    intervention = candidate["intervention"]
    zone = candidate["zone"]

    # Resource unavailable
    if resources[intervention] <= 0:
        continue

    # One immediate intervention per zone
    if any(item["zone"] == zone for item in allocated):
        continue

    allocated.append(candidate)

    resources[intervention] -= 1


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
    "temperature_score",
    "vegetation_heat_score",
    "building_heat_score",
    "heat_driver_score",
    "ml_risk",
    "vulnerability",
    "exposure_score",
    "cooling_stress",
    "water_stress",
    "healthcare_stress",
    "compound_risk"
]

print(
    df[risk_columns]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 16. INTERVENTION IMPACT
# ============================================================

print("\n======================================")
print("       INTERVENTION IMPACT")
print("======================================\n")

impact_columns = [
    "zone",
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
# 17. RESPONSE PLAN
# ============================================================

print("\n======================================")
print("       HEATWAVE RESPONSE PLAN")
print("======================================\n")

if not allocated:

    print("No interventions could be allocated.")

else:

    for priority, item in enumerate(allocated, start=1):

        print(
            f"Priority {priority}: "
            f"Zone {item['zone']} → "
            f"{item['intervention']}\n"
            f"    Impact: {item['impact']:.2f}\n"
            f"    Equity: {item['equity']:.2f}\n"
            f"    Decision Score: {item['decision_score']:.2f}\n"
        )


# ============================================================
# 18. REMAINING RESOURCES
# ============================================================

print("======================================")
print("       REMAINING RESOURCES")
print("======================================\n")

for resource, quantity in resources.items():

    print(f"{resource}: {quantity}")