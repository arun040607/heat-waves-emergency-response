import csv

FILE = "data/zones.csv"

REQUIRED_COLUMNS = [
    "zone",
    "temperature",
    "vegetation",
    "building_density",
    "population",
    "vulnerability",
    "cooling_access",
    "water_access",
    "hospital_capacity",
    "heat_risk_score",
    "risk_class"
]

with open(FILE, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("Rows:", len(rows))
print("Columns:", len(reader.fieldnames))

# 1. Check columns
missing = [
    column for column in REQUIRED_COLUMNS
    if column not in reader.fieldnames
]

if missing:
    print("❌ Missing columns:", missing)
else:
    print("✅ All required columns present")

# 2. Check missing values
missing_values = 0

for row in rows:
    for column in REQUIRED_COLUMNS:
        if row[column] == "":
            missing_values += 1

if missing_values == 0:
    print("✅ No missing values")
else:
    print("❌ Missing values:", missing_values)

# 3. Validate ranges
range_checks = {
    "vegetation": (0, 100),
    "building_density": (0, 100),
    "vulnerability": (0, 100),
    "cooling_access": (0, 100),
    "water_access": (0, 100),
    "hospital_capacity": (0, 100),
}

for column, (low, high) in range_checks.items():

    invalid = 0

    for row in rows:
        value = float(row[column])

        if value < low or value > high:
            invalid += 1

    if invalid == 0:
        print(f"✅ {column}: valid")
    else:
        print(f"❌ {column}: {invalid} invalid values")

# 4. Population check
invalid_population = sum(
    int(row["population"]) <= 0
    for row in rows
)

if invalid_population == 0:
    print("✅ population: valid")
else:
    print("❌ Invalid population:", invalid_population)

# 5. Risk class check
risk_classes = set(
    int(row["risk_class"])
    for row in rows
)

print("\nRisk classes found:", sorted(risk_classes))

if risk_classes == {0, 1, 2, 3}:
    print("✅ All four risk classes present")
else:
    print("⚠️ Some risk classes are missing")

print("\nDATA VALIDATION COMPLETE")