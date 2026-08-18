import csv
import random

random.seed(42)

N = 1000

rows = []

for i in range(1, N + 1):

    temperature = random.uniform(35, 45)
    vegetation = random.uniform(5, 80)
    building_density = random.uniform(30, 95)
    population = random.randint(5000, 50000)
    vulnerability = random.uniform(20, 100)
    cooling_access = random.uniform(10, 100)
    water_access = random.uniform(10, 100)
    hospital_capacity = random.uniform(20, 100)

    # Prototype heat-risk score
    risk_score = (
        0.30 * ((temperature - 35) / 10) * 100
        + 0.15 * ((100 - vegetation) / 100) * 100
        + 0.15 * (building_density / 100) * 100
        + 0.15 * (vulnerability / 100) * 100
        + 0.10 * ((100 - cooling_access) / 100) * 100
        + 0.10 * ((100 - water_access) / 100) * 100
        + 0.05 * ((100 - hospital_capacity) / 100) * 100
    )

    risk_score = max(0, min(100, risk_score))

    if risk_score <= 35:
        risk_class = 0
    elif risk_score <= 55:
        risk_class = 1
    elif risk_score <= 75:
        risk_class = 2
    else:
        risk_class = 3

    rows.append([
        f"Z{i:04d}",
        round(temperature, 2),
        round(vegetation, 2),
        round(building_density, 2),
        population,
        round(vulnerability, 2),
        round(cooling_access, 2),
        round(water_access, 2),
        round(hospital_capacity, 2),
        round(risk_score, 2),
        risk_class
    ])


columns = [
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


with open("zones.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(columns)
    writer.writerows(rows)


print(f"Generated {N} urban zones.")

print("Dataset saved as zones.csv")

print("\nFirst 5 rows:")

for row in rows[:5]:
    print(row)

print("\nRisk distribution:")

distribution = {}

for row in rows:
    risk = row[-1]
    distribution[risk] = distribution.get(risk, 0) + 1

for risk_class in sorted(distribution):
    print(f"Risk {risk_class}: {distribution[risk_class]} zones")