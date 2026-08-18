import joblib
import pandas as pd


MODEL_PATH = "models/rf_model.pkl"

FEATURES = [
    "temperature",
    "vegetation",
    "building_density",
    "population",
    "vulnerability",
    "cooling_access",
    "water_access",
    "hospital_capacity"
]


RISK_LABELS = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
    3: "CRITICAL"
}


def load_model():
    """Load the trained Random Forest model."""
    return joblib.load(MODEL_PATH)


def predict_zone(zone_data):
    """
    Predict heatwave risk for one urban zone.

    Parameters
    ----------
    zone_data : dict
        Dictionary containing the eight model features.

    Returns
    -------
    dict
        Risk class, probability and class probabilities.
    """

    model = load_model()

    # Convert input dictionary into DataFrame
    input_data = pd.DataFrame(
        [zone_data],
        columns=FEATURES
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probabilities = model.predict_proba(input_data)[0]

    classes = model.classes_

    probability_map = {
        int(cls): float(prob)
        for cls, prob in zip(classes, probabilities)
    }

    confidence = max(probabilities)

    return {
        "risk_class": int(prediction),
        "risk_label": RISK_LABELS.get(
            int(prediction),
            "UNKNOWN"
        ),
        "confidence": round(
            float(confidence),
            4
        ),
        "probabilities": {
            RISK_LABELS.get(
                int(cls),
                str(cls)
            ): round(
                float(prob),
                4
            )
            for cls, prob in zip(
                classes,
                probabilities
            )
        }
    }
def explain_zone(zone_data):
    """
    Generate a simple human-readable explanation
    of the main heat-risk factors.
    """

    factors = []

    if zone_data["temperature"] >= 42:
        factors.append(
            "Very high temperature"
        )

    if zone_data["vegetation"] <= 20:
        factors.append(
            "Low vegetation / limited cooling from greenery"
        )

    if zone_data["building_density"] >= 80:
        factors.append(
            "High building density"
        )

    if zone_data["vulnerability"] >= 80:
        factors.append(
            "High population vulnerability"
        )

    if zone_data["cooling_access"] <= 30:
        factors.append(
            "Limited cooling access"
        )

    if zone_data["water_access"] <= 30:
        factors.append(
            "Limited water access"
        )

    if zone_data["hospital_capacity"] <= 50:
        factors.append(
            "Limited hospital capacity"
        )

    return factors

if __name__ == "__main__":

    example_zone = {
        "temperature": 43.2,
        "vegetation": 8,
        "building_density": 92,
        "population": 24000,
        "vulnerability": 91,
        "cooling_access": 18,
        "water_access": 25,
        "hospital_capacity": 42
    }

    result = predict_zone(example_zone)

    print("\nHEATWAVE RISK PREDICTION")
    print("=" * 40)

    print(f"Risk: {result['risk_label']}")
    print(f"Confidence: {result['confidence']:.2%}")

    print("\nProbabilities:")

    for label, probability in result["probabilities"].items():
        print(
            f"  {label}: {probability:.2%}"
        )

    print("\nCONTRIBUTING FACTORS:")

    factors = explain_zone(example_zone)

    for factor in factors:
        print(f"  - {factor}")