import os
import joblib
import pandas as pd


# ============================================================
# MODEL PATH
# ============================================================

# Project root:
# heat-waves-emergency-response/
# ├── models/
# │   └── rf_model.pkl
# └── src/
#     └── ml/
#         └── predict.py

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "rf_model.pkl"
)


# ============================================================
# MODEL FEATURES
# ============================================================

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


# ============================================================
# RISK LABELS
# ============================================================

RISK_LABELS = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
    3: "CRITICAL"
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained Random Forest model.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# PREDICT ONE ZONE
# ============================================================

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
        Risk class, label, confidence and probabilities.
    """

    model = load_model()

    input_data = pd.DataFrame(
        [zone_data],
        columns=FEATURES
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    classes = model.classes_

    probability_map = {
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

        "probabilities": probability_map
    }


# ============================================================
# PREDICT MULTIPLE ZONES
# ============================================================

def predict_zones(df):
    """
    Predict heatwave risk for multiple zones.

    The Random Forest model is loaded once and all zones
    are predicted in a single batch.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the eight model features.

    Returns
    -------
    list
        Prediction result for every zone.
    """

    model = load_model()

    # Make sure all required features exist
    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    input_data = df[FEATURES].copy()

    # Predict all zones at once
    predictions = model.predict(
        input_data
    )

    probabilities = model.predict_proba(
        input_data
    )

    classes = model.classes_

    results = []

    for prediction, probs in zip(
        predictions,
        probabilities
    ):

        probability_map = {
            RISK_LABELS.get(
                int(cls),
                str(cls)
            ): round(
                float(prob),
                4
            )
            for cls, prob in zip(
                classes,
                probs
            )
        }

        confidence = max(probs)

        results.append({

            "risk_class": int(
                prediction
            ),

            "risk_label": RISK_LABELS.get(
                int(prediction),
                "UNKNOWN"
            ),

            "confidence": round(
                float(confidence),
                4
            ),

            "probabilities": probability_map

        })

    return results


# ============================================================
# EXPLAIN ONE ZONE
# ============================================================

def explain_zone(zone_data):
    """
    Generate human-readable heat-risk factors.
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


# ============================================================
# TEST SINGLE ZONE
# ============================================================

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

    result = predict_zone(
        example_zone
    )

    print("\nHEATWAVE RISK PREDICTION")
    print("=" * 40)

    print(
        f"Risk: {result['risk_label']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.2%}"
    )

    print("\nProbabilities:")

    for label, probability in result[
        "probabilities"
    ].items():

        print(
            f"  {label}: "
            f"{probability:.2%}"
        )

    print("\nCONTRIBUTING FACTORS:")

    factors = explain_zone(
        example_zone
    )

    for factor in factors:

        print(
            f"  - {factor}"
        )