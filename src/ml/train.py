import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ==========================================
# Configuration
# ==========================================

DATA_PATH = "data/zones.csv"
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

TARGET = "risk_class"


# ==========================================
# Load dataset
# ==========================================

df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("HEATWAVE RISK MODEL TRAINING")
print("=" * 50)

print(f"\nDataset shape: {df.shape}")


# ==========================================
# Validate columns
# ==========================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ==========================================
# Features and target
# ==========================================

X = df[FEATURES]
y = df[TARGET]

print("\nFeatures:")
for feature in FEATURES:
    print(f"  - {feature}")

print("\nTarget distribution:")
print(y.value_counts().sort_index())


# ==========================================
# Train / Test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ==========================================
# Random Forest
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ==========================================
# Train
# ==========================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training complete.")


# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"\nAccuracy : {accuracy:.3f}")
print(f"Macro F1 : {macro_f1:.3f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# Feature Importance
# ==========================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance.to_string(index=False))


# ==========================================
# Save model
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")

print("\nDONE.")