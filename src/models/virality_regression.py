import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/processed/virality_dataset.csv"
)

print("=" * 70)
print("VIRALITY SCORE PREDICTION - REGRESSION")
print("=" * 70)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

features = [
    "post_type",
    "content_length",
    "hashtags_count",
    "has_media",
    "media_type",
    "post_hour",
    "day_of_week"
]

target = "virality_score"

X = df[features]
y = df[target]


# ============================================================
# 3. FEATURE TYPES
# ============================================================

numerical_features = [
    "content_length",
    "hashtags_count",
    "post_hour"
]

categorical_features = [
    "post_type",
    "has_media",
    "media_type",
    "day_of_week"
]


# ============================================================
# 4. PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 5. RANDOM FOREST REGRESSOR
# ============================================================

regressor = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ]
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nDataset split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ============================================================
# 7. TRAIN
# ============================================================

print("\nTraining regression model...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# 8. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 9. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("REGRESSION PERFORMANCE")
print("=" * 70)

print(f"\nMAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# ============================================================
# 10. SAMPLE PREDICTIONS
# ============================================================

results = X_test.copy()

results["actual_virality_score"] = y_test
results["predicted_virality_score"] = y_pred

results["prediction_error"] = (
    results["actual_virality_score"]
    - results["predicted_virality_score"]
)

results = results.sort_values(
    "predicted_virality_score",
    ascending=False
)


print("\nTop predicted posts:")
print("-" * 70)

print(
    results.head(10).to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("VIRALITY REGRESSION COMPLETE")
print("=" * 70)