import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/processed/virality_dataset.csv"
)

print("=" * 70)
print("VIRALITY PREDICTION - RANDOM FOREST")
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

target = "virality_label"

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
# 5. RANDOM FOREST MODEL
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", random_forest)
    ]
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ============================================================
# 7. TRAIN
# ============================================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# 8. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 9. PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n" + "=" * 70)
print("RANDOM FOREST PERFORMANCE")
print("=" * 70)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print("-" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-Viral",
            "Viral"
        ]
    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")
print("-" * 70)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 12. SAMPLE PREDICTIONS
# ============================================================

results = X_test.copy()

results["actual"] = y_test
results["predicted"] = y_pred
results["viral_probability"] = y_probability

results = results.sort_values(
    "viral_probability",
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
print("RANDOM FOREST ANALYSIS COMPLETE")
print("=" * 70)