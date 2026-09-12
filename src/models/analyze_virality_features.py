import pandas as pd

# Load virality dataset
df = pd.read_csv("data/processed/virality_dataset.csv")

print("=" * 70)
print("VIRALITY FEATURE ANALYSIS")
print("=" * 70)

# Features available for prediction
features = [
    "post_type",
    "content_length",
    "hashtags_count",
    "has_media",
    "media_type",
    "post_hour",
    "day_of_week"
]

print("\n1. FEATURE INFORMATION")
print("-" * 70)

for feature in features:
    print(f"\n{feature}")
    print("Data type:", df[feature].dtype)
    print("Unique values:", df[feature].nunique())

    if df[feature].nunique() <= 15:
        print("Values:")
        print(df[feature].value_counts())


# Target distribution
print("\n\n2. VIRALITY TARGET DISTRIBUTION")
print("-" * 70)

print(df["virality_label"].value_counts())
print(
    (df["virality_label"].value_counts(normalize=True) * 100)
    .round(2)
)


# Numerical feature comparison
print("\n3. NUMERICAL FEATURES BY VIRALITY CLASS")
print("-" * 70)

numerical_features = [
    "content_length",
    "hashtags_count",
    "post_hour"
]

print(
    df.groupby("virality_label")[numerical_features]
    .mean()
    .round(2)
)


# Categorical feature comparison
print("\n4. POST TYPE VS VIRALITY")
print("-" * 70)

print(
    pd.crosstab(
        df["post_type"],
        df["virality_label"],
        normalize="index"
    ).round(3) * 100
)


print("\n5. MEDIA TYPE VS VIRALITY")
print("-" * 70)

print(
    pd.crosstab(
        df["media_type"],
        df["virality_label"],
        normalize="index"
    ).round(3) * 100
)


print("\n6. DAY OF WEEK VS VIRALITY")
print("-" * 70)

print(
    pd.crosstab(
        df["day_of_week"],
        df["virality_label"],
        normalize="index"
    ).round(3) * 100
)


print("\n" + "=" * 70)
print("FEATURE ANALYSIS COMPLETE")
print("=" * 70)