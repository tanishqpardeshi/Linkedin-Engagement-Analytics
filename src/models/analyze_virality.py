import pandas as pd

# Load processed dataset
df = pd.read_csv("data/processed/post_engagement_features.csv")

print("=" * 70)
print("VIRALITY SCORE ANALYSIS")
print("=" * 70)

# Basic statistics
print("\n1. BASIC STATISTICS")
print("-" * 70)

print(df["virality_score"].describe())


# Quantiles
print("\n2. VIRALITY SCORE QUANTILES")
print("-" * 70)

quantiles = df["virality_score"].quantile(
    [0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
)

print(quantiles)


# Number of unique values
print("\n3. UNIQUE VALUES")
print("-" * 70)

print("Unique virality scores:", df["virality_score"].nunique())


# Distribution ranges
print("\n4. VIRALITY SCORE RANGES")
print("-" * 70)

bins = [-float("inf"), 2, 4, 6, 8, 10, float("inf")]
labels = [
    "Very Low",
    "Low",
    "Medium",
    "High",
    "Very High",
    "Extremely High"
]

df["virality_range"] = pd.cut(
    df["virality_score"],
    bins=bins,
    labels=labels
)

print(df["virality_range"].value_counts().sort_index())


# Top posts
print("\n5. TOP 10 POSTS BY VIRALITY SCORE")
print("-" * 70)

top_posts = df.nlargest(
    10,
    "virality_score"
)[
    [
        "post_id",
        "post_type",
        "impressions",
        "likes",
        "comments",
        "shares",
        "engagement_rate",
        "virality_score"
    ]
]

print(top_posts.to_string(index=False))


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)