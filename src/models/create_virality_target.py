import pandas as pd

# Load feature-engineered dataset
df = pd.read_csv("data/processed/post_engagement_features.csv")

print("=" * 70)
print("CREATING VIRALITY TARGET")
print("=" * 70)

# Calculate the 75th percentile
virality_threshold = df["virality_score"].quantile(0.75)

print(f"\nVirality threshold (75th percentile): {virality_threshold:.6f}")

# Create binary target
df["virality_label"] = (
    df["virality_score"] >= virality_threshold
).astype(int)

# Display class distribution
print("\nTarget distribution:")
print(df["virality_label"].value_counts())

print("\nTarget distribution (%):")
print(
    (df["virality_label"].value_counts(normalize=True) * 100)
    .round(2)
)

# Display sample
print("\nSample:")
print(
    df[
        [
            "post_id",
            "virality_score",
            "virality_label"
        ]
    ].head(10).to_string(index=False)
)

# Save dataset
output_path = "data/processed/virality_dataset.csv"

df.to_csv(output_path, index=False)

print(f"\nSaved dataset to: {output_path}")

print("\n" + "=" * 70)
print("VIRALITY TARGET CREATED SUCCESSFULLY")
print("=" * 70)