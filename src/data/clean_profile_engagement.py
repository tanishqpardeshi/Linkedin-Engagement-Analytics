import pandas as pd


# ============================================================
# LOAD RAW DATA
# ============================================================

input_path = "data/raw/LinkedIn Profile Engagment.csv"

df = pd.read_csv(input_path)

print("=" * 70)
print("CLEANING LINKEDIN PROFILE ENGAGEMENT DATA")
print("=" * 70)

print(f"\nOriginal rows: {len(df)}")
print(f"Original unique URNs: {df['urn'].nunique()}")


# ============================================================
# SORT BY ENGAGEMENT
# ============================================================

# For posts that have multiple snapshots, keep the record
# with the highest total reaction count.

df = df.sort_values(
    by=["urn", "totalReactionCount"],
    ascending=[True, False]
)


# ============================================================
# KEEP ONE RECORD PER POST
# ============================================================

df_clean = df.drop_duplicates(
    subset="urn",
    keep="first"
).copy()


# ============================================================
# CLEAN INDEX
# ============================================================

df_clean = df_clean.reset_index(drop=True)


# ============================================================
# DROP UNNECESSARY COLUMN
# ============================================================

if "Unnamed: 0" in df_clean.columns:
    df_clean = df_clean.drop(
        columns=["Unnamed: 0"]
    )


# ============================================================
# VERIFY CLEAN DATA
# ============================================================

print("\nAfter cleaning:")
print(f"Rows: {len(df_clean)}")
print(f"Unique URNs: {df_clean['urn'].nunique()}")
print(f"Unique post URLs: {df_clean['postUrl'].nunique()}")


# ============================================================
# CHECK DUPLICATES
# ============================================================

print("\nDuplicate URNs:")
print(
    df_clean["urn"]
    .duplicated()
    .sum()
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    "data/processed/"
    "cleaned_profile_engagement.csv"
)

df_clean.to_csv(
    output_path,
    index=False
)

print(f"\nSaved cleaned dataset to:")
print(output_path)


# ============================================================
# FINAL PREVIEW
# ============================================================

print("\nFirst 5 cleaned posts:")
print("-" * 70)

print(
    df_clean[
        [
            "urn",
            "text",
            "totalReactionCount",
            "commentsCount",
            "repostsCount"
        ]
    ]
    .head()
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)