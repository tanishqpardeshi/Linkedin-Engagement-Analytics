import pandas as pd


# ============================================================
# LOAD DATA
# ============================================================

file_path = "data/raw/LinkedIn Profile Engagment.csv"

df = pd.read_csv(file_path)


print("=" * 70)
print("LINKEDIN PROFILE ENGAGEMENT ANALYSIS")
print("=" * 70)


# ============================================================
# 1. BASIC DATASET INFORMATION
# ============================================================

print("\n1. DATASET OVERVIEW")
print("-" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nUnique URNs:", df["urn"].nunique())
print("Unique post URLs:", df["postUrl"].nunique())


# ============================================================
# 2. POST TEXT ANALYSIS
# ============================================================

print("\n2. POST TEXT ANALYSIS")
print("-" * 70)

print("Missing text:", df["text"].isna().sum())

text_lengths = df["text"].dropna().str.len()

print("\nText length statistics:")
print(text_lengths.describe())


# ============================================================
# 3. ENGAGEMENT STATISTICS
# ============================================================

print("\n3. ENGAGEMENT STATISTICS")
print("-" * 70)

engagement_columns = [
    "totalReactionCount",
    "likeCount",
    "commentsCount",
    "repostsCount"
]

print(
    df[engagement_columns]
    .describe()
    .round(2)
)


# ============================================================
# 4. ENGAGEMENT QUANTILES
# ============================================================

print("\n4. ENGAGEMENT QUANTILES")
print("-" * 70)

for column in engagement_columns:

    print(f"\n{column}:")

    print(
        df[column]
        .quantile(
            [0, 0.25, 0.50, 0.75, 0.90, 1.00]
        )
        .round(2)
    )


# ============================================================
# 5. POSTED DATE
# ============================================================

print("\n5. POST DATE ANALYSIS")
print("-" * 70)

print("Missing postedAt:", df["postedAt"].isna().sum())

print("\nSample postedAt values:")
print(
    df["postedAt"]
    .dropna()
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 6. AUTHORS
# ============================================================

print("\n6. AUTHOR INFORMATION")
print("-" * 70)

print(
    "Unique authors:",
    df["author"].nunique()
)

print(
    "Unique companies:",
    df["company"].nunique()
)


# ============================================================
# 7. ARTICLE INFORMATION
# ============================================================

print("\n7. ARTICLE INFORMATION")
print("-" * 70)

print(
    "Posts with article data:",
    (df["article"] != "{}").sum()
)

print(
    "Posts without article data:",
    (df["article"] == "{}").sum()
)


# ============================================================
# 8. TOP POSTS
# ============================================================

print("\n8. TOP 10 POSTS BY REACTION COUNT")
print("-" * 70)

top_posts = df.nlargest(
    10,
    "totalReactionCount"
)[
    [
        "urn",
        "totalReactionCount",
        "likeCount",
        "commentsCount",
        "repostsCount",
        "text"
    ]
]

print(
    top_posts.to_string(index=False)
)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)