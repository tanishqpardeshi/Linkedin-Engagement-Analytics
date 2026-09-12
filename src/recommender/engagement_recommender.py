import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_PATH = Path("data/processed/post_engagement_features.csv")
OUTPUT_PATH = Path("data/processed/engagement_recommendations.csv")


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

if df.empty:
    raise ValueError("The input dataset is empty.")


required_columns = [
    "post_id",
    "engagement_rate",
    "virality_score",
    "impressions",
    "post_type",
    "day_of_week",
    "post_hour",
    "time_period",
    "has_media",
    "media_type",
    "content_length_category",
    "hashtags_count",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ---------------------------------------------------------
# Overall baseline
# ---------------------------------------------------------

overall_engagement = df["engagement_rate"].mean()
overall_virality = df["virality_score"].mean()
overall_impressions = df["impressions"].mean()


# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------

def calculate_group_stats(group_column):
    """
    Calculate historical performance statistics
    for a feature.
    """

    return (
        df.groupby(group_column)
        .agg(
            posts=("post_id", "count"),
            avg_engagement=("engagement_rate", "mean"),
            avg_virality=("virality_score", "mean"),
            avg_impressions=("impressions", "mean"),
        )
        .sort_values("avg_engagement", ascending=False)
    )


# ---------------------------------------------------------
# Historical performance
# ---------------------------------------------------------

post_type_stats = calculate_group_stats("post_type")
day_stats = calculate_group_stats("day_of_week")
hour_stats = calculate_group_stats("post_hour")
period_stats = calculate_group_stats("time_period")
media_stats = calculate_group_stats("has_media")
media_type_stats = calculate_group_stats("media_type")
length_stats = calculate_group_stats("content_length_category")


# ---------------------------------------------------------
# Hashtag statistics
# ---------------------------------------------------------

hashtag_stats = (
    df.groupby("hashtags_count")
    .agg(
        posts=("post_id", "count"),
        avg_engagement=("engagement_rate", "mean"),
        avg_virality=("virality_score", "mean"),
        avg_impressions=("impressions", "mean"),
    )
)

# Only consider hashtag counts with at least 5 posts
hashtag_stats = hashtag_stats[
    hashtag_stats["posts"] >= 5
].sort_values(
    "avg_engagement",
    ascending=False
)


# ---------------------------------------------------------
# Best historical values
# ---------------------------------------------------------

best_post_type = post_type_stats.index[0]
best_day = day_stats.index[0]
best_hour = int(hour_stats.index[0])
best_period = period_stats.index[0]

# IMPORTANT:
# Convert boolean value explicitly.
best_media = bool(media_stats.index[0])

best_media_type = media_type_stats.index[0]
best_length = length_stats.index[0]


if not hashtag_stats.empty:
    best_hashtags = int(hashtag_stats.index[0])
else:
    best_hashtags = int(df["hashtags_count"].median())


# ---------------------------------------------------------
# Get historical engagement values
# ---------------------------------------------------------

post_type_engagement = post_type_stats.loc[
    best_post_type, "avg_engagement"
]

day_engagement = day_stats.loc[
    best_day, "avg_engagement"
]

hour_engagement = hour_stats.loc[
    best_hour, "avg_engagement"
]

period_engagement = period_stats.loc[
    best_period, "avg_engagement"
]


# FIX:
# Instead of using .loc[True] / .loc[False],
# find the corresponding row explicitly.

media_engagement = media_stats[
    media_stats.index.astype(bool) == best_media
]["avg_engagement"].iloc[0]


media_type_engagement = media_type_stats.loc[
    best_media_type, "avg_engagement"
]

length_engagement = length_stats.loc[
    best_length, "avg_engagement"
]


if best_hashtags in hashtag_stats.index:
    hashtag_engagement = hashtag_stats.loc[
        best_hashtags, "avg_engagement"
    ]
else:
    hashtag_engagement = overall_engagement


# ---------------------------------------------------------
# Build recommendation table
# ---------------------------------------------------------

recommendations = pd.DataFrame(
    {
        "recommendation": [
            "Post Type",
            "Day of Week",
            "Posting Hour",
            "Time Period",
            "Media Usage",
            "Media Type",
            "Content Length",
            "Hashtag Count",
        ],

        "recommended_value": [
            best_post_type,
            best_day,
            best_hour,
            best_period,
            "Use Media" if best_media else "No Media",
            best_media_type,
            best_length,
            best_hashtags,
        ],

        "historical_avg_engagement": [
            post_type_engagement,
            day_engagement,
            hour_engagement,
            period_engagement,
            media_engagement,
            media_type_engagement,
            length_engagement,
            hashtag_engagement,
        ],
    }
)


# ---------------------------------------------------------
# Compare with overall performance
# ---------------------------------------------------------

recommendations["difference_from_overall"] = (
    recommendations["historical_avg_engagement"]
    - overall_engagement
)


recommendations["difference_percentage"] = (
    recommendations["difference_from_overall"]
    / overall_engagement
    * 100
)


# ---------------------------------------------------------
# Save recommendations
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

recommendations.to_csv(
    OUTPUT_PATH,
    index=False
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print()
print("ENGAGEMENT OPTIMIZATION RECOMMENDER")
print("=" * 50)

print(
    f"Overall average engagement: "
    f"{overall_engagement:.2f}%"
)

print(
    f"Overall average virality: "
    f"{overall_virality:.2f}"
)

print(
    f"Overall average impressions: "
    f"{overall_impressions:.2f}"
)

print()
print("Recommended posting strategy")
print("-" * 50)


for _, row in recommendations.iterrows():

    print(
        f"{row['recommendation']}: "
        f"{row['recommended_value']} "
        f"| Historical engagement: "
        f"{row['historical_avg_engagement']:.2f}% "
        f"| Difference: "
        f"{row['difference_percentage']:+.2f}%"
    )


print()
print("Recommendation file saved to:")
print(OUTPUT_PATH)