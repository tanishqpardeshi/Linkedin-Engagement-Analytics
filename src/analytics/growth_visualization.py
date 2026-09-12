import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_PATH = Path(
    "data/processed/post_engagement_features.csv"
)

FORECAST_PATH = Path(
    "data/processed/trend_forecast.csv"
)

OUTPUT_DIR = Path(
    "data/processed/visualizations"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

if df.empty:
    raise ValueError(
        "The post engagement dataset is empty."
    )


required_columns = [
    "post_id",
    "post_date",
    "post_type",
    "day_of_week",
    "post_hour",
    "engagement_rate",
    "impressions",
    "total_engagement",
    "virality_score",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


df["post_date"] = pd.to_datetime(
    df["post_date"],
    errors="coerce"
)

df = df.dropna(
    subset=["post_date"]
)


# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# 1. Engagement Trend
# ---------------------------------------------------------

daily_engagement = (
    df.groupby("post_date")
    .agg(
        avg_engagement=(
            "engagement_rate",
            "mean"
        )
    )
    .reset_index()
    .sort_values("post_date")
)


plt.figure(figsize=(12, 6))

plt.plot(
    daily_engagement["post_date"],
    daily_engagement["avg_engagement"],
    linewidth=2
)

plt.title(
    "LinkedIn Engagement Trend"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Average Engagement Rate (%)"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "engagement_trend.png",
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# 2. Impressions Trend
# ---------------------------------------------------------

daily_impressions = (
    df.groupby("post_date")
    .agg(
        avg_impressions=(
            "impressions",
            "mean"
        )
    )
    .reset_index()
    .sort_values("post_date")
)


plt.figure(figsize=(12, 6))

plt.plot(
    daily_impressions["post_date"],
    daily_impressions["avg_impressions"],
    linewidth=2
)

plt.title(
    "LinkedIn Impressions Trend"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Average Impressions"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "impressions_trend.png",
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# 3. Engagement by Post Type
# ---------------------------------------------------------

post_type_stats = (
    df.groupby("post_type")
    .agg(
        avg_engagement=(
            "engagement_rate",
            "mean"
        )
    )
    .sort_values(
        "avg_engagement",
        ascending=False
    )
)


plt.figure(figsize=(9, 6))

plt.bar(
    post_type_stats.index,
    post_type_stats["avg_engagement"]
)

plt.title(
    "Average Engagement by Post Type"
)

plt.xlabel(
    "Post Type"
)

plt.ylabel(
    "Average Engagement Rate (%)"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "engagement_by_post_type.png",
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# 4. Engagement by Day of Week
# ---------------------------------------------------------

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

day_stats = (
    df.groupby("day_of_week")
    .agg(
        avg_engagement=(
            "engagement_rate",
            "mean"
        )
    )
)

day_stats = day_stats.reindex(
    [
        day
        for day in day_order
        if day in day_stats.index
    ]
)


plt.figure(figsize=(10, 6))

plt.bar(
    day_stats.index,
    day_stats["avg_engagement"]
)

plt.title(
    "Average Engagement by Day of Week"
)

plt.xlabel(
    "Day"
)

plt.ylabel(
    "Average Engagement Rate (%)"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "engagement_by_day.png",
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# 5. Engagement by Posting Hour
# ---------------------------------------------------------

hour_stats = (
    df.groupby("post_hour")
    .agg(
        avg_engagement=(
            "engagement_rate",
            "mean"
        )
    )
    .sort_index()
)


plt.figure(figsize=(12, 6))

plt.plot(
    hour_stats.index,
    hour_stats["avg_engagement"],
    marker="o",
    linewidth=2
)

plt.title(
    "Average Engagement by Posting Hour"
)

plt.xlabel(
    "Posting Hour"
)

plt.ylabel(
    "Average Engagement Rate (%)"
)

plt.xticks(
    hour_stats.index
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "engagement_by_hour.png",
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# 6. Virality Distribution
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    df["virality_score"],
    bins=20
)

plt.title(
    "Virality Score Distribution"
)

plt.xlabel(
    "Virality Score"
)

plt.ylabel(
    "Number of Posts"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "virality_distribution.png",
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# 7. Historical Engagement + Forecast
# ---------------------------------------------------------

if FORECAST_PATH.exists():

    forecast = pd.read_csv(
        FORECAST_PATH
    )

    forecast["date"] = pd.to_datetime(
        forecast["date"],
        errors="coerce"
    )

    plt.figure(figsize=(13, 6))

    plt.plot(
        daily_engagement["post_date"],
        daily_engagement["avg_engagement"],
        linewidth=2,
        label="Historical Engagement"
    )

    plt.plot(
        forecast["date"],
        forecast["forecast_engagement"],
        linestyle="--",
        linewidth=2,
        label="14-Day Forecast"
    )

    plt.title(
        "Historical Engagement and 14-Day Forecast"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Engagement Rate (%)"
    )

    plt.legend()

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "engagement_forecast.png",
        dpi=150
    )

    plt.close()


# ---------------------------------------------------------
# 8. Top Performing Posts
# ---------------------------------------------------------

top_posts = (
    df[
        [
            "post_id",
            "post_date",
            "post_type",
            "impressions",
            "engagement_rate",
            "total_engagement",
            "virality_score",
        ]
    ]
    .sort_values(
        "engagement_rate",
        ascending=False
    )
    .head(10)
)


top_posts.to_csv(
    OUTPUT_DIR / "top_performing_posts.csv",
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()
print(
    "GROWTH VISUALIZATION MODULE"
)

print(
    "=" * 50
)

print(
    f"Posts analysed: {len(df)}"
)

print(
    f"Historical dates: "
    f"{df['post_date'].nunique()}"
)

print()
print(
    "Visualizations generated:"
)

print(
    "- Engagement trend"
)

print(
    "- Impressions trend"
)

print(
    "- Engagement by post type"
)

print(
    "- Engagement by day of week"
)

print(
    "- Engagement by posting hour"
)

print(
    "- Virality distribution"
)

if FORECAST_PATH.exists():
    print(
        "- Historical engagement + 14-day forecast"
    )

print(
    "- Top-performing posts table"
)

print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)