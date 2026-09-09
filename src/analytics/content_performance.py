import pandas as pd
from pathlib import Path


# Path to the analytics-ready dataset
INPUT_PATH = Path(
    "data/processed/post_engagement_features.csv"
)


def analyze_content_performance():

    # Load dataset
    df = pd.read_csv(INPUT_PATH)

    print("\n" + "=" * 70)
    print("CONTENT PERFORMANCE ANALYSIS")
    print("=" * 70)


    # ---------------------------------
    # OVERALL PERFORMANCE
    # ---------------------------------

    print("\nOVERALL PERFORMANCE")

    overall_metrics = {
        "Total Posts": len(df),
        "Average Impressions": df["impressions"].mean(),
        "Average Engagement Rate": df["engagement_rate"].mean(),
        "Average Total Engagement": df["total_engagement"].mean(),
        "Average Virality Score": df["virality_score"].mean()
    }

    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.2f}")


    # ---------------------------------
    # PERFORMANCE BY POST TYPE
    # ---------------------------------

    print("\n" + "-" * 70)
    print("PERFORMANCE BY POST TYPE")
    print("-" * 70)

    post_type_performance = (
        df.groupby("post_type")
        .agg(
            posts=("post_id", "count"),
            avg_impressions=("impressions", "mean"),
            avg_engagement_rate=("engagement_rate", "mean"),
            avg_total_engagement=("total_engagement", "mean"),
            avg_virality_score=("virality_score", "mean")
        )
        .round(2)
        .sort_values(
            by="avg_virality_score",
            ascending=False
        )
    )

    print(post_type_performance)


    # ---------------------------------
    # PERFORMANCE BY DAY OF WEEK
    # ---------------------------------

    print("\n" + "-" * 70)
    print("PERFORMANCE BY DAY OF WEEK")
    print("-" * 70)

    day_performance = (
        df.groupby("day_of_week")
        .agg(
            posts=("post_id", "count"),
            avg_engagement_rate=("engagement_rate", "mean"),
            avg_virality_score=("virality_score", "mean"),
            avg_total_engagement=("total_engagement", "mean")
        )
        .round(2)
        .sort_values(
            by="avg_engagement_rate",
            ascending=False
        )
    )

    print(day_performance)


    # ---------------------------------
    # PERFORMANCE BY MEDIA TYPE
    # ---------------------------------

    print("\n" + "-" * 70)
    print("PERFORMANCE BY MEDIA TYPE")
    print("-" * 70)

    media_performance = (
        df.groupby("media_type")
        .agg(
            posts=("post_id", "count"),
            avg_engagement_rate=("engagement_rate", "mean"),
            avg_virality_score=("virality_score", "mean"),
            avg_total_engagement=("total_engagement", "mean")
        )
        .round(2)
        .sort_values(
            by="avg_engagement_rate",
            ascending=False
        )
    )

    print(media_performance)


    # ---------------------------------
    # PERFORMANCE BY POSTING HOUR
    # ---------------------------------

    print("\n" + "-" * 70)
    print("PERFORMANCE BY POSTING HOUR")
    print("-" * 70)

    hour_performance = (
        df.groupby("post_hour")
        .agg(
            posts=("post_id", "count"),
            avg_engagement_rate=("engagement_rate", "mean"),
            avg_virality_score=("virality_score", "mean"),
            avg_total_engagement=("total_engagement", "mean")
        )
        .round(2)
        .sort_values(
            by="avg_engagement_rate",
            ascending=False
        )
    )

    print(hour_performance)


if __name__ == "__main__":
    analyze_content_performance()