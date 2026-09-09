import pandas as pd
from pathlib import Path


# Input and output file paths
INPUT_PATH = Path(
    "data/processed/cleaned_post_engagement.csv"
)

OUTPUT_PATH = Path(
    "data/processed/post_engagement_features.csv"
)


def get_time_period(hour):
    """
    Categorize a posting hour into a time period.
    """

    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


def create_features():

    # Load cleaned dataset
    df = pd.read_csv(INPUT_PATH)

    print("Original columns:")
    print(df.columns.tolist())


    # ---------------------------------
    # ENGAGEMENT FEATURES
    # ---------------------------------

    # Total engagement
    df["total_engagement"] = (
        df["likes"]
        + df["comments"]
        + df["shares"]
    )


    # Like rate
    df["like_rate"] = (
        df["likes"] / df["impressions"]
    ) * 100


    # Comment rate
    df["comment_rate"] = (
        df["comments"] / df["impressions"]
    ) * 100


    # Share rate
    df["share_rate"] = (
        df["shares"] / df["impressions"]
    ) * 100


    # ---------------------------------
    # VIRALITY FEATURE
    # ---------------------------------

    # Shares have the highest importance,
    # followed by comments and likes.
    df["virality_score"] = (
        (df["likes"] * 1)
        + (df["comments"] * 2)
        + (df["shares"] * 3)
    ) / df["impressions"] * 100


    # ---------------------------------
    # POSTING TIME FEATURES
    # ---------------------------------

    # Convert post_time temporarily for analysis
    time_data = pd.to_datetime(
        df["post_time"],
        format="%H:%M",
        errors="coerce"
    )


    # Extract posting hour
    df["post_hour"] = time_data.dt.hour


    # Categorize posting time
    df["time_period"] = df["post_hour"].apply(
        get_time_period
    )


    # ---------------------------------
    # SAVE FEATURE DATASET
    # ---------------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print("\nNew columns created:")
    print(
        [
            "total_engagement",
            "like_rate",
            "comment_rate",
            "share_rate",
            "virality_score",
            "post_hour",
            "time_period"
        ]
    )


    print("\nFeature dataset saved successfully.")
    print(f"Location: {OUTPUT_PATH}")


    print("\nFirst 5 rows:")

    print(
        df[
            [
                "post_id",
                "total_engagement",
                "like_rate",
                "comment_rate",
                "share_rate",
                "virality_score",
                "post_hour",
                "time_period"
            ]
        ].head()
    )


if __name__ == "__main__":
    create_features()