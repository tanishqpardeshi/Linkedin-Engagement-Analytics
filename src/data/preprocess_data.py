import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path(
    "data/raw/LinkedIn_Post_Engagement_Analytics_Medium.csv"
)

PROCESSED_DATA_PATH = Path(
    "data/processed/cleaned_post_engagement.csv"
)


def preprocess_post_engagement():

    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)

    print("Original shape:", df.shape)

    # Convert date column
    df["post_date"] = pd.to_datetime(
        df["post_date"],
        errors="coerce"
    )

    # Handle missing media types
    df["media_type"] = df["media_type"].fillna("No Media")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Standardize text columns
    df["post_type"] = df["post_type"].str.strip()
    df["day_of_week"] = df["day_of_week"].str.strip()
    df["media_type"] = df["media_type"].str.strip()

    # Save cleaned dataset
    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print("Cleaned shape:", df.shape)
    print("\nCleaned dataset saved successfully.")
    print(f"Location: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    preprocess_post_engagement()