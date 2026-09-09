import pandas as pd
from pathlib import Path


DATA_DIR = Path("data/raw")


def inspect_dataframe(df, dataset_name):

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())


def main():

    # Dataset 1: Comments
    comments = pd.read_csv(
        DATA_DIR / "comments.csv"
    )

    inspect_dataframe(
        comments,
        "comments.csv"
    )

    # Dataset 2: LinkedIn Post Analytics
    post_analytics = pd.read_excel(
        DATA_DIR / "linkedin_post_analytics_data.xlsx"
    )

    inspect_dataframe(
        post_analytics,
        "linkedin_post_analytics_data.xlsx"
    )

    # Dataset 3: Linked Post Engagement Analytics
    post_engagement = pd.read_csv(
        DATA_DIR / "linkedIn_post_engagement_analytics_medium.csv"
    )

    inspect_dataframe(
        post_engagement,
        "linkedIn_post_engagement_analytics_medium.csv"
    )


if __name__ == "__main__":
    main()