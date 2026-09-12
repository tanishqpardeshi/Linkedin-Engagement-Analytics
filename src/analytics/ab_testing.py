import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind


# ---------------------------------
# FILE PATH
# ---------------------------------

INPUT_PATH = Path(
    "data/processed/post_engagement_features.csv"
)


# ---------------------------------
# EFFECT SIZE
# ---------------------------------

def calculate_cohens_d(group_a, group_b):
    """
    Calculate Cohen's d for two independent groups.
    """

    mean_a = group_a.mean()
    mean_b = group_b.mean()

    std_a = group_a.std()
    std_b = group_b.std()

    n_a = len(group_a)
    n_b = len(group_b)

    # Pooled standard deviation
    pooled_std = (
        (
            (n_a - 1) * (std_a ** 2)
            + (n_b - 1) * (std_b ** 2)
        )
        / (n_a + n_b - 2)
    ) ** 0.5

    if pooled_std == 0:
        return 0

    return (mean_a - mean_b) / pooled_std


# ---------------------------------
# REUSABLE STATISTICAL TEST
# ---------------------------------

def run_ab_test(
    group_a,
    group_b,
    group_a_name,
    group_b_name,
    metric_name="engagement_rate"
):
    """
    Compare two independent groups using
    Welch's t-test and Cohen's d.
    """

    # Remove missing values
    group_a = group_a.dropna()
    group_b = group_b.dropna()

    # Calculate means
    mean_a = group_a.mean()
    mean_b = group_b.mean()

    # Welch's independent two-sample t-test
    t_statistic, p_value = ttest_ind(
        group_a,
        group_b,
        equal_var=False
    )

    # Calculate effect size
    cohens_d = calculate_cohens_d(
        group_a,
        group_b
    )

    # ---------------------------------
    # DISPLAY RESULTS
    # ---------------------------------

    print("\n" + "=" * 70)
    print(
        f"STATISTICAL COMPARISON: "
        f"{group_a_name} VS {group_b_name}"
    )
    print("=" * 70)

    print("\nGROUP INFORMATION")

    print(
        f"{group_a_name} Posts: {len(group_a)}"
    )

    print(
        f"{group_b_name} Posts: {len(group_b)}"
    )

    print(
        f"\nAverage {metric_name} - "
        f"{group_a_name}: {mean_a:.2f}%"
    )

    print(
        f"Average {metric_name} - "
        f"{group_b_name}: {mean_b:.2f}%"
    )

    print("\nSTATISTICAL TEST")

    print(
        f"T-statistic: {t_statistic:.4f}"
    )

    print(
        f"P-value: {p_value:.4f}"
    )

    print(
        f"Cohen's d: {cohens_d:.4f}"
    )

    # ---------------------------------
    # SIGNIFICANCE RESULT
    # ---------------------------------

    significance_level = 0.05

    if p_value < significance_level:

        print(
            "\nResult: Statistically Significant"
        )

        print(
            f"There is evidence of a significant "
            f"difference in {metric_name} between "
            f"{group_a_name} and {group_b_name}."
        )

    else:

        print(
            "\nResult: Not Statistically Significant"
        )

        print(
            f"There is not enough evidence to conclude "
            f"that {metric_name} differs significantly "
            f"between {group_a_name} and {group_b_name}."
        )

    # ---------------------------------
    # EFFECT SIZE INTERPRETATION
    # ---------------------------------

    absolute_d = abs(cohens_d)

    if absolute_d < 0.2:
        effect_description = "Negligible"

    elif absolute_d < 0.5:
        effect_description = "Small"

    elif absolute_d < 0.8:
        effect_description = "Medium"

    else:
        effect_description = "Large"

    print(
        f"Effect Size: {effect_description}"
    )


# ---------------------------------
# MAIN STATISTICAL ANALYSIS
# ---------------------------------

def perform_ab_tests():

    # Load analytics-ready dataset
    df = pd.read_csv(INPUT_PATH)


    # =================================
    # TEST 1
    # MEDIA VS NO MEDIA
    # =================================

    media_group = df[
        df["has_media"].str.lower() == "yes"
    ]["engagement_rate"]

    no_media_group = df[
        df["has_media"].str.lower() == "no"
    ]["engagement_rate"]

    run_ab_test(
        media_group,
        no_media_group,
        "Media",
        "No Media"
    )


    # =================================
    # TEST 2
    # MORNING VS EVENING
    # =================================

    morning_group = df[
        df["time_period"] == "Morning"
    ]["engagement_rate"]

    evening_group = df[
        df["time_period"] == "Evening"
    ]["engagement_rate"]

    run_ab_test(
        morning_group,
        evening_group,
        "Morning",
        "Evening"
    )


    # =================================
    # TEST 3
    # VIDEO VS TEXT
    # =================================

    video_group = df[
        df["post_type"] == "Video"
    ]["engagement_rate"]

    text_group = df[
        df["post_type"] == "Text"
    ]["engagement_rate"]

    run_ab_test(
        video_group,
        text_group,
        "Video",
        "Text"
    )


    # =================================
    # TEST 4
    # SHORT VS VERY LONG
    # =================================

    short_group = df[
        df["content_length_category"] == "Short"
    ]["engagement_rate"]

    very_long_group = df[
        df["content_length_category"] == "Very Long"
    ]["engagement_rate"]

    run_ab_test(
        short_group,
        very_long_group,
        "Short",
        "Very Long"
    )


# ---------------------------------
# RUN PROGRAM
# ---------------------------------

if __name__ == "__main__":
    perform_ab_tests()