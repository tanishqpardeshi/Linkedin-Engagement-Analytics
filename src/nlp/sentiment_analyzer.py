import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

input_path = "data/raw/comments.csv"
output_path = "data/processed/comment_sentiment_analysis.csv"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(input_path)

df["text"] = df["text"].fillna("").astype(str)

df = df[df["text"].str.strip() != ""].copy()

df = df.drop(
    columns=["Unnamed: 0", "threadUrn"],
    errors="ignore"
)


# --------------------------------------------------
# Initialize VADER
# --------------------------------------------------

vader = SentimentIntensityAnalyzer()


# --------------------------------------------------
# TextBlob + VADER analysis
# --------------------------------------------------

def analyze_text(text):

    textblob = TextBlob(text)

    vader_scores = vader.polarity_scores(text)

    return pd.Series({
        "textblob_polarity": textblob.sentiment.polarity,
        "textblob_subjectivity": textblob.sentiment.subjectivity,
        "vader_positive": vader_scores["pos"],
        "vader_negative": vader_scores["neg"],
        "vader_neutral": vader_scores["neu"],
        "vader_compound": vader_scores["compound"]
    })


scores = df["text"].apply(analyze_text)

df = pd.concat([df, scores], axis=1)


# --------------------------------------------------
# Final sentiment classification
# --------------------------------------------------

def classify_sentiment(row):

    vader_score = row["vader_compound"]
    textblob_score = row["textblob_polarity"]

    # Strong VADER sentiment
    if vader_score >= 0.05:
        vader_sentiment = "Positive"

    elif vader_score <= -0.05:
        vader_sentiment = "Negative"

    else:
        vader_sentiment = "Neutral"

    # If both models agree, use their result.
    if vader_sentiment == "Positive" and textblob_score > 0.1:
        return "Positive"

    if vader_sentiment == "Negative" and textblob_score < -0.1:
        return "Negative"

    # When models disagree, give VADER priority
    # because the data consists of social-media comments.
    return vader_sentiment


df["sentiment"] = df.apply(
    classify_sentiment,
    axis=1
)


# --------------------------------------------------
# Relatable / Neutral classification
# --------------------------------------------------

def classify_relatable(row):

    text = row["text"].lower()

    subjectivity = row["textblob_subjectivity"]

    personal_indicators = [
        "i ",
        "i'm",
        "i’ve",
        "i've",
        "me ",
        "my ",
        "we ",
        "our ",
        "us ",
        "personally",
        "in my",
        "for me",
        "my experience",
        "i agree",
        "i feel",
        "i think",
        "i believe",
        "i have",
        "i had",
        "i've been",
        "we have"
    ]

    has_personal_reference = any(
        indicator in text
        for indicator in personal_indicators
    )

    # Personal experience/opinion
    if has_personal_reference and subjectivity >= 0.25:
        return "Relatable"

    # Strongly subjective or emotional response
    if (
        abs(row["vader_compound"]) >= 0.5
        and subjectivity >= 0.40
    ):
        return "Relatable"

    return "Neutral"


df["engagement_type"] = df.apply(
    classify_relatable,
    axis=1
)


# --------------------------------------------------
# Save processed dataset
# --------------------------------------------------

df.to_csv(output_path, index=False)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("=" * 70)
print("AUDIENCE SENTIMENT ANALYZER")
print("=" * 70)

print("\nTotal comments analyzed:")
print(len(df))

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nSentiment percentages:")
print(
    (df["sentiment"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nRelatable / Neutral distribution:")
print(df["engagement_type"].value_counts())

print("\nRelatable / Neutral percentages:")
print(
    (df["engagement_type"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nAverage TextBlob polarity:")
print(round(df["textblob_polarity"].mean(), 4))

print("\nAverage TextBlob subjectivity:")
print(round(df["textblob_subjectivity"].mean(), 4))

print("\nAverage VADER compound score:")
print(round(df["vader_compound"].mean(), 4))


# --------------------------------------------------
# Detailed classification
# --------------------------------------------------

print("\n" + "=" * 70)
print("COMMENT CLASSIFICATIONS")
print("=" * 70)

for i, row in df.iterrows():

    print(f"\nComment {i + 1}:")
    print(
        row["text"][:250]
        .replace("\n", " ")
    )

    print(
        "TextBlob polarity:",
        round(row["textblob_polarity"], 3),
        "| TextBlob subjectivity:",
        round(row["textblob_subjectivity"], 3)
    )

    print(
        "VADER compound:",
        round(row["vader_compound"], 3)
    )

    print(
        "Sentiment:",
        row["sentiment"],
        "| Engagement Type:",
        row["engagement_type"]
    )


print("\n" + "=" * 70)
print("SENTIMENT ANALYSIS COMPLETE")
print("=" * 70)