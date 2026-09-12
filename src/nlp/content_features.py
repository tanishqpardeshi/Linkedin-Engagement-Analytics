import pandas as pd
import re


# ============================================================
# LOAD CLEANED DATA
# ============================================================

input_path = "data/processed/cleaned_profile_engagement.csv"

df = pd.read_csv(input_path)

print("=" * 70)
print("LINKEDIN CONTENT FEATURE EXTRACTION")
print("=" * 70)


# ============================================================
# HANDLE MISSING TEXT
# ============================================================

df["text"] = df["text"].fillna("")


# ============================================================
# TEXT FEATURE FUNCTIONS
# ============================================================

def count_words(text):
    return len(
        re.findall(r"\b\w+\b", text)
    )


def count_sentences(text):
    sentences = re.split(
        r"[.!?]+",
        text
    )

    return len(
        [s for s in sentences if s.strip()]
    )


def count_hashtags(text):
    return len(
        re.findall(
            r"#\w+",
            text
        )
    )


def count_urls(text):
    return len(
        re.findall(
            r"https?://\S+|www\.\S+",
            text
        )
    )


def count_questions(text):
    return text.count("?")


def count_exclamations(text):
    return text.count("!")


def average_word_length(text):
    words = re.findall(
        r"\b\w+\b",
        text
    )

    if not words:
        return 0

    return sum(
        len(word)
        for word in words
    ) / len(words)


# ============================================================
# EXTRACT FEATURES
# ============================================================

df["character_count"] = (
    df["text"]
    .str.len()
)

df["word_count"] = (
    df["text"]
    .apply(count_words)
)

df["sentence_count"] = (
    df["text"]
    .apply(count_sentences)
)

df["hashtag_count"] = (
    df["text"]
    .apply(count_hashtags)
)

df["url_count"] = (
    df["text"]
    .apply(count_urls)
)

df["question_count"] = (
    df["text"]
    .apply(count_questions)
)

df["exclamation_count"] = (
    df["text"]
    .apply(count_exclamations)
)

df["newline_count"] = (
    df["text"]
    .str.count("\n")
)

df["average_word_length"] = (
    df["text"]
    .apply(average_word_length)
)

df["has_article"] = (
    df["article"]
    .fillna("{}")
    .ne("{}")
    .astype(int)
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

new_features = [
    "character_count",
    "word_count",
    "sentence_count",
    "hashtag_count",
    "url_count",
    "question_count",
    "exclamation_count",
    "newline_count",
    "average_word_length",
    "has_article"
]

print("\nExtracted features:")
print("-" * 70)

print(
    df[new_features]
    .describe()
    .round(2)
)


# ============================================================
# CORRELATION WITH ENGAGEMENT
# ============================================================

print("\n\nFeature correlation with engagement:")
print("-" * 70)

correlation = (
    df[
        new_features + [
            "totalReactionCount",
            "commentsCount",
            "repostsCount"
        ]
    ]
    .corr(numeric_only=True)
)

print(
    correlation[
        [
            "totalReactionCount",
            "commentsCount",
            "repostsCount"
        ]
    ]
    .loc[new_features]
    .round(3)
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    "data/processed/"
    "profile_content_features.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\n\nSaved feature dataset to:")
print(output_path)


print("\n" + "=" * 70)
print("CONTENT FEATURE EXTRACTION COMPLETE")
print("=" * 70)