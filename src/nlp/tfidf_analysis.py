import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# LOAD DATA
# ============================================================

input_path = "data/processed/cleaned_profile_engagement.csv"

df = pd.read_csv(input_path)

print("=" * 70)
print("TF-IDF CONTENT ANALYSIS")
print("=" * 70)


# ============================================================
# PREPARE TEXT
# ============================================================

df["text"] = df["text"].fillna("").astype(str)

# Remove empty posts
df = df[
    df["text"].str.strip() != ""
].copy()

print("\nPosts with usable text:", len(df))


# ============================================================
# TF-IDF VECTORIZER
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=50,
    ngram_range=(1, 2),
    min_df=2
)


# ============================================================
# TRANSFORM TEXT
# ============================================================

tfidf_matrix = vectorizer.fit_transform(
    df["text"]
)

feature_names = vectorizer.get_feature_names_out()


print("\nTF-IDF matrix shape:")
print(tfidf_matrix.shape)

print("\nNumber of TF-IDF features:")
print(len(feature_names))


# ============================================================
# IDENTIFY IMPORTANT TERMS
# ============================================================

tfidf_scores = (
    tfidf_matrix
    .mean(axis=0)
    .A1
)

terms = pd.DataFrame(
    {
        "term": feature_names,
        "average_tfidf": tfidf_scores
    }
)

terms = terms.sort_values(
    "average_tfidf",
    ascending=False
)


print("\nTop TF-IDF terms:")
print("-" * 70)

print(
    terms.head(20)
    .to_string(index=False)
)


# ============================================================
# ADD ENGAGEMENT
# ============================================================

tfidf_dense = tfidf_matrix.toarray()

tfidf_df = pd.DataFrame(
    tfidf_dense,
    columns=feature_names,
    index=df.index
)


# ============================================================
# CORRELATE TERMS WITH ENGAGEMENT
# ============================================================

engagement = df[
    [
        "totalReactionCount",
        "commentsCount",
        "repostsCount"
    ]
]

correlations = []

for term in feature_names:

    term_values = tfidf_df[term]

    for metric in engagement.columns:

        correlation = term_values.corr(
            engagement[metric]
        )

        correlations.append(
            {
                "term": term,
                "metric": metric,
                "correlation": correlation
            }
        )


correlation_df = pd.DataFrame(
    correlations
)


# ============================================================
# TOP POSITIVE TERMS
# ============================================================

print("\n\nTerms positively associated with total reactions:")
print("-" * 70)

reaction_corr = (
    correlation_df[
        correlation_df["metric"]
        == "totalReactionCount"
    ]
    .sort_values(
        "correlation",
        ascending=False
    )
)

print(
    reaction_corr
    .head(15)
    .to_string(index=False)
)


# ============================================================
# TOP NEGATIVE TERMS
# ============================================================

print("\n\nTerms negatively associated with total reactions:")
print("-" * 70)

print(
    reaction_corr
    .tail(15)
    .sort_values(
        "correlation"
    )
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("TF-IDF ANALYSIS COMPLETE")
print("=" * 70)