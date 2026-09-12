import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LinkedIn Engagement Analytics",
    page_icon="LI",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
RAW = BASE_DIR / "data" / "raw"

POST_FILE = PROCESSED / "post_engagement_features.csv"
VIRALITY_FILE = PROCESSED / "virality_dataset.csv"
RECOMMENDATION_FILE = PROCESSED / "engagement_recommendations.csv"
FORECAST_FILE = PROCESSED / "trend_forecast.csv"
COMMENTS_FILE = RAW / "comments.csv"


# ============================================================
# UI STYLE
# ============================================================

st.markdown("""
<style>
.block-container {
    padding-top: 1.35rem;
    padding-bottom: 2.5rem;
    max-width: 1500px;
}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 14px;
    padding: 14px 16px;
    background: rgba(128,128,128,.035);
}
[data-testid="stMetricLabel"] {
    font-weight: 600;
}
.small-muted {
    color: #667085;
    font-size: .9rem;
}
.section-note {
    color: #667085;
    font-size: .88rem;
    margin-top: -8px;
    margin-bottom: 14px;
}
.module-card {
    border: 1px solid rgba(128,128,128,.18);
    border-radius: 14px;
    padding: 16px;
    min-height: 118px;
    background: rgba(128,128,128,.025);
}
.module-card h4 {
    margin: 0 0 6px 0;
}
.module-card p {
    margin: 0;
    color: #667085;
    font-size: .88rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA
# ============================================================

@st.cache_data
def load_posts():
    df = pd.read_csv(POST_FILE)
    df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
    return df


@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def analyze_comments(raw):
    if raw is None or "text" not in raw.columns:
        return None

    df = raw.copy()
    df["text"] = df["text"].fillna("").astype(str)

    vader = SentimentIntensityAnalyzer()

    positive_words = {
        "amazing","great","excellent","love","helpful","interesting",
        "exciting","kudos","good","thanks","thank","brilliant","awesome"
    }
    negative_words = {
        "bad","wrong","hate","terrible","awful","scam","fraud",
        "disappointing","disappointed","problem","problems","concern",
        "concerns","demise","misleading","misstatement","discrepancy",
        "discrepancies","unacceptable","never","cannot"
    }

    sentiments, polarity, subjectivity, vader_scores, types = [], [], [], [], []

    for text in df["text"]:
        lower = text.lower()
        blob = TextBlob(text)
        score = vader.polarity_scores(text)["compound"]

        pos_hits = sum(w in lower for w in positive_words)
        neg_hits = sum(w in lower for w in negative_words)

        if neg_hits >= 2 and neg_hits > pos_hits:
            sentiment = "Negative"
        elif pos_hits >= 2 and pos_hits > neg_hits:
            sentiment = "Positive"
        elif score >= 0.05:
            sentiment = "Positive"
        elif score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        if "?" in text or any(
            x in lower for x in
            ["please provide", "please explain", "why ", "how ",
             "can you", "could you", "what "]
        ):
            engagement_type = "Question / Concern"
        elif sentiment == "Negative":
            engagement_type = "Concern / Criticism"
        elif sentiment == "Positive":
            engagement_type = "Positive Engagement"
        else:
            engagement_type = "Neutral"

        sentiments.append(sentiment)
        polarity.append(blob.sentiment.polarity)
        subjectivity.append(blob.sentiment.subjectivity)
        vader_scores.append(score)
        types.append(engagement_type)

    df["sentiment"] = sentiments
    df["polarity"] = polarity
    df["subjectivity"] = subjectivity
    df["vader_compound"] = vader_scores
    df["engagement_type"] = types

    return df


@st.cache_data
def ab_tests(df):
    tests = [
        ("Media vs No Media", df["has_media"] == True, df["has_media"] == False),
        ("Morning vs Evening", df["time_period"] == "Morning", df["time_period"] == "Evening"),
        ("Video vs Text", df["post_type"] == "Video", df["post_type"] == "Text"),
        ("Short vs Very Long",
         df["content_length_category"] == "Short",
         df["content_length_category"] == "Very Long"),
    ]

    rows = []

    for name, a_mask, b_mask in tests:
        a = df.loc[a_mask, "engagement_rate"].dropna()
        b = df.loc[b_mask, "engagement_rate"].dropna()

        if len(a) < 2 or len(b) < 2:
            continue

        _, p = ttest_ind(a, b, equal_var=False)

        pooled = np.sqrt(
            ((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
            / (len(a)+len(b)-2)
        )
        d = (a.mean()-b.mean()) / pooled if pooled else 0

        rows.append({
            "Comparison": name,
            "Group A n": len(a),
            "Group B n": len(b),
            "Group A Avg %": a.mean(),
            "Group B Avg %": b.mean(),
            "Difference (pp)": a.mean()-b.mean(),
            "p-value": p,
            "Cohen's d": d,
            "Significant": "Yes" if p < .05 else "No",
        })

    return pd.DataFrame(rows)


@st.cache_data
def train_virality(df):
    features = [
        "post_type", "content_length", "hashtags_count",
        "has_media", "media_type", "post_hour", "day_of_week"
    ]

    if df is None or not all(c in df.columns for c in features + ["virality_label"]):
        return None

    X = df[features]
    y = df["virality_label"]

    numeric = ["content_length", "hashtags_count", "post_hour"]
    categorical = ["post_type", "has_media", "media_type", "day_of_week"]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
    ])

    model = Pipeline([
        ("preprocessor", pre),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
    }

    return metrics, confusion_matrix(y_test, pred)


posts = load_posts()
comments = analyze_comments(load_csv(COMMENTS_FILE))
virality = load_csv(VIRALITY_FILE)
recommendations = load_csv(RECOMMENDATION_FILE)
forecast = load_csv(FORECAST_FILE)

if forecast is not None and "date" in forecast.columns:
    forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("LinkedIn Analytics")
st.sidebar.caption("Data-Driven Social Engagement Platform")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Content Performance",
        "Virality Prediction",
        "Audience Sentiment",
        "A/B Testing",
        "Recommendations",
        "Growth Visualization",
        "Trend Forecasting",
    ]
)

st.sidebar.divider()
st.sidebar.caption(f"{len(posts):,} posts analysed")

if comments is not None:
    st.sidebar.caption(f"{len(comments):,} comments analysed")

st.sidebar.divider()
st.sidebar.caption(
    "Historical results are observational unless explicitly "
    "identified as model predictions."
)


# ============================================================
# HEADER
# ============================================================

st.title("LinkedIn Engagement Analytics")
st.markdown(
    '<div class="small-muted">'
    "A unified analytics dashboard for measuring, predicting, "
    "and optimizing LinkedIn engagement."
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Executive Overview")
    st.markdown(
        '<div class="section-note">A high-level view of historical performance, model outputs, and actionable patterns.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Posts Analysed", f"{len(posts):,}")
    c2.metric("Total Impressions", f"{posts.impressions.sum():,.0f}")
    c3.metric("Total Engagement", f"{posts.total_engagement.sum():,.0f}")
    c4.metric("Avg. Engagement Rate", f"{posts.engagement_rate.mean():.2f}%")
    c5.metric("Avg. Virality Score", f"{posts.virality_score.mean():.2f}")

    st.divider()

    daily = (
        posts.groupby("post_date")["engagement_rate"]
        .mean()
        .sort_index()
    )

    st.subheader("Engagement Over Time")
    st.markdown('<div class="section-note">Daily average engagement rate across the available post history.</div>', unsafe_allow_html=True)
    st.line_chart(daily, height=350)

    st.divider()

    st.subheader("Historical Performance Signals")
    st.markdown('<div class="section-note">These are historical associations in the available dataset, not causal effects.</div>', unsafe_allow_html=True)

    best_type = posts.groupby("post_type").engagement_rate.mean().idxmax()
    best_day = posts.groupby("day_of_week").engagement_rate.mean().idxmax()
    best_hour = posts.groupby("post_hour").engagement_rate.mean().idxmax()

    a, b, c = st.columns(3)
    a.info(f"**Highest historical post type**\n\n{best_type}")
    b.info(f"**Highest historical day**\n\n{best_day}")
    c.info(f"**Highest historical posting hour**\n\n{int(best_hour)}:00")

    if forecast is not None and "forecast_engagement" in forecast.columns:
        st.divider()
        st.subheader("Forecast Snapshot")
        first = forecast["forecast_engagement"].iloc[0]
        last = forecast["forecast_engagement"].iloc[-1]

        a, b, c = st.columns(3)
        a.metric("Forecast Start", f"{first:.2f}%")
        b.metric("Forecast End", f"{last:.2f}%")
        c.metric("14-Day Change", f"{last-first:+.2f} pp")

    st.divider()
    st.subheader("Project Module Status")
    status_cols = st.columns(4)
    modules = [
        ("Content Performance", "Historical post-level engagement analysis."),
        ("Virality Prediction", "Logistic-regression baseline using pre-publication metadata."),
        ("Audience Sentiment", "Exploratory NLP analysis of the available comment sample."),
        ("A/B Testing", "Welch t-tests and effect-size comparisons on historical data."),
        ("Recommendations", "Historical group-average based optimization signals."),
        ("Growth Visualization", "Interactive visual analysis of performance and virality trends."),
        ("Trend Forecasting", "Short-term baseline forecast of average engagement."),
        ("Data Layer", "300-post analytics dataset plus 49 available comments."),
    ]
    for i, (name, desc) in enumerate(modules):
        with status_cols[i % 4]:
            st.markdown(f'<div class="module-card"><h4>{name}</h4><p>{desc}</p></div>', unsafe_allow_html=True)
        if i % 4 == 3 and i != len(modules) - 1:
            st.write("")

    st.caption(
        "Model outputs are experimental where noted. Historical metrics and recommendations are observational. "
        "The current datasets do not provide follower-growth history, saves, or external hashtag-trend data."
    )


# ============================================================
# CONTENT PERFORMANCE
# ============================================================

elif page == "Content Performance":

    st.header("Content Performance Tracker")
    st.caption(
        "Interactive exploration of historical engagement patterns."
    )

    c1, c2, c3 = st.columns(3)

    types = c1.multiselect(
        "Post type",
        sorted(posts.post_type.dropna().unique()),
        default=sorted(posts.post_type.dropna().unique())
    )

    days = c2.multiselect(
        "Day",
        sorted(posts.day_of_week.dropna().unique()),
        default=sorted(posts.day_of_week.dropna().unique())
    )

    media = c3.multiselect(
        "Media type",
        sorted(posts.media_type.fillna("No Media").unique()),
        default=sorted(posts.media_type.fillna("No Media").unique())
    )

    filtered = posts[
        posts.post_type.isin(types)
        & posts.day_of_week.isin(days)
        & posts.media_type.fillna("No Media").isin(media)
    ].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts", f"{len(filtered):,}")
    c2.metric("Avg Engagement", f"{filtered.engagement_rate.mean():.2f}%")
    c3.metric("Avg Impressions", f"{filtered.impressions.mean():,.0f}")
    c4.metric("Avg Virality", f"{filtered.virality_score.mean():.2f}")

    st.divider()

    a, b = st.columns(2)

    with a:
        st.subheader("Engagement by Post Type")
        chart = filtered.groupby("post_type").engagement_rate.mean().sort_values()
        st.bar_chart(chart)

    with b:
        st.subheader("Engagement by Day")
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        chart = (
            filtered.groupby("day_of_week").engagement_rate.mean()
            .reindex([x for x in day_order if x in filtered.day_of_week.unique()])
        )
        st.bar_chart(chart)

    a, b = st.columns(2)

    with a:
        st.subheader("Engagement by Posting Hour")
        chart = filtered.groupby("post_hour").engagement_rate.mean().sort_index()
        st.line_chart(chart)

    with b:
        st.subheader("Virality by Post Type")
        chart = filtered.groupby("post_type").virality_score.mean().sort_values()
        st.bar_chart(chart)

    st.divider()

    st.subheader("Top Performing Posts")

    n = st.slider("Number of posts to display", 5, 20, 10)

    cols = [
        "post_id", "post_date", "post_type", "impressions",
        "likes", "comments", "shares",
        "engagement_rate", "virality_score"
    ]

    st.dataframe(
        filtered[cols]
        .sort_values("engagement_rate", ascending=False)
        .head(n),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# VIRALITY
# ============================================================

elif page == "Virality Prediction":

    st.header("Virality Prediction Engine")

    st.caption(
        "Virality Score is a project-defined proxy that weights comments and shares more heavily than likes; it is not a platform-provided virality label."
    )

    a, b, c, d = st.columns(4)

    a.metric("Mean Score", f"{posts.virality_score.mean():.2f}")
    b.metric("Median", f"{posts.virality_score.median():.2f}")
    c.metric("75th Percentile", f"{posts.virality_score.quantile(.75):.2f}")
    d.metric("Maximum", f"{posts.virality_score.max():.2f}")

    st.divider()

    a, b = st.columns(2)

    with a:
        st.subheader("Virality Distribution")
        hist = pd.cut(posts.virality_score, bins=10).value_counts().sort_index()
        hist.index = hist.index.astype(str)
        st.bar_chart(hist)

    with b:
        st.subheader("Average Virality by Post Type")
        st.bar_chart(
            posts.groupby("post_type").virality_score.mean().sort_values()
        )

    if virality is not None:

        st.divider()
        st.subheader("Prediction Model")

        result = train_virality(virality)

        if result is not None:

            metrics, cm = result

            a, b, c, d, e = st.columns(5)

            a.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
            b.metric("Precision", f"{metrics['Precision']:.2f}")
            c.metric("Recall", f"{metrics['Recall']:.2f}")
            d.metric("F1", f"{metrics['F1 Score']:.2f}")
            e.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2f}")

            st.subheader("Confusion Matrix")

            cm_df = pd.DataFrame(
                cm,
                index=["Actual Non-Viral", "Actual Viral"],
                columns=["Predicted Non-Viral", "Predicted Viral"]
            )

            st.dataframe(cm_df, use_container_width=True)

            st.warning(
                "Baseline interpretation: the current metadata-only model shows limited predictive signal. "
                "Use it as an experimental proof-of-concept, not as a production or guaranteed viral-content predictor."
            )

    else:
        st.info("Run the virality target-generation script to enable the model section.")


# ============================================================
# SENTIMENT
# ============================================================

elif page == "Audience Sentiment":

    st.header("Audience Sentiment Analyzer")

    st.caption(
        "Exploratory NLP analysis of the 49 available LinkedIn comments using lexicon-based TextBlob and VADER signals."
    )

    if comments is None:

        st.error("comments.csv was not found.")

    else:

        counts = comments.sentiment.value_counts()

        a, b, c = st.columns(3)
        a.metric("Positive", int(counts.get("Positive", 0)))
        b.metric("Neutral", int(counts.get("Neutral", 0)))
        c.metric("Negative", int(counts.get("Negative", 0)))

        st.divider()

        a, b = st.columns(2)

        with a:
            st.subheader("Sentiment Distribution")
            st.bar_chart(
                counts.reindex(
                    ["Positive", "Neutral", "Negative"]
                ).fillna(0)
            )

        with b:
            st.subheader("NLP Metrics")
            st.metric("Avg. Polarity", f"{comments.polarity.mean():.4f}")
            st.metric("Avg. Subjectivity", f"{comments.subjectivity.mean():.4f}")
            st.metric("Avg. VADER Compound", f"{comments.vader_compound.mean():.4f}")

        st.divider()

        sentiment_filter = st.selectbox(
            "Filter comments",
            ["All", "Positive", "Neutral", "Negative"]
        )

        view = comments
        if sentiment_filter != "All":
            view = comments[comments.sentiment == sentiment_filter]

        cols = [
            x for x in [
                "createdAtString", "text", "sentiment",
                "polarity", "subjectivity",
                "vader_compound", "engagement_type"
            ] if x in view.columns
        ]

        st.dataframe(
            view[cols],
            use_container_width=True,
            hide_index=True
        )

        st.warning(
            "Dataset limitation: the 49-comment sample is small and has no ground-truth sentiment labels. "
            "The current TextBlob/VADER approach is therefore an exploratory sentiment analysis, not a validated supervised classifier. "
            "Results may miss context, sarcasm, mixed sentiment, or domain-specific meaning."
        )


# ============================================================
# A/B TESTING
# ============================================================

elif page == "A/B Testing":

    st.header("A/B Testing & Statistical Comparison")

    st.caption(
        "Welch independent-samples t-tests and Cohen's d on historical observational data. These comparisons estimate association, not causation."
    )

    results = ab_tests(posts)

    st.dataframe(
        results.style.format({
            "Group A Avg %": "{:.2f}",
            "Group B Avg %": "{:.2f}",
            "Difference (pp)": "{:.2f}",
            "p-value": "{:.4f}",
            "Cohen's d": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    for _, row in results.iterrows():

        if row["Significant"] == "Yes":
            st.success(
                f"{row['Comparison']}: statistically significant at α = 0.05."
            )
        else:
            st.info(
                f"{row['Comparison']}: no statistically significant difference "
                "at α = 0.05."
            )

    st.warning(
        "These are statistical comparisons, not controlled A/B experiments. "
        "They do not establish causality."
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "Recommendations":

    st.header("Engagement Optimization Recommender")

    st.caption(
        "Recommendations rank historically stronger groups using average engagement rate; they are decision-support signals, not guaranteed outcomes."
    )

    overall = posts.engagement_rate.mean()

    # Recalculate directly so the page remains useful even if the
    # recommendation CSV has only recommendation/value columns.
    groups = [
        ("Post Type", "post_type"),
        ("Day of Week", "day_of_week"),
        ("Posting Hour", "post_hour"),
        ("Time Period", "time_period"),
        ("Content Length", "content_length_category"),
        ("Media Type", "media_type"),
    ]

    rows = []

    for label, column in groups:
        temp = posts.copy()
        temp[column] = temp[column].fillna("No Media")

        stats = (
            temp.groupby(column)
            .agg(
                posts=("post_id", "count"),
                avg_engagement=("engagement_rate", "mean")
            )
            .sort_values("avg_engagement", ascending=False)
        )

        best = stats.iloc[0]

        rows.append({
            "Factor": label,
            "Recommended Value": stats.index[0],
            "Posts": int(best["posts"]),
            "Historical Engagement %": best["avg_engagement"],
            "Difference vs Overall (pp)": best["avg_engagement"] - overall
        })

    # Media usage is boolean and should be presented cleanly.
    media_stats = (
        posts.groupby("has_media")
        .agg(
            posts=("post_id", "count"),
            avg_engagement=("engagement_rate", "mean")
        )
        .sort_values("avg_engagement", ascending=False)
    )

    media_best = media_stats.iloc[0]

    rows.append({
        "Factor": "Media Usage",
        "Recommended Value": "Use Media" if bool(media_stats.index[0]) else "No Media",
        "Posts": int(media_best["posts"]),
        "Historical Engagement %": media_best["avg_engagement"],
        "Difference vs Overall (pp)": media_best["avg_engagement"] - overall
    })

    # Hashtags: require at least five observations.
    hashtag_stats = (
        posts.groupby("hashtags_count")
        .agg(
            posts=("post_id", "count"),
            avg_engagement=("engagement_rate", "mean")
        )
    )

    hashtag_stats = hashtag_stats[hashtag_stats.posts >= 5]

    if not hashtag_stats.empty:
        best = hashtag_stats.sort_values("avg_engagement", ascending=False).iloc[0]
        rows.append({
            "Factor": "Hashtag Count",
            "Recommended Value": int(hashtag_stats.sort_values(
                "avg_engagement", ascending=False
            ).index[0]),
            "Posts": int(best["posts"]),
            "Historical Engagement %": best["avg_engagement"],
            "Difference vs Overall (pp)": best["avg_engagement"] - overall
        })

    result = pd.DataFrame(rows)

    a, b, c = st.columns(3)

    top = result.sort_values(
        "Historical Engagement %",
        ascending=False
    ).iloc[0]

    a.metric("Overall Engagement", f"{overall:.2f}%")
    b.metric("Strongest Historical Factor", top["Factor"])
    c.metric("Best Observed Value", str(top["Recommended Value"]))

    st.divider()

    st.subheader("Recommended Strategy")

    st.dataframe(
        result.style.format({
            "Historical Engagement %": "{:.2f}",
            "Difference vs Overall (pp)": "{:+.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The recommender identifies historically strong combinations. "
        "It does not guarantee future performance because the underlying "
        "dataset is observational."
    )


# ============================================================
# GROWTH VISUALIZATION
# ============================================================

elif page == "Growth Visualization":

    st.header("Growth Visualization")
    st.markdown(
        '<div class="section-note">Interactive visual analysis of engagement, impressions, posting patterns, and virality across the available history.</div>',
        unsafe_allow_html=True
    )

    daily = posts.groupby("post_date").agg(
        engagement_rate=("engagement_rate", "mean"),
        impressions=("impressions", "sum"),
        total_engagement=("total_engagement", "sum")
    ).sort_index()

    a, b, c, d = st.columns(4)
    a.metric("Historical Dates", f"{posts.post_date.nunique():,}")
    b.metric("Peak Daily Engagement", f"{daily.engagement_rate.max():.2f}%")
    c.metric("Peak Daily Impressions", f"{daily.impressions.max():,.0f}")
    d.metric("Peak Daily Engagement Volume", f"{daily.total_engagement.max():,.0f}")

    st.divider()

    st.subheader("Engagement Trend")
    st.line_chart(daily["engagement_rate"], height=300)

    st.subheader("Impressions Trend")
    st.line_chart(daily["impressions"], height=300)

    st.divider()

    a, b = st.columns(2)
    with a:
        st.subheader("Engagement by Post Type")
        st.bar_chart(
            posts.groupby("post_type").engagement_rate.mean().sort_values(),
            height=300
        )
    with b:
        st.subheader("Engagement by Day of Week")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_stats = posts.groupby("day_of_week").engagement_rate.mean().reindex(day_order).dropna()
        st.bar_chart(day_stats, height=300)

    a, b = st.columns(2)
    with a:
        st.subheader("Engagement by Posting Hour")
        st.bar_chart(posts.groupby("post_hour").engagement_rate.mean().sort_index(), height=300)
    with b:
        st.subheader("Virality Distribution")
        hist = pd.cut(posts.virality_score, bins=10).value_counts().sort_index()
        hist.index = hist.index.astype(str)
        st.bar_chart(hist, height=300)

    st.info(
        "Growth here refers to observed engagement and reach patterns in the available post history. "
        "The current dataset does not contain a follower-count time series, so follower growth cannot be measured directly."
    )


# ============================================================
# TREND FORECAST
# ============================================================

elif page == "Trend Forecasting":

    st.header("Trend Forecasting")

    if forecast is None:

        st.error("trend_forecast.csv was not found.")

    else:

        forecast = forecast.copy()

        first = forecast.forecast_engagement.iloc[0]
        last = forecast.forecast_engagement.iloc[-1]

        mae = forecast.model_mae.iloc[0]
        rmse = forecast.model_rmse.iloc[0]

        trend = (
            forecast.trend_direction.iloc[0]
            if "trend_direction" in forecast.columns
            else "Stable"
        )

        a, b, c, d = st.columns(4)

        a.metric("Historical Dates", f"{posts.post_date.nunique():,}")
        b.metric("Trend", str(trend))
        c.metric("MAE", f"{mae:.4f}")
        d.metric("RMSE", f"{rmse:.4f}")

        st.divider()

        st.subheader("Historical Engagement + Forecast")

        historical = (
            posts.groupby("post_date")["engagement_rate"]
            .mean()
            .sort_index()
            .rename("Historical")
        )

        predicted = (
            forecast.set_index("date")["forecast_engagement"]
            .rename("14-Day Forecast")
        )

        combined = pd.concat([historical, predicted], axis=1)

        st.line_chart(combined, height=400)

        st.divider()

        a, b, c = st.columns(3)
        a.metric("Forecast Start", f"{first:.2f}%")
        b.metric("Forecast End", f"{last:.2f}%")
        c.metric("Forecast Change", f"{last-first:+.2f} pp")

        st.subheader("14-Day Forecast")

        table = forecast[["date", "forecast_engagement"]].copy()
        table["date"] = table["date"].dt.strftime("%d %b %Y")
        table["forecast_engagement"] = table["forecast_engagement"].round(2)

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )

        st.warning(
            "This is a short-term baseline forecast. Its purpose is "
            "directional trend analysis, not precise future prediction."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "LinkedIn Engagement Analytics | Python • Pandas • Scikit-learn • NLP • Streamlit"
)
