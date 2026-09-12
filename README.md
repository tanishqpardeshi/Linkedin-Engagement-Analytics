# LinkedIn Engagement Analytics

A data-driven analytics dashboard for analyzing and optimizing LinkedIn content performance.

## Live Demo

**Streamlit App:**  
https://linkedin-engagement-analytics.streamlit.app/

## Project Overview

LinkedIn Engagement Analytics is a data-driven social engagement analysis system designed to help understand how LinkedIn content performs and identify patterns that can support better content strategies.

The project combines data analytics, machine learning, natural language processing, statistical analysis, recommendation techniques, and interactive visualization through Streamlit.

## Key Modules

### 1. Content Performance Tracker
Analyzes historical LinkedIn posts using metrics such as:

- Impressions
- Likes
- Comments
- Shares
- Engagement Rate
- Total Engagement
- Virality Score

Content performance can be explored by post type, day of week, posting hour, media usage, and other characteristics.

### 2. Virality Prediction
Uses a machine-learning classification model to estimate whether a post belongs to the higher-performing segment based on pre-publication characteristics such as:

- Post type
- Content length
- Hashtag count
- Media usage
- Media type
- Posting hour
- Day of week

The current implementation is presented as an experimental baseline because the available dataset provides limited predictive signal.

### 3. Audience Sentiment Analyzer
Analyzes LinkedIn comments using NLP techniques to estimate:

- Positive sentiment
- Neutral sentiment
- Negative sentiment
- Relatability
- Polarity
- Subjectivity

The sentiment module is exploratory and is not presented as a fully validated production sentiment model.

### 4. A/B Statistical Testing
Performs statistical comparisons between different content strategies using:

- Welch's independent t-test
- Cohen's d
- Statistical significance testing

Examples include comparisons between media/no-media posts, posting periods, post types, and content lengths.

Because the available data is observational rather than a controlled experiment, these results should be interpreted as statistical comparisons rather than causal A/B experiments.

### 5. Engagement Optimization Recommender
Generates data-driven posting recommendations based on historical performance patterns, including:

- Recommended post type
- Recommended posting day
- Recommended posting hour
- Media usage
- Media type
- Content length
- Hashtag count

These recommendations represent historical patterns and should not be interpreted as guaranteed causal effects.

### 6. Growth Visualization
Provides visual analysis of:

- Engagement trends
- Impressions trends
- Engagement by post type
- Engagement by day
- Engagement by posting hour
- Virality distribution
- Top-performing posts

### 7. Trend Forecasting
Uses historical engagement data to estimate the short-term engagement trend and generate a 14-day forecast.

The forecasting module reports model error metrics including:

- MAE
- RMSE

## Technology Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **SciPy**
- **TextBlob**
- **VADER Sentiment**
- **Matplotlib**
- **Streamlit**
- **OpenPyXL**

## Project Structure

```text
Linkedin-Engagement-Analytics/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── analytics/
│   ├── data/
│   ├── models/
│   ├── nlp/
│   └── recommender/
│
├── requirements.txt
└── README.md
