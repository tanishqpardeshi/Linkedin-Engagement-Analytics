import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_PATH = Path("data/processed/post_engagement_features.csv")
OUTPUT_PATH = Path("data/processed/trend_forecast.csv")


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

if df.empty:
    raise ValueError("The input dataset is empty.")


required_columns = [
    "post_date",
    "engagement_rate",
    "impressions",
    "total_engagement",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ---------------------------------------------------------
# Prepare date column
# ---------------------------------------------------------

df["post_date"] = pd.to_datetime(
    df["post_date"],
    errors="coerce"
)

df = df.dropna(
    subset=["post_date"]
)


# ---------------------------------------------------------
# Aggregate posts by date
# ---------------------------------------------------------

daily = (
    df.groupby("post_date")
    .agg(
        posts=("post_date", "count"),
        avg_engagement=("engagement_rate", "mean"),
        avg_impressions=("impressions", "mean"),
        total_engagement=("total_engagement", "sum"),
    )
    .reset_index()
    .sort_values("post_date")
)


if len(daily) < 10:
    raise ValueError(
        "Not enough historical dates for forecasting."
    )


# ---------------------------------------------------------
# Create time index
# ---------------------------------------------------------

daily["time_index"] = np.arange(
    len(daily)
)


# ---------------------------------------------------------
# Train / test split
# ---------------------------------------------------------

split_index = int(
    len(daily) * 0.80
)

train = daily.iloc[
    :split_index
].copy()

test = daily.iloc[
    split_index:
].copy()


# ---------------------------------------------------------
# Prepare training and testing data
# ---------------------------------------------------------

X_train = train[
    ["time_index"]
]

y_train = train[
    "avg_engagement"
]

X_test = test[
    ["time_index"]
]

y_test = test[
    "avg_engagement"
]


# ---------------------------------------------------------
# Train Linear Regression model
# ---------------------------------------------------------

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

test_predictions = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    test_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)


# ---------------------------------------------------------
# Determine historical trend
# ---------------------------------------------------------

slope = model.coef_[0]

if slope > 0.01:
    trend_direction = "Increasing"

elif slope < -0.01:
    trend_direction = "Decreasing"

else:
    trend_direction = "Stable"


# ---------------------------------------------------------
# Generate future dates
# ---------------------------------------------------------

forecast_days = 14

last_date = daily[
    "post_date"
].max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="D"
)


# ---------------------------------------------------------
# Generate future time indices
# ---------------------------------------------------------

future_indices = np.arange(
    len(daily),
    len(daily) + forecast_days
)


# ---------------------------------------------------------
# Prepare future data
#
# Keeping the "time_index" column ensures that
# the prediction data has the same feature name
# used during model training.
# ---------------------------------------------------------

future_X = pd.DataFrame(
    {
        "time_index": future_indices
    }
)


# ---------------------------------------------------------
# Generate forecast
# ---------------------------------------------------------

future_predictions = model.predict(
    future_X
)


# ---------------------------------------------------------
# Build forecast dataframe
# ---------------------------------------------------------

forecast = pd.DataFrame(
    {
        "date": future_dates,
        "forecast_engagement": future_predictions,
    }
)


# ---------------------------------------------------------
# Add forecasting metadata
# ---------------------------------------------------------

forecast["trend_direction"] = (
    trend_direction
)

forecast["model_mae"] = mae

forecast["model_rmse"] = rmse


# ---------------------------------------------------------
# Save forecast
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

forecast.to_csv(
    OUTPUT_PATH,
    index=False
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print()

print(
    "TREND FORECASTING"
)

print(
    "=" * 50
)

print(
    f"Historical dates analysed: "
    f"{len(daily)}"
)

print(
    f"Training dates: "
    f"{len(train)}"
)

print(
    f"Testing dates: "
    f"{len(test)}"
)

print(
    f"Model MAE: "
    f"{mae:.4f}"
)

print(
    f"Model RMSE: "
    f"{rmse:.4f}"
)

print(
    f"Historical trend: "
    f"{trend_direction}"
)

print(
    f"Trend slope: "
    f"{slope:.6f}"
)

print()

print(
    "14-Day Engagement Forecast"
)

print(
    "-" * 50
)


for _, row in forecast.iterrows():

    print(
        f"{row['date'].date()}: "
        f"{row['forecast_engagement']:.2f}%"
    )


print()

print(
    "Forecast file saved to:"
)

print(
    OUTPUT_PATH
)