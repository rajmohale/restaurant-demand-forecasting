import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# STEP 1: LOAD DATA
# --------------------------------------------------

df = pd.read_csv("daily_orders.csv", parse_dates=["Order Date"])

df = df.sort_values("Order Date").reset_index(drop=True)

print("Daily records loaded:", len(df))


# --------------------------------------------------
# STEP 2: CREATE A CONTINUOUS DAILY DATE SERIES
# --------------------------------------------------

# We only fill dates with zero orders in this first version.
# This is an assumption, not a confirmed fact about closures.

df = df.set_index("Order Date").asfreq("D")

df["Total Orders"] = df["Total Orders"].fillna(0)

df = df.reset_index()


# --------------------------------------------------
# STEP 3: CREATE FEATURES
# --------------------------------------------------

df["DayOfWeek"] = df["Order Date"].dt.dayofweek
df["Month"] = df["Order Date"].dt.month
df["DayOfMonth"] = df["Order Date"].dt.day

# Historical order counts
df["Lag_1"] = df["Total Orders"].shift(1)
df["Lag_7"] = df["Total Orders"].shift(7)
df["Rolling_7"] = df["Total Orders"].shift(1).rolling(7).mean()

# Remove rows without enough historical information
df = df.dropna().reset_index(drop=True)


# --------------------------------------------------
# STEP 4: CHRONOLOGICAL TRAIN/TEST SPLIT
# --------------------------------------------------

feature_columns = [
    "DayOfWeek",
    "Month",
    "DayOfMonth",
    "Lag_1",
    "Lag_7",
    "Rolling_7"
]

X = df[feature_columns]
y = df["Total Orders"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# STEP 5: DEFINE MODELS
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# STEP 6: EVALUATE MODELS
# --------------------------------------------------

results = []

# Baseline: use the same weekday's order count from
# seven days earlier.
baseline_predictions = df["Lag_7"].iloc[split_index:].to_numpy()

# Align the baseline with the exact test rows
baseline_predictions = baseline_predictions[-len(y_test):]

results.append({
    "Model": "Previous Week Baseline",
    "MAE": mean_absolute_error(y_test, baseline_predictions),
    "RMSE": np.sqrt(mean_squared_error(y_test, baseline_predictions)),
    "R2": r2_score(y_test, baseline_predictions)
})

trained_models = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = model


# --------------------------------------------------
# STEP 7: DISPLAY RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(3).to_string(index=False))


# --------------------------------------------------
# STEP 8: SELECT THE BEST ML MODEL BY MAE
# --------------------------------------------------

ml_results = results_df[
    results_df["Model"].isin(models.keys())
]

best_model_name = ml_results.sort_values("MAE").iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest ML model based on MAE:", best_model_name)


# --------------------------------------------------
# STEP 9: SAVE MODEL AND FEATURES
# --------------------------------------------------

model_package = {
    "model": best_model,
    "model_name": best_model_name,
    "feature_columns": feature_columns,
    "training_end_date": df["Order Date"].iloc[split_index - 1]
}

joblib.dump(model_package, "restaurant_demand_model.pkl")

print("\nModel saved as restaurant_demand_model.pkl")
print("Training completed successfully!")
