import pandas as pd
import joblib
import math

# Load the trained model
saved_data = joblib.load("restaurant_demand_model.pkl")

model = saved_data["model"]
model_name = saved_data["model_name"]
feature_columns = saved_data["feature_columns"]

print("Loaded model:", model_name)

# Load historical daily orders
df = pd.read_csv("daily_orders.csv")

# Match the actual column names in your CSV
df["Order Date"] = pd.to_datetime(df["Order Date"])
df = df.sort_values("Order Date")

# Rename columns to match the prediction code
df = df.rename(columns={
    "Order Date": "Date",
    "Total Orders": "Orders"
})

# Create a complete daily calendar
df = df.set_index("Date").asfreq("D")

# Assumption: dates without recorded orders represent zero orders
df["Orders"] = df["Orders"].fillna(0)

print("Historical data loaded successfully.")
print("First date:", df.index.min().date())
print("Last date:", df.index.max().date())
print("Total calendar days:", len(df))

# Store historical order counts
history = df["Orders"].tolist()

# Ask the user for a prediction date
date_input = input("Enter prediction date (YYYY-MM-DD): ")
target_date = pd.to_datetime(date_input)

last_date = df.index.max()

if target_date <= last_date:
    print("Please enter a date after", last_date.date())
else:
    current_date = last_date + pd.Timedelta(days=1)

    while current_date <= target_date:

        lag_1 = history[-1]
        lag_7 = history[-7]
        rolling_7 = sum(history[-7:]) / 7

        features = pd.DataFrame([{
            "DayOfWeek": current_date.dayofweek,
            "Month": current_date.month,
            "DayOfMonth": current_date.day,
            "Lag_1": lag_1,
            "Lag_7": lag_7,
            "Rolling_7": rolling_7
        }])

        # Keep features in the same order as during training
        features = features[feature_columns]

        prediction = float(model.predict(features)[0])

        # Order counts cannot be negative
        prediction = max(0, prediction)

        history.append(prediction)

        current_date += pd.Timedelta(days=1)

    predicted_orders = max(0, prediction)

    print("\n========== DEMAND FORECAST ==========")
    print("Prediction date:", target_date.date())
    print("Model used:", model_name)
    print("Predicted orders:", round(predicted_orders, 2))

    # Estimate staffing requirements
    orders_per_staff = 10

    staff_needed = math.ceil(predicted_orders / orders_per_staff)

    print("Estimated staff needed:", staff_needed)
    print("(Assumption: one staff member handles 10 orders.)")