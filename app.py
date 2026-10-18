import streamlit as st
import pandas as pd
import joblib
import math
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Restaurant Demand Forecasting",
    page_icon="🍽️",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("restaurant_demand_model.pkl")

saved_data = load_model()
model = saved_data["model"]
model_name = saved_data["model_name"]
feature_columns = saved_data["feature_columns"]

# Application title
st.title("🍽️ Smart Restaurant Demand Forecasting")
st.write(
    "Forecast restaurant order demand and plan staffing "
    "using Machine Learning."
)

st.info(f"Current forecasting model: **{model_name}**")

# Sidebar settings
st.sidebar.header("Forecast Settings")

orders_per_staff = st.sidebar.number_input(
    "Orders handled per staff member",
    min_value=1,
    max_value=100,
    value=10
)

# Upload historical dataset
st.header("1. Upload Historical Order Data")

uploaded_file = st.file_uploader(
    "Upload your cleaned daily_orders.csv",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        # Support the column names in our current dataset
        if "Order Date" in df.columns and "Total Orders" in df.columns:
            df = df.rename(columns={
                "Order Date": "Date",
                "Total Orders": "Orders"
            })

        if "Date" not in df.columns or "Orders" not in df.columns:
            st.error(
                "CSV must contain either 'Date' and 'Orders' "
                "or 'Order Date' and 'Total Orders'."
            )
            st.stop()

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df["Orders"] = pd.to_numeric(
            df["Orders"],
            errors="coerce"
        )

        df = df.dropna(subset=["Date", "Orders"])
        df = df.sort_values("Date")

        # Combine duplicate dates
        df = df.groupby("Date", as_index=False)["Orders"].sum()

        # -----------------------------------------
        # TEMPORARY MODEL TRAINING
        # -----------------------------------------

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error

        # Create daily historical data
        training_daily = (
            df.set_index("Date")["Orders"]
            .sort_index()
            .asfreq("D")
        )

        # Assumption: missing dates represent zero orders.
        # Verify this assumption for real restaurant datasets.
        training_daily = training_daily.fillna(0)

        # Prepare training features
        training_df = training_daily.to_frame(name="Orders")

        training_df["DayOfWeek"] = training_df.index.dayofweek
        training_df["Month"] = training_df.index.month
        training_df["DayOfMonth"] = training_df.index.day
        training_df["Lag_1"] = training_df["Orders"].shift(1)
        training_df["Lag_7"] = training_df["Orders"].shift(7)
        training_df["Rolling_7"] = (
            training_df["Orders"].shift(1).rolling(7).mean()
        )

        # Remove rows without enough historical information
        training_df = training_df.dropna()

        feature_columns = [
            "DayOfWeek",
            "Month",
            "DayOfMonth",
            "Lag_1",
            "Lag_7",
            "Rolling_7"
        ]

        X = training_df[feature_columns]
        y = training_df["Orders"]

        if len(training_df) < 30:
            st.error(
                "Not enough usable historical data. "
                "Please upload a dataset covering more days."
            )
            st.stop()

        # Train a temporary model in memory
        active_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        active_model.fit(X, y)


        if len(df) < 30:
            st.error("Please upload at least 30 days of historical data.")
            st.stop()

        st.success("Dataset uploaded successfully!")

        # Dataset summary
        col1, col2, col3 = st.columns(3)

        col1.metric("Historical Days", len(df))
        col2.metric("Total Orders", int(df["Orders"].sum()))
        col3.metric("Average Daily Orders", round(df["Orders"].mean(), 2))

        st.subheader("2. Historical Demand")

        # Convert dates into monthly periods
        chart_df = df.copy()
        chart_df["Month"] = chart_df["Date"].dt.to_period("M")

        # Calculate monthly average and maximum orders
        monthly_data = chart_df.groupby("Month").agg(
            Average_Orders=("Orders", "mean"),
            Maximum_Orders=("Orders", "max")
        ).reset_index()

        monthly_data["Month"] = monthly_data["Month"].dt.to_timestamp()

        # Create a cleaner chart
        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            monthly_data["Month"],
            monthly_data["Average_Orders"],
            label="Monthly Average Orders",
            linewidth=2
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Average Orders per Day")
        ax.set_title("Restaurant Demand Trend Over Time")

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

        fig.autofmt_xdate()
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # Display useful statistics
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average Daily Orders",
            round(df["Orders"].mean(), 2)
        )

        col2.metric(
            "Highest Daily Orders",
            int(df["Orders"].max())
        )

        col3.metric(
            "Lowest Daily Orders",
            int(df["Orders"].min())
        )

        # Menu Item Insights
        st.subheader("3. Menu Item Insights")

        st.write(
            "Upload your original restaurant transaction dataset "
            "to discover the most popular menu items."
        )

        menu_file = st.file_uploader(
            "Upload transaction-level CSV",
            type=["csv"],
            key="menu_upload"
        )

        if menu_file is not None:

            menu_df = pd.read_csv(menu_file)

            required_columns = ["Item Name", "Quantity"]

            if all(col in menu_df.columns for col in required_columns):

                menu_df["Quantity"] = pd.to_numeric(
                    menu_df["Quantity"],
                    errors="coerce"
                )

                menu_df = menu_df.dropna(
                    subset=["Item Name", "Quantity"]
                )

                item_sales = (
                    menu_df.groupby("Item Name")["Quantity"]
                    .sum()
                    .sort_values(ascending=False)
                )

                st.success("Menu-item data loaded successfully!")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Unique Menu Items",
                    int(item_sales.size)
                )

                col2.metric(
                    "Total Items Sold",
                    int(item_sales.sum())
                )

                st.write("### Top 10 Best-Selling Items")

                top_items = item_sales.head(10)

                fig, ax = plt.subplots(figsize=(10, 5))

                top_items.sort_values().plot(
                    kind="barh",
                    ax=ax
                )

                ax.set_xlabel("Total Quantity Sold")
                ax.set_ylabel("Menu Item")
                ax.set_title("Top 10 Best-Selling Menu Items")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.write("### Menu Sales Details")

                menu_details = item_sales.rename("Quantity Sold").reset_index()

                menu_details.index = range(1, len(menu_details) + 1)

                st.dataframe(
                    menu_details,
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.error(
                    "The uploaded CSV must contain "
                    "'Item Name' and 'Quantity' columns."
                )

        else:
            st.info(
                "Upload your original transaction-level CSV to see "
                "menu-item insights. You can use restaurant-2-orders.csv."
            )

        # Forecasting
        st.subheader("4. Forecast Future Demand")

        target_date = st.date_input(
            "Choose a future prediction date",
            value=df["Date"].max().date() + pd.Timedelta(days=1),
            min_value=df["Date"].max().date() + pd.Timedelta(days=1)
        )

        if st.button("Predict Demand"):

            daily = df.set_index("Date")["Orders"].asfreq("D")

            # Missing dates are assumed to have zero orders.
            # This assumption should be checked against the source data.
            daily = daily.fillna(0)

            history = daily.tolist()
            current_date = daily.index.max() + pd.Timedelta(days=1)
            target = pd.Timestamp(target_date)

            prediction = 0

            while current_date <= target:

                if len(history) < 7:
                    st.error("At least seven daily observations are required.")
                    st.stop()

                features = pd.DataFrame([{
                    "DayOfWeek": current_date.dayofweek,
                    "Month": current_date.month,
                    "DayOfMonth": current_date.day,
                    "Lag_1": history[-1],
                    "Lag_7": history[-7],
                    "Rolling_7": sum(history[-7:]) / 7
                }])

                features = features[feature_columns]

                prediction = max(
                    0,
                    float(active_model.predict(features)[0])
                )

                history.append(prediction)
                current_date += pd.Timedelta(days=1)

            predicted_orders = round(prediction)
            staff_needed = math.ceil(
                predicted_orders / orders_per_staff
            )

            st.success("Demand forecast generated!")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Orders",
                predicted_orders
            )

            col2.metric(
                "Estimated Staff Required",
                staff_needed
            )

            st.caption(
                f"Staff estimate assumes each staff member handles "
                f"{orders_per_staff} orders. This is a planning estimate, "
                f"not a separately trained ML prediction."
            )

            st.warning(
                "Forecasts are estimates. Missing dates are currently "
                "treated as zero-order days, and the loaded model was "
                "trained on the original restaurant dataset. "
                "For other restaurants, retraining the model on their "
                "own uploaded data is a necessary next improvement."
            )

    except Exception as e:
        st.error(f"Something went wrong: {e}")

else:
    st.warning("Upload your daily_orders.csv file to begin.")

st.divider()

st.caption(
    "Machine Learning project | Restaurant Demand Forecasting "
    "and Resource Planning"
)