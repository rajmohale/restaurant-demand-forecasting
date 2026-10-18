import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the cleaned dataset
df = pd.read_csv("cleaned_restaurant_data.csv")

# Step 2: Convert Order Date into datetime format
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Step 3: Create a daily order summary
daily_orders = (
    df.groupby(df["Order Date"].dt.date)["Order ID"]
    .nunique()
    .reset_index(name="Total Orders")
)

daily_orders["Order Date"] = pd.to_datetime(
    daily_orders["Order Date"]
)

# Step 4: Display basic statistics
print("\n========== DATASET SUMMARY ==========")

print("Total records:", len(df))
print("Unique orders:", df["Order ID"].nunique())
print("Unique menu items:", df["Item Name"].nunique())

print("\n========== DAILY ORDER STATISTICS ==========")

print("Average daily orders:",
      round(daily_orders["Total Orders"].mean(), 2))

print("Maximum daily orders:",
      daily_orders["Total Orders"].max())

print("Minimum daily orders:",
      daily_orders["Total Orders"].min())

print("Number of dates with recorded orders:",
      len(daily_orders))

# Step 5: Find the busiest recorded dates
print("\n========== TOP 10 BUSIEST DATES ==========")

print(
    daily_orders.nlargest(10, "Total Orders")
    .to_string(index=False)
)

# Step 6: Find the most popular menu items
popular_items = (
    df.groupby("Item Name")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n========== TOP 10 MENU ITEMS ==========")
print(popular_items.to_string())

# Step 7: Plot daily order counts
plt.figure(figsize=(12, 5))

plt.plot(
    daily_orders["Order Date"],
    daily_orders["Total Orders"]
)

plt.title("Daily Restaurant Order Demand")
plt.xlabel("Date")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 8: Plot the most popular menu items
plt.figure(figsize=(10, 5))

popular_items.sort_values().plot(kind="barh")

plt.title("Top 10 Most Popular Menu Items")
plt.xlabel("Total Quantity Sold")
plt.ylabel("Menu Item")
plt.tight_layout()
plt.show()

# Step 9: Save the daily summary for the next step
daily_orders.to_csv("daily_orders.csv", index=False)

print("\nDaily summary saved as daily_orders.csv")
print("EDA completed successfully!")

