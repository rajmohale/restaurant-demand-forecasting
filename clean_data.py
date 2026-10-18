import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("restaurant-2-orders.csv")

print("Original dataset shape:", df.shape)

# Step 2: Convert Order Date into datetime format
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%d/%m/%Y %H:%M",
    errors="coerce"
)

# Step 3: Remove rows with invalid dates
df = df.dropna(subset=["Order Date"])

# Step 4: Remove data before the long gap
cutoff_date = pd.Timestamp("2016-05-27")

df = df[df["Order Date"] >= cutoff_date].copy()

print("After removing old data:", df.shape)

# Step 5: Remove exact duplicate rows
duplicates = df.duplicated().sum()

print("Duplicate rows found:", duplicates)

df = df.drop_duplicates().copy()

# Step 6: Sort records by date
df = df.sort_values("Order Date").reset_index(drop=True)

# Step 7: Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Step 8: Save the cleaned dataset
df.to_csv("cleaned_restaurant_data.csv", index=False)

# Step 9: Display the final summary
print("\n========== CLEANING SUMMARY ==========")
print("Original records: 119183")
print("Final records:", len(df))
print("Unique orders:", df["Order ID"].nunique())
print("Unique menu items:", df["Item Name"].nunique())
print("First date:", df["Order Date"].min())
print("Last date:", df["Order Date"].max())
print("Remaining duplicates:", df.duplicated().sum())

print("\nCleaned dataset saved successfully!")
print("File: cleaned_restaurant_data.csv")
    