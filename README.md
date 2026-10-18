# 🍽️ Smart Restaurant Order Demand Forecasting and Resource Planning System Using Machine Learning

## 📌 Project Overview

The **Smart Restaurant Order Demand Forecasting and Resource Planning System** is a machine learning-based application designed to help restaurants predict future customer order demand and make better resource-planning decisions.

Restaurants often face difficulties in estimating daily demand, which can lead to overstaffing, understaffing, food wastage, and inefficient resource utilization.

This project uses historical restaurant order data to analyze demand patterns and predict the number of orders expected on a future date. It also provides visualizations of order trends, insights into popular menu items, and an estimated staffing requirement based on predicted demand.

The application is developed using Python, Machine Learning, Pandas, Matplotlib, and Streamlit.

## 🎯 Objectives

- Analyze historical restaurant order data.
- Clean and preprocess the dataset before analysis.
- Identify daily and monthly restaurant demand patterns.
- Train a machine learning model to predict future order demand.
- Estimate staffing requirements based on predicted orders.
- Identify popular menu items using sales quantity.
- Provide an interactive dashboard for demand analysis and planning.

## ✨ Key Features

### 1. Data Cleaning and Preprocessing
- Handles duplicate records.
- Converts order dates into the appropriate date format.
- Prepares data for further analysis and model training.

### 2. Exploratory Data Analysis
- Analyzes historical restaurant orders.
- Calculates average, maximum, and minimum daily demand.
- Visualizes monthly demand trends.

### 3. Machine Learning-Based Demand Forecasting
- Uses a Random Forest Regression model to predict future order demand.
- Considers historical demand, previous-day orders, previous-week orders, and calendar information.
- Evaluates model performance using MAE, RMSE, and R².

### 4. Menu Item Insights
- Identifies popular menu items based on their total sales quantity.
- Displays the top-selling items in a bar chart.
- Helps restaurants understand customer preferences.

### 5. Resource Planning
- Estimates the number of staff required based on predicted order demand.
- Allows users to configure the number of orders that one staff member can handle.
- Helps restaurants plan staffing more efficiently.

### 6. Interactive Web Application
- Built using Streamlit.
- Allows users to upload their own historical daily order data.
- Displays demand statistics, charts, menu insights, and forecasts.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data processing and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning model training and evaluation |
| Matplotlib | Data visualization |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web application |
| CSV | Dataset storage and exchange |
| Git and GitHub | Version control and project hosting |

## 📂 Project Structure

```text
restaurant-demand-forecasting/
│
├── app.py
├── clean_data.py
├── eda.py
├── train_model.py
├── predict_demand.py
├── generate_dummy_menu_data.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── restaurant-2-orders.csv
├── cleaned_restaurant_data.csv
├── daily_orders.csv
├── restaurant_demand_model.pkl
└── dummy_restaurant_menu_transactions_2026.csv
```

**Note:** Dataset and model files listed above may be stored locally and excluded from GitHub. The actual files present in the repository may differ from this example.

## ⚙️ Installation and Setup

### Prerequisites

Make sure you have installed:

- Python 3.10 or later
- Visual Studio Code
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/restaurant-demand-forecasting.git
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Open the Project Folder

```bash
cd restaurant-demand-forecasting
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application should open in your web browser.

## 📊 Dataset Information

The project uses historical restaurant order data containing information such as:

- Order ID
- Order Date
- Item Name
- Quantity
- Product Price
- Total Products

The data is processed to generate daily order summaries for demand forecasting.

Users can also upload their own daily order dataset to analyze demand and generate predictions.

### Supported Daily Dataset Formats

The application supports either of the following column formats:

**Format 1:**

| Date | Orders |
|---|---:|
| 2026-01-01 | 35 |
| 2026-01-02 | 42 |
| 2026-01-03 | 38 |

**Format 2:**

| Order Date | Total Orders |
|---|---:|
| 2026-01-01 | 35 |
| 2026-01-02 | 42 |
| 2026-01-03 | 38 |

For menu item analysis, the uploaded transaction dataset should contain:

| Item Name | Quantity |
|---|---:|
| Pizza | 25 |
| Burger | 18 |
| Pasta | 12 |

The application may require a minimum amount of historical data to train a useful model.

## 🤖 Machine Learning Approach

The project uses **Random Forest Regression** for demand forecasting.

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees to produce a final prediction.

### Features Used

The model uses features such as:

- Day of the week
- Month
- Day of the month
- Previous day's order demand
- Order demand from seven days earlier
- Rolling average of demand over the previous seven days

### Model Evaluation

The model is evaluated using the following metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted demand.
- **Root Mean Squared Error (RMSE):** Measures prediction error while penalizing larger errors more strongly.
- **R² Score:** Measures how well the model explains variation in the observed demand.

In the initial evaluation on the prepared historical dataset, the Random Forest model achieved:

| Metric | Result |
|---|---:|
| MAE | 5.461 |
| RMSE | 7.526 |
| R² Score | 0.689 |

These results apply to the initial evaluation dataset and are not a guarantee of future forecasting accuracy. Results may change when a different dataset is uploaded or the model is retrained.

## 📈 Expected Output

The application provides:

- Historical order-demand statistics.
- Monthly demand visualizations.
- Popular menu item analysis.
- Predicted future order demand.
- Estimated staffing requirements.

These outputs can help restaurant managers make more informed operational decisions.

## 🌍 Sustainable Development Goals (SDGs)

This project is primarily aligned with:

### SDG 12: Responsible Consumption and Production

The project can support responsible resource utilization by helping restaurants estimate demand, plan operations, and potentially reduce over-preparation and food waste.

### SDG 8: Decent Work and Economic Growth

The project supports more informed workforce planning by estimating staffing requirements based on expected order demand.

**Primary SDG:** SDG 12 – Responsible Consumption and Production.

The project's contribution depends on how its predictions are used in actual restaurant operations. The application does not directly measure or guarantee reductions in food waste.

## 🔮 Future Enhancements

- Integrate real-time restaurant order data.
- Improve forecasting through additional machine learning algorithms.
- Add weather, holidays, and local events as demand-prediction features.
- Forecast demand for individual menu items.
- Estimate ingredient requirements and inventory levels.
- Add food-waste tracking and reporting.
- Introduce user authentication and restaurant-specific dashboards.
- Store historical data in a database.
- Add downloadable forecasting reports.

## ⚠️ Limitations

- Forecasting accuracy depends on the quality and quantity of historical data.
- Predictions may not capture sudden changes in customer demand.
- Missing dates in the historical data may be interpreted as zero-order days, depending on the preprocessing method.
- Staffing estimates depend on the user-defined workload per staff member.
- Synthetic data, when used for demonstrations, does not represent verified real-world restaurant performance.

## 👨‍💻 Project Information

**Project Title:** Smart Restaurant Order Demand Forecasting and Resource Planning System Using Machine Learning

**Domain:** Machine Learning, Data Science, and Restaurant Resource Planning

**Application Framework:** Streamlit

**Programming Language:** Python

**Project Type:** Academic Mini Project

## 📜 License

This project is intended for educational and academic purposes.

A formal open-source license can be added if you decide how you want others to use, modify, and distribute the project.

---

⭐ If you find this project useful, consider giving the repository a star!
