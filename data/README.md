# Data

This directory contains the dataset used for the **Inventory Stockout Risk Prediction** project.

## Dataset

**File:** `raw_data.csv`

The project uses the **Retail Store Inventory Forecasting Dataset**, a synthetic retail dataset designed to represent inventory and demand behavior across multiple stores and products.

The dataset contains **73,100 daily store-product observations** and **15 variables**, including:

* Date
* Store ID
* Product ID
* Category
* Region
* Inventory Level
* Units Sold
* Units Ordered
* Demand Forecast
* Price
* Discount
* Weather Condition
* Holiday/Promotion
* Competitor Pricing
* Seasonality

## Data Source

The dataset was obtained from Kaggle:

**Retail Store Inventory Forecasting Dataset**
Creator: Anirudh Singh Chauhan
Platform: Kaggle

The dataset is distributed under the **CC0 1.0 Universal (Public Domain)** license.

## Use in This Project

The raw dataset is used to develop a machine learning framework for identifying store-product combinations at risk of stockout.

The analysis includes:

* Chronological sorting by store, product, and date
* Time-series demand analysis
* 7-day demand feature engineering
* Calendar feature extraction
* Categorical encoding
* Numerical feature scaling
* Stockout-risk target creation
* Safety-stock threshold sensitivity analysis
* Machine learning and LSTM model evaluation

The main analysis is available in:

`../notebooks/inventory_stockout_prediction.ipynb`

## Important Note

This is a **synthetic dataset** created for experimental retail analytics. It does not contain confidential customer or company information.

Because the data is simulated, relationships between variables may be cleaner and more deterministic than those found in real-world retail systems. Model performance in this project should therefore be interpreted as a proof of concept rather than expected real-world predictive performance.

For additional methodology, results, and limitations, see the main project `README.md` and the report in the `reports/` directory.
