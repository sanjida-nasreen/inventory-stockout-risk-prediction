# Inventory Stockout Risk Prediction

A machine learning framework for predicting **short-term retail stockout risk** using inventory and demand data. This project combines time-series feature engineering, safety-stock threshold analysis, and multiple classification models to identify store–product combinations at risk of stockout.

The project compares **Logistic Regression, Decision Tree, Random Forest, and Long Short-Term Memory (LSTM)** models using time-aware validation and evaluates whether increased model complexity provides meaningful improvement over simpler, interpretable approaches.

## Project Overview

Stockouts can result in lost sales, reduced customer satisfaction, and disruptions to retail operations. At the same time, maintaining excessive inventory increases holding costs and waste.

This project addresses the problem as a **binary classification task**:

> Given the current inventory position and expected near-term demand, is a particular store–product combination at risk of stockout?

The analysis uses a synthetic retail dataset containing **73,100 daily store–product observations** across multiple stores, products, categories, and regions.

The workflow includes:

* Exploratory analysis of demand patterns and seasonality
* Time-series feature engineering
* Stockout-risk target construction
* Safety-stock threshold sensitivity analysis
* Time-series cross-validation
* Comparison of traditional machine learning and deep learning models
* Operational interpretation of prediction errors

## Repository Structure

```text
inventory-stockout-risk-prediction/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── data/
│   ├── raw_data.csv
│   └── README.md
│
├── notebooks/
│   └── inventory_stockout_prediction.ipynb
│
├── reports/
│   └── project_report.pdf
│
└── requirements.txt
```

## Dataset

The project uses the **Retail Store Inventory Forecasting Dataset**, a synthetic dataset designed for retail inventory and demand-forecasting applications.

The dataset contains **73,100 observations and 15 variables**, including:

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

Because the dataset is synthetic, the results should be interpreted as a proof of concept rather than evidence of expected performance in a real retail environment.

**Source:** Kaggle — *Retail Store Inventory Forecasting Dataset*, Anirudh Singh Chauhan
**License:** CC0 1.0 Universal (Public Domain)

## Methodology

### 1. Exploratory Data Analysis

Demand behavior was examined at multiple temporal scales to understand variability, trends, and recurring patterns.

The analysis included:

* Daily units-sold behavior
* 7-day moving averages for short-term patterns
* 30-day moving averages for longer-term trends
* Sales distribution analysis
* Month × day-of-week heatmap analysis

These analyses revealed substantial daily variability together with recurring weekly and seasonal demand patterns.

### 2. Data Preprocessing

The data was converted to chronological format and sorted by:

```text
Store ID → Product ID → Date
```

Categorical variables including product category, region, weather condition, and seasonality were transformed using **one-hot encoding**.

Numerical features were standardized using **Z-score scaling** where appropriate for model training.

### 3. Feature Engineering

A key engineered variable was the expected demand over the upcoming seven-day period:

```text
forecast_next_7d
```

The feature was calculated separately for each store–product combination using the demand forecast.

Additional temporal features included:

```text
dayofweek
month
```

This allowed the models to incorporate short-term demand information together with recurring calendar patterns.

## Stockout Risk Definition

Stockout risk was formulated as a binary classification problem.

For store `s`, product `i`, and time `t`, an observation is classified as **at risk** when:

```text
Inventory(s,i,t) < α × Forecasted 7-Day Demand(s,i,t)
```

where `α` represents the safety-stock threshold.

Therefore:

```text
at_risk = 1   → Stockout risk
at_risk = 0   → Safe inventory position
```

Rather than assuming a single threshold, the project evaluated:

```text
α = 0.10
α = 0.15
α = 0.20
α = 0.25
α = 0.30
α = 0.35
```

This allowed the analysis to study how inventory policy affects class balance and predictive performance.

## Models

Four classification approaches were compared.

### Logistic Regression

Used as the interpretable linear baseline.

### Decision Tree

Used to capture nonlinear decision boundaries while maintaining relatively straightforward interpretation.

### Random Forest

Used as an ensemble model to reduce the prediction variance associated with individual decision trees.

### Long Short-Term Memory (LSTM)

Used as the deep-learning comparison model to investigate whether sequential modeling could capture additional temporal information beyond the engineered demand features.

## Time-Series Validation

Because the observations are time-dependent, conventional randomized cross-validation was avoided.

Instead, the project used **expanding-window time-series cross-validation**.

Conceptually:

```text
Fold 1:
[ TRAIN ] → [ VALIDATE ]

Fold 2:
[      TRAIN      ] → [ VALIDATE ]

Fold 3:
[           TRAIN           ] → [ VALIDATE ]
```

The training window expands over time while validation always occurs on later observations.

A final **chronological 80/20 train-test split** was also used for model evaluation.

This approach better represents how a forecasting system would operate in practice and reduces the risk of future observations being used to predict the past.

## Evaluation Metrics

Model performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Particular attention was given to **F1-score** because stockout prediction involves an operational tradeoff:

* **False Negative:** A stockout risk is missed, potentially resulting in lost sales.
* **False Positive:** An item is unnecessarily flagged for replenishment, potentially increasing inventory and holding costs.

Accuracy alone can therefore be misleading when the at-risk class is imbalanced.

## Results

Threshold sensitivity analysis indicated that **α = 0.25** provided a strong balance between class distribution and model stability.

Final performance on the chronological 80/20 test split at `α = 0.25`:

| Model                   |   Accuracy |   F1-Score |  Precision |     Recall |        AUC |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| **Logistic Regression** | **0.9974** | **0.9970** | **0.9940** | **1.0000** | **1.0000** |
| Decision Tree           |     0.9939 |     0.9928 |     0.9897 |     0.9960 |     0.9942 |
| Random Forest           |     0.9748 |     0.9705 |     0.9684 |     0.9726 |     0.9980 |
| LSTM                    |     0.9877 |     0.9858 |     0.9722 |     0.9997 |     0.9999 |

## Key Findings

### 1. Feature engineering mattered more than model complexity

The engineered 7-day demand feature provided a very strong signal for identifying stockout risk.

As a result, Logistic Regression performed as well as or better than substantially more complex models.

### 2. Logistic Regression provided the strongest practical solution

The linear model achieved:

```text
Accuracy:  99.74%
F1-Score:  99.70%
Recall:   100.00%
AUC:      100.00%
```

Given its performance, interpretability, and low computational requirements, Logistic Regression was the preferred model for this formulation.

### 3. LSTM complexity did not provide additional predictive value

Although LSTMs are designed to capture sequential dependencies, the LSTM did not outperform Logistic Regression.

A likely reason is that the engineered 7-day forecast already summarized much of the temporal information needed for classification.

### 4. Safety-stock policy strongly affected model behavior

At low values of `α`, relatively few observations were classified as at risk, creating substantial class imbalance.

As `α` increased, the at-risk class became better represented and F1 performance stabilized.

The analysis selected:

```text
α = 0.25
```

as the preferred threshold for the evaluated dataset.

## Important Modeling Consideration

The strong predictive results should be interpreted carefully.

The `at_risk` target is defined directly from **inventory level and forecasted demand**, and these variables are also highly informative inputs to the models. Consequently, the classification boundary is relatively deterministic.

The near-perfect performance therefore demonstrates that the models can successfully reproduce the defined inventory-risk policy; it should **not** be interpreted as evidence that the same accuracy would necessarily be achieved when predicting real-world stockout events.

This distinction is particularly important because the dataset is synthetic and contains cleaner relationships than would normally be expected in operational retail data.

## Business Interpretation

The project demonstrates how predictive analytics can translate inventory and demand information into an operational warning system.

A potential deployment workflow would be:

```text
Inventory & Demand Data
          ↓
7-Day Demand Estimation
          ↓
Stockout Risk Prediction
          ↓
At-Risk SKU Identification
          ↓
Replenishment Prioritization
```

Rather than treating every product equally, inventory managers could use predicted risk to prioritize replenishment decisions.

## Future Work

Several extensions could make the framework more representative of real inventory systems:

* Validate the methodology using real-world retail transaction data
* Predict **observed future stockout events** rather than a rule-derived label
* Incorporate supplier lead times and replenishment delays
* Include ordering and holding costs
* Develop product- or store-specific safety-stock thresholds
* Incorporate additional external demand signals
* Account for missing data, disruptions, and forecast uncertainty
* Extend prediction into prescriptive inventory optimization
* Explore reinforcement learning for dynamic replenishment decisions

## Tools & Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Seaborn
* Jupyter Notebook

## How to Run

Clone the repository:

```bash
git clone https://github.com/sanjida-nasreen/inventory-stockout-risk-prediction.git
cd inventory-stockout-risk-prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Then open:

```text
notebooks/inventory_stockout_prediction.ipynb
```

and run the notebook cells sequentially.

## Project Context

This project was completed as part of **IND E 528** and focused on applying machine learning and time-series analysis to an inventory-management problem.

The work demonstrates applications of:

* Inventory analytics
* Time-series data processing
* Feature engineering
* Classification
* Deep learning
* Model comparison
* Time-series cross-validation
* Threshold sensitivity analysis
* Operational decision support

## Authors

* Hailey Oh
* Sanjida Nasreen
* Morgan Lan

## License

The project code is released under the **MIT License**.

The dataset is distributed separately under its original **CC0 1.0 Universal (Public Domain)** license.
