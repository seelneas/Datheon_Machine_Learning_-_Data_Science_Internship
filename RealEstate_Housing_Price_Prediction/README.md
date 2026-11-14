# Real Estate Housing Price Prediction

A supervised regression pipeline built on the California Housing Dataset to benchmark and regularize linear models for predicting median home values.

---

## 📌 Project Overview

This project builds a regression pipeline to predict block-level median house values (`median_house_value`) across California. The workflow addresses common real-world data challenges—including missing data imputation, categorical encoding, multicollinearity, and feature scaling—while evaluating regularized linear models against ordinary least squares:
- **Ordinary Least Squares (OLS) Linear Regression**
- **Ridge Regression ($L_2$ Regularization)**
- **Lasso Regression ($L_1$ Regularization)**
- **Hyperparameter-Tuned Ridge & Lasso (5-Fold GridSearchCV)**

The final models are assessed using standard regression metrics (**RMSE**, **MAE**, and **$R^2$**) and residual diagnostic plots.

---

## 📂 Directory Structure

```text
RealEstate_Housing_Price_Prediction/
├── data/
│   └── housing.csv                 # Dataset (20,640 records, 10 attributes)
├── house_price_prediction.ipynb    # Main end-to-end pipeline notebook
└── README.md                       # Project documentation
```

---

## 📊 Dataset Description

The dataset contains 20,640 block group observations from the 1990 U.S. Census in California.

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `longitude` | Continuous (`float64`) | Longitude coordinate (degrees west) |
| `latitude` | Continuous (`float64`) | Latitude coordinate (degrees north) |
| `housing_median_age` | Continuous (`float64`) | Median age of houses within the block group |
| `total_rooms` | Continuous (`float64`) | Total number of rooms within the block |
| `total_bedrooms` | Continuous (`float64`) | Total number of bedrooms (contains 207 missing values) |
| `population` | Continuous (`float64`) | Total population residing in the block |
| `households` | Continuous (`float64`) | Total number of households (occupants) in the block |
| `median_income` | Continuous (`float64`) | Median household income (in tens of thousands of USD) |
| `ocean_proximity` | Categorical (`object`) | Proximity category: `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND` |
| `median_house_value` | Continuous (`float64`) | **Target Variable:** Median house value in USD (capped at \$500,001) |

---

## ⚙️ Technical Workflow & Pipeline Architecture

```mermaid
flowchart LR
    A["Raw Data (housing.csv)"] --> B["Missing Value Imputation"]
    B --> C["One-Hot Encoding"]
    C --> D["Exploratory Data Analysis (EDA)"]
    D --> E["Feature Engineering & Multicollinearity"]
    E --> F["80/20 Train-Test Split"]
    F --> G["Feature Scaling (StandardScaler)"]
    G --> H["Model Training (OLS, Ridge, Lasso)"]
    H --> I["5-Fold GridSearchCV Tuning"]
    I --> J["Evaluation & Residual Diagnostics"]
```

### 1. Data Cleaning & Preprocessing
- **Missing Value Imputation:** 207 missing entries in `total_bedrooms` (~1% of data) were imputed using the median strategy via `SimpleImputer`.
- **Categorical Encoding:** `ocean_proximity` was transformed into dummy features using `OneHotEncoder(drop='first')` to prevent dummy variable collinearity.
- **Dataset Splitting:** 80% training ($n = 16,512$), 20% testing ($n = 4,128$) with a fixed seed (`random_state=42`).
- **Standardization:** Input features were standardized ($\mu = 0, \sigma = 1$) via `StandardScaler`, fit exclusively on the training partition.

### 2. Exploratory Data Analysis & Feature Engineering
- **Strongest Predictor:** `median_income` exhibits the highest linear correlation with housing prices ($r \approx 0.69$).
- **Target Capping:** `median_house_value` displays an artificial cap at **\$500,001**, which causes linear models to underestimate luxury homes.
- **Engineered Features:**
  - `rooms_per_household` = $total\_rooms / households$
  - `bedrooms_per_room` = $total\_bedrooms / total\_rooms$
  - `population_per_household` = $population / households$
- **Multicollinearity:** High inter-correlation among aggregate block volume counts (`total_rooms`, `total_bedrooms`, `population`, `households`).

---

## 📈 Benchmark Results

All models were evaluated on the held-out 20% test set ($n = 4,128$):

| Model | Test RMSE (\$) | Test MAE (\$) | $R^2$ Score | Hyperparameters |
| :--- | :---: | :---: | :---: | :--- |
| **Tuned Ridge** | **\$70,031.31** | **\$50,646.17** | **0.6257** | **$\alpha = 10$** (via 5-fold CV) |
| **Tuned Lasso** | \$70,053.91 | \$50,664.70 | 0.6255 | **$\alpha = 10$** (via 5-fold CV) |
| **Ridge Baseline** | \$70,057.42 | \$50,668.12 | 0.6255 | $\alpha = 1.0$ |
| **Lasso Baseline** | \$70,059.85 | \$50,670.13 | 0.6254 | $\alpha = 1.0$ |
| **OLS Baseline** | \$70,060.52 | \$50,670.74 | 0.6254 | Unregularized |

### 5-Fold Cross-Validation Optimization
Hyperparameter tuning over $\alpha \in [0.001, 0.01, 0.1, 1, 10, 100, 1000]$ yielded:
- **Best Ridge:** $\alpha = 10$ with CV RMSE = **\$68,619.86**
- **Best Lasso:** $\alpha = 10$ with CV RMSE = **\$68,622.24**

Regularization successfully shrunk inflated coefficients from collinear features, improving test generalization error.

---

## 🔍 Model Diagnostics & Residual Analysis

The notebook visualizes the best model (**Tuned Ridge**):
1. **Actual vs. Predicted Plot:** Predictions adhere closely to the $y = x$ reference line up to roughly \$350,000, with a horizontal ceiling artifact visible at the \$500,001 cap.
2. **Residual Distribution:** Centered near zero and approximately bell-shaped, but right-skewed due to underestimating homes above the \$500,001 threshold.
3. **Residuals vs. Predicted Values:** Confirms mild heteroskedasticity and an artificial diagonal cutoff boundary generated by the upper price cap.

---

## 🏆 Key Takeaways & Recommendations

- **Selected Model:** **Tuned Ridge Regression ($\alpha = 10$)** is the champion linear baseline, delivering the lowest test RMSE (**\$70,031**) and MAE (**\$50,646**).
- **Linear Model Boundary:** The ~62.6% $R^2$ represents the performance limit of standard linear equations on non-linear spatial data.
- **Future Improvements:** 
  1. Transition to non-linear ensemble models (**Random Forest**, **XGBoost**, **LightGBM**) to capture complex geographic interactions (`latitude` $\times$ `longitude`).
  2. Implement censoring techniques or remove the capped \$500,001 boundary points to reduce high-end residual bias.

---

## 🚀 How to Run

### 1. Navigate to the project directory
```bash
cd RealEstate_Housing_Price_Prediction
```

### 2. Install requirements
```bash
pip install -r ../requirements.txt
```

### 3. Open the notebook
```bash
jupyter notebook house_price_prediction.ipynb
```
Select **Run All Cells** to execute the pipeline end-to-end.

