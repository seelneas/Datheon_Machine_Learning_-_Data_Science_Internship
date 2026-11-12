# Predictive Customer Churn Classifier

A supervised machine learning classification pipeline built on the Telco Customer Churn dataset to identify at-risk customers, mitigate class imbalance, benchmark ensemble models, and export a production-ready classifier.

---

## 📌 Project Overview

Customer attrition directly impacts recurring revenue in telecommunications and subscription-based business models. The goal of this project is to develop an automated predictive pipeline capable of accurately identifying churn-prone customers ahead of time, allowing retention teams to deploy targeted interventions.

The project addresses:
- **Class Imbalance:** Managing a ~73.5% non-churn vs. ~26.5% churn distribution without introducing synthetic noise.
- **Ensemble Benchmarking:** Comparing **Random Forest** (bagging) against **XGBoost** (gradient boosting).
- **Metric Prioritization:** Emphasizing **Recall** and **ROC-AUC** to minimize costly False Negatives (unidentified churners).
- **Explainability:** Extracting feature importance scores to provide actionable business recommendations.
- **Pipeline Deployment:** Exporting the end-to-end trained model artifact (`models/best_model.pkl`).

---

## 📂 Directory Structure

```text
Predictive_Customer_Churn_Classifier/
├── data/
│   └── Telco-Customer-Churn.csv       # Dataset (7,043 customers, 21 attributes)
├── models/
│   └── best_model.pkl                 # Serialized champion model (XGBoost pipeline)
├── customer_churn.ipynb               # Main pipeline notebook (EDA, training, evaluation)
└── README.md                          # Project documentation
```

---

## 📊 Dataset Description

The dataset comprises 7,043 customer accounts from a telecommunications provider, covering demographic profiles, subscribed services, account tenure, contract terms, and billing metrics.

| Category | Features |
| :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Services Subscribed** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Account Information** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| **Target Variable** | `Churn` (`No`: 5,174 [73.46%], `Yes`: 1,869 [26.54%]) |

### Preprocessing & Data Hygiene
- **Type Conversion & Null Handling:** `TotalCharges` contained 11 blank whitespace strings (`" "`) corresponding to zero-tenure customers. These were cast to `float64` and imputed using the median strategy.
- **Identifier Removal:** `customerID` was removed as it carries no predictive signal.
- **Target Encoding:** Mapped `No` $\rightarrow$ `0` and `Yes` $\rightarrow$ `1`.

---

## ⚙️ Technical Workflow & Pipeline Architecture

```mermaid
flowchart LR
    A["Raw Data (Telco-Customer-Churn.csv)"] --> B["Data Cleaning & Type Casting"]
    B --> C["Exploratory Data Analysis (EDA)"]
    C --> D["Stratified 80/20 Train-Test Split"]
    D --> E["ColumnTransformer Preprocessing"]
    E --> F["Class Imbalance Handling"]
    F --> G["Model Training (Random Forest & XGBoost)"]
    G --> H["5-Fold Stratified Cross-Validation"]
    H --> I["Evaluation (Recall, F1, ROC-AUC)"]
    I --> J["Feature Importance Analysis"]
    J --> K["Export Champion Model (best_model.pkl)"]
```

### 1. Data Splitting
- An **80/20 stratified split** was applied ($n_{\text{train}} = 5,634$, $n_{\text{test}} = 1,409$, `random_state=42`) to preserve the 26.5% minority class ratio across training and testing partitions.

### 2. Leakage-Free Preprocessing Pipeline
All transformations were encapsulated in a scikit-learn `Pipeline` and `ColumnTransformer`:
- **Numerical Pipeline:** `SimpleImputer(strategy='median')` for continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`).
- **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore')`.

### 3. Handling Class Imbalance
Rather than relying on synthetic oversampling (SMOTE) which can distort feature correlations in mixed data, class imbalance was addressed within the loss functions:
- **Random Forest:** Configured with `class_weight='balanced'`.
- **XGBoost:** Configured with `scale_pos_weight = 2.76` ($\frac{\text{Negative Samples}}{\text{Positive Samples}}$).

---

## 📈 Benchmark Results

Both models were benchmarked across a 5-fold stratified cross-validation setup and evaluated on the held-out test partition ($n = 1,409$):

| Evaluation Metric | Random Forest Classifier | XGBoost Classifier (Champion) |
| :--- | :---: | :---: |
| **Accuracy** | **76.65%** | 75.59% |
| **Precision** | **55.34%** | 52.69% |
| **Recall (Churn Class)** | 62.30% (233 / 374) | **78.61% (294 / 374)** |
| **F1-Score** | 0.5862 | **0.6309** |
| **Test ROC-AUC** | 0.8219 | **0.8446** |
| **5-Fold CV ROC-AUC** | 0.8220 ± 0.0107 | **0.8474 ± 0.0105** |

### 🏆 Model Selection Rationale
**XGBoost was chosen as the champion model:**
1. **Churn Economics Prioritizes Recall:** In retention strategy, a **False Negative** (an at-risk customer who leaves without intervention) incurs full customer acquisition cost replacement, whereas a **False Positive** incurs only the marginal cost of a retention offer. XGBoost catches **78.61% of all churners**, outperforming Random Forest by **+16.31%**.
2. **Discriminative Power:** XGBoost achieved an ROC-AUC of **0.8446** (and 0.8474 on 5-fold CV), proving superior probability calibration across all classification thresholds.

---

## 🔍 Key Feature Drivers & Business Takeaways

Feature importance analysis revealed the primary triggers driving customer churn:

1. **Contract Structure:** `Contract_Month-to-month` is the single most dominant churn predictor. Customers without long-term commitments churn at nearly **5x** the rate of 1- or 2-year contract holders.
2. **Tenure & Cumulative Spend:** `TotalCharges` and `tenure` heavily govern customer stickiness. The highest churn vulnerability window occurs within the **first 6 to 12 months**.
3. **Internet & Service Deficits:** Customers on `Fiber optic` internet churn significantly more when they lack bundled add-ons like `OnlineSecurity`, `TechSupport`, and `DeviceProtection`.

### Strategic Recommendations
- **Contract Migration Campaign:** Offer introductory discounts or loyalty credits to incentivize month-to-month users to transition to annual contracts.
- **Early-Tenure Onboarding Interventions:** Deploy automated customer success touchpoints during the critical first 90 days.
- **Value-Add Security Bundles:** Subsidize or bundle `OnlineSecurity` and `TechSupport` for fiber optic subscribers to improve retention.

---

## 💻 How to Run

### 1. Navigate to the project directory
```bash
cd Predictive_Customer_Churn_Classifier
```

### 2. Install dependencies
```bash
pip install -r ../requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook customer_churn.ipynb
```
Select **Run All Cells** to execute the pipeline end-to-end and recreate the model artifact.

