# Datheon Machine Learning & Data Science Internship Portfolio

A comprehensive collection of end-to-end machine learning and data science projects developed during the Datheon Internship. This repository covers diverse problem domains across supervised multiclass classification, continuous price regression, imbalanced customer churn prediction, and natural language processing (NLP).

---

## 📌 Portfolio Overview

| Project | Domain | Dataset | Core Techniques | Models Benchmarked | Champion Model & Metric |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[1. Multiclass Classification](#1-multiclass-benchmark-classification-pipeline)** | Botanical Categorization | Iris Dataset (150 samples) | `StandardScaler`, 80/20 Stratified Split, 2D Decision Boundaries | Logistic Regression, Decision Tree | **Logistic Regression** (94.96% 5-Fold CV Accuracy) |
| **[2. Housing Price Prediction](#2-real-estate-housing-price-prediction)** | Real Estate Economics | California Housing (20,640 records) | Median Imputation, One-Hot Encoding, Feature Engineering, Multicollinearity Diagnostics | OLS, Ridge ($\alpha=10$), Lasso ($\alpha=10$) | **Tuned Ridge** (RMSE: \$70,031, $R^2$: 0.6257) |
| **[3. Customer Churn Classifier](#3-predictive-customer-churn-classifier)** | Telecom Customer Retention | Telco Churn (7,043 customers) | Leakage-free `Pipeline`, `ColumnTransformer`, `scale_pos_weight` Imbalance Handling | Random Forest, XGBoost | **XGBoost Classifier** (Recall: 78.61%, ROC-AUC: 0.8446) |
| **[4. NLP Sentiment Analysis](#4-nlp-sentiment-classification-on-customer-reviews)** | Sentiment Classification | Customer Reviews Corpus | Text Normalization, Tokenization, TF-IDF Vectorization ($n$-grams) | Multinomial Naive Bayes, Logistic Regression | **TF-IDF + Logistic Regression** Benchmark Pipeline |

---

## 📂 Repository Structure

```text
Datheon_Machine_Learning_&_Data_Science_Internship/
├── Multiclass_Benchmark_Classification_Pipeline/
│   ├── data/
│   │   └── iris.csv                                # Iris flower dataset
│   ├── visualizations/                             # Exported high-res PNG plots (9 figures)
│   ├── iris_classification.ipynb                   # End-to-end classification notebook
│   └── README.md                                   # Project documentation
│
├── RealEstate_Housing_Price_Prediction/
│   ├── data/
│   │   └── housing.csv                             # California housing dataset (20,640 rows)
│   ├── visualizations/                             # Exported diagnostic & residual plots (10 figures)
│   ├── house_price_prediction.ipynb                # Regression modeling & regularization notebook
│   └── README.md                                   # Project documentation
│
├── Predictive_Customer_Churn_Classifier/
│   ├── data/
│   │   └── Telco-Customer-Churn.csv                # Telco churn customer records
│   ├── models/
│   │   └── best_model.pkl                          # Serialized champion model pipeline (XGBoost)
│   ├── visualizations/                             # Confusion matrices, ROC curves, feature importances (8 figures)
│   ├── customer_churn.ipynb                        # End-to-end churn prediction notebook
│   └── README.md                                   # Project documentation
│
├── NLP_Sentiment_Classification_on_Customer_Reviews/
│   ├── data/
│   │   └── sentiment_reviews.csv                   # Raw customer sentiment text reviews
│   ├── visualizations/                             # Review length & sentiment distribution figures (3 figures)
│   └── sentiment_analysis.ipynb                    # Text preprocessing & NLP classification notebook
│
├── requirements.txt                                # Unified environment dependencies
├── LICENSE                                         # Open-source license
└── README.md                                       # Root portfolio overview (this file)
```

---

## 🔬 Project Summaries

### 1. Multiclass Benchmark Classification Pipeline
- **Directory:** [`Multiclass_Benchmark_Classification_Pipeline/`](Multiclass_Benchmark_Classification_Pipeline/)
- **Objective:** Classify Iris flower specimens into three species (*Setosa*, *Versicolor*, *Virginica*) based on sepal and petal morphology.
- **Methodology:** 
  - Checked data types, nulls, and removed duplicate entries.
  - Explored distributions via histograms, boxplots, correlation heatmaps, and Seaborn pairplots.
  - Standardized features with `StandardScaler` fitted strictly on training data.
  - Implemented 2D decision boundary mesh-grid visualizations using the top discriminative features (`petal_length` and `petal_width`).
  - Benchmarked models via 5-fold stratified cross-validation across Accuracy, Precision, Recall, and F1-score.
- **Results & Key Takeaways:**
  - **Champion Model:** **Logistic Regression** achieved **94.96% ± 6.12%** cross-validated accuracy, outperforming the unconstrained Decision Tree (92.43% ± 3.14%).
  - Smooth linear hyperplanes generalized more reliably than the axis-parallel step boundaries of the Decision Tree, which over-partitioned sample noise around the *Versicolor*/*Virginica* boundary.

---

### 2. Real Estate Housing Price Prediction
- **Directory:** [`RealEstate_Housing_Price_Prediction/`](RealEstate_Housing_Price_Prediction/)
- **Objective:** Predict median home values (`median_house_value`) across California census block groups using demographic, geographical, and structural variables.
- **Methodology:**
  - Imputed missing bedroom counts using median strategy via `SimpleImputer`.
  - Encoded categorical `ocean_proximity` with `OneHotEncoder(drop='first')`.
  - Engineered ratio features: `rooms_per_household`, `bedrooms_per_room`, and `population_per_household`.
  - Conducted multicollinearity diagnostics across volumetric block metrics (`total_rooms`, `total_bedrooms`, `population`, `households`).
  - Standardized features and optimized regularization penalty ($\alpha$) across OLS, Ridge, and Lasso via 5-fold `GridSearchCV`.
  - Conducted residual diagnostics (heteroskedasticity and price capping analysis).
- **Results & Key Takeaways:**
  - **Champion Model:** **Tuned Ridge Regression ($\alpha = 10$)** achieved the lowest test error (**RMSE: \$70,031**, **MAE: \$50,646**, **$R^2$: 0.6257**).
  - `median_income` emerged as the primary price driver ($r \approx 0.69$).
  - Identified an artificial price ceiling at **\$500,001**, which causes linear models to underestimate high-end real estate and introduces right-skewed residual variance.

---

### 3. Predictive Customer Churn Classifier
- **Directory:** [`Predictive_Customer_Churn_Classifier/`](Predictive_Customer_Churn_Classifier/)
- **Objective:** Predict subscriber churn for a telecommunications provider to identify at-risk accounts before cancellation.
- **Methodology:**
  - Cleaned whitespace strings in `TotalCharges` and imputed missing values.
  - Designed an end-to-end, leakage-free scikit-learn `Pipeline` utilizing `ColumnTransformer` (median imputation for numeric columns, one-hot encoding for categorical attributes).
  - Handled class imbalance (~26.5% churners) at the loss-function level using `class_weight='balanced'` in Random Forest and `scale_pos_weight = 2.76` in XGBoost.
  - Evaluated models using 5-fold stratified cross-validation, confusion matrices, ROC-AUC curves, and feature importance rankings.
  - Serialized the trained pipeline artifact to `models/best_model.pkl`.
- **Results & Key Takeaways:**
  - **Champion Model:** **XGBoost Classifier** was selected as the production model due to its high **Recall (78.61%)** and **ROC-AUC (0.8446)**.
  - In customer retention, False Negatives (unidentified churners) are far costlier than False Positives. XGBoost caught **294 out of 374** test churners (vs. 233 by Random Forest).
  - Top churn drivers identified: `Month-to-month` contracts, low `tenure` (< 12 months), high `MonthlyCharges`, and absence of bundled online security/tech support.

---

### 4. NLP Sentiment Classification on Customer Reviews
- **Directory:** [`NLP_Sentiment_Classification_on_Customer_Reviews/`](NLP_Sentiment_Classification_on_Customer_Reviews/)
- **Objective:** Classify customer review text into positive or negative sentiment polarities using natural language processing techniques.
- **Methodology:**
  - Preprocessed text: converted to lowercase, stripped punctuation, removed stopwords, and tokenized review sequences.
  - Transformed unstructured text into sparse numerical feature matrices via `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) using unigram and bigram ranges.
  - Trained and benchmarked baseline probabilistic and linear NLP classifiers: **Multinomial Naive Bayes** and **Logistic Regression**.
  - Generated confusion matrix heatmaps and review-length distribution visualizations.
- **Key Takeaways & Scalability:**
  - Established a clean, modular text classification pipeline ready to scale to large-scale sentiment corpora (e.g., Amazon Reviews, IMDb, Yelp).
  - Highlighted the critical importance of training corpus volume in NLP vectorization to ensure vocabulary overlap and reliable generalization on unseen test reviews.

---

## 🛠️ Tech Stack & Dependencies

- **Programming Language:** Python 3.10+
- **Data Manipulation & Analysis:** `pandas`, `numpy`, `scipy`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Machine Learning & Modeling:** `scikit-learn`, `xgboost`, `statsmodels`
- **Model Serialization & Environment:** `joblib`, `jupyter`, `nbformat`

---

## 🚀 Getting Started & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/seelneas/Datheon_Machine_Learning_-_Data_Science_Internship.git
cd Datheon_Machine_Learning_-_Data_Science_Internship
```

### 2. Set Up a Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate on macOS/Linux:
source .venv/bin/activate

# Activate on Windows:
.venv\Scripts\activate
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Any Project Notebook
Navigate to any project directory and launch Jupyter:
```bash
# Example: Launch customer churn notebook
cd Predictive_Customer_Churn_Classifier
jupyter notebook customer_churn.ipynb
```
Select **Kernel $\rightarrow$ Restart & Run All** to reproduce all figures, tables, and model outputs.

---

## 📜 License
This repository is licensed under the [MIT License](LICENSE).
