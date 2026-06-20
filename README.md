# feature-engineering-lgbm-pipeline
### Credit Risk Assessment — Probability of Default

---

## Overview

This project builds an end-to-end machine learning pipeline to predict the **probability of default** for loan applicants. When a customer submits a loan application on the platform, their personal data and credit agency information is passed through this pipeline, which outputs a default probability. This probability determines whether the customer receives a loan and at what interest rate.

The model is built on historical loan application data using **LightGBM** — a high-performance gradient boosting framework well-suited for tabular financial data.

---

## Dataset

**Source:** Manu Siddhartha, November 6, 2020, *Bondora Peer-to-Peer Lending Data*, IEEE Dataport.  
doi: [https://dx.doi.org/10.21227/33kz-0s65](https://dx.doi.org/10.21227/33kz-0s65)

| File | Description |
|------|-------------|
| `loan_data.csv` | Historical loan applications — used for training and evaluation |
| `latest_customers.csv` | Most recent customer cohort — used for live model validation |

> ⚠️ These files contain sensitive customer financial information and are excluded from version control via `.gitignore`.

---

## Project Structure

```
feature-engineering-lgbm-pipeline/
│
├── notebook.ipynb              # Main development notebook (Tasks 1–10)
├── pipeline-task11.ipynb       # Clean deployment pipeline (Tasks 11–12)
│
├── encoder.pkl                 # Saved OrdinalEncoder (fitted on training data)
├── lightGBM.pkl                # Saved LightGBM model
│
├── requirements.txt            # Python dependencies
├── REPORT.md                   # Bug analysis and fix report
└── README.md                   # This file
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/feature-engineering-lgbm-pipeline.git
cd feature-engineering-lgbm-pipeline
```

### 2. Create and activate a virtual environment
```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Register the kernel and launch Jupyter
```bash
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
jupyter notebook
```

> In Jupyter, select **Kernel → Change Kernel → Python (venv)** before running cells.

---

## Pipeline Overview

The feature engineering pipeline (`feature_engineering_pipe`) takes raw customer and credit agency data and returns a preprocessed feature matrix ready for LightGBM. Below is a plain-language description of each step.

---

### Step 1 — Load and Split the Data
The dataset is split 80/20 into training and test sets using a fixed random seed for reproducibility. The target variable `default` indicates whether a customer has defaulted (1) or not (0). The overall default rate is approximately 65%.

---

### Step 2 — Impute Numerical Variables
Some numerical fields have missing values (e.g. `nr_dependants`, `credit_score_4`, repayment history). These are filled with **-1** — a value outside the natural range of all these variables. LightGBM can learn that -1 signals "not provided."

---

### Step 3 — Create Income and Debt Features
Three new features are engineered from existing columns:

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `total_income` | Sum of all 7 income sources | Total monthly income |
| `dti` | `total_debt / total_income` | Proportion of income going to debt |
| `cash` | `total_income - total_debt` | Discretionary income remaining |

> Customers who did not disclose income have `total_income = 0`, causing `dti` to be `inf` or `NaN`. These are replaced with 0.

---

### Step 4 — Create Datetime Features
Two datetime columns are processed:

- **`age`**: computed as `(application_date - date_of_birth) / 365` in whole years
- Customers under 18 are removed (the platform does not lend to minors)
- From `application_date`: `hour`, `dow` (day of week), `dom` (day of month), `month`

The raw datetime columns are dropped after extraction.

---

### Step 5 — Remove High-Cardinality Categorical Variables
Variables with too many unique values are removed:

| Variable | Unique values | Reason for removal |
|----------|-------------|-------------------|
| `employment_position` | 3,247 | Multi-language entries, too many future unseen values |
| `city` | 6,018 | Too granular, not generalisable |
| `county` | 951 | Too granular, not generalisable |

---

### Step 6 — Impute Categorical Variables
Remaining categorical variables with missing values are filled with the string `"missing"`. This preserves the information that a customer did not provide a value — the absence of information can itself be predictive of risk.

---

### Step 7 — Group Rare Categories
For 7 categorical variables, categories appearing in fewer than 5% of training observations are grouped into a single `"Rare"` label. This prevents overfitting to rare values and gracefully handles new, unseen categories at prediction time.

Variables grouped:
- `language`, `use_of_loan`, `occupation`, `home_ownership`
- `credit_score_1`, `credit_score_2`, `credit_score_3`

---

### Step 8 — Encode Categorical Variables
LightGBM's Python API requires numeric inputs. An `OrdinalEncoder` (from scikit-learn) converts each category to an integer. The encoder is **fitted on training data only** to prevent data leakage.

---

### Step 9 — Train and Evaluate LightGBM
A LightGBM classifier is trained with:
- Up to 1,000 boosting iterations
- **Early stopping** after 3 non-improving iterations (prevents overfitting)
- Evaluation metric: binary log-loss on the test set

Performance is measured using:
- **ROC-AUC**: how well the model ranks defaults above non-defaults (1.0 = perfect)
- **Classification report**: precision, recall, F1-score, and accuracy

---

### Step 10 — Evaluate Feature Importance
Three complementary methods are used to understand which features drive predictions:

| Method | What it measures |
|--------|-----------------|
| LightGBM built-in | How many times each feature is used to split decision trees |
| Permutation importance | Drop in ROC-AUC when a feature's values are randomly shuffled |
| Single-feature classifiers | ROC-AUC of a decision tree trained on each feature alone |

Features with permutation importance > 0 hurt performance when removed.  
Features with single-feature ROC-AUC > 0.5 are individually predictive (better than random).

---

### Step 11 — End-to-End Pipeline
All feature engineering steps are packaged into `feature_engineering_pipe()` — a single function that takes raw data and returns model-ready features. The encoder and model are serialised to `.pkl` files for deployment.

---

### Step 12 — Score Latest Customer Cohort
The pipeline is applied to `latest_customers.csv` to validate model performance on recent applicants. Lower metrics are expected here because:
- These customers have just received their loans
- There is not yet enough repayment history to accurately determine defaults
- Many customers are currently labelled "no default" even though they may default in coming months

---

## Key Design Decisions

**Why -1 for numerical imputation?**  
All numerical variables have natural values ≥ 0. Using -1 as a sentinel allows LightGBM to distinguish "missing" from "zero" — a meaningful difference (e.g. zero income vs unknown income).

**Why `"missing"` for categorical imputation?**  
Encoding NaN as a category preserves the signal that a customer did not provide a value. LightGBM can learn that missing values correlate with risk.

**Why group rare categories?**  
Categories seen in < 5% of training data are unreliable — the model has few examples to learn from. More importantly, entirely new categories will appear in production data. Grouping rare values into `"Rare"` ensures the pipeline handles them without errors.

**Why OrdinalEncoder over OneHotEncoder?**  
LightGBM natively handles ordinal-encoded categoricals efficiently through its `categorical_feature` parameter. One-hot encoding would significantly increase dimensionality with no benefit.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.2.2 | Data manipulation |
| numpy | 1.26.4 | Numerical operations |
| scikit-learn | 1.5.0 | Preprocessing, metrics, feature importance |
| lightgbm | 4.3.0 | Gradient boosting model |
| feature-engine | 1.9.4 | `ArbitraryNumberImputer` |
| matplotlib | 3.9.0 | Visualisations |
| seaborn | 0.13.2 | Enhanced plots |
| jupyter / notebook | 1.0.0 / 7.2.0 | Notebook environment |
| joblib | (scikit-learn dep) | Model serialisation |

---

## Known Issues and Limitations

- **Class imbalance**: ~65% of the training data are defaults. Future work could explore class weighting or resampling strategies.
- **Latest cohort dtype mismatch**: Several columns in `latest_customers.csv` have different dtypes (float instead of string) compared to `loan_data.csv`. The pipeline handles these gracefully through imputation and rare-grouping but the upstream data collection process should be reviewed.
- **Age calculation**: Age is computed using a simple `days / 365` approximation. Leap years are not accounted for, introducing a small error for some customers.

---

## License

This project is for educational and internal use only. The dataset is used under the IEEE Dataport terms of use.
