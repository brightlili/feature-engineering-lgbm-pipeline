# Notebook Analysis Report
## Credit Risk Feature Engineering Pipeline

---

## Executive Summary

The notebook was reviewed end-to-end across all 12 tasks. **5 bugs** were identified and fixed, ranging from a critical logic error in feature selection to deprecation warnings that will break in future pandas versions. Additionally, 2 redundant code blocks were cleaned up and 1 missing imputation entry was added.

---

## Bugs Found and Fixed

---

### Bug 1 — `inplace=True` FutureWarning (Multiple Locations)

**Severity:** Medium — works today, will silently break in pandas 3.0

**Where:** Task 6 (categorical imputation) and Task 11 pipeline function

**Original code:**
```python
X_train.fillna(value=imputation_cat_dict, inplace=True)
X_test.fillna(value=imputation_cat_dict, inplace=True)
```

**Problem:** pandas 3.0 is removing `inplace=True` support on chained operations. It already throws a `FutureWarning` and will raise an error in the next major release.

**Fix:**
```python
X_train = X_train.fillna(value=imputation_cat_dict)
X_test = X_test.fillna(value=imputation_cat_dict)
```

**Same fix applied inside `feature_engineering_pipe`:**
```python
# Before
df.fillna(IMPUTATION_DICT, inplace=True)

# After
df = df.fillna(IMPUTATION_DICT)
```

---

### Bug 2 — `dti` inf/nan Fix Applied Three Times (Redundant Cells)

**Severity:** Low — does not break results but creates confusing notebook state

**Where:** Task 3 — three consecutive cells all trying to fix the same problem

**What happened:** While debugging, three different approaches were tried and all left in the notebook. Only the first one (using `np.where`) is correct. The second and third cells run on an already-fixed column, making them harmless but misleading.

**Fix:** Kept only the correct cell:
```python
# Division by 0 creates nan and infinity — replace with 0
X_train["dti"] = np.where(np.isfinite(X_train["dti"]), X_train["dti"], 0)
X_test["dti"]  = np.where(np.isfinite(X_test["dti"]),  X_test["dti"],  0)
```

Removed: the `replace([inf, nan], median_dti)` cell and the `finite_vals.median()` cell.

---

### Bug 3 — Critical Logic Error in `feature_subset_1` (Feature Selection)

**Severity:** High — silently returns wrong results

**Where:** Task 10 — permutation importance feature selection

**Original code:**
```python
feature_subset_1 = list((result > 0).index)
```

**Problem:** `(result > 0)` returns a **boolean Series** (True/False). Calling `.index` on a boolean Series returns **all index labels**, regardless of whether the value is True or False. So `feature_subset_1` always contained every single feature — the filter had no effect.

**Demonstration:**
```python
dummy = pd.Series({'a': 0.1, 'b': -0.05, 'c': 0.3})
wrong   = list((dummy > 0).index)    # ['a', 'b', 'c'] — wrong!
correct = list(dummy[dummy > 0].index)  # ['a', 'c'] — correct
```

**Fix:**
```python
feature_subset_1 = list(perm_series[perm_series > 0].index)
```

The same bug existed in `feature_subset_2` for single-feature classifiers:
```python
# Before (wrong)
feature_subset_2 = list((result > 0.5).index)

# After (correct)
feature_subset_2 = list(single_feat_series[single_feat_series > 0.5].index)
```

---

### Bug 4 — Variable `result` Overwritten (Data Loss)

**Severity:** Medium — silently loses permutation importance data

**Where:** Task 10 — between permutation importance and single-feature classifiers

**Original code:**
```python
# Permutation importance
result = pd.Series(result.importances_mean, index=X_train.columns)

# ... later ...

# Single-feature classifiers — overwrites result!
result = {}
for var in X_train.columns:
    ...
result = pd.Series(result, index=X_train.columns)
```

**Problem:** The variable `result` is reused for the single-feature classifier dictionary, overwriting the permutation importance Series. After this point, the permutation importance data is lost.

**Fix:** Use separate, clearly named variables:
```python
perm_series        = pd.Series(perm_result.importances_mean, index=X_train.columns)
single_feat_result = {}   # separate dict for single-feature classifiers
single_feat_series = pd.Series(single_feat_result, index=X_train.columns)
```

---

### Bug 5 — `income_verification` Missing from `IMPUTATION_DICT`

**Severity:** Medium — causes `OrdinalEncoder` to fail on NaN values at prediction time

**Where:** Task 11 — pipeline constants

**Problem:** `income_verification` has 53 missing values in `loan_data.csv` and potentially more in `latest_customers.csv`. It was not included in `IMPUTATION_DICT`, meaning NaN values were passed directly into `OrdinalEncoder`, which cannot handle them.

**Fix:** Added to `IMPUTATION_DICT`:
```python
IMPUTATION_DICT = {
    ...
    'income_verification': 'missing',   # ← added
    ...
}
```

---

## Additional Observations

### `select_dtypes(include="O")` Deprecation Warning
In newer versions of pandas, `"O"` (object) dtype selection raises a warning. The fix is to use `include="object"` explicitly — this is minor but worth noting for future-proofing.

### `latest_customers.csv` — Dtype Mismatches
Several columns that are strings in `loan_data.csv` appear as `float64` in `latest_customers.csv` (e.g. `marital_status`, `employment_status`, `credit_score_3`). These columns contain numeric float values instead of string category labels. The pipeline handles this because:
1. Missing values are imputed to `"missing"` (a string)
2. Rare grouping converts unexpected values to `"Rare"`
3. `OrdinalEncoder` encodes the result

However, this is worth monitoring as the platform's data schema evolves.

---

## Summary Table

| # | Bug | Location | Severity | Status |
|---|-----|----------|----------|--------|
| 1 | `inplace=True` FutureWarning on `fillna` | Tasks 6 & 11 | Medium | ✅ Fixed |
| 2 | Three redundant `dti` fix cells | Task 3 | Low | ✅ Cleaned |
| 3 | Logic error in feature subset filter | Task 10 | **High** | ✅ Fixed |
| 4 | `result` variable overwritten | Task 10 | Medium | ✅ Fixed |
| 5 | `income_verification` missing from imputation dict | Task 11 | Medium | ✅ Fixed |
