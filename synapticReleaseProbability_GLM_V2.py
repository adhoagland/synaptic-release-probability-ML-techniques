import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv("E:/Ryan Analysis/P9RA3M4_Pooled_refined_scaled.csv")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error

# === SETTINGS ===
target = "Fs_All"
predictors = ["Unc13A", "RBP", "Brp"]
group_col = "Num"
cv_splits = 5

# === Load your data ===
# df = pd.read_csv("your_data.csv")
df["Num"] = df["Num"].astype(str)

# === Storage for results ===
results = []

# === Loop over each animal ===
for animal_id, group_df in df.groupby(group_col):
    X = group_df[predictors].values
    y = group_df[target].values

    if len(group_df) < cv_splits:
        print(f"Skipping {animal_id}: not enough data for {cv_splits}-fold CV")
        continue

    # === Cross-validation splitter ===
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # === Linear Regression CV ===
    lin_model = LinearRegression()
    r2_lin_cv = cross_val_score(lin_model, X, y, cv=kf, scoring='r2').mean()
    rmse_lin_cv = -cross_val_score(lin_model, X, y, cv=kf, scoring='neg_root_mean_squared_error').mean()

    # Fit once to extract coefficients
    lin_model.fit(X, y)
    lin_coefs = lin_model.coef_

    # = == Decision Tree CV ===
    tree_model = DecisionTreeRegressor(
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=42
    )
    r2_tree_cv = cross_val_score(tree_model, X, y, cv=kf, scoring='r2').mean()
    rmse_tree_cv = -cross_val_score(tree_model, X, y, cv=kf, scoring='neg_root_mean_squared_error').mean()

    # Fit once to extract importances
    tree_model.fit(X, y)
    tree_imps = tree_model.feature_importances_

    # === Store results ===
    result = {
        "animal_id": animal_id,
        "r2_lin": r2_lin_cv,
        "rmse_lin": rmse_lin_cv,
        "r2_tree": r2_tree_cv,
        "rmse_tree": rmse_tree_cv,
    }

    for name, coef in zip(predictors, lin_coefs):
        result[f"lin_coef_{name}"] = coef

    for name, imp in zip(predictors, tree_imps):
        result[f"tree_imp_{name}"] = imp

    results.append(result)

# === Compile Results ===
results_df = pd.DataFrame(results)

# === Output ===
print("\nPer-animal cross-validated model performance:")
print(results_df[["animal_id", "r2_lin", "r2_tree", "rmse_lin", "rmse_tree"]])

print("\n=== Average Feature Effects Across Animals ===")
print("\nLinear model average coefficients:")
print(results_df[[f"lin_coef_{f}" for f in predictors]].mean().round(4))

print("\nTree model average importances (deep trees):")
print(results_df[[f"tree_imp_{f}" for f in predictors]].mean().round(4))

# Optional: save results
# results_df.to_csv("per_animal_cv_model_comparison.csv", index=False)
