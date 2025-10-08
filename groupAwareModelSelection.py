# ==
# Model selection for Fs_All using multiple feature sets
# Group-aware CV (by animal 'Num') for honest evaluation
# ==

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# -----------------------------
# Config
# -----------------------------
FILE_PATH = r"E:/Ryan Analysis/P9RA3M4_Pooled_refined_scaled.csv"
TARGET    = "Fs_All"
GROUP     = "Num"

# Raw columns
BRP, UNC13A, RBP = "Brp", "Unc13A", "RBP"
# Normalized columns (optional)
BRP_N, UNC13A_N, RBP_N = "Brp_norm", "Unc13A_norm", "RBP_norm"

RANDOM_SEED = 424242
N_SPLITS    = 5                 # GroupKFold folds
N_ITER_XGB  = 30                # random search iterations per model
N_ITER_RF   = 30
N_ITER_EN   = 40

SAVE_OUTPUTS = True

# -----------------------------
# Load & basic cleaning
# -----------------------------
df = pd.read_csv(FILE_PATH)
df = df.replace([np.inf, -np.inf], np.nan)
if GROUP not in df.columns:
    raise ValueError(f"Missing grouping column '{GROUP}' in CSV.")
df[GROUP] = df[GROUP].astype(str)

# -----------------------------
# Feature engineering
# -----------------------------
EPS = 1e-6

def add_ratios(dfin, a, b, c, prefix=""):
    """Create pairwise ratios a/b, a/c, b/c. Returns list of new col names."""
    cols = []
    r1 = f"{prefix}Brp_Unc13A"; dfin[r1] = dfin[a] / (dfin[b] + EPS); cols.append(r1)
    r2 = f"{prefix}Brp_RBP";    dfin[r2] = dfin[a] / (dfin[c] + EPS); cols.append(r2)
    r3 = f"{prefix}Unc13A_RBP"; dfin[r3] = dfin[b] / (dfin[c] + EPS); cols.append(r3)
    return cols

# Build raw ratios
ratio_cols_raw = add_ratios(df, BRP, UNC13A, RBP, prefix="")

# Build normalized ratios if possible
has_norm = all(col in df.columns for col in [BRP_N, UNC13A_N, RBP_N])
ratio_cols_norm = []
if has_norm:
    ratio_cols_norm = add_ratios(df, BRP_N, UNC13A_N, RBP_N, prefix="norm_")

# Define feature sets
feature_sets = {
    "raw":              [BRP, UNC13A, RBP],
    "ratios":           ratio_cols_raw,
    "raw+ratios":       [BRP, UNC13A, RBP] + ratio_cols_raw,
}
if has_norm:
    feature_sets["normalized"] = [BRP_N, UNC13A_N, RBP_N]
    feature_sets["normalized+normalized_ratios"] = [BRP_N, UNC13A_N, RBP_N] + ratio_cols_norm
else:
    print("NOTE: Normalized columns not found; skipping 'normalized' and 'normalized+normalized_ratios'.")

# -----------------------------
# Models & hyperparameter spaces
# -----------------------------
def get_search_spaces():
    models = {}

    # # ElasticNet (with scaling)
    # models["ElasticNet"] = {
    #     "estimator": Pipeline([
    #         ("scaler", StandardScaler(with_mean=True, with_std=True)),
    #         ("model", ElasticNet(max_iter=20000, random_state=RANDOM_SEED))
    #     ]),
    #     "params": {
    #         "model__alpha":   np.logspace(-3, 1, 50),    # 0.001 .. 10
    #         "model__l1_ratio": np.linspace(0.0, 1.0, 50) # 0..1
    #     },
    #     "n_iter": N_ITER_EN
    # }

    # # RandomForest
    # models["RandomForest"] = {
    #     "estimator": RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
    #     "params": {
    #         "n_estimators":      np.linspace(200, 800, 7, dtype=int),
    #         "max_depth":         [None] + list(np.linspace(3, 20, 10, dtype=int)),
    #         "min_samples_leaf":  [1, 2, 3, 5, 8, 10],
    #         "min_samples_split": [2, 4, 6, 10],
    #         "max_features":      ["auto", "sqrt", 0.5, 0.7, 0.9]
    #     },
    #     "n_iter": N_ITER_RF
    # }

    # XGBoost
    models["XGBoost"] = {
        "estimator": xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
        ),
        "params": {
            "n_estimators":      np.linspace(200, 1200, 11, dtype=int),
            "learning_rate":     np.logspace(-3, -0.3, 20),  # 0.001 .. ~0.5
            "max_depth":         np.linspace(2, 8, 7, dtype=int),
            "subsample":         np.linspace(0.6, 1.0, 9),
            "colsample_bytree":  np.linspace(0.6, 1.0, 9),
            "reg_lambda":        np.logspace(-3, 2, 10),     # L2
            "min_child_weight":  np.linspace(1, 20, 10, dtype=int),
        },
        "n_iter": N_ITER_XGB
    }

    return models

# -----------------------------
# Utilities
# -----------------------------
def prepare_matrix(df_in, features):
    use = df_in[[*features, TARGET, GROUP]].copy()
    # coerce numeric
    for c in features + [TARGET]:
        use[c] = pd.to_numeric(use[c], errors="coerce")
    use = use.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [TARGET, GROUP])
    X = use[features].astype(float)
    y = use[TARGET].astype(float)
    groups = use[GROUP].astype(str)
    return X, y, groups, use

def grouped_cv():
    return GroupKFold(n_splits=N_SPLITS)

def run_random_search(name, estimator, param_dist, X, y, groups, n_iter):
    cv = grouped_cv()
    scoring = {"r2": "r2", "neg_mae": "neg_mean_absolute_error"}
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        refit="neg_mae",                 # refit the best (lowest) MAE
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
    )
    search.fit(X, y, **{"groups": groups})
    # Fetch both metrics for the refit (best index by neg_mae)
    best_idx = search.best_index_
    mean_r2  = search.cv_results_["mean_test_r2"][best_idx]
    mean_mae = -search.cv_results_["mean_test_neg_mae"][best_idx]
    return search, float(mean_r2), float(mean_mae)

def oof_grouped_metrics(estimator, X, y, groups):
    """Compute out-of-fold predictions with GroupKFold to report honest R²/MAE."""
    gkf = grouped_cv()
    y_true_all, y_pred_all = [], []
    for tr, te in gkf.split(X, y, groups):
        est = estimator.__class__(**estimator.get_params())
        est.fit(X.iloc[tr], y.iloc[tr])
        pred = est.predict(X.iloc[te])
        y_true_all.append(y.iloc[te].values)
        y_pred_all.append(pred)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    return y_true_all, y_pred_all, r2_score(y_true_all, y_pred_all), mean_absolute_error(y_true_all, y_pred_all)

# -----------------------------
# Main search over feature sets & models
# -----------------------------
models = get_search_spaces()
all_rows = []
best_tracker = {"mae": np.inf, "record": None}

for fs_name, cols in feature_sets.items():
    print(f"\n===== Feature set: {fs_name} | {len(cols)} features =====")
    X, y, groups, used_df = prepare_matrix(df, cols)

    for mdl_name, cfg in models.items():
        print(f"  > Tuning {mdl_name} ...")
        search, cv_r2, cv_mae = run_random_search(
            f"{fs_name}-{mdl_name}",
            cfg["estimator"],
            cfg["params"],
            X, y, groups,
            cfg["n_iter"],
        )
        row = {
            "feature_set": fs_name,
            "model": mdl_name,
            "best_params": search.best_params_,
            "cv_mean_R2": cv_r2,
            "cv_mean_MAE": cv_mae,
            "n_features": len(cols),
        }
        all_rows.append(row)
        print(f"    cv_mean_R2={cv_r2:.3f}  cv_mean_MAE={cv_mae:.4f}")

        # Track global best by MAE
        if cv_mae < best_tracker["mae"]:
            best_tracker["mae"] = cv_mae
            best_tracker["record"] = {
                "feature_set": fs_name,
                "model": mdl_name,
                "search": search,
                "X": X, "y": y, "groups": groups,
                "features": cols
            }

results_df = pd.DataFrame(all_rows).sort_values(["cv_mean_MAE", "cv_mean_R2"], ascending=[True, False])
print("\n=== Summary of model tuning (GroupKFold CV) ===")
print(results_df.to_string(index=False))

# -----------------------------
# Evaluate the overall winner with OOF & save artifacts
# -----------------------------
print("\n=== Best overall (by CV MAE) ===")
best = best_tracker["record"]
print(f"Feature set: {best['feature_set']}")
print(f"Model:       {best['model']}")
print(f"Best params: {best['search'].best_params_}")

best_estimator = best["search"].best_estimator_
Xb, yb, gb = best["X"], best["y"], best["groups"]

# Honest OOF metrics using fixed GroupKFold (match reporting style)
y_true_oof, y_pred_oof, r2_oof, mae_oof = oof_grouped_metrics(best_estimator, Xb, yb, gb)
print(f"\nOOF (GroupKFold) — {best['model']} on {best['feature_set']}:")
print(f"R²_groupedCV = {r2_oof:.3f}")
print(f"MAE_groupedCV = {mae_oof:.4f}")

# Per-animal metrics
oof_df = pd.DataFrame({
    "Num": gb.values,
    "Fs_All_true": y_true_oof,
    "Fs_All_pred": y_pred_oof
})
oof_df["abs_err"] = np.abs(oof_df["Fs_All_true"] - oof_df["Fs_All_pred"])

def _per_animal_metrics(g):
    yt = g["Fs_All_true"].values
    yp = g["Fs_All_pred"].values
    return pd.Series({
        "n": len(g),
        "R2": r2_score(yt, yp) if len(g) > 1 else np.nan,
        "MAE": mean_absolute_error(yt, yp)
    })

per_animal = (
    oof_df.groupby("Num", group_keys=False)[["Fs_All_true", "Fs_All_pred"]]
          .apply(_per_animal_metrics)
          .reset_index()
          .sort_values("MAE")
)

print("\nPer-animal performance (OOF):")
print(per_animal.to_string(index=False))

# Fit on FULL data to inspect importances/coefs
best_estimator.fit(Xb, yb)

feat_table = None
if best["model"] == "ElasticNet":
    # get coefficients from pipeline
    model = best_estimator.named_steps["model"]
    coefs = model.coef_
    feat_table = pd.DataFrame({"feature": best["features"], "coefficient": coefs}) \
                   .sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True)
elif best["model"] == "RandomForest":
    model = best_estimator
    if hasattr(model, "feature_importances_"):
        feat_table = pd.DataFrame({"feature": best["features"], "importance": model.feature_importances_}) \
                       .sort_values("importance", ascending=False).reset_index(drop=True)
elif best["model"] == "XGBoost":
    model = best_estimator
    if hasattr(model, "feature_importances_"):
        feat_table = pd.DataFrame({"feature": best["features"], "importance": model.feature_importances_}) \
                       .sort_values("importance", ascending=False).reset_index(drop=True)

if feat_table is not None:
    print("\nFeature importances / coefficients (fit on full data):")
    print(feat_table.to_string(index=False))

# Predicted vs Actual plot (OOF)
plt.figure(figsize=(6,6))
plt.scatter(y_true_oof, y_pred_oof, alpha=0.6)
lims = [min(y_true_oof.min(), y_pred_oof.min()), max(y_true_oof.max(), y_pred_oof.max())]
plt.plot(lims, lims, "--")
plt.xlabel("Actual Fs_All")
plt.ylabel("Predicted Fs_All")
plt.title(f"{best['model']} on {best['feature_set']} — OOF\nR²={r2_oof:.3f}, MAE={mae_oof:.4f}")
plt.tight_layout()

out_dir = os.path.dirname(FILE_PATH)
if SAVE_OUTPUTS:
    results_df.to_csv(os.path.join(out_dir, "tuning_summary_all_feature_sets.csv"), index=False)
    oof_df.to_csv(os.path.join(out_dir, "best_model_oof_predictions.csv"), index=False)
    per_animal.to_csv(os.path.join(out_dir, "best_model_per_animal_metrics.csv"), index=False)
    if feat_table is not None:
        feat_table.to_csv(os.path.join(out_dir, "best_model_feature_importance.csv"), index=False)
    plt.savefig(os.path.join(out_dir, "best_model_pred_vs_actual.png"), dpi=150)
    print(f"\nSaved outputs to: {out_dir}")
else:
    plt.show()
