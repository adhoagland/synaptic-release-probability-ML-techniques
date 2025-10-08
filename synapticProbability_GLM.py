# ============================================================
# GLM (Gaussian identity) for Fs_All with grouped CV by animal
# Feature sets: raw, ratios, raw+ratios, normalized, normalized+normalized_ratios
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

import statsmodels.api as sm


FILE_PATH = r"E:/Ryan Analysis/P9RA3M4_Pooled_refined_scaled.csv"
TARGET, GROUP = "Fs_All", "Num"

# Raw cols
BRP, U, R = "Brp", "Unc13A", "RBP"
# Normalized cols (optional)
BRP_N, U_N, R_N = "Brp_norm", "Unc13A_norm", "RBP_norm"

RANDOM_SEED = 424242
N_SPLITS = 5
EPS = 1e-6
SCALE_X = True   # standardize X within each train fold (recommended for stability)

SAVE_OUTPUTS = True

# Load & basic clean
df = pd.read_csv(FILE_PATH)
df = df.replace([np.inf, -np.inf], np.nan)
for c in [BRP, U, R, TARGET]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
if GROUP not in df.columns:
    raise ValueError(f"Missing grouping column '{GROUP}' in CSV.")
df[GROUP] = df[GROUP].astype(str)
df = df.dropna(subset=[BRP, U, R, TARGET, GROUP]).copy()

# Feature engineering
def add_ratios(dfin, a, b, c, prefix=""):
    cols = []
    r1 = f"{prefix}Brp_Unc13A";  dfin[r1] = dfin[a] / (dfin[b].abs() + EPS); cols.append(r1)
    r2 = f"{prefix}Brp_RBP";     dfin[r2] = dfin[a] / (dfin[c].abs() + EPS); cols.append(r2)
    r3 = f"{prefix}Unc13A_RBP";  dfin[r3] = dfin[b] / (dfin[c].abs() + EPS); cols.append(r3)
    return cols

ratio_cols_raw = add_ratios(df, BRP, U, R, prefix="")
has_norm = all(col in df.columns for col in [BRP_N, U_N, R_N])
ratio_cols_norm = add_ratios(df, BRP_N, U_N, R_N, prefix="norm_") if has_norm else []

feature_sets = {
    "raw":            [BRP, U, R],
    "ratios":         ratio_cols_raw,
    "raw+ratios":     [BRP, U, R] + ratio_cols_raw,
}
if has_norm:
    feature_sets["normalized"] = [BRP_N, U_N, R_N]
    feature_sets["normalized+normalized_ratios"] = [BRP_N, U_N, R_N] + ratio_cols_norm
else:
    print("NOTE: *_norm columns not found; skipping normalized feature sets.")

# GLM helpers (Gaussian, identity)
def fit_glm_gaussian(X_tr, y_tr):
    X_tr_const = sm.add_constant(X_tr, has_constant="add")
    model = sm.GLM(y_tr, X_tr_const, family=sm.families.Gaussian(sm.families.links.identity()))
    res = model.fit()
    return res

def predict_glm(res, X_te):
    X_te_const = sm.add_constant(X_te, has_constant="add")
    return res.predict(X_te_const)

def grouped_oof_glm(X, y, groups, n_splits=5, scale_X=True):
    gkf = GroupKFold(n_splits=n_splits)
    y_true_all, y_pred_all = [], []
    fold_stats = []
    for i, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_te = X.iloc[tr].values, X.iloc[te].values
        y_tr, y_te = y.iloc[tr].values, y.iloc[te].values

        scaler = None
        if scale_X:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        res = fit_glm_gaussian(X_tr, y_tr)
        pred = predict_glm(res, X_te)

        r2  = r2_score(y_te, pred)
        mae = mean_absolute_error(y_te, pred)
        fold_stats.append((i, len(te), r2, mae))

        y_true_all.append(y_te)
        y_pred_all.append(pred)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    r2_oof = r2_score(y_true_all, y_pred_all)
    mae_oof = mean_absolute_error(y_true_all, y_pred_all)
    return y_true_all, y_pred_all, fold_stats, r2_oof, mae_oof

def logo_oof_glm(X, y, groups, scale_X=True):
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []
    for tr, te in logo.split(X, y, groups):
        X_tr, X_te = X.iloc[tr].values, X.iloc[te].values
        y_tr, y_te = y.iloc[tr].values, y.iloc[te].values

        scaler = None
        if scale_X:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        res = fit_glm_gaussian(X_tr, y_tr)
        pred = predict_glm(res, X_te)

        y_true_all.append(y_te)
        y_pred_all.append(pred)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    return r2_score(y_true_all, y_pred_all), mean_absolute_error(y_true_all, y_pred_all)

# Run across feature sets

all_rows = []
best = {"mae": np.inf, "name": None, "X": None, "y": None, "groups": None, "features": None}

for fs_name, cols in feature_sets.items():
    # Prepare matrix
    use = df[[*cols, TARGET, GROUP]].copy()
    use = use.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + [TARGET, GROUP])
    for c in cols + [TARGET]:
        use[c] = pd.to_numeric(use[c], errors="coerce")
    use = use.dropna(subset=cols + [TARGET]).copy()

    X = use[cols].astype(float)
    y = use[TARGET].astype(float)
    groups = use[GROUP].astype(str)

    print(f"\n===== GLM on feature set: {fs_name} ({len(cols)} features) =====")
    # Grouped CV (main)
    y_true, y_pred, folds, r2_oof, mae_oof = grouped_oof_glm(X, y, groups, n_splits=N_SPLITS, scale_X=SCALE_X)
    for i, n, r2f, maef in folds:
        print(f"Fold {i}: n={n}  R2={r2f:.3f}  MAE={maef:.4f}")
    print(f"OOF (GroupKFold): R2={r2_oof:.3f}  MAE={mae_oof:.4f}")

    # In-sample (optimistic)
    X_all = X.values
    scaler_full = StandardScaler() if SCALE_X else None
    if scaler_full:
        X_all = scaler_full.fit_transform(X_all)
    res_full = fit_glm_gaussian(X_all, y.values)
    y_fit = predict_glm(res_full, X_all)
    r2_in = r2_score(y, y_fit); mae_in = mean_absolute_error(y, y_fit)
    print(f"In-sample: R2={r2_in:.3f}  MAE={mae_in:.4f}")

    # LOGO (leave one animal out)
    r2_logo, mae_logo = logo_oof_glm(X, y, groups, scale_X=SCALE_X)
    print(f"LOGO: R2={r2_logo:.3f}  MAE={mae_logo:.4f}")

    # Per-animal metrics from grouped OOF
    oof_df = pd.DataFrame({"Num": groups.values, "y_true": y_true, "y_pred": y_pred})
    def _per_animal(gp):
        yt, yp = gp["y_true"].values, gp["y_pred"].values
        return pd.Series({"n": len(gp),
                          "R2": r2_score(yt, yp) if len(gp) > 1 else np.nan,
                          "MAE": mean_absolute_error(yt, yp)})
    per_animal = (oof_df.groupby("Num", group_keys=False)[["y_true","y_pred"]]
                        .apply(_per_animal)
                        .reset_index()
                        .sort_values("MAE"))
    print("\nPer-animal (OOF, grouped CV):")
    print(per_animal.to_string(index=False))

    all_rows.append({
        "feature_set": fs_name,
        "R2_groupedCV": r2_oof,
        "MAE_groupedCV": mae_oof,
        "R2_in_sample": r2_in,
        "MAE_in_sample": mae_in,
        "R2_LOGO": r2_logo,
        "MAE_LOGO": mae_logo,
        "n_features": len(cols),
        "n_rows": len(use),
        "n_animals": use[GROUP].nunique()
    })

    if mae_oof < best["mae"]:
        best.update(mae=mae_oof, name=fs_name, X=X, y=y, groups=groups, features=cols,
                    oof_df=oof_df, res_full=res_full, scaler_full=scaler_full)


summary = pd.DataFrame(all_rows).sort_values(["MAE_groupedCV","R2_groupedCV"], ascending=[True, False])
print("\n=== GLM Summary (sorted by MAE_groupedCV) ===")
print(summary.to_string(index=False))

print("\n=== BEST FEATURE SET (by MAE_groupedCV) ===")
print(f"Best: {best['name']}  |  MAE_groupedCV={best['mae']:.4f}")

print("\nCoefficients on FULL data (best feature set):")
params = best["res_full"].params
coef_table = pd.DataFrame({"term": ["const"] + list(best["features"]),
                           "coef": params})
print(coef_table.to_string(index=False))

plt.figure(figsize=(6,6))
plt.scatter(best["oof_df"]["y_true"], best["oof_df"]["y_pred"], alpha=0.6)
lims = [min(best["oof_df"]["y_true"].min(), best["oof_df"]["y_pred"].min()),
        max(best["oof_df"]["y_true"].max(), best["oof_df"]["y_pred"].max())]
plt.plot(lims, lims, "--")
plt.xlabel("Actual Fs_All")
plt.ylabel("Predicted Fs_All")
plt.title(f"GLM (Gaussian) — {best['name']} OOF\nMAE={best['mae']:.4f}")
plt.tight_layout()

# Save 
out_dir = os.path.dirname(FILE_PATH)
if SAVE_OUTPUTS:
    summary.to_csv(os.path.join(out_dir, "glm_summary_across_feature_sets.csv"), index=False)
    best["oof_df"].to_csv(os.path.join(out_dir, f"glm_oof_predictions_{best['name']}.csv"), index=False)
    coef_table.to_csv(os.path.join(out_dir, f"glm_full_coefficients_{best['name']}.csv"), index=False)
    plt.savefig(os.path.join(out_dir, f"glm_pred_vs_actual_{best['name']}.png"), dpi=150)
    print(f"\nSaved outputs to: {out_dir}")
else:
    plt.show()
