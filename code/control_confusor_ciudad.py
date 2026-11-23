"""
control_confusor_ciudad.py — Station Confounder vs Physics Diagnostic
======================================================================
Research question: Does the high feature importance of city indicators (e.g. ciudad_Barcelona)
in the full model confound the thermodynamic variables (choque_clausius, et0_sum), or are
the physical indicators genuinely independent of station identity?

Method: Retrains XGBoost excluding one-hot station indicators and compares the
ranking/gain of physical features before and after.

Usage (after loading ds from checkpoint 01_data.pt):
    from control_confusor_ciudad import comparar_importancias_con_sin_ciudad
    rank_full, rank_sin = comparar_importancias_con_sin_ciudad(ds)
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score


def _entrenar_xgboost_confusor(ds):
    """Train XGBoost and return (auc, predictions, importances, model).
    Self-contained to avoid depending on external model APIs."""
    from xgboost import XGBClassifier

    feature_names = (list(ds.feature_names) if hasattr(ds, 'feature_names')
                      else [f'f{i}' for i in range(ds.X_sk.shape[1])])

    imp = SimpleImputer(strategy='mean')
    X_tr_clean = imp.fit_transform(ds.X_sk)
    X_te_clean = imp.transform(ds.X_all[ds.s2:])

    n_neg = (ds.y_sk == 0).sum()
    n_pos = (ds.y_sk == 1).sum()
    xgb = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric='logloss', random_state=42, n_jobs=-1
    )
    xgb.fit(X_tr_clean, ds.y_sk, verbose=False)

    y_te = ds.y_te.numpy() if hasattr(ds.y_te, 'numpy') else ds.y_te
    p_xgb = xgb.predict_proba(X_te_clean)[:, 1]
    auc = roc_auc_score(y_te, p_xgb)
    importancias = dict(zip(feature_names, xgb.feature_importances_))

    return auc, p_xgb, importancias, xgb


def _filtrar_columnas_ciudad(ds):
    """
    Constructs a shallow copy of ds attributes needed for XGBoost,
    with 'ciudad_*' columns removed from X_sk/X_all.
    Does not mutate the original ds instance.
    """
    feature_names = (list(ds.feature_names) if hasattr(ds, 'feature_names')
                      else [f'f{i}' for i in range(ds.X_sk.shape[1])])

    mask_sin_ciudad = np.array([not nm.startswith('ciudad_') for nm in feature_names])
    ciudad_cols = [nm for nm in feature_names if nm.startswith('ciudad_')]

    if not ciudad_cols:
        print("  [!] No 'ciudad_*' columns found in feature_names. "
              "Verify that ds.feature_names is populated with that exact prefix.")

    print(f"  [Control] Excluding {len(ciudad_cols)} station columns: {ciudad_cols}")

    class _DatasetFiltrado:
        pass

    ds_f = _DatasetFiltrado()
    ds_f.X_sk  = ds.X_sk[:, mask_sin_ciudad]
    ds_f.X_all = ds.X_all[:, mask_sin_ciudad]
    ds_f.y_sk  = ds.y_sk
    ds_f.y_va  = ds.y_va
    ds_f.y_te  = ds.y_te
    ds_f.s1    = ds.s1
    ds_f.s2    = ds.s2
    ds_f.feature_names = [nm for nm, keep in zip(feature_names, mask_sin_ciudad) if keep]

    return ds_f


def entrenar_xgboost_sin_ciudad(ds):
    """Retrains XGBoost without station dummy features and returns feature importances."""
    ds_f = _filtrar_columnas_ciudad(ds)
    _, _, importancias, _ = _entrenar_xgboost_confusor(ds_f)
    return importancias


def comparar_importancias_con_sin_ciudad(ds, variables_interes=None):
    """
    Runs full XGBoost (with station indicators) and filtered XGBoost (without station indicators),
    displaying a comparative gain/rank table of key physical variables.

    Returns (rank_full, rank_sin), two dicts {feature_name: (rank, gain)}.
    """
    if variables_interes is None:
        variables_interes = [
            'choque_clausius', 'et0_sum', 'et0_lag3m',
            'potencial_convectivo', 'dp_std', 'ia_racha', 't_anomaly'
        ]

    print("\n" + "=" * 70)
    print("CONTROL: XGBoost Feature Importance — WITH vs WITHOUT Station Indicators")
    print("=" * 70)

    print("\n  [1/2] Training full XGBoost (with ciudad_*)...")
    _, _, importancias_full, _ = _entrenar_xgboost_confusor(ds)

    print("\n  [2/2] Training XGBoost without station columns...")
    importancias_sin_ciudad = entrenar_xgboost_sin_ciudad(ds)

    def _rank(importancias):
        ordenado = sorted(importancias.items(), key=lambda kv: kv[1], reverse=True)
        return {nm: (i + 1, val) for i, (nm, val) in enumerate(ordenado)}

    rank_full = _rank(importancias_full)
    rank_sin  = _rank(importancias_sin_ciudad)

    print(f"\n  {'Feature':<22} {'Rank WITH':>10} {'Gain WITH':>11}   {'Rank W/O':>10} {'Gain W/O':>11}   {'Delta Gain':>10}")
    print("  " + "-" * 82)
    for v in variables_interes:
        r_f, g_f = rank_full.get(v, ('N/A', 0.0))
        r_s, g_s = rank_sin.get(v, ('N/A', 0.0))
        delta = (g_s - g_f) if isinstance(r_f, int) and isinstance(r_s, int) else float('nan')
        print(f"  {v:<22} {str(r_f):>10} {g_f:>11.4f}   {str(r_s):>10} {g_s:>11.4f}   {delta:>+10.4f}")

    print("\n  Top-5 predictors WITH station indicators:")
    for nm, val in sorted(importancias_full.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        print(f"    {nm:<22} {val:.4f}")

    print("\n  Top-5 predictors WITHOUT station indicators:")
    for nm, val in sorted(importancias_sin_ciudad.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        print(f"    {nm:<22} {val:.4f}")

    # Automated check on whether physical features rise when excluding station identity
    r_cc_full = rank_full.get('choque_clausius', (None, None))[0]
    r_cc_sin  = rank_sin.get('choque_clausius', (None, None))[0]
    if isinstance(r_cc_full, int) and isinstance(r_cc_sin, int):
        print(f"\n  [Verdict] choque_clausius: rank #{r_cc_full} (with station) -> #{r_cc_sin} (without station)")
        if r_cc_sin < r_cc_full:
            print("  -> Rank rises without station identity: partial confounding present "
                  "(physics was partially masked by station-specific effects).")
        elif r_cc_sin > r_cc_full:
            print("  -> Rank holds or decreases: physical feature was not dependent on station identity; "
                  "the thermodynamic mechanism stands independently.")
        else:
            print("  -> Rank unchanged: inspect absolute gain values.")

    return rank_full, rank_sin


# English aliases for public API
train_xgboost_without_city = entrenar_xgboost_sin_ciudad
compare_feature_importances_with_without_city = comparar_importancias_con_sin_ciudad

