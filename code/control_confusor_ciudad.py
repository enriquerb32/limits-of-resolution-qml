"""
control_confusor_ciudad.py — Diagnóstico de confusión ciudad vs física
=======================================================================
Pregunta a responder: la alta importancia de ciudad_Barcelona en el modelo
completo, ¿está "robándole" importancia a las variables termodinámicas
(choque_clausius, et0_sum) porque estas correlacionan con la identidad de
estación, o son predictores genuinamente independientes de qué ciudad es?

Método: reentrena XGBoost excluyendo las columnas one-hot de ciudad y
compara el ranking/gain de las variables termodinámicas antes y después.

Uso (en Colab, tras cargar ds desde el checkpoint 01_data.pt):
    from control_confusor_ciudad import comparar_importancias_con_sin_ciudad
    rank_full, rank_sin = comparar_importancias_con_sin_ciudad(ds)
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score


def _entrenar_xgboost_confusor(ds):
    """Train XGBoost and return (auc, predictions, importances, model).
    Self-contained to avoid depending on a non-existent modelos API."""
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
    Construye una copia superficial de los atributos de ds necesarios para
    XGBoost, con las columnas 'ciudad_*' eliminadas de X_sk/X_all.
    No modifica el objeto ds original (que puede reutilizarse en el resto
    del pipeline).
    """
    feature_names = (list(ds.feature_names) if hasattr(ds, 'feature_names')
                      else [f'f{i}' for i in range(ds.X_sk.shape[1])])

    mask_sin_ciudad = np.array([not nm.startswith('ciudad_') for nm in feature_names])
    ciudad_cols = [nm for nm in feature_names if nm.startswith('ciudad_')]

    if not ciudad_cols:
        print("  [!] No se encontraron columnas 'ciudad_*' en feature_names. "
              "Verifica que ds.feature_names está poblado y usa ese prefijo exacto.")

    print(f"  [Control] Excluyendo {len(ciudad_cols)} columnas de ciudad: {ciudad_cols}")

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
    """Reentrena XGBoost sin las dummies de ciudad y devuelve su feature importance."""
    ds_f = _filtrar_columnas_ciudad(ds)
    _, _, importancias, _ = _entrenar_xgboost_confusor(ds_f)
    return importancias


def comparar_importancias_con_sin_ciudad(ds, variables_interes=None):
    """
    Corre XGBoost completo (con ciudad) y filtrado (sin ciudad), e imprime
    una tabla comparativa del rango/gain de las variables termodinámicas
    clave en ambos escenarios.

    Devuelve (rank_full, rank_sin), dos dicts {nombre: (rank, gain)}.
    """
    if variables_interes is None:
        variables_interes = [
            'choque_clausius', 'et0_sum', 'et0_lag3m',
            'potencial_convectivo', 'dp_std', 'ia_racha', 't_anomaly'
        ]

    print("\n" + "=" * 70)
    print("CONTROL: Feature importance XGBoost — CON vs SIN identidad de ciudad")
    print("=" * 70)

    print("\n  [1/2] Entrenando XGBoost completo (con ciudad_*)...")
    _, _, importancias_full, _ = _entrenar_xgboost_confusor(ds)

    print("\n  [2/2] Entrenando XGBoost sin columnas de ciudad...")
    importancias_sin_ciudad = entrenar_xgboost_sin_ciudad(ds)

    def _rank(importancias):
        ordenado = sorted(importancias.items(), key=lambda kv: kv[1], reverse=True)
        return {nm: (i + 1, val) for i, (nm, val) in enumerate(ordenado)}

    rank_full = _rank(importancias_full)
    rank_sin  = _rank(importancias_sin_ciudad)

    print(f"\n  {'Variable':<22} {'Rank CON':>9} {'Gain CON':>10}   {'Rank SIN':>9} {'Gain SIN':>10}   {'Δ Gain':>8}")
    print("  " + "-" * 78)
    for v in variables_interes:
        r_f, g_f = rank_full.get(v, ('N/A', 0.0))
        r_s, g_s = rank_sin.get(v, ('N/A', 0.0))
        delta = (g_s - g_f) if isinstance(r_f, int) and isinstance(r_s, int) else float('nan')
        print(f"  {v:<22} {str(r_f):>9} {g_f:>10.4f}   {str(r_s):>9} {g_s:>10.4f}   {delta:>+8.4f}")

    print("\n  Top-5 predictores CON ciudad:")
    for nm, val in sorted(importancias_full.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        print(f"    {nm:<22} {val:.4f}")

    print("\n  Top-5 predictores SIN ciudad:")
    for nm, val in sorted(importancias_sin_ciudad.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        print(f"    {nm:<22} {val:.4f}")

    # Veredicto automático simple: ¿sube choque_clausius al excluir ciudad?
    r_cc_full = rank_full.get('choque_clausius', (None, None))[0]
    r_cc_sin  = rank_sin.get('choque_clausius', (None, None))[0]
    if isinstance(r_cc_full, int) and isinstance(r_cc_sin, int):
        print(f"\n  [Veredicto] choque_clausius: rank #{r_cc_full} (con ciudad) -> #{r_cc_sin} (sin ciudad)")
        if r_cc_sin < r_cc_full:
            print("  → Sube al excluir ciudad: consistente con un efecto de confusión parcial "
                  "(la física estaba parcialmente enmascarada por el efecto de estación).")
        elif r_cc_sin > r_cc_full:
            print("  → Baja o se mantiene al excluir ciudad: la importancia de choque_clausius "
                  "no dependía de la identidad de estación; el argumento físico se sostiene mejor "
                  "de lo esperado.")
        else:
            print("  → Sin cambio de rango: resultado ambiguo, revisar valores de gain absolutos.")

    return rank_full, rank_sin
