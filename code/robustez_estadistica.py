"""
robustez_estadistica.py — Statistical Robustness and Multi-Seed Replication
===========================================================================
Self-contained module providing:
  1. bootstrap_pareado_operativo()   — 95% paired bootstrap CI of Rc-p90/Rc-ACP/FPR deltas
  2. ejecutar_multisemilla_qml()     — Multi-seed QML replication with operational metrics,
                                        checkpoint/resume capability, and progress persistence
  3. veredicto_orden_dial()          — Model order stability verdict across random seeds

Usage:
    from modelos import entrenar_qml, entrenar_qtl, entrenar_qnn_layerbylayer

    resumen = ejecutar_multisemilla_qml(
        ds, res_base, entrenar_qml, entrenar_qtl, entrenar_qnn_layerbylayer,
        seeds=range(15),
        guardar_en='./resultados/checkpoints/09_multisemilla.pt',
    )
    veredicto_orden_dial(resumen, metrica='fpr')
    veredicto_orden_dial(resumen, metrica='rc_ACP')
"""

import time
import numpy as np
from sklearn.metrics import recall_score, confusion_matrix

import math

METRICAS_SEGUIDAS = ['acc', 'f1', 'auc', 'fpr', 'rc_ACP', 'rc_p90']

def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score confidence interval for binomial proportions."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1.0 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ══════════════════════════════════════════════════════════════════════════
# 1. PAIRED BOOTSTRAP AT OPERATIONAL WORKING POINT
# ══════════════════════════════════════════════════════════════════════════

def bootstrap_pareado_operativo(y_te, y_te_ext, ACP_te, prob_a, prob_b,
                                  umbral_a, umbral_b,
                                  nombre_a: str = "Model A", nombre_b: str = "Model B",
                                  n_boot: int = 2000, seed: int = 42,
                                  imprimir: bool = True) -> dict:
    """95% CI (paired bootstrap) for the difference (B - A) in Rc-p90, Rc-ACP,
    and FPR, evaluating each model at its predetermined operating threshold."""
    rng = np.random.RandomState(seed)
    n = len(y_te)

    pred_a_full = (np.asarray(prob_a) >= umbral_a).astype(float)
    pred_b_full = (np.asarray(prob_b) >= umbral_b).astype(float)

    diffs = {'rc_p90': [], 'rc_ACP': [], 'fpr': []}

    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb, yxb, db = y_te[idx], y_te_ext[idx], ACP_te[idx]
        pa, pb = pred_a_full[idx], pred_b_full[idx]

        rc_p90_a = recall_score(yxb, pa, zero_division=0)
        rc_p90_b = recall_score(yxb, pb, zero_division=0)
        diffs['rc_p90'].append(rc_p90_b - rc_p90_a)

        if db.sum() > 0:
            rc_ACP_a = recall_score(yb[db == 1], pa[db == 1], zero_division=0)
            rc_ACP_b = recall_score(yb[db == 1], pb[db == 1], zero_division=0)
            diffs['rc_ACP'].append(rc_ACP_b - rc_ACP_a)

        tn_a, fp_a, fn_a, tp_a = confusion_matrix(yb, pa, labels=[0, 1]).ravel()
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
        fpr_a = fp_a / (fp_a + tn_a) if (fp_a + tn_a) > 0 else np.nan
        fpr_b = fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else np.nan
        diffs['fpr'].append(fpr_b - fpr_a)

    resultados = {}
    for k, vals in diffs.items():
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            resultados[k] = {'delta_medio': float('nan'), 'ic95': (float('nan'), float('nan')),
                              'significativo': False, 'n_boot_validos': 0}
            continue
        lo, hi = np.percentile(vals, [2.5, 97.5])
        resultados[k] = {
            'delta_medio': float(vals.mean()),
            'ic95': (float(lo), float(hi)),
            'significativo': not (lo <= 0.0 <= hi),
            'n_boot_validos': int(len(vals)),
        }

    if imprimir:
        print(f"\n  Paired bootstrap ({n_boot} resamples): Delta = {nombre_b} - {nombre_a}")
        print("  " + "-" * 78)
        for k, r in resultados.items():
            etiqueta = {'rc_p90': 'Rc-p90 ', 'rc_ACP': 'Rc-ACP', 'fpr': 'FPR    '}[k]
            sig = "significant (CI excludes 0)" if r['significativo'] else "NOT significant (CI spans 0)"
            print(f"  Delta-{etiqueta} = {r['delta_medio']:+.4f}   "
                  f"95% CI=[{r['ic95'][0]:+.4f}, {r['ic95'][1]:+.4f}]   [{sig}]   (n={r['n_boot_validos']})")
        print()

    return resultados


# ══════════════════════════════════════════════════════════════════════════
# 2. MULTI-SEED: Operational Metrics + Checkpoint/Resume
# ══════════════════════════════════════════════════════════════════════════

def _metricas_operativas(y_te, y_te_ext, ACP_te, prob, umbral: float = 0.5) -> dict:
    """Computes fpr/rc_ACP/rc_p90 from predicted probabilities at nominal threshold 0.5."""
    prob = np.asarray(prob)
    pred = (prob >= umbral).astype(float)

    rc_p90 = recall_score(y_te_ext, pred, zero_division=0)
    if ACP_te.sum() > 0:
        rc_ACP = recall_score(y_te[ACP_te == 1], pred[ACP_te == 1], zero_division=0)
    else:
        rc_ACP = float('nan')

    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')

    return {'fpr': float(fpr), 'rc_ACP': float(rc_ACP), 'rc_p90': float(rc_p90)}


def _construir_resumen(acumulado: dict, tiempos: dict, semillas_completadas: set) -> dict:
    """Builds summary dictionary (mean/std/min/max) from accumulated metric runs."""
    resumen = {}
    for nm, metricas in acumulado.items():
        resumen[nm] = {}
        for m, vals in metricas.items():
            arr = np.asarray(vals, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) > 0:
                resumen[nm][m] = {
                    'mean': float(arr.mean()), 'std': float(arr.std()),
                    'min': float(arr.min()), 'max': float(arr.max()),
                    'valores': arr.tolist(),
                }
    resumen['_tiempos_min'] = {nm: float(np.mean(t)) / 60 for nm, t in tiempos.items() if t}
    resumen['_estado_interno'] = {
        'acumulado': acumulado,
        'tiempos': tiempos,
        'semillas_completadas': semillas_completadas,
    }
    return resumen


def ejecutar_multisemilla_qml(ds, res_base, entrenar_qml, entrenar_qtl,
                                entrenar_qnn_layerbylayer, seeds=range(15),
                                guardar_en: str = None,
                                descargar_cada_semilla: bool = False,
                                frecuencia_descarga: int = 1) -> dict:
    """Replicates QML models across multiple seeds, recording operational metrics.
    Automatically resumes from existing checkpoint if available."""
    import torch
    import os
    import gc

    seeds = list(seeds)
    n_seeds = len(seeds)

    acumulado = {}
    tiempos = {}
    semillas_completadas = set()

    y_te = ds.y_te.numpy() if hasattr(ds.y_te, 'numpy') else ds.y_te
    y_te_ext = ds.y_te_ext.numpy() if hasattr(ds.y_te_ext, 'numpy') else ds.y_te_ext
    ACP_te = ds.ACP_te.numpy() if hasattr(ds.ACP_te, 'numpy') else ds.ACP_te

    if guardar_en and os.path.exists(guardar_en):
        print(f"  [+] Found previous checkpoint at {guardar_en}. Resuming...")
        try:
            res_previo = torch.load(guardar_en, weights_only=False)
            if '_estado_interno' in res_previo:
                acumulado = res_previo['_estado_interno']['acumulado']
                tiempos = res_previo['_estado_interno']['tiempos']
                semillas_completadas = set(res_previo['_estado_interno']['semillas_completadas'])
                print(f"  [+] Resumed successfully. {len(semillas_completadas)} seeds already completed.")
        except Exception as e:
            print(f"  [!] Failed to load checkpoint: {e}")

    tiene_qtl = '_modelo_lstm' in res_base
    if not tiene_qtl:
        print("  [!] res_base lacks '_modelo_lstm' — QTL will be skipped.")

    for i, seed in enumerate(seeds):
        if seed in semillas_completadas:
            print(f"\n  -- Seed {seed} ({i + 1}/{n_seeds}) -- [ALREADY COMPLETED, SKIPPING]")
            continue

        print(f"\n  -- Seed {seed} ({i + 1}/{n_seeds}) --")

        # --- LSTM+QNN and LSTM+PI-QNN ---
        torch.manual_seed(seed); np.random.seed(seed)
        t0 = time.time()
        res_qml = entrenar_qml(ds)
        dt_qml = time.time() - t0
        for nm in ('LSTM+QNN', 'LSTM+PI-QNN'):
            if nm not in res_qml:
                continue
            if '_prob' in res_qml[nm]:
                res_qml[nm].update(_metricas_operativas(y_te, y_te_ext, ACP_te, res_qml[nm]['_prob']))
            acumulado.setdefault(nm, {m: [] for m in METRICAS_SEGUIDAS})
            tiempos.setdefault(nm, [])
            for m in METRICAS_SEGUIDAS:
                v = res_qml[nm].get(m)
                acumulado[nm][m].append(float(v) if v is not None else float('nan'))
            tiempos[nm].append(dt_qml / 2)

        # --- LSTM+QTL ---
        res_qtl = {}
        if tiene_qtl:
            torch.manual_seed(seed); np.random.seed(seed)
            t0 = time.time()
            res_qtl = entrenar_qtl(ds, res_base['_modelo_lstm'])
            dt_qtl = time.time() - t0
            nm = 'LSTM+QTL'
            if nm in res_qtl:
                if '_prob' in res_qtl[nm]:
                    res_qtl[nm].update(_metricas_operativas(y_te, y_te_ext, ACP_te, res_qtl[nm]['_prob']))
                acumulado.setdefault(nm, {m: [] for m in METRICAS_SEGUIDAS})
                tiempos.setdefault(nm, [])
                for m in METRICAS_SEGUIDAS:
                    v = res_qtl[nm].get(m)
                    acumulado[nm][m].append(float(v) if v is not None else float('nan'))
                tiempos[nm].append(dt_qtl)

        # --- LbL-QNN ---
        torch.manual_seed(seed); np.random.seed(seed)
        t0 = time.time()
        res_lbl = entrenar_qnn_layerbylayer(ds)
        dt_lbl = time.time() - t0
        nm = 'LbL-QNN'
        if nm in res_lbl:
            if '_prob' in res_lbl[nm]:
                res_lbl[nm].update(_metricas_operativas(y_te, y_te_ext, ACP_te, res_lbl[nm]['_prob']))
            acumulado.setdefault(nm, {m: [] for m in METRICAS_SEGUIDAS})
            tiempos.setdefault(nm, [])
            for m in METRICAS_SEGUIDAS:
                v = res_lbl[nm].get(m)
                acumulado[nm][m].append(float(v) if v is not None else float('nan'))
            tiempos[nm].append(dt_lbl)

        for res in (res_qml, res_qtl, res_lbl):
            for k in list(res.keys()):
                if isinstance(res[k], dict) and '_modelo' in res[k]:
                    del res[k]['_modelo']

        semillas_completadas.add(seed)

        resumen = _construir_resumen(acumulado, tiempos, semillas_completadas)

        if guardar_en:
            torch.save(resumen, guardar_en)
            print(f"  [v] PROGRESS SAVED (completed seeds: {len(semillas_completadas)}/{n_seeds})")

        del res_qml, res_qtl, res_lbl
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    resumen = _construir_resumen(acumulado, tiempos, semillas_completadas)

    print(f"\n  {'Model':<15} {'Metric':<9} {'Mean':>8} {'+- Std':>8}   "
          f"[min, max]  (n={len(semillas_completadas)} seeds)")
    print("  " + "-" * 78)
    for nm, metricas in resumen.items():
        if nm.startswith('_'):
            continue
        for m, data in metricas.items():
            if m in ('auc', 'fpr', 'rc_ACP'):
                print(f"  {nm:<15} {m:<9} {data['mean']:>8.4f} {data['std']:>8.4f}   "
                      f"[{data['min']:.3f}, {data['max']:.3f}]")

    return resumen


# ══════════════════════════════════════════════════════════════════════════
# 3. SEED STABILITY VERDICT
# ══════════════════════════════════════════════════════════════════════════

def veredicto_orden_dial(resumen: dict, metrica: str = 'fpr',
                          modelos=('LSTM+PI-QNN', 'LbL-QNN')) -> None:
    """Checks whether the ranking between two models is consistent across all seeds."""
    a, b = modelos
    if a not in resumen or b not in resumen or metrica not in resumen.get(a, {}):
        print(f"  [!] Missing data for '{a}' or '{b}' on metric '{metrica}'.")
        return
    va = np.asarray(resumen[a][metrica]['valores'])
    vb = np.asarray(resumen[b][metrica]['valores'])
    n = min(len(va), len(vb))
    menor_mejor = metrica in ('fpr',)
    if menor_mejor:
        gana_a = int((va[:n] < vb[:n]).sum()); gana_b = int((vb[:n] < va[:n]).sum())
    else:
        gana_a = int((va[:n] > vb[:n]).sum()); gana_b = int((vb[:n] > va[:n]).sum())

    print(f"\n  Stability Verdict — {metrica} ({'lower' if menor_mejor else 'higher'}=better):")
    print(f"    {a:<15} wins in {gana_a}/{n} seeds")
    print(f"    {b:<15} wins in {gana_b}/{n} seeds")
    if gana_a == n or gana_b == n:
        print("    -> CONSISTENT order across all seeds: statistically stable finding.")
    else:
        print("    -> Order INVERTS between seeds: sensitive to initialization. "
              "Report mean+-std rather than single-run rankings.")


# English aliases for public API
run_qml_multiseed = ejecutar_multisemilla_qml
seed_stability_verdict = veredicto_orden_dial
paired_operational_bootstrap = bootstrap_pareado_operativo

