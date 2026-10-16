"""
robustez_estadistica.py — VERSIÓN FINAL (única, completa, para pegar tal cual)

Contiene TODO lo necesario en un solo archivo:
  1. bootstrap_pareado_operativo()   — IC de la diferencia Rc-p90/Rc-ACP/FPR
  2. ejecutar_multisemilla_qml()     — multi-seed CON fpr/rc_ACP/rc_p90,
                                        checkpoint/resume, guardado incremental
                                        y descarga automática en Colab
  3. veredicto_orden_dial()          — ¿el orden entre dos modelos es estable
                                        entre semillas?

USO:
    from modelos import entrenar_qml, entrenar_qtl, entrenar_qnn_layerbylayer

    resumen = ejecutar_multisemilla_qml(
        ds, res_base, entrenar_qml, entrenar_qtl, entrenar_qnn_layerbylayer,
        seeds=range(15),
        guardar_en='/content/drive/MyDrive/resultados/checkpoints/09_multisemilla.pt',
        descargar_cada_semilla=True,
        frecuencia_descarga=1,
    )
    veredicto_orden_dial(resumen, metrica='fpr')
    veredicto_orden_dial(resumen, metrica='rc_ACP')

IMPORTANTE: usa una ruta de checkpoint NUEVA (distinta de cualquier .pt
generado por versiones anteriores de este wrapper) — los checkpoints viejos
no tienen fpr/rc_ACP/rc_p90 y no se pueden completar retroactivamente
(ver el aviso que imprime la función si detecta uno).
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
# 1. BOOTSTRAP PAREADO DE LA DIFERENCIA EN EL PUNTO DE OPERACIÓN
# ══════════════════════════════════════════════════════════════════════════

def bootstrap_pareado_operativo(y_te, y_te_ext, ACP_te, prob_a, prob_b,
                                  umbral_a, umbral_b,
                                  nombre_a: str = "Modelo A", nombre_b: str = "Modelo B",
                                  n_boot: int = 2000, seed: int = 42,
                                  imprimir: bool = True) -> dict:
    """IC 95% (bootstrap pareado) de la diferencia (B - A) en Rc-p90, Rc-ACP
    y FPR, evaluando cada modelo en su umbral operativo ya fijado."""
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
        print(f"\n  Bootstrap pareado ({n_boot} remuestreos): Δ = {nombre_b} − {nombre_a}")
        print("  " + "-" * 78)
        for k, r in resultados.items():
            etiqueta = {'rc_p90': 'Rc-p90 ', 'rc_ACP': 'Rc-ACP', 'fpr': 'FPR    '}[k]
            sig = "✅ significativo (IC no cruza 0)" if r['significativo'] else "⚠️  NO significativo (IC cruza 0)"
            print(f"  Δ{etiqueta}  = {r['delta_medio']:+.4f}   "
                  f"IC95%=[{r['ic95'][0]:+.4f}, {r['ic95'][1]:+.4f}]   {sig}   (n={r['n_boot_validos']})")
        print()

    return resultados


# ══════════════════════════════════════════════════════════════════════════
# 2. MULTI-SEMILLA: métricas operativas + checkpoint/resume + descarga
# ══════════════════════════════════════════════════════════════════════════

def _metricas_operativas(y_te, y_te_ext, ACP_te, prob, umbral: float = 0.5) -> dict:
    """fpr/rc_ACP/rc_p90 a partir de las probabilidades predichas (umbral=0.5,
    el punto de operación "crudo", igual que analisis_extremos)."""
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
    """Construye el resumen (mean/std/min/max). Separado en su propia función
    porque se necesita tanto en el guardado incremental como al final del
    todo — incluyendo cuando todas las semillas ya estaban completadas y el
    bucle no se ejecuta ni una vez (evita el NameError de versiones previas)."""
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


def _descargar_en_colab(ruta: str) -> None:
    try:
        from google.colab import files
        files.download(ruta)
    except ImportError:
        print(f"  [i] No estás en Colab: omitiendo descarga automática. Archivo en {ruta}.")
    except Exception as e:
        print(f"  [!] No se pudo disparar la descarga automática ({e}). Archivo en {ruta}.")


def ejecutar_multisemilla_qml(ds, res_base, entrenar_qml, entrenar_qtl,
                                entrenar_qnn_layerbylayer, seeds=range(15),
                                guardar_en: str = None,
                                descargar_cada_semilla: bool = True,
                                frecuencia_descarga: int = 1) -> dict:
    """Repite el entrenamiento de LSTM+QNN, LSTM+PI-QNN, LSTM+QTL y
    LbL-QNN con varias semillas, capturando acc/f1/auc/fpr/rc_ACP/rc_p90
    para cada una. Reanuda automáticamente desde `guardar_en` si ya existe,
    guarda progreso tras cada semilla, y descarga el checkpoint al navegador
    en Colab tras cada guardado (configurable)."""
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
        print(f"  [+] Detectado archivo previo en {guardar_en}. Intentando reanudar...")
        try:
            res_previo = torch.load(guardar_en, weights_only=False)
            if '_estado_interno' in res_previo:
                acumulado = res_previo['_estado_interno']['acumulado']
                tiempos = res_previo['_estado_interno']['tiempos']
                semillas_completadas = set(res_previo['_estado_interno']['semillas_completadas'])
                print(f"  [+] Reanudando. {len(semillas_completadas)} semillas ya completadas.")
                for nm, metricas in acumulado.items():
                    faltan = [m for m in ('fpr', 'rc_ACP', 'rc_p90') if m not in metricas or not metricas[m]]
                    if faltan:
                        print(f"  [!] AVISO: '{nm}' en el checkpoint cargado no tiene {faltan}. "
                              f"Viene de una versión anterior del wrapper — usa una ruta nueva "
                              f"en guardar_en si necesitas estas métricas también para las "
                              f"semillas ya completadas.")
        except Exception as e:
            print(f"  [!] No se pudo cargar el progreso previo: {e}")

    tiene_qtl = '_modelo_lstm' in res_base
    if not tiene_qtl:
        print("  [!] res_base no tiene '_modelo_lstm' — QTL se omitirá en todas las semillas.")

    for i, seed in enumerate(seeds):
        if seed in semillas_completadas:
            print(f"\n  ── Semilla {seed} ({i + 1}/{n_seeds}) ── [YA COMPLETADA, SALTANDO]")
            continue

        print(f"\n  ── Semilla {seed} ({i + 1}/{n_seeds}) ──")

        # --- LSTM+QNN y LSTM+PI-QNN ---
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
            print(f"  [v] PROGRESO GUARDADO EN DISCO (semillas completadas: "
                  f"{len(semillas_completadas)}/{n_seeds})")
            if descargar_cada_semilla and (len(semillas_completadas) % frecuencia_descarga == 0):
                _descargar_en_colab(guardar_en)

        del res_qml, res_qtl, res_lbl
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    resumen = _construir_resumen(acumulado, tiempos, semillas_completadas)

    print(f"\n  {'Modelo':<15} {'Métrica':<9} {'Media':>8} {'± Std':>8}   "
          f"[min, max]  (n={len(semillas_completadas)} semillas)")
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
# 3. VEREDICTO DE ESTABILIDAD ENTRE SEMILLAS
# ══════════════════════════════════════════════════════════════════════════

def veredicto_orden_dial(resumen: dict, metrica: str = 'fpr',
                          modelos=('LSTM+PI-QNN', 'LbL-QNN')) -> None:
    """¿El orden entre dos modelos es consistente en TODAS las semillas?"""
    a, b = modelos
    if a not in resumen or b not in resumen or metrica not in resumen.get(a, {}):
        print(f"  [!] Faltan datos de '{a}' o '{b}' para la métrica '{metrica}'.")
        return
    va = np.asarray(resumen[a][metrica]['valores'])
    vb = np.asarray(resumen[b][metrica]['valores'])
    n = min(len(va), len(vb))
    menor_mejor = metrica in ('fpr',)
    if menor_mejor:
        gana_a = int((va[:n] < vb[:n]).sum()); gana_b = int((vb[:n] < va[:n]).sum())
    else:
        gana_a = int((va[:n] > vb[:n]).sum()); gana_b = int((vb[:n] > va[:n]).sum())

    print(f"\n  Veredicto de estabilidad — {metrica} ({'menor' if menor_mejor else 'mayor'}=mejor):")
    print(f"    {a:<15} gana en {gana_a}/{n} semillas")
    print(f"    {b:<15} gana en {gana_b}/{n} semillas")
    if gana_a == n or gana_b == n:
        print("    → Orden CONSISTENTE en todas las semillas: hallazgo estable.")
    else:
        print("    → El orden CAMBIA entre semillas: no es un hallazgo estable. "
              "Usa media±std, no el ranking de una sola corrida.")
