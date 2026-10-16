#!/usr/bin/env python3
"""
main.py — Pipeline orchestrator for the PI-QNN precipitation study
====================================================================
Single entry point that runs all 11 steps of the research pipeline:
  1-3. ETL, physics diagnostics, feature engineering
  4.   Classical baselines (single-run + multi-seed)
  5.   Quantum models (QNN, PI-QNN, QTL, LbL)
  5.5  NISQ noise simulation
  6.   Quantum Kernel SVM (QKSVM)
  7.   Ablation study
  8.   Extreme-event analysis (ACP/p90)
  9.   Statistical validation (Bootstrap, DeLong)
  10.  JSON metrics export
  11.  Paper figures (PDF + LaTeX table)

Usage:
    python main.py                             # Full run with checkpoints
    python main.py --skip-qml                  # Classical-only run
    python main.py --forzar-reentreno          # Force full retraining
    python main.py --datos ./datos --salida ./resultados
"""

import os
import sys
import json
import warnings
import argparse

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float64)


def parse_args():
    parser = argparse.ArgumentParser(
        description='PI-QNN Precipitation Study — Full Pipeline'
    )
    parser.add_argument('--datos', type=str, default='./datos',
                        help='Directory containing raw AEMET data per city')
    parser.add_argument('--salida', type=str, default='./resultados',
                        help='Output directory for results, checkpoints, figures')
    parser.add_argument('--ciudades', nargs='+',
                        default=['Barcelona', 'Valencia', 'Alicante', 'San_Javier'],
                        help='List of city names to process')
    parser.add_argument('--skip-qml', action='store_true',
                        help='Skip all quantum model steps')
    parser.add_argument('--usar-checkpoints', action='store_true', default=True,
                        help='Load from checkpoints if available (default: True)')
    parser.add_argument('--no-checkpoints', dest='usar_checkpoints', action='store_false',
                        help='Disable checkpoint loading (force full recompute)')
    parser.add_argument('--forzar-reentreno', action='store_true',
                        help='Force retraining of ALL models even if checkpoints exist')
    parser.add_argument('--forzar-reentreno-qml', action='store_true',
                        help='Force retraining of QML models only (keeps classical checkpoints)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Lazy imports to allow --help without installing heavy deps
    from etl           import cargar_datos
    from fisica        import construir_features
    from features      import preparar_dataset
    from validacion    import ejecutar_validacion
    from visualizacion import generar_figuras
    from experimento_ruido import ejecutar_evaluacion_ruido

    from modelos import (entrenar_baselines, entrenar_qml,
                         entrenar_qksvm, entrenar_ablacion, analisis_extremos,
                         entrenar_qtl, entrenar_qnn_layerbylayer,
                         entrenar_baselines_multisemilla, LstmQNN)

    os.makedirs(args.salida, exist_ok=True)
    ckpt_dir = os.path.join(args.salida, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Checkpoint paths ──
    ckpt_data  = f"{ckpt_dir}/01_data.pt"
    ckpt_base  = f"{ckpt_dir}/02_baselines.pt"
    ckpt_qml   = f"{ckpt_dir}/03_qml.pt"
    ckpt_qksvm = f"{ckpt_dir}/04_qksvm.pt"
    ckpt_abl   = f"{ckpt_dir}/05_ablacion.pt"
    ckpt_ruido_pesos = f"{ckpt_dir}/06_pesos_ruido.pt"
    ckpt_ruido_metricas = f"{ckpt_dir}/07_metricas_ruido.pt"
    ckpt_val   = f"{ckpt_dir}/08_validacion.pt"
    ckpt_multisemilla = f"{ckpt_dir}/09_multisemilla.pt"
    ckpt_multisemilla_clasico = f"{ckpt_dir}/10_multisemilla_clasico.pt"

    forzar_qml = args.forzar_reentreno or args.forzar_reentreno_qml

    # =====================================================================
    # STEPS 1-3: ETL, Physics, Features
    # =====================================================================
    print("\n" + "═"*65)
    print("PASOS 1-3 · ETL Multi-Estación, Física Diagnóstica y Features")
    print("═"*65)

    if args.usar_checkpoints and os.path.exists(ckpt_data) and not args.forzar_reentreno:
        print("  [+] Cargando datos procesados desde checkpoint...")
        agg, ds = torch.load(ckpt_data, weights_only=False)
    else:
        df_diario = cargar_datos(args.datos, ciudades=args.ciudades)
        agg = construir_features(df_diario)
        if 'ciudad' in agg.columns:
            agg = pd.get_dummies(agg, columns=['ciudad'], dtype=float)
        ds = preparar_dataset(agg)
        torch.save((agg, ds), ckpt_data)
        print("  [v] Datos multi-estación procesados y guardados en checkpoint.")

    # =====================================================================
    # STEP 4: Classical Baselines + Multi-Seed
    # =====================================================================
    print("\n" + "═"*65)
    print("PASO 4 · Baselines Clásicos y Estadísticos")
    print("═"*65)

    if args.usar_checkpoints and os.path.exists(ckpt_base) and not args.forzar_reentreno:
        print("  [+] Cargando modelos clásicos (single-run) desde checkpoint...")
        res_base = torch.load(ckpt_base, weights_only=False)
    else:
        res_base = entrenar_baselines(ds)
        torch.save(res_base, ckpt_base)
        print("  [v] Baselines (single-run) entrenados y guardados en checkpoint.")

    # Multi-Seed Classical
    if args.usar_checkpoints and os.path.exists(ckpt_multisemilla_clasico) and not args.forzar_reentreno:
        print("  [+] Cargando réplica multi-semilla clásica desde checkpoint...")
        resultados_cls_multi = torch.load(ckpt_multisemilla_clasico, weights_only=False)
    else:
        print("  [!] Ejecutando réplica Multi-Semilla Clásica (15 semillas)...")
        resultados_cls_multi = entrenar_baselines_multisemilla(ds, seeds=range(15))
        torch.save(resultados_cls_multi, ckpt_multisemilla_clasico)
        print("  [v] Multi-semilla clásica completada y guardada en checkpoint.")

    # Print classical summary table
    print("\n" + "="*80)
    print(f" {'Modelo Clásico':<20} | {'AUC':<15} | {'FPR':<15} | {'Rc-ACP':<15} ")
    print("-" * 80)
    for nm, met in resultados_cls_multi.items():
        auc_m, auc_s = met['auc']['mean'], met['auc']['std']
        fpr_m, fpr_s = met['fpr']['mean'], met['fpr']['std']
        rcd_m, rcd_s = met['rc_ACP']['mean'], met['rc_ACP']['std']
        print(f" {nm:<20} | {auc_m:.3f} ± {auc_s:.3f} | {fpr_m:.3f} ± {fpr_s:.3f} | {rcd_m:.3f} ± {rcd_s:.3f} ")
    print("="*80)

    # =====================================================================
    # STEP 5: QML Models
    # =====================================================================
    if args.skip_qml:
        print("\n  [--skip-qml] Saltando pasos cuánticos")
        res_qml, res_qksvm, res_abl, res_val, res_ruido = {'_probs': {}}, {}, {}, {}, {}
    else:
        print("\n" + "═"*65)
        print("PASO 5 · Modelos QML (Base, PI-QNN Termodinámica, QTL y LbL)")
        print("═"*65)

        if args.usar_checkpoints and os.path.exists(ckpt_qml) and not forzar_qml:
            print("  [+] Cargando modelos cuánticos (QML) desde checkpoint...")
            res_qml = torch.load(ckpt_qml, weights_only=False)
        else:
            res_qml = entrenar_qml(ds)
            res_qml['_probs'] = {}
            for nm in ['LSTM+QNN', 'LSTM+PI-QNN']:
                if nm in res_qml and '_prob' in res_qml[nm]:
                    res_qml['_probs'][nm] = res_qml[nm]['_prob']

            if '_modelo_lstm' in res_base:
                res_qtl = entrenar_qtl(ds, res_base['_modelo_lstm'])
                res_qml.update(res_qtl)
                if '_probs' not in res_qml: res_qml['_probs'] = {}
                res_qml['_probs']['LSTM+QTL'] = res_qtl['LSTM+QTL']['_prob']
            else:
                print("  [!] No se pudo entrenar QTL porque falta '_modelo_lstm'")

            res_lbl = entrenar_qnn_layerbylayer(ds)
            res_qml.update(res_lbl)
            res_qml['_probs']['LbL-QNN'] = res_lbl['LbL-QNN']['_prob']

            # Save weights for noise experiment before cleanup
            if 'LSTM+QNN' in res_qml and 'LbL-QNN' in res_qml:
                torch.save({
                    'LSTM+QNN_state': res_qml['LSTM+QNN']['_modelo'].state_dict(),
                    'LbL-QNN_state': res_qml['LbL-QNN']['_modelo'].state_dict(),
                }, ckpt_ruido_pesos)

            # Clean up model objects (not serializable with QNodes)
            for nm in ['LSTM+QNN', 'LSTM+PI-QNN', 'LSTM+QTL', 'LbL-QNN']:
                if nm in res_qml and '_modelo' in res_qml[nm]:
                    del res_qml[nm]['_modelo']

            torch.save(res_qml, ckpt_qml)
            print("  [v] Modelos QML entrenados y guardados en checkpoint.")

        # =====================================================================
        # STEP 5.5: NISQ Noise Simulation
        # =====================================================================
        print("\n" + "═"*65)
        print("PASO EXTRA · Simulación de Hardware NISQ (Ruido)")
        print("═"*65)

        if args.usar_checkpoints and os.path.exists(ckpt_ruido_metricas) and not args.forzar_reentreno and not forzar_qml:
            print("  [+] Cargando métricas de ruido cuántico desde checkpoint...")
            res_ruido = torch.load(ckpt_ruido_metricas, weights_only=False)
        else:
            if os.path.exists(ckpt_ruido_pesos):
                print("  [+] Reconstruyendo modelos desde los pesos para la simulación...")
                pesos = torch.load(ckpt_ruido_pesos, weights_only=False)
                modelo_qnn = LstmQNN(ds.in_dim, n_q=4, n_l=2, physics=False).double()
                modelo_qnn.load_state_dict(pesos['LSTM+QNN_state'])
                modelo_lbl = LstmQNN(ds.in_dim, n_q=4, n_l=3, physics=True).double()
                modelo_lbl.load_state_dict(pesos['LbL-QNN_state'])

                print("  [!] Ejecutando simulación de hardware NISQ (Ruido p=0.01)...")
                res_ruido = ejecutar_evaluacion_ruido(ds, modelo_qnn, modelo_lbl, p_error=0.01)
                torch.save(res_ruido, ckpt_ruido_metricas)
                print("  [v] Métricas de ruido completadas y guardadas.")
            else:
                print("  [!] ERROR: No se encontró el archivo de pesos '06_pesos_ruido.pt'.")
                res_ruido = {}

        # =====================================================================
        # STEP 6: QKSVM
        # =====================================================================
        print("\n" + "═"*65)
        print("PASO 6 · Quantum Kernel Support Vector Machine (QKSVM)")
        print("═"*65)

        if args.usar_checkpoints and os.path.exists(ckpt_qksvm) and not args.forzar_reentreno:
            print("  [+] Cargando QKSVM desde checkpoint...")
            res_qksvm = torch.load(ckpt_qksvm, weights_only=False)
        else:
            res_qksvm = entrenar_qksvm(ds)
            torch.save(res_qksvm, ckpt_qksvm)
            print("  [v] QKSVM entrenado y guardado en checkpoint.")

        # =====================================================================
        # STEP 7: Ablation
        # =====================================================================
        print("\n" + "═"*65)
        print("PASO 7 · Ablación cuántica")
        print("═"*65)

        res_abl_cache = None
        if args.usar_checkpoints and os.path.exists(ckpt_abl) and not args.forzar_reentreno:
            res_abl_cache = torch.load(ckpt_abl, weights_only=False)
            if not res_abl_cache:
                print("  [!] Checkpoint de ablación vacío detectado — se descarta y se reentrena.")
                res_abl_cache = None

        if res_abl_cache is not None:
            print("  [+] Cargando Ablación desde checkpoint...")
            res_abl = res_abl_cache
        else:
            res_abl = entrenar_ablacion(ds)
            torch.save(res_abl, ckpt_abl)
            print("  [v] Ablación entrenada y guardada en checkpoint.")

    # =====================================================================
    # STEPS 8-11: Analysis, JSON, Figures
    # =====================================================================
    print("\n" + "═"*65)
    print("PASO 8 · Análisis de eventos extremos (ACP / p90)")
    print("═"*65)
    res_extremos = analisis_extremos(ds, res_base, res_qml, res_qksvm if not args.skip_qml else None)

    print("\n" + "═"*65)
    print("PASO 9 · Validación estadística (Bootstrapping & DeLong)")
    print("═"*65)
    if args.skip_qml:
        res_val = {}
    elif args.usar_checkpoints and os.path.exists(ckpt_val) and not args.forzar_reentreno and not forzar_qml:
        print("  [+] Cargando resultados de Validación y Nyström desde checkpoint...")
        res_val = torch.load(ckpt_val, weights_only=False)
    else:
        res_val = ejecutar_validacion(ds, res_base, res_qml, res_qksvm, skip_nystrom=False)
        torch.save(res_val, ckpt_val)
        print("  [v] Validación y Nyström completados y guardados en checkpoint.")

    # STEP 10: JSON metrics
    def _clean(d):
        if d is None: return None
        if not isinstance(d, dict):
            if isinstance(d, (float, np.floating)): return float(d)
            if isinstance(d, (int, np.integer)): return int(d)
            if isinstance(d, (bool, np.bool_)): return bool(d)
            if isinstance(d, (list, tuple)): return [_clean(i) for i in d]
            return d
        return {str(k): _clean(v) for k, v in d.items() if not str(k).startswith('_')}

    metricas_json = {
        "baselines":  _clean(res_base),
        "qml":        _clean(res_qml),
        "qksvm":      _clean(res_qksvm if not args.skip_qml else {}),
        "ablacion":   _clean(res_abl if not args.skip_qml else {}),
        "extremos":   _clean(res_extremos),
        "ruido":      _clean(res_ruido),
        "validacion": _clean(res_val),
        "fisica": {
            "umbral_p60":   float(ds.umbral),
            "umbral_p90":   float(ds.umbral_ext),
            "n_ACP_test":  int(ds.ACP_te.sum()),
            "in_dim":       ds.in_dim,
            "termodinamica_aplicada": True
        },
    }
    ruta_json = f"{args.salida}/main_metrics.json"
    with open(ruta_json, 'w') as f:
        json.dump(metricas_json, f, indent=2)

    # STEP 11: Figures
    print("\n" + "═"*65)
    print("PASO 11 · Generando figuras y PDF")
    print("═"*65)

    res_qml_figs = res_qml.copy()
    if not args.skip_qml and res_qksvm:
        res_qml_figs.update(res_qksvm)
        if '_probs' not in res_qml_figs: res_qml_figs['_probs'] = {}
        res_qml_figs['_probs']['QKSVM'] = res_qksvm['QKSVM']['_prob']

    lbl_res = res_qml.get('LbL-QNN', {})
    if not lbl_res:
        print("  [!] AVISO: 'LbL-QNN' no está en res_qml — figuras usarán baseline hardcodeado.")

    res_multi = {}
    if os.path.exists(ckpt_multisemilla):
        print("  [+] Cargando datos multi-semilla para el gráfico de violines...")
        res_multi = torch.load(ckpt_multisemilla, weights_only=False)

    generar_figuras(
        agg, ds, res_base, res_qml_figs,
        res_abl if not args.skip_qml else {},
        None, None,
        res_validacion=res_val,
        salida=args.salida,
        baseline_acc=lbl_res.get('acc', 0.6136) * 100,
        baseline_f1=lbl_res.get('f1', 0.555),
        res_multisemilla=res_multi
    )

    _resumen(res_base, res_qml, res_qksvm if not args.skip_qml else {}, res_val, metricas_json, args.salida)


def _resumen(res_base, res_qml, res_qksvm, res_val, metricas_json, salida):
    print("\n" + "═"*65)
    print("RESUMEN FINAL")
    print("═"*65)

    todos = {k: v for k, v in res_base.items() if not k.startswith('_')}
    todos.update({k: v for k, v in res_qml.items() if not k.startswith('_')})
    if res_qksvm:
        todos.update({k: v for k, v in res_qksvm.items() if not k.startswith('_')})

    print(f"\n  {'Modelo':<22} {'Accuracy':>9}  {'F1':>6}  {'AUC':>6}")
    print("  " + "-"*48)
    for nm, mv in todos.items():
        auc_s = f"{mv['auc']:.3f}" if mv.get('auc') else "  —  "
        mk = " ◄ PI-QNN" if "PI" in nm else (" ◄ QTL" if "QTL" in nm else (" ◄ QKSVM" if "QKSVM" in nm else (" ◄ QNN" if "QNN" in nm else "")))
        print(f"  {nm:<22} {mv['acc']*100:>8.2f}%  {mv['f1']:>6.3f}  {auc_s:>6}{mk}")

    tests = res_val.get('tests_delong', {})
    if tests:
        print(f"\n  Tests de DeLong (significancia AUC):")
        for etiq, t in tests.items():
            sig = "✅ sig." if t['significativo_95'] else "⚠️  no sig."
            nm_a = etiq.split('_vs_')[0].replace('_', '+')
            nm_b = etiq.split('_vs_')[1].replace('_', ' ')
            print(f"    {nm_a:<18} vs {nm_b:<18}  ΔAUC={t['delta_auc']:+.4f}  p={t['p_valor']:.4f}  {sig}")

    res_ruido = metricas_json.get('ruido', {})
    if res_ruido:
        print(f"\n  [NISQ] Impacto del Ruido Cuántico (Monte Carlo 15 seeds, p=0.01):")
        for nm, mv in res_ruido.items():
            acc_s = mv.get('acc', 0) * 100
            f1_mean, f1_std = mv.get('f1', 0), mv.get('f1_std', 0)
            rc_ACP_mean, rc_ACP_std = mv.get('rc_ACP', 0), mv.get('rc_ACP_std', 0)
            fpr_mean, fpr_std = mv.get('fpr', 0), mv.get('fpr_std', 0)
            print(f"    {nm:<18} Acc: {acc_s:>5.2f}% | F1: {f1_mean:.3f}±{f1_std:.3f} | Rc-ACP: {rc_ACP_mean:.3f}±{rc_ACP_std:.3f} | FPR: {fpr_mean:.3f}±{fpr_std:.3f}")

    print(f"\n  Archivos generados en {salida}/:")
    for f in ['main_metrics.json', 'paper_figures.pdf', 'tabla_latex.tex', 'checkpoints/']:
        print(f"    ✅  {f}")

    print("\n" + "═"*65)
    print("✅  PIPELINE COMPLETO")
    print("═"*65)


if __name__ == '__main__':
    main()
