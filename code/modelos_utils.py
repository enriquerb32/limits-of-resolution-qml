"""
modelos_utils.py — Shared utilities for classical and quantum models
=====================================================================
Contains metrics, sklearn helpers, PyTorch evaluation, training loop,
probability extraction, and extreme-event analysis.

Both modelos_clasicos.py and modelos_cuanticos.py import from here.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from sklearn.impute import SimpleImputer


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _metricas(y_true, y_pred, y_prob=None) -> dict:
    d = {
        'acc': float(accuracy_score(y_true, y_pred)),
        'f1':  float(f1_score(y_true, y_pred, zero_division=0))
    }
    if y_prob is not None:
        try:
            d['auc'] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            d['auc'] = float('nan')
    return d


# ──────────────────────────────────────────────────────────────────────────────
# Sklearn helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_fit_sklearn(model, X, y):
    imp = SimpleImputer(strategy='mean')
    X_clean = imp.fit_transform(X)
    model.fit(X_clean, y)
    return model, imp


def _safe_predict_sklearn(model, imp, X):
    X_clean = imp.transform(X)
    return model.predict(X_clean), model.predict_proba(X_clean)[:, 1]


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _evaluar_pytorch(model, X_te, y_te_np):
    model.eval()
    with torch.no_grad():
        out = model(X_te)
        logits = out[0] if isinstance(out, tuple) else out
        prob = torch.sigmoid(logits).numpy()
    pred = (prob >= 0.5).astype(float)
    return _metricas(y_te_np, pred, prob), prob


# ──────────────────────────────────────────────────────────────────────────────
# Generic training loop (no physics)
# ──────────────────────────────────────────────────────────────────────────────

def _loop(model, dl, X_va, y_va, epochs=25, patience=6, lr=5e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    y_tr = dl.dataset.tensors[1]
    n_neg = (y_tr == 0).sum().float()
    n_pos = (y_tr == 1).sum().float()
    pos_w = torch.tensor([n_neg / (n_pos + 1e-5)])
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    best, bs, ni = float('inf'), None, 0
    hist = {'tr': [], 'va': [], 'acc_va': []}

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for bX, by in dl:
            opt.zero_grad()
            out = model(bX)
            logits = out[0] if isinstance(out, tuple) else out
            loss = crit(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()

        model.eval()
        with torch.no_grad():
            vout = model(X_va)
            v_logits = vout[0] if isinstance(vout, tuple) else vout
            vl = crit(v_logits, y_va).item()
            v_probs = torch.sigmoid(v_logits)
            va = ((v_probs >= 0.5).float() == y_va).float().mean().item()

        sch.step(vl)
        hist['tr'].append(ep_loss / len(dl))
        hist['va'].append(vl)
        hist['acc_va'].append(va)

        if vl < best - 1e-4:
            best, bs, ni = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            ni += 1
            if ni >= patience:
                break

    model.load_state_dict(bs)
    return model, hist


# ──────────────────────────────────────────────────────────────────────────────
# Probability extraction from result dicts
# ──────────────────────────────────────────────────────────────────────────────

def _extraer_probs(res: dict) -> dict:
    """Extrae probabilidades predichas de un diccionario de resultados,
    soportando ambos formatos usados en este módulo:
      (a) agrupadas bajo la clave '_probs' (entrenar_baselines), o
      (b) anidadas por modelo bajo '_prob' (entrenar_qml, entrenar_qtl,
          entrenar_qnn_layerbylayer, entrenar_qksvm)."""
    out = dict(res.get('_probs', {}))
    for k, v in res.items():
        if k.startswith('_'):
            continue
        if isinstance(v, dict) and '_prob' in v:
            out[k] = v['_prob']
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Extreme-event analysis (ACP / p90)
# ──────────────────────────────────────────────────────────────────────────────

def analisis_extremos(ds, res_base: dict, res_qml: dict,
                       res_qksvm: dict = None, res_qtl: dict = None,
                       res_lbl: dict = None) -> dict:
    from sklearn.metrics import confusion_matrix, roc_curve
    y_te   = ds.y_te.numpy()
    y_te_x = ds.y_te_ext.numpy()

    probs_todos = {}
    for r in (res_base, res_qml, res_qksvm, res_qtl, res_lbl):
        if r:
            probs_todos.update(_extraer_probs(r))

    resultados  = {}

    print(f"\n  {'Modelo':<28} {'Acc':>8} {'F1':>6} {'Rc-p90':>8} {'Rc-ACP':>9} {'FPR':>7} {'CSI':>7}")
    print("  " + "-"*82)

    ACP_te_np = ds.ACP_te.numpy() if hasattr(ds.ACP_te, 'numpy') else ds.ACP_te

    for nm, prob in probs_todos.items():
        pred   = (prob >= .5).astype(float)
        met    = _metricas(y_te, pred, prob)
        rc_ext = recall_score(y_te_x, pred, zero_division=0)

        rc_ACP= (recall_score(y_te[ACP_te_np == 1], pred[ACP_te_np == 1], zero_division=0)
                  if ACP_te_np.sum() > 0 else float('nan'))

        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        resultados[nm] = {**met, 'rc_p90': rc_ext, 'rc_ACP': rc_ACP, 'fpr': fpr, 'csi': csi}
        print(f"  {nm:<28} {met['acc']*100:>7.2f}%  {met['f1']:>5.3f}  {rc_ext:>7.3f}  {rc_ACP:>8.3f}  {fpr:>5.3f}  {csi:>5.3f}")

    # Threshold Matching: LbL-QNN vs LSTM pura
    if 'LbL-QNN' in resultados and 'LSTM pura' in probs_todos:
        fpr_objetivo = resultados['LbL-QNN']['fpr']
        prob_lstm = probs_todos['LSTM pura']

        fprs, tprs, thresholds = roc_curve(y_te, prob_lstm)
        idx = np.argmin(np.abs(fprs - fpr_objetivo))
        umbral_opt = thresholds[idx]

        pred_adj = (prob_lstm >= umbral_opt).astype(float)
        rc_ext_adj = recall_score(y_te_x, pred_adj, zero_division=0)
        rc_ACP_adj = recall_score(y_te[ACP_te_np == 1], pred_adj[ACP_te_np == 1], zero_division=0) if ACP_te_np.sum() > 0 else float('nan')

        tn, fp, fn, tp = confusion_matrix(y_te, pred_adj, labels=[0, 1]).ravel()
        fpr_adj = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
        csi_adj = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        met_adj = _metricas(y_te, pred_adj, prob_lstm)

        nm_adj = "LSTM pura (FPR Matched)"
        resultados[nm_adj] = {**met_adj, 'rc_p90': rc_ext_adj, 'rc_ACP': rc_ACP_adj, 'fpr': fpr_adj, 'csi': csi_adj,
                               '_umbral': float(umbral_opt), '_fpr_objetivo': float(fpr_objetivo)}
        print("  " + "-"*82)
        print(f"  {nm_adj:<28} {met_adj['acc']*100:>7.2f}%  {met_adj['f1']:>5.3f}  {rc_ext_adj:>7.3f}  {rc_ACP_adj:>8.3f}  {fpr_adj:>5.3f}  {csi_adj:>5.3f}")
        print(f"  [!] Umbral del LSTM ajustado de 0.5 a {umbral_opt:.3f} "
              f"(FPR objetivo={fpr_objetivo:.3f}, FPR real alcanzado={fpr_adj:.3f}) "
              f"para igualar el punto operativo del LbL-QNN.")
    elif 'LSTM pura' in probs_todos:
        print("  [!] Threshold-matching omitido: 'LbL-QNN' no está en resultados "
              "(pasa res_lbl=... a analisis_extremos).")

    return resultados
