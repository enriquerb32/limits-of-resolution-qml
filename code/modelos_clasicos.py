"""
modelos_clasicos.py — Classical baseline models
=================================================
Contains PureLSTM, sklearn-based baselines (LogReg, RF, GB, XGBoost),
and their multi-seed evaluation routines.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.impute import SimpleImputer

from features import DatasetResult
from modelos_utils import (
    _metricas,
    _safe_fit_sklearn,
    _safe_predict_sklearn,
    _evaluar_pytorch,
    _loop,
    deterministic_seed,
)


# ──────────────────────────────────────────────────────────────────────────────
# LSTM Pura (Baseline clásico profundo)
# ──────────────────────────────────────────────────────────────────────────────

class PureLSTM(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, 2, batch_first=True, dropout=0.3, bidirectional=True)
        self.bn = nn.BatchNorm1d(32 + in_dim)
        self.fc = nn.Sequential(
            nn.Linear(32 + in_dim, 16),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x_lstm = x[:, :12].unsqueeze(-1).to(self.lstm.weight_ih_l0.dtype)
        _, (hn, _) = self.lstm(x_lstm)
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        combined = torch.cat([h, x.to(h.dtype)], dim=1)
        try:
            out_bn = self.bn(combined)
        except RuntimeError:
            out_bn = self.bn(combined.float()).to(combined.dtype)
        return self.fc(out_bn).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Classical baseline training (single-run)
# ──────────────────────────────────────────────────────────────────────────────

def entrenar_baselines(ds: DatasetResult) -> dict:
    y_te = ds.y_te.numpy()
    res = {}

    y_pers = np.array([1.0 if ds.precip_media_mes[m] >= 0.5 else 0.0 for m in ds.m_te])
    res['Persistencia'] = {**_metricas(y_te, y_pers), 'auc': None}
    print(f"  [0] Persistence       Acc={res['Persistencia']['acc']*100:.2f}%  F1={res['Persistencia']['f1']:.3f}")

    lr_m, lr_imp = _safe_fit_sklearn(LogisticRegression(C=0.1, max_iter=1000, random_state=42), ds.X_sk, ds.y_sk)
    y_lr, p_lr = _safe_predict_sklearn(lr_m, lr_imp, ds.X_all[ds.s2:])
    res['Reg. Logística'] = _metricas(y_te, y_lr, p_lr)
    print(f"  [1] Logistic Reg.     Acc={res['Reg. Logística']['acc']*100:.2f}%  F1={res['Reg. Logística']['f1']:.3f}  AUC={res['Reg. Logística']['auc']:.3f}")

    rf_m, rf_imp = _safe_fit_sklearn(
        RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1),
        ds.X_sk, ds.y_sk
    )
    y_rf, p_rf = _safe_predict_sklearn(rf_m, rf_imp, ds.X_all[ds.s2:])
    res['Random Forest'] = _metricas(y_te, y_rf, p_rf)
    print(f"  [2] Random Forest     Acc={res['Random Forest']['acc']*100:.2f}%  F1={res['Random Forest']['f1']:.3f}  AUC={res['Random Forest']['auc']:.3f}")

    gb_m, gb_imp = _safe_fit_sklearn(
        GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
        ds.X_sk, ds.y_sk
    )
    y_gb, p_gb = _safe_predict_sklearn(gb_m, gb_imp, ds.X_all[ds.s2:])
    res['Gradient Boosting'] = _metricas(y_te, y_gb, p_gb)
    print(f"  [3] Grad. Boosting    Acc={res['Gradient Boosting']['acc']*100:.2f}%  F1={res['Gradient Boosting']['f1']:.3f}  AUC={res['Gradient Boosting']['auc']:.3f}")

    try:
        from xgboost import XGBClassifier
        imp = SimpleImputer(strategy='mean')
        X_tr_clean = imp.fit_transform(ds.X_sk)
        X_va_clean = imp.transform(ds.X_all[ds.s1:ds.s2])
        X_te_clean = imp.transform(ds.X_all[ds.s2:])

        n_neg, n_pos = (ds.y_sk == 0).sum(), (ds.y_sk == 1).sum()
        xgb = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / max(n_pos, 1),
            eval_metric='logloss', random_state=42, n_jobs=-1
        )
        xgb.fit(X_tr_clean, ds.y_sk, eval_set=[(X_va_clean, ds.y_va.numpy() if hasattr(ds.y_va, 'numpy') else ds.y_va)], verbose=False)
        y_xgb = xgb.predict(X_te_clean)
        p_xgb = xgb.predict_proba(X_te_clean)[:, 1]
        res['XGBoost'] = _metricas(y_te, y_xgb, p_xgb)
        res['XGBoost']['_feat_importances'] = dict(zip(getattr(ds, 'feature_names', [f'f{i}' for i in range(X_tr_clean.shape[1])]), xgb.feature_importances_))
        print(f"  [4] XGBoost           Acc={res['XGBoost']['acc']*100:.2f}%  F1={res['XGBoost']['f1']:.3f}  AUC={res['XGBoost']['auc']:.3f}")
    except Exception as e:
        p_xgb = np.zeros_like(y_te)
        print(f"  [4] XGBoost not executed: {e}")

    lstm_m = PureLSTM(ds.in_dim)
    dl = DataLoader(TensorDataset(ds.X_tr, ds.y_tr), batch_size=16, shuffle=True, drop_last=True)
    lstm_m, _ = _loop(lstm_m, dl, ds.X_va, ds.y_va)
    res['LSTM pura'], prob_lstm = _evaluar_pytorch(lstm_m, ds.X_te, y_te)
    res['LSTM pura']['_prob'] = prob_lstm
    print(f"  [5] Standalone LSTM   Acc={res['LSTM pura']['acc']*100:.2f}%  F1={res['LSTM pura']['f1']:.3f}  AUC={res['LSTM pura']['auc']:.3f}")

    res['_probs'] = {'Reg. Logística': p_lr, 'Random Forest': p_rf, 'Gradient Boosting': p_gb, 'XGBoost': p_xgb, 'LSTM pura': prob_lstm}
    res['_modelo_lstm'] = lstm_m
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed classical baselines
# ──────────────────────────────────────────────────────────────────────────────

def entrenar_baselines_multisemilla(ds: DatasetResult, seeds=range(15)) -> dict:
    """
    Executes a multi-seed replicate of classical baselines (RF, GB, XGBoost, and Standalone LSTM)
    to assess variance and address Q1 methodological review criteria.
    """
    from xgboost import XGBClassifier

    y_te = ds.y_te.numpy()
    y_te_x = ds.y_te_ext.numpy()
    ACP_te_np = ds.ACP_te.numpy() if hasattr(ds.ACP_te, 'numpy') else ds.ACP_te

    resultados_multi = {
        'Random Forest':     {'auc': [], 'fpr': [], 'rc_ACP': [], 'rc_p90': []},
        'Gradient Boosting': {'auc': [], 'fpr': [], 'rc_ACP': [], 'rc_p90': []},
        'XGBoost':           {'auc': [], 'fpr': [], 'rc_ACP': [], 'rc_p90': []},
        'LSTM pura':         {'auc': [], 'fpr': [], 'rc_ACP': [], 'rc_p90': []}
    }

    print(f"\n  Starting Classical Multi-Seed Replicate ({len(seeds)} seeds)...")

    for i, seed in enumerate(seeds):
        print(f"    -> Training Classical Seed {seed} ({i+1}/{len(seeds)})...")

        with deterministic_seed(seed):
            # 1. Random Forest
            rf_m, rf_imp = _safe_fit_sklearn(
                RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=seed, n_jobs=-1),
                ds.X_sk, ds.y_sk
            )
            _, p_rf = _safe_predict_sklearn(rf_m, rf_imp, ds.X_all[ds.s2:])

            # 2. Gradient Boosting
            gb_m, gb_imp = _safe_fit_sklearn(
                GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=seed),
                ds.X_sk, ds.y_sk
            )
            _, p_gb = _safe_predict_sklearn(gb_m, gb_imp, ds.X_all[ds.s2:])

            # 3. XGBoost
            imp = SimpleImputer(strategy='mean')
            X_tr_clean = imp.fit_transform(ds.X_sk)
            X_te_clean = imp.transform(ds.X_all[ds.s2:])
            n_neg, n_pos = (ds.y_sk == 0).sum(), (ds.y_sk == 1).sum()

            xgb = XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=n_neg / max(n_pos, 1),
                eval_metric='logloss', random_state=seed, n_jobs=-1
            )
            xgb.fit(X_tr_clean, ds.y_sk, verbose=False)
            p_xgb = xgb.predict_proba(X_te_clean)[:, 1]

            # 4. LSTM pura
            lstm_m = PureLSTM(ds.in_dim)
            dl = DataLoader(TensorDataset(ds.X_tr, ds.y_tr), batch_size=16, shuffle=True, drop_last=True)
            lstm_m, _ = _loop(lstm_m, dl, ds.X_va, ds.y_va)
            _, p_lstm = _evaluar_pytorch(lstm_m, ds.X_te, y_te)

        # Evaluate and store metrics for this seed
        preds_dict = {
            'Random Forest': p_rf, 'Gradient Boosting': p_gb,
            'XGBoost': p_xgb, 'LSTM pura': p_lstm
        }

        for nm, prob in preds_dict.items():
            pred = (prob >= 0.5).astype(float)
            met = _metricas(y_te, pred, prob)
            auc = met['auc']

            rc_90 = recall_score(y_te_x, pred, zero_division=0)
            rc_ACP = recall_score(y_te[ACP_te_np == 1], pred[ACP_te_np == 1], zero_division=0) if ACP_te_np.sum() > 0 else float('nan')

            tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')

            resultados_multi[nm]['auc'].append(auc)
            resultados_multi[nm]['fpr'].append(fpr)
            resultados_multi[nm]['rc_ACP'].append(rc_ACP)
            resultados_multi[nm]['rc_p90'].append(rc_90)

    # Build statistical summary (Mean +- Std)
    res_summary = {}
    for nm, metrics in resultados_multi.items():
        res_summary[nm] = {}
        for m_name, vals in metrics.items():
            arr = np.array(vals)
            arr = arr[~np.isnan(arr)]
            res_summary[nm][m_name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'valores': arr.tolist()
            }

    return res_summary


def entrenar_lstm_standalone_semilla(ds: DatasetResult, seed: int = 0) -> dict:
    """Trains a standalone LSTM with a specific seed and returns its metrics.
    Used in backfill_lstm_standalone() to pair with quantum seeds."""
    y_te = ds.y_te.numpy()
    y_te_ext = ds.y_te_ext.numpy()
    ACP_te_np = ds.ACP_te.numpy() if hasattr(ds.ACP_te, 'numpy') else ds.ACP_te

    with deterministic_seed(seed):
        lstm_m = PureLSTM(ds.in_dim)
        dl = DataLoader(TensorDataset(ds.X_tr, ds.y_tr), batch_size=16, shuffle=True, drop_last=True)
        lstm_m, _ = _loop(lstm_m, dl, ds.X_va, ds.y_va)
        met, prob = _evaluar_pytorch(lstm_m, ds.X_te, y_te)

    pred = (prob >= 0.5).astype(float)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
    rc_ACP = recall_score(y_te[ACP_te_np == 1], pred[ACP_te_np == 1], zero_division=0) if ACP_te_np.sum() > 0 else float('nan')
    rc_p90 = recall_score(y_te_ext, pred, zero_division=0)

    return {**met, 'fpr': fpr, 'rc_ACP': rc_ACP, 'rc_p90': rc_p90, '_prob': prob}


# English aliases for public API
train_classical_baselines = entrenar_baselines
train_classical_baselines_multiseed = entrenar_baselines_multisemilla
train_lstm_standalone_seed = entrenar_lstm_standalone_semilla

