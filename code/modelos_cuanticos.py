"""
modelos_cuanticos.py — Quantum models and hybrid architectures
===============================================================
Contains quantum circuit definitions, LSTM+QNN, PI-QNN with
thermodynamic physics loss, Quantum Transfer Learning (QTL),
Layer-by-Layer training (LbL), Quantum Kernel SVM (QKSVM),
and ablation experiments.
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

from features import DatasetResult
from modelos_utils import _metricas, _evaluar_pytorch, _loop


# ──────────────────────────────────────────────────────────────────────────────
# Quantum circuit definition
# ──────────────────────────────────────────────────────────────────────────────

def _make_qnode(n_q: int, n_l: int):
    dev = qml.device('default.qubit', wires=n_q)
    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circ(W, x):
        for l in range(n_l):
            for i in range(n_q):
                qml.RY(x[i % x.shape[0]], wires=i)
                qml.RZ(x[i % x.shape[0]], wires=i)
            qml.StronglyEntanglingLayers(W[l:l+1], wires=range(n_q))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]
    return circ


# ──────────────────────────────────────────────────────────────────────────────
# LSTM+QNN Hybrid Model
# ──────────────────────────────────────────────────────────────────────────────

class LstmQNN(nn.Module):
    def __init__(self, in_dim: int, n_q: int = 4, n_l: int = 2, physics: bool = False):
        super().__init__()
        self.n_q = n_q
        self.physics = physics
        self.lstm = nn.LSTM(1, 8, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(16 + in_dim, n_q), nn.Dropout(0.1))
        self.W = nn.Parameter(torch.empty(n_l, n_q, 3).uniform_(-math.pi, math.pi))
        self.circ = _make_qnode(n_q, n_l)
        self.norm = nn.LayerNorm(n_q)
        self.clf = nn.Linear(n_q, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x[:, :12].unsqueeze(-1))
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        st = self.proj(torch.cat([h, x], dim=1))
        qi = torch.sigmoid(st) * math.pi
        qo = torch.stack([torch.stack(self.circ(self.W, xi)) for xi in qi])
        pr_logits = self.clf(self.norm(qo)).squeeze(-1)
        return (pr_logits, qo) if self.physics else pr_logits


# ──────────────────────────────────────────────────────────────────────────────
# Thermodynamic physics-informed loss (Clausius-Clapeyron + MSE)
# ──────────────────────────────────────────────────────────────────────────────

def make_thermodynamic_physics_loss(pos_weight: torch.Tensor):
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    CP = 7.0

    def loss_fn(logits, ytrue, qout, T_anomaly_batch, epoch, total_epochs,
                lambda_cc=0.2, lambda_mse=0.1):
        bce = crit(logits, ytrue)
        probs = torch.sigmoid(logits)
        alpha = min(1.0, epoch / (total_epochs * 0.6))

        cc_capacity = torch.exp((CP / 100.0) * T_anomaly_batch * 10.0)
        L_cc = (torch.relu(probs - 0.6) * torch.relu(1.0 - cc_capacity)).mean()

        energy = torch.relu(qout[:, 0]) + torch.relu(qout[:, 1])
        L_mse = ((energy - torch.relu(qout[:, 3]) - 0.5) ** 2).mean()

        return bce + alpha * (lambda_cc * L_cc + lambda_mse * L_mse)
    return loss_fn


# ──────────────────────────────────────────────────────────────────────────────
# Training loop with physics-informed loss
# ──────────────────────────────────────────────────────────────────────────────

def _loop_with_physics(model, dl, X_va, y_va, epochs=25, patience=6, lr=5e-3):
    """Loop con física usando pos_weight también en validación."""
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
        for bX, by, bT in dl:
            opt.zero_grad()
            logits, qo = model(bX)
            loss = model.physics_fn(logits, by, qo, bT, ep, epochs)
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
# Train LSTM+QNN (base) and LSTM+PI-QNN (thermodynamic)
# ──────────────────────────────────────────────────────────────────────────────

def entrenar_qml(ds: DatasetResult) -> dict:
    y_te = ds.y_te.numpy()
    results = {}

    n_neg_t = (ds.y_tr == 0).sum().float()
    n_pos_t = (ds.y_tr == 1).sum().float()
    pos_w_t = torch.tensor([n_neg_t / (n_pos_t + 1e-5)])
    physics_fn = make_thermodynamic_physics_loss(pos_w_t)

    T_anomaly_tr = torch.tensor([ds.t_anomaly_mes[m] for m in ds.m_tr], dtype=torch.float64)

    dl_base = DataLoader(TensorDataset(ds.X_tr, ds.y_tr), batch_size=16, shuffle=True, drop_last=True)
    dl_pi   = DataLoader(TensorDataset(ds.X_tr, ds.y_tr, T_anomaly_tr), batch_size=16, shuffle=True, drop_last=True)

    print("\n  Entrenando LSTM+QNN base ...")
    qnn = LstmQNN(ds.in_dim, physics=False)
    qnn, h_qnn = _loop(qnn, dl_base, ds.X_va, ds.y_va)
    m_qnn, p_qnn = _evaluar_pytorch(qnn, ds.X_te, y_te)
    results['LSTM+QNN'] = {**m_qnn, '_prob': p_qnn, '_hist': h_qnn, '_modelo': qnn}
    print(f"  [6] LSTM+QNN          Acc={m_qnn['acc']*100:.2f}%  F1={m_qnn['f1']:.3f}  AUC={m_qnn['auc']:.3f}")

    print("\n  Entrenando LSTM+PI-QNN (Thermodynamic) ...")
    piqnn = LstmQNN(ds.in_dim, physics=True)
    piqnn.physics_fn = physics_fn

    piqnn, h_pi = _loop_with_physics(piqnn, dl_pi, ds.X_va, ds.y_va)
    m_pi, p_pi = _evaluar_pytorch(piqnn, ds.X_te, y_te)
    results['LSTM+PI-QNN'] = {**m_pi, '_prob': p_pi, '_hist': h_pi}
    print(f"  [7] LSTM+PI-QNN (CC+MSE) Acc={m_pi['acc']*100:.2f}%  F1={m_pi['f1']:.3f}  AUC={m_pi['auc']:.3f}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Quantum Kernel SVM (QKSVM)
# ──────────────────────────────────────────────────────────────────────────────

def entrenar_qksvm(ds: DatasetResult) -> dict:
    from sklearn.svm import SVC
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.decomposition import PCA

    N_Q        = 4
    N_LANDMARK = 64
    dev = qml.device('default.qubit', wires=N_Q)

    @qml.qnode(dev, interface='autograd')
    def _qk_circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(N_Q), rotation='Y')
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(N_Q), rotation='Y')
        return qml.probs(wires=range(N_Q))

    def _kernel_row(xi, X_ref):
        return np.array([_qk_circuit(xi, xj)[0] for xj in X_ref])

    std_sc = StandardScaler()
    X_tr_std = std_sc.fit_transform(ds.X_sk)
    X_te_std = std_sc.transform(ds.X_all[ds.s2:])

    pca = PCA(n_components=N_Q, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_std)
    X_te_pca = pca.transform(X_te_std)

    sc = MinMaxScaler(feature_range=(0, np.pi))
    X_tr_q = sc.fit_transform(X_tr_pca)
    X_te_q = sc.transform(X_te_pca)

    rng = np.random.RandomState(42)
    idx_pos = np.where(ds.y_sk == 1)[0]
    idx_neg = np.where(ds.y_sk == 0)[0]
    n_pos_lm = min(N_LANDMARK // 2, len(idx_pos))
    n_neg_lm = N_LANDMARK - n_pos_lm
    landmarks = np.concatenate([
        rng.choice(idx_pos, n_pos_lm, replace=False),
        rng.choice(idx_neg, min(n_neg_lm, len(idx_neg)), replace=False),
    ])
    X_lm = X_tr_q[landmarks]

    t0 = time.time()
    K_lm_lm = np.vstack([_kernel_row(x, X_lm) for x in X_lm])
    K_tr_lm = np.vstack([_kernel_row(x, X_lm) for x in X_tr_q])
    K_te_lm = np.vstack([_kernel_row(x, X_lm) for x in X_te_q])

    K_lm_inv = np.linalg.pinv(K_lm_lm + 1e-6 * np.eye(len(landmarks)))
    K_tr = K_tr_lm @ K_lm_inv @ K_tr_lm.T
    K_te = K_te_lm @ K_lm_inv @ K_tr_lm.T

    n_neg = (ds.y_sk == 0).sum()
    n_pos = (ds.y_sk == 1).sum()
    cw    = {0: 1.0, 1: float(n_neg / max(n_pos, 1))}

    svm = SVC(kernel='precomputed', C=1.0, class_weight=cw, probability=True, random_state=42)
    svm.fit(K_tr, ds.y_sk)

    y_pred_q = svm.predict(K_te)
    y_prob_q = svm.predict_proba(K_te)[:, 1]

    dt  = time.time() - t0
    met = _metricas(ds.y_te.numpy(), y_pred_q, y_prob_q)
    print(f"  [8] QKSVM             Acc={met['acc']*100:.2f}%  F1={met['f1']:.3f}  AUC={met['auc']:.3f}  ({dt:.0f}s)")
    return {'QKSVM': {**met, '_prob': y_prob_q}}


# ──────────────────────────────────────────────────────────────────────────────
# Quantum ablation study
# ──────────────────────────────────────────────────────────────────────────────

def entrenar_ablacion(ds: DatasetResult) -> dict:
    y_te  = ds.y_te.numpy()
    dl    = DataLoader(TensorDataset(ds.X_tr, ds.y_tr), batch_size=16, shuffle=True, drop_last=True)
    res   = {}
    configs = [(2,2,'A'),(4,2,'A'),(6,2,'A'), (4,1,'B'),(4,3,'B'),(4,4,'B')]

    for nq, nl, grupo in configs:
        tag = f"{nq}q-{nl}L"
        print(f"  [{grupo}] {tag} ...", end=" ", flush=True)
        t0  = time.time()
        m   = LstmQNN(ds.in_dim, n_q=nq, n_l=nl, physics=False)
        m, _= _loop(m, dl, ds.X_va, ds.y_va, epochs=20, patience=5)
        met, _ = _evaluar_pytorch(m, ds.X_te, y_te)
        dt  = time.time() - t0
        res[tag] = {**met, 't': round(dt, 1), 'grupo': grupo}
        print(f"Acc={met['acc']*100:.2f}%  F1={met['f1']:.3f}  ({dt:.0f}s)")
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Quantum Transfer Learning (QTL)
# ──────────────────────────────────────────────────────────────────────────────

class QuantumTransferLearning(nn.Module):
    def __init__(self, encoder_clasico: nn.Module, in_embed: int,
                 n_q: int = 4, n_l: int = 2, use_physics: bool = True):
        super().__init__()
        self.n_q = n_q
        self.encoder = encoder_clasico
        self.use_physics = use_physics

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.encoder.double()

        self.adaptador = nn.Sequential(
            nn.Linear(in_embed, n_q * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(n_q * 2, n_q),
            nn.Sigmoid()
        ).double()

        dev = qml.device('default.qubit', wires=n_q)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def qtl_circuit(W, x):
            qml.AngleEmbedding(x * math.pi, wires=range(n_q), rotation='Y')
            qml.StronglyEntanglingLayers(W, wires=range(n_q))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

        self.circ = qtl_circuit
        self.W = nn.Parameter(torch.empty(n_l, n_q, 3, dtype=torch.float64).uniform_(-0.1, 0.1))
        self.norm = nn.LayerNorm(n_q).double()
        self.clf = nn.Linear(n_q, 1).double()

        if use_physics:
            n_neg = torch.tensor([1.0])
            self.physics_fn = make_thermodynamic_physics_loss(n_neg)
        else:
            self.physics_fn = None

    def forward(self, x):
        with torch.no_grad():
            x64 = x.double()
            _, (hn, _) = self.encoder.lstm(x64[:, :12].unsqueeze(-1))
            h_lstm = torch.cat([hn[-2], hn[-1]], dim=1)
            embed_raw = torch.cat([h_lstm, x64], dim=1)
            try:
                embed = self.encoder.bn(embed_raw.float()).double()
            except RuntimeError:
                embed = self.encoder.bn(embed_raw)

        qi = self.adaptador(embed) * math.pi
        qo = torch.stack([torch.stack(self.circ(self.W, xi)) for xi in qi])
        logits = self.clf(self.norm(qo)).squeeze(-1)

        if self.use_physics:
            return logits, qo
        return logits


def entrenar_qtl(ds: DatasetResult, lstm_preentrenado: nn.Module, use_physics: bool = True) -> dict:
    in_embed = 32 + ds.in_dim
    qtl = QuantumTransferLearning(lstm_preentrenado, in_embed=in_embed, n_q=4, n_l=2, use_physics=use_physics)

    n_neg = (ds.y_tr == 0).sum().float()
    n_pos = (ds.y_tr == 1).sum().float()
    pos_w = torch.tensor([n_neg / (n_pos + 1e-5)], dtype=torch.float64)

    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    crit_val = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    if use_physics:
        qtl.physics_fn = make_thermodynamic_physics_loss(pos_w)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, qtl.parameters()), lr=2e-3, weight_decay=1e-4)

    best_vl = float('inf')
    best_state = None
    ni = 0

    T_anomaly_tr = torch.tensor([ds.t_anomaly_mes[m] for m in ds.m_tr], dtype=torch.float64)
    dl = DataLoader(TensorDataset(ds.X_tr, ds.y_tr, T_anomaly_tr), batch_size=8, shuffle=True, drop_last=True)

    print("\n  [QTL] Entrenando Quantum Transfer Learning ...")

    for ep in range(1, 16):
        qtl.train()
        for bX, by, bT in dl:
            opt.zero_grad()
            bX64, by64, bT64 = bX.double(), by.double(), bT.double()

            if use_physics:
                logits, qo_out = qtl(bX64)
                loss = qtl.physics_fn(logits, by64, qo_out, bT64, ep, 16)
            else:
                logits = qtl(bX64)
                loss = crit(logits, by64)

            loss.backward()
            nn.utils.clip_grad_norm_(qtl.parameters(), 1.0)
            opt.step()

        qtl.eval()
        with torch.no_grad():
            vout = qtl(ds.X_va.double())
            va_logits = vout[0] if isinstance(vout, tuple) else vout
            vl = crit_val(va_logits, ds.y_va.double()).item()

        if vl < best_vl - 1e-4:
            best_vl = vl
            best_state = {k: v.clone() for k, v in qtl.state_dict().items()}
            ni = 0
        else:
            ni += 1
            if ni >= 4:
                break

    if best_state:
        qtl.load_state_dict(best_state)

    y_te = ds.y_te.numpy()
    qtl.eval()
    with torch.no_grad():
        out_te = qtl(ds.X_te.double())
        logits_te = out_te[0] if isinstance(out_te, tuple) else out_te
        prob_te = torch.sigmoid(logits_te).numpy()

    pred_te = (prob_te >= 0.5).astype(float)
    met = _metricas(y_te, pred_te, prob_te)

    print(f"  [QTL] LSTM+QTL        Acc={met['acc']*100:.2f}%  F1={met['f1']:.3f}  AUC={met['auc']:.3f}")
    return {'LSTM+QTL': {**met, '_prob': prob_te}}


# ──────────────────────────────────────────────────────────────────────────────
# Layer-by-Layer Training (LbL) with physics loss
# ──────────────────────────────────────────────────────────────────────────────

def entrenar_qnn_layerbylayer(ds: DatasetResult) -> dict:
    N_Q, N_L = 4, 3
    model = LstmQNN(ds.in_dim, n_q=N_Q, n_l=N_L, physics=True)

    n_neg = (ds.y_tr == 0).sum().float()
    n_pos = (ds.y_tr == 1).sum().float()
    pos_w = torch.tensor([n_neg / (n_pos + 1e-5)], dtype=torch.float64)

    physics_fn = make_thermodynamic_physics_loss(pos_w)
    model.physics_fn = physics_fn

    T_anomaly_tr = torch.tensor([ds.t_anomaly_mes[m] for m in ds.m_tr], dtype=torch.float64)
    dl = DataLoader(TensorDataset(ds.X_tr, ds.y_tr, T_anomaly_tr), batch_size=16, shuffle=True, drop_last=True)

    crit_val = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    print("\n  [LbL] Layer-by-Layer Training (3 capas × 10 épocas + fine-tune)")

    for capa_activa in range(N_L):
        print(f"  [LbL]   Fase {capa_activa+1}/{N_L}: entrenando capa {capa_activa+1} del circuito ...")

        def _make_mask_hook(capa):
            def _hook(grad):
                mask = torch.zeros_like(grad)
                mask[capa] = 1.0
                return grad * mask
            return _hook

        handle = model.W.register_hook(_make_mask_hook(capa_activa))
        opt_fase = torch.optim.Adam([model.W], lr=5e-3)

        for ep in range(1, 11):
            model.train()
            for bX, by, bT in dl:
                opt_fase.zero_grad()
                pr, qo = model(bX)
                loss = physics_fn(pr, by, qo, bT, ep, 10)
                loss.backward()
                opt_fase.step()
        handle.remove()

    print("  [LbL]   Fine-tuning global (25 épocas, AdamW lr=5e-4) ...")
    for param in model.parameters():
        param.requires_grad_(True)

    opt_ft = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ft, patience=4, factor=0.5)

    best_vl, best_state, ni = float('inf'), None, 0

    for ep in range(1, 26):
        model.train()
        for bX, by, bT in dl:
            opt_ft.zero_grad()
            pr, qo = model(bX)
            loss = physics_fn(pr, by, qo, bT, ep, 25)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_ft.step()

        model.eval()
        with torch.no_grad():
            vout = model(ds.X_va)
            vl = crit_val(vout[0].double(), ds.y_va.double()).item()
        sch_ft.step(vl)

        if vl < best_vl - 1e-4:
            best_vl = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ni = 0
        else:
            ni += 1
            if ni >= 7:
                break

    if best_state:
        model.load_state_dict(best_state)

    y_te = ds.y_te.numpy()
    met, prob = _evaluar_pytorch(model, ds.X_te, y_te)
    print(f"  [LbL] LbL-QNN (CC+MSE) Acc={met['acc']*100:.2f}%  F1={met['f1']:.3f}  AUC={met['auc']:.3f}")
    return {'LbL-QNN': {**met, '_prob': prob, '_modelo': model}}
