"""
experimentos_control.py — Control & supplementary experiments
==============================================================
Contains experiments extracted from the Colab notebook that serve as
controls and supplementary analyses for the paper:

  1. LstmAmplitudeQNN     — Amplitude Embedding circuit variant
  2. LstmReuploadingQNN   — Data Re-uploading circuit variant
  3. LstmLinear4D          — Linear parity control (20 params)
  4. LstmLeakyMLP4D        — LeakyReLU MLP parity control (28 params)
  5. Lambda sensitivity    — Sweep over λ_cc in LbL-QNN
  6. PR-AUC calculation    — Precision-Recall AUC from checkpoints
  7. ACP diagnostics      — Mann-Whitney tests on top-day share

Usage:
    python experimentos_control.py --experiment amplitude
    python experimentos_control.py --experiment reuploading
    python experimentos_control.py --experiment parity
    python experimentos_control.py --experiment lambda-sweep
    python experimentos_control.py --experiment pr-auc
    python experimentos_control.py --experiment ACP-diagnostics
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import wilcoxon
import pennylane as qml

from modelos_utils import _metricas, _evaluar_pytorch, _loop


# ══════════════════════════════════════════════════════════════════════════
# 1. AMPLITUDE EMBEDDING VARIANT
# ══════════════════════════════════════════════════════════════════════════

class LstmAmplitudeQNN(nn.Module):
    """LSTM encoder → Amplitude Embedding → StronglyEntanglingLayers."""
    def __init__(self, in_dim: int, n_q: int = 4, n_l: int = 2):
        super().__init__()
        self.n_q = n_q
        self.lstm = nn.LSTM(1, 8, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(16 + in_dim, 2**n_q), nn.Dropout(0.1))
        self.W = nn.Parameter(torch.empty(n_l, n_q, 3).uniform_(-math.pi, math.pi))

        dev = qml.device('default.qubit', wires=n_q)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def amplitude_circuit(W, x):
            qml.AmplitudeEmbedding(x, wires=range(n_q), normalize=True)
            qml.StronglyEntanglingLayers(W, wires=range(n_q))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

        self.circ = amplitude_circuit
        self.norm = nn.LayerNorm(n_q)
        self.clf = nn.Linear(n_q, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x[:, :12].unsqueeze(-1))
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        st = self.proj(torch.cat([h, x], dim=1))
        amp = torch.softmax(st, dim=-1)
        qo = torch.stack([torch.stack(self.circ(self.W, xi)) for xi in amp])
        return self.clf(self.norm(qo)).squeeze(-1)


def run_amplitude_experiment(ds, seeds=range(15)):
    """Run 15-seed Amplitude Embedding experiment."""
    aucs = []
    print("\n  Running 15-seed Amplitude Embedding experiment...")

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = LstmAmplitudeQNN(ds.in_dim)
        dl = DataLoader(TensorDataset(ds.X_tr, ds.y_tr), batch_size=16, shuffle=True, drop_last=True)
        model, _ = _loop(model, dl, ds.X_va, ds.y_va, epochs=25, patience=6, lr=5e-3)
        met, _ = _evaluar_pytorch(model, ds.X_te, ds.y_te.numpy())
        aucs.append(met['auc'])
        print(f"    Seed {seed}: AUC = {met['auc']:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs, ddof=1)
    print(f"\n  Amplitude Embedding: AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    return aucs


# ══════════════════════════════════════════════════════════════════════════
# 2. DATA RE-UPLOADING VARIANT
# ══════════════════════════════════════════════════════════════════════════

class LstmReuploadingQNN(nn.Module):
    """LSTM encoder → Data Re-uploading (interleaved RX-RY per layer)."""
    def __init__(self, in_dim: int, n_q: int = 4, n_l: int = 2):
        super().__init__()
        self.n_q = n_q
        self.n_l = n_l
        self.lstm = nn.LSTM(1, 8, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(16 + in_dim, n_q), nn.Dropout(0.1))
        self.W = nn.Parameter(torch.empty(n_l, n_q, 2).uniform_(-math.pi, math.pi))

        dev = qml.device('default.qubit', wires=n_q)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def reuploading_circuit(W, x, n_q, n_l):
            for l in range(n_l):
                for i in range(n_q):
                    qml.RX(x[i % x.shape[0]], wires=i)
                    qml.RY(W[l, i, 0], wires=i)
                    qml.RX(W[l, i, 1], wires=i)
                for i in range(n_q - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_q > 1:
                    qml.CNOT(wires=[n_q - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

        self.circ = reuploading_circuit
        self.norm = nn.LayerNorm(n_q)
        self.clf = nn.Linear(n_q, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x[:, :12].unsqueeze(-1))
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        st = self.proj(torch.cat([h, x], dim=1))
        qi = torch.sigmoid(st) * math.pi
        qo = torch.stack([torch.stack(self.circ(self.W, xi, self.n_q, self.n_l)) for xi in qi])
        return self.clf(self.norm(qo)).squeeze(-1)


def run_reuploading_experiment(ds, seeds=range(15)):
    """Run 15-seed Data Re-uploading experiment."""
    aucs = []
    print("\n  Running 15-seed Data Re-uploading experiment...")

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = LstmReuploadingQNN(ds.in_dim)
        dl = DataLoader(TensorDataset(ds.X_tr, ds.y_tr), batch_size=16, shuffle=True, drop_last=True)
        model, _ = _loop(model, dl, ds.X_va, ds.y_va, epochs=25, patience=6, lr=5e-3)
        met, _ = _evaluar_pytorch(model, ds.X_te, ds.y_te.numpy())
        aucs.append(met['auc'])
        print(f"    Seed {seed}: AUC = {met['auc']:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs, ddof=1)
    print(f"\n  Data Re-uploading: AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    return aucs


# ══════════════════════════════════════════════════════════════════════════
# 3. MLP PARITY CONTROLS
# ══════════════════════════════════════════════════════════════════════════

class LinearBottleneckControl(nn.Module):
    """Linear bottleneck (4→4, 20 parameters)."""
    def __init__(self, n_q: int = 4):
        super().__init__()
        self.linear = nn.Linear(n_q, n_q)  # 4*4 + 4 = 20 params

    def forward(self, qi):
        return self.linear(qi)


class LstmLinear4D(nn.Module):
    """LSTM + Linear bottleneck parity control (20 params)."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(1, 8, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(16 + in_dim, 4), nn.Dropout(0.1))
        self.bottleneck = LinearBottleneckControl(n_q=4)
        self.norm = nn.LayerNorm(4)
        self.clf = nn.Linear(4, 1)

    def forward(self, x):
        x = x.float()
        _, (hn, _) = self.lstm(x[:, :12].unsqueeze(-1))
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        st = self.proj(torch.cat([h, x], dim=1))
        qi = torch.sigmoid(st) * math.pi
        out = self.bottleneck(qi)
        return self.clf(self.norm(out)).squeeze(-1)


class LstmLeakyMLP4D(nn.Module):
    """LSTM + LeakyReLU MLP parity control (28 params)."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(1, 8, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(16 + in_dim, 4), nn.Dropout(0.1))

        self.mlp = nn.Sequential(
            nn.Linear(4, 2),
            nn.LeakyReLU(0.1),
            nn.Linear(2, 2),
            nn.LeakyReLU(0.1),
            nn.Linear(2, 4)
        )

        # He initialization
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')

        self.norm = nn.LayerNorm(4)
        self.clf = nn.Linear(4, 1)

    def forward(self, x):
        x = x.float()
        _, (hn, _) = self.lstm(x[:, :12].unsqueeze(-1))
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        st = self.proj(torch.cat([h, x], dim=1))
        qi = torch.sigmoid(st) * math.pi
        out = self.mlp(qi)
        return self.clf(self.norm(out)).squeeze(-1)


def run_parity_experiment(ds, qnn_auc_array, model_class, model_name, seeds=range(15)):
    """Run parity control experiment with Wilcoxon test against QNN."""
    aucs = []
    print(f"\n  Starting 15-seed evaluation for {model_name}...")

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model_class(ds.in_dim)
        dl = DataLoader(TensorDataset(ds.X_tr, ds.y_tr), batch_size=16, shuffle=True, drop_last=True)
        model, _ = _loop(model, dl, ds.X_va, ds.y_va, epochs=25, patience=6, lr=5e-3)
        met, _ = _evaluar_pytorch(model, ds.X_te, ds.y_te.numpy())
        aucs.append(met['auc'])
        print(f"    Seed {seed} -> AUC: {met['auc']:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs, ddof=1)
    max_auc = np.max(aucs)

    print("\n" + "="*50)
    print(f"RESULTS FOR {model_name}:")
    print("="*50)
    print(f"Mean AUC:  {mean_auc:.4f}")
    print(f"Std. Dev:  {std_auc:.4f}")
    print(f"Max AUC:   {max_auc:.4f}")

    if qnn_auc_array is not None and len(qnn_auc_array) == len(aucs):
        stat, p_val = wilcoxon(aucs, qnn_auc_array)
        print(f"Wilcoxon p-value vs LSTM+QNN: {p_val:.4f}")

    return aucs


# ══════════════════════════════════════════════════════════════════════════
# 4. LAMBDA SENSITIVITY SWEEP
# ══════════════════════════════════════════════════════════════════════════

def run_lambda_sensitivity(ds, lambda_grid=None, n_seeds=3):
    """Sweep over λ_cc values for LbL-QNN."""
    from modelos_cuanticos import LstmQNN, make_thermodynamic_physics_loss

    if lambda_grid is None:
        lambda_grid = [0.1, 0.2, 0.5]

    torch.set_default_dtype(torch.float64)

    X_tr = ds.X_tr.double()
    y_tr = ds.y_tr.double()
    X_te = ds.X_te.double()
    y_te = ds.y_te.double()

    n_neg = (y_tr == 0).sum().double()
    n_pos = (y_tr == 1).sum().double()
    POS_W = torch.tensor([n_neg / (n_pos + 1e-5)], dtype=torch.float64)

    T_ANOMALY_TR = torch.tensor([ds.t_anomaly_mes[m] for m in ds.m_tr], dtype=torch.float64)
    train_dataset = TensorDataset(X_tr, y_tr, T_ANOMALY_TR)
    dl = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

    def make_physics_loss_lambda(pos_weight, lambda_cc, lambda_mse=0.1):
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        CP = 7.0
        def loss_fn(logits, ytrue, qout, T_anomaly_batch, epoch, total_epochs):
            bce = crit(logits, ytrue)
            probs = torch.sigmoid(logits)
            alpha = min(1.0, epoch / (total_epochs * 0.6))

            cc_capacity = torch.exp((CP / 100.0) * T_anomaly_batch * 10.0)
            L_cc = (torch.relu(probs - 0.6) * torch.relu(1.0 - cc_capacity)).mean()

            energy = torch.relu(qout[:, 0]) + torch.relu(qout[:, 1])
            L_mse = ((energy - torch.relu(qout[:, 3]) - 0.5) ** 2).mean()

            return bce + alpha * (lambda_cc * L_cc + lambda_mse * L_mse)
        return loss_fn

    def entrenar_lbl_con_lambda(ds, lambda_cc, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        N_Q, N_L = 4, 3

        model = LstmQNN(ds.in_dim, n_q=N_Q, n_l=N_L, physics=True).double()
        model.physics_fn = make_physics_loss_lambda(POS_W, lambda_cc)

        for capa_activa in range(N_L):
            def _make_mask_hook(capa):
                def _hook(grad):
                    mask = torch.zeros_like(grad); mask[capa] = 1.0
                    return grad * mask
                return _hook

            handle = model.W.register_hook(_make_mask_hook(capa_activa))
            opt_fase = torch.optim.Adam([model.W], lr=5e-3)
            for ep in range(1, 11):
                model.train()
                for bX, by, bT in dl:
                    opt_fase.zero_grad()
                    pr, qo = model(bX)
                    loss = model.physics_fn(pr, by, qo, bT, ep, 10)
                    loss.backward()
                    opt_fase.step()
            handle.remove()

        for p in model.parameters():
            p.requires_grad_(True)
        opt_ft = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

        for ep in range(1, 26):
            model.train()
            for bX, by, bT in dl:
                opt_ft.zero_grad()
                pr, qo = model(bX)
                loss = model.physics_fn(pr, by, qo, bT, ep, 25)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_ft.step()

        met, _ = _evaluar_pytorch(model, X_te, y_te.numpy())
        return met['auc']

    resultados = {}
    print(f"[+] Starting lambda sensitivity ({len(lambda_grid)*n_seeds} total runs)...")
    start_time = time.time()

    for lam in lambda_grid:
        print(f"  -> Training for λ_cc={lam}...")
        aucs = []
        for seed in range(n_seeds):
            seed_start = time.time()
            auc_val = entrenar_lbl_con_lambda(ds, lam, seed)
            aucs.append(auc_val)
            print(f"     [v] Seed {seed} done in {(time.time()-seed_start)/60:.2f} min. AUC={auc_val:.4f}")

        resultados[lam] = {'mean': float(np.mean(aucs)), 'std': float(np.std(aucs, ddof=1)), 'valores': aucs}
        print(f"  [v] λ_cc={lam}: AUC={resultados[lam]['mean']:.4f} ± {resultados[lam]['std']:.4f}\n")

    print(f"[v] Total time: {(time.time()-start_time)/3600:.2f} hours")
    return resultados


# ══════════════════════════════════════════════════════════════════════════
# 5. PR-AUC CALCULATION FROM CHECKPOINTS
# ══════════════════════════════════════════════════════════════════════════

def calculate_pr_auc(ckpt_dir='./resultados/checkpoints'):
    """Calculate PR-AUC from saved checkpoints."""
    import os
    from sklearn.metrics import average_precision_score, roc_auc_score

    data_loaded = torch.load(os.path.join(ckpt_dir, '01_data.pt'), weights_only=False)
    ds = None
    if isinstance(data_loaded, tuple):
        for item in data_loaded:
            if hasattr(item, 'y_te'):
                ds = item
                break
    else:
        ds = data_loaded

    if ds is None or not hasattr(ds, 'y_te'):
        raise ValueError("Could not find DatasetResult in checkpoint.")

    y_te = ds.y_te.numpy()

    def extraer_probs(res: dict) -> dict:
        out = dict(res.get('_probs', {}))
        for k, v in res.items():
            if k.startswith('_'):
                continue
            if isinstance(v, dict) and '_prob' in v:
                out[k] = v['_prob']
        return out

    checkpoints = [
        os.path.join(ckpt_dir, '02_baselines.pt'),
        os.path.join(ckpt_dir, '03_qml.pt'),
        os.path.join(ckpt_dir, '04_qksvm.pt'),
        os.path.join(ckpt_dir, '08_validacion.pt')
    ]

    probs_todos = {}
    for ckpt in checkpoints:
        try:
            res = torch.load(ckpt, weights_only=False)
            probs_todos.update(extraer_probs(res))
        except FileNotFoundError:
            print(f"[!] {ckpt} not found, skipping.")
        except Exception as e:
            print(f"[!] {ckpt} unexpected format ({e}), skipping.")

    base_rate = y_te.mean()
    print(f"\nBase rate (positive prevalence in test): {base_rate:.4f}\n")

    resultados_pr = {}
    print(f"{'Model':<28} {'ROC-AUC':>9} {'PR-AUC':>9}")
    print("-" * 50)
    for nm, prob in sorted(probs_todos.items()):
        roc = roc_auc_score(y_te, prob)
        pr = average_precision_score(y_te, prob)
        resultados_pr[nm] = {'roc_auc': roc, 'pr_auc': pr}
        print(f"{nm:<28} {roc:>9.4f} {pr:>9.4f}")

    return resultados_pr


# ══════════════════════════════════════════════════════════════════════════
# 6. ACP DIAGNOSTICS (Mann-Whitney)
# ══════════════════════════════════════════════════════════════════════════

def run_ACP_diagnostics(ckpt_path='./resultados/checkpoints/01_data.pt'):
    """Run ACP vs non-ACP diagnostic comparison using Mann-Whitney U."""
    from scipy import stats

    loaded = torch.load(ckpt_path, weights_only=False)
    df_raw = loaded[0].copy()
    ds = loaded[1]

    print(f"P60 threshold used in pipeline (ds.umbral): {ds.umbral}")

    df_raw['P60_exceeded'] = df_raw['precip'] > ds.umbral
    df_raw['is_ACP_month'] = df_raw['acp_flag'].astype(bool)
    df_raw['top_day_share'] = df_raw['precip_max_dia'] / df_raw['precip'].replace(0, np.nan)

    p60_months = df_raw[df_raw['P60_exceeded']]

    print(f"\nTotal months in panel: {len(df_raw)}")
    print(f"Months exceeding P60: {len(p60_months)}")
    print(f"  ACP: {p60_months['is_ACP_month'].sum()}")
    print(f"  Non-ACP: {(~p60_months['is_ACP_month']).sum()}")

    resultados = {}
    grupos_share = {}
    grupos_extreme_days = {}

    for label, mask in [("ACP", p60_months['is_ACP_month']),
                         ("No-ACP (P60-exceeding)", ~p60_months['is_ACP_month'])]:
        sub = p60_months[mask]
        grupos_share[label] = sub['top_day_share'].dropna().values
        grupos_extreme_days[label] = sub['dias_extremos'].values

        resultados[label] = {
            'n_meses': int(len(sub)),
            'mean_top_day_share': float(sub['top_day_share'].mean()),
            'std_top_day_share': float(sub['top_day_share'].std(ddof=1)),
            'mean_dias_extremos': float(sub['dias_extremos'].mean()),
            'std_dias_extremos': float(sub['dias_extremos'].std(ddof=1)),
        }

    u_share, p_share = stats.mannwhitneyu(
        grupos_share["ACP"], grupos_share["No-ACP (P60-exceeding)"], alternative='two-sided'
    )
    u_days, p_days = stats.mannwhitneyu(
        grupos_extreme_days["ACP"], grupos_extreme_days["No-ACP (P60-exceeding)"], alternative='two-sided'
    )

    resultados['test_top_day_share'] = {'U': float(u_share), 'p': float(p_share)}
    resultados['test_dias_extremos'] = {'U': float(u_days), 'p': float(p_days)}

    print("\n=== RESULTS ===")
    for grupo in ["ACP", "No-ACP (P60-exceeding)"]:
        r = resultados[grupo]
        print(f"\n{grupo} (n={r['n_meses']}):")
        print(f"  top_day_share: {r['mean_top_day_share']:.3f} ± {r['std_top_day_share']:.3f}")
        print(f"  dias_extremos: {r['mean_dias_extremos']:.2f} ± {r['std_dias_extremos']:.2f}")

    print(f"\nMann-Whitney U (top_day_share): U={resultados['test_top_day_share']['U']:.1f}, "
          f"p={resultados['test_top_day_share']['p']:.4f}")
    print(f"Mann-Whitney U (dias_extremos):  U={resultados['test_dias_extremos']['U']:.1f}, "
          f"p={resultados['test_dias_extremos']['p']:.4f}")

    return resultados


# ══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Control & supplementary experiments')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['amplitude', 'reuploading', 'parity', 'lambda-sweep',
                                 'pr-auc', 'ACP-diagnostics'],
                        help='Which experiment to run')
    parser.add_argument('--ckpt-dir', type=str, default='./resultados/checkpoints',
                        help='Checkpoint directory')
    args = parser.parse_args()

    if args.experiment in ('amplitude', 'reuploading', 'parity', 'lambda-sweep'):
        import os
        ckpt_data = os.path.join(args.ckpt_dir, '01_data.pt')
        loaded = torch.load(ckpt_data, weights_only=False)
        ds = loaded[1] if isinstance(loaded, tuple) else loaded

    if args.experiment == 'amplitude':
        run_amplitude_experiment(ds)

    elif args.experiment == 'reuploading':
        run_reuploading_experiment(ds)

    elif args.experiment == 'parity':
        # Reference QNN AUCs from the paper's 15-seed run
        qnn_15_seed_aucs = [
            0.6717, 0.6639, 0.6636, 0.6448, 0.6545,
            0.6808, 0.6483, 0.6731, 0.6596, 0.6277,
            0.6761, 0.6530, 0.6369, 0.6544, 0.6750
        ]
        run_parity_experiment(ds, qnn_15_seed_aucs, LstmLinear4D, "LstmLinear4D (20 params)")
        run_parity_experiment(ds, qnn_15_seed_aucs, LstmLeakyMLP4D, "LstmLeakyMLP4D (28 params)")

    elif args.experiment == 'lambda-sweep':
        results = run_lambda_sensitivity(ds)
        torch.save(results, os.path.join(args.ckpt_dir, 'lambda_sensibilidad_resultados.pt'))

    elif args.experiment == 'pr-auc':
        calculate_pr_auc(args.ckpt_dir)

    elif args.experiment == 'ACP-diagnostics':
        ckpt_data = os.path.join(args.ckpt_dir, '01_data.pt')
        run_ACP_diagnostics(ckpt_data)
