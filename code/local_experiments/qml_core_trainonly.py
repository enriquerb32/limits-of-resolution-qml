"""
qml_core_trainonly.py — Matched 15-Seed QML Core Experiment
============================================================
Fits the 4 models of the matched core comparison on the train-only pipeline:
  * data      : trainonly_dataset.build_dataset(csv, train_only=True)
                (anomalies standardized with year <= 1996 only)
  * fitting   : every model is fitted on the SAME 2,273 training rows
                (validation 514 rows used only for early stopping)
  * models    : RF_2273        Random Forest, 300 trees, max_depth 8, same rows
                PureLSTM       standalone classical LSTM (same as project baseline)
                CTRL_LINEAR4D  classical control: same LSTM + 51->4 projection, circuit replaced by Linear(4,4) (20 params)
                LSTM_QNN       hybrid: same LSTM + 51->4 projection + 4-qubit, 2-layer circuit (24 quantum params)
  * seeds     : N_SEEDS (default 15) for every stochastic model; the 528 test probabilities of EVERY seed are stored
  * outputs   : results_qml_core_trainonly.json (resumable, rewritten after each seed) and
                results_qml_core_trainonly_probs.npz (keys '<MODEL>_probs' (S x 528) and 'y_te')

Feed the npz to exp10_paired_ci.py (--extra) for the paired intervals.
"""
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pennylane as qml

import multiprocessing
torch.set_num_threads(multiprocessing.cpu_count())
torch.set_default_dtype(torch.float64)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from trainonly_dataset import build_dataset

N_SEEDS = 15
RAW_CSV = "datos_limpios_multiestacion.csv"
if not os.path.exists(RAW_CSV) and os.path.exists(os.path.join(SCRIPT_DIR, RAW_CSV)):
    RAW_CSV = os.path.join(SCRIPT_DIR, RAW_CSV)

OUT_JSON = "results_qml_core_trainonly.json"
OUT_NPZ = "results_qml_core_trainonly_probs.npz"
REF_NPZ = "results_rerun_train_only_canonical_probs.npz"
if not os.path.exists(REF_NPZ) and os.path.exists(os.path.join(SCRIPT_DIR, REF_NPZ)):
    REF_NPZ = os.path.join(SCRIPT_DIR, REF_NPZ)

STOP_ON_IMPLAUSIBLE = True   # stop if a seed-0 AUC is outside 0.55-0.78 (set False to continue anyway)
MODELS = ["RF_2273", "PureLSTM", "CTRL_LINEAR4D", "LSTM_QNN"]   # cheap models first, QNN last
device = torch.device("cpu")


# ------------------------------------------------------------------ model code
class PureLSTM(nn.Module):
    """
    Classical PureLSTM baseline exactly as implemented in code/modelos_clasicos.py:
    2-layer BiLSTM (16 units/dir), BatchNorm1d(32 + in_dim), GELU head with LayerNorm.
    """
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
        _, (hn, _) = self.lstm(x[:, :12].unsqueeze(-1))
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        combined = torch.cat([h, x], dim=1)
        try:
            out_bn = self.bn(combined)
        except RuntimeError:
            out_bn = self.bn(combined.float()).to(combined.dtype)
        return self.fc(out_bn).squeeze(-1)


def _loop(model, dl, X_va, y_va, epochs=25, patience=6, lr=5e-3):
    """Early-stopping AdamW loop matching modelos_utils.py."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    y_tr = dl.dataset.tensors[1]
    n_neg = (y_tr == 0).sum().float()
    n_pos = (y_tr == 1).sum().float()
    pos_w = torch.tensor([n_neg / (n_pos + 1e-5)], device=X_va.device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    best_loss = float('inf')
    best_state = None
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        for bX, by in dl:
            opt.zero_grad()
            out = model(bX)
            logits = out[0] if isinstance(out, tuple) else out
            loss = crit(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vout = model(X_va)
            v_logits = vout[0] if isinstance(vout, tuple) else vout
            vl = crit(v_logits, y_va).item()

        sch.step(vl)
        if vl < best_loss - 1e-4:
            best_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def make_qnode_circuit(n_q=4, n_l=2):
    dev = qml.device('default.qubit', wires=n_q)
    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circ(W, x):
        is_batched = (x.ndim == 2)
        for l in range(n_l):
            for i in range(n_q):
                val = x[:, i % x.shape[1]] if is_batched else x[i % x.shape[0]]
                qml.RY(val, wires=i)
                qml.RZ(val, wires=i)
            qml.StronglyEntanglingLayers(W[l:l+1], wires=range(n_q))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]
    return circ


class HybridQNNControl(nn.Module):
    """
    Modular Hybrid QNN supporting:
      - mode='trained': standard variational gradient updates on W
      - mode='frozen_random': W ~ Uniform(-pi, pi) frozen (0 quantum trainable params)
      - mode='frozen_identity': W = 0 frozen (0 quantum trainable params)
    """
    def __init__(self, in_dim=35, n_q=4, n_l=2, mode="trained"):
        super().__init__()
        self.n_q = n_q
        self.n_l = n_l
        self.mode = mode
        
        self.lstm = nn.LSTM(1, 8, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(16 + in_dim, n_q), nn.Dropout(0.1))
        
        if mode == "trained":
            self.W = nn.Parameter(torch.empty(n_l, n_q, 3).uniform_(-math.pi, math.pi), requires_grad=True)
        elif mode == "frozen_random":
            self.W = nn.Parameter(torch.empty(n_l, n_q, 3).uniform_(-math.pi, math.pi), requires_grad=False)
        elif mode == "frozen_identity":
            self.W = nn.Parameter(torch.zeros(n_l, n_q, 3), requires_grad=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        self.circ = make_qnode_circuit(n_q, n_l)
        self.norm = nn.LayerNorm(n_q)
        self.clf = nn.Linear(n_q, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x[:, :12].unsqueeze(-1))
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        st = self.proj(torch.cat([h, x], dim=1))
        qi = torch.sigmoid(st) * math.pi
        try:
            res = self.circ(self.W, qi)
            qo = torch.stack(res, dim=-1) if isinstance(res, (list, tuple)) else res
        except Exception:
            qo = torch.stack([torch.stack(self.circ(self.W, xi)) for xi in qi])
        qo = qo.to(qi.dtype)
        return self.clf(self.norm(qo)).squeeze(-1)


class LinearControl4D(nn.Module):
    """Same classical backbone as HybridQNNControl; the circuit is replaced by a linear 4->4 map (20 parameters)."""
    def __init__(self, in_dim=35, n_q=4):
        super().__init__()
        self.lstm = nn.LSTM(1, 8, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(16 + in_dim, n_q), nn.Dropout(0.1))
        self.ctrl = nn.Linear(n_q, n_q)
        self.norm = nn.LayerNorm(n_q)
        self.clf = nn.Linear(n_q, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x[:, :12].unsqueeze(-1))
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        st = self.proj(torch.cat([h, x], dim=1))
        qi = torch.sigmoid(st) * math.pi
        return self.clf(self.norm(self.ctrl(qi))).squeeze(-1)


if __name__ == "__main__":
    # ------------------------------------------------------------------ data
    print("Building TRAIN-ONLY dataset from", RAW_CSV)
    agg, ds = build_dataset(RAW_CSV, train_only=True)
    y_te = ds.y_te.numpy().astype(int)
    print(f"rows train/val/test = {len(ds.X_tr)}/{len(ds.X_va)}/{len(ds.X_te)} | P60 = {ds.umbral:.2f} | positives test = {int(y_te.sum())}")
    assert (len(ds.X_tr), len(ds.X_va), len(ds.X_te)) == (2273, 514, 528), "split sizes differ from the paper: do not continue"
    assert abs(ds.umbral - 28.1) < 0.2, "P60 differs from 28.1"
    ref = np.load(REF_NPZ) if os.path.exists(REF_NPZ) else None
    if ref is not None:
        assert (ref["y_te"].astype(int) == y_te).all(), "test labels differ from the earlier train-only rerun"
        print("test labels identical to the earlier train-only rerun: OK")

    def seed_all(s):
        torch.manual_seed(s)
        np.random.seed(s)

    def loader():
        return DataLoader(TensorDataset(ds.X_tr.to(device).double(), ds.y_tr.to(device).double()), batch_size=16, shuffle=True, drop_last=True)

    def predict(m):
        m.eval()
        with torch.no_grad():
            return torch.sigmoid(m(ds.X_te.to(device).double())).cpu().numpy().astype(float)

    def fit_nn(model):
        m = _loop(model.to(device).double(), loader(), ds.X_va.to(device).double(), ds.y_va.to(device).double(), epochs=25, patience=6, lr=5e-3)
        return predict(m)

    def t_rf(seed):
        imp = SimpleImputer(strategy="mean")
        X = imp.fit_transform(ds.X_sk)
        rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=seed, n_jobs=-1)
        rf.fit(X, ds.y_sk)
        return rf.predict_proba(imp.transform(ds.X_te.numpy()))[:, 1].astype(float)

    def t_lstm(seed):
        seed_all(seed)
        return fit_nn(PureLSTM(in_dim=ds.in_dim))

    def t_ctrl(seed):
        seed_all(seed)
        return fit_nn(LinearControl4D(in_dim=ds.in_dim))

    def t_qnn(seed):
        seed_all(seed)
        return fit_nn(HybridQNNControl(in_dim=ds.in_dim, mode="trained"))

    TRAIN = {"RF_2273": t_rf, "PureLSTM": t_lstm, "CTRL_LINEAR4D": t_ctrl, "LSTM_QNN": t_qnn}

    state = {"y_te": y_te.tolist(), "n_seeds": N_SEEDS, "models": {k: {"probs": [], "auc": [], "seconds": []} for k in MODELS}}
    if os.path.exists(OUT_JSON):
        old = json.load(open(OUT_JSON, encoding="utf-8"))
        if old.get("y_te") == state["y_te"]:
            for k in MODELS:
                if k in old["models"]:
                    state["models"][k].update(old["models"][k])
            print("[resume]", {k: len(state["models"][k]["probs"]) for k in MODELS})

    def save():
        json.dump(state, open(OUT_JSON, "w", encoding="utf-8"))
        out = {"y_te": y_te}
        for k in MODELS:
            if state["models"][k]["probs"]:
                out[f"{k}_probs"] = np.asarray(state["models"][k]["probs"])
        np.savez_compressed(OUT_NPZ, **out)

    def run(name, seed):
        t0 = time.time()
        p = TRAIN[name](seed)
        assert p.shape == (528,) and np.isfinite(p).all() and 0 <= p.min() and p.max() <= 1, name
        return p, float(roc_auc_score(y_te, p)), time.time() - t0

    # ------------------------------------------------------------------ sanity gate on seed 0 of the cheap models
    for name in MODELS:
        st = state["models"][name]
        if not st["probs"]:
            p, a, s = run(name, 0)
            st["probs"].append(p.tolist())
            st["auc"].append(a)
            st["seconds"].append(s)
            save()
        print(f"gate seed 0 {name:14s} AUC {st['auc'][0]:.4f}  ({st['seconds'][0]/60:.1f} min)")
        if not (0.55 < st["auc"][0] < 0.78):
            msg = f"{name}: seed-0 AUC {st['auc'][0]:.3f} outside the plausible range 0.55-0.78 - check the pipeline before spending hours"
            if STOP_ON_IMPLAUSIBLE:
                raise SystemExit(msg)
            print("WARNING:", msg)
    if ref is not None:
        print("reference (earlier classical rerun, train-only, seed 0): LSTM %.4f | RF(2787 rows) %.4f" % (
            roc_auc_score(y_te, ref["train_only_LSTM_probs"][0]), roc_auc_score(y_te, ref["train_only_RF_probs"][0])))

    # ------------------------------------------------------------------ main loop (resumable)
    for name in MODELS:
        st = state["models"][name]
        for seed in range(len(st["probs"]), N_SEEDS):
            p, a, s = run(name, seed)
            st["probs"].append(p.tolist())
            st["auc"].append(a)
            st["seconds"].append(s)
            save()
            print(f"{name:14s} seed {seed:2d} AUC {a:.4f}  {s/60:.1f} min")
    for name in MODELS:
        a = np.array(state["models"][name]["auc"])
        print(f"{name:14s} n={len(a):2d} mean AUC {a.mean():.4f} sd {a.std(ddof=1) if len(a)>1 else float('nan'):.4f}")
    print("saved", OUT_JSON, "and", OUT_NPZ)
