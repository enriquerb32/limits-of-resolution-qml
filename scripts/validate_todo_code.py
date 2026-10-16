"""
scripts/validate_todo_code.py
Validates the syntax of all Colab code blocks for todo_experiments.md.
"""
import ast

def check(code_str, name):
    try:
        ast.parse(code_str)
        print(f"[{name}] Syntax VALID")
    except SyntaxError as e:
        print(f"[{name}] SYNTAX ERROR: {e}")
        raise

# Block 1: Setup & Data Loading
b0 = '''
import os, sys, json, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, confusion_matrix
import pennylane as qml

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
'''
check(b0, "Block 0: Imports")

# Block 1: Priority 1 Code
b1 = '''
def run_priority1_train_only_rerun(ckpt_path="01_data.pt", n_seeds=15):
    data = torch.load(ckpt_path, map_location="cpu")
    agg = data[0] if isinstance(data, (tuple, list)) else data["agg"]
    ds = data[1] if isinstance(data, (tuple, list)) else data["ds"]
    
    # 1. Anomaly standardization strictly on training partition (year <= 1996)
    train_mask = (agg.index.year <= 1996) if hasattr(agg.index, "year") else (agg["fecha"].dt.year <= 1996)
    city_cols = [c for c in agg.columns if c.startswith("ciudad_")]
    
    # Recompute station-specific monthly climatology on train only
    agg_strict = agg.copy()
    for city in city_cols:
        c_mask = (agg[city] == 1.0)
        c_tr_mask = c_mask & train_mask
        m_mean = agg.loc[c_tr_mask].groupby(agg.loc[c_tr_mask].index.month)["t_med"].mean()
        m_std = agg.loc[c_tr_mask].groupby(agg.loc[c_tr_mask].index.month)["t_med"].std().replace(0, 1.0)
        
        # Apply strict anomalies
        months = agg.loc[c_mask].index.month
        agg_strict.loc[c_mask, "t_anomaly"] = (agg.loc[c_mask, "t_med"] - months.map(m_mean)) / months.map(m_std)
        
    print("Strict train-only anomaly standardization computed.")
'''
check(b1, "Block 1: Priority 1")

# Block 2: Priority 2 Code (MatchedClassicalLSTM)
b2 = '''
class MatchedClassicalLSTM(nn.Module):
    """
    Capacity-matched classical LSTM control (2,609 parameters).
    Matches the 2,589-parameter classical backbone of LSTM+QNN:
      - 2-layer BiLSTM (1 -> 8 hidden units per dir): 2,368 params
      - Projection Linear(51, 4): 208 params
      - LayerNorm(4): 8 params
      - Linear head Linear(4, 1): 5 params
      Total = 2,589 params (+ 20 params with LinearBottleneck = 2,609 params).
    """
    def __init__(self, in_dim=35, hidden_dim=8, bottleneck_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(hidden_dim * 2 + in_dim, bottleneck_dim), nn.Dropout(0.1))
        self.bottleneck = nn.Linear(bottleneck_dim, bottleneck_dim) # 20 params
        self.norm = nn.LayerNorm(bottleneck_dim)
        self.clf = nn.Linear(bottleneck_dim, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x[:, :12].unsqueeze(-1))
        h = torch.cat([hn[-2], hn[-1]], dim=1) # 16 dimensions
        st = self.proj(torch.cat([h, x], dim=1)) # 51 -> 4
        qi = torch.sigmoid(st) * math.pi
        out = self.bottleneck(qi)
        return self.clf(self.norm(out)).squeeze(-1)
'''
check(b2, "Block 2: Priority 2 Matched LSTM")

# Block 3: Priority 3 Code (Untrained Quantum Controls)
b3 = '''
def make_qnode_circuit(n_q=4, n_l=2):
    dev = qml.device("default.qubit", wires=n_q)
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circ(W, x):
        for l in range(n_l):
            for i in range(n_q):
                qml.RY(x[i % x.shape[0]], wires=i)
                qml.RZ(x[i % x.shape[0]], wires=i)
            qml.StronglyEntanglingLayers(W[l:l+1], wires=range(n_q))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]
    return circ

class RandomFrozenQNN(nn.Module):
    def __init__(self, in_dim=35, n_q=4, n_l=2, mode="random"):
        super().__init__()
        self.n_q = n_q
        self.lstm = nn.LSTM(1, 8, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(16 + in_dim, n_q), nn.Dropout(0.1))
        
        # Frozen parameter tensor
        if mode == "random":
            W_init = torch.empty(n_l, n_q, 3).uniform_(-math.pi, math.pi)
        elif mode == "identity":
            W_init = torch.zeros(n_l, n_q, 3)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        self.W = nn.Parameter(W_init, requires_grad=False)
        self.circ = make_qnode_circuit(n_q, n_l)
        self.norm = nn.LayerNorm(n_q)
        self.clf = nn.Linear(n_q, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x[:, :12].unsqueeze(-1))
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        st = self.proj(torch.cat([h, x], dim=1))
        qi = torch.sigmoid(st) * math.pi
        qo = torch.stack([torch.stack(self.circ(self.W, xi)) for xi in qi])
        return self.clf(self.norm(qo)).squeeze(-1)
'''
check(b3, "Block 3: Priority 3 Untrained Controls")

# Block 4: Priority 4 Export Multi-Seed Probabilities
b4 = '''
def export_multiseed_probabilities(models_dict, ds, n_seeds=15):
    results_export = {"models": {}, "metadata": {"n_test": len(ds.y_te), "seeds": list(range(n_seeds))}}
    y_te = ds.y_te.cpu().numpy() if hasattr(ds.y_te, "numpy") else np.array(ds.y_te)
    # Loop over models and seeds
    return results_export
'''
check(b4, "Block 4: Priority 4 Export Probabilities")

print("\nALL SCRIPT BLOCKS PASSED AST SYNTAX CHECK!")
