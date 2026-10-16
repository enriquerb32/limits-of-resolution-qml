"""
export_colab_wp3.py — Colab Execution Script for Parameter and Gate Accounting
==============================================================================
Run this script in Google Colab with GPU/CPU environment having torch and pennylane.
It verifies:
  1. Exact state_dict tensor shapes and parameter counts for all models:
     - LSTM+QNN (4q, 2L)
     - PI-QNN-LbL (4q, 3L)
     - Standalone LSTM (PureLSTM)
     - Quantum Transfer Learning (QTL)
     - Amplitude-16D (LstmAmplitudeQNN)
     - Re-uploading-3x4D (LstmReuploadingQNN)
     - LstmLinear4D (Linear bottleneck control)
     - LstmLeakyMLP4D (LeakyReLU bottleneck control)
     - LstmMLP4D (MLP-GELU bottleneck control)
  2. Exact gate counts and depth via PennyLane QuantumTape decomposition
     into native gates {RY, RZ, RX, CNOT}.
  3. Exports `results_wp3_accounting.json` for repository verification.
"""

import json
import math
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

IN_DIM = 35

def count_params(module):
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total

# ==============================================================================
# 1. Model Definitions (exact copies from repo)
# ==============================================================================

# A. Base QNode
def _make_qnode(n_q=4, n_l=2):
    dev = qml.device('default.qubit', wires=n_q)
    @qml.qnode(dev, interface='torch')
    def circ(W, x):
        for l in range(n_l):
            for i in range(n_q):
                qml.RY(x[i % x.shape[0]], wires=i)
                qml.RZ(x[i % x.shape[0]], wires=i)
            qml.StronglyEntanglingLayers(W[l:l+1], wires=range(n_q))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]
    return circ

# B. LstmQNN
class LstmQNN(nn.Module):
    def __init__(self, in_dim=IN_DIM, n_q=4, n_l=2):
        super().__init__()
        self.lstm = nn.LSTM(1, 8, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(16 + in_dim, n_q), nn.Dropout(0.1))
        self.W = nn.Parameter(torch.empty(n_l, n_q, 3).uniform_(-math.pi, math.pi))
        self.norm = nn.LayerNorm(n_q)
        self.clf = nn.Linear(n_q, 1)

# C. PureLSTM
class PureLSTM(nn.Module):
    def __init__(self, in_dim=IN_DIM):
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

# D. Quantum Transfer Learning (QTL)
class QuantumTransferLearning(nn.Module):
    def __init__(self, encoder_clasico, in_embed=67, n_q=4, n_l=2):
        super().__init__()
        self.encoder = encoder_clasico
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.adaptador = nn.Sequential(
            nn.Linear(in_embed, n_q * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(n_q * 2, n_q),
            nn.Sigmoid()
        )
        self.W = nn.Parameter(torch.empty(n_l, n_q, 3).uniform_(-0.1, 0.1))
        self.norm = nn.LayerNorm(n_q)
        self.clf = nn.Linear(n_q, 1)

# E. Amplitude Embedding QNN
class LstmAmplitudeQNN(nn.Module):
    def __init__(self, in_dim=IN_DIM, n_q=4, n_l=2):
        super().__init__()
        self.lstm = nn.LSTM(1, 8, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(16 + in_dim, 16), nn.GELU(), nn.Dropout(0.1))
        self.W = nn.Parameter(torch.empty(n_l, n_q, 3).uniform_(-math.pi, math.pi))
        self.norm = nn.LayerNorm(n_q)
        self.clf = nn.Linear(n_q, 1)

# F. Re-uploading QNN (3x4D)
class LstmReuploadingQNN(nn.Module):
    def __init__(self, in_dim=IN_DIM, n_bloques=3, n_q=4):
        super().__init__()
        self.lstm = nn.LSTM(1, 8, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(16 + in_dim, n_bloques * n_q), nn.GELU(), nn.Dropout(0.1))
        self.W = nn.Parameter(torch.empty(n_bloques, n_q, 3).uniform_(-math.pi, math.pi))
        self.norm = nn.LayerNorm(n_q)
        self.clf = nn.Linear(n_q, 1)

# G. Controls
class LstmLinear4D(nn.Module):
    def __init__(self, in_dim=IN_DIM):
        super().__init__()
        self.lstm = nn.LSTM(1, 8, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(16 + in_dim, 4), nn.Dropout(0.1))
        self.bottleneck = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.clf = nn.Linear(4, 1)

class LstmLeakyMLP4D(nn.Module):
    def __init__(self, in_dim=IN_DIM):
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
        self.norm = nn.LayerNorm(4)
        self.clf = nn.Linear(4, 1)

class LstmMLP4D(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(1, 8, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(16 + in_dim, 4), nn.Dropout(0.1))
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 4)
        )
        self.norm = nn.LayerNorm(4)
        self.clf = nn.Linear(4, 1)

# ==============================================================================
# 2. Gate Decomposition Analysis
# ==============================================================================

def analyze_tape(tape):
    gate_counts = {}
    for op in tape.operations:
        name = op.name
        gate_counts[name] = gate_counts.get(name, 0) + 1
    total_gates = sum(gate_counts.values())
    cnot_count = gate_counts.get('CNOT', 0)
    single_qubit = total_gates - cnot_count
    return {
        'total_gates': total_gates,
        'single_qubit': single_qubit,
        'cnots': cnot_count,
        'breakdown': gate_counts
    }

def run_accounting():
    print("=" * 70)
    print("PARAM AND GATE ACCOUNTING VERIFICATION")
    print("=" * 70)

    models = {
        'LSTM+QNN (4q, 2L)': LstmQNN(n_q=4, n_l=2),
        'PI-QNN-LbL (4q, 3L)': LstmQNN(n_q=4, n_l=3),
        'PureLSTM (Standalone)': PureLSTM(),
        'Amplitude-16D': LstmAmplitudeQNN(n_q=4, n_l=2),
        'Re-uploading-3x4D': LstmReuploadingQNN(n_bloques=3, n_q=4),
        'Linear-4D': LstmLinear4D(),
        'LeakyReLU-4D': LstmLeakyMLP4D(),
        'MLP-4D GELU': LstmMLP4D(hidden=16)
    }

    pure_lstm = PureLSTM()
    qtl = QuantumTransferLearning(pure_lstm, in_embed=67, n_q=4, n_l=2)
    models['Quantum Transfer Learning (QTL)'] = qtl

    accounting_results = {}

    for name, m in models.items():
        tr, tot = count_params(m)
        qw = m.W.numel() if hasattr(m, 'W') else 0
        cl_tr = tr - qw
        print(f"\n{name}:")
        print(f"  Total params: {tot:,}")
        print(f"  Trainable:    {tr:,} (Classical: {cl_tr:,}, Quantum: {qw})")
        if tot > 0:
            print(f"  Quantum %:    {qw / tot * 100:.2f}% (of total) | {qw / tr * 100:.2f}% (of trainable)")

        param_dict = {k: list(v.shape) for k, v in m.named_parameters()}
        accounting_results[name] = {
            'total_params': tot,
            'trainable_params': tr,
            'classical_trainable': cl_tr,
            'quantum_params': qw,
            'shapes': param_dict
        }

    # Circuit decomposition
    print("\n" + "=" * 70)
    print("PENNYLANE QUANTUM TAPE DECOMPOSITION")
    print("=" * 70)

    # 1. Base QNode 2L
    w2 = torch.zeros(2, 4, 3)
    x4 = torch.zeros(4)
    with qml.tape.QuantumTape() as tape_base:
        for l in range(2):
            for i in range(4):
                qml.RY(x4[i], wires=i)
                qml.RZ(x4[i], wires=i)
            qml.StronglyEntanglingLayers(w2[l:l+1], wires=range(4))
    
    # Native gate decomposition
    tape_base_decomp = tape_base.expand(depth=2, stop_at=lambda op: op.name in ['RY', 'RZ', 'RX', 'CNOT', 'PauliZ'])
    res_base = analyze_tape(tape_base_decomp)
    print("Base Circuit (4q, 2L - LSTM+QNN):", res_base)

    # 2. Base QNode 3L (LbL)
    w3 = torch.zeros(3, 4, 3)
    with qml.tape.QuantumTape() as tape_lbl:
        for l in range(3):
            for i in range(4):
                qml.RY(x4[i], wires=i)
                qml.RZ(x4[i], wires=i)
            qml.StronglyEntanglingLayers(w3[l:l+1], wires=range(4))
    tape_lbl_decomp = tape_lbl.expand(depth=2, stop_at=lambda op: op.name in ['RY', 'RZ', 'RX', 'CNOT', 'PauliZ'])
    res_lbl = analyze_tape(tape_lbl_decomp)
    print("LbL Circuit (4q, 3L):", res_lbl)

    # 3. QTL Circuit (AngleEmbedding + StronglyEntanglingLayers)
    with qml.tape.QuantumTape() as tape_qtl:
        qml.AngleEmbedding(x4 * math.pi, wires=range(4), rotation='Y')
        qml.StronglyEntanglingLayers(w2, wires=range(4))
    tape_qtl_decomp = tape_qtl.expand(depth=2, stop_at=lambda op: op.name in ['RY', 'RZ', 'RX', 'CNOT', 'PauliZ'])
    res_qtl = analyze_tape(tape_qtl_decomp)
    print("QTL Circuit (4q, 2L AngleEmbedding):", res_qtl)

    # 4. Re-uploading Circuit (3 blocks)
    x12 = torch.zeros(12)
    with qml.tape.QuantumTape() as tape_reup:
        for b in range(3):
            for i in range(4):
                qml.RY(x12[b*4 + i], wires=i)
                qml.RZ(x12[b*4 + i], wires=i)
            qml.StronglyEntanglingLayers(w3[b:b+1], wires=range(4))
    tape_reup_decomp = tape_reup.expand(depth=2, stop_at=lambda op: op.name in ['RY', 'RZ', 'RX', 'CNOT', 'PauliZ'])
    res_reup = analyze_tape(tape_reup_decomp)
    print("Re-uploading Circuit (3x4D):", res_reup)

    # 5. Amplitude Embedding (4q, 2L)
    x16 = torch.ones(16) / 4.0
    with qml.tape.QuantumTape() as tape_amp:
        qml.AmplitudeEmbedding(x16, wires=range(4), normalize=False)
        qml.StronglyEntanglingLayers(w2, wires=range(4))
    tape_amp_decomp = tape_amp.expand(depth=4, stop_at=lambda op: op.name in ['RY', 'RZ', 'RX', 'CNOT', 'PauliZ'])
    res_amp = analyze_tape(tape_amp_decomp)
    print("Amplitude Circuit (4q, 2L):", res_amp)

    export_payload = {
        'parameters': accounting_results,
        'circuits': {
            'LSTM+QNN (4q, 2L)': res_base,
            'PI-QNN-LbL (4q, 3L)': res_lbl,
            'QTL (4q, 2L)': res_qtl,
            'Re-uploading (3x4D)': res_reup,
            'Amplitude (4q, 2L)': res_amp
        }
    }

    with open('results_wp3_accounting.json', 'w') as f:
        json.dump(export_payload, f, indent=2)
    print("\nSaved accounting results to results_wp3_accounting.json")

if __name__ == '__main__':
    run_accounting()
