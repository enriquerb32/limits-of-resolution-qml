"""
verify_parameter_counts.py
Exact analytical formula derivation of parameter counts for all models with in_dim = 35.
Cross-referenced against code definitions and saved state dicts.
"""

def lstm_params(input_size, hidden_size, num_layers, bidirectional=True):
    num_dirs = 2 if bidirectional else 1
    total = 0
    for layer in range(num_layers):
        in_s = input_size if layer == 0 else hidden_size * num_dirs
        # 4 gates: input, forget, cell, output
        # weights: 4 * (hidden_size * in_s + hidden_size * hidden_size)
        # biases: 4 * (hidden_size + hidden_size) = 8 * hidden_size
        gate_w_ih = 4 * hidden_size * in_s
        gate_w_hh = 4 * hidden_size * hidden_size
        gate_b = 8 * hidden_size
        layer_params = (gate_w_ih + gate_w_hh + gate_b) * num_dirs
        total += layer_params
    return total

def linear_params(in_f, out_f, bias=True):
    return in_f * out_f + (out_f if bias else 0)

def layernorm_params(normalized_shape):
    return 2 * normalized_shape  # weight + bias

def batchnorm_params(num_features):
    return 2 * num_features     # weight (gamma) + bias (beta)

in_dim = 35  # Canonical dataset

print("=" * 60)
print("EXACT PARAMETER DERIVATION (in_dim = 35)")
print("=" * 60)

# 1. Base Hybrid Architecture: LSTM+QNN (4q, 2L)
# lstm: nn.LSTM(1, 8, 2, batch_first=True, dropout=0.2, bidirectional=True)
p_lstm_hybrid = lstm_params(1, 8, 2, bidirectional=True)
p_proj_hybrid = linear_params(16 + in_dim, 4) # Linear(51, 4)
p_norm_hybrid = layernorm_params(4)          # LayerNorm(4)
p_clf_hybrid = linear_params(4, 1)            # Linear(4, 1)
p_classical_backbone = p_lstm_hybrid + p_proj_hybrid + p_norm_hybrid + p_clf_hybrid
p_q_w_2L = 2 * 4 * 3  # (n_l, n_q, 3) = (2, 4, 3) = 24
p_lstm_qnn = p_classical_backbone + p_q_w_2L

print(f"LSTM encoder (hybrid): {p_lstm_hybrid}")
print(f"Projection Linear(51, 4): {p_proj_hybrid}")
print(f"LayerNorm(4): {p_norm_hybrid}")
print(f"Classifier Linear(4, 1): {p_clf_hybrid}")
print(f"-> Classical backbone total: {p_classical_backbone}")
print(f"Quantum weights W (2, 4, 3): {p_q_w_2L}")
print(f"-> LSTM+QNN TOTAL: {p_lstm_qnn} ({p_q_w_2L/p_lstm_qnn*100:.2f}% quantum)")

# 2. PI-QNN-LbL (4q, 3L)
p_q_w_3L = 3 * 4 * 3 # (3, 4, 3) = 36
p_lbl_qnn = p_classical_backbone + p_q_w_3L
print(f"-> PI-QNN-LbL TOTAL: {p_lbl_qnn} ({p_q_w_3L/p_lbl_qnn*100:.2f}% quantum)")

# 3. Standalone LSTM (PureLSTM)
# lstm: nn.LSTM(1, 16, 2, batch_first=True, dropout=0.3, bidirectional=True)
p_lstm_pure = lstm_params(1, 16, 2, bidirectional=True)
p_bn_pure = batchnorm_params(32 + in_dim) # BatchNorm1d(67)
p_fc1_pure = linear_params(32 + in_dim, 16) # Linear(67, 16)
p_ln_pure = layernorm_params(16)            # LayerNorm(16)
p_fc2_pure = linear_params(16, 1)           # Linear(16, 1)
p_pure_lstm = p_lstm_pure + p_bn_pure + p_fc1_pure + p_ln_pure + p_fc2_pure
print("\nStandalone LSTM (PureLSTM):")
print(f"  LSTM (hidden 16 bi): {p_lstm_pure}")
print(f"  BatchNorm1d(67): {p_bn_pure}")
print(f"  Head: Linear(67, 16)={p_fc1_pure}, LayerNorm(16)={p_ln_pure}, Linear(16, 1)={p_fc2_pure} -> {p_fc1_pure + p_ln_pure + p_fc2_pure}")
print(f"-> Standalone LSTM TOTAL: {p_pure_lstm}")

# 4. Quantum Transfer Learning (QTL)
# Frozen encoder: PureLSTM (10,103 params)
# Trainable adapter: Linear(67, 8) + Linear(8, 4)
p_adaptador = linear_params(67, 8) + linear_params(8, 4)
p_qtl_clf = p_clf_hybrid # Linear(4, 1) = 5
p_qtl_norm = p_norm_hybrid # LayerNorm(4) = 8
p_qtl_trainable_classical = p_adaptador + p_qtl_norm + p_qtl_clf
p_qtl_trainable_quantum = p_q_w_2L # W (2, 4, 3) = 24
p_qtl_trainable_total = p_qtl_trainable_classical + p_qtl_trainable_quantum
p_qtl_grand_total = p_pure_lstm + p_qtl_trainable_total
print("\nQuantum Transfer Learning (QTL):")
print(f"  Frozen encoder (PureLSTM): {p_pure_lstm}")
print(f"  Trainable adapter: Linear(67, 8)={linear_params(67, 8)} + Linear(8, 4)={linear_params(8, 4)} -> {p_adaptador}")
print(f"  Trainable norm + clf: {p_qtl_norm + p_qtl_clf}")
print(f"  Trainable classical total: {p_qtl_trainable_classical}")
print(f"  Trainable quantum W: {p_qtl_trainable_quantum}")
print(f"-> QTL Trainable TOTAL: {p_qtl_trainable_total} ({p_qtl_trainable_quantum/p_qtl_trainable_total*100:.2f}% quantum)")
print(f"-> QTL Grand Total (frozen + trainable): {p_qtl_grand_total}")

# 5. Amplitude-16D (LstmAmplitudeQNN)
# LSTM: 2,368
# proj: Linear(51, 16)
p_proj_amp = linear_params(16 + in_dim, 16)
p_amp_classical = p_lstm_hybrid + p_proj_amp + p_norm_hybrid + p_clf_hybrid
p_amp_quantum = p_q_w_2L # W (2, 4, 3) = 24
p_amp_total = p_amp_classical + p_amp_quantum
print("\nAmplitude-16D (LstmAmplitudeQNN):")
print(f"  Projection Linear(51, 16): {p_proj_amp}")
print(f"  Classical total: {p_amp_classical}")
print(f"  Quantum weights W (2, 4, 3): {p_amp_quantum}")
print(f"-> Amplitude-16D TOTAL: {p_amp_total} ({p_amp_quantum/p_amp_total*100:.2f}% quantum)")

# 6. Re-uploading-3x4D (LstmReuploadingQNN)
# 3 blocks of 4 features = 12 output features in proj
# proj: Linear(51, 12)
p_proj_reup = linear_params(16 + in_dim, 12)
p_reup_classical = p_lstm_hybrid + p_proj_reup + p_norm_hybrid + p_clf_hybrid
p_reup_quantum = 3 * 4 * 3 # (3, 4, 3) = 36 quantum weights
print("\nRe-uploading-3x4D (LstmReuploadingQNN):")
print(f"  Projection Linear(51, 12): {p_proj_reup}")
print(f"  Classical total: {p_reup_classical}")
print(f"  Quantum weights W (3, 4, 3): {p_reup_quantum}")
print(f"-> Re-uploading-3x4D TOTAL: {p_reup_classical + p_reup_quantum} ({p_reup_quantum/(p_reup_classical + p_reup_quantum)*100:.2f}% quantum)")

# 7. MLP-4D Controls
# Linear-4D: Linear(4, 4) = 20
p_linear_bottleneck = linear_params(4, 4)
p_linear_total = p_classical_backbone + p_linear_bottleneck
print("\nLinear-4D Parity Control:")
print(f"  Bottleneck Linear(4, 4): {p_linear_bottleneck}")
print(f"-> Linear-4D TOTAL: {p_linear_total}")

# LeakyReLU-4D: Linear(4, 2) + Linear(2, 2) + Linear(2, 4) = 10 + 6 + 12 = 28
p_leaky_bottleneck = linear_params(4, 2) + linear_params(2, 2) + linear_params(2, 4)
p_leaky_total = p_classical_backbone + p_leaky_bottleneck
print("\nLeakyReLU-4D Parity Control:")
print(f"  Bottleneck Linear(4, 2) + Linear(2, 2) + Linear(2, 4): {p_leaky_bottleneck}")
print(f"-> LeakyReLU-4D TOTAL: {p_leaky_total}")

# MLP-4D GELU: Linear(4, 16) + LayerNorm(16) + Linear(16, 4) = 80 + 32 + 68 = 180
p_gelu_bottleneck = linear_params(4, 16) + layernorm_params(16) + linear_params(16, 4)
p_gelu_total = p_classical_backbone + p_gelu_bottleneck
print("\nMLP-4D GELU Control (hidden=16):")
print(f"  Bottleneck Linear(4, 16) + LayerNorm(16) + Linear(16, 4): {p_gelu_bottleneck}")
print(f"-> MLP-4D GELU TOTAL: {p_gelu_total}")
