"""Writes the results of exp30 (full Fourier surrogates) and exp31 (shot noise) into main.tex. Run once."""
import sys
path = sys.argv[1] if len(sys.argv) > 1 else 'main.tex'
s = open(path, encoding='utf-8').read()
if 'Full Fourier' in s:
    sys.exit('already applied')

def rep(old, new):
    global s
    assert s.count(old) == 1, f'not unique/not found: {old[:80]}'
    s = s.replace(old, new)

# abstract
rep(r"a linear map of Fourier features of the circuit's input angles matched the trained circuit to within 0.002 AUC, as classical-surrogate arguments predict, and a classical head",
    r"a classical model on the circuit's full Fourier basis matched the trained circuit ($+0.004$ AUC) and one on its degree-1 sub-basis scored 0.013 above it, consistent with classical-surrogate arguments, and a classical head")

# contribution 4
rep(r"(linear, Fourier surrogates with the spectrum of the circuit, frozen circuits and a classical transfer head)",
    r"(linear, separable and full tensor-product Fourier surrogates with the spectrum of the circuit, frozen circuits and a classical transfer head), a finite-shot simulation of inference")

# methods: describe the full surrogates
rep(r"Because the Fourier controls lack the cross-coordinate products of the circuit, they are a strict, separable subset of its function class: if they match it, those products carry no detectable signal.",
    r"Because these two controls lack the cross-coordinate products of the circuit, they are a strict, separable subset of its function class. Two further surrogates, \texttt{Full Fourier-1} and \texttt{Full Fourier-2}, use the full tensor-product bases $\prod_{j=1}^{4}\{1,\cos k q_j,\sin k q_j\}_{k\le d}$ for $d=1$ (81 features, 2{,}917 trainable parameters in all) and $d=2$ (625 features, 5{,}093), followed by the same linear map, LayerNorm and output layer. Each $\langle Z_j\rangle$ of the circuit is a linear combination of the $d=2$ features, so \texttt{Full Fourier-2} contains every function the circuit can produce at its readout, with free coefficients where the circuit's are tied to its 24 weights (\path{exp30_fourier_full.py}).")

# controls AUC table rows
rep(r"Fourier-1 control (36) & 0.6639 & 0.0105 & $+0.0005$ \\",
    r"Fourier-1 control (36) & 0.6639 & 0.0105 & $+0.0005$ \\" + "\n" +
    r"Full Fourier-1 control (328; 81 features) & 0.6761 & 0.0058 & $+0.013$ \\" + "\n" +
    r"Full Fourier-2 control (2{,}504; 625 features) & 0.6669 & 0.0114 & $+0.0035$ \\")

# paired controls table rows (block SE and block MDE from results_exp10_fourier_full.json)
rep(r"Fourier-2 $-$ LSTM+QNN & $+0.0016$ & 0.0043 & 0.38 (0.703) & $[-0.006,+0.009]$ (0.0039) & $[-0.009,+0.013]$ & 0.012 / 0.011 \\",
    r"Fourier-2 $-$ LSTM+QNN & $+0.0016$ & 0.0043 & 0.38 (0.703) & $[-0.006,+0.009]$ (0.0039) & $[-0.009,+0.013]$ & 0.012 / 0.011 \\" + "\n" +
    r"Full Fourier-1 $-$ LSTM+QNN & $+0.0127$ & 0.0043 & 2.95 (0.003) & $[+0.007,+0.018]$ (\tbd{}) & $[+0.003,+0.023]$ & 0.012 / \tbd{} \\" + "\n" +
    r"Full Fourier-2 $-$ LSTM+QNN & $+0.0035$ & 0.0037 & 0.95 (0.348) & $[-0.003,+0.010]$ (\tbd{}) & $[-0.008,+0.015]$ & 0.010 / \tbd{} \\" + "\n" +
    r"Full Fourier-2 $-$ Fourier-2 & $+0.0019$ & 0.0037 & 0.51 (0.615) & $[-0.004,+0.008]$ (\tbd{}) & $[-0.008,+0.012]$ & 0.010 / \tbd{} \\")

# controls discussion
rep(r"This is the empirical counterpart of the classical-surrogate argument \citep{schreiber2023,landman2023}, in a stronger form: the surrogate used here is separable, without the cross-frequency terms of the circuit, and fitted end to end like the hybrid rather than derived from it.",
    r"This is the empirical counterpart of the classical-surrogate argument \citep{schreiber2023,landman2023}, in a stronger form: the surrogate used here is separable, without the cross-frequency terms of the circuit, and fitted end to end like the hybrid rather than derived from it. Adding the cross-coordinate products does not change the conclusion. \texttt{Full Fourier-2}, which contains the circuit's whole readout function class, is not distinguished from the circuit ($+0.0035$, $p=0.35$; year-block $[-0.003,+0.010]$; the cluster-robust interval includes zero for all 12 start months) nor from the separable \texttt{Fourier-2} ($+0.0019$, $p=0.62$), and its seed SD (0.011) is twice that of \texttt{Full Fourier-1} (0.006), as expected of 625 free features fitted on 2{,}273 rows. \texttt{Full Fourier-1} scores 0.013 above the trained circuit ($p=0.003$; year-block $[+0.007,+0.018]$, years$\times$seeds $[+0.003,+0.023]$, cluster-robust interval excluding zero for all 12 start months), the largest neural-model AUC of the panel (0.676, against 0.683 for Random Forest). This comparison is secondary and outside the Holm families, and $p=0.003$ would not survive a Bonferroni correction over the roughly forty secondary comparisons; but its sign agrees under every procedure, and it points in the direction the surrogate argument allows: the circuit is not limited by its spectrum, which a classical model can use at least as well, but by the way its 24 weights tie the Fourier coefficients together.")

# multiplicity list
rep(r"the comparisons of the other three hybrids (\Cref{S-tab:hyb_paired}) and of the six controls (\Cref{tab:controls})",
    r"the comparisons of the other three hybrids (\Cref{S-tab:hyb_paired}) and of the eight controls (\Cref{tab:controls}), the finite-shot simulation")

# experiment map row
rep(r"Controls and corrected transfer model & train-only & 2{,}273 (as the core) & 15 & all seeds & primary (\Cref{tab:case,tab:controls}) \\",
    r"Controls and corrected transfer model & train-only & 2{,}273 (as the core) & 15 & all seeds & primary (\Cref{tab:case,tab:controls}) \\" + "\n" +
    r"Full Fourier surrogates & train-only & 2{,}273 (as the core) & 15 & all seeds & secondary (\Cref{tab:case,tab:controls}) \\" + "\n" +
    r"Finite-shot inference & train-only & 2{,}273 (\texttt{LSTM+QNN} refitted) & 15 & 50 shot repetitions per seed & secondary (\Cref{sec:limitations}) \\")

# discussion and conclusions
rep(r"Fourier features of the four angles match the trained circuit, and a linear head",
    r"Fourier features of the four angles match the trained circuit, with or without the cross-coordinate products (a degree-1 tensor-product surrogate even scores 0.013 above it), and a linear head")
rep(r"a linear map of Fourier features of the circuit's angles matches the trained circuit, and a linear head on a frozen extractor matches the quantum head, both within the resolution of the comparison.",
    r"a classical model on the circuit's full Fourier basis matches the trained circuit and one on its degree-1 sub-basis scores 0.013 above it, and a linear head on a frozen extractor matches the quantum head. Finite-shot inference costs less than 0.002 AUC from about 1{,}000 shots per circuit evaluation.")

# protocol item 7
rep(r"one with the frequency spectrum of the circuit (a Fourier-feature surrogate),",
    r"one with the frequency spectrum of the circuit (a Fourier-feature surrogate, including the cross-coordinate products),")

# limitation: shot noise now simulated
rep(r"This term was not simulated.",
    r"We simulated it for inference (\path{exp31_shot_noise.py}): the 15 \texttt{LSTM+QNN} models were refitted (mean exact AUC 0.6654, against 0.6634 for the archived fits), and for each seed the 16-outcome distribution of the circuit was sampled jointly, 50 times per shot budget. The 15-seed mean AUC falls by 0.031, 0.013, 0.002 and 0.0003 at 32, 100, 1{,}000 and 10{,}000 shots per circuit evaluation, and the shot SD of the seed-mean AUC is 0.004, 0.002, 0.001 and 0.0003. From about 1{,}000 shots (528{,}000 shots for one evaluation of the test set) the third term is an order of magnitude below the seed SD (0.010) and the sampling SE (0.024) and does not change the resolution statements; at 32 shots the loss is the size of the MDE. Training on hardware was not simulated: with parameter-shift gradients each row and step needs 81 circuits (one forward pass, 48 for the 24 weights and 32 for the 16 encoding-gate occurrences through which the classical layers are trained), about $1.8\times10^{8}$ shots per epoch at 1{,}000 shots.")

open(path, 'w', encoding='utf-8').write(s)
print('exp30/exp31 results written into', path)
