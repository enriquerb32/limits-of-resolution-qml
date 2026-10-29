# Informe de revisión — *Limits of Resolution in Quantum Machine Learning: A Statistical Noise Budget for Small Benchmarks*

**Revisor:** Reviewer 2 (Física / QML / Climatología)
**Archivos revisados:** `paper/main.tex` (680 líneas), `paper/supp.tex` (1404 líneas)
**Fecha:** 2026-10-26
**Recomendación:** *Minor-to-major revision.* El trabajo es metodológicamente honesto y está muy por encima de la media del campo en cuanto a transparencia inferencial. Sin embargo, hay (a) varias inconsistencias internas entre main y supp, (b) algunos argumentos estadísticos y físicos que deben corregirse o matizarse, y (c) un punto sustantivo, el de la escala de la "seed share" en las comparaciones híbridas, que matiza el mensaje central del artículo precisamente para QML.

> **Convenciones.** `M:L` = `main.tex`, línea L; `S:L` = `supp.tex`, línea L. Los cálculos marcados **[cálculo del revisor]** se han hecho a partir de valores publicados en el propio manuscrito; no se ha modificado ningún número del artículo. Las frases propuestas en inglés son sugerencias de redacción, no cambios de resultados.

---

## 0. Resumen ejecutivo: los 12 puntos prioritarios

| # | Tipo | Ubicación | Problema |
|---|------|-----------|----------|
| 1 | Rigor / QML | M:82, M:202, M:463 | La "seed share" (2–7 %) solo se da en la escala de *un AUC* y para modelos clásicos. En la escala de *diferencias pareadas* y para el híbrido (SD = 0.0152), la varianza de semilla es comparable a la de muestreo. Esto matiza la tesis central precisamente para QML (§1.1). |
| 2 | Consistencia | M:138 vs S:224 | M dice que la diferencia RF 2,273 frente a 2,787 filas (0.005) está "dentro de la varianza de semilla"; con 15 semillas son ≈4 errores estándar de semilla (§1.2). S dice lo contrario ("consistent with a training-size effect"). |
| 3 | Consistencia | M:455 | La diferencia LSTM 0.6630 (core) frente a 0.6682 (rerun) se atribuye a "shuffle and initialization variance", pero el supp documenta una configuración distinta (dropout 0.3 frente a 0.2) (§2). |
| 4 | Consistencia | M:159 (Tab. expmap) vs S:75 | La tabla indica que el LSTM+QNN histórico tiene probabilidades de todas las semillas; el supp dice que solo de la semilla 0. |
| 5 | Estadística | M:498 | "negative inter-cluster correlation across years" es conceptualmente incorrecto: el bootstrap por bloques supone independencia *entre* clusters. Debe decir *within-cluster* (§1.3). |
| 6 | Estadística | M:172, §3.6 | DeLong y la simulación se dirigen a estimandos distintos (condicional en los años observados frente a superpoblación de años nuevos). "Anticonservative" debe matizarse (§1.3). |
| 7 | Climatología | S:1038–1047 | El signo del prior "Clausius–Clapeyron" (penalizar lluvia con anomalía fría) es físicamente discutible en un régimen DANA (§1.5). |
| 8 | Climatología | M:109 | "humid Barcelona/Valencia": Valencia es semiárida (BSk). Köppen de Alicante/San Javier es BSh, no BSk (§1.5). |
| 9 | Consistencia | M:392 vs S:716 | "the hybrid models have a high ECE (0.112–0.149)". El QKSVM, listado como híbrido en M:136, tiene ECE 0.051 y BSS −0.027. El rango incluye además al LSTM clásico. |
| 10 | Consistencia | S:548 vs S:676 | La climatología binarizada a 0.5 da Acc 64.58 % y F1 0.522 en el texto, pero 61.2 % y 0.547 en la Tabla S (mismo umbral). |
| 11 | Rigor | S:1097 | "…which must be checked against the feature-construction code": es una tarea pendiente dentro del texto. Debe resolverse antes del envío. |
| 12 | LaTeX | M:243–282 | Las figuras `fig:paired_ci` y `fig:sim` nunca se citan en el texto. Hay `\bibitem` sin citar en ambos ficheros. `\affil[*]` no está enlazado a ningún autor. |

---

## 1. Revisión física, matemática y de planteamiento QML

### 1.1 La "seed share" depende de la escala elegida y el mensaje cambia para el híbrido (sustantivo)

El artículo define la seed share como σ²_seed / (σ²_seed + SE²) con SE = error estándar de **un** AUC (M:180, M:673). Con esa elección, 2–7 % para RF/XGB/LSTM es correcto (verificado: 2.2 %, 5.0 %, 6.6 %). Pero la pregunta del artículo es sobre **comparaciones**, y en la escala de la diferencia pareada el cociente cambia mucho:

**[cálculo del revisor]**, suponiendo semillas independientes entre modelos (lo respalda r ≈ 0.05 en S:758):

| Comparación (una semilla por modelo) | Var. de semilla de la diferencia | SE pareado de muestreo (Tab. S-mde) | Seed share en la escala de la diferencia |
|---|---|---|---|
| RF − XGB | 0.0035² + 0.0054² | 0.013 | ≈ 20 % |
| RF − LSTM | 0.0035² + 0.0063² | 0.014 | ≈ 21 % |
| XGB − LSTM | 0.0054² + 0.0063² | 0.018 | ≈ 18 % |
| LSTM+QNN (un AUC) | 0.0152² | 0.024 | ≈ 29 % |
| Linear-4D − LSTM+QNN | 0.0152² + 0.0074² (SD ≈ 0.017) | no reportado para semilla única (0.0066 para medias) | probablemente ≥ 50 % |

Consecuencias:
- M:82 ("the sampling error of each AUC is much larger than the variation caused by the seed") no se cumple para el híbrido: 0.024 frente a 0.015 es un factor 1.6, no "much larger".
- M:202 calcula la seed share solo para modelos clásicos, aunque para LSTM+QNN (core, 15 semillas, probabilidades guardadas) también se puede calcular. Debería añadirse a la Tabla `tab:budget`.
- El intervalo two-way de Linear-4D − QNN (±0.016) es claramente más ancho que el year-block (±0.012). Eso ya muestra que la varianza de semilla pesa en la comparación QML clave.

**Sugerencia:** añadir una fila "seed share on the paired-difference scale" y la columna del híbrido. Matizar la tesis: *"For the classical models, test-sampling error dominates; for the hybrid, whose seed SD is two to four times larger, seed variance is a non-negligible share of the uncertainty of single-run comparisons."*

### 1.2 Tamaño de entrenamiento de RF: la conclusión de M:138 no se sigue de los datos

M:138 afirma que 0.6802 ± 0.0035 frente a 0.6851 ± 0.0026 (canónico) y 0.6846 frente a 0.6830 (train-only) demuestran que "variations of this scale (~0.002–0.005) are within optimization and seed variance rather than a systematic sample-size effect".

**[cálculo del revisor]:** SE de la diferencia de medias con 15 semillas = √(0.0035²/15 + 0.0026²/15) ≈ 0.0011, así que 0.0049 / 0.0011 ≈ 4.3. La diferencia canónica **no** está dentro de la varianza de semilla. En train-only la diferencia es +0.0016 (≈1.2 SE) y de signo opuesto. Lo defendible es: *"the direction of the training-size effect is not consistent across pipelines, and its magnitude (≤0.005) is small relative to the test-sampling SE of one AUC (≈0.024)."* Hay que eliminar "demonstrating". S:224 y S:340 ("consistent with a training-size effect") contradicen M:138, así que debe unificarse la interpretación.

### 1.3 Inferencia estadística

1. **Estimandos de DeLong frente a la simulación (M:172, M:255, M:284).** DeLong estima la varianza de muestreo *a nivel de fila condicionada a los años observados*. El objetivo de la simulación es la diferencia poblacional "for new years drawn with equal weight". Con efectos de año específicos del scorer, DeLong no está "mal calibrado" para su propio estimando condicional: responde a otra pregunta. Conviene decirlo explícitamente en §3.1 y en la lectura de la Tabla `tab:sim`; si no, el lector concluye que DeLong es incorrecto per se. Igualmente, §3.1 habla de "evaluation population": hay que definir qué superpoblación se asume en un split cronológico no estacionario (¿años intercambiables?).
2. **M:498, explicación de SE_block < SE_DeLong.** "negative inter-cluster correlation across years" es incorrecto: el bootstrap por clusters supone clusters independientes y no puede reflejar correlación entre clusters. Un SE por bloques menor que el iid surge de (a) correlación negativa *intra-cluster* (dentro del año) de los componentes de la diferencia pareada, o (b) sesgo a la baja y ruido del bootstrap con G = 19 clusters (factor ≈ √((G−1)/G) más la variabilidad del propio estimador). Propuesta: *"…can arise from negative within-year correlation of the paired-difference components or from the downward bias and instability of cluster-bootstrap variance estimates with few clusters."*
3. **Bootstrap con pocos clusters.** Con 19 clusters de tamaño desigual (10–48 filas, y 2026 parcial), el bootstrap por percentiles es conocido por ser liberal, y la propia simulación lo confirma (0.074–0.092). Sugerencia: reportar al menos una alternativa más robusta (cluster jackknife, bootstrap studentizado o wild cluster bootstrap; cf. Cameron, Gelbach & Miller, 2008). Künsch (1989) es la referencia del *moving-block* bootstrap para series estacionarias, no del bootstrap por clusters de calendario fijo; conviene añadir una referencia de cluster bootstrap (p. ej. Field & Welsh, 2007).
4. **p-valores bootstrap.** M:222 dice "no bootstrap p-values were computed", pero M:250 ("year-block bootstrap p=0.001"), S:707 (p = 0.5420) y S:718 (p = 0.0400, 0.0010, 0.9280) sí los reportan. Hay que definir cómo se calculan (¿inversión de intervalo? ¿proporción de réplicas ≤ 0?) o eliminarlos.
5. **Estimación puntual dependiente del procedimiento (S:707).** RF − climatología: ΔAUC = +0.0079 con DeLong y +0.0068 con block bootstrap, sobre los mismos datos. La estimación puntual no debe cambiar con el método de varianza. Si +0.0068 es la media de las réplicas, debe indicarse y reportarse la estimación muestral.
6. **Tercer estimando no declarado (M:286).** El "panel-matched check" usa el AUC de las *predicciones promediadas entre semillas* (ensemble, gap 0.0115), distinto de los dos estimandos de §3.1. Debe declararse como tercer estimando o justificarse.
7. **Tests pareados con emparejamiento nominal (S:735–760, S:643).** El texto reconoce que el índice de semilla "is a label, not a pairing device", y r ≈ 0.05 lo confirma. Aun así se usan Wilcoxon signed-rank, Friedman y TOST *pareados*. Lo coherente es usar las versiones no pareadas (Mann–Whitney, TOST de Welch, Kruskal–Wallis). S:541 usa además el Friedman de forma inferencial ("confers no statistically significant performance advantage (Friedman p = 0.458)"), mientras S:741 lo declara "not an inferential test".
8. **Levene (S:358–366).** Resultado contraintuitivo: σ 0.0089 frente a 0.0171 (cociente 1.92) da p = 0.191, y σ 0.0089 frente a 0.0154 (cociente 1.73) da p = 0.011. Puede deberse a valores atípicos (PI-QNN mín. 0.605), pero conviene indicar la variante (mediana/Brown–Forsythe o media), reportar también un F-test y advertir de la baja potencia con n = 15 y de la multiplicidad.
9. **Familias Holm.** La agrupación de RF − QNN con Linear − QNN y LSTM − QNN es una decisión que afecta a la conclusión principal del caso QML (p_Holm = 0.079). Debe justificarse a priori y no solo describirse.

### 1.4 Planteamiento QML

1. **Barren plateaus (M:80).** Citar McClean et al. (2018) para un circuito de 4 qubits sugiere implícitamente que los BP explican la alta varianza de semilla del híbrido. Con n = 4 los gradientes no se concentran exponencialmente. La varianza se explica mejor por mínimos locales, por el backbone clásico y por la periodicidad del encoding. Propuesta: *"…although barren plateaus are not expected at four qubits, variational circuits remain sensitive to initialization."*
2. **LayerNorm tras la lectura ⟨Z⟩ (S:158–160, S:1017).** LayerNorm sobre 4 valores elimina media y escala y deja ≈2 grados de libertad efectivos (la dirección en el subespacio de media cero). Esto tiene dos implicaciones que deben discutirse:
   - Es un cuello de botella adicional común al híbrido y al control Linear-4D. La comparación sigue siendo justa, pero la "interfaz de 4 dimensiones" es efectivamente menor.
   - Hace casi trivial la "robustez" al ruido despolarizante (S:1014–1017): una contracción uniforme de ⟨Z_i⟩ es exactamente invariante bajo LayerNorm (salvo ε). El experimento de ruido no informa sobre la sensibilidad del circuito y debería decirse así, en lugar de "notable structural robustness".
3. **Experimento de ruido sin control de shots (S:988–1024).** El modelo noiseless es statevector exacto; el ruidoso usa 1024 shots. Falta la condición "noiseless + 1024 shots" para separar el ruido de puerta del ruido de muestreo.
4. **Kernel cuántico (S:266, S:645).** Bien señalado que el kernel factoriza como ∏cos²((x_j − x'_j)/2) (correcto para angle embedding RX/RY sin entrelazamiento). Pero entonces la comparación QKSVM frente a RBF-SVM es entre dos kernels clásicos, y la diferencia puede deberse solo al *bandwidth* (escala fija de MinMax a [0, π] frente a γ ajustado). Recomendación: mencionar en el texto principal (M:136) que el QKSVM es clasicamente factorizable, y citar la literatura sobre bandwidth de kernels cuánticos (Shaydulin & Wild, 2022; Canatar et al., 2023).
5. **Recuento de puertas de Amplitude Embedding (S:1154).** 28 puertas de un qubit en total, pero dos capas SEL ya aportan 24 (Rot descompuesto en RZ·RY·RZ). Quedarían 4 rotaciones para Möttönen con 16 amplitudes, lo que es imposible para un estado genérico (hacen falta ≈15 RY). Probablemente el recuento se hizo sobre una entrada concreta en la que se podan ángulos nulos, o con otra convención. Debe verificarse y aclararse.
6. **Conjunto de puertas nativas.** S:1111 dice {RY, RZ, RX, CNOT}; S:1161 y S:1168 dicen {RY, RZ, CNOT}. El AngleEmbedding por defecto del QKSVM usa RX.
7. **Argumento del cuello de botella (S:647, S:261).** "the information bottleneck alone cannot explain the classical-over-quantum gap, because full-feature tree ensembles such as XGBoost (0.6575) achieve performance comparable to LSTM+QNN" es un *non sequitur*: que XGB sea tan malo como el híbrido no dice nada sobre si el cuello de botella limita al híbrido. La evidencia pertinente son los controles a 4D y el LSTM sin cuello de botella. Recomiendo eliminar el argumento de XGBoost.
8. **Dead ReLU (S:344–348).** La solución cambió dos cosas a la vez (LeakyReLU **y** He init), así que "the pathology was associated with the activation rather than with capacity" no queda aislado. Además, la muerte de unidades ReLU suele producirse durante el entrenamiento y no solo en la inicialización.

### 1.5 Climatología y física de las variables

1. **Clasificación climática (M:107, M:109).** Alicante y San Javier/Murcia son **BSh** (semiárido cálido) en el Atlas Climático Ibérico de AEMET; Valencia es BSk o límite Csa/BSk; Barcelona es Csa. Valencia (≈450 mm/año) no es "humid". Propuesta M:109: *"Because regimes differ between sub-humid Barcelona and the semi-arid southern stations…"*.
2. **Signo del prior Clausius–Clapeyron (S:1038–1047).** C_CC < 1 cuando T_anomaly < 0, así que se penalizan probabilidades altas de lluvia con anomalías *frías*. En el Mediterráneo occidental, los episodios de lluvia intensa de otoño asociados a DANA combinan aire frío en altura con un mar cálido, y muchos meses lluviosos tienen anomalía térmica superficial negativa (nubosidad, advección). El prior podría ir **en contra** del mecanismo que motiva el estudio (M:84). El supp ya reconoce que es heurístico. Recomiendo discutir explícitamente el signo, no solo la dimensionalidad.
3. **`iir` (S:414):** "Sunshine duration / theoretical extraterrestrial maximum" es terminología incorrecta. El cociente estándar es la insolación relativa n/N, con N = duración astronómica del día. "Extraterrestrial" se refiere a la radiación R_a (FAO-56).
4. **`ib` "Barometric instability index" (S:413):** σ_p / rango diurno no mide inestabilidad atmosférica (que depende del gradiente vertical de temperatura y humedad). Sugiero "pressure-variability-to-diurnal-range ratio".
5. **`vuln_costera` (S:407, S:727):** las cuatro estaciones son costeras, así que un "coastal proximity indicator" es constante. Además, S:727 lo trata como "spatially invariant station property", pero el glosario dice que interactúa con el déficit barométrico, lo que lo haría variable en el tiempo. Hay que definirlo con precisión.
6. **Presión (S:400, S:727):** el glosario dice "sea-level pressure", pero los valores diarios de AEMET OpenData (`presMax`/`presMin`) son presión **a nivel de estación**. Si es presión de estación, `pmed` codifica la altitud (y S:727 tendría razón en parte). Si es a nivel del mar, no. Hay que verificarlo. "altitude remain available to the models" (S:727) es incorrecto: la altitud no figura entre las 35 variables.
7. **ET₀ FAO-56 (M:107):** Penman–Monteith requiere radiación, humedad, viento y temperatura. Hay que especificar si se usó PM completo, PM con datos faltantes estimados (FAO-56 cap. 3) o Hargreaves, dado que solo hay cinco ficheros de variables diarias.
8. **"days above the daily 90th percentile" (M:107, S:307):** hay que especificar si el P90 se calcula sobre *días con lluvia* (≥ 1 mm, convención ETCCDI) o sobre todos los días; en estaciones semiáridas la diferencia es enorme.
9. **Umbral regional agrupado.** Un P60 regional único hace que el AUC agrupado premie la identificación de la estación. **[Corregido en el bloque C, `exp16_block_c.py`]:** un predictor que solo usa la tasa base de *entrenamiento* por estación obtiene un AUC agrupado de **0.599** (mi estimación inicial de 0.617 usaba por error las tasas del test). Es decir, algo más de la mitad del exceso sobre 0.5 que alcanza RF (0.68) lo explica la estación sola. Recomiendo reportar este baseline "station-only" y, como sensibilidad, un umbral por estación (percentil local), que es la convención climatológica estándar para excedencias.
10. **Climatología con más datos.** RF usa 2,787 filas (1938–2007) y la climatología solo el periodo de entrenamiento (1938–1996). Con calentamiento y no estacionariedad, una climatología ajustada en 1938–2007 o en una ventana móvil reciente sería la referencia equivalente y probablemente más exigente.
11. **Procedencia de datos (M:107):** *datosclima.es* es un repositorio de terceros. Recomiendo citar los indicativos AEMET (p. ej. 0076, 8416/8414A, 8025/8019, 7031), coordenadas, altitud y fecha de acceso, e indicar si las series combinan estaciones (Alicante ciudad/aeropuerto), lo que agrava la inhomogeneidad ya reconocida.
12. **Selección por el inner join (M:318, M:504).** 897 station-months excluidos frente a 3,363 incluidos (≈27 %) es una selección grande, concentrada en Valencia y San Javier después de 2010–2012, y su efecto "was not examined". Al menos la climatología y un RF con solo predictores de precipitación podrían evaluarse en el panel ampliado.

---

## 2. Consistencia interna main ↔ supp

| # | Ubicación | Discrepancia | Sugerencia |
|---|-----------|--------------|------------|
| C1 | M:159 vs S:75 | Tab. expmap: "seed 0 only, except LSTM+QNN" (histórico). S: "The historical LSTM+QNN run likewise has stored test probabilities for seed 0 only." | Corregir la tabla: la excepción es el experimento Core, no el histórico. |
| C2 | M:138 vs S:224 / S:340 | M: "rather than a systematic sample-size effect". S: "consistent with a training-size effect". | Unificar (ver §1.2). |
| C3 | M:455 vs S:942 | LSTM 0.6630 (core) frente a 0.6682 (rerun), atribuido a "shuffle and initialization variance". **[cálculo del revisor]** ≈2.2 SE de semilla. *Corrección tras revisar el código (bloque B):* la arquitectura es idéntica en ambos (dropout 0.3 en el LSTM y 0.2 en la cabeza, `modelos_clasicos.py` y `qml_core_trainonly.py`); lo erróneo era la Tabla S-hyperparams ("dropout=0.2") y la frase de S:942. La causa de la diferencia entre las dos ejecuciones no está identificada. | Corregido en el bloque B: tabla, S:942 y M:455 (causa "not isolated"). |
| C4 | M:392 vs M:136, S:716 | "the hybrid models have a high ECE (0.112–0.149) … BSS −0.058 to −0.140". El QKSVM (híbrido según M:136) tiene ECE 0.051 y BSS −0.027; el rango 0.112–0.149 incluye al LSTM clásico (S:716). | "the variational hybrids and the standalone LSTM have…". |
| C5 | S:716 | "all trained models yield negative Brier skill … BSS_clim ∈ [−0.003, −0.027]" y a continuación los híbridos en [−0.058, −0.140]. | "these models yield…". |
| C6 | S:548 vs S:676 | Climatología binarizada a 0.5: Acc 64.58 %, F1 0.522 (texto) frente a 61.2 %, 0.547 (tabla, "default threshold 0.5"). **[cálculo del revisor]:** la fila de la tabla es internamente coherente (FPR 0.360 ⇒ TP = 124, F1 = 0.547). | Reconciliar o explicar la diferencia de definición. |
| C7 | M:396 | "the full set of ten models is in the Supplement": el supp no tiene ninguna tabla con el Brier de los diez modelos (faltan XGB, LSTM, PI-QNN y QTL). | Añadir una columna Brier/ECE a `tab:single_run_results`. |
| C8 | S:662 vs S:336 | S:662: los deltas train-only "reflecting optimization seed variability rather than systematic leakage inflation". S:336: "they do not isolate the differences as seed noise"; el Wilcoxon pareado de XGB es significativo (p_Holm = 0.037). | Eliminar "reflecting optimization seed variability" en S:662. |
| C9 | S:718 vs S:705 | RF frente a LbL-QNN (single run): "confirms that Random Forest significantly outperforms" (bloques, p = 0.040) frente a "tracking Random Forest … no statistically significant difference" (DeLong, p = 0.2124). | Es justamente la dependencia del procedimiento que el artículo defiende: redactar en neutro ("the year-block interval excludes zero, whereas DeLong does not"). |
| C10 | S:279 vs S:415 | `choque_clausius` = T_{i−1} − T_{i−4} frente a T_{i+10} − T_{i+7}. | Unificar la indexación (relativa a la ventana i…i+11). |
| C11 | S:280 vs S:416 | `potencial_convectivo` = "Evaporation–variability interaction proxy" frente a "Convective potential". | Usar un único nombre (preferiblemente el primero). |
| C12 | S:249, S:788 vs S:872, S:829 | SD de LSTM+PI-QNN: 0.0172 frente a 0.0171; S:829 cita "[0.0089, 0.0171] in tab:multiseed", pero esa tabla muestra 0.0172. | Unificar el redondeo. |
| C13 | S:224 | "Tab. multiseed gives the 15-seed AUC of all models", pero faltan Linear-4D (TO 0.6559), LSTM+QNN (TO 0.6552), LSTM core (0.6630) y los controles 4D canónicos. | Completar o cambiar a "selected models". |
| C14 | M:222 | "Δ = +0.015 for RF − LSTM on the full 2,787-row classical baseline hierarchy": el LSTM de esa comparación usa 2,273 filas. | "…with RF fitted on 2,787 rows and the LSTM on 2,273". |
| C15 | M:109 vs S:332 | "shifts the AUC … by at most 0.004" frente a XGB +0.0041. | "by about 0.004 or less". |
| C16 | M:305 vs M:84/M:109 | Tab. coverage: proyecto P60 = 28.0; en el resto del texto 28.1. Se explica en M:316, pero después de la tabla. | Nota en el caption de la tabla. |
| C17 | M:250 | "a preprocessing detail that moves the AUC by less than 0.005": por modelo es cierto, pero el *gap* RF − XGB pasa de 0.0276 a 0.021 (≈0.007). | Mencionar el cambio del gap, que es lo relevante para el % de pares significativos. |
| C18 | M:506 vs S:981 | "Hyperparameters were set by validation-based manual grids" frente a "manually specified; a reproducible tuning history is unavailable". | Unificar. |
| C19 | S:101 | "an approximate version of this check is given in app:multiseed": esa sección contiene la matriz LbL frente a QNN (otras arquitecturas), no un control de la misma arquitectura. | Corregir la referencia o añadir el control. |
| C20 | S:263 | "(624 and 832 weights, adding 416 and 624 parameters)" en orden inverso al de la frase anterior (amplitude 16D, 832; re-uploading 12D, 624). | Reordenar: "832 and 624 weights, adding 624 and 416". |
| C21 | S:831 | "Friedman test across all eight architectures": no se dice cuáles. Tampoco se dice qué RF (2,273 o 2,787) entra en "224 of 225 pairs". | Especificar ambos. |
| C22 | S:1104 | "3.5 to 7.4 times": con los valores redondeados de Tab. S-operational_multiseed sale 3.7–7.5. | Verificar con los valores sin redondear o citar el rango redondeado. |
| C23 | S:182 | Referencia nula: "0.0000 (0.0002)", con un MC SE no nulo, aunque M:255 dice que la diferencia "is exactly 0". | Poner "—" o explicar el valor. |
| C24 | S:875 | `tab:bottleneck_relaxation` está ordenada por rango medio (LbL 0.6605 aparece debajo de PI-QNN 0.6555) sin indicarlo. | "sorted by mean rank". |

**Verificaciones que cuadran** (no requieren acción): totales de la Tabla 1 (2,273 / 514 / 528 / 3,315; 217 positivos; porcentajes por estación); 3,363 − 48 = 3,315; recuentos de parámetros (2,368 + 208 + 8 + 5 + 24 = 2,613; PureLSTM 10,103; QTL 617; GELU 180; LeakyReLU 28; amplitude 3,237; re-uploading 3,041); recuentos de puertas y profundidades de LSTM+QNN, LbL y QTL (incluido el depth 13 de QTL con rango de CNOT 2 en la segunda capa); z, p y MDE de `tab:paired_ci` y `tab:qnn_paired`; Holm 0.079; seed shares (2.3 / 4.9 / 6.6 %); error estándar de Hanley–McNeil 0.024 y toda la Tabla `tab:hm_mde`; BSS (0.082, −0.003 … −0.140); Brier de la constante 0.4017 = 0.2422; Wilson [0.699, 0.972]; TOST (p ≈ 0.041; 0.035; 0.111; 0.015; 0.277); Kendall W (0.058; 0.174; 0.509); Cliff δ = 0.9911; ratios SD/SE de la simulación (1.07 / 1.17 / 1.57; 1.01 / 1.08 / 1.06); coherencia interna F1–Acc–FPR de las 12 filas de `tab:single_run_results`; flips y prevalencias S0–S7.

---

## 3. Precisión del inglés científico

### 3.1 Frases ambiguas o poco naturales

| Ubicación | Original | Propuesta |
|---|---|---|
| M:67 | "Random Forest--LSTM (+0.015) is excluded from zero by year-block bootstrap intervals but not by DeLong" | "for Random Forest–LSTM (+0.015) the year-block bootstrap intervals exclude zero, whereas DeLong does not reject the null" (una diferencia no "se excluye de cero"; el intervalo excluye el cero) |
| M:67 | "Re-evaluating fixed classical predictors under alternative label definitions, given that 23 % of the test months are incomplete, preserved…" | "Because 23 % of the test months are incomplete, we re-evaluated fixed predictors under alternative label definitions; this preserved…" |
| M:82 | "the remedy addresses the source of error that is smaller on the scale of an individual AUC" | "the remedy targets the smaller of the two error sources, at least on the scale of a single AUC" |
| M:86 | "In the time-series and weather QML studies we read" | "In the time-series and weather QML studies cited above" (además, "we read" sugiere una selección no sistemática) |
| M:86 | "report no interval or test on test sets of 14 and 5 days" | "report no interval or test, although their test sets span only 14 and 5 days" |
| M:107 | "monthly totals are sums over the available days, so incomplete months enter the data" | "…; incomplete months therefore remain in the panel" |
| M:107 | "loses its first 12 months as antecedent window" | "as the antecedent window" |
| M:172 | "the empirical statistic of this paper" | "the main test statistic of this paper" |
| M:193 | Paréntesis anidado de ~70 palabras con tres punto y coma | Sacar a frases independientes: "Holm correction is applied within two families … . Bootstrap intervals are pointwise … . For reference, table captions quote the first-step threshold 0.017." |
| M:222 | "…the block estimate is itself imprecise: its MDE (0.017) is substantially smaller…, and given the limited number…" (dos dos puntos en una frase) | Dividir en dos frases. |
| M:255 | Una frase de ~130 palabras con siete punto y coma que describe el diseño | Dividir en 3–4 frases o pasar los parámetros a la Tabla `tab:sim`. |
| M:286 | "At this ensemble RF−LSTM gap of 0.0115" | "At the observed ensemble gap of 0.0115" |
| M:294 | "An incomplete month can only under-count precipitation, so negatives may be over-represented" | "…so some months labelled negative may in fact exceed the threshold" |
| M:392 | "(ECE 0.112–0.149; its direction was not analysed)" | "(ECE 0.112–0.149; whether they are over- or under-confident was not analysed)" |
| M:455 | Párrafo de ~450 palabras | Dividir en tres: RF − híbrido / híbrido − control / otros híbridos. |
| M:463 | "…do not establish skill beyond the seasonal-geographic base rate, whatever its architecture." | "…for any of the architectures evaluated." ("its" no tiene antecedente) |
| M:498 | "rather than definitive proof of variance reduction"; "inferential sensitivity" aparece dos veces seguidas | "rather than as evidence of variance reduction"; eliminar la repetición. |
| M:533 | Primera frase de las conclusiones: ~110 palabras, con punto y coma y dos puntos | Dividir; las conclusiones repiten casi todos los números del abstract. Recomiendo reducir a los 3–4 mensajes clave. |
| S:298 | "We emphasize that no station data were fabricated or artificially extended" | "No gap filling or extension was applied to any station record." |
| S:328 | "a 15-seed replication rerun" | "a 15-seed rerun" |
| S:1022 | "…we did not perform this owing to …; we consider it …; This evaluation…" | "We did not perform this because of the cost of a 15-seed replication under noisy gradients. We consider it the most consequential open extension." |
| S:1207 | "There is a classical projection from 51 → 4 hidden inside the models." | "Inside the models, a classical linear layer projects 51 → 4 dimensions." |
| S:1244 | "(the `for xi in qi` loop in `forward()`)" | Eliminar o pasar a nota; es un detalle de código. |

### 3.2 Comma splices

El texto está en general bien puntuado; el uso de punto y coma es correcto, aunque excesivo (ver §3.1). Casos a revisar:
- S:1090: "…single-run AUC did not degrade (…), and thermodynamic features rose in the importance ranking. However, because this check was conducted on a single initialization seed, it is reported…": correcto, pero muy largo.
- M:284: "…coverage 0.916–0.926), so nominal coverage was not achieved uniformly in these scenarios; additionally, in scenarios C and E…": empezar frase nueva en "Additionally".
- M:202: "(a descriptive comparison of scales, not an estimate …)" seguido de otra cláusula: dividir.

### 3.3 Registro informal o retórico (sobre todo en el supp)

Estos términos contrastan con el tono cuidadosamente matizado del texto principal. Recomiendo eliminarlos o sustituirlos:

| Ubicación | Término | Sugerencia |
|---|---|---|
| S:707 | "Crucially, this apparent single-run superiority dissolves under closer examination" | "The single-run ordering is not robust:" |
| S:328, S:353 | "To rigorously benchmark / isolate" | "To assess / To separate" |
| S:537 | "culminate in a rigorous 15-seed … protocol", "explicitly isolates" | "are each trained with 15 seeds", "separates" |
| S:361 | "For full transparency" | eliminar |
| S:551, S:605, S:1047 | "We emphasize" | eliminar |
| S:348 | "completely resolved" | "removed the collapse in all 15 seeds" |
| S:611 | "is nonetheless robust to this caveat" | "does not depend on this caveat, because…" |
| S:711 | "is entirely an artifact of default threshold selection" | "is largely explained by the default threshold" (τ se eligió sobre el test) |
| S:1014 | "notable structural robustness" | ver §1.4.2 |
| S:1090 | "genuine dynamical signal" | "dynamical signal" |
| S:1199, S:1235 | "Key observation:", "Implication:" | integrar en prosa |
| S:1254 | "trivially small" | "small (256 bytes)" |
| S:930 | título "Deconstructing the Operational Paradox" | "Threshold Dependence of Operational Metrics" |
| S:931 | "suffers from severe overprediction bias" | "has a high false-positive rate (0.617)" |

### 3.4 Ortografía mixta británica/americana

El texto mezcla ambas variantes: M:86 *modelling*, M:202/M:392 *analysed*, M:675 *centred*, S:1042 *vapour* (británico) frente a S:303/S:762 *analyzed*, S:386 *centered*, S:266 *favor*, S:1032 *behavior*, *standardized* (americano). Hay que elegir una variante según la revista y unificar.

---

## 4. Sintaxis LaTeX y referencias

**Verificado automáticamente:** no hay `\ref`/`\Cref` locales rotos ni etiquetas duplicadas en ninguno de los dos ficheros. Todas las referencias externas (`S-…` desde main y `M-…` desde supp) apuntan a etiquetas existentes. Todas las claves `\cite*` tienen su `\bibitem`.

| # | Ubicación | Problema | Corrección |
|---|-----------|----------|------------|
| L1 | M:243–248, M:277–282 | `fig:paired_ci` y `fig:sim` no se citan en el texto (la mayoría de revistas lo exigen). | Añadir "(\Cref{fig:paired_ci})" en M:222 y "(\Cref{fig:sim})" en M:284. |
| L2 | M:147 | `tab:expmap` no se cita en el texto principal (solo desde el supp). | Citarla en §2.2 o al inicio de §3. |
| L3 | M:645 | `\bibitem{schuld2021fourier}` no se cita en main (se imprime igualmente con `thebibliography`). | Citarla (p. ej. en M:136 o M:455) o eliminarla. |
| L4 | S:1324, S:1339, S:1365, S:1383 | `breiman2001`, `grinsztajn2022`, `mcclean2018` y `schuld2022` no se citan en el supp. | Eliminarlas de la bibliografía del supp. |
| L5 | M:55–57 / S:55–57 | `\affil[*]{Corresponding author…}` sin `\author[1,*]`: el asterisco aparece como una afiliación huérfana. | `\author[1,*]{…}` o usar `\thanks`. Considerar `\orcidlink` en lugar del `\href` en superíndice. |
| L6 | M:109, M:136, M:250, M:289, M:455, M:506 | Referencias genéricas "(Supplement)" sin destino. Solo hay **una** `\Cref{S-…}` en todo main. | Sustituir por `\Cref{S-app:leakage_correction}`, `\Cref{S-app:architectures}`, `\Cref{S-tab:mde}`, `\Cref{S-sec:positive_control}`, `\Cref{S-app:noise}`, etc. |
| L7 | M:539 | `\todo{archive … DOI}` sigue pendiente (oculto con `\showtodofalse`). | Resolver antes del envío. |
| L8 | S:1290 | `tab:cost_summary`: especificación `{lcccccc}` (7 columnas) para 6 columnas usadas. | `{lccccc}`. |
| L9 | S:973, S:1161 | `\footnotesize{…}` usa un *switch* como si fuera un comando con argumento. | `{\footnotesize …}`. |
| L10 | M:286 vs M:289 | `\paragraph{Panel-matched check.}` frente a `\textbf{Conditional perturbation…}`: formato de encabezado inconsistente; además, el control positivo no es una simulación y queda dentro de §4.2. | Usar `\paragraph` en ambos o mover el control positivo. |
| L11 | S:400 | `\multirow{8}{*}{Physical \&}`: el texto del grupo queda truncado ("Physical &"). | "Physical \& synoptic". `dp_std` aparece en el grupo "Antecedent" pero cuenta entre los 17 físicos. |
| L12 | M:41–43 | `\textcolor` sin cargar `xcolor` explícitamente (funciona solo porque `tikz` lo carga). | `\usepackage{xcolor}` antes del bloque `\todo`. |
| L13 | M:229, M:262, M:330, M:372, M:443, S:86… | `\resizebox` sobre tablas: produce tamaños de letra distintos en cada tabla (muchas revistas lo rechazan). | `\small`/`\scriptsize` + `tabularx`/`\tabcolsep`, o `adjustbox` con `max width`. |
| L14 | Global | Formatos numéricos inconsistentes: `2{,}273` frente a `2,273`; `$B=2000$` (M:322, M:350) frente a `$B=2{,}000$`; `$+$0.021` (Tab. label_sens_delta) frente a `$+0.021$`. | Unificar, idealmente con `\num{}` de siunitx (ya cargado). |
| L15 | M:28–31 / S:28–31 | `xr-hyper` exige compilar main y supp de forma cruzada (dos pasadas cada uno). Muchas revistas piden un único fichero fuente. | Documentar el orden de compilación (README/latexmkrc) o preparar una versión con el supp combinado. |
| L16 | S:1071 | Notación `L_v q` (humedad específica) frente a `q_j` (expectación de qubit): choque de símbolos. | Usar ⟨Z_j⟩ o r_j para la lectura. |
| L17 | S:1127 | `Qiskit Aer ≥0.17`: versión no fijada, lo que contradice la tabla de reproducibilidad. | Fijar la versión exacta. |
| L18 | S:1080 | Sección `app:lambda_sweep` sin contenido ("no results from it are reported"). | Eliminar la subsección. |

Los logs existentes (`main.log`, `supp.log`) no muestran referencias indefinidas ni avisos de *overfull*. Hay que confirmarlo con una compilación limpia tras los cambios.

---

## 5. Rigor argumental: afirmaciones desmedidas o que requieren matización

1. **M:109:** "making the out-of-sample split a test of robustness under distribution shift". Un único split cronológico no constituye un test de robustez, y M:506 lo reconoce ("with one chronological split we cannot determine how distribution shift modifies model performance"). Propuesta: *"…so the out-of-sample evaluation is carried out under a covariate shift whose effect cannot be isolated with a single split."*
2. **M:138:** "demonstrating that variations of this scale … are within optimization and seed variance" (ver §1.2).
3. **M:82 / M:202:** generalización de la dominancia del error de muestreo a QML a partir de modelos clásicos (ver §1.1).
4. **M:533 (Conclusiones):** "The defensible conclusion is not that Random Forest is superior, but that this benchmark cannot resolve differences of this size reliably." Para RF − híbrido (0.029), los tres procedimientos excluyen el cero de forma nominal, el signo se mantiene en los 8 escenarios de etiquetas y la diferencia supera el MDE por bloques (0.024). El propio protocolo (ítem 2, M:482) pide no tratar el MDE como umbral. La conclusión debería ser simétrica: *"the evidence that Random Forest exceeds the hybrid is moderate and stable in sign but does not survive multiplicity correction; differences among the classical models remain unresolved."* El artículo aquí matiza en exceso en la dirección contraria a la habitual, lo cual también es un problema de rigor.
5. **M:88 / Contribución 1:** "A statistical framework": DeLong, block bootstrap, two-way bootstrap y MDE son herramientas estándar. Sugiero "a combination of standard procedures, applied to two explicit estimands".
6. **Título:** "Limits of Resolution in Quantum Machine Learning". La mayor parte del análisis (budget, simulación, etiquetas) es sobre modelos clásicos y el argumento es agnóstico a QML (M:508). Considerar "…in Small Machine-Learning Benchmarks, with a Quantum Case Study".
7. **M:86:** la caracterización de \citet{marchisio2026} y \citet{dasilva2026} debe comprobarse literalmente contra los artículos (tamaños de test, número de runs). Es la única afirmación sobre trabajos ajenos que sostiene la motivación.
8. **S:365:** "The variance-stabilizing effect of LbL-QNN is therefore supported by the depth-matched comparison". "Effect" es causal, mientras que la nota al pie S:557 reconoce el confundido con presupuesto de épocas y LR. Propuesta: "the lower seed variance of the combined LbL configuration is observed in the depth-matched comparison, but is not attributable to layer-wise training alone".
9. **S:709:** "a substantial fraction of pooled discriminative performance arises from separating high-base-rate from low-base-rate stations". Es plausible y el cálculo de §1.5.9 (AUC solo con la estación ≈ 0.617) lo apoya, pero hay que cuantificarlo en el artículo en lugar de inferirlo de los AUC por estación.
10. **S:711:** "multi-seed replication reveals that these single-run margins reflect initialization variance rather than robust architectural advantages": los datos a nivel de semilla no permiten separar esto del error de muestreo (lo dice el propio S:257). Propuesta: "…are not reproduced across seeds".
11. **S:727 / S:1090:** "Consistent with the hypothesis that geographic dummies do not substitute for genuine dynamical signal": una ablación de una semilla sobre XGBoost, cuyo cambio (+0.0046) es menor que la SD de semilla de XGB, no permite apoyar esa hipótesis. Además, `vuln_costera` y `pmed` (ver §1.5.5–6) pueden codificar la estación.
12. **S:1097:** "the absence of temporal leakage rests on that timeline, which must be checked against the feature-construction code". Esto debe verificarse y afirmarse (o corregirse), no dejarse como tarea en el manuscrito.
13. **S:541:** inferencia a partir de un Friedman declarado descriptivo (ver §1.3.7).
14. **S:266:** "it shows that the low-dimensional compression is not the explanation for the modest QKSVM score". La comparación PCA-4 lo sugiere, pero el confusor de *bandwidth* (§1.4.4) impide afirmarlo.
15. **M:463:** "The observed gaps … (0.003–0.03) are mostly smaller than these values. This does not mean that the models are equal." Es correcto, pero conviene añadir la idea complementaria: tampoco significa que no difieran, y para eso falta un análisis de equivalencia con margen justificado (protocolo, ítem 9).

---

## 6. Sugerencias adicionales (opcionales, no bloqueantes)

- Añadir un baseline **station-only** y otro **station × month frequency fitted on 1938–2007** (§1.5.9–10).
- Reportar la **seed share del híbrido** y la **seed share en la escala de la diferencia** (§1.1).
- Un análisis **within-station** (AUC estratificado o un modelo con efectos fijos de estación) eliminaría el componente de tasa base del AUC agrupado.
- Para el protocolo (Tabla `tab:protocol`): añadir un ítem "Report a station-/group-only baseline when the target threshold is pooled across heterogeneous sites".
- Para el ítem 3 del protocolo: indicar que la simulación debe especificar **qué estimando** (condicional o superpoblacional) se evalúa.

---

*Fin del informe.*
