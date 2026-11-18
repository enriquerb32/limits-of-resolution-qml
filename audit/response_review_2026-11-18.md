# Respuesta a la revisión del 18-11-2026

Rama `v2`. Cada cifra nueva del manuscrito está archivada en `results/` y comprobada por `tests/manuscript/test_review_2026_11_18.py` (y las tablas, por `test_v2_tables.py`). `M` = `paper/main.tex`, `S` = `paper/supp.tex`.

## Resumen

| Punto | Estado | Cambio en las conclusiones |
|---|---|---|
| 1. Fuga temporal en la imputación | Confirmada y medida; corregida en el código (`V2_CAUSAL=1`) | Ninguno: no afecta a ninguna ventana de test; reajustar 4 modelos cambia la AUC ≤ 0,0023 |
| 1b. Etiquetas con días faltantes | Frase corregida; sensibilidad hecha | Ninguno: ningún mes usado tiene días faltantes |
| 2. Inferencia principal y estimando | Estimando fijado; CR2 con gl de Bell–McCaffrey elegido por calibración | Dos pares pasan a «resuelto antes de Holm»; ninguno tras Holm; RF − SEAS5 deja de estar resuelto |
| 2b. Bootstrap de semillas | Corregido en código y texto (había además una discrepancia código/texto) | Intervalos años×semillas cambian ≤ 0,002 |
| 3. Uso del MDE | Corregido en resumen, discusión y conclusión | — |
| 4. Cuatro afirmaciones | Corregidas; MDE predicho por contraste en la Tabla 13 | — |
| Añadido 2 (simulación) | Hecha la calibración en el diseño real; **no** se reproduce la cancelación | Se declara como limitación |
| Añadido 2b (Apéndice A) | AUC analítica incorporada | Coincide con las referencias Monte Carlo |
| Añadido 3 (control positivo) | Repetido con 4 teachers, 2 intensidades, 2 extracciones y un nulo | **Sí:** el circuito entrenado no supera a sus sustitutos; se reescribe la conclusión de la Tabla 8 |
| Añadido 1 (benchmark externo) | Hecho, con diseño registrado antes de ajustar (`1e88a96`); no prospectivo | P1–P3 fallan y P4–P5 se cumplen; DeLong resuelve con signo erróneo hasta un 16 % de los tests pequeños, CR2 nunca |
| Añadido 3a (emulación / sustitución / aprendizaje) | Hecho: rejilla 9⁴ y extractor común congelado | Emulación exacta; sobre los mismos ángulos, un circuito nuevo queda por debajo de los sustitutos clásicos |
| Sensibilidad a años influyentes | Hecha (quitando un año cada vez) | RF − LSTM cambia de veredicto con 9 de 19 años |
| Tabla 14 como herramienta | Hecha: `qml_resolution.report` (librería y CLI) | — |
| Recomendación editorial | Hecha: variantes, shots y rankings secundarios al suplemento | — |

## 1. Fuga temporal

**Confirmada.** Hay tres pasos que usan datos posteriores a la fecha de emisión (`dataset_v2.py`, S «Lookahead Audit»):

1. La interpolación de covariables mensuales es bidireccional. Un hueco en t−1 se rellena con t. En un hueco de más de 2 meses, los dos primeros meses se interpolan hacia el final del hueco, hasta 18 meses después. El texto anterior decía que los huecos largos quedaban sin predictores, lo que era falso.
2. `et0_verano_prev` se interpolaba hacia el siguiente verano completo y se rellenaba hacia atrás.
3. Un `et0_lag1m` faltante tomaba la media del registro completo, incluidos los años de test.

**Prueba de truncado** (`exp47_truncation_audit.py`). Para cada mes objetivo de test, el panel se reconstruyó con el fichero diario cortado al final de t−1, usando las transformaciones ajustadas en entrenamiento.
- Las **529** ventanas de test conservan los 35 predictores y la elegibilidad. Ninguna se vuelve elegible solo tras el corte.
- La fuga alcanza a 54 estación-meses, de los que 26 son el último mes antecedente de una ventana elegible: 24 de entrenamiento, 2 de validación y 0 de test.
- La prueba es sensible: con el pipeline causal también da 0 cambios, pero detecta que el número de ventanas elegibles pasa a 533.
- Se describe como comprobación retrospectiva, porque no hay fechas históricas de disponibilidad de los datos.

**Reajuste con imputación causal**, sobre las 529 filas comunes, con IC por bloques de años:

| Modelo | Archivado | Causal | Δ | IC 95 % |
|---|---|---|---|---|
| RF | 0,6784 | 0,6797 | −0,0013 | [−0,003; +0,000] |
| XGBoost | 0,6684 | 0,6707 | −0,0023 | [−0,009; +0,004] |
| LSTM | 0,6638 | 0,6634 | +0,0004 | [−0,006; +0,006] |
| LSTM+QNN | 0,6587 | 0,6579 | +0,0008 | [−0,006; +0,006] |

Las diferencias son un orden de magnitud menores que el error estándar de una AUC. Por eso los resultados se mantienen con el pipeline archivado, y así se declara en M §2.1.

**Etiquetas.** La frase «no month is labelled from an incomplete record» se ha sustituido.
- La regla admite hasta 3 días faltantes, y en ese caso la suma observada es un límite inferior del total real.
- En los datos, los 529 objetivos de test, los 528 meses de validación y los 2.847 meses de entrenamiento con los que se estima P60 están completos. Hay un mes etiquetado del periodo de test con días faltantes, pero no es objetivo de ninguna ventana.
- La sensibilidad con solo meses completos reproduce exactamente las AUC y P60 = 29,9 mm.
- P60 se estima con todos los meses etiquetados de entrenamiento de las cuatro estaciones (699/809/629/710), no solo con los objetivos de ventana.

## 2. Estimando e inferencia principal

**Estimando** (M §3.1): la diferencia de AUC media sobre semillas para años nuevos en las mismas estaciones, condicionada a los datos de entrenamiento y a los modelos ajustados. El panel observado se describe, pero no es objeto de inferencia. La frase «DeLong conditions on the observed test years» se ha sustituido: DeLong es una referencia bajo independencia.

**Método principal: CR2 con grados de libertad de Bell–McCaffrey** (ν = 15,9 con los tamaños desiguales de los años; `qml_resolution.cluster_robust_delta`). Se eligió por calibración en `exp48_primary_calibration.py`, con 600 réplicas por celda.

El diseño de la simulación usa:
- la máscara real de cobertura;
- efectos de año sobre la prevalencia y las puntuaciones;
- efectos año×estación propios de cada modelo, con cargas de signo opuesto;
- un dipolo de prevalencia entre estaciones;
- dependencia AR(1) entre años;
- 10, 19 o 38 años, y 2 o 4 estaciones por año.

La verdad es analítica o se calcula con 300 copias independientes del diseño.

| Celdas | DeLong | Bloques (percentil) | CR2-BM |
|---|---|---|---|
| 19 años independientes | 0,845–0,960 | 0,885–0,930 | 0,940–0,955 |
| 10 años | 0,897–0,947 | 0,873–0,908 | 0,935–0,967 |
| AR(1) + año×estación | 0,822–0,848 | 0,843–0,872 | 0,883–0,908 |

- El bootstrap por bloques con percentiles rechaza el nulo en el 6–13 % de las réplicas, así que pasa al suplemento.
- La regla de unanimidad se elimina. Holm se aplica a los p-valores de CR2, no a los de DeLong.
- Sensibilidad a bloques de años consecutivos: con clusters de 2 años, CR2 recupera la cobertura con AR(1) simple (0,937 → 0,953), apenas mejora con AR(1) + año×estación (0,908 → 0,915) y pierde potencia (0,28 → 0,22).

**Veredictos que cambian:**
- **RF − LSTM** (+0,015): CR2 p = 0,041, y **RF − LSTM+QNN** (+0,020): p = 0,031. Ambos quedan resueltos antes de Holm, pero no después (umbral 0,017). Con clusters de 2 años dejan de estar resueltos (p = 0,061 y 0,051). El texto los presenta como los veredictos menos seguros del paper.
- **RF − SEAS5** (lead 1): con CR2, p = 0,056, así que deja de estar resuelto. Antes se daba por resuelto con DeLong y los bootstraps.
- **Congelar el circuito** (Δ = −0,012/−0,013) sigue resuelto con bloques de 1 a 3 años (p ≤ 0,018).

**Corrección de una hipótesis propia.** En ningún mecanismo simulado aparece un cociente cluster/iid ≤ 0,59 en más del 1,2 % de las réplicas con 19 años (≤ 0,72: como mucho 5,8 %). Los cocientes de 0,60–0,72 del primer panel no se explican, por tanto, como ruido de un estimador con 19 clusters, y tampoco he conseguido construir un mecanismo que los produzca. El texto lo declara: la calibración de CR2 vale para los mecanismos simulados, no para este patrón.

**Bootstrap de semillas.**
- `exp42_v2_summary.py` remuestreaba conjuntamente los índices de semilla de **todos** los pares con 15 semillas, mientras el texto decía «independently between models». Ambas cosas eran incorrectas en algún par.
- Ahora se remuestrea conjuntamente solo donde hay un componente ajustado compartido: LSTM+QTL, QTL-classical, QTL tensor-81 y la LSTM de la misma semilla. Se verificó que `PureLSTM_probs == QTL_EXTRACTOR_probs`. En los demás pares el remuestreo es independiente.
- La librería tiene la opción `shared_seeds`. Los intervalos años×semillas cambian como mucho 0,002.

## 3. MDE

- La conclusión ya no usa el MDE como umbral. Cita el contraejemplo (RF − LSTM+QTL = +0,0235, frente a un MDE de 0,028) y recuerda que un efecto por encima del MDE puede no excluir el cero.
- «0,022–0,032» queda acotado a los contrastes entre modelos completos con DeLong nominal. Los contrastes en la interfaz tienen MDE de 0,011–0,016 y SEAS5 frente a climatología, de 0,057.
- La cifra a N ≈ 2.100 se presenta como extrapolación, condicionada a que los casos añadidos aporten información independiente y a que la covarianza entre modelos no cambie.
- La parte cuántica pasa de «los sustitutos igualan al circuito» a «no se detectó una ventaja; los intervalos la acotan sin demostrar equivalencia».

## 4. Cuatro afirmaciones

- **§4.6.** El P60 por estación×mes elimina la señal en las etiquetas de entrenamiento, pero no por construcción en las de test. Se añade la AUC dentro de cada celda estación×mes con la etiqueta original (`exp49`): 0,449–0,495 para todos los modelos. En esa AUC la climatología vale exactamente 0,5, así que el contraste con 0,5 coincide con el contraste con la climatología.
- **§4.7.**
  - El lead 2 es la comparación operativa. El lead 1 ya no se llama cota superior.
  - Comparación incremental frente a SEAS5 + climatología: añadir el modelo cambia la AUC entre −0,008 y −0,001 (lead 2) y entre −0,003 y +0,001 (lead 1). Todos los IC incluyen el 0 y el Brier no mejora.
- **§4.8.**
  - La banda registrada (0,025–0,035) se calculó a partir de los MDE del pipeline v1. La Tabla 13 añade la predicción por contraste con escalado N^−1/2 (RF − XGB: 0,017). Los MDE observados quedan entre 0,79 y 1,26 veces su predicción.
  - El rango 0,91–1,35 se restringe a los seis contrastes entre modelos completos (Fourier-2: 0,74).
  - El cambio de signo del ranking se presenta como descriptivo.
- **§2.2/§4.4/S5.2.**
  - El R² = 0,984 describe los readouts antes del LayerNorm y solo sobre los ángulos visitados.
  - La invariancia a una contracción común es aproximada, por el ε del LayerNorm.
  - Se elimina «whatever its weights».

## Añadidos

- **Simulación** (añadido 2): es la calibración descrita en el punto 2. Varía por separado el número de años (10/19/38) y las estaciones por año (2/4), e incluye DeLong, percentiles, CR1, CR2-BM, jackknife, wild bootstrap y la regla conjunta. No reproduce la cancelación (ver arriba).
- **Apéndice A:** añadida la AUC analítica Φ(μ_k/√(2(1+τ_s²+τ_m²))). A Δ de diseño 0,03 da 0,0290 / 0,0280 / 0,0280 / 0,0238 en los escenarios B–E, de acuerdo con las referencias Monte Carlo.
- **Control positivo** (`exp50_teacher_grid.py`). Se ha corregido la incoherencia teacher/student: el teacher pasa por el mismo LayerNorm. Hay 4 teachers no seleccionados, intensidades AUC oráculo de 0,70 y 0,85, y una segunda extracción de etiquetas: 10 configuraciones, más 2 nulas. Con CR2 sobre medias de 5 semillas:

  | Contraste | Circuito por encima | Control por encima | Nulo: falsos positivos |
  |---|---|---|---|
  | Circuito − Linear-4D | 6/10 | 0/10 | 0/2 |
  | Circuito − Fourier-2 | 0/10 | 2/10 | 0/2 |
  | Circuito − tensor 81 | 0/10 | 10/10 | 0/2 |

  **Esto cambia la conclusión.** El resultado de la Tabla 8 (circuito por encima de los mapas separables) no se generaliza. Los sustitutos tensoriales recuperan siempre la señal plantada, pero una cabeza con circuito entrenada no aprende una función generada por un circuito mejor que sus sustitutos clásicos. Se han reescrito el resumen, «What is new», §4.4, la conclusión y el ítem 7 del protocolo.

## Segunda tanda (puntos que quedaban pendientes)

**Benchmark externo** (`exp53_external_benchmark.py`; diseño, contrastes, margen de 0,01 y cinco predicciones registrados en `audit/prediction_external.txt`, commit `1e88a96`, antes de ajustar ningún modelo). California Housing, objetivo binario por encima de la mediana, celdas de 0,25° como clusters, partición por celdas (182 celdas y 10.282 filas de test). Las AUC van de 0,905 (LR) a 0,921 (RF): la señal es fuerte, pero las diferencias entre modelos son del mismo orden que en los paneles de precipitación.
- La dependencia espacial domina: en el test completo, el error estándar CR2 es entre 4,4 y 6,6 veces el de DeLong, y RF − LR (+0,017) no se resuelve ni con 10.282 filas.
- En 1.000 tests pequeños de 19 celdas, DeLong cubre la referencia solo en el 55–63 % de los casos y resuelve MLP − RF (dentro del margen) con signo erróneo en el 16 %. CR2 cubre en el 94–99,8 % y no comete errores de signo.
- **Predicciones:** P1–P3 fallan. La población de referencia (39 celdas) era demasiado pequeña, y es un error de diseño que no he corregido a posteriori; la potencia de CR2 es del 4 % y su cobertura es conservadora. P4 se cumple. P5 se cumple en el límite y en parte por la corrección por población finita.
- La evaluación prospectiva no es posible: necesita datos que todavía no existen.

**Emulación, sustitución y aprendizaje** (`exp52_emulation_substitution.py`).
- **Emulación:** la rejilla de 9⁴ puntos reconstruye exactamente los 15 circuitos entrenados. El error en los logits tras el mismo LayerNorm es como mucho 2,6·10⁻¹⁴ y las AUC son idénticas. Truncar la serie a frecuencia 1 cuesta 0,058 de AUC de media.
- **Sustitución sobre ángulos comunes:** con el extractor de cada semilla congelado, un circuito entrenado desde cero (0,652) queda por debajo de Fourier-2 (0,664; p = 0,002) y de los mapas tensoriales (0,667–0,668), y no se distingue de Linear-4D. El circuito original (0,660) no se distingue de Fourier-2 y queda 0,008 por debajo del mapa de 81 funciones (CR2 p = 0,035; el bootstrap años×semillas no lo resuelve).

**Años influyentes** (`exp51_review_extras.py`). Al quitar un año cada vez:
- RF − LSTM cambia de veredicto con 9 de los 19 años, y RF − LSTM+QNN con 2.
- RF − XGBoost pasa a resolverse sin 2020 o sin 2022.
- RF − LSTM+QTL y los circuitos congelados no cambian.

**Tabla 13.** Ya muestra el intervalo principal; ningún contraste del segundo panel queda resuelto (p ≥ 0,107).

**Tabla 14 como herramienta:** `qml_resolution.report`, como librería y como comando (`python -m qml_resolution.report`). Devuelve el contraste pareado, CR2-BM, DeLong, el bootstrap años×semillas con semillas compartidas declaradas, los MDE, Holm dentro de la familia y una curva de potencia bajo supuestos declarados. Reproduce la Tabla 6 desde el archivo.

**Material movido al suplemento:**
- la variante PI-QNN y los controles Fourier-1/3/4 de las Tablas 5 y 7 (con la Figura 1 ajustada);
- los detalles de shots y de re-ejecución;
- las variantes de SEAS5 no especificadas antes del análisis.

**Curva de resolución (Figura 2):** el texto aclara que es una comprobación de consistencia y no evidencia independiente, porque los subconjuntos se solapan, los modelos son los mismos y se usa DeLong.

## Lo que sigue abierto

- Evaluación prospectiva del benchmark externo y una población de referencia mayor, con un diseño nuevo que habría que registrar antes de analizar.
- La suite completa de tests no ha terminado en esta máquina: la paró el sistema por falta de memoria al 71 %, sin fallos hasta ese punto. Los tests de manuscrito (56) y de unidad (63) pasan.
