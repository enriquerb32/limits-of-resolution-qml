# Respuesta a la revisión editorial del 19-11-2026

Rama `v2`. Solo hay cambios de redacción. No hay cifras nuevas: todas las que aparecen ya estaban en el texto y en `results/`. Los tests `tests/manuscript` pasan (57/57) y el resumen tiene 199 palabras (límite del test: 200). `M` = `paper/main.tex`.

## Confirmado y corregido

| Punto | Cambio |
|---|---|
| 1. Resumen: «no learned model beat…» | La frase era cierta para las estimaciones puntuales (ningún modelo supera a la climatología), pero «added information to SEAS5» iba más allá de la evidencia. Ahora dice «was resolvably better than / added detectable information», y la ausencia de ventaja del circuito lleva «though equivalence is not established». Lo mismo en Discusión §5.1 y Conclusiones. |
| 4. 19 clústeres | Se nombran en el resumen. En la Introducción (Scope), RF − LSTM y RF − LSTM+QNN se declaran exploratorios. Esas dos filas llevan † en las Tablas 3 y 6, con una nota que explica por qué. También se repite en las Conclusiones. RF − LSTM+QTL (secundario, CR2 p = 0,008) no se toca: el texto no tiene un análisis de bloques para ese contraste. |
| «the data cannot tell the models apart» | En la Fig. 3 ahora dice «this test set and procedure cannot tell the two models apart». |
| Definir «resolved» antes | La definición está en la Introducción (Scope), con un enlace a §3.3. |
| §4.7: cobertura de un conjunto finito | Se dice en la primera mención de la cobertura, no después. |
| «clusters, not rows, set the resolution» | Está en el resumen y en las Conclusiones (ya estaba en §4.7 y §5.3). |
| 2. Contribución en una frase | Se añade un párrafo tras la lista de contribuciones. |
| 3. Mensaje cuántico en positivo | Se añade en §5.1: una contribución cuántica solo se puede atribuir con sustitutos clásicos emparejados en interfaz y capacidad, más señales plantadas. |
| Añadido 6: alcance cuántico | En §5.1: el resultado se limita a circuitos pequeños, sin ruido y detrás de un cuello de botella de baja dimensión. |
| 7. Regla práctica sobre DeLong | Se añade una regla práctica en §5.4. Va como párrafo y no como tabla nueva, para no alargar el texto. |

## Ya estaba en el manuscrito (la revisión no lo vio)

- **Posicionamiento** («no como quantum vs classical»). El título y el objetivo ya plantean un problema de inferencia en benchmarks pequeños, con QML como caso de estudio.
- **Diagrama de decisión** (añadido 1). Es la Fig. 3 (`fig:inference_map`): entrada → estimando → CR2, semillas y MDE → Holm → decisión → robustez.
- **Ejemplo numérico** (añadido 3). La Tabla 4 da 0,049 para N = 500, π = 0,41 y AUC 0,68, que es justo el ejemplo que propone la revisión.
- **Simulación de cuándo falla CR2** (punto 6). La Fig. 4 y §3.2 lo muestran con AR(1) más efectos año×estación: ningún procedimiento basado en años es calibrado (0,87–0,91).
- **El MDE no es un umbral** (añadido 4). Está en §3.3, en el ítem 2 del protocolo y en el último párrafo de las Conclusiones.
- **Selección de modelos en validación** (añadido 5). §5.3 calcula el MDE de la validación (0,030–0,048), y el ítem 6 del protocolo pide hacerlo.
- **Checklist mínimo** (añadido 2). Esa función ya la cumple la Tabla 12; un segundo checklist la duplicaría.

## Discrepancia parcial

- **Cambiar el nombre «external benchmark».** No se cambia, porque California Housing es un benchmark. §4.7 ya dice que fallaron 3 de las 5 predicciones registradas, que la potencia de CR2 fue del 4 % y que el experimento no demuestra la detección de diferencias reales con tests pequeños.

## No hecho (requiere trabajo nuevo o una decisión del autor)

- **5. Reorganizar el cuerpo y adelantar la Fig. 3.** Es un cambio estructural. El texto ha pasado de 29 a 30 páginas con estos añadidos.
- **6. Una validación verdaderamente independiente** (un tercer dominio, un test prospectivo o un benchmark público con clústeres). Requiere experimentos nuevos y sigue declarada en Limitaciones.

---

# Segunda parte: revisión de las figuras (19-11-2026)

Se regeneran cuatro figuras con sus scripts (`scripts/make_*_plot.py`), sin datos nuevos.

## Error factual encontrado al revisar la Fig. 1

El texto decía «2008–2012, the only five years with all four stations», en M §4.1, en M Limitaciones y en S «cancellation by period». Los datos del panel dicen otra cosa:

- San Javier tiene ventanas hasta 2011 (una sola en ese año).
- En 2012 solo hay tres estaciones.

Las cuatro estaciones coinciden solo en **2008–2011**, y 2008–2012 son los cinco años con **más de dos** estaciones. Las cifras de `exp45` (periodo 2008–2012) no cambian; se corrige la descripción en M, en S y en el docstring de `exp45`. `make_design_plot.py` comprueba ahora ambos datos con asserts: 19 años de test y 2008–2011 como únicos años con las cuatro estaciones.

## Confirmado y corregido

| Figura | Cambio |
|---|---|
| 1. Diseño | Recuadro sobre 2008–2011 y nota «19 test years = 19 clusters; the only 4 with all four stations». Los recuentos de test pasan junto al nombre de la estación. El pie se reescribe. |
| 2. Arquitectura | La etiqueta «swap» pasa a «replaces the circuit». |
| 4. Forest plot | Solo CR2 (el procedimiento que decide) va en color; DeLong y el bootstrap van en grises y con marcador propio. RF − LSTM y RF − LSTM+QNN llevan †. El pie dice que ninguna diferencia de las familias Holm pasa la corrección. |
| 5. Calibración | La línea de 0,95 es más marcada y lleva rótulo. Las celdas AR(1) + año×estación, el único régimen sin calibración, se dibujan con un rombo propio y tienen entrada en la leyenda. |
| 6. Resolución | Se añade un eje superior con el número de años (clústeres) de cada punto. La extrapolación se rotula como «illustrative… if added years are independent». |

## La revisión no acierta

- **Fig. 1: colores de partición parecidos.** Son azul, verde y naranja, y ya había una línea temporal etiquetada encima.
- **Fig. 4: tres marcadores difíciles de distinguir.** Ya tenían color y forma a la vez. El cambio a color/gris se hace por jerarquía, no por legibilidad.
- **Fig. 4: «Cluster/iid» poco explicativo.** Esa abreviatura no aparece en la figura; está en las tablas, y sus pies la definen.
- **Fig. 5: «seis colores».** La figura ya usaba un solo color de acento (CR2), con los demás procedimientos en gris.
- **Fig. 6: etiquetas numéricas internas.** Ya estaban solo en N = 529 y en el punto extrapolado.
- **Fig. 6: anotar que añadir filas dentro de los mismos clústeres no reduce el MDE.** La figura no muestra eso: sus subconjuntos son años completos. La anotación afirmaría algo que el gráfico no contiene. El eje superior de clústeres transmite la idea sin afirmarlo.

## Desaconsejado

- **Banda de MDE en el forest plot.** Una banda ±MDE alrededor de cero invita a leer el MDE como umbral de significación, que es justo lo que el texto prohíbe (§3.3, protocolo ítem 2). El MDE de cada contraste sigue en las tablas.
- **Ordenar los contrastes por tamaño del efecto.** Se rompería la agrupación por familias, que es la que sigue el texto y la corrección de Holm.
- **Franja de resultados en la Fig. 2.** La figura aparece en §2, antes de los resultados. El hallazgo está en §4.4 y en las Conclusiones.

## Decisiones sobre lo pendiente

| Propuesta | Decisión | Motivo |
|---|---|---|
| Mapa de inferencia como Figura 1 | **Hecho** | Pasa a la Introducción, al final de las contribuciones. Es la figura que resume la contribución metodológica. El diseño pasa a ser la Fig. 2. |
| Recortar la prosa de la Fig. 1 (antes Fig. 3) | **Hecho** | Cada caja tiene ahora una frase. El pie ya no se dirige a «readers without a statistical background». |
| Partir las Figs. 3 y 4 en paneles | **No** | La Fig. 3 ya separa visualmente el modelo (arriba) de los controles (recuadro discontinuo); partirla repetiría el bloque común. La Fig. 4 ya agrupa por familias con encabezados; en paneles separados, los intervalos dejarían de compartir eje y no se podrían comparar entre familias. |
| Figura «headline» | **No** | Repetiría resultados que ya están en las Figs. 4–6 y en §4.7, y añadiría una página. |
| Sensibilidad de los veredictos frágiles | **Hecho, en el suplemento (Fig. S2)** | `scripts/make_sensitivity_plot.py`, solo con resultados archivados (bloques de 1–3 años, dejar fuera un año, Alicante+Barcelona, segundo panel). Comprueba que los ficheros coinciden entre sí. Se cita desde M §4.1 y se registra en `run_v2.ps1`. |

Ninguna figura existente se retira: las seis del texto principal y la del suplemento sostienen cada una una afirmación que no está en otra figura. El texto principal sigue en 30 páginas y el suplemento pasa de 20 a 21.
---

# Tercera parte: cierre del análisis (revisión del 19-11-2026, cuatro puntos)

Las cifras nuevas salen de `exp55_review_closing.py` y `exp54_power_clusters_rows.py`, ambos sin entrenamiento. Además se reentrenó Linear-4D con imputación causal (`run_v2.ps1`, paso `03c_linear4d_causal`). Las comprueba `tests/manuscript/test_review_2026_11_19.py`. Pasan los 130 tests de `manuscript` y `unit`. M pasa a tener 31 páginas y S, 23.

| Punto | Estado | Cambio en las conclusiones |
|---|---|---|
| 3. 1,2 % atribuido a ambos cocientes | **Corregido.** En las celdas de 19 años, 1,2 % para ≤ 0,59 y 5,8 % para ≤ 0,72 | — |
| 3. Alicante + Barcelona | **Corregido.** RF − QNN (p = 0,014) pasaría el primer paso de Holm; se presenta como sensibilidad exploratoria | — |
| 3. Conclusiones | **Corregido.** CR2 infracubre con AR(1) + año×estación; su calibración bajo la cancelación «no está establecida» | — |
| 1. P60 | **Documentado.** 2.847 meses hasta 1996; los registros empiezan antes de la primera ventana (Barcelona en 1924). No es una fuga. S decía «1938–1996»; se corrige | — |
| 1. Pipeline causal | **Hecho** para las dos familias principales (5 modelos, incluido Linear-4D reentrenado). Tabla S2 y fila «Causal refit» en la Fig. S2 | **Sí.** RF − LSTM deja de estar resuelto (p = 0,070) y RF − LSTM+QNN pasa Holm (p = 0,009), pero no con bloques de 3 años (p = 0,076). Ningún contraste no resuelto pasa a resuelto. |
| 2. Resolución del segundo panel | **Hecho.** Columna «MDE CR2 (pred.)» en la Tabla 11 | **Sí.** «El escalado replicó» solo vale para DeLong. Con CR2, el MDE del panel 2 es 1,36–1,67 veces el predicho y mayor que el del panel 1 en los seis contrastes entre modelos completos. Fourier-2 es la excepción, porque su cociente cluster/iid baja. |
| 2. Frase «Clusters, not rows» | **Sustituida** en el resumen («Independent information, not row count, sets the resolution»), §4.7, §5.3 y Conclusiones | — |
| 4. Referencias | **Hecho.** La Tabla 10 se reconstruye con CR2: modelo − climatología, SEAS5 a lead 2 (operacional, primero) y lead 1, y el contraste incremental a lead 2. Los incrementos a lead 1 y las mezclas de sustitución pasan a la Tabla S23. La remisión rota de §4.5 a la Tabla 3 se corrige | No se detecta valor incremental (CR2 p ≥ 0,25 a lead 2; MDE 0,010–0,019). «The statistical models add nothing to it» pasa a «We did not detect incremental skill». |
| Simulación de clústeres frente a filas | **Hecha.** 128 celdas con 1.000 réplicas: K = 10–76 clústeres y m = 12–96 filas por clúster, en 4 patrones de dependencia. Es la nueva Fig. 6 | Con casos independientes, filas y clústeres son intercambiables. Con dependencia, cada fila añadida aporta menos: el punto de igualdad está en unas 110 filas por clúster con efectos año×estación y en 290–460 en los otros casos. CR2 cubre a 0,930–0,967 en todas las celdas; DeLong rechaza un nulo verdadero hasta en un 17,6 %. La curva empírica de subconjuntos, basada en DeLong, pasa al suplemento (Fig. S4). |

## Decisión: el pipeline archivado sigue siendo el principal

El revisor proponía hacer principal el pipeline causal. No se hace, por tres razones:

1. El archivado era el análisis principal antes de los reajustes causales. Cambiar al causal después de ver que mueve dos veredictos haría depender la elección de los resultados.
2. Todo el paper comparte el pipeline archivado: controles, circuitos congelados, teachers, SEAS5 y segundo panel. Solo se han reajustado los cinco modelos de las familias principales.
3. El cambio de veredictos ante una perturbación que mueve cada AUC como mucho 0,002 se presenta como resultado en §2.1, §4.1, en los pies de las Tablas 3 y 6 y en la Fig. 4, en Limitaciones y en Conclusiones. Refuerza la tesis: cerca del límite de resolución, un veredicto no es estable ni frente al pipeline de ajuste.

## Renombrado

El script de cierre se llama `exp55_review_closing.py`, porque `exp53` ya lo usaba `exp53_external_benchmark.py`.
