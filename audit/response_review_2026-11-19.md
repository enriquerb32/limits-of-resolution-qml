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
