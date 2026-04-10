# Ejercicio 1 – Respuestas

## Pregunta 1: ¿Cómo se obtienen las coordenadas del biplot y cómo se interpreta?

### ¿Qué es un biplot?

Un biplot es un gráfico que **superpone dos cosas en el mismo plano**:

1. Las **observaciones** (los 477 estudiantes), representadas como puntos `o`.
2. Las **variables originales** (horas, asistencia, tareas, retrasos, planificación), representadas como flechas.

Todo esto se visualiza en el espacio de las dos primeras componentes principales (PC1 y PC2).

---

### ¿Cómo se calculan las coordenadas?

El biplot se generó con `biplot(pca_res, scale = 0, xlabs = rep('o', 477))`.

El parámetro `scale = 0` indica la siguiente convención:

#### Coordenadas de las observaciones (puntos `o`)
Son las **proyecciones (scores)** de cada observación sobre PC1 y PC2. Matemáticamente:

$$Z = X \cdot W$$

donde $X$ es la matriz de datos estandarizados (centrados) y $W$ es la matriz de vectores propios (loadings). Estas coordenadas se leen en los **ejes inferior e izquierdo** del gráfico.

#### Coordenadas de las variables (flechas)
Con `scale = 0`, cada flecha tiene como extremo el vector loading **multiplicado por la desviación estándar** de su componente principal (es decir, $\sqrt{\lambda_j}$). Esto hace que las flechas sean proporcionales a la varianza explicada. Estas coordenadas se leen en los **ejes superior y derecho** del gráfico.

---

### ¿Cómo se interpreta el biplot?

Mirando el gráfico entregado, se pueden sacar las siguientes conclusiones:

| Elemento | Interpretación |
|---|---|
| **Longitud de las flechas** | Variables con flechas más largas tienen más peso en las componentes. |
| **Dirección de las flechas** | Indica en qué componente principal contribuye más la variable. |
| **Flechas apuntando en la misma dirección** | Variables positivamente correlacionadas entre sí. |
| **Flechas en dirección opuesta** | Variables negativamente correlacionadas. |
| **Flechas perpendiculares** | Variables aproximadamente independientes (baja correlación). |

**Observaciones sobre el biplot específico:**

- Las variables **horas, asistencia, tareas y planificación** apuntan hacia la **derecha** (PC1 positivo). Esto significa que estas cuatro variables están positivamente correlacionadas entre sí y están muy relacionadas con PC1. PC1 puede interpretarse como un *"factor de comportamiento académico general"*: estudiantes con valores altos en PC1 tienden a asistir más, estudiar más horas, entregar más tareas y planificar mejor.

- La variable **retrasos** apunta hacia **arriba** (PC2 positivo), siendo casi perpendicular a las demás. Esto indica que **retrasos es casi independiente** de las otras cuatro variables. PC2 puede interpretarse como un *"factor de impuntualidad"* separado del rendimiento general.

- Los **puntos (observaciones)** muy a la derecha del gráfico corresponden a estudiantes con valores muy altos en tareas, asistencia, horas y planificación. Los puntos muy arriba tienen muchos retrasos. La mayoría de los estudiantes se concentra cerca del origen, sin valores extremos.

---

## Pregunta 2: Comparación entre `prcomp` y el cálculo manual. ¿Qué contiene el objeto resultante?

### Comparación de resultados

En el notebook se realizó el PCA de dos maneras:

**Método manual:** se calcularon los valores y vectores propios de la **matriz de covarianza empírica** `cov(scaled_df)` usando `eigen()`.

**Método automático:** se usó `prcomp(scaled_df, scale = FALSE)`.

Los resultados son **equivalentes**, con la única diferencia de un posible cambio de signo en los vectores (lo cual es normal, ya que los vectores propios están definidos salvo un signo, y en el código se corrige con `w <- -mdat.eigen$vectors[,1:2]`).

Por ejemplo, los loadings para PC1 obtenidos manualmente son:

| Variable | PC1 (manual) | PC1 (prcomp) |
|---|---|---|
| horas | 0.4768 | 0.4768 |
| asistencia | 0.4708 | 0.4708 |
| tareas | 0.5238 | 0.5238 |
| retrasos | 0.2400 | 0.2400 |
| planificacion | 0.4680 | 0.4680 |

Son idénticos ✅

---

### Componentes del objeto `prcomp`

Al ejecutar `pca_res <- prcomp(scaled_df, scale = FALSE)`, el objeto retornado tiene 5 componentes:

#### `pca_res$sdev` — Desviaciones estándar de cada PC
Son la **raíz cuadrada de los valores propios** ($\sqrt{\lambda_j}$), uno por cada componente principal. Indican cuánta variabilidad captura cada componente.

```
sdev = sqrt(c(3.1523, 0.8780, 0.4170, 0.3696, 0.1831))
```

La varianza explicada por cada PC es `sdev^2`, y la proporción es `sdev^2 / sum(sdev^2)`.

---

#### `pca_res$rotation` — Matriz de loadings (pesos)
Es la **matriz de vectores propios** (los $w_j$). Cada columna es un vector propio y representa una componente principal. Cada fila corresponde a una variable original.

Equivale a la matriz `w` calculada manualmente con `eigen()`.

Esta matriz se usa para **proyectar los datos** sobre las nuevas dimensiones.

---

#### `pca_res$center` — Medias de las variables
Son los **valores con los que se centró la matriz** antes de hacer PCA (es decir, las medias de cada columna de `scaled_df`). PCA requiere centrar los datos para que tengan media cero.

---

#### `pca_res$scale` — Escalas utilizadas
Indica si se escaló la matriz (dividiendo por la desviación estándar de cada variable). En este caso, como se usó `scale = FALSE`, no se escaló dentro de `prcomp` porque los datos ya estaban estandarizados en el paso anterior. El valor será `FALSE`.

> ⚠️ **Nota importante:** Si las variables están en escalas muy distintas (como en este caso: *tareas* tiene varianza ~1663 y *planificacion* tiene varianza ~1.57), es **indispensable estandarizar** antes de aplicar PCA, de lo contrario la variable con mayor varianza dominaría las componentes. Por eso se usa `scale = TRUE` (o se estandariza manualmente antes).

---

#### `pca_res$x` — Scores (coordenadas de las observaciones)
Es la **proyección de cada observación** sobre las componentes principales. Tiene dimensión $n \times d$ (477 filas × 5 columnas en este caso).

Se calcula como:

$$Z = X_{centrado} \cdot W$$

Equivale a `scaled_df %*% w` calculado manualmente.

Estas son exactamente las coordenadas que se grafican como puntos `o` en el **eje inferior/izquierdo del biplot**.

---

### Resumen visual

```
pca_res
├── $sdev       → √(valores propios) → cuánta varianza explica cada PC
├── $rotation   → vectores propios (loadings) → dirección de cada PC
├── $center     → medias usadas para centrar los datos
├── $scale      → si se escaló (FALSE en este caso)
└── $x          → scores → coordenadas de las observaciones en el nuevo espacio
```
