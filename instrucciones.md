Algoritmos de optimización bioinspirados
Planteo de la actividad

---

**Resumen**

Este documento define el trabajo a efectuar en el módulo de Algoritmos de optimización bioinspirados del ramo de Inteligencia Artificial. La actividad busca conocer y comparar dos metaheurísticas bioinspiradas en una función de prueba estándar. Se definen los entregables y el esquema de evaluación.

---

**Contexto**

Las metaheurísticas son métodos generales de resolución de problemas. Representan un nivel superior ('meta') a las heurísticas, que son estrategias que permiten obtener soluciones de calidad razonable a un costo aceptable para problemas particulares.

Las metaheurísticas se utilizan normalmente para resolver problemas que son difíciles de resolver mediante otras técnicas, ya sea porque los espacios de búsqueda son demasiado extensos, la estructura del mismo es compleja, la función que define la calidad de la solución no cumple con condiciones requeridas por otros métodos, o bien sencillamente no existe.

Las metaheurísticas tienen diversas inspiraciones: procesos térmicos, extensiones de métodos matemáticos, etc. En este módulo nos enfocaremos en dos técnicas inspiradas de la naturaleza: Particle Swarm Optimization (PSO) y Evolutionary Algorithms (EA). Ambos son métodos de optimización que mantienen un conjunto de individuos que interactúan entre ellos.

**PSO** fue propuesto en 1995 y se inspira del comportamiento emergente observado en cardúmenes de peces o bandadas de aves, en los que cada individuo actúa de acuerdo a información recopilada individual o socialmente.

**EA** fue propuesto originalmente como Algoritmos Genéticos en 1975, encontrándose algunos antecedentes en las estrategias evolutivas, propuestas en los años 60. En los EAs se modela un proceso Evolutivo en el que cada individuo lucha por sobrevivir y transmitir su información genética a las generaciones posteriores. Aparte de competir por ser los más aptos, los individuos mutan y se cruzan para dar origen a nuevos individuos que idealmente recojan las mejores características de sus padres.

---

**Resultados de aprendizaje**

Durante este módulo, se espera que el estudiante desarrolle las siguientes competencias:

1. Comprender los conceptos asociados a la optimización con metaheurísticas, tales como función de evaluación, fitness, codificación de soluciones, multimodalidad, espacio de búsqueda, operadores de modificación de soluciones (incluyendo mutación y cruzamiento), vecindad, sesgo de selección, reinserción, explosión combinatoria, exploración, explotación, diversidad poblacional, convergencia y convergencia prematura.
2. Explicar el funcionamiento de los algoritmos PSO y EA.
3. Reconocer conceptos comunes a distintas metaheurísticas, tales como exploración, explotación, convergencia y diversidad.
4. Relacionar los conceptos anteriores, pudiendo predecir el impacto que podría tener la modificación de los parámetros asociados a ellos sobre el funcionamiento del algoritmo.

---

**Planificación de la actividad**

| Fecha actividad | |
|---|---|
| 9 marzo | Presentación del curso / Introducción a la unidad / Planteo del proyecto |
| 13 marzo | Introducción a Metaheurísticas / Inscripción de grupos (4 personas) / Particle Swarm Optimization (PSO) / Trabajo en clase |
| 16 marzo | Pre-revisión PSO |
| 20 marzo | Algoritmos Evolutivos (AE) / Trabajo en clase |
| 23 marzo | Pre-revisión AE |
| 27 marzo | Resolución de dudas para trabajo y prueba |
| 30 marzo | Prueba |
| 5 abril | Entrega informe final del trabajo (23h59, vía Siveduc) |

---

**Evaluación**

Existirán dos ítem de evaluación, los cuales serán incorporados a la evaluación global de acuerdo al programa del curso:

**1. Reporte grupal**, con una extensión referencial de 8 páginas hoja tamaño carta. Se deben entregar adjuntos las fuentes del programa debe incluir lo aprendido, documentando las experiencias y generando conclusiones acerca de los resultados obtenidos. Los criterios de evaluación del reporte serán la concordancia del contenido de cada sección descrita a continuación, además de criterios de forma como la claridad de la exposición y la buena presentación. El reporte debe estructurarse de la siguiente forma:

- **Resumen**: Resumen de menos de 150 palabras sobre el contenido del informe, destacando los puntos más importantes.
- **Introducción**: (~1 página) Contextualización del trabajo, presentación de elementos que permitan comprender el objetivo del trabajo y el resto del informe. Presentación de la organización del informe.
- **Marco experimental**: (~2-3 páginas) presentación de los algoritmos a utilizar (representación, operadores), y los valores de sus parámetros. Se deben implementar dos algoritmos (PSO y EA) y ejecutarlos para minimizar la función Rastrigin en 2 dimensiones **en el dominio continuo -3<xᵢ<7**. Esta función está definida de forma genérica para n dimensiones como:

$$f(x) = 10n + \sum_{i=1}^{n} \left[ x_i^2 - 10\cos(2\pi x_i) \right]$$

- **Resultados**: (~3 páginas) comparación del desempeño de los dos algoritmos sobre la función Rastrigin. Tablas, gráficos y comentarios sobre los resultados obtenidos. Se debe proveer link a dos videos (Youtube u otro), en donde se muestre el despliegue gráfico de la ejecución de ambos algoritmos.
- **Conclusiones**: principales descubrimientos obtenidos de la experimentación, interpretación de resultados. (~1 página)
- **Referencias**: Lista de referencias bibliográficas utilizadas en el informe. (<½ página)

**Errores comunes, consideraciones**

- **Función de evaluación**: en el ejemplo está basada en la imagen, para Rastrigin se usa la fórmula.
- **Espacio de búsqueda**: en el ejemplo son los pixeles de la imagen, en Rastrigin es el dominio dado. En la implementación para Rastrigin se deben manejar ambos. Se aconseja **no mezclarlos**.
- **Usar la visualización**: comportamientos anormales pueden dar pistas sobre problemas en la implementación y orientar la parametrización.
- **Rol de la aleatoriedad**: UNA ejecución exitosa o no no es necesaria para caracterizar el algoritmo. Es necesario promediar varias ejecuciones (~30).