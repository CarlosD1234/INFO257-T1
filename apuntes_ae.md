Algoritmos de optimización bioinspirados: El modelo evolutivo

Diapositiva 2: La evolución de las Especies

Charles Darwin (1809-1882)

"El origen de las especies" (1859)

Viaje en el HMS Beagle, 1831-1836

Concepto: "La supervivencia del más apto"

Idea de evolución: Jean-Baptiste Lamarck (1744-1829)

Teoría alquímica basada en dos fuerzas:

"El poder de la vida"

"La influencia de las circunstancias"

Diapositiva 3: Genética y Evolución "Darwiniana"

Gregor Mendel (1822-1884)

Estudio de variación en plantas.

Modelo genético combinatorio:

Genes dominantes.

Genes recesivos.

El genotipo define las características.

Redescubierto por Hugo de Vries y Carl Correns en 1900.

Características de las plantas de guisante utilizadas por Gregor Mendel en sus experimentos de herencia:

|

| Semillas (Seeds) | Color de flor (Flower) | Vaina (Pod) | Tallo (Stem) |
| Forma: Redonda / Arrugada | Color: Blanca / Violeta-rojo | Forma: Llena / Constreñida entre semillas | Posición de inflorescencia: Axial / Terminal |
| Cotiledones: Amarillo / Verde |  | Color: Amarilla / Verde | Tamaño: Largo / Corto |

Diapositiva 4: Modelo Evolutivo (muy simplificado)

Los genes determinan las características de los individuos.

Las características determinan las opciones de los individuos:

Lucha por los recursos.

Lucha por las parejas.

"Supervivencia del más apto".

Evolución = Mejora continua = Optimización

(Nota visual de la diapositiva: Se muestra una comparación de la estructura ósea (Húmero, Radio, Cúbito, Metacarpos, Falanges) entre un humano, un gato, una ballena y un murciélago, demostrando el parentesco evolutivo).

Diapositiva 5: Evolución natural y artificial

Paralelo entre naturaleza y problemas de optimización:

| Natural | Artificial (Algoritmos) |
| Individuo | Solución |
| ADN del individuo | Conjunto de variables que constituyen la solución |
| Aptitud | Evaluación de la solución mediante una función objetivo |
| Población de individuos | Conjunto de soluciones |
| Competencia | Selección orientada hacia los mejores |
| Cruzamiento | Combinación de variables de soluciones |
| Mutación | Perturbación aleatoria de soluciones |

Diapositiva 6: Algoritmo Evolutivo

Propuesta de base: Algoritmo Genético

Propuesto por John Holland en 1975.

Ciclo del Algoritmo (Diagrama):

Inicialización (i): Se crea una población inicial (conjunto de soluciones X).

Evaluación: Se evalúa cada solución mediante la función de aptitud f(X).

¿Criterio de término? (?:)

Si se cumple -> Se entrega la solución final (*X*)*.

Si no se cumple -> Se continúa con el ciclo.

Selección (Se): Se eligen los mejores individuos.

Cruzamiento (Cr): Combinación entre individuos seleccionados.

Mutación (Mu): Alteración aleatoria en los individuos.

Evaluación: Se evalúa la aptitud f(X) de la nueva generación.

Reinserción (Re): La nueva generación reemplaza a la antigua y el ciclo se repite desde el paso 3.

Diapositiva 7: Aplicaciones de los EAs (Evolutionary Algorithms)

Principal atributo: Lograr soluciones no intuitivas.

Prueba de miles de soluciones generadas al azar.

Combinaciones de ellas.

Sin sesgo humano hacia "soluciones correctas".

Ejemplo: Diseño de antenas

Antena para la nave espacial ST5 de la NASA (2005).

Montado en satélites.

Considerado el primer objeto obtenido por evolución artificial usado en el espacio.

Diapositiva 8: Aplicaciones de los EAs (Criaturas Virtuales)

Evolved Virtual Creatures (Sims, 1994):

El proceso evolutivo mide el desempeño de criaturas en un ambiente virtual.

Existen varios trabajos inspirados en este:

Karl Sims, 1994 (evolución de criaturas en agua/tierra).

Geijtenbeek et al., SIGGRAPH Asia 2013 (evolución en bipedestación y velocidad objetivo).

Cheney, Clune, Lipson, GECCO 2013 (robots simulados hechos de celdas "voxel" blandas que evolucionan la habilidad de correr).

Diapositiva 9: Aplicaciones de los EAs (Diseño de Vehículos)

Evolución de diseños de vehículos:

El "ADN" del individuo determina su forma y estructura.

No existe una función de evaluación estática; la aptitud se calcula directamente interactuando en la simulación física (ej. distancia recorrida en un terreno irregular).

(Referencia visual: Simulador web Genetic Cars).

Diapositiva 10: Aplicaciones de los EAs (Diseño de Estructuras)

Diseño de arcos de puentes de celosía:

Integrado con el método de elementos finitos.

Determinar la forma del puente.

Minimizar el peso total de la estructura.

Asegurar que puede soportar la carga.

Evaluar restricciones de tensión y deformación máximas.

(Referencia: "Optimum design of steel truss arch bridges using a hybrid genetic algorithm", Jin Cheng, 2010).

Diapositiva 11: Aplicaciones de los EAs (Layout design industrial)

Layout design (Diseño de distribución):

Distribuir máquinas en líneas de producción.

Maximizar el uso del área disponible.

Disminuir costos de transporte entre estaciones.

Gran impacto en la reducción de costos operacionales.

(Referencia: "Plant Layout Optimization Using Evolutionary Algorithms", Kubalík et al.).

Diapositiva 12: Aplicaciones de los EAs (Wind farm layout)

Optimización de distribución de granjas eólicas:

Considerar la dirección predominante del viento (Rosa de los vientos).

Considerar el efecto aerodinámico entre generadores (estelas de viento).

Maximizar la eficiencia global del parque eólico.

(Referencia: "Wind farm layout optimization using self-informed genetic algorithm with information guided exploitation", Xinglong Ju et al.).

Diapositiva 13: Aplicaciones de los EAs (Bioinformática)

Construcción de árboles filogenéticos:

Reconstruir la historia evolutiva en base a la comparación de ADN de distintas especies.

Construir el árbol que maximice la parsimonia (es decir, la simpleza de la hipótesis de descendencia).

(Referencia: "A Genetic Algorithm for Maximum-Likelihood Phylogeny Inference Using Nucleotide Sequence Data", Paul O. Lewis).

Diapositiva 14: Aplicaciones de los EAs (Planificación Minera)

Planificación de extracción minera:

En base a un modelo minero de bloques geológicos.

Definir las rutas de acarreo, puntos óptimos de hundimiento y la secuencia temporal de extracción.

Objetivo: Maximizar el Valor Actual Neto (VAN) de la extracción minera.

Se deben considerar restricciones geomecánicas muy complejas, como la zona de subsidencia (hundimiento del terreno).

Diapositiva 15: Proyección de los EAs (Neuroevolution - Intro)

Neuroevolution (Neuroevolución):

Evolucionar la arquitectura de redes neuronales artificiales utilizando algoritmos evolutivos, imitando cómo se desarrollaron los cerebros biológicos.

(Referencia: Artículo "Neuroevolution: A different kind of deep learning", Kenneth O. Stanley, O'Reilly Radar).

Diapositiva 16: Proyección de los EAs (Evolución de la IA)

Impacto Científico:

El campo avanza al punto donde "La Inteligencia Artificial está evolucionando por sí misma".

Uso de conceptos Darwinianos para construir programas de IA que mejoran generación tras generación sin intervención humana.

(Referencia: Artículo de Science Magazine "Artificial intelligence is evolving all by itself", Edd Gent, 2020).

Diapositiva 17: Proyección de los EAs (Búsqueda de Arquitecturas)

Arquitecturas de Redes Neuronales (AutoML):

Crece el interés en redes neuronales profundas (Deep Learning).

Surge la fuerte necesidad de automatizar el diseño de su estructura, ya que diseñarlas a mano toma años de investigación.

Objetivo: Maximizar exactitud (accuracy) de las redes con algoritmos evolutivos masivos.

(Referencia: Google AI Blog - "Using Evolutionary AutoML to Discover Neural Network Architectures").

Diapositiva 18: Proyección de los EAs (El caso de Uber AI Labs)

Uber Engineering y la Neuroevolución profunda:

Neuroevolution: Optimización de redes neuronales (DNNs) mediante algoritmos evolutivos en lugar (o además) de la tradicional "propagación hacia atrás" (Backpropagation).

Descubrimiento de que un Algoritmo Genético (GA) extremadamente simple puede entrenar redes convolucionales profundas con más de 4 millones de parámetros para jugar juegos de Atari, rivalizando con los métodos de Aprendizaje por Refuerzo tradicionales.

(Referencia: "Welcoming the Era of Deep Neuroevolution / Genetic algorithms as a competitive alternative for training deep neural networks", Uber AI Labs).



Algoritmos de optimización bioinspirados: Algoritmos Evolutivos

Diapositiva 2: Algoritmo Evolutivo

Ciclo del Algoritmo (Diagrama):

Inicialización (i): Creación de la población inicial de individuos.

Evaluación (f(x)): Se evalúa la aptitud o fitness de la población actual.

¿Criterio de término? (?): * Si se cumple -> Se entrega la solución final (X)*.

Si no se cumple -> Continúa el ciclo.

Selección (Se): Se eligen individuos para la reproducción.

Cruzamiento (Cr): Se combinan los individuos seleccionados.

Mutación (Mu): Se aplican cambios aleatorios a los nuevos individuos.

Evaluación (f(X)): Se evalúa la nueva generación generada.

Reinserción (Re): La nueva generación se integra reemplazando a la antigua y el ciclo vuelve a la pregunta del criterio de término.

Diapositiva 3: ¿Cómo funciona un Algoritmo Evolutivo?

Ejemplo: Optimización de una función en 2D.

(Referencia visual: Se muestra el gráfico 3D de una función matemática compleja con múltiples picos y valles (óptimos locales), ilustrando la dificultad del espacio de búsqueda para un algoritmo tradicional).

Función de ejemplo (Z. Michalewicz): $f(x_1,x_2) = 21.5 + x_1 \sin(4\pi x_1) + x_2 \sin(20\pi x_2)$

Diapositiva 4: Representación (encoding) del ejemplo

Usando Algoritmos Genéticos (AG), propuestos por John Holland (1975).

Rango de variables de ejemplo: $3.0 \le x_1 \le 12.1$ y $4.1 \le x_2 \le 5.8$

Representación clásica de los Algoritmos Genéticos: strings binarios

Cada variable será representada por un string de bits.

000...0 corresponde al valor mínimo, 111...1 al valor máximo.

Los Algoritmos Evolutivos (Clase más general) admiten otras representaciones:

Reales

Permutaciones

Árboles

Cualquier otra estructura de datos, variable o no.

Se determina un tamaño de población (e.g., 20) y se genera una población de individuos inicializados al azar.

Diapositiva 5: Evaluación y selección

Cada individuo es decodificado y evaluado mediante la función de fitness.

Se crea una nueva población (ej. de 20 individuos) seleccionando con preferencia a los mejores.

Idea central: los individuos más aptos se reproducen con mayor frecuencia que el resto.

Método base: Roulette wheel (Ruleta)

La selección es proporcional a su fitness (el individuo más apto tiene la porción más grande de la ruleta).

Nota: Algunos pueden quedar seleccionados más de una vez, y los menos aptos tienen menos probabilidades (porción más pequeña).

Mejor opción moderna: Tournament selection (Selección por torneo). Se eligen K cromosomas al azar y se selecciona al mejor de ese grupo como padre.

Diapositiva 6: Cruzamiento (Crossover)

Se elige con alguna probabilidad distintos individuos para mezclar sus cromosomas.

Típicamente probabilidad entre 0.2 y 0.8.

Cruzamiento "de un punto" (One point crossover):

Una vez elegidos los dos padres, se elige al azar un punto de corte.

Se intercambian las dos mitades para producir dos hijos.

Ejemplo visual: Padre 1 (A B | C D E) + Padre 2 (F G | H I J) $\rightarrow$ Hijo 1 (A B | H I J) e Hijo 2 (F G | C D E).

¡Existen otros métodos!, dependiendo de la codificación y el problema.

Los hijos reemplazan a los padres en la población (la secundaria).

Diapositiva 7: Mutación

Una vez cruzados, comienza la mutación.

Es un cambio con probabilidad baja (aprox. ~0.01).

Para la codificación de string de bits, se acostumbra a decidir bit a bit:

Si un número al azar es inferior a la probabilidad de mutación, se hace un swap del bit ($0 \rightarrow 1$ ó $1 \rightarrow 0$).

Ejemplo visual: 01101010110 muta a 01101000110.

¡Existen otras formas de mutar!, dependiendo de la codificación y el problema.

Diapositiva 8: Reinserción

En general existen dos estrategias de reinserción de la nueva población:

Generacional:

Los individuos nuevos "pisan" (reemplazan) a los antiguos.

La población se renueva constantemente.

Exploración relativamente alta.

Steady State (Estado Estacionario):

Los individuos hijos deben competir con sus padres para entrar a la población.

A medida que avanza la búsqueda, aumenta la explotación.

Riesgo de eventual atasco en la búsqueda (convergencia prematura).

Diapositiva 9: Exploración vs Explotación

Existen dos formas de buscar en el espacio de soluciones, y son distintas estrategias (ambas necesarias para el éxito del algoritmo):

Buscar ampliamente $\rightarrow$ Explorar / diversificar.

Refinar búsqueda $\rightarrow$ Explotar / intensificar.

Diapositiva 10: Lidiando con el azar

La aleatoriedad propia del algoritmo dificulta el debug y la evaluación del mismo.

Tips:

Fijar o registrar la semilla aleatoria (seed) para poder reproducir errores.

Evitar usar una misma semilla siempre (para no evaluar solo un caso puntual).

Usar siempre un gráfico de convergencia para entender el comportamiento.

(Se muestra un gráfico estándar de convergencia iterativa donde se plotea el mejor "best" y el promedio "average" de la población a lo largo de las generaciones).

Diapositiva 11: Gráficas típicas de convergencia - Exceso de explotación

(Se muestra un gráfico donde las curvas caen muy rápido y se vuelven planas horizontalmente en etapas muy tempranas, cerca de la iteración 20).

Efectos del exceso de explotación:

Los individuos se concentran rápidamente en un óptimo local.

La población es incapaz de salir, quedando "atrapada".

El proceso de selección comienza a privilegiar a los mejores de manera muy agresiva, y la diversidad entre ellos disminuye.

Se pierde tiempo de cómputo sobreexplotando la misma zona del espacio de búsqueda.

Conclusión: Se produce convergencia prematura.

Diapositiva 12: Gráficas típicas de convergencia - Exceso de exploración

(Se muestra un gráfico donde las curvas no bajan significativamente, se mantienen ruidosas, horizontales y separadas a lo largo de todas las generaciones).

Efectos del exceso de exploración:

Los individuos vagan eternamente por el espacio de búsqueda.

La población es incapaz de concentrarse en las zonas de mayor fitness.

Las buenas soluciones que se llegan a encontrar son producto puramente del azar.

Diapositiva 13: Gráficas típicas de convergencia - Búsqueda balanceada

(Se muestra un gráfico donde la curva baja gradualmente asintótica hacia el final. El "average" sigue de cerca al "best" marcando una mejora constante).

Efectos de un buen balance entre exploración y explotación:

La población se mantiene diversa a lo largo de la búsqueda.

Abierta a encontrar nuevos óptimos locales de mejor calidad.

Abierta a identificar nuevas zonas de alto fitness en problemas dinámicos.

Se observa una mejora continua (asintótica hacia el final) del mejor individuo.

Buena utilización de los recursos computacionales.

Estrategia útil: explorar mucho al principio y explotar más hacia el final de la ejecución.