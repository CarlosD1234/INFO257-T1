# Optimización de Función Rastrigin: PSO vs AE

Este proyecto implementa dos de los algoritmos metaheurísticos más populares para la optimización de funciones continuas: **Optimización por Enjambre de Partículas (PSO)** y **Algoritmos Evolutivos (AE)**. El objetivo es encontrar el mínimo global de la función **Rastrigin** en un dominio de 2 dimensiones (entre -3 y 7).

## Características principales

*   **Visualización Dual interactiva**: Ambos scripts incluyen una animación en tiempo real con Matplotlib que muestra el mapa de calor de la función y la curva de convergencia simultáneamente.
*   **Modo Estudio (Sin Animación)**: Opción para ejecutar los algoritmos a máxima velocidad sin interfaz gráfica, ideal para la recolección masiva de datos.
*   **Exportación automática a CSV**: Los resultados de cada iteración se guardan de forma persistente en la carpeta `data/` con marcas de tiempo, facilitando su análisis posterior en Excel o Python.
*   **Aceleración por GPU**: El archivo `plot.py` incluye funciones experimentales para pre-visualizar la superficie usando **Plotly (WebGL)** aprovechando tu tarjeta gráfica.

## Requisitos Previos

*   Python 3.10 o superior.
*   Escritorio de Windows (para las animaciones de Matplotlib).

## Instalación y Configuración

Sigue estos pasos para configurar el entorno virtual e instalar las dependencias necesarias:

```powershell
# 1. Clonar el repositorio y entrar a la carpeta del proyecto
cd INFO257-T1/P1

# 2. Crear el entorno virtual (venv)
python -m venv venv

# 3. Activar el entorno virtual
.\venv\Scripts\activate

# 4. Instalar dependencias
pip install -r requirements.txt
```

> **Nota**: Si tienes problemas instalando `plotly` o `matplotlib`, asegúrate de tener una versión reciente de `pip`.

## Cómo ejecutar

### Optimización por Enjambre de Partículas (PSO)
```powershell
python .\src\pso.py
```

### Algoritmo Evolutivo (AE)
```powershell
python .\src\ae.py
```

## Configuración de Banderas (Modo Estudio)

Al final de los archivos `src/pso.py` y `src/ae.py`, puedes configurar las siguientes variables para tu estudio:

```python
# pso.py o ae.py
if __name__ == "__main__":
    # ...
    MOSTRAR_ANIMACION = True  # Cambia a False para ejecución rápida
    GUARDAR_DATOS = True      # Cambia a False para no generar archivos CSV
    # ...
```

## Estructura de Datos (CSV)

Los archivos de salida se encuentran en la carpeta `data/`:
*   `data_pso.csv`: Resultados de las ejecuciones de PSO.
*   `data_ae.csv`: Resultados de las ejecuciones de AE.

**Columnas registradas:**
1.  **ejecucion**: Fecha y hora de inicio de la prueba.
2.  **iteracion**: Número de generación/ciclo.
3.  **gbest_gen**: Mejor fitness encontrado en *esa* generación específica.
4.  **average**: Fitness promedio de toda la población en esa generación.
5.  **gbest_historico**: El mejor valor absoluto encontrado desde el inicio de la ejecución.

---
Proyecto desarrollado para el curso de Inteligencia Artificial - INFO257.
