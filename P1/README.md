# Optimización de Función Rastrigin: PSO vs AE

Este proyecto implementa dos algoritmos metaheurísticos para la optimización de funciones continuas: **Optimización por Enjambre de Partículas (PSO)** y **Algoritmos Evolutivos (AE)**. El objetivo es minimizar la función **Rastrigin** en un dominio bidimensional (-3 a 7).

## 1. 🧠 Análisis del propósito

El proyecto resuelve el problema de la optimización global en superficies con múltiples mínimos locales (como Rastrigin). Proporciona una plataforma para:

- Visualizar la convergencia de poblaciones/enjambres en tiempo real.
- Recolectar datos de rendimiento (fitness promedio y mejor fitness) para estudios comparativos.
- Ejecutar pruebas masivas mediante un "modo estudio" sin interfaz gráfica.

Como resultado, genera archivos CSV detallados con el historial de cada ejecución y animaciones interactivas de la evolución del algoritmo.

---

## 📁 Estructura del proyecto

```bash
.
├── README.md           # Documentación del proyecto
├── requirements.txt    # Dependencias de Python
├── data/               # Resultados de las ejecuciones
│   ├── data_pso.csv    # Historial de PSO
│   └── data_ae.csv     # Historial de AE
├── src/                # Código fuente
│   ├── pso.py          # Implementación y animación de PSO
│   ├── ae.py           # Implementación y animación de AE
│   └── plot.py         # Utilidades de graficación y Heatmap
└── venv/               # Entorno virtual (generado localmente)
```

- **src/**: Contiene los módulos principales de lógica y visualización.
- **data/**: Carpeta generada automáticamente donde se guardan los logs de las ejecuciones.
- **requirements.txt**: Lista de librerías necesarias (numpy, matplotlib, plotly).

---

## ⚠️ Problemas / posibles errores

- **Dependencias de GUI**: Las animaciones de Matplotlib requieren un entorno con soporte para ventanas (Desktop). No funcionarán en entornos headless (como servidores SSH sin X11) a menos que se use el modo estudio (`animation=False`).
- **Codificación de requirements.txt**: El archivo generado via PowerShell puede estar en UTF-16LE, lo que podría causar problemas en algunos editores de texto si no se detecta correctamente.
- **Bloqueo de archivos**: Evitar tener abiertos los archivos CSV en Excel mientras se corre el algoritmo, ya que el script intenta hacer *append* y podría fallar por permisos de escritura.

---

## ⚙️ Setup

### 4.1 Instalación

Sigue estos comandos en tu terminal (Windows PowerShell preferido):

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
.\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## ▶️ Ejecución

Desde la carpeta raíz del proyecto, puedes ejecutar ambos algoritmos:

### Ejecutar PSO (Particle Swarm Optimization)

```bash
python src/pso.py
```

### Ejecutar AE (Algoritmo Evolutivo)

```bash
python src/ae.py
```

> **Tip**: Puedes modificar las banderas `MOSTRAR_ANIMACION` y `GUARDAR_DATOS` al final de cada archivo `.py` para cambiar el comportamiento del script.

---

## ⚠️ Disclaimer

* Fecha de generación: 2026-04-03

Esta documentación fue generada automáticamente por una IA utilizando un workflow.

---

**Revisor:** Renato Atencio
**Revisado:** [x]
