"""
Visualización de Resultados PSO - Función Rastrigin
=====================================================
Script para generar gráficos profesionales a partir de los datos
de múltiples ejecuciones del algoritmo PSO.

Autor: Generado automáticamente
Fecha: 2026-04-03
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuración de estilo global
# ============================================================
# ============================================================
# Configuración de estilo global (Tema Claro)
# ============================================================
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': 'black',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.color': 'black',
    'legend.facecolor': 'white',
    'legend.edgecolor': '#cccccc',
    'legend.fontsize': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans'],
    'grid.color': '#e0e0e0',
    'grid.alpha': 0.8,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
})

# Paleta de colores clásica
COLORS = {
    'best':       'red',       # Rojo para el mejor gbest
    'average':    'blue',      # Azul para el promedio
    'band':       '#58a6ff',   # Banda de confianza (azul suave)
    'accent1':    'green',     # Verde para otros indicadores
    'grid':       '#e0e0e0',
}

# Gradiente para múltiples ejecuciones
EXEC_CMAP = plt.cm.plasma

# ============================================================
# Configuración de Ejecución
# ============================================================
# Cambiar a 'PSO' o 'AE' según se desee visualizar
TIPO_ALGORITMO = 'AE' 

CONFIG = {
    'PSO': {
        'nombre': 'PSO (Particle Swarm Optimization)',
        'label': 'PSO',
        'data': 'data/data_pso.csv',
        'prefix': 'pso_',
        'watermark': 'PSO - Rastrigin'
    },
    'AE': {
        'nombre': 'AE (Algoritmo Evolutivo)',
        'label': 'AE',
        'data': 'data/data_ae.csv',
        'prefix': 'ae_',
        'watermark': 'AE - Rastrigin'
    }
}

# Selección actual
SEL = CONFIG[TIPO_ALGORITMO]
data_path = SEL['data']
watermark_text = SEL['watermark']
file_prefix = SEL['prefix']
algo_name = SEL['nombre']
algo_label = SEL['label']

def load_data(filepath=data_path):
    """Carga y preprocesa los datos del CSV."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    # Crear un ID numérico para cada ejecución basado en el timestamp
    timestamps = df['ejecucion'].unique()
    ts_to_id = {ts: i + 1 for i, ts in enumerate(timestamps)}
    df['run_id'] = df['ejecucion'].map(ts_to_id)
    
    print(f"📊 Datos cargados: {len(timestamps)} ejecuciones, "
          f"{df['iteracion'].max()} iteraciones cada una")
    print(f"   Rango de fitness: [{df['gbest_historico'].min():.6e}, "
          f"{df['gbest_historico'].max():.4f}]")
    
    return df

def add_watermark(ax, text = watermark_text):
    """Añade una marca de agua sutil al gráfico."""
    ax.text(0.98, 0.02, text, transform=ax.transAxes,
            fontsize=8, color='#999999', ha='right', va='bottom',
            fontstyle='italic', alpha=0.5)

def plot_01_best_vs_average(df, save_path):
    """
    Gráfico 1: Comparativa del Mejor Histórico vs Promedio del Enjambre/Población.
    Muestra la media ± desviación estándar across todas las ejecuciones.
    """
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle(f'{algo_label} — Mejor Histórico vs Promedio',
                 fontsize=16, fontweight='bold', color='black', y=1.02)
    
    # Agrupar por iteración y calcular estadísticas
    stats = df.groupby('iteracion').agg(
        gbest_mean=('gbest_historico', 'mean'),
        gbest_std=('gbest_historico', 'std'),
        gbest_min=('gbest_historico', 'min'),
        gbest_max=('gbest_historico', 'max'),
        avg_mean=('average', 'mean'),
        avg_std=('average', 'std'),
        avg_min=('average', 'min'),
        avg_max=('average', 'max'),
    ).reset_index()
    
    iters = stats['iteracion']
    
    # --- Panel izquierdo: Escala lineal ---
    ax = axes
    
    # Banda de confianza para promedio
    ax.fill_between(iters, stats['avg_mean'] - stats['avg_std'],
                    stats['avg_mean'] + stats['avg_std'],
                    alpha=0.15, color=COLORS['average'], linewidth=0)
    # Banda de confianza para mejor
    ax.fill_between(iters, stats['gbest_mean'] - stats['gbest_std'],
                    stats['gbest_mean'] + stats['gbest_std'],
                    alpha=0.2, color=COLORS['best'], linewidth=0)
    
    # Líneas principales
    ax.plot(iters, stats['avg_mean'], color=COLORS['average'],
            linewidth=1.5, label='Promedio general', alpha=0.8, linestyle='--')
    ax.plot(iters, stats['gbest_mean'], color=COLORS['best'],
            linewidth=2, label='Mejor histórico', alpha=1.0)
    
    ax.set_xlabel('Iteración')
    ax.set_ylabel('Fitness (Rastrigin)')
    ax.set_title('Escala Lineal', fontsize=12, pad=10)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 200)
    ax.set_ylim(bottom=0)
    add_watermark(ax)
    
    
    plt.tight_layout()
    filename = f'{file_prefix}01_mejor_vs_promedio.png'
    fig.savefig(save_path + filename)
    plt.close(fig)
    print(f"✅ Gráfico 1 guardado: {filename}")

def plot_02_convergence_all_runs(df, save_path):
    """
    Gráfico 2: Curvas de convergencia individuales de todas las ejecuciones.
    Cada ejecución se muestra como una línea con transparencia.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n_runs = df['run_id'].nunique()
    
    for i, run_id in enumerate(sorted(df['run_id'].unique())):
        run_data = df[df['run_id'] == run_id]
        color = EXEC_CMAP(i / max(n_runs - 1, 1))
        ax.semilogy(run_data['iteracion'], run_data['gbest_historico'].clip(lower=1e-16),
                    color=color, alpha=0.45, linewidth=0.8)
    
    # Superponer la media
    mean_gbest = df.groupby('iteracion')['gbest_historico'].mean().clip(lower=1e-16)
    ax.semilogy(mean_gbest.index, mean_gbest.values, color='black',
                linewidth=2.5, label='Media de ejecuciones', zorder=10, alpha=0.95)
    
    ax.set_xlabel('Iteración')
    ax.set_ylabel('Mejor Fitness Histórico (log)')
    ax.set_title(f'Convergencia de {algo_label} — Todas las Ejecuciones',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(1, 200)
    
    # Barra de color para ejecuciones
    sm = plt.cm.ScalarMappable(cmap=EXEC_CMAP, norm=plt.Normalize(1, n_runs))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Nº de Ejecución', color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')
    
    add_watermark(ax)
    plt.tight_layout()
    filename = f'{file_prefix}02_convergencia_todas.png'
    fig.savefig(save_path + filename)
    plt.close(fig)
    print(f"✅ Gráfico 2 guardado: {filename}")

def plot_05_convergence_speed(df, save_path):
    """
    Gráfico 5: Velocidad de convergencia - ¿En qué iteración cada ejecución
    alcanza ciertos umbrales de fitness?
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{algo_label} — Velocidad de Convergencia',
                 fontsize=16, fontweight='bold', color='black', y=1.02)
    
    thresholds = [1.0, 0.1, 0.01, 1e-3, 1e-6, 1e-10, 1e-14]
    threshold_labels = ['1', '0.1', '0.01', '10⁻³', '10⁻⁶', '10⁻¹⁰', '10⁻¹⁴']
    
    convergence_iters = {t: [] for t in thresholds}
    
    for run_id in sorted(df['run_id'].unique()):
        run_data = df[df['run_id'] == run_id].sort_values('iteracion')
        for t in thresholds:
            reached = run_data[run_data['gbest_historico'] <= t]
            if not reached.empty:
                convergence_iters[t].append(reached['iteracion'].iloc[0])
            else:
                convergence_iters[t].append(np.nan)
    
    # --- Panel izquierdo: Barras con % de ejecuciones que alcanzan cada umbral ---
    ax = axes[0]
    success_rates = []
    mean_iters = []
    for t in thresholds:
        vals = convergence_iters[t]
        valid = [v for v in vals if not np.isnan(v)]
        success_rates.append(len(valid) / len(vals) * 100)
        mean_iters.append(np.mean(valid) if valid else np.nan)
    
    bars = ax.barh(range(len(thresholds)), success_rates,
                   color=[EXEC_CMAP(i / (len(thresholds) - 1)) for i in range(len(thresholds))],
                   alpha=0.85, edgecolor='#30363d', linewidth=0.5, height=0.6)
    
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        if rate > 0:
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'{rate:.0f}%', va='center', ha='left', color='black',
                    fontsize=10, fontweight='bold')
    
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f'f ≤ {lbl}' for lbl in threshold_labels])
    ax.set_xlabel('% de Ejecuciones Exitosas')
    ax.set_title('Tasa de Éxito por Umbral', fontsize=12, pad=10)
    ax.set_xlim(0, 115)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    add_watermark(ax)
    
    # --- Panel derecho: Box plot de iteración de convergencia ---
    ax = axes[1]
    box_data = []
    box_labels_valid = []
    box_colors = []
    
    for i, (t, lbl) in enumerate(zip(thresholds, threshold_labels)):
        vals = [v for v in convergence_iters[t] if not np.isnan(v)]
        if len(vals) >= 2:
            box_data.append(vals)
            box_labels_valid.append(f'f ≤ {lbl}')
            box_colors.append(EXEC_CMAP(i / (len(thresholds) - 1)))
    
    if box_data:
        bplot = ax.boxplot(box_data, labels=box_labels_valid, patch_artist=True,
                           widths=0.5, vert=True,
                           medianprops=dict(color='white', linewidth=2),
                           whiskerprops=dict(color='#8b949e'),
                           capprops=dict(color='#8b949e'),
                           flierprops=dict(markerfacecolor='#ff7b72', marker='D',
                                          markersize=5, alpha=0.7))
        
        for patch, color in zip(bplot['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('black')
        
        ax.set_xticklabels(box_labels_valid, rotation=30, ha='right')
    
    ax.set_ylabel('Iteración de Convergencia')
    ax.set_title('Iteración para Alcanzar Cada Umbral', fontsize=12, pad=10)
    ax.grid(True, alpha=0.3, axis='y')
    add_watermark(ax)
    
    plt.tight_layout()
    filename = f'{file_prefix}05_velocidad_convergencia.png'
    fig.savefig(save_path + filename)
    plt.close(fig)
    print(f"✅ Gráfico 5 guardado: {filename}")

def plot_07_summary_dashboard(df, save_path):
    """
    Gráfico 7: Tabla resumen con métricas clave (Solo Tabla).
    """
    # Estadísticas globales
    max_iter = df['iteracion'].max()
    final_data = df[df['iteracion'] == max_iter]
    n_runs = df['run_id'].nunique()
    
    convergence_to_zero = []
    for run_id in df['run_id'].unique():
        run = df[df['run_id'] == run_id]
        zero_iter = run[run['gbest_historico'] == 0.0]
        if not zero_iter.empty:
            convergence_to_zero.append(zero_iter['iteracion'].iloc[0])
    
    # Preparar datos para la tabla (Solo Éxito y Convergencia)
    table_data = [
        ["⚡ Éxito (conv. a 0)", f"{len(convergence_to_zero)}/{n_runs}"]
    ]
    
    if convergence_to_zero:
        table_data.append(["🕒 Media iter. conv.", f"{np.mean(convergence_to_zero):.0f}"])
    else:
        table_data.append(["🕒 Media iter. conv.", "N/A"])

    # Crear figura y eje
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis('off')
    
    # Crear la tabla
    table = ax.table(
        cellText=table_data,
        colLabels=["Métrica", "Valor"],
        loc='center',
        cellLoc='left',
        colWidths=[0.55, 0.4]
    )
    
    # Estilo de la tabla
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.2) # Ancho, Alto
    
    # Colores y fuentes
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(1.0)
        if row == 0:  # Encabezados
            cell.set_text_props(weight='bold', color='black', fontsize=12)
            cell.set_facecolor('#f2f2f2')
        else:  # Celdas de datos
            cell.set_facecolor('white')
            cell.set_text_props(color='#333333')
            if col == 1:  # Columna de valores con fuente mono
                cell.set_text_props(fontfamily='monospace')

    ax.set_title(f'Resumen de Métricas {algo_label}', fontsize=16, fontweight='bold', pad=30, color='black')
    
    plt.tight_layout()
    filename = f'{file_prefix}07_dashboard_metricas.png'
    fig.savefig(save_path + filename)
    plt.close(fig)
    print(f"✅ Gráfico 7 guardado: {filename}")
   

# ============================================================
# Ejecución principal
# ============================================================
if __name__ == '__main__':
    save_path = "data/graficos/"
    
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    print("=" * 60)
    print(f"  🔬 Visualización de Resultados: {algo_name}")
    print("=" * 60)
    print()
    
    df = load_data(data_path)
    print()
    
    print(f"📈 Generando gráficos para {algo_label}...")
    print("-" * 40)
    
    plot_01_best_vs_average(df, save_path)
    plot_02_convergence_all_runs(df, save_path)
    plot_05_convergence_speed(df, save_path)
    plot_07_summary_dashboard(df, save_path)
    
    print()
    print("=" * 60)
    print(f"  ✨ ¡Gráficos de {algo_label} generados exitosamente!")
    print("=" * 60)
