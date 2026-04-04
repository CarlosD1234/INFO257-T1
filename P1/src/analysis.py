import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Configuración de archivos a analizar (Añade o quita según necesites)
FILES_TO_ANALYZE = [
    {'path': 'data/data_pso.csv', 'label': 'PSO'},
    # {'path': 'data/data_ae.csv', 'label': 'AE'}, # Comentado por ahora
]

# Asegurar carpeta de resultados
OS_RESULTS_DIR = 'results'
os.makedirs(OS_RESULTS_DIR, exist_ok=True)

def load_and_clean_data(filepath, label):
    """Carga el CSV y limpia posibles cabeceras duplicadas."""
    if not os.path.exists(filepath):
        print(f"Aviso: No se encontró el archivo {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        # Filtrar filas que puedan ser cabeceras repetidas
        if 'iteracion' in df.columns:
            df = df[df['iteracion'] != 'iteracion'].copy()
            df['iteracion'] = pd.to_numeric(df['iteracion'])
            df['gbest_historico'] = pd.to_numeric(df['gbest_historico'])
            df['average'] = pd.to_numeric(df['average'])
            df['algoritmo'] = label
            return df
    except Exception as e:
        print(f"Error procesando {filepath}: {e}")
    return None

def analyze():
    all_dfs = []
    
    print("Cargando archivos seleccionados...")
    for config in FILES_TO_ANALYZE:
        df = load_and_clean_data(config['path'], config['label'])
        if df is not None:
            all_dfs.append(df)
    
    if not all_dfs:
        print("No se encontraron datos para analizar.")
        return

    # Combinar todos los datasets encontrados
    df_combined = pd.concat(all_dfs, ignore_index=True)

    # 1. Gráfico de Convergencia
    print("Generando gráfico de convergencia...")
    plt.figure()
    
    # Añadir un pequeño epsilon para que el log(0) no corte el gráfico
    plot_df = df_combined.copy()
    plot_df['gbest_historico_plot'] = plot_df['gbest_historico'] + 1e-15
    
    sns.lineplot(data=plot_df, x='iteracion', y='gbest_historico_plot', hue='algoritmo', errorbar='sd')
    plt.yscale('log')
    plt.title('Convergencia del Mejor Fitness (Promedio +/- Desviación)')
    plt.xlabel('Generación / Iteración')
    plt.ylabel('Mejor Fitness Histórico (Log)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.savefig(os.path.join(OS_RESULTS_DIR, 'convergencia.png'))
    plt.close()

    # 2. Boxplot de Desempeño Final
    print("Generando boxplot de resultados finales...")
    # Tomar la última iteración de cada ejecución única
    df_final = df_combined.sort_values('iteracion').groupby(['algoritmo', 'ejecucion']).tail(1)
    
    plt.figure(figsize=(8, 6))
    # Usar hue=algoritmo para evitar el aviso de deprecación
    sns.boxplot(data=df_final, x='algoritmo', y='gbest_historico', hue='algoritmo', palette='Set2', legend=False)
    plt.title('Distribución del Fitness Final Alcanzado')
    plt.ylabel('Fitness Final')
    plt.savefig(os.path.join(OS_RESULTS_DIR, 'boxplot_final.png'))
    plt.close()

    # 3. Reporte en Consola
    print("\n" + "="*45)
    print(f"{'ALGORITMO':<12} | {'MEJOR':<10} | {'PROMEDIO':<10} | {'STD':<8}")
    print("-" * 45)
    
    stats = df_final.groupby('algoritmo')['gbest_historico'].agg(['min', 'mean', 'std']).round(6)
    for idx, row in stats.iterrows():
        print(f"{idx:<12} | {row['min']:<10} | {row['mean']:<10} | {row['std']:<8}")
    print("="*45)
    
    print(f"\nAnálisis completado. Gráficos en: '{OS_RESULTS_DIR}/'")

if __name__ == "__main__":
    analyze()
