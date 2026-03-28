"""
Genera graficos comparativos EA vs PSO a partir de los CSVs generados.
Requiere: pip install matplotlib pandas numpy
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results"


def load_data():
    ea_hist = pd.read_csv(os.path.join(RESULTS_DIR, "ea_history.csv"))
    pso_hist = pd.read_csv(os.path.join(RESULTS_DIR, "pso_history.csv"))
    ea_sum = pd.read_csv(os.path.join(RESULTS_DIR, "ea_summary.csv"))
    pso_sum = pd.read_csv(os.path.join(RESULTS_DIR, "pso_summary.csv"))
    return ea_hist, pso_hist, ea_sum, pso_sum


def plot_convergence(ea_hist, pso_hist):
    """Curva de convergencia usando MEDIANA + rango intercuartilico (mas robusto que media)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, hist, color in [("EA", ea_hist, "#e74c3c"), ("PSO", pso_hist, "#3498db")]:
        grouped = hist.groupby("generation")["best_fitness"]
        median = grouped.median()
        q25 = grouped.quantile(0.25)
        q75 = grouped.quantile(0.75)
        ax.plot(median.index, median.values, label=f"{label} (mediana)", color=color, linewidth=2)
        ax.fill_between(median.index, q25.values, q75.values, alpha=0.2, color=color,
                        label=f"{label} (Q1–Q3)")

    ax.set_xlabel("Generacion / Iteracion", fontsize=12)
    ax.set_ylabel("Mejor Fitness (mediana de 30 runs)", fontsize=12)
    ax.set_title("Curvas de Convergencia — EA vs PSO", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "convergencia.png"), dpi=150)
    print("Guardado: results/convergencia.png")
    plt.close(fig)


def plot_boxplot(ea_sum, pso_sum):
    """Violin plot + strip plot para mostrar la distribucion real."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1, 1]})

    # Panel izquierdo: escala lineal
    ax = axes[0]
    data = [ea_sum["final_best_fitness"].values, pso_sum["final_best_fitness"].values]
    parts = ax.violinplot(data, positions=[1, 2], showmedians=True, showextrema=True)
    parts["bodies"][0].set_facecolor("#e74c3c")
    parts["bodies"][0].set_alpha(0.4)
    parts["bodies"][1].set_facecolor("#3498db")
    parts["bodies"][1].set_alpha(0.4)
    # Strip plot (puntos individuales con jitter)
    for i, (d, color) in enumerate(zip(data, ["#e74c3c", "#3498db"])):
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d, c=color, s=25, alpha=0.7,
                   edgecolors="black", linewidths=0.3, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["EA", "PSO"])
    ax.set_ylabel("Mejor Fitness Final", fontsize=12)
    ax.set_title("Escala lineal", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel derecho: escala log (para ver detalle cerca de 0)
    ax = axes[1]
    # Agregar epsilon para evitar log(0)
    eps = 1e-16
    data_log = [np.maximum(d, eps) for d in data]
    for i, (d, color, label) in enumerate(zip(data_log, ["#e74c3c", "#3498db"], ["EA", "PSO"])):
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d, c=color, s=25, alpha=0.7,
                   edgecolors="black", linewidths=0.3, zorder=3, label=label)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["EA", "PSO"])
    ax.set_ylabel("Mejor Fitness Final (log)", fontsize=12)
    ax.set_title("Escala logaritmica", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=10)

    fig.suptitle("Distribucion del Mejor Fitness Final — EA vs PSO", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "boxplot.png"), dpi=150)
    print("Guardado: results/boxplot.png")
    plt.close(fig)


def plot_final_positions(ea_sum, pso_sum):
    """Scatter con jitter para que los puntos superpuestos sean visibles."""
    fig, ax = plt.subplots(figsize=(7, 7))

    rng = np.random.default_rng(42)
    jitter_scale = 0.15  # ruido visual para separar puntos superpuestos

    ea_x = ea_sum["final_best_x"].values + rng.uniform(-jitter_scale, jitter_scale, len(ea_sum))
    ea_y = ea_sum["final_best_y"].values + rng.uniform(-jitter_scale, jitter_scale, len(ea_sum))
    pso_x = pso_sum["final_best_x"].values + rng.uniform(-jitter_scale, jitter_scale, len(pso_sum))
    pso_y = pso_sum["final_best_y"].values + rng.uniform(-jitter_scale, jitter_scale, len(pso_sum))

    ax.scatter(pso_x, pso_y, c="#3498db", label=f"PSO (n={len(pso_sum)})",
               s=50, alpha=0.7, edgecolors="black", linewidths=0.5, zorder=3)
    ax.scatter(ea_x, ea_y, c="#e74c3c", label=f"EA (n={len(ea_sum)})",
               s=50, alpha=0.7, edgecolors="black", linewidths=0.5, zorder=4)

    # Marcar optimo global (0, 0)
    ax.scatter([0], [0], c="gold", s=250, marker="*", edgecolors="black",
               linewidths=1, zorder=5, label="Optimo global (0,0)")

    # Marcar minimos locales mas cercanos de Rastrigin
    for lx, ly in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
        ax.scatter([lx], [ly], c="gray", s=30, marker="x", alpha=0.5, zorder=2)
    ax.scatter([], [], c="gray", marker="x", label="Minimos locales", alpha=0.5)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Posiciones Finales del Mejor Individuo (con jitter visual)", fontsize=14)
    ax.set_xlim(-3, 7)
    ax.set_ylim(-3, 7)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "posiciones.png"), dpi=150)
    print("Guardado: results/posiciones.png")
    plt.close(fig)


def plot_avg_fitness(ea_hist, pso_hist):
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, hist, color in [("EA", ea_hist, "#e74c3c"), ("PSO", pso_hist, "#3498db")]:
        grouped = hist.groupby("generation")["avg_fitness"]
        mean = grouped.mean()
        std = grouped.std()
        ax.plot(mean.index, mean.values, label=label, color=color, linewidth=2)
        ax.fill_between(mean.index, (mean - std).values, (mean + std).values,
                        alpha=0.15, color=color)

    ax.set_xlabel("Generacion / Iteracion", fontsize=12)
    ax.set_ylabel("Fitness Promedio Poblacional", fontsize=12)
    ax.set_title("Fitness Promedio de la Poblacion — EA vs PSO", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "avg_fitness.png"), dpi=150)
    print("Guardado: results/avg_fitness.png")
    plt.close(fig)


def main():
    ea_hist, pso_hist, ea_sum, pso_sum = load_data()

    plot_convergence(ea_hist, pso_hist)
    plot_boxplot(ea_sum, pso_sum)
    plot_final_positions(ea_sum, pso_sum)
    plot_avg_fitness(ea_hist, pso_hist)

    # Tabla resumen
    n_ea = len(ea_sum)
    n_pso = len(pso_sum)
    ea_success = (ea_sum["final_best_fitness"] < 0.01).sum()
    pso_success = (pso_sum["final_best_fitness"] < 0.01).sum()

    print(f"\n--- Tabla Resumen ({n_ea} runs cada uno) ---")
    print(f"{'':>25} {'EA':>15} {'PSO':>15}")
    print(f"{'Fitness promedio':>25} {ea_sum['final_best_fitness'].mean():>15.6f} {pso_sum['final_best_fitness'].mean():>15.6f}")
    print(f"{'Fitness mediana':>25} {ea_sum['final_best_fitness'].median():>15.6f} {pso_sum['final_best_fitness'].median():>15.6f}")
    print(f"{'Fitness std':>25} {ea_sum['final_best_fitness'].std():>15.6f} {pso_sum['final_best_fitness'].std():>15.6f}")
    print(f"{'Fitness min':>25} {ea_sum['final_best_fitness'].min():>15.6f} {pso_sum['final_best_fitness'].min():>15.6f}")
    print(f"{'Fitness max':>25} {ea_sum['final_best_fitness'].max():>15.6f} {pso_sum['final_best_fitness'].max():>15.6f}")
    print(f"{'Tasa exito (<0.01)':>25} {ea_success:>11}/{n_ea} {pso_success:>11}/{n_pso}")
    print(f"{'Gen convergencia (med)':>25} {ea_sum['convergence_gen'].median():>15.1f} {pso_sum['convergence_gen'].median():>15.1f}")


if __name__ == "__main__":
    main()
