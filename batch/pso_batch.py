"""
Batch runner para Particle Swarm Optimization (PSO) sobre Rastrigin 2D.
Replica fielmente pso/PSO_Rastrigin.pde.
"""
import random
import math
import csv
import os

from rastrigin import rastrigin, clamp, X_MIN, X_MAX

# --- Parametros (mismos que PSO_Rastrigin.pde) ---
N_PARTICLES = 100
MAX_ITER = 500
W = 0.7        # inercia
C1 = 1.5       # factor cognitivo
C2 = 1.5       # factor social
MAX_V = 0.5    # velocidad maxima (magnitud)
N_RUNS = 30


class Particle:
    def __init__(self):
        self.x = random.uniform(X_MIN, X_MAX)
        self.y = random.uniform(X_MIN, X_MAX)
        self.vx = random.uniform(-1, 1) * MAX_V
        self.vy = random.uniform(-1, 1) * MAX_V
        self.fit = float('inf')
        self.px = self.x
        self.py = self.y
        self.pfit = float('inf')


def run_pso(seed):
    random.seed(seed)
    history = []

    # Inicializar particulas
    particles = [Particle() for _ in range(N_PARTICLES)]

    gbest_fit = float('inf')
    gbest_x = 0.0
    gbest_y = 0.0
    convergence_iter = 0

    # Evaluar posiciones iniciales
    for p in particles:
        p.fit = rastrigin(p.x, p.y)
        if p.fit < p.pfit:
            p.pfit = p.fit
            p.px = p.x
            p.py = p.y
        if p.fit < gbest_fit:
            gbest_fit = p.fit
            gbest_x = p.x
            gbest_y = p.y

    # Registrar iteracion 0
    avg_fit = sum(p.fit for p in particles) / N_PARTICLES
    history.append((0, gbest_fit, avg_fit, gbest_x, gbest_y))

    for it in range(1, MAX_ITER):
        # Para cada particula secuencialmente (update asincrono de gbest)
        for p in particles:
            # Actualizar velocidad
            r1x = random.random()
            r1y = random.random()
            r2x = random.random()
            r2y = random.random()
            p.vx = W * p.vx + r1x * C1 * (p.px - p.x) + r2x * C2 * (gbest_x - p.x)
            p.vy = W * p.vy + r1y * C1 * (p.py - p.y) + r2y * C2 * (gbest_y - p.y)

            # Clamp magnitud de velocidad
            mag = math.sqrt(p.vx**2 + p.vy**2)
            if mag > MAX_V:
                p.vx = p.vx / mag * MAX_V
                p.vy = p.vy / mag * MAX_V

            # Actualizar posicion
            p.x += p.vx
            p.y += p.vy

            # Reflexion en bordes
            if p.x > X_MAX:
                p.x = X_MAX
                p.vx = -p.vx
            if p.x < X_MIN:
                p.x = X_MIN
                p.vx = -p.vx
            if p.y > X_MAX:
                p.y = X_MAX
                p.vy = -p.vy
            if p.y < X_MIN:
                p.y = X_MIN
                p.vy = -p.vy

            # Evaluar
            p.fit = rastrigin(p.x, p.y)
            if p.fit < p.pfit:
                p.pfit = p.fit
                p.px = p.x
                p.py = p.y
            if p.fit < gbest_fit:
                gbest_fit = p.fit
                gbest_x = p.x
                gbest_y = p.y
                convergence_iter = it

        avg_fit = sum(p.fit for p in particles) / N_PARTICLES
        history.append((it, gbest_fit, avg_fit, gbest_x, gbest_y))

    summary = (gbest_fit, gbest_x, gbest_y, convergence_iter)
    return history, summary


def main():
    os.makedirs("results", exist_ok=True)

    all_history = []
    all_summary = []

    for run_id in range(N_RUNS):
        seed = run_id + 42
        history, summary = run_pso(seed)
        for row in history:
            all_history.append((run_id, *row))
        all_summary.append((run_id, *summary))
        print(f"PSO Run {run_id + 1}/{N_RUNS}: best_fitness = {summary[0]:.6f}, "
              f"pos = ({summary[1]:.4f}, {summary[2]:.4f}), conv_iter = {summary[3]}")

    # Escribir history CSV
    with open("results/pso_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "generation", "best_fitness", "avg_fitness", "best_x", "best_y"])
        writer.writerows(all_history)

    # Escribir summary CSV
    with open("results/pso_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "final_best_fitness", "final_best_x", "final_best_y", "convergence_gen"])
        writer.writerows(all_summary)

    # Estadisticas finales
    final_fits = [s[1] for s in all_summary]
    print(f"\n--- PSO Resumen ({N_RUNS} runs) ---")
    print(f"Mejor fitness promedio: {sum(final_fits) / len(final_fits):.6f}")
    print(f"Mejor fitness minimo:   {min(final_fits):.6f}")
    print(f"Mejor fitness maximo:   {max(final_fits):.6f}")
    print(f"Resultados en results/pso_history.csv y results/pso_summary.csv")


if __name__ == "__main__":
    main()
