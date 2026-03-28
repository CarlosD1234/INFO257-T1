"""
Batch runner para Algoritmo Evolutivo (EA) sobre Rastrigin 2D.
Replica fielmente ae/AE_Rastrigin.pde con elitismo.
"""
import random
import math
import csv
import os

from rastrigin import rastrigin, clamp, X_MIN, X_MAX

# --- Parametros (mismos que AE_Rastrigin.pde) ---
POP_SIZE = 100
MAX_GEN = 500
TOURNAMENT_K = 3
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.05      # por gen
MUTATION_RANGE = 0.5      # uniforme [-range, +range]
N_RUNS = 30


def tournament_selection(pop, fitnesses):
    best = random.randint(0, POP_SIZE - 1)
    for _ in range(TOURNAMENT_K - 1):
        candidate = random.randint(0, POP_SIZE - 1)
        if fitnesses[candidate] < fitnesses[best]:
            best = candidate
    return pop[best]


def crossover(p1, p2):
    if random.random() < CROSSOVER_PROB:
        alpha = random.random()
        c1 = (alpha * p1[0] + (1 - alpha) * p2[0],
              alpha * p1[1] + (1 - alpha) * p2[1])
        c2 = ((1 - alpha) * p1[0] + alpha * p2[0],
              (1 - alpha) * p1[1] + alpha * p2[1])
        return c1, c2
    else:
        return (p1[0], p1[1]), (p2[0], p2[1])


def mutate(ind):
    x, y = ind
    if random.random() < MUTATION_PROB:
        x += random.uniform(-MUTATION_RANGE, MUTATION_RANGE)
    if random.random() < MUTATION_PROB:
        y += random.uniform(-MUTATION_RANGE, MUTATION_RANGE)
    return (clamp(x, X_MIN, X_MAX), clamp(y, X_MIN, X_MAX))


def run_ea(seed):
    random.seed(seed)
    history = []

    # Inicializar poblacion
    pop = [(random.uniform(X_MIN, X_MAX), random.uniform(X_MIN, X_MAX))
           for _ in range(POP_SIZE)]
    fitnesses = [rastrigin(x, y) for x, y in pop]

    gbest_fit = min(fitnesses)
    gbest_idx = fitnesses.index(gbest_fit)
    gbest_pos = pop[gbest_idx]
    convergence_gen = 0

    # Registrar gen 0
    avg_fit = sum(fitnesses) / POP_SIZE
    best_fit = min(fitnesses)
    history.append((0, best_fit, avg_fit, gbest_pos[0], gbest_pos[1]))

    for gen in range(1, MAX_GEN):
        new_pop = []

        # Elitismo: preservar el mejor
        elite_idx = fitnesses.index(min(fitnesses))
        new_pop.append(pop[elite_idx])

        # Llenar el resto con seleccion + cruce + mutacion
        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(pop, fitnesses)
            p2 = tournament_selection(pop, fitnesses)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)

        pop = new_pop
        fitnesses = [rastrigin(x, y) for x, y in pop]

        # Actualizar gbest
        gen_best_fit = min(fitnesses)
        if gen_best_fit < gbest_fit:
            gbest_fit = gen_best_fit
            gbest_idx = fitnesses.index(gen_best_fit)
            gbest_pos = pop[gbest_idx]
            convergence_gen = gen

        avg_fit = sum(fitnesses) / POP_SIZE
        history.append((gen, gbest_fit, avg_fit, gbest_pos[0], gbest_pos[1]))

    summary = (gbest_fit, gbest_pos[0], gbest_pos[1], convergence_gen)
    return history, summary


def main():
    os.makedirs("results", exist_ok=True)

    all_history = []
    all_summary = []

    for run_id in range(N_RUNS):
        seed = run_id + 42
        history, summary = run_ea(seed)
        for row in history:
            all_history.append((run_id, *row))
        all_summary.append((run_id, *summary))
        print(f"EA Run {run_id + 1}/{N_RUNS}: best_fitness = {summary[0]:.6f}, "
              f"pos = ({summary[1]:.4f}, {summary[2]:.4f}), conv_gen = {summary[3]}")

    # Escribir history CSV
    with open("results/ea_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "generation", "best_fitness", "avg_fitness", "best_x", "best_y"])
        writer.writerows(all_history)

    # Escribir summary CSV
    with open("results/ea_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "final_best_fitness", "final_best_x", "final_best_y", "convergence_gen"])
        writer.writerows(all_summary)

    # Estadisticas finales
    final_fits = [s[1] for s in all_summary]
    print(f"\n--- EA Resumen ({N_RUNS} runs) ---")
    print(f"Mejor fitness promedio: {sum(final_fits) / len(final_fits):.6f}")
    print(f"Mejor fitness minimo:   {min(final_fits):.6f}")
    print(f"Mejor fitness maximo:   {max(final_fits):.6f}")
    print(f"Resultados en results/ea_history.csv y results/ea_summary.csv")


if __name__ == "__main__":
    main()
