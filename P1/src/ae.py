import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from datetime import datetime
import os
import plot  # Importar nuestras funciones de plot.py

# Asegurar que la carpeta de datos existe
OS_DATA_DIR = 'data'
os.makedirs(OS_DATA_DIR, exist_ok=True)

def rastrigin(x, y):
    """
    f(x,y) = 20 + (x^2 - 10*cos(2*pi*x)) + (y^2 - 10*cos(2*pi*y))
    """
    return 20 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))

class Individual:
    def __init__(self, dom_min, dom_max, x=None, y=None):
        self.dom_min = dom_min
        self.dom_max = dom_max
        
        # Genes (Posición)
        if x is None:
            self.x = np.random.uniform(dom_min, dom_max)
        else:
            self.x = x
            
        if y is None:
            self.y = np.random.uniform(dom_min, dom_max)
        else:
            self.y = y
            
        self.fit = float('inf')

    def evaluate(self, func):
        self.fit = func(self.x, self.y)
        return self.fit

class AE:
    def __init__(self, func, pop_size=100, dom_min=-3, dom_max=7, 
                 tournament_k=3, crossover_prob=0.7, mutation_prob=0.05, mutation_range=0.5, max_generations=100):
        self.func = func
        self.pop_size = pop_size
        self.dom_min = dom_min
        self.dom_max = dom_max
        self.tournament_k = tournament_k
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_range = mutation_range
        self.initial_mutation_range = mutation_range
        self.max_generations = max_generations
        
        # Inicializar población
        self.pop = [Individual(dom_min, dom_max) for _ in range(pop_size)]
        for ind in self.pop:
            ind.evaluate(self.func)
            
        self.gbest_x = 0
        self.gbest_y = 0
        self.gbest_fit = float('inf')
        self.generation = 0
        
        self.history = []        # GBest Histórico
        self.avg_history = []    # Promedio de la generación
        self.best_gen_history = [] # Mejor de la generación actual
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%fZ")

    def tournament_selection(self):
        # Seleccionar K individuos al azar y devolver el mejor
        contenders = np.random.choice(self.pop, self.tournament_k)
        best = contenders[0]
        for c in contenders[1:]:
            if c.fit < best.fit:
                best = c
        return best

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            # Cruzamiento aritmético (BLX-alpha con alpha=0)
            alpha = np.random.rand()
            c1_x = alpha * parent1.x + (1 - alpha) * parent2.x
            c1_y = alpha * parent1.y + (1 - alpha) * parent2.y
            
            c2_x = (1 - alpha) * parent1.x + alpha * parent2.x
            c2_y = (1 - alpha) * parent1.y + alpha * parent2.y
            
            return Individual(self.dom_min, self.dom_max, c1_x, c1_y), \
                   Individual(self.dom_min, self.dom_max, c2_x, c2_y)
        else:
            # Sin cruzamiento: copias de los padres
            return Individual(self.dom_min, self.dom_max, parent1.x, parent1.y), \
                   Individual(self.dom_min, self.dom_max, parent2.x, parent2.y)

    def mutate(self, ind):
        # Mutación por gen
        if np.random.rand() < self.mutation_prob:
            ind.x += np.random.uniform(-self.mutation_range, self.mutation_range)
        if np.random.rand() < self.mutation_prob:
            ind.y += np.random.uniform(-self.mutation_range, self.mutation_range)
            
        # Constrain (Límites del dominio)
        ind.x = np.clip(ind.x, self.dom_min, self.dom_max)
        ind.y = np.clip(ind.y, self.dom_min, self.dom_max)

    def step(self, save_data=True):
        # Adaptación del rango de mutación (disminución en 1/4 cada hito)
        if not hasattr(self, 'hitos_alcanzados'):
            self.hitos_alcanzados = set()

        hitos = [0.2, 0.4, 0.6, 0.8]
        for hito in hitos:
            if hito not in self.hitos_alcanzados and self.generation >= int(self.max_generations * hito):
                self.mutation_range -= self.initial_mutation_range * 0.22
                self.mutation_range = max(self.mutation_range, self.initial_mutation_range * 0.01)
                self.hitos_alcanzados.add(hito)
                print(f"--- Gen {self.generation}: Hito {hito} alcanzado. Rango ahora: {self.mutation_range:.4f} ---")

        new_pop = []
        
        # Elitismo: Preservar el mejor
        best_current = min(self.pop, key=lambda ind: ind.fit)
        elite = Individual(self.dom_min, self.dom_max, best_current.x, best_current.y)
        elite.fit = best_current.fit
        new_pop.append(elite)
        
        # Generar nueva población
        while len(new_pop) < self.pop_size:
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            
            c1, c2 = self.crossover(p1, p2)
            
            self.mutate(c1)
            self.mutate(c2)
            
            c1.evaluate(self.func)
            new_pop.append(c1)
            if len(new_pop) < self.pop_size:
                c2.evaluate(self.func)
                new_pop.append(c2)
        
        self.pop = new_pop
        self.generation += 1
        
        # Actualizar GBest
        gen_best = min(self.pop, key=lambda ind: ind.fit)
        if gen_best.fit < self.gbest_fit:
            self.gbest_fit = gen_best.fit
            self.gbest_x = gen_best.x
            self.gbest_y = gen_best.y
            
        # Métricas
        current_fits = [ind.fit for ind in self.pop]
        current_avg = np.mean(current_fits)
        current_best = np.min(current_fits)
        
        self.history.append(self.gbest_fit)
        self.avg_history.append(current_avg)
        self.best_gen_history.append(current_best)
        
        if save_data:
            self._save_to_csv(current_best, current_avg)

    def _save_to_csv(self, current_best, current_avg):
        csv_path = os.path.join(OS_DATA_DIR, 'data_ae.csv')
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['ejecucion', 'iteracion', 'gbest_gen', 'average', 'gbest_historico'])
            writer.writerow([self.timestamp, self.generation, current_best, current_avg, self.gbest_fit])

    def ejecutar(self, iterations=100, animation=True, save_data=True):
        if animation:
            self.run_animation(iterations, save_data)
        else:
            print(f"Ejecutando AE en modo estudio (Sin animación)... Guardando: {save_data}")
            for i in range(iterations):
                self.step(save_data=save_data)
            print(f"Completado. Mejor fitness final: {self.gbest_fit}")

    def run_animation(self, iterations=100, save_data=True):
        fig = plt.figure(figsize=(15, 6))
        
        # Subplot 1: Map
        ax_map = fig.add_subplot(1, 2, 1)
        res = 100
        x = np.linspace(self.dom_min, self.dom_max, res)
        y = np.linspace(self.dom_min, self.dom_max, res)
        X, Y = np.meshgrid(x, y)
        Z = self.func(X, Y)
        
        im = ax_map.imshow(Z, extent=[self.dom_min, self.dom_max, self.dom_min, self.dom_max], 
                    origin='lower', cmap='magma', aspect='auto', alpha=0.9)
        scat = ax_map.scatter([], [], c='cyan', edgecolors='black', s=25, label='Individuos')
        best_dot = ax_map.scatter([], [], c='blue', s=60, marker='X', label='GBest Global')
        ax_map.set_title("Población AE")
        ax_map.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax_map, label='Fitness')
        
        # Subplot 2: Convergencia
        ax_conv = fig.add_subplot(1, 2, 2)
        line_gbest, = ax_conv.plot([], [], color='red', lw=2, label='GBest Histórico')
        line_best_gen, = ax_conv.plot([], [], color='green', lw=1, alpha=0.8, label='Mejor Gen.')
        line_avg, = ax_conv.plot([], [], color='blue', lw=1, ls='--', alpha=0.8, label='Promedio Gen.')
        
        ax_conv.set_title("Métricas de AE")
        ax_conv.set_xlabel("Generación")
        ax_conv.set_ylabel("Fitness")
        ax_conv.grid(True, ls="-", alpha=0.3)
        ax_conv.set_xlim(0, iterations)
        ax_conv.legend(fontsize=9)
        
        info_text = fig.text(0.5, 0.01, '', ha='center', fontsize=12, fontweight='bold', color='blue')

        def init():
            scat.set_offsets(np.empty((0, 2)))
            line_gbest.set_data([], [])
            line_best_gen.set_data([], [])
            line_avg.set_data([], [])
            info_text.set_text('')
            return scat, line_gbest, line_best_gen, line_avg, info_text

        def update(frame):
            self.step(save_data=save_data)
            pos = np.array([[ind.x, ind.y] for ind in self.pop])
            scat.set_offsets(pos)
            best_dot.set_offsets([[self.gbest_x, self.gbest_y]])
            
            gens = np.arange(len(self.history))
            line_gbest.set_data(gens, self.history)
            line_best_gen.set_data(gens, self.best_gen_history)
            line_avg.set_data(gens, self.avg_history)
            
            if len(self.history) > 1:
                all_vals = self.history + self.best_gen_history + self.avg_history
                ymin = max(1e-12, min(all_vals) * 0.1)
                ymax = max(all_vals) * 2
                ax_conv.set_ylim(ymin, ymax)
            
            info_text.set_text(f"Gen: {self.generation} | GBest: {int(self.gbest_fit)} | Avg: {int(self.avg_history[-1])}")
            return scat, line_gbest, line_best_gen, line_avg, info_text

        ani = FuncAnimation(fig, update, frames=iterations, init_func=init, blit=False, interval=50, repeat=False)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Flags de ejecución
    MOSTRAR_ANIMACION = True
    GUARDAR_DATOS = False

    iterations = 200
    ae = AE(
        rastrigin, 
        pop_size=100, 
        dom_min=-3, 
        dom_max=7, 
        tournament_k=3,
        crossover_prob=0.8, 
        mutation_prob=0.2,        # 10% es más estándar que 10
        mutation_range=1,       # Empezamos con un rango mayor para explorar valles
        max_generations=iterations
    )
    ae.ejecutar(iterations=iterations, animation=MOSTRAR_ANIMACION, save_data=GUARDAR_DATOS)
    
    # # Iniciar proceso para estudio (30 ejecuciones independientes)
    # print(f"Iniciando estudio de 30 ejecuciones...")
    # for i in range(30):
    #     ae = AE(
    #         rastrigin, 
    #         pop_size=100, 
    #         dom_min=-3, 
    #         dom_max=7, 
    #         tournament_k=3,
    #         crossover_prob=0.8, 
    #         mutation_prob=0.2,        # 10% es más estándar que 10
    #         mutation_range=1,         # Empezamos con un rango mayor para explorar valles
    #         max_generations=iterations
    #     )
    #     ae.ejecutar(iterations=iterations, animation=MOSTRAR_ANIMACION, save_data=GUARDAR_DATOS)
    #     print(f"Ejecución {i+1}/30 completada.")
