import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from datetime import datetime
import plot  # Importar nuestras funciones de plot.py
import os

# Asegurar que la carpeta de datos existe
OS_DATA_DIR = 'data'
os.makedirs(OS_DATA_DIR, exist_ok=True)

def rastrigin(x, y):
    """
    Función de Rastrigin para optimización.
    f(x,y) = 20 + (x^2 - 10*cos(2*pi*x)) + (y^2 - 10*cos(2*pi*y))
    """
    return 20 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))

class Particle:
    def __init__(self, dom_min, dom_max, max_v):
        self.dom_min = dom_min
        self.dom_max = dom_max
        self.max_v = max_v
        
        # Posición inicial aleatoria
        self.x = np.random.uniform(dom_min, dom_max)
        self.y = np.random.uniform(dom_min, dom_max)
        
        # Velocidad inicial aleatoria
        self.vx = np.random.uniform(-1, 1) * max_v
        self.vy = np.random.uniform(-1, 1) * max_v
        
        # Historial personal (pbest)
        self.px = self.x
        self.py = self.y
        self.pfit = float('inf')
        
        self.fit = float('inf')

    def evaluate(self, func):
        self.fit = func(self.x, self.y)
        if self.fit < self.pfit:
            self.pfit = self.fit
            self.px = self.x
            self.py = self.y
        return self.fit

    def move(self, gbest_x, gbest_y, w, c1, c2):
        # Actualización de velocidad (fórmula estándar PSO)
        r1, r2 = np.random.rand(), np.random.rand()
        self.vx = w * self.vx + r1 * c1 * (self.px - self.x) + r2 * c2 * (gbest_x - self.x)
        
        r3, r4 = np.random.rand(), np.random.rand()
        self.vy = w * self.vy + r3 * c1 * (self.py - self.y) + r4 * c2 * (gbest_y - self.y)
        
        # Truncar velocidad
        mod = np.sqrt(self.vx**2 + self.vy**2)
        if mod > self.max_v:
            self.vx = (self.vx / mod) * self.max_v
            self.vy = (self.vy / mod) * self.max_v
            
        # Actualizar posición
        self.x += self.vx
        self.y += self.vy
        
        # Rebote en los límites (como en Processing)
        if self.x > self.dom_max:
            self.x = self.dom_max
            self.vx *= -1
        elif self.x < self.dom_min:
            self.x = self.dom_min
            self.vx *= -1
            
        if self.y > self.dom_max:
            self.y = self.dom_max
            self.vy *= -1
        elif self.y < self.dom_min:
            self.y = self.dom_min
            self.vy *= -1

class PSO:
    def __init__(self, func, n_particles=50, dom_min=-3, dom_max=7, w=0.7, c1=1.5, c2=1.5, max_v=0.5):
        self.func = func
        self.n_particles = n_particles
        self.dom_min = dom_min
        self.dom_max = dom_max
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_v = max_v
        
        # Inicializar enjambre
        self.particles = [Particle(dom_min, dom_max, max_v) for _ in range(n_particles)]
        self.gbest_x = 0
        self.gbest_y = 0
        self.gbest_fit = float('inf')
        self.evals = 0
        self.history = []        # GBest Histórico
        self.avg_history = []    # Promedio de la generación
        self.best_gen_history = [] # Mejor de la generación actual
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def step(self, save_data=True):
        current_fits = []
        for p in self.particles:
            fit = p.evaluate(self.func)
            current_fits.append(fit)
            self.evals += 1
            if fit < self.gbest_fit:
                self.gbest_fit = fit
                self.gbest_x = p.x
                self.gbest_y = p.y
        
        # Guardar métricas de la generación
        current_avg = np.mean(current_fits)
        current_best = np.min(current_fits)
        
        self.history.append(self.gbest_fit)
        self.avg_history.append(current_avg)
        self.best_gen_history.append(current_best)
        
        if save_data:
            self._save_to_csv(current_best, current_avg)
        
        for p in self.particles:
            p.move(self.gbest_x, self.gbest_y, self.w, self.c1, self.c2)

    def _save_to_csv(self, current_best, current_avg):
        csv_path = os.path.join(OS_DATA_DIR, 'data_pso.csv')
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['ejecucion', 'iteracion', 'gbest_gen', 'average', 'gbest_historico'])
            writer.writerow([self.timestamp, len(self.history), current_best, current_avg, self.gbest_fit])

    def ejecutar(self, iterations=100, animation=True, save_data=True):
        """
        Punto de entrada principal para ejecutar el PSO.
        """
        if animation:
            self.run_animation(iterations, save_data)
        else:
            print(f"Ejecutando PSO en modo estudio (Sin animación)... Guardando: {save_data}")
            for i in range(iterations):
                self.step(save_data=save_data)
            print(f"Completado. Mejor fitness final: {self.gbest_fit}")

    def run_animation(self, iterations=100, save_data=True):
        fig = plt.figure(figsize=(15, 6))
        
        # Subplot 1: 2D Heatmap
        ax_map = fig.add_subplot(1, 2, 1)
        res = 100
        x = np.linspace(self.dom_min, self.dom_max, res)
        y = np.linspace(self.dom_min, self.dom_max, res)
        X, Y = np.meshgrid(x, y)
        Z = self.func(X, Y)
        
        im = ax_map.imshow(Z, extent=[self.dom_min, self.dom_max, self.dom_min, self.dom_max], 
                    origin='lower', cmap='magma', aspect='auto', alpha=0.9)
        scat = ax_map.scatter([], [], c='cyan', edgecolors='black', s=25, label='Partículas')
        best_dot = ax_map.scatter([], [], c='blue', s=60, marker='X', label='GBest Global')
        ax_map.set_title("Evolución del Enjambre")
        ax_map.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax_map, label='Fitness')
        
        # Subplot 2: Convergencia (Múltiples líneas)
        ax_conv = fig.add_subplot(1, 2, 2)
        line_gbest, = ax_conv.plot([], [], color='red', lw=2, label='GBest Histórico')
        line_best_gen, = ax_conv.plot([], [], color='green', lw=1, alpha=0.8, label='Mejor Gen.')
        line_avg, = ax_conv.plot([], [], color='blue', lw=1, ls='--', alpha=0.8, label='Promedio Gen.')
        
        ax_conv.set_title("Métricas de Convergencia")
        ax_conv.set_xlabel("Generación")
        ax_conv.set_ylabel("Fitness")
        ax_conv.grid(True, ls="-", alpha=0.3)
        ax_conv.set_xlim(0, iterations)
        ax_conv.legend(fontsize=9)
        
        # Texto de info
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
            
            # Actualizar Map
            pos = np.array([[p.x, p.y] for p in self.particles])
            scat.set_offsets(pos)
            best_dot.set_offsets([[self.gbest_x, self.gbest_y]])
            
            # Actualizar Curvas
            gens = np.arange(len(self.history))
            line_gbest.set_data(gens, self.history)
            line_best_gen.set_data(gens, self.best_gen_history)
            line_avg.set_data(gens, self.avg_history)
            
            # Ajustar límites de Y (con margen de seguridad para log)
            if len(self.history) > 1:
                all_vals = self.history + self.best_gen_history + self.avg_history
                ymin = max(1e-12, min(all_vals) * 0.1) # Evitar <= 0
                ymax = max(all_vals) * 2
                ax_conv.set_ylim(ymin, ymax)
            
            info_text.set_text(f"Iter: {frame} | GBest: {int(self.gbest_fit)} | Avg: {int(self.avg_history[-1])} | Pos: ({int(self.gbest_x)}, {int(self.gbest_y)})")
            
            return scat, line_gbest, line_best_gen, line_avg, info_text

        ani = FuncAnimation(fig, update, frames=iterations, init_func=init, blit=False, interval=50, repeat=False)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Crear instancia de PSO con Rastrigin
    pso = PSO(rastrigin, n_particles=50, dom_min=-3, dom_max=7)
    
    # Flags de ejecución
    MOSTRAR_ANIMACION = True
    GUARDAR_DATOS = True
    
    # Iniciar proceso
    pso.ejecutar(iterations=200, animation=MOSTRAR_ANIMACION, save_data=GUARDAR_DATOS)
