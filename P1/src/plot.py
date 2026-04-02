import numpy as np
import matplotlib.pyplot as plt

def plot_2d_heatmap(func, x_range, y_range, resolution=200):
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig, ax = plt.gcf(), plt.gca()
    # Usar un mapa de color con más contraste
    im = ax.imshow(
        Z,
        extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
        origin='lower',
        cmap='magma',
        aspect='auto',
    )
    
    plt.colorbar(im, ax=ax, label='Fitness')
    return fig, ax

if __name__ == "__main__":
    rastrigin = lambda x, y: 10 * 2 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))
    plot_2d_heatmap(rastrigin, (-3, 7), (-3, 7))
    plt.show()