// AE (Algoritmo Evolutivo) para minimizar Rastrigin en 2D entre -3 < xi < 7

// ===============================================================
// Parámetros del dominio
float domMin = -3;  // límite inferior del dominio
float domMax = 7;   // límite superior del dominio

int popSize = 100;         // tamaño de la población
Individual[] pop;          // arreglo de individuos (población actual)
Individual[] newPop;       // arreglo de individuos (nueva generación)
float d = 10;              // radio del círculo, solo para despliegue
float gbestx, gbesty, gbest; // posición y fitness del mejor global (en coordenadas del dominio)
int evals = 0, evals_to_best = 0; // número de evaluaciones, sólo para despliegue
int generation = 0;        // contador de generaciones

// Parámetros del AE
int tournamentK = 3;       // tamaño del torneo para selección
float crossoverProb = 0.7; // probabilidad de cruzamiento
float mutationProb = 0.05; // probabilidad de mutación por gen
float mutationRange = 0.5; // rango máximo de perturbación en mutación

PImage surfImg; // imagen del mapa de calor de Rastrigin para despliegue

// ===============================================================
// Función Rastrigin para 2 dimensiones
// f(x) = 10*n + sum(xi^2 - 10*cos(2*PI*xi))
float rastrigin(float x1, float x2) {
  int n = 2;
  float sum = (x1 * x1 - 10 * cos(TWO_PI * x1))
    + (x2 * x2 - 10 * cos(TWO_PI * x2));
  return 10 * n + sum;
}

// ===============================================================
// Convierte coordenada del dominio a pixel de pantalla
float domToScreenX(float domX) {
  return map(domX, domMin, domMax, 0, width);
}
float domToScreenY(float domY) {
  return map(domY, domMin, domMax, 0, height);
}

// Convierte pixel de pantalla a coordenada del dominio
float screenToDomX(float sx) {
  return map(sx, 0, width, domMin, domMax);
}
float screenToDomY(float sy) {
  return map(sy, 0, height, domMin, domMax);
}

// ===============================================================
// Genera la imagen del mapa de calor de la función Rastrigin
PImage generarSuperficieRastrigin() {
  PImage img = createImage(width, height, RGB);
  img.loadPixels();

  // Primero calcula el rango de valores para normalizar colores
  float fMin = Float.MAX_VALUE;
  float fMax = -Float.MAX_VALUE;
  float[] vals = new float[width * height];
  for (int py = 0; py < height; py++) {
    for (int px = 0; px < width; px++) {
      float rx = screenToDomX(px);
      float ry = screenToDomY(py);
      float val = rastrigin(rx, ry);
      vals[py * width + px] = val;
      if (val < fMin) fMin = val;
      if (val > fMax) fMax = val;
    }
  }

  // Asigna colores: zonas bajas (mínimos) oscuras/azul, zonas altas (máximos) brillantes/rojo
  for (int py = 0; py < height; py++) {
    for (int px = 0; px < width; px++) {
      float val = vals[py * width + px];
      float norm = map(val, fMin, fMax, 0, 1);
      // Mapa de color: azul oscuro (bajo) -> cyan -> amarillo -> rojo (alto)
      color c;
      if (norm < 0.33) {
        float t = norm / 0.33;
        c = lerpColor(color(0, 0, 80), color(0, 180, 200), t);
      } else if (norm < 0.66) {
        float t = (norm - 0.33) / 0.33;
        c = lerpColor(color(0, 180, 200), color(255, 255, 0), t);
      } else {
        float t = (norm - 0.66) / 0.34;
        c = lerpColor(color(255, 255, 0), color(255, 50, 0), t);
      }
      img.pixels[py * width + px] = c;
    }
  }
  img.updatePixels();
  return img;
}

// ===============================================================
class Individual {
  float x, y, fit; // posición actual en el dominio (genes) y fitness

  // ---------------------------- Constructor
  Individual() {
    x = random(domMin, domMax);
    y = random(domMin, domMax); // posición aleatoria en el dominio (genes aleatorios)
    fit = Float.MAX_VALUE; // inicializa alto porque se minimiza
  }

  // ---------------------------- Constructor copia (para crear hijos)
  Individual(float _x, float _y) {
    x = _x;
    y = _y;
    fit = Float.MAX_VALUE;
  }

  // ---------------------------- Evalúa individuo usando Rastrigin
  float Eval() {
    evals++;
    fit = rastrigin(x, y); // evalúa la función Rastrigin en la posición actual
    if (fit < gbest) { // actualiza global best
      gbest = fit;
      gbestx = x;
      gbesty = y;
      evals_to_best = evals;
      println("Nuevo gbest: " + str(gbest) + " en (" + nf(gbestx, 1, 4) + ", " + nf(gbesty, 1, 4) + ") gen=" + generation);
    }
    return fit;
  }

  // ------------------------------ despliega individuo
  void display() {
    float sx = domToScreenX(x);
    float sy = domToScreenY(y);
    // color del individuo según su fitness (blanco = bueno, gris = malo)
    float normFit = (fit >= Float.MAX_VALUE) ? 80 : map(fit, 0, 80, 255, 80);
    normFit = constrain(normFit, 80, 255);
    fill(normFit);
    stroke(0);
    strokeWeight(1);
    ellipse(sx, sy, d, d);
  }
} // fin de la definición de la clase Individual


// ===============================================================
// Selección por torneo: elige K individuos al azar y retorna el mejor
Individual tournamentSelection(Individual[] population) {
  Individual best = population[int(random(popSize))];
  for (int i = 1; i < tournamentK; i++) {
    Individual contender = population[int(random(popSize))];
    if (contender.fit < best.fit) { // minimización: menor es mejor
      best = contender;
    }
  }
  return best;
}

// ===============================================================
// Cruzamiento aritmético (blend): genera dos hijos mezclando genes de los padres
// Retorna un arreglo de dos hijos
Individual[] crossover(Individual parent1, Individual parent2) {
  Individual[] children = new Individual[2];
  if (random(1) < crossoverProb) {
    // Cruzamiento BLX-alpha (blend crossover) con alpha=0
    // Equivale a cruzamiento aritmético: hijo = alpha*p1 + (1-alpha)*p2
    float alpha = random(0, 1);
    float child1x = alpha * parent1.x + (1 - alpha) * parent2.x;
    float child1y = alpha * parent1.y + (1 - alpha) * parent2.y;
    float child2x = (1 - alpha) * parent1.x + alpha * parent2.x;
    float child2y = (1 - alpha) * parent1.y + alpha * parent2.y;
    children[0] = new Individual(child1x, child1y);
    children[1] = new Individual(child2x, child2y);
  } else {
    // Sin cruzamiento: los hijos son copias de los padres
    children[0] = new Individual(parent1.x, parent1.y);
    children[1] = new Individual(parent2.x, parent2.y);
  }
  return children;
}

// ===============================================================
// Mutación: perturba cada gen con probabilidad mutationProb
void mutate(Individual ind) {
  // Muta gen x
  if (random(1) < mutationProb) {
    ind.x += random(-mutationRange, mutationRange);
  }
  // Muta gen y
  if (random(1) < mutationProb) {
    ind.y += random(-mutationRange, mutationRange);
  }
  // Asegura que los genes queden dentro del dominio
  ind.x = constrain(ind.x, domMin, domMax);
  ind.y = constrain(ind.y, domMin, domMax);
}

// ===============================================================
// dibuja punto azul en la mejor posición y despliega números
void despliegaBest() {
  float sx = domToScreenX(gbestx);
  float sy = domToScreenY(gbesty);
  fill(#0000ff);
  noStroke();
  ellipse(sx, sy, d + 4, d + 4);
  // Texto informativo
  PFont f = createFont("Arial", 16, true);
  textFont(f, 15);
  fill(#00ff00);
  text("Best fitness: " + nf(gbest, 1, 6)
    + "\nBest pos: (" + nf(gbestx, 1, 4) + ", " + nf(gbesty, 1, 4) + ")"
    + "\nGeneracion: " + str(generation)
    + "\nEvals to best: " + str(evals_to_best)
    + "\nEvals: " + str(evals), 10, 20);
}

// ===============================================================

void setup() {
  size(600, 600); // tamaño de la ventana
  smooth();

  // genera la imagen de la superficie Rastrigin para el fondo
  surfImg = generarSuperficieRastrigin();

  // inicializa gbest con valor alto (minimización)
  gbest = Float.MAX_VALUE;

  // crea arreglo de individuos (población inicial)
  pop = new Individual[popSize];
  for (int i = 0; i < popSize; i++) {
    pop[i] = new Individual();
    pop[i].Eval(); // evalúa la población inicial
  }
}

void draw() {
  // despliega mapa de calor de Rastrigin, posiciones y otros
  image(surfImg, 0, 0);
  for (int i = 0; i < popSize; i++) {
    pop[i].display();
  }
  despliegaBest();

  // === Ciclo evolutivo: Selección -> Cruzamiento -> Mutación -> Evaluación -> Reinserción ===
  newPop = new Individual[popSize];
  int idx = 0;

  // Genera nueva población mediante selección, cruzamiento y mutación
  while (idx < popSize) {
    // Selección por torneo de dos padres
    Individual parent1 = tournamentSelection(pop);
    Individual parent2 = tournamentSelection(pop);

    // Cruzamiento: genera dos hijos
    Individual[] children = crossover(parent1, parent2);

    // Mutación: perturba los hijos
    mutate(children[0]);
    mutate(children[1]);

    // Evaluación de los hijos
    children[0].Eval();
    if (idx + 1 < popSize) {
      children[1].Eval();
    }

    // Agrega hijos a la nueva población
    newPop[idx] = children[0];
    idx++;
    if (idx < popSize) {
      newPop[idx] = children[1];
      idx++;
    }
  }

  // Reinserción generacional: la nueva población reemplaza a la antigua
  pop = newPop;
  generation++;
}
