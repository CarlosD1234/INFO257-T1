// PSO para minimizar Rastrigin en 2D entre -3 < xi < 7

// ===============================================================
// Parámetros del dominio
float domMin = -3;  // límite inferior del dominio
float domMax = 7;   // límite superior del dominio

int puntos = 100;
Particle[] fl; // arreglo de partículas
float d = 10; // radio del círculo, solo para despliegue
float gbestx, gbesty, gbest; // posición y fitness del mejor global (en coordenadas del dominio)
float w = 0.7;   // inercia: alta (~0.9): exploración, baja (~0.4): explotación
float C1 = 1.5, C2 = 1.5; // learning factors (C1: own, C2: social)
int evals = 0, evals_to_best = 0; // número de evaluaciones, sólo para despliegue
float maxv = 0.5; // max velocidad en el dominio (modulo)

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
class Particle {
  float x, y, fit; // posición actual en el dominio y fitness
  float px, py, pfit; // posición y fitness del mejor encontrado por la partícula (personal best)
  float vx, vy; // vector de avance (velocidad en el dominio)

  // ---------------------------- Constructor
  Particle() {
    x = random(domMin, domMax);
    y = random(domMin, domMax); // posición aleatoria en el dominio
    vx = random(-1, 1) * maxv;
    vy = random(-1, 1) * maxv;  // velocidad inicial aleatoria
    pfit = Float.MAX_VALUE;
    fit = Float.MAX_VALUE; // inicializa alto porque se minimiza
  }

  // ---------------------------- Evalúa partícula usando Rastrigin
  float Eval() {
    evals++;
    fit = rastrigin(x, y); // evalúa la función Rastrigin en la posición actual
    if (fit < pfit) { // actualiza local best si es mejor (minimización: menor es mejor)
      pfit = fit;
      px = x;
      py = y;
    }
    if (fit < gbest) { // actualiza global best
      gbest = fit;
      gbestx = x;
      gbesty = y;
      evals_to_best = evals;
      println("Nuevo gbest: " + str(gbest) + " en (" + nf(gbestx, 1, 4) + ", " + nf(gbesty, 1, 4) + ")");
    }
    return fit;
  }

  // ------------------------------ mueve la partícula
  void move() {
    // actualiza velocidad (fórmula mezclada)
    vx = w * vx + random(0, 1) * C1 * (px - x) + random(0, 1) * C2 * (gbestx - x);
    vy = w * vy + random(0, 1) * C1 * (py - y) + random(0, 1) * C2 * (gbesty - y);
    // trunca velocidad a maxv
    float modu = sqrt(vx * vx + vy * vy);
    if (modu > maxv) {
      vx = vx / modu * maxv;
      vy = vy / modu * maxv;
    }
    // actualiza posición
    x = x + vx;
    y = y + vy;
    // rebota en los límites del dominio
    if (x > domMax) {
      x = domMax;
      vx = -vx;
    }
    if (x < domMin) {
      x = domMin;
      vx = -vx;
    }
    if (y > domMax) {
      y = domMax;
      vy = -vy;
    }
    if (y < domMin) {
      y = domMin;
      vy = -vy;
    }
  }

  // ------------------------------ despliega partícula
  void display() {
    float sx = domToScreenX(x);
    float sy = domToScreenY(y);
    // color de la partícula según su fitness (blanco = bueno, gris = malo)
    float normFit = (fit >= Float.MAX_VALUE) ? 80 : map(fit, 0, 80, 255, 80); // evita warning con MAX_VALUE
    normFit = constrain(normFit, 80, 255);
    fill(normFit);
    stroke(0);
    strokeWeight(1);
    ellipse(sx, sy, d, d);
    // dibuja vector
    stroke(#ff0000);
    float svx = map(vx, 0, domMax - domMin, 0, width); // escala velocidad a pixeles
    float svy = map(vy, 0, domMax - domMin, 0, height);
    line(sx, sy, sx - 10 * svx, sy - 10 * svy);
  }
} // fin de la definición de la clase Particle


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

  // crea arreglo de objetos partículas
  fl = new Particle[puntos];
  for (int i = 0; i < puntos; i++)
    fl[i] = new Particle();
}

void draw() {
  // despliega mapa de calor de Rastrigin, posiciones y otros
  image(surfImg, 0, 0);
  for (int i = 0; i < puntos; i++) {
    fl[i].display();
  }
  despliegaBest();
  // mueve puntos y evalúa
  for (int i = 0; i < puntos; i++) {
    fl[i].move();
    fl[i].Eval();
  }
}
