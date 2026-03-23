// ============================================================
//  Algoritmo Evolutivo para minimizar la función de Rastrigin
//  Visualización interactiva en Processing
// ============================================================

// --- Parámetros del EA ---
int    POP_SIZE      = 80;
int    MAX_GEN       = 500;
float  MUTATION_RATE = 0.15;
float  MUTATION_STD  = 0.3;
float  CROSSOVER_RATE= 0.8;
float X_MIN = -3.0;
float X_MAX =  7.0;
int    TOURNAMENT_K  = 3;

// --- Estado global ---
float[][] population;
float[]   fitness;
float[]   bestPerGen;
float[]   avgPerGen;
int       generation  = 0;
float[]   globalBest;
float     globalBestFit;
boolean   running      = true;
boolean   paused       = false;
int       stepsPerFrame = 3;

// --- Visualización ---
int PLOT_W, PLOT_H;          // panel izquierdo: mapa de calor
int GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H; // panel derecho: historial
int SIDE_W;
color[] heatPalette;
int HEAT_RES = 120;          // resolución del mapa de calor
float[][] heatValues;
float heatMin, heatMax;

// Fuente y colores
color COL_BG     = color(18, 18, 28);
color COL_PANEL  = color(26, 26, 42);
color COL_ACCENT = color(90, 200, 255);
color COL_BEST   = color(255, 210, 60);
color COL_AVG    = color(140, 255, 160);
color COL_IND    = color(255, 100, 120, 200);
color COL_TEXT   = color(220, 220, 235);
color COL_DIM    = color(120, 120, 145);
color COL_GBEST  = color(255, 240, 80);

// ============================================================
void setup() {
  size(1100, 620);
  colorMode(RGB, 255);
  textFont(createFont("Consolas", 13));

  SIDE_W  = 520;
  PLOT_W  = SIDE_W - 20;
  PLOT_H  = height - 140;

  GRAPH_X = SIDE_W + 20;
  GRAPH_Y = 60;
  GRAPH_W = width - SIDE_W - 40;
  GRAPH_H = height - 160;

  // Precomputar mapa de calor de Rastrigin
  buildHeatmap();

  // Inicializar EA
  initPopulation();
  bestPerGen = new float[MAX_GEN];
  avgPerGen  = new float[MAX_GEN];
  evaluateFitness();
  recordStats();

  surface.setTitle("EA — Función de Rastrigin  |  [P] pausa  [R] reiniciar  [+/-] velocidad");
}

// ============================================================
void draw() {
  background(COL_BG);

  // Correr el EA
  if (running && !paused && generation < MAX_GEN - 1) {
    for (int s = 0; s < stepsPerFrame; s++) {
      if (generation < MAX_GEN - 1) {
        evolve();
        generation++;
        recordStats();
      }
    }
  }
  if (generation >= MAX_GEN - 1) running = false;

  // ---- Panel izquierdo: mapa de calor + individuos ----
  drawHeatmap();
  drawPopulationOnMap();
  drawGlobalBestOnMap();
  drawLeftLabels();

  // ---- Panel derecho: gráficas ----
  drawGraph();
  drawInfoBox();
}

// ============================================================
// ---- FUNCIÓN DE RASTRIGIN ----
float rastrigin(float[] x) {
  int n = x.length;
  float A = 10.0;
  float sum = A * n;
  for (int i = 0; i < n; i++) {
    sum += x[i]*x[i] - A * cos(TWO_PI * x[i]);
  }
  return sum;
}

// ============================================================
// ---- INICIALIZACIÓN ----
void initPopulation() {
  population  = new float[POP_SIZE][2];
  fitness     = new float[POP_SIZE];
  globalBest  = new float[]{0, 0};
  globalBestFit = Float.MAX_VALUE;

  for (int i = 0; i < POP_SIZE; i++) {
    population[i][0] = random(X_MIN, X_MAX);
    population[i][1] = random(X_MIN, X_MAX);
  }
}

// ============================================================
// ---- EVALUACIÓN ----
void evaluateFitness() {
  for (int i = 0; i < POP_SIZE; i++) {
    fitness[i] = rastrigin(population[i]);
    if (fitness[i] < globalBestFit) {
      globalBestFit = fitness[i];
      globalBest = population[i].clone();
    }
  }
}

// ============================================================
// ---- PASO EVOLUTIVO ----
void evolve() {
  float[][] newPop = new float[POP_SIZE][2];

  // Elitismo: conservar el mejor
  int eliteIdx = getEliteIndex();
  newPop[0] = population[eliteIdx].clone();

  for (int i = 1; i < POP_SIZE; i++) {
    // Selección por torneo
    float[] parent1 = tournament();
    float[] parent2 = tournament();

    // Cruce aritmético
    float[] child = new float[2];
    if (random(1) < CROSSOVER_RATE) {
      float alpha = random(1);
      child[0] = alpha * parent1[0] + (1 - alpha) * parent2[0];
      child[1] = alpha * parent1[1] + (1 - alpha) * parent2[1];
    } else {
      child = (random(1) < 0.5) ? parent1.clone() : parent2.clone();
    }

    // Mutación gaussiana
    if (random(1) < MUTATION_RATE) {
      child[0] += randomGaussian() * MUTATION_STD;
      child[1] += randomGaussian() * MUTATION_STD;
    }

    // Clamp al dominio
    child[0] = constrain(child[0], X_MIN, X_MAX);
    child[1] = constrain(child[1], X_MIN, X_MAX);
    newPop[i] = child;
  }

  population = newPop;
  evaluateFitness();
}

float[] tournament() {
  int best = (int) random(POP_SIZE);
  for (int k = 1; k < TOURNAMENT_K; k++) {
    int candidate = (int) random(POP_SIZE);
    if (fitness[candidate] < fitness[best]) best = candidate;
  }
  return population[best].clone();
}

int getEliteIndex() {
  int best = 0;
  for (int i = 1; i < POP_SIZE; i++) {
    if (fitness[i] < fitness[best]) best = i;
  }
  return best;
}

void recordStats() {
  if (generation >= MAX_GEN) return;
  float minF = Float.MAX_VALUE, sumF = 0;
  for (int i = 0; i < POP_SIZE; i++) {
    if (fitness[i] < minF) minF = fitness[i];
    sumF += fitness[i];
  }
  bestPerGen[generation] = minF;
  avgPerGen[generation]  = sumF / POP_SIZE;
}

// ============================================================
// ---- MAPA DE CALOR ----
void buildHeatmap() {
  heatValues = new float[HEAT_RES][HEAT_RES];
  heatMin = Float.MAX_VALUE;
  heatMax = -Float.MAX_VALUE;
  for (int xi = 0; xi < HEAT_RES; xi++) {
    for (int yi = 0; yi < HEAT_RES; yi++) {
      float xv = map(xi, 0, HEAT_RES - 1, X_MIN, X_MAX);
      float yv = map(yi, 0, HEAT_RES - 1, X_MIN, X_MAX);
      float v  = rastrigin(new float[]{xv, yv});
      heatValues[xi][yi] = v;
      if (v < heatMin) heatMin = v;
      if (v > heatMax) heatMax = v;
    }
  }
  // Paleta: azul oscuro → cyan → amarillo → rojo
  heatPalette = new color[256];
  for (int i = 0; i < 256; i++) {
    float t = i / 255.0;
    if (t < 0.33) {
      float s = t / 0.33;
      heatPalette[i] = color(lerp(10, 20, s), lerp(15, 130, s), lerp(60, 220, s));
    } else if (t < 0.66) {
      float s = (t - 0.33) / 0.33;
      heatPalette[i] = color(lerp(20, 230, s), lerp(130, 200, s), lerp(220, 40, s));
    } else {
      float s = (t - 0.66) / 0.34;
      heatPalette[i] = color(lerp(230, 220, s), lerp(200, 40, s), lerp(40, 20, s));
    }
  }
}

void drawHeatmap() {
  int ox = 10, oy = 60;
  float cellW = (float) PLOT_W / HEAT_RES;
  float cellH = (float) PLOT_H / HEAT_RES;
  noStroke();
  for (int xi = 0; xi < HEAT_RES; xi++) {
    for (int yi = 0; yi < HEAT_RES; yi++) {
      float t = map(heatValues[xi][yi], heatMin, heatMax, 0, 255);
      t = constrain(t, 0, 255);
      fill(heatPalette[(int) t]);
      rect(ox + xi * cellW, oy + yi * cellH, cellW + 0.5, cellH + 0.5);
    }
  }
  // Borde
  noFill();
  stroke(COL_ACCENT, 100);
  strokeWeight(1.5);
  rect(ox, oy, PLOT_W, PLOT_H);
}

// Convierte coordenada del dominio a px del mapa
float mapX(float v) { return map(v, X_MIN, X_MAX, 10, 10 + PLOT_W); }
float mapY(float v) { return map(v, X_MIN, X_MAX, 60, 60 + PLOT_H); }

void drawPopulationOnMap() {
  noStroke();
  for (int i = 0; i < POP_SIZE; i++) {
    float px = mapX(population[i][0]);
    float py = mapY(population[i][1]);
    float t  = map(fitness[i], 0, 80, 0, 1);
    fill(lerp(red(COL_IND), 60, t),
         lerp(green(COL_IND), 200, t),
         lerp(blue(COL_IND), 100, t), 210);
    ellipse(px, py, 7, 7);
  }
}

void drawGlobalBestOnMap() {
  float px = mapX(globalBest[0]);
  float py = mapY(globalBest[1]);
  // Halo
  noFill();
  stroke(COL_GBEST, 80);
  strokeWeight(2);
  ellipse(px, py, 22, 22);
  stroke(COL_GBEST, 40);
  ellipse(px, py, 34, 34);
  // Centro
  fill(COL_GBEST);
  noStroke();
  ellipse(px, py, 10, 10);
  // Etiqueta
  fill(COL_GBEST);
  textSize(11);
  textAlign(LEFT, BOTTOM);
  text("★ MEJOR", px + 8, py - 4);
}

void drawLeftLabels() {
  // Título
  fill(COL_ACCENT);
  textSize(15);
  textAlign(LEFT, TOP);
  text("Función de Rastrigin  f(x,y)", 10, 8);

  // Escala de color
  int bx = 10, by = 60 + PLOT_H + 8, bw = PLOT_W, bh = 14;
  for (int i = 0; i < bw; i++) {
    int idx = (int) map(i, 0, bw, 0, 255);
    stroke(heatPalette[constrain(idx, 0, 255)]);
    line(bx + i, by, bx + i, by + bh);
  }
  noStroke();
  fill(COL_DIM);
  textSize(10);
  textAlign(LEFT, TOP);
  text(nf(heatMin, 0, 1), bx, by + bh + 2);
  textAlign(RIGHT, TOP);
  text(nf(heatMax, 0, 1), bx + bw, by + bh + 2);
  textAlign(CENTER, TOP);
  text("Fitness (bajo = mejor)", bx + bw / 2, by + bh + 2);
}

// ============================================================
// ---- GRÁFICA DE CONVERGENCIA ----
void drawGraph() {
  int maxGen = max(1, generation);

  // Fondo panel
  fill(COL_PANEL);
  noStroke();
  rect(GRAPH_X - 10, GRAPH_Y - 30, GRAPH_W + 20, GRAPH_H + 50, 8);

  // Título
  fill(COL_ACCENT);
  textSize(13);
  textAlign(LEFT, TOP);
  text("Convergencia  —  Generación " + (generation + 1) + " / " + MAX_GEN, GRAPH_X, GRAPH_Y - 26);

  // Ejes
  stroke(COL_DIM, 100);
  strokeWeight(1);
  int ax = GRAPH_X, ay = GRAPH_Y;
  int aw = GRAPH_W, ah = GRAPH_H;
  line(ax, ay, ax, ay + ah);
  line(ax, ay + ah, ax + aw, ay + ah);

  // Ticks Y
  float yMin = 0, yMax = 0;
  for (int g = 0; g <= generation; g++) {
    if (bestPerGen[g] > yMax) yMax = bestPerGen[g];
    if (avgPerGen[g]  > yMax) yMax = avgPerGen[g];
  }
  yMax = max(yMax, 1);
  for (int t = 0; t <= 5; t++) {
    float fy = map(t, 0, 5, ay + ah, ay);
    float fv = map(t, 0, 5, yMin, yMax);
    stroke(COL_DIM, 60);
    line(ax, fy, ax + aw, fy);
    fill(COL_DIM);
    textSize(10);
    textAlign(RIGHT, CENTER);
    text(nf(fv, 0, 1), ax - 4, fy);
  }
  // Ticks X
  for (int t = 0; t <= 5; t++) {
    float fx = map(t, 0, 5, ax, ax + aw);
    float gv = map(t, 0, 5, 0, MAX_GEN);
    fill(COL_DIM);
    textSize(10);
    textAlign(CENTER, TOP);
    text((int) gv, fx, ay + ah + 4);
  }

  // Curva promedio
  if (generation > 0) {
    stroke(COL_AVG, 180);
    strokeWeight(1.5);
    noFill();
    beginShape();
    for (int g = 0; g <= generation; g++) {
      float fx = map(g, 0, MAX_GEN, ax, ax + aw);
      float fy = map(avgPerGen[g], yMin, yMax, ay + ah, ay);
      vertex(fx, constrain(fy, ay, ay + ah));
    }
    endShape();

    // Curva mejor
    stroke(COL_BEST, 230);
    strokeWeight(2.5);
    noFill();
    beginShape();
    for (int g = 0; g <= generation; g++) {
      float fx = map(g, 0, MAX_GEN, ax, ax + aw);
      float fy = map(bestPerGen[g], yMin, yMax, ay + ah, ay);
      vertex(fx, constrain(fy, ay, ay + ah));
    }
    endShape();

    // Punto actual
    float cx = map(generation, 0, MAX_GEN, ax, ax + aw);
    float cy = map(bestPerGen[generation], yMin, yMax, ay + ah, ay);
    noStroke();
    fill(COL_BEST);
    ellipse(cx, constrain(cy, ay, ay + ah), 8, 8);
  }

  // Leyenda
  noStroke();
  fill(COL_BEST);
  rect(ax, ay + ah + 22, 18, 4);
  fill(COL_TEXT);
  textSize(11);
  textAlign(LEFT, CENTER);
  text("Mejor individuo", ax + 22, ay + ah + 24);

  fill(COL_AVG);
  rect(ax + 130, ay + ah + 22, 18, 4);
  fill(COL_TEXT);
  text("Fitness promedio", ax + 152, ay + ah + 24);
}

// ============================================================
// ---- INFO BOX ----
void drawInfoBox() {
  int bx = GRAPH_X, by = GRAPH_Y + GRAPH_H + 42;
  fill(COL_DIM, 60);
  noStroke();
  rect(bx - 10, by - 6, GRAPH_W + 20, 68, 6);

  fill(COL_TEXT);
  textSize(12);
  textAlign(LEFT, TOP);
  String state = paused ? "PAUSADO" : (running ? "CORRIENDO" : "FINALIZADO");
  fill(running && !paused ? COL_AVG : (paused ? COL_BEST : COL_ACCENT));
  text("● " + state, bx, by);

  fill(COL_TEXT);
  text("Mejor fitness: " + nf(globalBestFit, 0, 4), bx, by + 18);
  text("Mejor pos:  x=" + nf(globalBest[0], 0, 4) +
       "  y=" + nf(globalBest[1], 0, 4), bx, by + 34);

  fill(COL_DIM);
  textSize(10);
  text("[P] Pausar / reanudar    [R] Reiniciar    [+] Más rápido    [-] Más lento    Vel: " + stepsPerFrame + "x",
       bx, by + 52);
}

// ============================================================
// ---- INTERACCIÓN ----
void keyPressed() {
  if (key == 'p' || key == 'P') paused = !paused;
  if (key == 'r' || key == 'R') {
    generation = 0;
    initPopulation();
    bestPerGen = new float[MAX_GEN];
    avgPerGen  = new float[MAX_GEN];
    evaluateFitness();
    recordStats();
    running = true;
    paused  = false;
  }
  if (key == '+' || key == '=') stepsPerFrame = min(stepsPerFrame + 1, 20);
  if (key == '-' || key == '_') stepsPerFrame = max(stepsPerFrame - 1, 1);
}