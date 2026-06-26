#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sistema RAG sobre obesidad - Tarea Inteligencia Artificial (Equipo 2)

Uso:
    1) python embeddings.py            (una sola vez: construye el indice vectorial)
    2) python Equipo2_RAG_1.py preguntas.csv   (genera las respuestas)

Entrada : CSV con columnas (numero, pregunta)
Salida  : respuestas.csv con columnas (numero, respuesta, contexto)

Arquitectura:
    - Corpus: 18 documentos clinicos sobre obesidad -> 2158 chunks (data_extracted/chunks.json)
    - Embeddings: text-embedding-3-small (OpenAI), mismo modelo para indexar y consultar.
      Dimension 1536.
    - Vector store: ChromaDB persistente (construido previamente por embeddings.py)
    - Generacion: gpt-4o-mini (OpenAI) con grounding estricto anti-alucinacion

Este script NO construye el indice: asume que ya fue generado por embeddings.py.
Solo recupera contexto y genera respuestas. Requiere una OPENAI_API_KEY valida.
"""

import os
import sys
import csv

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

# Rutas relativas al directorio del script (para que funcione desde cualquier cwd)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "tesis_docs"

# Embeddings: text-embedding-3-small (OpenAI). Debe coincidir con el modelo usado
# por embeddings.py al construir el indice.
EMBEDDING_MODEL = "text-embedding-3-small"

CHAT_MODEL = "gpt-4o-mini"   # generacion via OpenAI

TOP_K = 5                 # fragmentos recuperados por pregunta

OUTPUT_CSV_PATH = "respuestas.csv"


# ---------------------------------------------------------------------------
# Dependencias
# ---------------------------------------------------------------------------

# Cargar variables de entorno desde .env si esta disponible
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    load_dotenv()  # tambien intenta en el cwd
except ImportError:
    pass

try:
    import chromadb
    from openai import OpenAI
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    print(f"\n[Error] Faltan dependencias: {e}", file=sys.stderr)
    print("Instala con: pip install openai chromadb python-dotenv\n", file=sys.stderr)


# ---------------------------------------------------------------------------
# Sistema RAG
# ---------------------------------------------------------------------------

class RAGSystem:
    def __init__(self):
        self.client = None          # OpenAI (embeddings + generacion de respuestas)
        self.collection = None

    # -- Inicializacion -----------------------------------------------------
    def initialize(self):
        """Prepara el cliente de OpenAI y la coleccion vectorial."""
        if not DEPS_AVAILABLE:
            return False

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[Error] No se encontro OPENAI_API_KEY.", file=sys.stderr)
            print("Crea un archivo .env junto al script con: OPENAI_API_KEY=tu_clave", file=sys.stderr)
            return False

        print("[1/3] Inicializando cliente de OpenAI (embeddings + chat)...")
        # Mismo cliente para embeddings y generacion de respuestas.
        self.client = OpenAI(api_key=api_key)

        print("[2/3] Abriendo base vectorial (ChromaDB)...")
        if not os.path.exists(CHROMA_PATH):
            print(f"[Error] No existe el indice vectorial en '{CHROMA_PATH}'.", file=sys.stderr)
            print("Ejecuta primero: python embeddings.py", file=sys.stderr)
            return False

        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            self.collection = chroma_client.get_collection(name=COLLECTION_NAME)
        except Exception:
            print(f"[Error] No se encontro la coleccion '{COLLECTION_NAME}' en el indice.",
                  file=sys.stderr)
            print("Ejecuta primero: python embeddings.py", file=sys.stderr)
            return False

        # Verificar que el indice se construyo con el mismo modelo de embeddings.
        indexed_model = (self.collection.metadata or {}).get("embedding_model")
        if indexed_model and indexed_model != EMBEDDING_MODEL:
            print(f"[Error] El indice se construyo con '{indexed_model}', pero este script "
                  f"usa '{EMBEDDING_MODEL}'. Las dimensiones no coinciden.", file=sys.stderr)
            print("Reconstruye el indice con: python embeddings.py", file=sys.stderr)
            return False

        try:
            existing = self.collection.count()
        except Exception:
            existing = 0

        if existing == 0:
            print(f"[Error] El indice '{COLLECTION_NAME}' esta vacio.", file=sys.stderr)
            print("Ejecuta primero: python embeddings.py", file=sys.stderr)
            return False

        print(f"  -> Coleccion lista ({existing} chunks indexados).")
        print("[3/3] Sistema RAG listo.")
        return True

    # -- Embeddings ---------------------------------------------------------
    def _embed(self, texts):
        """Genera embeddings con text-embedding-3-small (OpenAI) para una lista de textos."""
        cleaned = [t.replace("\n", " ") for t in texts]
        resp = self.client.embeddings.create(input=cleaned, model=EMBEDDING_MODEL)
        return [d.embedding for d in resp.data]

    # -- Consulta -----------------------------------------------------------
    def query(self, question):
        """Recupera contexto relevante y genera una respuesta con grounding estricto.

        Devuelve (respuesta, contexto_utilizado).
        """
        if not self.collection or not self.client:
            return ("Respuesta no disponible (sistema no inicializado).", "")

        try:
            # 1. Recuperar contexto (similarity search con el mismo modelo de embeddings)
            q_emb = self._embed([question])[0]
            results = self.collection.query(
                query_embeddings=[q_emb],
                n_results=TOP_K,
                include=["documents", "metadatas", "distances"],
            )

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            # 2. Formatear contexto con trazabilidad (documento + chunk + distancia)
            context_parts = []
            for j, (text, meta, dist) in enumerate(zip(docs, metas, dists)):
                fuente = meta.get("file_name", "desconocido")
                idx = meta.get("chunk_index", "N/A")
                context_parts.append(
                    f"Fragmento {j + 1} [Doc: {fuente}, chunk: {idx}, dist: {dist:.3f}]:\n"
                    f"{text.strip()}"
                )
            full_context = "\n\n---\n\n".join(context_parts)

            # 3. Prompt con grounding estricto (la pauta penaliza fuertemente las alucinaciones)
            system_prompt = (
                "Eres un asistente de salud especializado en el manejo de la obesidad.\n"
                "Tu objetivo es responder de manera clara, pedagogica, util y clinicamente segura.\n"
                "INSTRUCCIONES CRITICAS:\n"
                "1. Responde unicamente basandote en el CONTEXTO provisto abajo.\n"
                "2. Si el contexto no contiene informacion suficiente, di con honestidad: "
                "'No dispongo de suficiente informacion en las guias oficiales para responder a tu pregunta'.\n"
                "3. No inventes datos, medicamentos ni recomendaciones que no esten explicitamente en el contexto.\n"
                "4. Cuando sea pertinente, indica de que documento proviene la informacion.\n"
                "5. Manten un tono empatico pero formal y cientificamente responsable."
            )
            user_prompt = f"CONTEXTO:\n{full_context}\n\nPREGUNTA:\n{question}\n\nRESPUESTA:"

            # 4. Generar respuesta
            response = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            answer = response.choices[0].message.content.strip()
            return answer, full_context

        except Exception as e:
            error_msg = f"Error al procesar la consulta: {e}"
            print(f"[Error] {error_msg}", file=sys.stderr)
            return (f"Lo siento, ocurrio un error al procesar tu pregunta. ({error_msg})", "")


# ---------------------------------------------------------------------------
# Lectura de preguntas
# ---------------------------------------------------------------------------

def leer_preguntas(input_csv_path):
    """Lee el CSV de entrada y devuelve una lista de tuplas (numero, pregunta)."""
    preguntas = []
    with open(input_csv_path, "r", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)

        idx_id, idx_question = 0, 1
        if header:
            header_lower = [h.lower().strip() for h in header]
            if "numero" in header_lower:
                idx_id = header_lower.index("numero")
            elif "id" in header_lower:
                idx_id = header_lower.index("id")
            if "pregunta" in header_lower:
                idx_question = header_lower.index("pregunta")

        for row in reader:
            if not row or len(row) <= max(idx_id, idx_question):
                continue
            preguntas.append((row[idx_id].strip(), row[idx_question].strip()))
    return preguntas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Uso: python Equipo2_RAG_1.py <ruta_preguntas.csv>", file=sys.stderr)
        sys.exit(1)

    input_csv_path = sys.argv[1]
    if not os.path.exists(input_csv_path):
        print(f"[Error] El archivo de entrada '{input_csv_path}' no existe.", file=sys.stderr)
        sys.exit(1)

    rag = RAGSystem()
    if not rag.initialize():
        print("[Error] No se pudo inicializar el sistema RAG. Abortando.", file=sys.stderr)
        sys.exit(1)

    print(f"\nLeyendo preguntas desde '{input_csv_path}'...")
    try:
        preguntas = leer_preguntas(input_csv_path)
    except Exception as e:
        print(f"[Error] Error leyendo el CSV: {e}", file=sys.stderr)
        sys.exit(1)

    total = len(preguntas)
    print(f"Se encontraron {total} preguntas. Iniciando procesamiento...\n")

    # Escritura incremental: cada fila se guarda al instante (resistente a interrupciones).
    # utf-8-sig para que Excel en Windows muestre tildes correctamente.
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["numero", "respuesta", "contexto"])

        for i, (q_id, q_text) in enumerate(preguntas):
            print(f"[{i + 1}/{total}] Procesando pregunta ID {q_id}...")
            respuesta, contexto = rag.query(q_text)
            writer.writerow([q_id, respuesta, contexto])
            csvfile.flush()

    print(f"\nProceso finalizado. Respuestas guardadas en '{OUTPUT_CSV_PATH}'.")


if __name__ == "__main__":
    main()
