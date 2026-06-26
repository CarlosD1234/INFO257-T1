# ===========================================================
# GENERAR EMBEDDINGS (OpenAI - text-embedding-3-small) + CHROMADB
# ===========================================================
#
# Construye el indice vectorial UNA sola vez a partir de chunks.json.
# Una vez creado, no hay que volver a ejecutarlo: Equipo2_RAG_1.py
# reutiliza este mismo indice para generar respuestas.
#
# Requiere una OPENAI_API_KEY valida (en un archivo .env junto al script
# o como variable de entorno).
#
# Uso:
#     python embeddings.py
# ===========================================================

import os
import json
import sys

import chromadb
from tqdm import tqdm
from openai import OpenAI

# Cargar variables de entorno desde .env si esta disponible
try:
    from dotenv import load_dotenv
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    load_dotenv()  # tambien intenta en el cwd
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===========================================================
# CONFIG  (debe coincidir con Equipo2_RAG_1.py)
# ===========================================================

CHUNKS_PATH = os.path.join(BASE_DIR, "data_extracted", "chunks.json")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "tesis_docs"

# Embeddings: text-embedding-3-small (OpenAI). Mismo modelo que usa el RAG al consultar.
EMBEDDING_MODEL = "text-embedding-3-small"

BATCH_SIZE = 64  # la API de OpenAI acepta lotes grandes

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[Error] No se encontro OPENAI_API_KEY.", file=sys.stderr)
    print("Crea un archivo .env junto al script con: OPENAI_API_KEY=tu_clave", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=api_key)


# ===========================================================
# 1. CARGAR CHUNKS
# ===========================================================

def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================
# 2. GENERAR EMBEDDINGS (en batches)
# ===========================================================

def get_embeddings_batch(texts, model=EMBEDDING_MODEL):
    texts = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(input=texts, model=model)
    return [d.embedding for d in response.data]


# ===========================================================
# 3. INDEXAR EN CHROMADB
# ===========================================================

def build_chroma_collection(chunks, chroma_path=CHROMA_PATH,
                             collection_name=COLLECTION_NAME,
                             batch_size=BATCH_SIZE):

    chroma_client = chromadb.PersistentClient(path=chroma_path)

    # Si ya existe una coleccion previa (p.ej. de otro modelo), la borramos
    # para reconstruir desde cero y evitar mezclar dimensiones.
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Coleccion previa '{collection_name}' eliminada. Reconstruyendo...")
    except Exception:
        pass

    # cosine + metadata del modelo: mismo esquema que espera Equipo2_RAG_1.py
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )

    for i in tqdm(range(0, len(chunks), batch_size), desc="Generando embeddings"):
        batch = chunks[i:i + batch_size]

        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        embeddings = get_embeddings_batch(texts)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    print(f"Total de chunks indexados: {collection.count()}")
    return collection


# ===========================================================
# 4. EJECUCION
# ===========================================================

if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Chunks cargados: {len(chunks)}")

    build_chroma_collection(chunks)

    print(f"TERMINADO - embeddings ({EMBEDDING_MODEL}) guardados en ChromaDB")
