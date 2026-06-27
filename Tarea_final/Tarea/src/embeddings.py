# ===========================================================
# GENERAR EMBEDDINGS (LM Studio - Qwen3-Embedding-8B) + CHROMADB
# ===========================================================

import json
import chromadb
from tqdm import tqdm
from openai import OpenAI

# ===========================================================
# CONFIG
# ===========================================================

CHUNKS_PATH = "./data_extracted/chunks.json"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "tesis_docs"

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-8b"

BATCH_SIZE = 16 

client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")


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
    collection = chroma_client.get_or_create_collection(name=collection_name)

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
# 4. EJECUCIÓN
# ===========================================================

if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Chunks cargados: {len(chunks)}")

    build_chroma_collection(chunks)

    print("TERMINADO - embeddings guardados en ChromaDB")