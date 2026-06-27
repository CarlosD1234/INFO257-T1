# ===========================================================
# RAG SIMPLE — USANDO CHROMADB EXISTENTE + PREGUNTAS EMBEBIDAS
# ===========================================================
#
# Requiere:
#   pip install python-dotenv chromadb openai langchain-openai
#
# Archivo .env en la raíz del proyecto con:
#   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
#
# ===========================================================

import os
import json
from dotenv import load_dotenv

import chromadb
from langchain_openai import ChatOpenAI

# ===========================================================
# CONFIG
# ===========================================================

load_dotenv()

CHROMA_PATH = "../chroma_db"
COLLECTION_NAME = "tesis_docs"
PREGUNTAS_EMBEDDINGS_PATH = "../data_extracted/preguntas_embeddings.json"
OUTPUT_PATH = "../data_extracted/rag_resultados.json"

TOP_K = 5

# LLM de generación. La api_key se toma de .env (OPENAI_API_KEY)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ===========================================================
# 1. CARGAR DB Y PREGUNTAS
# ===========================================================

def load_collection(path=CHROMA_PATH, name=COLLECTION_NAME):
    client = chromadb.PersistentClient(path=path)
    return client.get_collection(name)


def load_preguntas(path=PREGUNTAS_EMBEDDINGS_PATH):
    """
    Carga el JSON generado previamente, con la forma:
    [{"numero": ..., "pregunta": ..., "embedding": [...]}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================
# 2. RETRIEVAL
# ===========================================================

def retrieve(query_embedding, collection, k=TOP_K):
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"],
    )
    return res["documents"][0], res["metadatas"][0]


# ===========================================================
# 3. CONSTRUCCIÓN DE CONTEXTO
# ===========================================================

def build_context(docs, metadatas):
    """
    Junta los chunks recuperados en un solo bloque de texto,
    identificando la fuente de cada uno si está disponible en metadata.
    Ajusta la clave "source" según cómo guardaste tu metadata en chunks.json.
    """
    parts = []
    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        fuente = (meta or {}).get("source", f"chunk_{i}")
        parts.append(f"[Fuente: {fuente}]\n{doc}")
    return "\n\n---\n\n".join(parts)


# ===========================================================
# 4. GENERACIÓN
# ===========================================================

def generate_answer(question, context):
    prompt = f"""Eres un asistente que responde preguntas basándose ÚNICAMENTE en el contexto proporcionado.

CONTEXTO:
{context}

PREGUNTA:
{question}

INSTRUCCIONES:
- Responde solo con información presente en el contexto.
- Si el contexto no contiene la respuesta, dilo explícitamente, no inventes.
- Sé claro y conciso.
"""
    response = llm.invoke(prompt)
    return response.content


# ===========================================================
# 5. PIPELINE PRINCIPAL
# ===========================================================

def main():
    collection = load_collection()
    preguntas = load_preguntas()

    print(f"Preguntas cargadas: {len(preguntas)}")
    print(f"Chunks en ChromaDB: {collection.count()}")

    resultados = []

    for item in preguntas:
        numero = item["numero"]
        pregunta = item["pregunta"]
        embedding = item["embedding"]

        docs, metadatas = retrieve(embedding, collection, k=TOP_K)
        context = build_context(docs, metadatas)
        respuesta = generate_answer(pregunta, context)

        print(f"\n--- Pregunta {numero} ---")
        print(f"P: {pregunta}")
        print(f"R: {respuesta}")

        resultados.append({
            "numero": numero,
            "pregunta": pregunta,
            "respuesta": respuesta,
            "docs_recuperados": docs,
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print(f"\nResultados guardados en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()