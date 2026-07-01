# ===========================================================
# RAG VARIANTE 2 — RETRIEVE-THEN-RERANK CON MMR
# ===========================================================
#
# Idea: en lugar de quedarnos directo con los top-k chunks más
# parecidos a la pregunta (que muchas veces son casi el mismo
# párrafo repetido o fragmentos redundantes), recuperamos un
# conjunto más grande de candidatos (CANDIDATE_K) y luego aplicamos
# Maximal Marginal Relevance (MMR) para elegir los TOP_K finales
# que sean relevantes a la pregunta PERO diversos entre sí.
#
# Esto da contexto que cubre más "ángulos" del tema en vez de
# concentrarse en una sola zona del documento, lo cual suele
# mejorar respuestas a preguntas amplias o de varias partes.
#
# Requiere lo mismo que el script original, no necesita generar
# embeddings nuevos (usa el embedding de la pregunta que ya
# viene precomputado en el JSON), solo numpy adicional:
#   pip install numpy
#
# ===========================================================

import os
import csv
import json
from dotenv import load_dotenv

import numpy as np
import chromadb
from langchain_openai import ChatOpenAI

# ===========================================================
# CONFIG
# ===========================================================

load_dotenv()

CHROMA_PATH = "../chroma_db"
COLLECTION_NAME = "tesis_docs"
PREGUNTAS_EMBEDDINGS_PATH = "../data_extracted/preguntas_tarea_unidad4_2026_embeddings.json"
OUTPUT_PATH = "../data_extracted/RESP_preguntas_tarea_unidad4_2026_RAG_MMR.csv"

TOP_K = 5            # chunks finales usados como contexto
CANDIDATE_K = 15      # candidatos iniciales sobre los que aplicar MMR
MMR_LAMBDA = 0.6      # 1.0 = solo relevancia, 0.0 = solo diversidad

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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================
# 2. RETRIEVAL + MMR
# ===========================================================

def retrieve_candidates(query_embedding, collection, k=CANDIDATE_K):
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "embeddings"],
    )
    return res["documents"][0], res["metadatas"][0], res["embeddings"][0]


def cosine_sim(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def mmr_select(query_embedding, doc_embeddings, k=TOP_K, lambda_param=MMR_LAMBDA):
    """
    Selecciona k índices de doc_embeddings maximizando relevancia
    a la query y minimizando redundancia entre los ya seleccionados.
    """
    n = len(doc_embeddings)
    if n == 0:
        return []
    k = min(k, n)

    sim_to_query = [cosine_sim(query_embedding, e) for e in doc_embeddings]
    candidates = list(range(n))
    selected = []

    # primer elemento: el más relevante a la query
    first = max(candidates, key=lambda i: sim_to_query[i])
    selected.append(first)
    candidates.remove(first)

    while len(selected) < k and candidates:
        mmr_scores = {}
        for i in candidates:
            redundancia = max(cosine_sim(doc_embeddings[i], doc_embeddings[j]) for j in selected)
            mmr_scores[i] = lambda_param * sim_to_query[i] - (1 - lambda_param) * redundancia
        best = max(mmr_scores, key=mmr_scores.get)
        selected.append(best)
        candidates.remove(best)

    return selected


def retrieve(query_embedding, collection, k=TOP_K):
    docs, metas, embeddings = retrieve_candidates(query_embedding, collection, k=CANDIDATE_K)
    idxs = mmr_select(query_embedding, embeddings, k=k)
    docs_sel = [docs[i] for i in idxs]
    metas_sel = [metas[i] for i in idxs]
    return docs_sel, metas_sel


# ===========================================================
# 3. CONSTRUCCIÓN DE CONTEXTO
# ===========================================================

def build_context(docs, metadatas):
    parts = []
    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        meta = meta or {}
        fuente = meta.get("file_name") or meta.get("source") or f"chunk_{i}"
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
        id = item["id"]
        pregunta = item["pregunta"]
        embedding = item["embedding"]

        docs, metadatas = retrieve(embedding, collection, k=TOP_K)
        context = build_context(docs, metadatas)
        respuesta = generate_answer(pregunta, context)

        print(f"\n--- Pregunta {id} ---")
        print(f"P: {pregunta}")
        print(f"R: {respuesta}")

        resultados.append({
            "id": id,
            "answer": respuesta,
            "context": context,
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "answer", "context"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\nResultados guardados en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()