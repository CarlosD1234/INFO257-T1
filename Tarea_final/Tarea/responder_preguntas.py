# ===========================================================
# RAG SOBRE 100 PREGUNTAS -> respuestas.csv
# ===========================================================

import json
import csv
from tqdm import tqdm
from openai import OpenAI
import chromadb

# ===========================================================
# CONFIG
# ===========================================================

EMBEDDINGS_PATH = "./preguntas_embeddings.json"
OUTPUT_CSV = "./respuestas.csv"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "tesis_docs"

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
CHAT_MODEL = "qwen3-8b"  # <-- AJUSTAR al id exacto que muestre /v1/models
TOP_K = 4

client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)


# ===========================================================
# 1. CARGAR EMBEDDINGS DE LAS PREGUNTAS
# ===========================================================

def load_question_embeddings(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================
# 2. RETRIEVAL (ya con el embedding pre-calculado)
# ===========================================================

def retrieve_context(query_embedding, top_k=TOP_K):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "file_name": results["metadatas"][0][i]["file_name"],
        })
    return chunks


# ===========================================================
# 3. PROMPT + LLAMADA AL LLM
# ===========================================================

def build_prompt(query, chunks):
    context_block = "\n\n".join(
        f"[Fuente: {c['file_name']}]\n{c['text']}"
        for c in chunks
    )
    system_prompt = (
        "Eres un asistente que responde preguntas SOLO usando el contexto "
        "proporcionado. Si la respuesta no está en el contexto, dilo "
        "explícitamente en lugar de inventar información."
    )
    user_prompt = f"Contexto:\n{context_block}\n\nPregunta: {query}"
    return system_prompt, user_prompt


def ask_llm(query, chunks, model=CHAT_MODEL):
    system_prompt, user_prompt = build_prompt(query, chunks)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ===========================================================
# 4. PIPELINE PRINCIPAL
# ===========================================================

def main():
    items = load_question_embeddings(EMBEDDINGS_PATH)
    print(f"Preguntas cargadas: {len(items)}")

    rows = []

    for item in tqdm(items, desc="Generando respuestas"):
        numero = item["numero"]
        pregunta = item["pregunta"]
        embedding = item["embedding"]

        chunks = retrieve_context(embedding)
        respuesta = ask_llm(pregunta, chunks)

        rows.append({
            "id": numero,
            "pregunta": pregunta,
            "respuesta": respuesta,
        })

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "pregunta", "respuesta"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Respuestas guardadas en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()