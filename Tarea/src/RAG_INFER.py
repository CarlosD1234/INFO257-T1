# ===========================================================
# RAG VARIANTE 3 — TOP_K AMPLIADO + INFERENCIA PERMITIDA
# ===========================================================
#
# Idea: igual de simple que el script original (mismo retrieval,
# mismo embedding precomputado de la pregunta), pero con dos
# cambios puntuales:
#
#   1) TOP_K sube de 5 a 10 -> más contexto disponible.
#   2) El prompt deja de exigir "responde SOLO con lo literal del
#      contexto" y en cambio permite que el modelo INFIERA y
#      conecte información entre distintos chunks, siempre que
#      esa inferencia esté razonablemente respaldada por el
#      contexto (no que invente datos que no tienen ninguna base).
#      Además le pedimos que distinga explícitamente qué es dato
#      directo del contexto y qué es inferencia suya, para que el
#      CSV resultante siga siendo auditable.
#
# Todo lo demás (ChromaDB, modelo, formato de salida) es idéntico
# al script original.
#
# ===========================================================

import os
import csv
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
PREGUNTAS_EMBEDDINGS_PATH = "../data_extracted/preguntas_tarea_unidad4_2026_embeddings.json"
OUTPUT_PATH = "../data_extracted/RESP_preguntas_tarea_unidad4_2026_RAG_INFER.csv"

TOP_K = 10  # antes 5

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
# 2. RETRIEVAL (idéntico al original, solo cambia TOP_K)
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
    parts = []
    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        meta = meta or {}
        fuente = meta.get("file_name") or meta.get("source") or f"chunk_{i}"
        parts.append(f"[Fuente: {fuente}]\n{doc}")
    return "\n\n---\n\n".join(parts)


# ===========================================================
# 4. GENERACIÓN — prompt que permite inferencia razonada
# ===========================================================

def generate_answer(question, context):
    prompt = f"""Eres un asistente experto que responde preguntas apoyándote en el contexto proporcionado.

CONTEXTO:
{context}

PREGUNTA:
{question}

INSTRUCCIONES:
- Usa el contexto como base principal de tu respuesta.
- Puedes INFERIR y conectar información entre distintos fragmentos del contexto
  cuando sea razonable hacerlo (por ejemplo, relacionar una causa mencionada en
  un fragmento con un efecto mencionado en otro), aunque esa conexión no esté
  dicha de forma literal y explícita en un solo lugar.
- No inventes datos, cifras, nombres o hechos que no tengan ninguna base en el
  contexto. La inferencia debe ser una conclusión razonable a partir de lo que
  SÍ está presente, no información externa o inventada.
- Si haces una inferencia (en vez de citar algo directo del contexto), indícalo
  brevemente, por ejemplo con una frase como "esto sugiere que..." o "se puede
  inferir que...", para que quede claro qué es dato directo y qué es tu lectura.
- Si el contexto simplemente no da ninguna base ni siquiera para inferir una
  respuesta, dilo explícitamente.
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