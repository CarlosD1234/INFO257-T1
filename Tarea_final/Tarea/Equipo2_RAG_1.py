# ===========================================================
# RAG: Pregunta por consola -> retrieval -> respuesta con LLM local
# ===========================================================

import chromadb
from openai import OpenAI

# ===========================================================
# CONFIG
# ===========================================================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "tesis_docs"

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-8b"
CHAT_MODEL = "qwen3-8b"  # <-- AJUSTAR al id exacto que te muestre /v1/models

TOP_K = 4  # cuántos chunks recuperar como contexto

client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)


# ===========================================================
# 1. PREPROCESAR Y EMBEBER LA PREGUNTA
# ===========================================================

def embed_query(text, model=EMBEDDING_MODEL):
    """
    Qwen3-Embedding espera un formato de instrucción específico
    para queries (no para documentos, esos van planos).
    """
    instruct_text = (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query.\n"
        f"Query: {text}"
    )
    response = client.embeddings.create(input=[instruct_text], model=model)
    return response.data[0].embedding


# ===========================================================
# 2. RETRIEVAL DESDE CHROMADB
# ===========================================================

def retrieve_context(query, top_k=TOP_K):
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "file_name": results["metadatas"][0][i]["file_name"],
            "distance": results["distances"][0][i],
        })
    return chunks


# ===========================================================
# 3. ARMAR PROMPT CON CONTEXTO
# ===========================================================

def build_prompt(query, chunks):
    context_block = "\n\n".join(
        f"[Fuente: {c['file_name']}]\n{c['text']}"
        for c in chunks
    )

    system_prompt = (
        "Eres un asistente que responde preguntas SOLO usando el contexto "
        "proporcionado. Si la respuesta no está en el contexto, dilo "
        "explícitamente en lugar de inventar información. Cita el archivo "
        "fuente cuando sea relevante."
    )

    user_prompt = f"Contexto:\n{context_block}\n\nPregunta: {query}"

    return system_prompt, user_prompt


# ===========================================================
# 4. CONSULTAR AL LLM LOCAL
# ===========================================================

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
# 5. LOOP POR CONSOLA
# ===========================================================

def main():
    print("=== RAG local (Qwen3-Embedding-8B + LM Studio) ===")
    print("Escribe 'salir' para terminar.\n")

    while True:
        query = input("Pregunta: ").strip()

        if query.lower() in ("salir", "exit", "quit"):
            break
        if not query:
            continue

        chunks = retrieve_context(query)

        print("\n--- Chunks recuperados ---")
        for c in chunks:
            print(f"  [{c['file_name']}] distancia={c['distance']:.4f}")

        answer = ask_llm(query, chunks)

        print("\n--- Respuesta ---")
        print(answer)
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()