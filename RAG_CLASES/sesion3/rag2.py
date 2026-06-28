# ===========================================================
# RAG ONLINE v1 — WITH QUERY REWRITING
# ===========================================================

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

import warnings
import os

warnings.filterwarnings("ignore")

os.environ["SENTENCE_TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from huggingface_hub import logging as hf_logging
hf_logging.set_verbosity_error()

# ===========================================================
# 1. MODELOS
# ===========================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key="[ENCRYPTION_KEY]"

)

# ===========================================================
# 2. CARGAR BASE VECTORIAL
# ===========================================================

def load_db(db_path="./chroma_db", collection_name="rag_index"):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)
    return collection

# ===========================================================
# 3. QUERY REWRITING
# ===========================================================

def rewrite_query(question):
    """
    Transforma la pregunta del usuario en una consulta más útil
    para búsqueda semántica en documentos médicos.
    """

    prompt = f"""
Convierte la siguiente pregunta en una consulta optimizada para buscar información en documentos médicos sobre obesidad.

Reglas:
- Mantén significado clínico
- Expande conceptos implícitos
- Agrega sinónimos médicos relevantes
- Agrega una lista de palabras clave y sinónimos al final
- Usa más términos técnicos cuando sea posible
- NO respondas la pregunta

Pregunta: {question}

Consulta optimizada:
"""

    return llm.invoke(prompt).content.strip()

# ===========================================================
# 4. RETRIEVAL
# ===========================================================

def retrieve(query, collection, k=50):
    """
    Recupera chunks relevantes desde ChromaDB.
    """

    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas

# ===========================================================
# 5. CONTEXTO
# ===========================================================

def build_context(documents, metadatas):
    """
    Construye el contexto que se entrega al LLM.
    """

    context_parts = []

    for i, (doc, meta) in enumerate(zip(documents, metadatas)):

        source = meta.get("file_name", "unknown")

        context_parts.append(
            f"[CHUNK {i} | SOURCE: {source}]\n{doc}"
        )

    return "\n\n".join(context_parts)

# ===========================================================
# 6. GENERACIÓN
# ===========================================================

def generate_answer(question, context):
    """
    LLM responde SOLO usando el contexto.
    """

    prompt = f"""
Eres un asistente médico especializado en obesidad.

Responde la pregunta usando el contexto.

IMPORTANTE:
- Usa la información del contexto aunque no sea literal.
- Puedes combinar ideas de distintos fragmentos.
- Si la información es parcialmente relevante, infiere la respuesta.
- Si no hay nada relevante, recién ahí di que no sabes.

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:
"""

    response = llm.invoke(prompt)
    return response.content

# ===========================================================
# 7. PIPELINE RAG v1
# ===========================================================

def rag_pipeline(question, collection):
    """
    Pipeline completo:
    1. rewriting
    2. retrieval
    3. contexto
    4. generación
    """

    rewritten_query = rewrite_query(question)

    print("\n==============================")
    print("QUERY ORIGINAL  :", question)
    print("QUERY REWRITTEN :", rewritten_query)
    print("==============================\n")

    documents, metadatas = retrieve(rewritten_query, collection, k=10)

    context = build_context(documents, metadatas)

    answer = generate_answer(question, context)

    return answer, context

# ===========================================================
# 8. CHAT LOOP
# ===========================================================

def chat_loop(collection):
    """
    Interfaz simple tipo chatbot en consola.
    """

    print("\nRAG v2 listo (con rewriting). Escribe 'exit' para salir.\n")

    while True:

        question = input("Pregunta: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer, context = rag_pipeline(question, collection)

        print("\nRESPUESTA:\n")
        print(answer)

        print("\n--- CONTEXTO USADO (recortado) ---\n")
        print(context[:3000])

# ===========================================================
# 9. MAIN
# ===========================================================

if __name__ == "__main__":

    collection = load_db("./chroma_db", "rag_index")

    chat_loop(collection)