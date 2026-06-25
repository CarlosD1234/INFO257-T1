# ===========================================================
# RAG ONLINE SIMPLE
# Consulta sobre una base vectorial ya indexada
# ===========================================================

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
import warnings
warnings.filterwarnings("ignore")
import os

# evitar logs de HF
os.environ["SENTENCE_TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import logging as hf_logging
hf_logging.set_verbosity_error()

from tqdm import tqdm
tqdm.disable = True

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
    """
    Carga la base Chroma ya existente en disco.
    """

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)

    return collection


# ===========================================================
# 3. RETRIEVAL (BUSQUEDA SEMÁNTICA)
# ===========================================================

def retrieve(question, collection, k=10):
    """
    Busca los chunks más similares a la pregunta.
    """

    query_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
    query_embeddings=query_embedding,
    n_results=k,
    include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas


# ===========================================================
# 4. CONSTRUIR CONTEXTO
# ===========================================================

def build_context(documents, metadatas):
    context_parts = []

    for i, (doc, meta) in enumerate(zip(documents, metadatas)):

        source = meta.get("file_name", "unknown")

        context_parts.append(
            f"[CHUNK {i} | SOURCE: {source}]\n{doc}"
        )

    return "\n\n".join(context_parts)


# ===========================================================
# 5. GENERACIÓN DE RESPUESTA
# ===========================================================

def generate_answer(question, context):
    """
    LLM responde usando SOLO el contexto recuperado.
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
# 6. PIPELINE RAG
# ===========================================================

def rag_pipeline(question, collection):
    """
    Flujo completo RAG:
    pregunta → retrieval → contexto → LLM
    """

    documents, metadatas = retrieve(question, collection)

    context = build_context(documents, metadatas)

    answer = generate_answer(question, context)

    return answer, context


# ===========================================================
# 7. INTERFAZ SIMPLE (CONSOLE)
# ===========================================================

def chat_loop(collection):
    """
    Interacción simple tipo chatbot en consola.
    """

    print("RAG listo. Escribe una pregunta (o 'exit'):")

    while True:

        question = input("\nPregunta: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer, context = rag_pipeline(question, collection)

        print("\nRESPUESTA:")
        print(answer)

        print("\n--- CONTEXTO USADO ---")
        print(context[:10000])  # recortado para no saturar


# ===========================================================
# 8. MAIN
# ===========================================================

if __name__ == "__main__":

    collection = load_db("./chroma_db", "rag_index")

    chat_loop(collection)