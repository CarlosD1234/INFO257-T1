# ===========================================================
# RAG v3 — HYBRID SEARCH + RERANKING + CONTEXT COMPRESSION
# ===========================================================

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
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
# MODELOS
# ===========================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key="[ENCRYPTION_KEY]"

)

# ===========================================================
# DB
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
- Usa más términos técnicos cuando sea posible
- NO respondas la pregunta

Pregunta: {question}

Consulta optimizada:
"""

    return llm.invoke(prompt).content.strip()

# ===========================================================
# 2. HYBRID SEARCH SETUP (BM25)
# ===========================================================

def build_bm25_index(collection):
    docs = collection.get()["documents"]
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, docs

# ===========================================================
# 3. EMBEDDING RETRIEVAL
# ===========================================================

def vector_search(query, collection, k=50):

    q_emb = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas"]
    )

    return results["documents"][0], results["metadatas"][0]

# ===========================================================
# 4. HYBRID FUSION
# ===========================================================

def hybrid_search(query, collection, bm25, all_docs, k=20):

    # embeddings
    emb_docs, emb_meta = vector_search(query, collection, k)

    # BM25
    bm25_scores = bm25.get_scores(query.lower().split())

    bm25_top_idx = np.argsort(bm25_scores)[::-1][:k]
    bm25_docs = [all_docs[i] for i in bm25_top_idx]

    # fusion simple (deduplicación)
    combined = list(dict.fromkeys(emb_docs + bm25_docs))

    return combined

# ===========================================================
# 5. RERANKING (embedding-based)
# ===========================================================

def rerank(query, documents):

    q_emb = embedding_model.encode([query])[0]
    d_embs = embedding_model.encode(documents)

    scores = np.dot(d_embs, q_emb)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [d for d, _ in ranked]

# ===========================================================
# 6. CONTEXT COMPRESSION
# ===========================================================

def compress_context(documents):
    """
    Reduce ruido antes de enviar al LLM final.
    """

    joined = "\n\n".join(documents)

    prompt = f"""
Resume los siguientes fragmentos de documentos médicos eliminando redundancia.
Mantén solo información relevante para responder preguntas sobre obesidad.

TEXTO:
{joined}

RESUMEN:
"""

    return llm.invoke(prompt).content

# ===========================================================
# 7. ANSWER GENERATION
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

    return llm.invoke(prompt).content

# ===========================================================
# 8. PIPELINE RAG v2
# ===========================================================

def rag_pipeline(question, collection, bm25, all_docs):

    # 1. rewriting
    rewritten = rewrite_query(question)

    print("\nQUERY ORIGINAL:", question)
    print("QUERY REWRITTEN:", rewritten)

    # 2. hybrid retrieval
    docs = hybrid_search(rewritten, collection, bm25, all_docs, k=20)

    # 3. reranking
    docs = rerank(rewritten, docs)[:10]

    # 4. compression
    context = compress_context(docs)

    # 5. generation
    answer = generate_answer(question, context)

    return answer, context

# ===========================================================
# 9. CHAT LOOP
# ===========================================================

def chat_loop(collection):

    print("\nRAG v2 listo. Escribe 'exit' para salir.\n")

    bm25, all_docs = build_bm25_index(collection)

    while True:

        q = input("Pregunta: ")

        if q.lower() in ["exit", "quit"]:
            break

        answer, context = rag_pipeline(q, collection, bm25, all_docs)

        print("\nRESPUESTA:\n")
        print(answer)

        print("\n--- CONTEXTO COMPRIMIDO ---\n")
        print(context[:3000])

# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":

    collection = load_db("./chroma_db", "rag_index")

    chat_loop(collection)