# ===========================================================
# RAG v4 — HYBRID + RERANK + COMPRESSION + CITATION-AWARE
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
    return client.get_collection(collection_name)


# ===========================================================
# 1. QUERY REWRITING
# ===========================================================

def rewrite_query(question):

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
# 2. LOAD ALL DOCS (for BM25)
# ===========================================================

def build_bm25_index(collection):

    data = collection.get(include=["documents", "metadatas"])

    docs = data["documents"]
    metas = data["metadatas"]

    tokenized = [d.lower().split() for d in docs]

    bm25 = BM25Okapi(tokenized)

    return bm25, docs, metas


# ===========================================================
# 3. VECTOR SEARCH
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
# 4. HYBRID SEARCH
# ===========================================================

def hybrid_search(query, collection, bm25, all_docs, k=20):

    emb_docs, emb_meta = vector_search(query, collection, k)

    bm25_scores = bm25.get_scores(query.lower().split())

    top_idx = np.argsort(bm25_scores)[::-1][:k]
    bm25_docs = [all_docs[i] for i in top_idx]

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
# 6. SAFE CONTEXT COMPRESSION (NO LOSES CITATION)
# ===========================================================

def compress_context(documents):
    """
    Compresión SIN perder trazabilidad.
    Mantiene evidencia explícita por DOC ID.
    """

    structured = "\n\n".join(
        [f"[DOC {i}] {doc}" for i, doc in enumerate(documents)]
    )

    prompt = f"""
Eres un asistente médico.

Reduce ruido del texto pero RESPETA estructura.

REGLAS:
- Mantén etiquetas de los documentos y paginas [DOC X, PAG Y]
- No mezcles documentos
- No elimines información médica relevante
- No inventes contenido

TEXTOS:
{structured}

SALIDA:
Resumen estructurado por documento:
"""

    return llm.invoke(prompt).content


# ===========================================================
# 7. GENERACIÓN FINAL CON CITAS
# ===========================================================

def generate_answer(question, context):

    prompt = f"""
Eres un asistente médico especializado en obesidad.

REGLAS OBLIGATORIAS:
- Usa SOLO el contexto
- Cita SIEMPRE usando [DOC X, PAG Y]
- No inventes información
- Si no hay evidencia, dilo explícitamente

FORMATO:

1. Respuesta directa
2. Evidencia (con citas [DOC X])
3. Explicación breve
4. Sugerir preguntas relacionadas para profundizar    

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:
"""

    return llm.invoke(prompt).content


# ===========================================================
# 8. PIPELINE RAG
# ===========================================================

def rag_pipeline(question, collection, bm25, all_docs):

    print("\n--- QUERY ORIGINAL ---")
    print(question)

    rewritten = rewrite_query(question)

    print("\n--- QUERY REWRITTEN ---")
    print(rewritten)

    # retrieval híbrido
    docs = hybrid_search(rewritten, collection, bm25, all_docs, k=20)

    # reranking
    docs = rerank(rewritten, docs)[:10]

    # compression segura
    context = compress_context(docs)

    # generación final
    answer = generate_answer(question, context)

    return answer, context


# ===========================================================
# 9. CHAT LOOP
# ===========================================================

def chat_loop(collection):

    print("\nRAG v3 listo. Escribe 'exit' para salir.\n")

    bm25, all_docs, metas = build_bm25_index(collection)

    while True:

        q = input("Pregunta: ")

        if q.lower() in ["exit", "quit"]:
            break

        answer, context = rag_pipeline(q, collection, bm25, all_docs)

        print("\nRESPUESTA:\n")
        print(answer)

        print("\n--- CONTEXTO ---\n")
        print(context[:3000])


# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":

    collection = load_db("./chroma_db", "rag_index")

    chat_loop(collection)