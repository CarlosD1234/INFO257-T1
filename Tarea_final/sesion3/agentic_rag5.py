# ===========================================================
# RAG AGENTIC — LANGGRAPH (CLÍNICO + MODULAR)
# ===========================================================

import os
import numpy as np
import chromadb

from typing import TypedDict, Annotated, List
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from rank_bm25 import BM25Okapi

# evitar logs de HF
os.environ["SENTENCE_TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import logging as hf_logging
hf_logging.set_verbosity_error()

from tqdm import tqdm
tqdm.disable = True
import warnings
warnings.filterwarnings("ignore")

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
# PERFIL PACIENTE (MEMORIA CLÍNICA)
# ===========================================================

PATIENT_PROFILE = """
Paciente: Matthieu
Edad: 41 años
Riesgo: alto riesgo de abandono de tratamiento
Motivación: baja para hacer deporte
Condición: colesterol ligeramente elevado
Seguimiento: última cita medical hace 6 meses
Objetivo: mejorar adherencia y motivación sin culpabilizar
"""


# ===========================================================
# STATE
# ===========================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    use_rag: bool
    context: str
    docs: list


# ===========================================================
# DB
# ===========================================================

def load_db(path="./chroma_db", name="rag_index"):
    client = chromadb.PersistentClient(path=path)
    return client.get_collection(name)


# ===========================================================
# ROUTER (planner)
# ===========================================================

def router_node(state: State):
    question = state["messages"][-1].content

    prompt = f"""
Clasifica la intención del usuario:

Opciones:
- CHAT (saludos, conversación simple)
- MEDICAL_RAG (salud, obesidad, sueño, nutrición, peso)
- OTHER

Pregunta:
{question}

Respuesta (una palabra):
"""

    decision = llm.invoke(prompt).content.strip()

    return {
        "use_rag": decision == "MEDICAL_RAG"
    }


# ===========================================================
# QUERY REWRITING
# ===========================================================

def rewrite_query(q: str):
    prompt = f"""
Convierte la siguiente pregunta en una consulta optimizada para buscar información en documentos médicos sobre obesidad.

Reglas:
- Mantén significado clínico
- Expande conceptos implícitos
- Agrega sinónimos médicos relevantes
- Usa más términos técnicos cuando sea posible
- NO respondas la pregunta

Pregunta: {q}

Consulta optimizada:
"""
    return llm.invoke(prompt).content.strip()


# ===========================================================
# HYBRID SEARCH
# ===========================================================

def build_bm25(collection):
    data = collection.get(include=["documents"])
    docs = data["documents"]
    bm25 = BM25Okapi([d.lower().split() for d in docs])
    return bm25, docs


def vector_search(query, collection, k=20):
    q_emb = embedding_model.encode([query]).tolist()

    res = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas"]
    )

    return res["documents"][0]


def hybrid_search(query, collection, bm25, all_docs, k=20):
    vdocs = vector_search(query, collection, k)

    b_scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(b_scores)[::-1][:k]
    bdocs = [all_docs[i] for i in top_idx]

    return list(dict.fromkeys(vdocs + bdocs))


# ===========================================================
# RERANKING (cross-encoder simple por embeddings)
# ===========================================================

def rerank(query, docs):
    q = embedding_model.encode([query])[0]
    d = embedding_model.encode(docs)

    scores = np.dot(d, q)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return [r[0] for r in ranked]


# ===========================================================
# CONTEXT COMPRESSION (SAFE + CITABLE)
# ===========================================================

def compress_context(docs):
    """
    Compresión SIN perder trazabilidad.
    Mantiene evidencia explícita por DOC ID.
    """

    structured = "\n\n".join(
        [f"[DOC {i}] {doc}" for i, doc in enumerate(docs)]
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
# RAG NODE
# ===========================================================

def rag_node(state: State, collection, bm25, all_docs):

    q = state["messages"][-1].content

    rq = rewrite_query(q)

    docs = hybrid_search(rq, collection, bm25, all_docs)
    docs = rerank(rq, docs)[:10]

    context = compress_context(docs)

    return {
        "context": context,
        "docs": docs
    }


# ===========================================================
# WRITER CLÍNICO
# ===========================================================

def writer_node(state: State):

    q = state["messages"][-1].content
    context = state.get("context", "")

    prompt = f"""
Eres un asistente médico especializado en obesidad.

Debes responder con enfoque clínico y motivacional.

PERFIL DEL PACIENTE:
{PATIENT_PROFILE}

REGLAS:
- Usa SOLO el contexto
- Sé empático y personaliza la respuesta al perfil del paciente
- Cita como [DOC X, PAG Y] SIEMPRE que uses información del contexto
- Si no sabes, dilo

CONTEXTO:
{context}

PREGUNTA:
{q}

FORMATO:
1. Respuesta médica
2. Evidencia
3. Recomendación práctica
4. Mensaje motivacional
5. Sugerencia de preguntas para profundizar
"""

    res = llm.invoke(prompt)

    return {"messages": [res]}


# ===========================================================
# ROUTING FINAL
# ===========================================================

def should_rag(state: State):
    return "rag" if state["use_rag"] else "final"


# ===========================================================
# GRAPH
# ===========================================================

def build_graph(collection):

    bm25, all_docs = build_bm25(collection)

    graph = StateGraph(State)

    graph.add_node("router", router_node)

    graph.add_node("rag", lambda s: rag_node(s, collection, bm25, all_docs))
    graph.add_node("writer", writer_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        should_rag,
        {
            "rag": "rag",
            "final": "writer"
        }
    )

    graph.add_edge("rag", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


# ===========================================================
# MAIN LOOP
# ===========================================================

if __name__ == "__main__":

    collection = load_db("./chroma_db", "rag_index")

    app = build_graph(collection)

    print("\nRAG AGENT LISTO\n")

    while True:
        q = input("Pregunta: ")

        if q.lower() in ["exit", "quit"]:
            break

        result = app.invoke({
            "messages": [{"role": "user", "content": q}]
        })

        print("\nRESPUESTA:\n")
        print(result["messages"][-1].content)