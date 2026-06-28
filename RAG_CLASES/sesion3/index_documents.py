# ===========================================================
# RAG OFFLINE PIPELINE
# Construcción de base vectorial a partir de PDFs
# ===========================================================

import os
import glob
from tqdm import tqdm

import chromadb
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter


# ============================================================
# MODELOS GLOBALES
# ============================================================

converter = DocumentConverter()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 1. PARSEO PDF CON DOCLING
# ============================================================

def parse_pdf(pdf_path):
    """
    Convierte PDF → DoclingDocument
    """

    result = converter.convert(pdf_path)
    doc = result.document

    file_name = os.path.basename(pdf_path)

    return file_name, doc


# ============================================================
# 2. CHUNKING ROBUSTO (MARKDOWN ONLY)
# ============================================================

def extract_chunks(file_name, doc):
    """
    Cada chunk = párrafo basado en markdown.

    Decisión de diseño:
    Se usa SOLO export_to_markdown para evitar problemas de estructura.
    """

    chunks = []
    chunk_id = 0

    markdown = doc.export_to_markdown()

    paragraphs = markdown.split("\n\n")

    for p in paragraphs:
        p = p.strip()

        if len(p) < 20:
            continue

        chunks.append({
            "id": f"{file_name}_chunk_{chunk_id}",
            "text": p,
            "metadata": {
                "file_name": file_name,
                "chunk_index": chunk_id
            }
        })

        chunk_id += 1

    return chunks


# ============================================================
# 3. CARGA DE DOCUMENTOS
# ============================================================

def load_all_documents(pdf_folder):
    """
    PDF → Docling → chunks
    """

    all_chunks = []

    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

    print(f"PDFs encontrados: {len(pdf_files)}")

    for pdf_path in tqdm(pdf_files):

        file_name, doc = parse_pdf(pdf_path)

        chunks = extract_chunks(file_name, doc)

        all_chunks.extend(chunks)

    print(f"Total chunks creados: {len(all_chunks)}")

    return all_chunks


# ============================================================
# 4. VECTOR DB (CHROMA)
# ============================================================

def create_vector_db(db_path="./chroma_db"):
    """
    Base vectorial persistente
    """

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="rag_index")

    return client, collection


# ============================================================
# 5. INDEXACIÓN
# ============================================================

def index_chunks(collection, chunks):
    """
    Genera embeddings + guarda en ChromaDB
    """

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    if not texts:
        raise ValueError("No hay chunks para indexar")

    print("Generando embeddings...")

    embeddings = embedding_model.encode(texts).tolist()

    print("Guardando en ChromaDB...")

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    print("Indexación completada")


# ============================================================
# 6. PIPELINE OFFLINE COMPLETO
# ============================================================

def build_index(pdf_folder, db_path="./chroma_db"):
    """
    Pipeline completo offline.
    Se ejecuta una sola vez.
    """

    print("===================================")
    print("BUILD INDEX PIPELINE (OFFLINE)")
    print("===================================")

    print("Leyendo PDFs...")
    chunks = load_all_documents(pdf_folder)

    print("Inicializando base vectorial...")
    client, collection = create_vector_db(db_path)

    print("Indexando chunks...")
    index_chunks(collection, chunks)

    print("TERMINADO")
    print("===================================")


# ============================================================
# 7. EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    pdf_folder = "./pdfs"

    build_index(pdf_folder)