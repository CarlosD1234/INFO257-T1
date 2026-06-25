# ===========================================================
# PARSEO DE PDFs A MARKDOWN + CHUNKS (VERSIÓN OPTIMIZADA)
# ===========================================================

import os
import glob
from tqdm import tqdm
from docling.document_converter import DocumentConverter


# ===========================================================
# CONFIG
# ===========================================================

converter = DocumentConverter()

# ===========================================================
# 1. PARSEO PDF → MARKDOWN
# ===========================================================

def parse_pdf_to_markdown(pdf_path):
    """
    Convierte PDF → Markdown (una sola vez)
    """
    result = converter.convert(pdf_path)
    doc = result.document

    markdown = doc.export_to_markdown()
    file_name = os.path.basename(pdf_path)

    return file_name, markdown


# ===========================================================
# 2. GUARDAR MARKDOWN
# ===========================================================

def save_markdown(file_name, markdown, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    md_name = file_name.replace(".pdf", ".md")
    output_path = os.path.join(output_folder, md_name)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)


# ===========================================================
# 3. CHUNKING DESDE MARKDOWN
# ===========================================================

def extract_chunks(file_name, markdown):
    """
    Cada chunk = párrafo basado en markdown
    """
    chunks = []
    chunk_id = 0

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


# ===========================================================
# 4. PIPELINE PRINCIPAL
# ===========================================================

def process_pdfs(pdf_folder, md_folder):
    """
    Pipeline completo:
    PDF → Markdown (guardado) → chunks (en memoria)
    """
    all_chunks = []

    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

    print("===================================")
    print(f"PDFs encontrados: {len(pdf_files)}")
    print("===================================")

    for pdf_path in tqdm(pdf_files):

        # 1. Parseo
        file_name, markdown = parse_pdf_to_markdown(pdf_path)

        # 2. Guardar markdown
        save_markdown(file_name, markdown, md_folder)

        # 3. Chunking
        chunks = extract_chunks(file_name, markdown)
        all_chunks.extend(chunks)

    print("===================================")
    print(f"Total chunks creados: {len(all_chunks)}")
    print("===================================")

    return all_chunks


# ===========================================================
# 5. EJECUCIÓN
# ===========================================================

if __name__ == "__main__":
    pdf_folder = "./data_raw"
    md_folder = "./data_extracted"

    print("INICIANDO PARSEO PDF → MARKDOWN")
    chunks = process_pdfs(pdf_folder, md_folder)
    print("TERMINADO")