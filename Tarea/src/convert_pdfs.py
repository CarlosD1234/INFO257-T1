# ===========================================================
# PARSEO DE PDFs A MARKDOWN + CHUNKS (VERSIÓN OPTIMIZADA)
# ===========================================================

import os
import glob
import json
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat


# ===========================================================
# CONFIG
# ===========================================================

pipeline_options = PdfPipelineOptions(do_ocr=False)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


# ===========================================================
# 1. PARSEO PDF → MARKDOWN
# ===========================================================

def parse_pdf_to_markdown(pdf_path):
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )
    pieces = splitter.split_text(markdown)

    chunks = []
    for i, p in enumerate(pieces):
        p = p.strip()

        if len(p) < 20:
            continue

        chunks.append({
            "id": f"{file_name}_chunk_{i}",
            "text": p,
            "metadata": {
                "file_name": file_name,
                "chunk_index": i
            }
        })

    return chunks


# ===========================================================
# 4. GUARDAR CHUNKS
# ===========================================================

def save_chunks(all_chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)


# ===========================================================
# 5. PIPELINE PRINCIPAL
# ===========================================================

def process_pdfs(pdf_folder, md_folder):
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
# 6. EJECUCIÓN
# ===========================================================

if __name__ == "__main__":
    pdf_folder = "./data_raw"
    md_folder = "./data_extracted"
    chunks_path = os.path.join(md_folder, "chunks.json")

    print("INICIANDO PARSEO PDF → MARKDOWN")
    chunks = process_pdfs(pdf_folder, md_folder)

    save_chunks(chunks, chunks_path)
    print(f"Chunks guardados en: {chunks_path}")

    print("TERMINADO")