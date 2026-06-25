#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para la Tarea RAG - Inteligencia Artificial
Uso: python Equipo2_RAG_1.py preguntas.csv
"""

import os
import sys
import csv

# Cargar variables de entorno si existe un archivo .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Verificar la clave de API
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[Error] No se encontró la variable de entorno OPENAI_API_KEY.", file=sys.stderr)
    print("Asegúrate de crear un archivo .env en la carpeta de ejecución con: OPENAI_API_KEY=tu_clave_api", file=sys.stderr)


# Importaciones condicionales de LangChain para permitir la validación del formato sin requerir todas las dependencias
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import ChatOpenAI
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("\n[Advertencia] Algunas librerías de LangChain no están instaladas.", file=sys.stderr)
    print("Para instalarlas ejecuta: pip install langchain langchain-community langchain-openai langchain-chroma langchain-huggingface sentence-transformers chromadb pypdf\n", file=sys.stderr)

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("\n[Advertencia] La librería docling no está instalada.", file=sys.stderr)
    print("Para instalarla ejecuta: pip install docling\n", file=sys.stderr)


class SimpleRAGSystem:
    def __init__(self, doc_path=None, persist_dir="chroma_db_tarea"):
        self.doc_path = doc_path
        self.persist_dir = persist_dir
        self.vector_db = None
        self.retriever = None
        self.llm = None
        
    def initialize(self):
        """Inicializa el RAG indexando el documento base si está disponible."""
        if not LANGCHAIN_AVAILABLE:
            print("[Error] No se puede inicializar el RAG sin las librerías de LangChain.", file=sys.stderr)
            return False
            
        print("[1/3] Inicializando modelos de embeddings y LLM...")
        # Inicializar embeddings locales gratuitos
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Inicializar el LLM con la API key obtenida
        self.llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0)
        
        # Si ya existe base de datos persistida, cargarla
        if os.path.exists(self.persist_dir) and len(os.listdir(self.persist_dir)) > 0:
            print("[2/3] Cargando base de datos vectorial existente de ChromaDB...")
            self.vector_db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        else:
            # Si no existe, indexar el documento base
            if not self.doc_path or not os.path.exists(self.doc_path):
                print(f"[Error] No se encontró el documento base '{self.doc_path}' para indexar y crear la base vectorial.", file=sys.stderr)
                return False
                
            print(f"[2/3] Procesando documento con Docling...")
            
            if DOCLING_AVAILABLE:
                try:
                    # Usar Docling para convertir el PDF a Markdown limpio
                    print(f"  -> Convirtiendo PDF '{self.doc_path}' a Markdown usando Docling...")
                    converter = DocumentConverter()
                    result = converter.convert(self.doc_path)
                    markdown_text = result.document.export_to_markdown()
                    
                    # Envolver el contenido en un objeto Document de LangChain
                    documents = [
                        Document(
                            page_content=markdown_text,
                            metadata={"source": self.doc_path}
                        )
                    ]
                except Exception as e:
                    print(f"[Error] Falló la conversión con Docling: {e}. Intentando fallback con PyPDFLoader...", file=sys.stderr)
                    loader = PyPDFLoader(self.doc_path)
                    documents = loader.load()
            else:
                print("  -> [Advertencia] Docling no está disponible. Usando PyPDFLoader como fallback...", file=sys.stderr)
                loader = PyPDFLoader(self.doc_path)
                documents = loader.load()
            
            # Aplicar splitter semántico o por caracteres
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=1000,
                chunk_overlap=150
            )
            chunks = splitter.split_documents(documents)
            
            # Crear y persistir la base vectorial
            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
            print(f"Indexación completada. Creados {len(chunks)} chunks.")
            
        # Definir retriever
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={"k": 4} # Obtener los 4 fragmentos más relevantes
        )
        print("[3/3] Sistema RAG inicializado con éxito.")
        return True

    def query(self, question):
        """Consulta el RAG para obtener una respuesta y el contexto utilizado."""
        if not LANGCHAIN_AVAILABLE or not self.vector_db:
            # Fallback simple si no está disponible LangChain o la base de datos
            return "Respuesta simulada (instala LangChain y configura la API para respuestas reales).", "Contexto simulado."
            
        try:
            # 1. Recuperar contexto
            retrieved_docs = self.retriever.invoke(question)
            
            # 2. Formatear el contexto recuperado
            context_list = []
            for i, doc in enumerate(retrieved_docs):
                source_info = f"[Doc: {os.path.basename(doc.metadata.get('source', 'unknown'))}, Pág: {doc.metadata.get('page', 'N/A')}]"
                context_list.append(f"Fragmento {i+1} {source_info}:\n{doc.page_content.strip()}")
                
            full_context = "\n\n---\n\n".join(context_list)
            
            # 3. Construir el prompt del sistema y usuario para Grounded Generation
            system_prompt = (
                "Eres un asistente de salud especializado en el manejo de la obesidad.\n"
                "Tu objetivo es responder de manera clara, pedagógica, útil y clínicamente segura.\n"
                "INSTRUCCIONES CRÍTICAS:\n"
                "1. Responde únicamente basándote en el CONTEXTO provisto abajo.\n"
                "2. Si el contexto no contiene información suficiente para responder, di con honestidad: 'No dispongo de suficiente información en las guías oficiales para responder a tu pregunta'.\n"
                "3. No inventes datos, medicamentos ni recomendaciones que no estén explícitamente en el contexto.\n"
                "4. Mantén un tono empático pero formal y científicamente responsable."
            )
            
            user_prompt = f"CONTEXTO:\n{full_context}\n\nPREGUNTA:\n{question}\n\nRESPUESTA:"
            
            # 4. Generar respuesta con el LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = self.llm.invoke(messages)
    
            return response.content, full_context
            
        except Exception as e:
            error_msg = f"Error al procesar la consulta: {str(e)}"
            print(f"[Error] {error_msg}", file=sys.stderr)
            return f"Lo siento, ocurrió un error al procesar tu pregunta. ({error_msg})", ""


def main():
    # Validar argumentos de línea de comando
    if len(sys.argv) < 2:
        print("Uso del script:", file=sys.stderr)
        print("  python Equipo2_RAG_1.py <ruta_preguntas.csv>", file=sys.stderr)
        sys.exit(1)
        
    input_csv_path = sys.argv[1]
    output_csv_path = "respuestas.csv"
    
    if not os.path.exists(input_csv_path):
        print(f"[Error] El archivo de entrada '{input_csv_path}' no existe.", file=sys.stderr)
        sys.exit(1)
        
    # Inicializar el RAG utilizando el PDF de la Sesión 2 como corpus base inicial
    # Ajustar la ruta del PDF dependiendo de dónde se ejecute el script
    pdf_path = "../sesion2-Agentes-LangGraph/articulo.pdf"
    if not os.path.exists(pdf_path):
        pdf_path = "sesion2-Agentes-LangGraph/articulo.pdf"
        
    rag = SimpleRAGSystem(doc_path=pdf_path)
    rag_initialized = rag.initialize()
    
    if not rag_initialized:
        print("[Advertencia] Iniciando en MODO SIMULACIÓN (sin embeddings ni LLM real).", file=sys.stderr)

    preguntas_procesadas = []
    
    print(f"\nLeyendo preguntas desde '{input_csv_path}'...")
    try:
        # Detectar delimitador y codificación
        with open(input_csv_path, 'r', encoding='utf-8-sig') as csvfile:
            # Leer el archivo utilizando csv.reader
            reader = csv.reader(csvfile)
            header = next(reader, None) # Leer cabecera
            
            # Mapear columnas para manejar variaciones (ej. 'numero' vs 'id' y 'pregunta')
            idx_id = 0
            idx_question = 1
            
            if header:
                # Normalizar cabecera a minúsculas
                header_lower = [h.lower().strip() for h in header]
                if "numero" in header_lower:
                    idx_id = header_lower.index("numero")
                elif "id" in header_lower:
                    idx_id = header_lower.index("id")
                    
                if "pregunta" in header_lower:
                    idx_question = header_lower.index("pregunta")
            
            for row in reader:
                if not row or len(row) <= max(idx_id, idx_question):
                    continue
                q_id = row[idx_id].strip()
                q_text = row[idx_question].strip()
                preguntas_procesadas.append((q_id, q_text))
                
    except Exception as e:
        print(f"[Error] Error leyendo el archivo CSV: {e}", file=sys.stderr)
        sys.exit(1)
        
    total_preguntas = len(preguntas_procesadas)
    print(f"Se encontraron {total_preguntas} preguntas. Iniciando procesamiento...\n")
    
    # Procesar preguntas y escribir respuestas en tiempo real
    respuestas_generadas = []
    
    # Usamos utf-8-sig para que Excel en Windows muestre tildes y caracteres especiales correctamente
    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        # Escribir la cabecera exacta requerida por la pauta
        writer.writerow(["numero", "respuesta", "contexto"])
        
        for index, (q_id, q_text) in enumerate(preguntas_procesadas):
            print(f"[{index + 1}/{total_preguntas}] Procesando pregunta ID {q_id}...")
            
            # Obtener respuesta y contexto del RAG
            respuesta, contexto = rag.query(q_text)
            
            # Escribir fila inmediatamente para asegurar el guardado ante interrupciones
            writer.writerow([q_id, respuesta, contexto])
            
    print(f"\nProceso finalizado. Respuestas guardadas en '{output_csv_path}'.")


if __name__ == "__main__":
    main()
