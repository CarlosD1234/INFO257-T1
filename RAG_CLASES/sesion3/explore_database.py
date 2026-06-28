import chromadb
import random


# ============================================================
# 1. CARGAR BASE VECTORIAL
# ============================================================

def load_db(db_path="./chroma_db", collection_name="rag_index"):
    """
    Carga la base vectorial ya creada en disco.
    """

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)

    return collection


# ============================================================
# 2. OBTENER TODOS LOS IDS
# ============================================================

def get_all_ids(collection):
    """
    Recupera todos los IDs almacenados en la base.
    """

    data = collection.get()

    return data["ids"]


# ============================================================
# 3. MUESTREO ALEATORIO
# ============================================================

def sample_chunks(collection, n=5):
    """
    Muestra N chunks aleatorios desde la base vectorial.
    """

    data = collection.get()

    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    if len(ids) == 0:
        print("La base está vacía")
        return

    # selección aleatoria
    indices = random.sample(range(len(ids)), min(n, len(ids)))

    print("\n==============================")
    print(f"MOSTRANDO {len(indices)} CHUNKS ALEATORIOS")
    print("==============================\n")

    for i in indices:

        print(f"ID: {ids[i]}")
        print(f"Archivo: {metadatas[i].get('file_name')}")
        print(f"Chunk index: {metadatas[i].get('chunk_index')}")
        print("\nTEXTO:")
        print(documents[i])
        print("\n" + "-" * 80 + "\n")


# ============================================================
# 4. EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    collection = load_db("./chroma_db", "rag_index")

    sample_chunks(collection, n=5)