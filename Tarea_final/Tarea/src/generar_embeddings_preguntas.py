# ===========================================================
# GENERAR EMBEDDINGS DE LAS PREGUNTAS (preguntas.csv -> JSON)
# ===========================================================

import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ===========================================================
# CONFIG
# ===========================================================

INPUT_CSV = "./data_raw/preguntas.csv"
OUTPUT_EMBEDDINGS = "./data_extracted/preguntas_embeddings.json"

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-8b"
BATCH_SIZE = 16

INSTRUCT_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query.\nQuery: "
)

client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")


# ===========================================================
# 1. CARGAR PREGUNTAS
# ===========================================================

def load_questions(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


# ===========================================================
# 2. EMBEBER EN BATCHES
# ===========================================================

def get_embeddings_batch(texts, model=EMBEDDING_MODEL):
    formatted = [f"{INSTRUCT_PREFIX}{t}".replace("\n", " ") for t in texts]
    response = client.embeddings.create(input=formatted, model=model)
    return [d.embedding for d in response.data]


# ===========================================================
# 3. PIPELINE PRINCIPAL
# ===========================================================

def main():
    df = load_questions(INPUT_CSV)
    print(f"Preguntas cargadas: {len(df)}")

    results = []

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Generando embeddings"):
        batch = df.iloc[i:i + BATCH_SIZE]
        numeros = batch["numero"].tolist()
        textos = batch["pregunta"].tolist()

        embeddings = get_embeddings_batch(textos)

        for numero, pregunta, emb in zip(numeros, textos, embeddings):
            results.append({
                "numero": numero,
                "pregunta": pregunta,
                "embedding": emb,
            })

    with open(OUTPUT_EMBEDDINGS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    print(f"Embeddings guardados en: {OUTPUT_EMBEDDINGS}")


if __name__ == "__main__":
    main()