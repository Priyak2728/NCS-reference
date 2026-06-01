from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# -----------------------------
# STEP 1 — Load Documents
# -----------------------------

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = text.split("\n\n")

print("\nDOCUMENT CHUNKS:\n")

for i, chunk in enumerate(chunks):
    print(f"{i+1}. {chunk}")

# -----------------------------
# STEP 2 — Embeddings
# -----------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

chunk_embeddings = model.encode(chunks)

chunk_embeddings = np.array(
    chunk_embeddings
).astype("float32")

# -----------------------------
# STEP 3 — FAISS Vector DB
# -----------------------------

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(chunk_embeddings)

print("\nFAISS index created.")

# -----------------------------
# STEP 4 — User Query
# -----------------------------

query = input("\nAsk a question: ")

query_embedding = model.encode([query])

query_embedding = np.array(
    query_embedding
).astype("float32")

# -----------------------------
# STEP 5 — Retrieval
# -----------------------------

k = 2

distances, indices = index.search(
    query_embedding,
    k
)

retrieved_chunks = []

print("\nRETRIEVED CHUNKS:\n")

for idx in indices[0]:
    print(chunks[idx])
    print("------------------")
    retrieved_chunks.append(chunks[idx])

# -----------------------------
# STEP 6 — Prompt Augmentation
# -----------------------------

context = "\n".join(retrieved_chunks)

prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

# -----------------------------
# STEP 7 — Send to Ollama
# -----------------------------

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
)

result = response.json()

# -----------------------------
# STEP 8 — Final Answer
# -----------------------------

print("\nFINAL ANSWER:\n")

print(result["response"])