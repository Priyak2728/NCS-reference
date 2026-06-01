from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------------
# Load Document
# -----------------------------

with open("data1.txt", "r", encoding="utf-8") as f:
    text = f.read()

# -----------------------------
# Chunking Functions
# -----------------------------

def no_chunking(text):
    return [text]


def fixed_chunking(text, chunk_size=100):

    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


def overlap_chunking(
    text,
    chunk_size=100,
    overlap=30
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += (chunk_size - overlap)

    return chunks


def semantic_chunking(text):

    sections = text.split("\n\n")

    chunks = []

    current_chunk = ""

    for section in sections:

        if (
            "Policy" in section
            and current_chunk != ""
        ):

            chunks.append(current_chunk)

            current_chunk = section

        else:

            current_chunk += "\n" + section

    chunks.append(current_chunk)

    return chunks


# -----------------------------
# Menu
# -----------------------------

print("\nCHUNKING STRATEGIES")
print("1 - No Chunking")
print("2 - Fixed Chunking")
print("3 - Overlap Chunking")
print("4 - Semantic Chunking")

choice = input("\nChoose Strategy: ")

if choice == "1":

    chunks = no_chunking(text)
    strategy = "No Chunking"

elif choice == "2":

    chunks = fixed_chunking(text)
    strategy = "Fixed Chunking"

elif choice == "3":

    chunks = overlap_chunking(text)
    strategy = "Overlap Chunking"

elif choice == "4":

    chunks = semantic_chunking(text)
    strategy = "Semantic Chunking"

else:

    print("Invalid Choice")
    exit()

# -----------------------------
# Display Chunks
# -----------------------------

print("\n")
print("=" * 60)
print(f"USING: {strategy}")
print("=" * 60)

for i, chunk in enumerate(chunks):

    print(f"\nChunk {i+1}")
    print("-" * 40)
    print(chunk)

# -----------------------------
# Embedding Model
# -----------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Create Embeddings
# -----------------------------

embeddings = model.encode(chunks)

embeddings = np.array(
    embeddings
).astype("float32")

# -----------------------------
# Create FAISS Index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("\nFAISS INDEX CREATED")

# -----------------------------
# User Question
# -----------------------------

query = input("\nAsk a Question: ")

query_embedding = model.encode([query])

query_embedding = np.array(
    query_embedding
).astype("float32")

# -----------------------------
# Similarity Search
# -----------------------------

k = 2

distances, indices = index.search(
    query_embedding,
    k
)

# -----------------------------
# Show Retrieved Chunks
# -----------------------------

print("\n")
print("=" * 60)
print("TOP RETRIEVED CHUNKS")
print("=" * 60)

for idx in indices[0]:

    print("\n")
    print(chunks[idx])
    print("-" * 40)

print("\nDemo Complete")