from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings
)

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding
)

import yfinance as yf


# ==========================
# OLLAMA
# ==========================

Settings.llm = Ollama(
    model="llama3",
    request_timeout=120
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


# ==========================
# USER INPUT
# ==========================

symbol = input(
    "Enter Stock Symbol: "
).upper()


# ==========================
# GET STOCK DATA
# ==========================

stock = yf.Ticker(symbol)

info = stock.info

history = stock.history(period="5d")


stock_report = f"""
Company: {info.get('longName')}

Current Price:
{info.get('currentPrice')}

Previous Close:
{info.get('previousClose')}

Market Cap:
{info.get('marketCap')}

Recent Performance:

{history.to_string()}
"""


# ==========================
# CREATE DOCUMENT
# ==========================

documents = [
    Document(
        text=stock_report
    )
]


# ==========================
# CREATE INDEX
# ==========================

index = VectorStoreIndex.from_documents(
    documents
)


# ==========================
# QUERY ENGINE
# ==========================

query_engine = index.as_query_engine()


# ==========================
# CHAT LOOP
# ==========================

while True:

    question = input(
        "\nAsk a question (exit to quit): "
    )

    if question.lower() == "exit":
        break

    response = query_engine.query(
        question
    )

    print("\nAnswer:")
    print(response)