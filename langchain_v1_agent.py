from langchain_ollama import ChatOllama
from langchain.tools import tool
import yfinance as yf


# ==========================
# LLM
# ==========================

llm = ChatOllama(
    model="llama3"
)


# ==========================
# TOOLS
# ==========================

@tool
def stock_price(symbol: str) -> str:
    """
    Get current stock price.
    Example: AAPL, MSFT, TSLA
    """

    stock = yf.Ticker(symbol)

    info = stock.info

    return (
        f"Company: {info.get('longName')}\n"
        f"Current Price: {info.get('currentPrice')}\n"
        f"Previous Close: {info.get('previousClose')}\n"
        f"Market Cap: {info.get('marketCap')}"
    )


@tool
def stock_history(symbol: str) -> str:
    """
    Get recent stock performance.
    """

    stock = yf.Ticker(symbol)

    hist = stock.history(period="5d")

    return hist.to_string()


# ==========================
# AGENT LOOP
# ==========================

while True:

    symbol = input(
        "\nEnter stock symbol (or exit): "
    )

    if symbol.lower() == "exit":
        break

    print("\nFetching market data...\n")

    current_data = stock_price.invoke(symbol)

    history_data = stock_history.invoke(symbol)

    prompt = f"""
    You are a professional financial analyst.

    Analyze this stock.

    CURRENT DATA:
    {current_data}

    PRICE HISTORY:
    {history_data}

    Provide:

    1. Current Situation
    2. Trend Analysis
    3. Risks
    4. Opportunities
    5. Investment Summary

    Do not provide financial advice.
    """

    response = llm.invoke(prompt)

    print("\n========================")
    print("ANALYSIS")
    print("========================\n")

    print(response.content)
    