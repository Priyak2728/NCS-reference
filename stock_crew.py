from crewai import Agent, Task, Crew
import yfinance as yf


# ==========================
# GET STOCK DATA
# ==========================

symbol = input("Enter Stock Symbol: ").upper()

stock = yf.Ticker(symbol)

info = stock.info

history = stock.history(period="5d")

stock_data = f"""
Company: {info.get('longName', 'N/A')}
Current Price: {info.get('currentPrice', 'N/A')}
Previous Close: {info.get('previousClose', 'N/A')}
Market Cap: {info.get('marketCap', 'N/A')}

Recent 5 Day History:

{history.to_string()}
"""

print("\nStock Data Retrieved Successfully\n")


# ==========================
# AGENTS
# ==========================

research_agent = Agent(
    role="Stock Market Researcher",
    goal="Analyze stock performance and trends",
    backstory="""
    You are a professional stock market researcher.
    You analyze company performance and stock trends.
    """,
    llm="ollama/llama3",
    verbose=True
)

risk_agent = Agent(
    role="Risk Analyst",
    goal="Identify potential investment risks",
    backstory="""
    You specialize in identifying market risks,
    volatility, weaknesses and threats.
    """,
    llm="ollama/llama3",
    verbose=True
)

advisor_agent = Agent(
    role="Investment Strategist",
    goal="Create executive investment summary",
    backstory="""
    You prepare executive reports for investors.
    Focus on opportunities and risks.
    Never provide financial advice.
    """,
    llm="ollama/llama3",
    verbose=True
)


# ==========================
# TASKS
# ==========================

research_task = Task(
    description=f"""
    Analyze the following stock information:

    {stock_data}

    Provide:

    1. Company Overview
    2. Recent Stock Trend
    3. Key Observations
    4. Market Position
    """,
    expected_output="Detailed stock research report",
    agent=research_agent
)

risk_task = Task(
    description=f"""
    Review this stock information:

    {stock_data}

    Identify:

    1. Risks
    2. Volatility Indicators
    3. Weaknesses
    4. Potential Challenges
    """,
    expected_output="Risk analysis report",
    agent=risk_agent
)

advisor_task = Task(
    description="""
    Using the outputs from the researcher and risk analyst,
    create an executive summary.

    Include:

    - Business Summary
    - Opportunities
    - Risks
    - Overall Assessment

    Do not provide financial advice.
    """,
    expected_output="Executive stock summary",
    agent=advisor_agent
)


# ==========================
# CREW
# ==========================

crew = Crew(
    agents=[
        research_agent,
        risk_agent,
        advisor_agent
    ],
    tasks=[
        research_task,
        risk_task,
        advisor_task
    ],
    verbose=True
)


# ==========================
# RUN CREW
# ==========================

result = crew.kickoff()

print("\n")
print("=" * 60)
print("FINAL STOCK REPORT")
print("=" * 60)
print(result)