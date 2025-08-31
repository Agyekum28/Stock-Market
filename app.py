# Stock Assistant: Top 3 Trending Stocks with Tavily + Groq LLM
import streamlit as st
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_core.messages import HumanMessage, SystemMessage

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()

# ------------------------------
# Initialize caching and LLM
# ------------------------------
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
set_llm_cache(InMemoryCache())

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_retries=3,
    timeout=120
)

search_tool = TavilySearch()

# ------------------------------
# Functions
# ------------------------------
def get_top_3_tickers():
    """
    Fetch top 3 trending US stock tickers from Tavily search results
    """
    search_query = "Top 3 US trending stock tickers in the US stock market this week"
    search_results = search_tool.invoke({"query": search_query})

    tickers = []
    for result in search_results.get("results", []):
        words = result.get("content", "").split()
        for w in words:
            if len(w) <= 5 and w.isalpha() and w.isupper():  # crude ticker filter
                tickers.append(w)

    top_3_tickers = list(dict.fromkeys(tickers))[:3]
    print("Extracted tickers:", top_3_tickers)
    return top_3_tickers


def get_stock_insights(ticker):
    """
    Generate insights for a given stock ticker using the LLM
    """
    messages = [
        SystemMessage(
            "You're a helpful financial assistant with 20+ years experience analyzing stocks."
        ),
        HumanMessage(content=f"""
        Provide a concise analysis and insights for the stock ticker: {ticker}.
        Include recent news, social media trends, and market analysis. 
        Output in markdown format.
        """)
    ]
    model_response = llm.invoke(messages)
    return model_response.content

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📊 Hottest Stock Tickers of the Week")
st.subheader("Tracking Top 3 Trending Stocks This Week 📈")
st.divider()

st.sidebar.title("📈 Stock Market Insights")
st.sidebar.markdown("""
Stay ahead of the market with our weekly roundup of the hottest stock tickers. 
From breakout movers to trending trades, we track the week’s most talked-about 
stocks so you can spot momentum, gauge sentiment, and make informed decisions.
""")

if st.button("Get Top 3 Trending Tickers This Week"):
    with st.spinner("Fetching top 3 trending tickers..."):
        top_tickers = get_top_3_tickers()

        if not top_tickers:
            st.warning("⚠️ No tickers found. Check Tavily results or widen the filter.")
        else:
            for ticker in top_tickers:
                st.markdown(f"### 📈 {ticker}")
                insights = get_stock_insights(ticker)
                st.markdown(insights, unsafe_allow_html=True)

st.divider()
