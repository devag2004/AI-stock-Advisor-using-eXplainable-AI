import streamlit as st
import pandas as pd
import json
import ollama
import glob

st.set_page_config(page_title="AI Financial Assistant", page_icon="🤖")

st.title("AI Financial Assistant")

st.write("""
Ask questions about:

• Technical indicators  
• Model accuracy  
• Explainable AI results  
• Forecast predictions
""")


# Load latest ticker used

try:
    with open("latest_run.json","r") as f:
        ticker = json.load(f)["ticker"]
except:
    ticker = None

if ticker is None:
    st.warning("Run the main dashboard first to generate results.")
    st.stop()

st.success(f"Using latest model results for: {ticker}")


# Indicator explanations


INDICATOR_INFO = """
Dataset columns used by the model:

Price Data
Open – Opening price of the stock for the day.
High – Highest traded price during the day.
Low – Lowest traded price during the day.
Close – Final traded price of the day.
Volume – Total number of shares traded.
Dividends – Dividend issued on that date.
Stock Splits – Stock split adjustment factor.

Trend Indicators
SMA_20 – 20-day Simple Moving Average showing overall price trend.
EMA_20 – 20-day Exponential Moving Average giving more weight to recent prices.
EMA_50 – 50-day Exponential Moving Average used for longer trend direction.

MACD Indicators
MACD_12_26_9 – Difference between 12-EMA and 26-EMA measuring momentum.
MACDh_12_26_9 – MACD histogram showing distance between MACD and signal line.
MACDs_12_26_9 – Signal line (9-period EMA of MACD) used for trend signals.

Momentum Indicators
RSI_14 – Relative Strength Index (0-100). Above 70 = overbought, below 30 = oversold.
STOCHk_14_3_3 – Fast stochastic %K measuring position of close within recent range.
STOCHd_14_3_3 – Smoothed stochastic %D used for trading signals.
STOCHh_14_3_3 – Difference between %K and %D indicating momentum shifts.

Volatility Indicators
BBL_20_2.0_2.0 – Lower Bollinger Band showing lower volatility boundary.
BBM_20_2.0_2.0 – Middle Bollinger Band (20-day moving average).
BBU_20_2.0_2.0 – Upper Bollinger Band showing upper volatility boundary.
BBB_20_2.0_2.0 – Bollinger Band width measuring market volatility.
BBP_20_2.0_2.0 – Bollinger Band percentage showing price position within bands.

ATR_14 – Average True Range measuring market volatility.

Volume Indicator
OBV – On Balance Volume tracking buying and selling pressure using volume flow.
"""

MODEL_ARCHITECTURE = """
Model Architecture:

The system uses an Informer-style transformer model for time-series forecasting.

1. Input Layer – historical stock prices and technical indicators are provided as sequences.
2. Feature Embedding – converts indicator values into high-dimensional vectors.
3. Positional Encoding – adds time-step information so the model understands order in the sequence.
4. Encoder Attention Layers – capture long-term dependencies between past market signals.
5. Feed Forward Layers – learn nonlinear relationships between indicators and price movements.
6. Output Layer – predicts the next timestep Open and Close prices.

Attention weights and explainability methods (Integrated Gradients, Saliency, SHAP, Permutation Importance) are used to interpret which indicators influenced predictions.
"""

# Load context from CSV files

def load_context(ticker):

    context = ""

    context += "\nTechnical Indicators:\n"
    context += INDICATOR_INFO

    try:
        forecast = pd.read_csv(f"{ticker}_forecast_summary.csv")
        context += "\nForecast Summary:\n"
        context += forecast.to_string(index=False) + "\n"
    except:
        pass

    try:
        ig = pd.read_csv(f"{ticker}_ig_summary_last10.csv").head(10)
        context += "\nIntegrated Gradients Results:\n"
        context += ig.to_string(index=False) + "\n"
    except:
        pass

    try:
        sal = pd.read_csv(f"{ticker}_saliency_summary_last10.csv").head(10)
        context += "\nSaliency Results:\n"
        context += sal.to_string(index=False) + "\n"
    except:
        pass

    try:
        perm = pd.read_csv(f"{ticker}_permutation_importance.csv").head(10)
        context += "\nPermutation Importance:\n"
        context += perm.to_string(index=False) + "\n"
    except:
        pass

    try:
        shap = pd.read_csv(f"{ticker}_shap_explained_last10.csv").head(10)
        context += "\nSHAP Results:\n"
        context += shap.to_string(index=False) + "\n"
    except:
        pass

    return context


context = load_context(ticker)

# Chat interface

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question about the model")

if question:

    st.session_state.messages.append({"role":"user","content":question})

    with st.chat_message("user"):
        st.write(question)

    prompt = f"""
You are a financial AI assistant explaining a stock forecasting system.

Model context:
Explain indicators using the definitions provided in the context exactly.
{context}

{MODEL_ARCHITECTURE}


User question:
{question}

Instructions:
• Answer clearly in simple language
• Use model results when relevant
• Explain indicators if asked
• around 4 sentences
"""

    response = ollama.chat(
        model="gemma2:2b",
        messages=[{"role":"user","content":prompt}],
        options={"temperature":0.2,"num_predict":400},
        keep_alive="30m"
    )

    answer = response["message"]["content"]

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append({"role":"assistant","content":answer})