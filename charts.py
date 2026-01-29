# charts.py
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px

# Add technical indicators using pandas_ta
import pandas as pd
import numpy as np
import pandas_ta as ta

def add_indicators(df):
    df = df.copy()

    # Trend Indicators
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["EMA_20"] = ta.ema(df["Close"], length=20)
    df["EMA_50"] = ta.ema(df["Close"], length=50)
    macd = ta.macd(df["Close"])
    df = df.join(macd)

    # Momentum Indicators
    df["RSI_14"] = ta.rsi(df["Close"], length=14)
    stoch = ta.stoch(df["High"], df["Low"], df["Close"])
    df = df.join(stoch)

    # Volatility Indicators
    bbands = ta.bbands(df["Close"], length=20, std=2)
    df = df.join(bbands)
    df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Volume Indicator
    df["OBV"] = ta.obv(df["Close"], df["Volume"])

    return df


# Explanations used in app
EXPLANATIONS = {
    "close_open": (
        "Close vs Open — explanation:\n\n"
        "- The **Close** price is where the market finished for the day; **Open** is where it started.\n"
        "- Repeated days where Close > Open indicate bullish sessions; Close < Open indicate bearish.\n"
        "- Use in combination with volume and trend indicators."
    ),
    "moving_averages": (
        "Moving averages — explanation:\n\n"
        "- **SMA (Simple Moving Average)** smooths noise. **EMA** gives more weight to recent prices.\n"
        "- When price crosses above SMA/EMA, it often signals a trend shift upward; crossing below signals possible downtrend.\n"
        "- Crossovers between short-term and long-term MAs are commonly used (e.g., EMA20 vs EMA50)."
    ),
    "bollinger": (
        "Bollinger Bands & ATR — explanation:\n\n"
        "- Bollinger Bands show a volatility envelope around price (upper, middle, lower).\n"
        "- Price above the upper band can indicate overbought conditions and a possible pullback; price near the lower band can indicate oversold.\n"
        "- **ATR** measures volatility magnitude; rising ATR means larger price moves (higher risk/reward). Use ATR for position sizing."
    ),
    "rsi": (
        "RSI — explanation:\n\n"
        "- RSI ranges 0–100. Readings above 70 are typically considered overbought (possible pullback); below 30 considered oversold (possible bounce).\n"
        "- Use with trend context — RSI divergence vs price can signal reversals."
    ),
    "stochastic": (
        "Stochastic Oscillator — explanation:\n\n"
        "- %K and %D lines indicate momentum; values above 80 are often overbought and below 20 oversold.\n"
        "- Crosses of %K over %D in oversold/overbought zones are short-term signals."
    ),
    "volume": (
        "Volume & OBV — explanation:\n\n"
        "- Volume confirms moves: price increases on rising volume are stronger than on low volume.\n"
        "- **OBV (On-Balance Volume)** accumulates volume according to price direction; rising OBV supports bullish trends."
    ),
    "candlestick": (
        "Candlestick chart — explanation:\n\n"
        "- Each candlestick shows Open/High/Low/Close for a period. Look for patterns (e.g., hammer, engulfing) and confirm with volume.\n"
        "- Combine candlestick patterns with indicators for more reliable signals."
    )
}

# -------------------------
# Plotly plot functions
# -------------------------
def plot_close_open(df, title="Close & Open Prices"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    if "Open" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Open"], mode="lines", name="Open", line=dict(dash="dash")))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", hovermode="x unified", height=450)
    return fig

def plot_moving_averages(df, title="Close with Moving Averages"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close", opacity=0.7))
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode="lines", name="SMA 20"))
    if "EMA_50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"], mode="lines", name="EMA 50"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", hovermode="x unified", height=450)
    return fig

def plot_bollinger_bands_and_atr(df, title="Bollinger Bands"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    if "BBU_20_2.0_2.0" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BBU_20_2.0_2.0"], mode="lines", name="Upper Band", line=dict(dash="dash")))
    if "BBM_20_2.0_2.0" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BBM_20_2.0_2.0"], mode="lines", name="Middle Band"))
    if "BBL_20_2.0_2.0" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BBL_20_2.0_2.0"], mode="lines", name="Lower Band", line=dict(dash="dash")))

    # fill between BBU and BBL if present
    if "BBU_20_2.0_2.0" in df.columns and "BBL_20_2.0_2.0" in df.columns:
        fig.add_traces([
            go.Scatter(x=df.index, y=df["BBU_20_2.0_2.0"], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'),
            go.Scatter(x=df.index, y=df["BBL_20_2.0_2.0"], mode='lines', fill='tonexty', fillcolor='rgba(173,216,230,0.1)', line=dict(width=0), showlegend=False, hoverinfo='skip')
        ])
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", hovermode="x unified", height=450)

    # ATR figure
    fig_atr = go.Figure()
    if "ATR_14" in df.columns:
        fig_atr.add_trace(go.Scatter(x=df.index, y=df["ATR_14"], mode="lines", name="ATR 14"))
    fig_atr.update_layout(title="Average True Range (ATR)", xaxis_title="Date", height=250)
    return fig, fig_atr

def plot_rsi_stochastic(df):
    figs = []
    if "RSI_14" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI"))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="blue")
        fig.update_layout(title="Relative Strength Index (RSI 14)", xaxis_title="Date", height=300)
        figs.append(("RSI", fig))
    if "STOCHk_14_3_3" in df.columns and "STOCHd_14_3_3" in df.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df["STOCHk_14_3_3"], name="%K"))
        fig2.add_trace(go.Scatter(x=df.index, y=df["STOCHd_14_3_3"], name="%D"))
        fig2.add_hline(y=80, line_dash="dash", line_color="red")
        fig2.add_hline(y=20, line_dash="dash", line_color="blue")
        fig2.update_layout(title="Stochastic Oscillator", xaxis_title="Date", height=300)
        figs.append(("STOCH", fig2))
    return figs

def plot_volume_and_obv(df):
    fig_vol = go.Figure()
    if "Volume" in df.columns:
        fig_vol.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
    fig_vol.update_layout(title="Daily Trading Volume", xaxis_title="Date", height=300)

    fig_obv = go.Figure()
    if "OBV" in df.columns:
        fig_obv.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV"))
    fig_obv.update_layout(title="On-Balance Volume (OBV)", xaxis_title="Date", height=300)
    return fig_vol, fig_obv

def plot_candlestick_plotly(df, months=6, title_prefix="Candlestick"):
    """
    Returns a Plotly candlestick figure for the last `months` months.
    """
    df_ = df.copy()
    df_.index = pd.to_datetime(df_.index)
    last_date = df_.index.max()
    start_date = last_date - pd.DateOffset(months=months)
    last_n = df_.loc[start_date:last_date].copy()

    if last_n.empty:
        raise ValueError("Not enough data to create candlestick for the requested range.")

    fig = go.Figure(data=[go.Candlestick(
        x=last_n.index,
        open=last_n["Open"],
        high=last_n["High"],
        low=last_n["Low"],
        close=last_n["Close"],
        name="OHLC"
    )])

    # optional add bollinger bands
    if "BBU_20_2.0_2.0" in last_n.columns and "BBL_20_2.0_2.0" in last_n.columns and "BBM_20_2.0_2.0" in last_n.columns:
        fig.add_trace(go.Scatter(x=last_n.index, y=last_n["BBU_20_2.0_2.0"], mode="lines", name="BB Upper", line=dict(color="green", dash="dash")))
        fig.add_trace(go.Scatter(x=last_n.index, y=last_n["BBM_20_2.0_2.0"], mode="lines", name="BB Middle", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=last_n.index, y=last_n["BBL_20_2.0_2.0"], mode="lines", name="BB Lower", line=dict(color="red", dash="dash")))

    fig.update_layout(title=f"{title_prefix} (last {months} months)", xaxis_title="Date", yaxis_title="Price", height=600)
    return fig
