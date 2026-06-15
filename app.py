
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import bcrypt
import re
import pandas_ta as ta
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import mplfinance as mpf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ollama
# ML libs (original script)
import random
import json
from datetime import datetime, timedelta
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except Exception:
    CAPTUM_AVAILABLE = False


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


START_DATE = "2019-01-01"
SEQ_LEN = 60         
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-4
D_MODEL = 128        
N_HEADS = 4
N_ENCODER_LAYERS = 3
N_DECODER_LAYERS = 2
FFN_DIM = 256
DROPOUT = 0.1
TEST_DAYS = 120
PATIENCE = 8

def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        return

INTRO_HTML = r"""
<style>
:root {
  --bg: #0f172a;
  --card: #0b1220;
  --accent: #1e40af;
  --text: #e6eef8;
}
body { margin:0; padding:0; background:var(--bg); color:var(--text); font-family: 'Segoe UI', Roboto, sans-serif;}
.container { display:flex; align-items:center; justify-content:center; height:420px; }
.slidewrap { width:100%; max-width:900px; height:300px; overflow:hidden; position:relative; }
.slide {
  position:absolute;
  width:100%;
  top:50%; left:50%;
  transform:translate(-50%, -50%);
  white-space:normal;
  text-align:center;
  transition: transform 0.7s ease, opacity 0.7s ease;
  opacity:0;
}
.slide.visible {
  transform:translate(-30%, -50%);
  opacity:1;
}
.big {
  font-size:64px;
  font-weight:800;
  letter-spacing:1px;
  margin-bottom:12px;
  color: #ffffff;
}
.title {
  font-size:36px;
  font-weight:700;
  margin-bottom:10px;
  color:#dbeafe;
}
.copy {
  font-size:18px;
  color:#cbd5e1;
  max-width:900px;
  margin: 0 auto;
  line-height:1.4;
}

/* Continue button style */
.btn {
  display:inline-block;
  margin-top:26px;
  padding:10px 22px;
  border-radius:10px;
  background:linear-gradient(90deg,#2563eb,#7dd3fc);
  color:#002244;
  font-weight:700;
  cursor:pointer;
  box-shadow: 0 6px 20px rgba(2,6,23,0.6);
  border:none;
}
.footer { text-align:center; margin-top:16px; color:#94a3b8; font-size:13px; }
</style>

<div class="container">
  <div style="text-align:center; width:100%;">
    <div class="slidewrap" id="slidewrap">
      <div class="slide" id="s0">
        <div class="big">Welcome</div>
      </div>
      <div class="slide" id="s1">
        <div class="title">Stock Advisor — what you needed.</div>
      </div>
      <div class="slide" id="s2">
        <div class="title">This stock advisor is what you always needed.</div>
        <div class="copy">We value transparency and trust. Not only do we forecast prices using robust models, we also explain the factors that affect those forecasts and suggest trading decisions — helping you trade smarter, not blindly.</div>
      </div>
    </div>
  </div>
</div>

<script>
const slides = [document.getElementById('s0'), document.getElementById('s1'), document.getElementById('s2')];
let idx = 0;
const D = 6000; // 

function showIndex(i){
  slides.forEach((s, j) => {
    s.classList.remove('visible');
    s.style.transform = 'translate(-50%, -50%)';
  });
  const s = slides[i];
  setTimeout(()=> {
    s.classList.add('visible');
  }, 20);
}

// show first slide
showIndex(0);

let timer = setInterval(() => {
  if (idx < slides.length - 1) {
    idx++;
    showIndex(idx);
  } else {
    clearInterval(timer);
    document.getElementById('continueBtn').style.display = 'inline-block';
  }
}, D);

// Continue button posts a message to parent; Streamlit can't reliably capture postMessage cross-deploy,
// so we also include a Streamlit-side Continue button below the component for deterministic behavior.
const continueBtn = document.getElementById('continueBtn');
if (continueBtn) {
  continueBtn.addEventListener('click', () => {
    const payload = { type: 'continue' };
    window.parent.postMessage({ isStreamlitMessage: true, type: 'CUSTOM_EVENT', data: payload }, "*");
  });
}
</script>
"""

def show_intro():
    components.html(INTRO_HTML, height=520, scrolling=False)
    if st.button("Continue"):
        st.session_state.screen = "auth"
        # no need to force rerun — state change will reflect on next interaction
        safe_rerun()



# --- Explanations used under charts (short, actionable) ---
EXPL_OPEN_CLOSE = """
**How to read:** On a price chart, each day has two key numbers: the Open (price at market open) and Close (price at market close). Hover over the lines to see the exact values for any date.
**Why it matters:** When the Close is higher than Open (often shown as a green candle), it means the stock price rose that day, signaling buying pressure. When Close is lower than Open (red candle), it indicates selling pressure. The difference shows intraday sentiment and strength of buying or selling.
"""

EXPL_MA = """
**How to read:** SMA (Simple Moving Average) and EMA (Exponential Moving Average) smooth out short-term price noise over a window length (e.g., 20 days). On charts, you'll see them as lines tracking overall price trends.
**How it helps:** If the current price stays above these averages, it suggests a strong upward trend (bullish). Crossovers—like the price crossing above the EMA—can act as “green lights” to buy, while crossing below can signal selling opportunities.
"""

EXPL_BB_ATR = """
**How to read:** Bollinger Bands have three lines—
BBM (the middle line) is a simple moving average of price.
BBU (upper band) is above the middle line by a certain amount (usually 2 standard deviations).
BBL (lower band) is below the middle line by the same amount.  
**How it helps:** If price approaches or crosses the upper band (BBU), it is considered “overbought” and may bounce back down; if it touches the lower band (BBL), it might rebound upward.
The BBB is the band width showing how wide the bands are (volatility measure), and BBP shows where the current price sits inside the bands (0 = at bottom, 1 = at top).
ATR (Average True Range) is a volatility measure showing how much price moves on average. Rising ATR means the price is making bigger moves recently, which can mean higher risk but also opportunity.
"""

EXPL_RSI_STOCH = """
**How to read:** RSI ranges from 0 to 100; readings above 70 mean the stock might be “overbought” (too high, might fall), below 30 means “oversold” (too low, might rise).
Stochastic oscillator uses two lines: %K (fast, sensitive to price changes) and %D (a smoothed version of %K). When %K crosses above %D below 20, it can mean a good buying opportunity; when %K crosses below %D above 80, it could be a sell signal.  
**How it helps:** Shows momentum extremes and possible trend reversals earlier than price alone.
"""

EXPL_VOL_OBV = """
**How to read:** Volume bars show how many shares changed hands each day.
OBV adds daily volume when price closes up and subtracts volume when price closes down—showing flow of buying vs selling pressure over time.  
**How it helps:** If OBV rises with price, it confirms the strength of the move; if price rises but OBV falls, it signals a possible weakening rally (divergence).
"""

EXPL_CANDLE = """
**How to read:** Candlesticks show open/high/low/close per day. 
A green candle means the Close is above the Open—its a “bullish” day.
A red candle means the Close is below the Open—its a “bearish” day.
The vertical line (wick) shows the high and low prices during the day.
Look for candle shapes and patterns like “hammer” (potential bottom reversal) or “engulfing” (strong reversal patterns).
Volume combined with certain candle patterns strengthens these signals.  
**How it helps:** Gives a detailed snapshot of market psychology and potential turning points.
"""

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", 1))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", 5))

if not DATABASE_URL:
    st.error("DATABASE_URL not set in .env. Add it and restart.")
    st.stop()

if "pg_pool" not in st.session_state:
    try:
        st.session_state.pg_pool = SimpleConnectionPool(
            minconn=DB_POOL_MIN,
            maxconn=DB_POOL_MAX,
            dsn=DATABASE_URL,
            sslmode="require"
        )
    except Exception as e:
        st.error(f"Failed to create DB pool: {e}")
        st.stop()

def get_conn():
    pool = st.session_state.get("pg_pool", None)
    if pool is None:
        raise RuntimeError("DB pool not available in session.")
    return pool.getconn()

def put_conn(conn):
    pool = st.session_state.get("pg_pool", None)
    if pool is None:
        return
    pool.putconn(conn)

def init_db():
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash BYTEA NOT NULL,
                age INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cur.close()
    except Exception as e:
        st.error(f"DB init error: {e}")
    finally:
        if conn is not None:
            put_conn(conn)

init_db()

if "screen" not in st.session_state:
    st.session_state.screen = "intro"   

if "show_signup" not in st.session_state:
    st.session_state.show_signup = False

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.age = None



def create_user(username, password, age):
    username = username.strip()
    if not username or not password:
        return False, "Username and password are required."
    try:
        age_int = int(age)
    except Exception:
        return False, "Invalid age."
    if age_int < 18:
        return False, "You must be 18 or older to create an account."
    if not re.match(r"^[A-Za-z0-9_.-]{3,30}$", username):
        return False, "Username must be 3-30 chars (letters/numbers . _ -)."
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password_hash, age) VALUES (%s, %s, %s)",
                    (username, psycopg2.Binary(hashed), age_int))
        conn.commit()
        cur.close()
        return True, "Account created."
    except Exception as e:
        err = str(e).lower()
        if "unique" in err:
            return False, "Username already exists."
        print("create_user error:", e)
        return False, "Database error."
    finally:
        if conn is not None:
            put_conn(conn)

def get_user(username):
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, username, password_hash, age, created_at FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
        cur.close()
        return row
    except Exception as e:
        print("get_user error:", e)
        return None
    finally:
        if conn is not None:
            put_conn(conn)

def verify_password(stored_hash, provided_password):
    if stored_hash is None:
        return False
    stored_bytes = stored_hash.tobytes() if isinstance(stored_hash, memoryview) else bytes(stored_hash)
    try:
        return bcrypt.checkpw(provided_password.encode("utf-8"), stored_bytes)
    except Exception:
        return False

def authenticate_user(username, password):
    row = get_user(username)
    if not row:
        return False, None
    ok = verify_password(row[2], password)
    return ok, row


# Indicator 
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

@st.cache_data(ttl=3600)
def fetch_ticker_history(ticker_symbol: str, start_date="2019-01-01"):
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start=start_date, end=None, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker_symbol}")
    for c in ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]:
        if c not in df.columns:
            df[c] = np.nan
    df = df[["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]].copy()
    for c in ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).copy()
    if df.shape[0] < 10:
        raise ValueError(f"Not enough rows for {ticker_symbol} after cleaning ({df.shape[0]} rows).")
    return df

def plot_open_close(df: pd.DataFrame, title="Open & Close Prices (interactive)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Open"], name="Open", mode="lines+markers",
                             hovertemplate="Date: %{x}<br>Open: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", mode="lines+markers",
                             hovertemplate="Date: %{x}<br>Close: %{y:.2f}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price(INR)", legend=dict(orientation="h"))
    return fig

def plot_moving_averages(df: pd.DataFrame, title="Price with SMA/EMA"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(width=2)))
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20", line=dict(dash="dash")))
    if "EMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], name="EMA 20", line=dict(dash="dot")))
    if "EMA_50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"], name="EMA 50"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", legend=dict(orientation="h"))
    return fig

def plot_bollinger_and_atr(df: pd.DataFrame, title="Bollinger Bands & ATR"):
    bbl = next((c for c in df.columns if c.upper().startswith("BBL_")), None)
    bbm = next((c for c in df.columns if c.upper().startswith("BBM_")), None)
    bbu = next((c for c in df.columns if c.upper().startswith("BBU_")), None)
    atr = "ATR_14" if "ATR_14" in df.columns else None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.08,
                        subplot_titles=(title, "ATR (Volatility)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", mode="lines"), row=1, col=1)
    if bbl:
        fig.add_trace(go.Scatter(x=df.index, y=df[bbl], name="Lower Band", line=dict(dash="dash")), row=1, col=1)
    if bbm:
        fig.add_trace(go.Scatter(x=df.index, y=df[bbm], name="Middle Band"), row=1, col=1)
    if bbu:
        fig.add_trace(go.Scatter(x=df.index, y=df[bbu], name="Upper Band", line=dict(dash="dash")), row=1, col=1)
    if bbl and bbu:
        x_vals = np.concatenate([df.index.values, df.index.values[::-1]])
        y_vals = np.concatenate([df[bbu].values, df[bbl].values[::-1]])
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, fill="toself", name="BB range",
                                 fillcolor="rgba(173,216,230,0.1)", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
    if atr:
        fig.add_trace(go.Scatter(x=df.index, y=df[atr], name="ATR_14", line=dict(color="purple")), row=2, col=1)
    fig.update_layout(height=650, legend=dict(orientation="h"))
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price(INR)", row=1, col=1)
    fig.update_yaxes(title_text="ATR", row=2, col=1)
    return fig

def plot_rsi_stoch(df: pd.DataFrame, title: str = "RSI & Stochastic"):
    df = df.copy()
    rsi_col = next((c for c in df.columns if c.upper().startswith("RSI_")), None)
    stoch_k_col = next((c for c in df.columns if c.upper().startswith("STOCHK") or c.upper().startswith("STOCHK")), None)
    stoch_d_col = next((c for c in df.columns if c.upper().startswith("STOCHD") or c.upper().startswith("STOCHD")), None)
    if (stoch_k_col is None or stoch_d_col is None) and {"High", "Low", "Close"}.issubset(df.columns):
        try:
            st = ta.stoch(df["High"], df["Low"], df["Close"])
            for c in st.columns:
                if c not in df.columns:
                    df[c] = st[c]
            stoch_k_col = next((c for c in df.columns if c.upper().startswith("STOCHK") or c.upper().startswith("STOCHK")), None)
            stoch_d_col = next((c for c in df.columns if c.upper().startswith("STOCHD") or c.upper().startswith("STOCHD")), None)
        except Exception:
            pass
    if rsi_col is None and "Close" in df.columns:
        try:
            df["RSI_14"] = ta.rsi(df["Close"], length=14)
            rsi_col = "RSI_14"
        except Exception:
            rsi_col = None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45],
                        vertical_spacing=0.08,
                        subplot_titles=("Relative Strength Index (RSI)", "Stochastic Oscillator (%K / %D)"))
    x = df.index
    if rsi_col and rsi_col in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[rsi_col], name="RSI", mode="lines"), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=x, y=[None]*len(x), name="RSI (missing)"), row=1, col=1)
    if stoch_k_col and stoch_k_col in df.columns and stoch_d_col and stoch_d_col in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[stoch_k_col], name="%K", mode="lines"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=df[stoch_d_col], name="%D", mode="lines"), row=2, col=1)
    else:
        fig.add_trace(go.Scatter(x=x, y=[None]*len(x), name="%K/%D (missing)"), row=2, col=1)
    fig.update_layout(title=title, height=620, legend=dict(orientation="h"))
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def plot_volume_and_obv(df: pd.DataFrame, title="Volume & OBV"):
    dfc = df.copy()
    vol_col = "Volume" if "Volume" in dfc.columns else None
    obv_col = "OBV" if "OBV" in dfc.columns else None
    if obv_col is None and vol_col and {"Close", "Volume"}.issubset(dfc.columns):
        try:
            dfc["OBV"] = ta.obv(dfc["Close"], dfc["Volume"])
            obv_col = "OBV"
        except Exception:
            obv_col = None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.45, 0.45], vertical_spacing=0.08,
                        subplot_titles=("Daily Volume", "On-Balance Volume (OBV)"))
    if vol_col and vol_col in dfc.columns:
        fig.add_trace(go.Bar(x=dfc.index, y=dfc[vol_col], name="Volume"), row=1, col=1)
    else:
        fig.add_trace(go.Bar(x=dfc.index, y=[0]*len(dfc.index), name="Volume (missing)"), row=1, col=1)
    if obv_col and obv_col in dfc.columns:
        fig.add_trace(go.Scatter(x=dfc.index, y=dfc[obv_col], name="OBV", mode="lines"), row=2, col=1)
    else:
        fig.add_trace(go.Scatter(x=dfc.index, y=[None]*len(dfc.index), name="OBV (missing)"), row=2, col=1)
    fig.update_layout(height=600, legend=dict(orientation="h"))
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def plot_candlestick(df: pd.DataFrame, last_n_days: int = None, title: str = None):
    df_plot = df.tail(last_n_days) if (last_n_days is not None and len(df) > last_n_days) else df
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot["Open"] if "Open" in df_plot.columns else None,
        high=df_plot["High"] if "High" in df_plot.columns else None,
        low=df_plot["Low"] if "Low" in df_plot.columns else None,
        close=df_plot["Close"] if "Close" in df_plot.columns else None,
        name="Candlestick",
        increasing_line_color="green",
        decreasing_line_color="red"
    ))
    bbu = next((c for c in df_plot.columns if c.upper().startswith("BBU_")), None)
    bbm = next((c for c in df_plot.columns if c.upper().startswith("BBM_")), None)
    bbl = next((c for c in df_plot.columns if c.upper().startswith("BBL_")), None)
    if bbm:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[bbm], name="BBM"))
    if bbu:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[bbu], name="BBU"))
    if bbl:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[bbl], name="BBL"))
    fig.update_layout(title=title or "Candlestick", xaxis_title="Date", yaxis_title="Price(INR)", height=650)
    fig.update_xaxes(rangeslider_visible=False)
    return fig


class TimeSeriesSeqDataset(Dataset):
    def __init__(self, df_scaled, seq_len=SEQ_LEN, feature_cols=None, target_cols=None):
        self.X = df_scaled.values.astype(np.float32)
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.seq_len = seq_len
        self.indices = []
        for i in range(seq_len - 1, len(self.X) - 1):
            self.indices.append(i)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        i = self.indices[idx]
        seq = self.X[i - (self.seq_len - 1): i + 1]
        target_row = self.X[i + 1]
        target = np.array([target_row[self.feature_cols.index(c)] for c in self.target_cols], dtype=np.float32)
        return torch.from_numpy(seq), torch.from_numpy(target)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  
        self.register_buffer("pe", pe)
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class MultiHeadAttentionWithWeights(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        out, attn_weights = self.mha(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True)
        return out, attn_weights

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttentionWithWeights(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        attn_out, attn_w = self.mha(x, x, x)
        x = self.norm1(x + attn_out)
        f = self.ffn(x)
        x = self.norm2(x + f)
        return x, attn_w

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_mha = MultiHeadAttentionWithWeights(d_model, n_heads, dropout)
        self.cross_mha = MultiHeadAttentionWithWeights(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        self_attn_out, self_attn_w = self.self_mha(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self_attn_out)
        cross_attn_out, cross_attn_w = self.cross_mha(tgt, memory, memory, attn_mask=memory_mask)
        tgt = self.norm2(tgt + cross_attn_out)
        f = self.ffn(tgt)
        tgt = self.norm3(tgt + f)
        return tgt, self_attn_w, cross_attn_w

class InformerLike(nn.Module):
    def __init__(self, n_features, d_model=128, n_heads=4, enc_layers=3, dec_layers=2, ffn_dim=256, dropout=0.1, target_size=2):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=2000)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(enc_layers)])
        self.decoder_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(dec_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, target_size)
        )
    def forward(self, src):
        batch_size = src.size(0)
        x = self.input_proj(src)
        x = self.pos_enc(x)
        attn_record = {"encoder": [], "decoder_self": [], "decoder_cross": []}
        mem = x
        for layer in self.encoder_layers:
            mem, attn_w = layer(mem)
            attn_record["encoder"].append(attn_w.detach().cpu().numpy())
        tgt = self.decoder_query.repeat(batch_size, 1, 1)
        for layer in self.decoder_layers:
            tgt, self_attn_w, cross_attn_w = layer(tgt, mem)
            attn_record["decoder_self"].append(self_attn_w.detach().cpu().numpy())
            attn_record["decoder_cross"].append(cross_attn_w.detach().cpu().numpy())
        out = tgt[:, -1, :]
        preds = self.fc(out)
        return preds, attn_record


# Utility functions
def fit_scaler_on_train(train_df, feature_cols):
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols].values)
    return scaler

def scale_df(a_df, scaler, feature_cols):
    arr = scaler.transform(a_df[feature_cols].values)
    return pd.DataFrame(arr, index=a_df.index, columns=feature_cols)

def inverse_targets_from_scaled_original(scaled_arr, scaler, feature_cols, target_cols):
    scaled_arr = np.array(scaled_arr)
    if scaled_arr.ndim == 1:
        scaled_arr = scaled_arr.reshape(1, -1)
    n = scaled_arr.shape[0]
    filler = np.zeros((n, len(feature_cols)))
    for i, col in enumerate(feature_cols):
        if col in target_cols:
            filler[:, feature_cols.index(col)] = scaled_arr[:, target_cols.index(col)]
    inv = scaler.inverse_transform(filler)
    return inv[:, [feature_cols.index(c) for c in target_cols]]

def iterative_forecast_original(model, last_window_scaled, steps, feature_cols, target_cols, scaler):
    model.eval()
    seq = last_window_scaled.copy().astype(np.float32)
    preds_scaled_all = []
    with torch.no_grad():
        for s in range(steps):
            x = torch.from_numpy(seq[np.newaxis, :, :]).to(DEVICE)
            pred_scaled, _ = model(x)
            pred_scaled_np = pred_scaled.cpu().numpy().reshape(-1)
            preds_scaled_all.append(pred_scaled_np)
            last_row = seq[-1].copy()
            for t_idx, tcol in enumerate(target_cols):
                col_index = feature_cols.index(tcol)
                last_row[col_index] = pred_scaled_np[t_idx]
            seq = np.vstack([seq[1:, :], last_row])
    preds_scaled_all = np.vstack(preds_scaled_all)
    preds_inv = inverse_targets_from_scaled_original(preds_scaled_all, scaler, feature_cols, target_cols)
    return preds_inv


# Evolution Strategy Trading Agent Classes
class DeepEvolutionStrategy:

    def __init__(self, weights, reward_function,
                 population_size=15,
                 sigma=0.1,
                 learning_rate=0.03):

        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):

        weights_population = []

        for w, p in zip(weights, population):
            weights_population.append(w + self.sigma * p)

        return weights_population

    def get_weights(self):
        return self.weights

    def train(self, epoch=120):

        for _ in range(epoch):

            population = []
            rewards = np.zeros(self.population_size)

            for k in range(self.population_size):

                member = []
                for w in self.weights:
                    member.append(np.random.randn(*w.shape))

                population.append(member)

            for k in range(self.population_size):

                weights = self._get_weight_from_population(self.weights, population[k])
                rewards[k] = self.reward_function(weights)

            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)

            for index, w in enumerate(self.weights):

                A = np.array([p[index] for p in population])

                grad = np.tensordot(rewards, A, axes=(0, 0)) / (self.population_size * self.sigma)

                self.weights[index] = w + self.learning_rate * grad


class ESModel:

    def __init__(self, input_size, hidden_size, output_size):

        self.weights = [
            np.random.randn(input_size, hidden_size) * 0.1,
            np.random.randn(hidden_size, output_size) * 0.1,
            np.random.randn(1, hidden_size) * 0.1
        ]

    def predict(self, inputs):

        W1, W2, b1 = self.weights

        h = np.dot(inputs, W1) + b1
        h = np.maximum(h, 0)

        out = np.dot(h, W2)

        return out

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class ESAgent:

    def __init__(self,
                 model,
                 window_size,
                 forecast_series,
                 actual_series,
                 initial_money=10000):

        self.model = model
        self.window_size = window_size

        self.forecast = forecast_series
        self.actual = actual_series

        self.initial_money = initial_money

        self.es = DeepEvolutionStrategy(
            self.model.get_weights(),
            self.get_reward
        )

    def get_state(self, t):

        window = self.window_size

        d = t - window + 1

        if d >= 0:
            block = self.forecast[d:t+1]
        else:
            block = np.concatenate([
                np.full(-d, self.forecast[0]),
                self.forecast[0:t+1]
            ])

        res = [(block[i+1] - block[i]) for i in range(window - 1)]

        return np.array([res])

    def act(self, state):

        decision = self.model.predict(state)

        return np.argmax(decision[0])

    def get_reward(self, weights):

        self.model.set_weights(weights)

        money = self.initial_money
        inventory = []

        state = self.get_state(0)

        for t in range(len(self.forecast)-1):

            action = self.act(state)

            price = self.actual[t]

            if action == 1 and money >= price:
                inventory.append(price)
                money -= price

            elif action == 2 and len(inventory):

                bought = inventory.pop(0)
                money += price

            state = self.get_state(t+1)

        while len(inventory):

            inventory.pop(0)
            money += self.actual[-1]

        return ((money - self.initial_money) / self.initial_money) * 100

    def train(self):

        self.es.train()

    def trade(self):

        self.model.set_weights(self.es.get_weights())

        money = self.initial_money
        inventory = []

        buys = []
        sells = []

        cash_hist = []

        state = self.get_state(0)

        for t in range(len(self.forecast)-1):

            action = self.act(state)

            price = self.actual[t]

            if action == 1 and money >= price:

                inventory.append(price)
                money -= price

                buys.append(t)

            elif action == 2 and len(inventory):

                inventory.pop(0)
                money += price

                sells.append(t)

            cash_hist.append(money)

            state = self.get_state(t+1)

        while len(inventory):

            inventory.pop(0)
            money += self.actual[-1]

        total_gains = money - self.initial_money
        pct_return = (total_gains / self.initial_money) * 100

        return buys, sells, money, total_gains, pct_return, cash_hist


def explain_model_performance_llm(mae_open, mae_close, rmse_open, rmse_close, test_days):

    prompt = f"""
A transformer-based stock forecasting model was evaluated on the last {test_days} days of data.

Evaluation metrics obtained:

MAE (Close Price): {mae_close:.4f} INR
RMSE (Close Price): {rmse_close:.4f} INR

Important clarification:
• These errors represent absolute price differences in Indian Rupees (INR), NOT percentages.
• For example, an MAE of {mae_close:.2f} means the prediction differs from the real price by about ₹{mae_close:.2f} on average.

The forecasting model architecture:

• Uses a 60-day historical input window of market data
• Input projection converts indicators into a 128-dimension representation
• Positional encoding preserves the chronological order of the sequence
• A transformer encoder with 3 layers and 4 attention heads learns relationships across the 60-day window
• A 2-layer transformer decoder interprets the learned market patterns
• Final dense layers produce the predicted stock prices

Explain the results in simple language.

Focus on:
• what MAE and RMSE represent in this context
• how the model performed during the test period
• how the 60-day transformer architecture helps capture market patterns
• mention that small prediction differences are normal because financial markets are inherently volatile
Focus on positives and do not state room for improvements keep optimistic and state how good model was.
Do NOT convert these errors into percentages.
Keep the explanation short (~10 lines) and neutral-professional in tone.
"""

    try:
        response = ollama.chat(
            model="gemma2:2b",
            messages=[
                {
                    "role":"system",
                    "content":"You explain machine learning model performance clearly and professionally."
                },
                {
                    "role":"user",
                    "content":prompt
                }
            ],
            options={"temperature":0.2, "num_predict":300},
            keep_alive="30m"
        )

        return response["message"]["content"]

    except Exception as e:
        return f"LLM explanation failed: {e}"
    

def explain_ig_results_llm(ticker):

    try:
        df = pd.read_csv(f"{ticker}_ig_summary_last10.csv")

        top_features = (
            df.groupby("feature")["score"]
            .mean()
            .sort_values(ascending=False)
            .head(8)
            .to_string()
        )

        prompt = f"""
Integrated Gradients (IG) was applied to explain a transformer-based stock prediction model.

Model context:
- Each prediction uses the **previous 60 days of market data**.
- IG was calculated for the **last 10 prediction days**.
- Each IG heatmap therefore shows how the **previous 60 days of features influenced that specific day's prediction**.

Average IG feature importance across the 10 explanations:

{top_features}

Write a **concise analytical explanation** covering the following:

1. Briefly explain what Integrated Gradients measures in this model (1 sentence only).

2. Explain how to interpret the IG heatmap plots in this experiment:
   - each plot corresponds to one prediction day
   - x-axis represents the previous 60 days used as model input
   - y-axis represents the indicators/features
   - stronger red = stronger negative contribution
   - stronger blue = stronger positive contribution.

3. Analyze the **actual results above**:
   - identify which indicators appear most influential
   - explain why indicators such as Bollinger Bands, price values, or moving averages might matter for stock prediction.

4. Provide a **clear interpretation of what the model has learned** from the data.

5. Write 2–3 specific sentences that directly reference the most important indicators listed above.
Explain:
• what the dominance of these indicators suggests about the model's strategy
• whether the model relies more on price levels, trend indicators (SMA/EMA), or volatility indicators (Bollinger Bands)
• what this indicates about how the model predicts stock movement.

Do not write a generic conclusion. The conclusion must mention at least two of the top indicators explicitly.

Important instructions:
- Do NOT write a generic textbook explanation.
- Focus specifically on the **results provided above**.
- Keep the explanation clear and analytical.
- around 8–12 lines.
"""

        response = ollama.chat(
            model="gemma2:2b",
            messages=[{"role":"user","content":prompt}],
            options={"temperature":0.2,"num_predict":600},
            keep_alive="30m"
        )

        return response["message"]["content"]

    except Exception as e:
        return f"IG explanation failed: {e}"
    

def explain_saliency_results_llm(ticker):

    try:
        df = pd.read_csv(f"{ticker}_saliency_summary_last10.csv")

        top_features = (
            df.groupby("feature")["score"]
            .mean()
            .sort_values(ascending=False)
            .head(8)
            .to_string()
        )

        prompt = f"""
Saliency (gradient × input) analysis was applied to explain a transformer-based stock prediction model.

Model context:
- Each prediction uses the **previous 60 days of market data**.
- Saliency was calculated for the **last 10 prediction days**.
- Each saliency heatmap therefore shows how the **previous 60 days of features influenced that specific day's prediction**.

Average saliency feature importance across the 10 explanations:

{top_features}

Write a **concise analytical explanation** covering the following:

1. Briefly explain what Saliency (gradient × input) measures in this model (1 sentence only).

2. Explain how to interpret the saliency heatmap plots in this experiment:
   - each plot corresponds to one prediction day
   - x-axis represents the previous 60 days used as model input
   - y-axis represents the indicators/features
   - stronger red = stronger negative contribution
   - stronger blue = stronger positive contribution.

3. Analyze the **actual results above**:
   - identify which indicators appear most influential
   - explain why indicators such as Bollinger Bands, price values, or moving averages might matter for stock prediction.

4. Provide a **clear interpretation of what the model has learned** from the data.

5. Write 2–3 specific sentences that directly reference the most important indicators listed above.
Explain:
• what the dominance of these indicators suggests about the model's strategy
• whether the model relies more on price levels, trend indicators (SMA/EMA), or volatility indicators (Bollinger Bands)
• what this indicates about how the model predicts stock movement.

Do not write a generic conclusion. The conclusion must mention at least two of the top indicators explicitly.

Important instructions:
- Do NOT write a generic textbook explanation.
- Focus specifically on the **results provided above**.
- Keep the explanation clear and analytical.
- around 8–12 lines.
"""

        response = ollama.chat(
            model="gemma2:2b",
            messages=[{"role":"user","content":prompt}],
            options={"temperature":0.2,"num_predict":600},
            keep_alive="30m"
        )

        return response["message"]["content"]

    except Exception as e:
        return f"Saliency explanation failed: {e}"
    

def explain_permutation_results_llm(ticker):

    try:
        df = pd.read_csv(f"{ticker}_permutation_importance.csv")

        top_features = df.head(8).to_string()

        prompt = f"""
Permutation feature importance was applied to explain a transformer-based stock prediction model.

Model context:
- The model predicts stock prices using multiple technical indicators.
- Permutation importance measures how prediction accuracy changes when each feature is randomly shuffled.
- A larger increase in RMSE means that feature is more important for the model.

Top features based on permutation importance:

{top_features}

Write a **concise analytical explanation** covering the following:

1. Briefly explain what permutation importance measures in this model (1 sentence only).

2. Explain how to interpret the permutation importance bar plot:
   - each bar represents one feature
   - the x-axis represents the increase in RMSE after shuffling that feature
   - larger RMSE increase means the model depends more on that feature.

3. Analyze the **actual results above**:
   - identify which indicators appear most influential
   - explain why indicators such as Bollinger Bands, price values, or moving averages might matter for stock prediction.

4. Provide a **clear interpretation of what the model has learned** from the data.

5. Write 2–3 specific sentences that directly reference the most important indicators listed above.
Explain:
• what the dominance of these indicators suggests about the model's strategy
• whether the model relies more on price levels, trend indicators (SMA/EMA), or volatility indicators (Bollinger Bands)
• what this indicates about how the model predicts stock movement.

Do not write a generic conclusion. The conclusion must mention at least two of the top indicators explicitly.

Important instructions:
- Do NOT write a generic textbook explanation.
- Focus specifically on the **results provided above**.
- Keep the explanation clear and analytical.
- around 8–12 lines.
"""

        response = ollama.chat(
            model="gemma2:2b",
            messages=[{"role":"user","content":prompt}],
            options={"temperature":0.2,"num_predict":600},
            keep_alive="30m"
        )

        return response["message"]["content"]

    except Exception as e:
        return f"Permutation explanation failed: {e}"
    

def explain_shap_results_llm(ticker):

    try:
        df = pd.read_csv(f"{ticker}_shap_explained_last10.csv")

        top_features = (
            df.groupby("feature")["shap"]
            .apply(lambda x: abs(x).mean())
            .sort_values(ascending=False)
            .head(8)
            .to_string()
        )

        prompt = f"""
SHAP (SHapley Additive Explanations) analysis was applied to explain a transformer-based stock prediction model.

Model context:
- Each prediction uses the **previous 60 days of market data**.
- SHAP values were computed for the **last 10 prediction days**.
- SHAP measures how much each feature contributes positively or negatively to a prediction.

Average SHAP feature importance across the 10 explanations:

{top_features}

Write a **concise analytical explanation** covering the following:

1. Briefly explain what SHAP values measure in this model (1 sentence only).

2. Explain how to interpret the SHAP plots:
   - each plot corresponds to one prediction day
   - features push predictions either higher or lower
   - positive SHAP values increase predictions
   - negative SHAP values decrease predictions.

3. Analyze the **actual results above**:
   - identify which indicators appear most influential
   - explain why indicators such as Bollinger Bands, price values, or moving averages might matter for stock prediction.

4. Provide a **clear interpretation of what the model has learned** from the data.

5. Write 2–3 specific sentences that directly reference the most important indicators listed above.
Explain:
• what the dominance of these indicators suggests about the model's strategy
• whether the model relies more on price levels, trend indicators (SMA/EMA), or volatility indicators (Bollinger Bands)
• what this indicates about how the model predicts stock movement.

Do not write a generic conclusion. The conclusion must mention at least two of the top indicators explicitly.

Important instructions:
- Do NOT write a generic textbook explanation.
- Focus specifically on the **results provided above**.
- Keep the explanation clear and analytical.
- around 8–12 lines.
"""

        response = ollama.chat(
            model="gemma2:2b",
            messages=[{"role":"user","content":prompt}],
            options={"temperature":0.2,"num_predict":600},
            keep_alive="30m"
        )

        return response["message"]["content"]

    except Exception as e:
        return f"SHAP explanation failed: {e}"


def explain_combined_xai_llm(ig_summary, saliency_summary, permutation_summary, shap_summary):

    prompt = f"""
Multiple explainability techniques were used to analyze a transformer-based stock forecasting model.

Below are the summaries from each technique:

Integrated Gradients Summary:
{ig_summary}

Saliency Summary:
{saliency_summary}

Permutation Importance Summary:
{permutation_summary}

SHAP Summary:
{shap_summary}

Write a **combined analytical interpretation** covering the following:

1. Identify which indicators appear consistently important across multiple methods.
2. Explain what this consistency suggests about the model's decision-making process.
3. Discuss whether the model relies more on:
   - price levels (Open, High, Low, Close)
   - trend indicators (SMA, EMA)
   - volatility indicators (Bollinger Bands)
   - momentum indicators (Stochastic, RSI).

4. Explain what this tells us about the strategy the model has learned for predicting stock prices.

5. Provide a clear final conclusion about how the model uses technical indicators when making predictions.

Important instructions:
- Do NOT repeat textbook definitions of the XAI techniques.
- Focus on **comparing the results across methods**.
- The explanation must reference **specific indicators appearing in the summaries above**.
- Avoid vague language.
- Write about **8–12 analytical lines**.
"""

    try:
        response = ollama.chat(
            model="gemma2:2b",
            messages=[{"role":"user","content":prompt}],
            options={"temperature":0.2,"num_predict":700},
            keep_alive="30m"
        )

        return response["message"]["content"]

    except Exception as e:
        return f"Combined XAI explanation failed: {e}"


def explain_future_forecast_llm(pred_prices,pct):

    start_price = float(pred_prices[0])
    end_price = float(pred_prices[-1])
    pct_change = ((end_price - start_price) / start_price) * 100

    if pct_change > 2:
        trend = "increase"
    elif pct_change < -2:
        trend = "decrease"
    else:
        trend = "remain relatively stable"

    prompt = f"""
The stock forecasting model has been retrained on the full historical dataset.

The model then generated a **45 day future price forecast**.

Forecast statistics:

Starting predicted price: {start_price:.2f}
Ending predicted price: {end_price:.2f}
Percentage change over forecast horizon: {pct:.2f}%

Explain the expected price trend.

Focus on:
• whether the price is expected to rise, fall, or stay relatively stable
• what the predicted percentage change indicates and sould user buy or sell stocks presently
• mention that the forecast is based on historical market patterns learned by the model

Keep the explanation concise (~5 sentences).
Do not be overly technical.
"""

    try:
        response = ollama.chat(
            model="gemma2:2b",
            messages=[{"role":"user","content":prompt}],
            options={"temperature":0.2,"num_predict":400},
            keep_alive="30m"
        )

        return response["message"]["content"]

    except Exception as e:
        return f"Forecast explanation failed: {e}"
    


# Training and evaluation 
def train_and_eval_informer_from_df(df: pd.DataFrame, ticker: str):
    """
    Adapts your original training script to take df (with indicators) instead of a CSV path.
    Returns models, scalers and the phase1/phase2 plots and metrics (displayed via Streamlit).
    """
    try:
        df = df.copy()
        df.sort_index(inplace=True)

        
        df_timezone = df.index.tz
        start_date_aware = pd.to_datetime(START_DATE)
        if df_timezone is not None:
            start_date_aware = start_date_aware.tz_localize(df_timezone)

        df = df[df.index >= start_date_aware]

        # ffill/bfill/dropna
        df = df.ffill().bfill()
        df.dropna(inplace=True)


        TARGET_COLS = ["Open", "Close"]
        FEATURE_COLS = df.columns.tolist()
        for t in TARGET_COLS:
            if t not in FEATURE_COLS:
                st.error(f"Target {t} not in columns; aborting.")
                return

        if TEST_DAYS >= len(df):
            st.error("TEST_DAYS too large for dataset.")
            return
        train_val_df = df.iloc[:-TEST_DAYS]
        test_df = df.iloc[-TEST_DAYS:].copy()

        val_frac = 0.1
        val_size = int(len(train_val_df) * val_frac)
        train_df = train_val_df.iloc[:-val_size].copy()
        val_df = train_val_df.iloc[-val_size:].copy()

        scaler = MinMaxScaler()
        scaler.fit(train_df[FEATURE_COLS].values)

        train_scaled = scale_df(train_df, scaler, FEATURE_COLS)
        val_scaled = scale_df(val_df, scaler, FEATURE_COLS)
        test_scaled = scale_df(test_df, scaler, FEATURE_COLS)

        train_ds = TimeSeriesSeqDataset(train_scaled, seq_len=SEQ_LEN, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS)
        val_ds = TimeSeriesSeqDataset(val_scaled, seq_len=SEQ_LEN, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS)
        
        test_ds = TimeSeriesSeqDataset(pd.concat([train_scaled.tail(SEQ_LEN), test_scaled]),
                                       seq_len=SEQ_LEN, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        st.write("Train samples:", len(train_ds), "Val samples:", len(val_ds))

        n_features = len(FEATURE_COLS)
        model = InformerLike(n_features, d_model=D_MODEL, n_heads=N_HEADS,
                             enc_layers=N_ENCODER_LAYERS, dec_layers=N_DECODER_LAYERS,
                             ffn_dim=FFN_DIM, dropout=DROPOUT, target_size=len(TARGET_COLS)).to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        
        best_val = 1e9
        counter = 0
        SAVE_PATH = f"{ticker}_informer_once.pth"
        for epoch in range(1, EPOCHS + 1):
            
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                optimizer.zero_grad()
                preds, _ = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())
            
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(DEVICE)
                    yb = yb.to(DEVICE)
                    preds, _ = model(xb)
                    loss = criterion(preds, yb)
                    val_losses.append(loss.item())
            train_loss = np.mean(train_losses) if train_losses else float('nan')
            val_loss = np.mean(val_losses) if val_losses else float('nan')
            scheduler.step(val_loss)
            #st.write(f"Epoch {epoch:03d} | Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f}")
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "scaler": scaler,
                    "feature_cols": FEATURE_COLS,
                    "target_cols": TARGET_COLS
                }, SAVE_PATH)
       
            else:
                counter += 1
                if counter >= PATIENCE:
                    break

        if not os.path.exists(SAVE_PATH):
            st.error("Training finished but checkpoint missing (unexpected).")
            return
        checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        scaler = checkpoint["scaler"]

        full_scaled = pd.DataFrame(scaler.transform(df[FEATURE_COLS].values), index=df.index, columns=FEATURE_COLS)
        test_inputs = []
        test_targets = []
        test_dates = []
        for i in range(len(df) - TEST_DAYS, len(df)):
            seq_end = i - 1
            seq_start = seq_end - (SEQ_LEN - 1)
            if seq_start < 0:
                continue
            seq = full_scaled.iloc[seq_start: seq_end + 1].values.astype(np.float32)
            target_row = full_scaled.iloc[i].values.astype(np.float32)
            target = np.array([target_row[FEATURE_COLS.index(c)] for c in TARGET_COLS], dtype=np.float32)
            test_inputs.append(seq)
            test_targets.append(target)
            test_dates.append(df.index[i])
        if len(test_inputs) == 0:
            st.error("Not enough test sequences generated.")
            return
        test_inputs = np.stack(test_inputs)
        test_targets = np.stack(test_targets)

        model.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(test_inputs).to(DEVICE)
            preds_scaled, attn_test = model(X_test_t)
            preds_scaled = preds_scaled.cpu().numpy()

        def inverse_targets(scaled_arr):
            if scaled_arr.ndim == 1:
                scaled_arr = scaled_arr.reshape(1, -1)
            n = scaled_arr.shape[0]
            filler = np.zeros((n, len(FEATURE_COLS)))
            for i, col in enumerate(FEATURE_COLS):
                if col in TARGET_COLS:
                    filler[:, FEATURE_COLS.index(col)] = scaled_arr[:, TARGET_COLS.index(col)]
            inv = scaler.inverse_transform(filler)
            return inv[:, [FEATURE_COLS.index(c) for c in TARGET_COLS]]

        preds_inv = inverse_targets(preds_scaled)
        targets_inv = inverse_targets(test_targets)

        rmse_open = math.sqrt(mean_squared_error(targets_inv[:,0], preds_inv[:,0]))
        rmse_close = math.sqrt(mean_squared_error(targets_inv[:,1], preds_inv[:,1]))
        mae_open = mean_absolute_error(targets_inv[:,0], preds_inv[:,0])
        mae_close = mean_absolute_error(targets_inv[:,1], preds_inv[:,1])

        st.write(f"Test MAE  Open: {mae_open:.4f} | Close: {mae_close:.4f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_dates, y=targets_inv[:,1], name="Actual Close"))
        fig.add_trace(go.Scatter(x=test_dates, y=preds_inv[:,1], name="Pred Close", line=dict(dash="dash")))
        fig.update_layout(title=f"{ticker} — Test: Close price Actual vs Predicted | RMSE={rmse_close:.3f} MAE={mae_close:.3f}",
                          xaxis_title="Date", yaxis_title="Close")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### AI Explanation of Model Performance")

        with st.spinner("AI analyzing model performance..."):
            model_explanation = explain_model_performance_llm(
                mae_open,
                mae_close,
                rmse_open,
                rmse_close,
                TEST_DAYS
            )

        st.write(model_explanation)

    
        # XAI POST-EVAL 
        # Uses Captum (IG, Saliency), permutation importance, and SHAP.
        st.markdown("## Model Explainability (XAI) — Test set (last 10 test dates)")

        missing_libs = []
        if not CAPTUM_AVAILABLE:
            missing_libs.append("captum (IntegratedGradients, Saliency)")
        try:
            import shap
            SHAP_AVAILABLE = True
        except Exception:
            SHAP_AVAILABLE = False
            missing_libs.append("shap")

        if missing_libs:
            st.warning("XAI libraries missing: " + ", ".join(missing_libs) +
                       ". Install them to enable full XAI: `pip install captum shap` (on your environment).")

        feature_cols = FEATURE_COLS
        target_cols = TARGET_COLS
        device = DEVICE if "DEVICE" in globals() else torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
        model.to(device)
        model.eval()

        def inverse_targets_local(scaled_arr):
            arr = np.array(scaled_arr)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            n = arr.shape[0]
            filler = np.zeros((n, len(feature_cols)))
            for i, col in enumerate(feature_cols):
                if col in target_cols:
                    filler[:, feature_cols.index(col)] = arr[:, target_cols.index(col)]
            inv = scaler.inverse_transform(filler)
            return inv[:, [feature_cols.index(c) for c in target_cols]]

        
        def predict_unscaled_batch(X_np):
            with torch.no_grad():
                t = torch.from_numpy(X_np.astype(np.float32)).to(device)
                out = model(t)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                preds_scaled = out.cpu().numpy()
            preds_unscaled = inverse_targets_local(preds_scaled)
            return preds_unscaled

        try:
            base_preds = predict_unscaled_batch(test_inputs)
            base_rmse_close = math.sqrt(mean_squared_error(targets_inv[:,1], base_preds[:,1]))
            st.write(f"Base test RMSE (Close): **{base_rmse_close:.4f}**")
        except Exception as e:
            st.warning(f"Could not compute base RMSE: {e}")

        def run_in_train_mode(fn, *a, **k):
            prev_mode = model.training
            try:
                model.train()
                return fn(*a, **k)
            finally:
                if not prev_mode:
                    model.eval()

        # Choose the last 10 samples 
        N_test = len(test_inputs)
        XAI_SAMPLE_COUNT = min(10, N_test)
        sample_indices = list(range(N_test - XAI_SAMPLE_COUNT, N_test))  

        # 1) Integrated Gradients (per-sample) 
        if CAPTUM_AVAILABLE:
            from captum.attr import IntegratedGradients
            st.markdown("### Integrated Gradients for last 10 test dates")
            IG_STEPS = 50

            ig_summary_rows = []
            pbar = st.progress(0)
            for idx_count, sample_idx in enumerate(sample_indices):
                class IGWrapper(nn.Module):
                    def __init__(self, base, target_idx=1):
                        super().__init__()
                        self.base = base
                        self.target_idx = target_idx
                    def forward(self, x):
                        out = self.base(x)
                        if isinstance(out, (tuple, list)):
                            out = out[0]
                        return out[:, self.target_idx]

                wmod = IGWrapper(model, target_idx=1).to(device)
                ig = IntegratedGradients(wmod)

                inp = torch.from_numpy(test_inputs[sample_idx:sample_idx+1]).to(device).float()
                baseline = torch.zeros_like(inp).to(device)

                def _compute_ig():
                    attrs, delta = ig.attribute(inputs=inp, baselines=baseline, n_steps=IG_STEPS, return_convergence_delta=True)
                    return attrs.cpu().detach().numpy()[0], float(delta)

                try:
                    at_np, delta = run_in_train_mode(_compute_ig)
                    feat_scores = np.sum(np.abs(at_np), axis=0)  
                    pct = 100.0 * feat_scores / (feat_scores.sum() + 1e-9)
                    df_feat = pd.DataFrame({"feature": feature_cols, "score": feat_scores, "pct": pct}).sort_values("score", ascending=False).reset_index(drop=True)

                    topn = df_feat.head(12)
                    bar_fig = go.Figure(go.Bar(
                        x=topn["score"].values[::-1],
                        y=topn["feature"].values[::-1],
                        orientation="h",
                        hovertemplate="%{y}: %{x:.6f}<extra></extra>"
                    ))
                    bar_fig.update_layout(title=f"IG — Top features (sample idx {sample_idx}) | date {str(test_dates[sample_idx].date())}",
                                          xaxis_title="IG score (sum abs over time)", yaxis_title="Feature", height=420)
                    st.plotly_chart(bar_fig, use_container_width=True)

                    heat_z = at_np.T  
                    heat_fig = go.Figure(go.Heatmap(
                        z=heat_z,
                        x=[f"t-{SEQ_LEN-1-i}" for i in range(SEQ_LEN)][::-1],
                        y=feature_cols,
                        colorscale="RdBu",
                        zmid=0,
                        colorbar=dict(title="IG")
                    ))
                    heat_fig.update_layout(title=f"IG per-timestep (features × time) sample idx {sample_idx}", height=600,
                                           xaxis_title="Time steps (older → newer)", yaxis_title="Feature")
                    st.plotly_chart(heat_fig, use_container_width=True)

                    
                    for rank, row in enumerate(topn.itertuples(), 1):
                        ig_summary_rows.append({"sample_idx": sample_idx, "date": str(test_dates[sample_idx].date()), "rank": rank, "feature": row.feature, "score": float(row.score), "pct": float(row.pct)})
                except Exception as e:
                    st.warning(f"IG failed for sample index {sample_idx}: {e}")

                pbar.progress(int((idx_count+1)/len(sample_indices)*100))

            if ig_summary_rows:
                pd.DataFrame(ig_summary_rows).to_csv(f"{ticker}_ig_summary_last10.csv", index=False)
                st.info(f"Saved IG summary to `{ticker}_ig_summary_last10.csv`")
                st.markdown("### AI Summary of Integrated Gradients")
                with st.spinner("AI analyzing IG results..."):
                    ig_summary = explain_ig_results_llm(ticker)
                st.write(ig_summary)
        else:
            st.info("Captum not available — skipping Integrated Gradients.")


        # 2) Saliency (grad × input) per-sample 
        if CAPTUM_AVAILABLE:
            from captum.attr import Saliency
            st.markdown("### Saliency (gradient × input) — per-sample for last test dates")
            sal_summary_rows = []
            pbar2 = st.progress(0)
            for idx_count, sample_idx in enumerate(sample_indices):
                class SalWrapper(nn.Module):
                    def __init__(self, base, target_idx=1):
                        super().__init__(); self.base = base; self.target_idx = target_idx
                    def forward(self, x):
                        out = self.base(x)
                        if isinstance(out, (tuple, list)): out = out[0]
                        return out[:, self.target_idx]

                wmod = SalWrapper(model, target_idx=1).to(device)
                sal = Saliency(wmod)

                inp = torch.from_numpy(test_inputs[sample_idx:sample_idx+1]).to(device).float()
                inp.requires_grad = True

                def _compute_sal():
                    attr = sal.attribute(inp)  
                    return attr.cpu().detach().numpy()[0]

                try:
                    sal_np = run_in_train_mode(_compute_sal)
                    grad_x_input = sal_np * test_inputs[sample_idx] 
                    feat_scores = np.sum(np.abs(grad_x_input), axis=0)
                    pct = 100.0 * feat_scores / (feat_scores.sum() + 1e-9)
                    df_sal = pd.DataFrame({"feature": feature_cols, "score": feat_scores, "pct": pct}).sort_values("score", ascending=False).reset_index(drop=True)

                    # bar
                    topn = df_sal.head(12)
                    bar_fig = go.Figure(go.Bar(
                        x=topn["score"].values[::-1],
                        y=topn["feature"].values[::-1],
                        orientation="h",
                        hovertemplate="%{y}: %{x:.6f}<extra></extra>"
                    ))
                    bar_fig.update_layout(title=f"Saliency — Top features (sample idx {sample_idx}) | date {str(test_dates[sample_idx].date())}",
                                          xaxis_title="grad×input score (sum abs over time)", yaxis_title="Feature", height=420)
                    st.plotly_chart(bar_fig, use_container_width=True)

                    # heatmap
                    heat_z = grad_x_input.T
                    heat_fig = go.Figure(go.Heatmap(
                        z=heat_z,
                        x=[f"t-{SEQ_LEN-1-i}" for i in range(SEQ_LEN)][::-1],
                        y=feature_cols,
                        colorscale="RdBu",
                        zmid=0,
                        colorbar=dict(title="grad×input")
                    ))
                    heat_fig.update_layout(title=f"Saliency (grad×input) per-timestep sample idx {sample_idx}", height=600,
                                           xaxis_title="Time steps (older → newer)", yaxis_title="Feature")
                    st.plotly_chart(heat_fig, use_container_width=True)

                    for rank, row in enumerate(topn.itertuples(), 1):
                        sal_summary_rows.append({"sample_idx": sample_idx, "date": str(test_dates[sample_idx].date()), "rank": rank, "feature": row.feature, "score": float(row.score), "pct": float(row.pct)})
                except Exception as e:
                    st.warning(f"Saliency failed for sample index {sample_idx}: {e}")

                pbar2.progress(int((idx_count+1)/len(sample_indices)*100))

            if sal_summary_rows:
                pd.DataFrame(sal_summary_rows).to_csv(f"{ticker}_saliency_summary_last10.csv", index=False)
                st.info(f"Saved saliency summary to `{ticker}_saliency_summary_last10.csv`")
                st.markdown("### AI Summary of Saliency Analysis")
                with st.spinner("AI analyzing saliency results..."):
                    sal_summary = explain_saliency_results_llm(ticker)
                st.write(sal_summary)
        else:
            st.info("Captum not available — skipping Saliency.")

      
        # 3) Permutation Feature Importance 
        st.markdown("### Permutation Feature Importance (global, Close) — evaluated on full test set")
        def permutation_feature_importance(X_np, y_true_unscaled, feature_names, target_idx=1, n_repeats=3):
            preds_base = predict_unscaled_batch(X_np)
            base_rmse = math.sqrt(mean_squared_error(y_true_unscaled, preds_base[:, target_idx]))
            st.write(f"Base RMSE (Close): {base_rmse:.4f}")
            importances = []
            N, T, F = X_np.shape
            p = st.progress(0)
            for f_idx, fname in enumerate(feature_names):
                rmses = []
                for _ in range(n_repeats):
                    Xp = X_np.copy()
                    perm = Xp[:,:,f_idx].flatten()
                    np.random.shuffle(perm)
                    Xp[:,:,f_idx] = perm.reshape(N, T)
                    preds_perm = predict_unscaled_batch(Xp)
                    rmse = math.sqrt(mean_squared_error(y_true_unscaled, preds_perm[:, target_idx]))
                    rmses.append(rmse)
                mean_rmse = np.mean(rmses)
                importances.append((fname, mean_rmse - base_rmse))
                p.progress(int((f_idx+1)/len(feature_names)*100))
            df = pd.DataFrame(importances, columns=["feature","delta_RMSE"]).sort_values("delta_RMSE", ascending=False).reset_index(drop=True)
            return df

        try:
            perm_df = permutation_feature_importance(test_inputs, targets_inv[:,1], feature_cols, target_idx=1, n_repeats=3)
            st.write("Top permutation importances (increase in RMSE when permuted):")
            st.dataframe(perm_df.head(20))
            top_perm = perm_df.head(12)
            perm_fig = go.Figure(go.Bar(
                x=top_perm["delta_RMSE"].values[::-1],
                y=top_perm["feature"].values[::-1],
                orientation="h",
                hovertemplate="%{y}: %{x:.6f}<extra></extra>"
            ))
            perm_fig.update_layout(title="Permutation Importance (Close) — top 12", xaxis_title="ΔRMSE", yaxis_title="Feature", height=480)
            st.plotly_chart(perm_fig, use_container_width=True)
            perm_df.to_csv(f"{ticker}_permutation_importance.csv", index=False)
            st.info(f"Saved permutation importances to `{ticker}_permutation_importance.csv`")
            st.markdown("### AI Summary of Permutation Importance")
            with st.spinner("AI analyzing Permutation results..."):
                per_summary = explain_permutation_results_llm(ticker)
            st.write(per_summary)    
        except Exception as e:
            st.warning(f"Permutation importance failed: {e}")


        # 4) SHAP last 10 days
        if SHAP_AVAILABLE:
            st.markdown("### SHAP on last test dates")
            try:
                X_avg = np.mean(test_inputs, axis=1) 
                SHAP_BACKGROUND = min(80, len(X_avg))
                SHAP_EXPLAIN_N = min(10, len(X_avg))  
                SHAP_NSAMPLES = 200

                bg_idx = np.random.choice(len(X_avg), SHAP_BACKGROUND, replace=False)
                background = X_avg[bg_idx]

                def predict_flat(x_flat):
                   
                    seq = np.repeat(x_flat[:, np.newaxis, :], SEQ_LEN, axis=1).astype(np.float32)
                    preds_unscaled = predict_unscaled_batch(seq)
                    return preds_unscaled[:, 1]  # Close

                explainer = shap.KernelExplainer(predict_flat, background)

                to_explain = X_avg[-SHAP_EXPLAIN_N:]
                shap_values = explainer.shap_values(to_explain, nsamples=SHAP_NSAMPLES)

                arr = np.array(shap_values)
                shap_out = arr
                shap_rows = []
                
                explained_indices = list(range(len(X_avg)-SHAP_EXPLAIN_N, len(X_avg)))
                for j, i in enumerate(explained_indices):
                    sv = shap_out[j]
                    df_sh = pd.DataFrame({"feature": feature_cols, "shap": sv})
                    df_sh["abs"] = np.abs(df_sh["shap"])
                    df_sh = df_sh.sort_values("abs", ascending=False).reset_index(drop=True)
                    st.write(f"SHAP top features for sample idx {i} (date {str(test_dates[i].date())}):")
                    st.dataframe(df_sh.head(12))
                    
                    top_sh = df_sh.head(12)
                    sh_fig = go.Figure(go.Bar(
                        x=top_sh["shap"].values[::-1],
                        y=top_sh["feature"].values[::-1],
                        orientation="h",
                        hovertemplate="%{y}: %{x:.6f}<extra></extra>"
                    ))
                    sh_fig.update_layout(title=f"SHAP contributions (sample idx {i})", xaxis_title="SHAP value", yaxis_title="Feature", height=420)
                    st.plotly_chart(sh_fig, use_container_width=True)
                    
                    for row in df_sh.itertuples():
                        shap_rows.append({"sample_idx": i, "date": str(test_dates[i].date()), "feature": row.feature, "shap": float(row.shap)})
                if shap_rows:
                    pd.DataFrame(shap_rows).to_csv(f"{ticker}_shap_explained_last10.csv", index=False)
                    st.info(f"Saved SHAP results to `{ticker}_shap_explained_last10.csv`")
                    st.markdown("### AI Summary of SHAP Analysis")
                    with st.spinner("AI analyzing SHAP results..."):
                        shap_summary = explain_shap_results_llm(ticker)
                    st.write(shap_summary)
            except Exception as e:
                st.warning(f"SHAP explanation failed: {e}")
        else:
            st.info("SHAP not available — skipping SHAP explanations. Install `shap` to enable.")

        st.success("XAI run complete for last test dates. Check CSV artifacts saved alongside the app for deeper analysis.")
        st.markdown("### Combined AI Interpretation of Model Behaviour")

        with st.spinner("AI comparing results across XAI techniques..."):
            combined_xai = explain_combined_xai_llm(
                ig_summary,
                sal_summary,
                per_summary,
                shap_summary
            )
        st.write(combined_xai)



    # Forecast next 365 days and interactive Plotly
        try:
            st.markdown("## Forecast: Next 45 days ")

            # Build scaled series for entire df 
            full_scaled_all = pd.DataFrame(scaler.transform(df[FEATURE_COLS].values),
                                           index=df.index, columns=FEATURE_COLS)

            if len(full_scaled_all) < SEQ_LEN:
                st.warning("Not enough history to build SEQ_LEN window for forecasting. Skipping future forecast.")
            else:
                last_window_scaled = full_scaled_all.iloc[-SEQ_LEN:].values  # shape (seq_len, n_features)
                future_steps = 45

                preds_future_inv = iterative_forecast_original(model, last_window_scaled,
                                                               future_steps, FEATURE_COLS, TARGET_COLS, scaler)

                last_date = df.index[-1]
                if isinstance(last_date, pd.Timestamp) and last_date.tz is not None:
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                                 periods=future_steps, freq='D', tz=last_date.tz)
                else:
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                                 periods=future_steps, freq='D')

                context_days = min(180, len(df))  
                context_dates = df.index[-context_days:]
                context_close = df["Close"].values[-context_days:]

                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(
                    x=context_dates, y=context_close,
                    name="Actual Close (recent)", mode="lines",
                    hovertemplate="%{x}<br>Actual Close: %{y:.2f}<extra></extra>"
                ))
                fig_future.add_trace(go.Scatter(
                    x=future_dates, y=preds_future_inv[:, TARGET_COLS.index("Close")],
                    name=f"Forecast Close (next {future_steps} days)", mode="lines",
                    hovertemplate="%{x}<br>Forecast Close: %{y:.2f}<extra></extra>"
                ))

                fig_future.update_layout(
                    title=f"{ticker} — Forecast: Close price for next {future_steps} days",
                    xaxis=dict(
                        title="Date",
                        rangeselector=dict(
                            buttons=list([
                                dict(count=30, label="1M", step="day", stepmode="backward"),
                                dict(count=120, label="4M", step="day", stepmode="backward"),
                                dict(count=365, label="1Y", step="day", stepmode="backward"),
                                dict(step="all", label="All")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type="date"
                    ),
                    yaxis=dict(title="Close")
                )

                st.plotly_chart(fig_future, use_container_width=True)

        except Exception as e:
            st.warning(f"Future forecasting/plotting failed: {e}")

        
        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "feature_cols": FEATURE_COLS,
            "target_cols": TARGET_COLS
        }, f"{ticker}_informer_like_reliance_final.pth")
        
        
        return {
            "model": model,
            "scaler": scaler,
            "feature_cols": FEATURE_COLS,
            "target_cols": TARGET_COLS,
            "preds_inv": preds_inv,
            "targets_inv": targets_inv,
            "test_dates": test_dates,
            "attn_test": attn_test
        }

    except Exception as e:
        st.error(f"Training pipeline failed: {e}")
        import traceback
        st.text(traceback.format_exc())
        return


def show_auth():
    st.markdown("<div style='display:flex; justify-content:space-between; align-items:center;'>"
                "<h2 style='margin:0'> Login</h2>"
                "<div style='color:#64748b'>Have an account? Login — New user? Click sign up</div>"
                "</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("Login")
        with st.form("login_form"):
            login_user = st.text_input("Username", key="login_user_in")
            login_pass = st.text_input("Password", type="password", key="login_pass_in")
            submitted = st.form_submit_button("Login")
            if submitted:
                if not login_user or not login_pass:
                    st.error("Enter both username and password.")
                else:
                    ok, row = authenticate_user(login_user.strip(), login_pass)
                    if ok:
                        st.success(f"Welcome back, **{login_user.strip()}**")
                        st.session_state.logged_in = True
                        st.session_state.username = login_user.strip()
                        st.session_state.age = row[3] if row else None
                        st.session_state.screen = "dashboard"
                        safe_rerun()
                    else:
                        st.error("Invalid username or password.")

        st.write("")
        if not st.session_state.show_signup:
            if st.button("New user? Sign up"):
                st.session_state.show_signup = True
                

    with col2:
        if st.session_state.show_signup:
            st.subheader("Create an account")
            with st.form("signup_form"):
                signup_user = st.text_input("Choose username", key="signup_user_in")
                signup_pass = st.text_input("Choose password", type="password", key="signup_pass_in")
                signup_confirm = st.text_input("Confirm password", type="password", key="signup_confirm_in")
                age = st.number_input("Age", min_value=1, max_value=120, value=18, step=1)
                st.markdown("<small style='color:#64748b'> Must be 18+.</small>", unsafe_allow_html=True)
                create_sub = st.form_submit_button("Create account")
                if create_sub:
                    
                    if not signup_user or not signup_pass or not signup_confirm:
                        st.warning("Complete all fields.")
                    elif signup_pass != signup_confirm:
                        st.error("Passwords do not match.")
                    elif age < 18:
                        st.error("You must be 18+ to sign up.")
                    else:
                        ok, msg = create_user(signup_user, signup_pass, age)
                        if ok:
                            st.success("Account created! Please login on the left.")
                            st.session_state.show_signup = False
                            st.session_state.login_user_in = signup_user
                            safe_rerun()
                        else:
                            st.error(msg)
        else:
            st.info("If you don't have an account, click **New user? Sign up** on the left.")

def explain_indicators_llm():

    prompt = """
Explain the following stock technical indicators in very simple language.
Give only 1–2 short sentences for each indicator.

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

    try:
        response = ollama.chat(
            model="gemma2:2b",
            messages=[
                {"role":"system","content":"You are a financial assistant. Explain technical indicators simply. do not add things like sure here is the explanation or how can i help you next. just explain what is said and then stop."},
                {"role":"user","content":prompt}
            ],
            options={"temperature":0.2, "num_predict":700},
            keep_alive="30m"
        )

        return response["message"]["content"]

    except Exception as e:
        return f"LLM explanation failed: {e}"

def show_dashboard():
    st.markdown(
        "<div style='display:flex; justify-content:space-between; align-items:center;'>"
        f"<h2 style='margin:0'>Hello, <span style=\"color:#064e3b\">{st.session_state.username}</span></h2>"
        f"<div style='color:#64748b'>Select a stock to fetch data from yfinance</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    
    tick = st.selectbox("Choose ticker", ["INFY.NS", "RELIANCE.NS", "TCS.NS", "SBIN.NS", "SWSOLAR.NS" ])

    if st.button("Fetch Data (OHLCV + Indicators)"):
        try:
            with st.spinner(f"Fetching {tick} from yfinance..."):
                raw_df = fetch_ticker_history(tick, start_date="2019-01-01")
            raw_csv = f"{tick}_data.csv"
            raw_df.to_csv(raw_csv, index=True)

           
            try:
                info = yf.Ticker(tick).info
                if isinstance(info, dict):
                    st.markdown("## Basic Info")
                    keys = ["shortName", "longName", "sector", "industry", "website"]
                    for k in keys:
                        st.markdown(f"**{k}:** {info.get(k, 'N/A')}")
                    st.markdown("---")
            except Exception:
                pass

            st.success(f"Saved raw data to `{raw_csv}` ({len(raw_df)} rows).")
            st.markdown("### Raw OHLCV Data (sample)")
            st.dataframe(raw_df.head())

            
            for c in ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]:
                if c not in raw_df.columns:
                    raw_df[c] = np.nan
                raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce")

            df_with_ind = add_indicators(raw_df)
            with open("latest_run.json", "w") as f:
                json.dump({"ticker": tick}, f)

            out_csv = f"{tick}_with_indicators.csv"
            df_with_ind.to_csv(out_csv, index=True)
            st.success(f"Indicators computed and saved to `{out_csv}` ({len(df_with_ind)} rows).")
            st.markdown("### Data with Technical Indicators (last 5 rows)")
            st.dataframe(df_with_ind.tail(5))
            # LLM explanation of indicators
            st.markdown("## AI Explanation of Technical Indicators")

            with st.spinner("Generating simple explanations..."):
                explanation = explain_indicators_llm()

            st.write(explanation)

            # interactive charts
            st.markdown("## Interactive Charts")
            fig1 = plot_open_close(df_with_ind, title=f"{tick} — Open & Close")
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown(EXPL_OPEN_CLOSE)
            if "EMA_30" not in df_with_ind.columns and "Close" in df_with_ind.columns:
                df_with_ind["EMA_30"] = ta.ema(df_with_ind["Close"], length=30)
            fig2 = plot_moving_averages(df_with_ind, title=f"{tick} — Moving Averages")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown(EXPL_MA)
            fig3 = plot_bollinger_and_atr(df_with_ind, title=f"{tick} — Bollinger Bands & ATR")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown(EXPL_BB_ATR)
            fig4 = plot_rsi_stoch(df_with_ind, title=f"{tick} — RSI & Stochastic")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown(EXPL_RSI_STOCH)
            fig5 = plot_volume_and_obv(df_with_ind, title=f"{tick} — Volume & OBV")
            st.plotly_chart(fig5, use_container_width=True)
            st.markdown(EXPL_VOL_OBV)
            fig6 = plot_candlestick(df_with_ind, last_n_days=None, title=f"{tick} — Candlestick ")
            st.plotly_chart(fig6, use_container_width=True)
            st.markdown(EXPL_CANDLE)

            
            st.markdown("## Forecasting: ")

            #training
            res = train_and_eval_informer_from_df(df_with_ind, tick)

            if res is not None:
                try:
                    preds_inv = res.get("preds_inv")       
                    targets_inv = res.get("targets_inv")   
                    test_dates = res.get("test_dates")    
                    model = res.get("model")
                    scaler = res.get("scaler")
                    feature_cols = res.get("feature_cols")
                    target_cols = res.get("target_cols")

                    if preds_inv is None or targets_inv is None or test_dates is None:
                        st.warning("AB=CD: Missing preds_inv/targets_inv/test_dates from model run — skipping AB=CD agent.")
                    else:
                        
                        forecast_close = np.asarray(preds_inv[:, 1], dtype=float)   
                        actual_close = np.asarray(targets_inv[:, 1], dtype=float)  
                        N_test = len(forecast_close)

                        try:
                            test_dates_arr = pd.to_datetime(test_dates)
                        except Exception:
                            test_dates_arr = np.arange(N_test)
                        
                        # ES Strategy execution
                        forecast_close = np.asarray(preds_inv[:,1])
                        actual_close = np.asarray(targets_inv[:,1])
                        window_size = 30
                        es_model = ESModel(
                            input_size=window_size-1,
                            hidden_size=64,
                            output_size=3
                        )

                        agent = ESAgent(
                            model=es_model,
                            window_size=window_size,
                            forecast_series=forecast_close,
                            actual_series=actual_close,
                            initial_money=10000
                        )

                        agent.train()

                        buys, sells, final_cash, total_gains, pct_return, cash_hist = agent.trade()

                        INITIAL_MONEY = 10000
                        st.markdown("### ES Strategy Results on Test Set")

                        st.write(f"Trades → buys: {len(buys)} sells: {len(sells)}")

                        st.write(
                            f"Agent final cash: {final_cash:.2f} | "
                            f"Total gains: {total_gains:.2f} | "
                            f"Return: {pct_return:.2f}%"
                        )
                        
                        # Plot cash history
                        # ----------------------------------------------------------
                        # ES Trade Execution Chart (Buy / Sell markers)
                        # ----------------------------------------------------------

                        trade_fig = go.Figure()

                        # Actual price line
                        trade_fig.add_trace(go.Scatter(
                            x=test_dates_arr,
                            y=actual_close,
                            name="Actual Close",
                            mode="lines",
                            line=dict(width=2)
                        ))

                        # Buy markers
                        if len(buys) > 0:
                            trade_fig.add_trace(go.Scatter(
                                x=[test_dates_arr[i] for i in buys],
                                y=[actual_close[i] for i in buys],
                                mode="markers",
                                marker=dict(symbol="triangle-up", size=12, color="green"),
                                name="Buy Signal"
                            ))

                        # Sell markers
                        if len(sells) > 0:
                            trade_fig.add_trace(go.Scatter(
                                x=[test_dates_arr[i] for i in sells],
                                y=[actual_close[i] for i in sells],
                                mode="markers",
                                marker=dict(symbol="triangle-down", size=12, color="red"),
                                name="Sell Signal"
                            ))

                        trade_fig.update_layout(
                            title=f"{tick} — ES Strategy Trades on Test Set",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            legend=dict(orientation="h"),
                            height=540
                        )

                        st.plotly_chart(trade_fig, use_container_width=True)
                        ch_fig = go.Figure()
                        ch_fig.add_trace(go.Scatter(x=test_dates_arr, y=cash_hist, name="Cash over time", mode="lines+markers"))
                        ch_fig.update_layout(title="Cash history while trading", xaxis_title="Date", yaxis_title="Cash", height=380)
                        st.plotly_chart(ch_fig, use_container_width=True)
                        
                        
                        st.markdown("### Next 45-day Forecast & Recommendation ")
                        try:
                            full_scaled_all = pd.DataFrame(scaler.transform(df_with_ind[feature_cols].values),
                                                           index=df_with_ind.index, columns=feature_cols)
                        except Exception as e:
                            st.warning(f"Could not build full_scaled_all for future forecast: {e}")
                            full_scaled_all = None

                        if full_scaled_all is None or len(full_scaled_all) < SEQ_LEN:
                            st.warning("Not enough history to create a seed window for 45-day forecast. Skipping future recommendation.")
                        else:
                            last_window_scaled = full_scaled_all.iloc[-SEQ_LEN:].values 
                            future_steps = 45
                            preds_future_inv = iterative_forecast_original(model, last_window_scaled,
                                                                           future_steps, feature_cols, target_cols, scaler)
                            future_close_prices = preds_future_inv[:,1]
                            last_date = df_with_ind.index[-1]
                            if isinstance(last_date, pd.Timestamp) and last_date.tz is not None:
                                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                                             periods=future_steps, freq='D', tz=last_date.tz)
                            else:
                                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                                             periods=future_steps, freq='D')

                            
                            if "Close" in feature_cols:
                                try:
                                    close_idx = target_cols.index("Close") if ("Close" in target_cols) else 1
                                except Exception:
                                    close_idx = 1
                                future_close = preds_future_inv[:, close_idx]
                            else:
                                future_close = preds_future_inv[:, 1] if preds_future_inv.shape[1] > 1 else preds_future_inv[:, 0]

                            last_actual_close = float(df_with_ind["Close"].iloc[-1])
                            mean_future = float(np.nanmean(future_close))
                            pct_change = ((mean_future - last_actual_close) / last_actual_close) * 100.0 if last_actual_close != 0 else 0.0

                            # decision thresholds 
                            BUY_TH = 2.5   
                            SELL_TH = -2.5 

                            if pct_change >= BUY_TH:
                                rec = "BUY"
                            elif pct_change <= SELL_TH:
                                rec = "SELL"
                            else:
                                rec = "HOLD"

                            st.write(f"Last actual close: {last_actual_close:.2f}")
                            st.write(f"Mean predicted close (next {future_steps} days): {mean_future:.2f} ({pct_change:.2f}% change)")
                            forecast_df = pd.DataFrame({
                                "ticker":[tick],
                                "last_close":[last_actual_close],
                                "mean_future_close":[mean_future],
                                "pct_change":[pct_change]
                            })
                            forecast_df.to_csv(f"{tick}_forecast_summary.csv", index=False)
                            st.markdown(f"**Recommendation: {rec}** — (thresholds: BUY >= +{BUY_TH}%, SELL <= {SELL_TH}%)")

                            
                            fut_fig = go.Figure()
                            context_days = min(180, len(df_with_ind))
                            cont_dates = df_with_ind.index[-context_days:]
                            cont_close = df_with_ind["Close"].values[-context_days:]
                            fut_fig.add_trace(go.Scatter(x=cont_dates, y=cont_close, name="Actual Close (recent)", mode="lines"))
                            fut_fig.add_trace(go.Scatter(x=future_dates, y=future_close, name=f"Forecast Close (next {future_steps} days)", mode="lines"))
                            fut_fig.add_annotation(
                                x=future_dates[int(min(5, len(future_dates)-1))],
                                y=float(future_close[min(5, len(future_close)-1)]),
                                text=f"Rec: {rec}",
                                showarrow=True,
                                arrowhead=2
                            )
                            fut_fig.update_layout(title=f"{tick} — Forecast Close next {future_steps} days (Recommendation: {rec})",
                                                  xaxis_title="Date", yaxis_title="Price", height=540)
                            st.plotly_chart(fut_fig, use_container_width=True)
                            st.markdown("### AI Explanation of Future Forecast")

                            with st.spinner("AI analyzing predicted trend..."):
                                forecast_summary = explain_future_forecast_llm(future_close_prices,pct_change)

                            st.write(forecast_summary)

                            st.markdown("## AI Financial Assistant")

                            st.markdown(
"""
Ask questions about:

• Technical indicators  
• Model accuracy (MAE / RMSE)  
• Explainable AI results (IG, Saliency, SHAP, Permutation)  
• Forecast interpretation  

"""
                            )

                            st.link_button(
                                "Open AI Chatbot",
                                "http://localhost:8502"
                            )

                            

                except Exception as e:
                    st.warning(f"AB=CD agent or future recommendation failed: {e}")
                    import traceback
                    st.text(traceback.format_exc())

        except Exception as e:
            st.error(f"Data fetch or indicator computation failed: {e}")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.age = None
        st.session_state.screen = "auth"
        safe_rerun()


# Page routing
st.set_page_config(page_title="Stock Advisor", page_icon="📈", layout="centered")
if st.session_state.screen == "intro":
    show_intro()
elif st.session_state.screen == "auth":
    show_auth()
elif st.session_state.screen == "dashboard":
    if st.session_state.logged_in:
        show_dashboard()
    else:
        st.warning("Please login first.")
        st.session_state.screen = "auth"
        safe_rerun()
