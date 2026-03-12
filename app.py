import streamlit as st
import yfinance as yf
import mstarpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import re
from datetime import datetime, timedelta
import io
import math

# ==========================================
# CONFIGURAZIONE & CSS
# ==========================================
st.set_page_config(page_title="RRG & Historical Data Engine", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .stApp { background-color: #F0F4FA; color: #0F2A56; }
    section[data-testid="stSidebar"] { background-color: #E4EDF8; border-right: 1px solid #C4D6ED; }
    section[data-testid="stSidebar"] * { color: #0F2A56 !important; }
    .stTextArea textarea { background-color: #FFFFFF; color: #0F2A56 !important; border: 1px solid #AEC4E5; border-radius: 8px; }
    .stTextArea textarea:focus { border-color: #3B82F6; box-shadow: 0 0 0 1px #3B82F6; }
    h1, h2, h3 { color: #1A3A72 !important; font-weight: 700; }
    div[data-testid="stDataFrame"] { border: 1px solid #C4D6ED; border-radius: 10px; background-color: #FFFFFF; }
    .metric-card { background: #FFFFFF; border: 1px solid #C4D6ED; border-radius: 12px; padding: 16px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(15, 42, 86, 0.05); }
    button[data-baseweb="tab"] { color: #3B82F6; font-weight: 600; }
    div.stButton > button { background: linear-gradient(90deg, #3B82F6 0%, #2563EB 100%); color: white !important; border: none; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 600; width: 100%; }
    div.stButton > button:hover { transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

if 'master_df' not in st.session_state: st.session_state.master_df = None

# ==========================================
# FUNZIONI MATEMATICHE RRG
# ==========================================
def ema_sma_seed(series: pd.Series, period: int) -> pd.Series:
    k = 2.0 / (period + 1)
    arr = series.values.astype(float)
    out = np.full(len(arr), np.nan)
    valid_pos = np.where(~np.isnan(arr))[0]
    if len(valid_pos) < period: return pd.Series(out, index=series.index)
    seed_pos = valid_pos[period - 1]
    out[seed_pos] = np.nanmean(arr[valid_pos[:period]])
    for i in range(seed_pos + 1, len(arr)):
        if np.isnan(arr[i]): out[i] = np.nan
        else: out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return pd.Series(out, index=series.index)

def compute_jdk_method(df, benchmark_col, sector_cols, ema_short=12, ema_long=26, ratio_window=52, momentum_window=14):
    results = {}
    for col in sector_cols:
        rs_raw = (df[col] / df[benchmark_col]).replace([np.inf, -np.inf], np.nan)
        ema12 = ema_sma_seed(rs_raw, ema_short)
        rs_s = ema_sma_seed(ema12, ema_long)
        rs_s_arr = rs_s.values.astype(float)
        ratio_arr = np.full(len(rs_s_arr), np.nan)
        valid_rs_pos = np.where(~np.isnan(rs_s_arr))[0]
        if len(valid_rs_pos) >= ratio_window:
            anchor_pos = valid_rs_pos[0]
            for i in range(anchor_pos, len(rs_s_arr)):
                if np.isnan(rs_s_arr[i]): continue
                window = rs_s_arr[anchor_pos: i + 1]
                window_valid = window[~np.isnan(window)]
                if len(window_valid) < ratio_window: continue
                mean_val = np.mean(window_valid)
                if mean_val != 0: ratio_arr[i] = 100.0 * rs_s_arr[i] / mean_val
        rs_ratio = pd.Series(ratio_arr, index=df.index)
        ratio_vals = rs_ratio.values.astype(float)
        mom_arr = np.full(len(ratio_vals), np.nan)
        for i in range(len(ratio_vals)):
            if np.isnan(ratio_vals[i]): continue
            start = i - momentum_window + 1
            if start < 0: continue
            window = ratio_vals[start: i + 1]
            window_valid = window[~np.isnan(window)]
            if len(window_valid) < momentum_window: continue
            mean_val = np.mean(window_valid)
            if mean_val != 0: mom_arr[i] = 100.0 * ratio_vals[i] / mean_val
        rs_momentum = pd.Series(mom_arr, index=df.index)
        results[col] = {"rs_raw": rs_raw, "ema12": ema12, "rs_s": rs_s, "rs_ratio": rs_ratio, "rs_momentum": rs_momentum}
    return results

def compute_zscore_method(df, benchmark_col, sector_cols, ema_short=12, ema_long=26, zscore_window=52, momentum_window=10):
    results = {}
    for col in sector_cols:
        rs_raw = (df[col] / df[benchmark_col]).replace([np.inf, -np.inf], np.nan)
        ema12  = ema_sma_seed(rs_raw, ema_short)
        rs_s   = ema_sma_seed(ema12, ema_long)
        roll_mean = rs_s.rolling(zscore_window, min_periods=zscore_window).mean()
        roll_std  = rs_s.rolling(zscore_window, min_periods=zscore_window).std(ddof=0)
        rs_ratio  = 100.0 + 10.0 * (rs_s - roll_mean) / roll_std.replace(0, np.nan)
        d_ratio = rs_ratio.diff()
        m_mean  = d_ratio.rolling(momentum_window, min_periods=momentum_window).mean()
        m_std   = d_ratio.rolling(momentum_window, min_periods=momentum_window).std(ddof=0)
        rs_mom  = 100.0 + 10.0 * (d_ratio - m_mean) / m_std.replace(0, np.nan)
        results[col] = {"rs_raw": rs_raw, "ema12": ema12, "rs_s": rs_s, "rs_ratio": rs_ratio, "rs_momentum": rs_mom}
    return results

def get_quadrant(x, y):
    if x >= 100 and y >= 100: return "leading"
    if x >= 100 and y <  100: return "weakening"
    if x <  100 and y <  100: return "lagging"
    return "improving"

# ==========================================
# ESTRAZIONE DATI
# ==========================================
ALIAS_MAP = {"SP500": "^GSPC", "NASDAQ": "^NDX", "GOLD": "GC=F"}

def get_data_yahoo(ticker, start_dt, end_dt):
    try:
        df = yf.download(ticker, start=start_dt, end=end_dt, progress=False)
        if not df.empty:
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            s = df[col].squeeze()
            if isinstance(s, pd.Series) and not s.empty: return s.ffill()
    except: return None
    return None

def resample_prices(df, freq_code):
    if freq_code == "D": return df
    rule = "W-FRI" if freq_code == "W" else "ME"
    df_resampled = df.resample(rule).last().dropna(how="all")
    real_last_dates = pd.Series(df.index, index=df.index).resample(rule).last()
    df_resampled.index = pd.DatetimeIndex([real_last_dates[bl] if bl in real_last_dates.index else bl for bl in df_resampled.index])
    return df_resampled

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("⚙️ Motore Dati")
raw_input = st.sidebar.text_area("Lista Tickers (Primo = Benchmark)", "SP500\n^NDX\nGC=F", height=100)
years = st.sidebar.selectbox("Anni", [1, 3, 5, 10], index=1)
freq_code = {"Giornaliero": "D", "Settimanale": "W", "Mensile": "ME"}[st.sidebar.selectbox("Frequenza", ["Giornaliero", "Settimanale", "Mensile"], index=1)]

if st.sidebar.button("📥 Estrai Dati"):
    tickers_input = [ALIAS_MAP.get(t, t) for t in re.findall(r"[\w\.\-\^\=]+", raw_input.upper())]
    if tickers_input:
        all_series = {}
        with st.spinner('Estrazione...'):
            for t in tickers_input:
                s = get_data_yahoo(t, datetime.now() - timedelta(days=years*365), datetime.now())
                if s is not None and not s.empty:
                    s.name = t
                    all_series[t] = s
        if all_series:
            st.session_state.master_df = resample_prices(pd.DataFrame(all_series).ffill().dropna(), freq_code)

# ==========================================
# APP MAIN
# ==========================================
st.title("Asset & Macro Intelligence")
tab_rrg, tab_data = st.tabs(["🎯 1. RRG & Macro Map", "📉 2. Serie Storiche"])

with tab_rrg:
    if st.session_state.master_df is not None and len(st.session_state.master_df.columns) >= 2:
        df_rrg = st.session_state.master_df
        cols = list(df_rrg.columns)
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1: benchmark_col = st.selectbox("Benchmark", cols, index=0)
        with c2: sector_cols = st.multiselect("Asset", [c for c in cols if c != benchmark_col], default=[c for c in cols if c != benchmark_col])
        with c3: method = st.selectbox("Metodo", ["JdK Originale", "Z-Score Statistico"])
        
        if sector_cols:
            results = compute_jdk_method(df_rrg, benchmark_col, sector_cols) if "JdK" in method else compute_zscore_method(df_rrg, benchmark_col, sector_cols)
            
            # --- RRG CHART ---
            fig_rrg = go.Figure()
            max_dev = 0
            for name, v in results.items():
                r, m = v["rs_ratio"].dropna(), v["rs_momentum"].dropna()
                if r.empty: continue
                xs, ys = r.tail(8).values, m.tail(8).values
                max_dev = max(max_dev, max([abs(x-100) for x in xs] + [abs(y-100) for y in ys]))
                fig_rrg.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', line=dict(width=2), opacity=0.4, showlegend=False))
                fig_rrg.add_trace(go.Scatter(x=[xs[-1]], y=[ys[-1]], mode='markers+text', name=name, text=[f"<b>{name}</b>"], textposition="top center", marker=dict(size=12)))
            
            fig_rrg.add_vline(x=100, line_color="#1A3A72"); fig_rrg.add_hline(y=100, line_color="#1A3A72")
            pad = max(2, max_dev * 1.1)
            fig_rrg.update_layout(template="plotly_white", height=500, title="Relative Rotation Graph", xaxis=dict(range=[100-pad, 100+pad]), yaxis=dict(range=[100-pad, 100+pad]), showlegend=False)
            st.plotly_chart(fig_rrg, use_container_width=True)
            
            # --- MACRO MAP (SINUSOIDE) ---
            st.markdown("---")
            st.subheader("Macro Map (Economic Cycle)")
            
            fig_macro = go.Figure()
            
            # Generazione sinusoide (0 a 4*pi per rappresentare i 4 stati)
            x_curve = np.linspace(0, 4, 200)
            y_curve = np.sin(x_curve * np.pi / 2 - np.pi/2) # Parte dal basso (-1 a 0), sale a 1 (2), scende a 0 (3), scende a -1 (4)
            
            # Aggiunta aree colorate in background
            fig_macro.add_vrect(x0=0, x1=1, fillcolor="rgba(34, 197, 94, 0.1)", line_width=0, annotation_text="Expansion", annotation_position="bottom right")
            fig_macro.add_vrect(x0=1, x1=2, fillcolor="rgba(239, 68, 68, 0.1)", line_width=0, annotation_text="Contraction", annotation_position="bottom right")
            fig_macro.add_vrect(x0=2, x1=3, fillcolor="rgba(239, 68, 68, 0.1)", line_width=0, annotation_text="Recession", annotation_position="bottom right")
            fig_macro.add_vrect(x0=3, x1=4, fillcolor="rgba(34, 197, 94, 0.1)", line_width=0, annotation_text="Recovery", annotation_position="bottom right")

            fig_macro.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', line=dict(color="#1A3A72", width=2), showlegend=False))
            
            # Posizionamento asset sulla sinusoide tramite calcolo angolo RRG
            for name, v in results.items():
                r, m = v["rs_ratio"].dropna(), v["rs_momentum"].dropna()
                if r.empty: continue
                
                rx, ry = r.iloc[-1] - 100, m.iloc[-1] - 100
                angle = math.atan2(ry, rx) # da -pi a pi
                if angle < 0: angle += 2 * math.pi # da 0 a 2pi
                
                # Mappatura angoli RRG sulla sinusoide
                # RRG Q1 (Leading, 0 a pi/2) -> Expansion (0 a 1)
                # RRG Q4 (Weakening, 3pi/2 a 2pi) -> Contraction (1 a 2)
                # RRG Q3 (Lagging, pi a 3pi/2) -> Recession (2 a 3)
                # RRG Q2 (Improving, pi/2 a pi) -> Recovery (3 a 4)
                
                if rx >= 0 and ry >= 0: # Leading
                    macro_x = 0 + (angle / (math.pi/2))
                elif rx >= 0 and ry < 0: # Weakening
                    macro_x = 1 + ((2*math.pi - angle) / (math.pi/2))
                elif rx < 0 and ry < 0: # Lagging
                    macro_x = 2 + ((angle - math.pi) / (math.pi/2))
                else: # Improving
                    macro_x = 3 + ((math.pi - angle) / (math.pi/2))
                
                macro_y = np.sin(macro_x * np.pi / 2 - np.pi/2)
                
                fig_macro.add_trace(go.Scatter(x=[macro_x], y=[macro_y], mode='markers+text', name=name, text=[f"<b>{name}</b>"], textposition="top center", marker=dict(size=14, line=dict(width=2, color="white"))))

            fig_macro.update_layout(template="plotly_white", height=400, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), margin=dict(l=20, r=20, t=20, b=40), showlegend=False)
            st.plotly_chart(fig_macro, use_container_width=True)

with tab_data:
    if st.session_state.master_df is not None:
        st.line_chart((st.session_state.master_df / st.session_state.master_df.iloc[0]) * 100)
