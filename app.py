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
# CONFIGURAZIONE & CSS (Light Blue)
# ==========================================
st.set_page_config(page_title="RRG & Historical Data Engine", layout="wide", page_icon="📊")

st.markdown("""
<style>
    /* Tema Light Blue Professionale */
    .stApp { background-color: #F0F4FA; color: #0F2A56; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #E4EDF8; border-right: 1px solid #C4D6ED; }
    section[data-testid="stSidebar"] * { color: #0F2A56 !important; }
    
    /* Input testuali */
    .stTextArea textarea { background-color: #FFFFFF; color: #0F2A56 !important; border: 1px solid #AEC4E5; border-radius: 8px; }
    .stTextArea textarea:focus { border-color: #3B82F6; box-shadow: 0 0 0 1px #3B82F6; }
    
    /* Titoli */
    h1, h2, h3 { color: #1A3A72 !important; font-weight: 700; }
    
    /* Tabelle e metriche */
    div[data-testid="stDataFrame"] { border: 1px solid #C4D6ED; border-radius: 10px; background-color: #FFFFFF; }
    .metric-card { background: #FFFFFF; border: 1px solid #C4D6ED; border-radius: 12px; padding: 16px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(15, 42, 86, 0.05); }
    .mc-label { font-size: 11px; color: #6B8CC4; text-transform: uppercase; font-weight: 600; }
    .mc-value { font-size: 24px; font-weight: bold; color: #1A3A72; }
    
    /* Tabs */
    button[data-baseweb="tab"] { color: #3B82F6; font-weight: 600; }
    
    /* Bottoni Sidebar */
    div.stButton > button {
        background: linear-gradient(90deg, #3B82F6 0%, #2563EB 100%);
        color: white !important; border: none; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 600;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2); width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%);
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3); transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# INIZIALIZZAZIONE STATO
# ==========================================
if 'master_df' not in st.session_state:
    st.session_state.master_df = None
if 'data_source_type' not in st.session_state:
    st.session_state.data_source_type = None

# ==========================================
# FUNZIONI MATEMATICHE RRG (Identiche al file Excel)
# ==========================================
def ema_sma_seed(series: pd.Series, period: int) -> pd.Series:
    k = 2.0 / (period + 1)
    arr = series.values.astype(float)
    out = np.full(len(arr), np.nan)
    valid_pos = np.where(~np.isnan(arr))[0]
    if len(valid_pos) < period:
        return pd.Series(out, index=series.index)
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
# FUNZIONI ESTRAZIONE E PARSING DATI
# ==========================================
ALIAS_MAP = {
    "SP500": "^GSPC", "NASDAQ": "^NDX", "DOWJONES": "^DJI", "DAX": "^GDAXI", 
    "EUROSTOXX": "^STOXX50E", "GOLD": "GC=F", "OIL": "CL=F", "BTC": "BTC-USD"
}

def get_data_yahoo(ticker, start_dt, end_dt):
    try:
        df = yf.download(ticker, start=start_dt, end=end_dt, progress=False)
        if not df.empty:
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            series = df[col].squeeze()
            if isinstance(series, pd.Series) and not series.empty: 
                return series.ffill()
    except: return None
    return None

def get_data_morningstar(isin, start_dt, end_dt):
    try:
        fund = mstarpy.Funds(term=isin, country="it")
        history = fund.nav(start_date=start_dt, end_date=end_dt, frequency="daily")
        if history:
            df = pd.DataFrame(history)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            series = df['nav']
            series.index = series.index.normalize().tz_localize(None)
            if not series.empty:
                return series
    except: return None
    return None

def parse_file(uploaded):
    raw = uploaded.read()
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        sample = raw[:4096].decode("utf-8", errors="replace")
        sep = ";" if sample.count(";") > sample.count(",") else ","
        df = pd.read_csv(io.BytesIO(raw), sep=sep, decimal="," if sep==";" else ".")
    else:
        df = pd.read_excel(io.BytesIO(raw))
    
    date_col = next((c for c in df.columns if any(k in str(c).lower() for k in ("date", "data", "time"))), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(how="all")[df.index.notna()]

def resample_prices(df, freq_code):
    if freq_code == "D": return df
    rule = "W-FRI" if freq_code == "W" else "ME"
    df_resampled = df.resample(rule).last().dropna(how="all")
    real_last_dates = pd.Series(df.index, index=df.index).resample(rule).last()
    df_resampled.index = pd.DatetimeIndex([real_last_dates[bl] if bl in real_last_dates.index else bl for bl in df_resampled.index])
    return df_resampled

# ==========================================
# UI SIDEBAR: MOTORE DATI
# ==========================================
st.sidebar.title("⚙️ Motore Dati")
data_mode = st.sidebar.radio("Sorgente Dati", ["Scarica da API (Yahoo/MStar)", "Carica File (CSV/Excel)"])

if data_mode == "Scarica da API (Yahoo/MStar)":
    raw_input = st.sidebar.text_area("Lista Tickers/ISIN (Il primo sarà il Benchmark per l'RRG)", "SP500\nSWDA.MI\nLU1287022708\nGOLD", height=100)
    years = st.sidebar.selectbox("Orizzonte Temporale (Anni)", [1, 3, 5, 10, 20], index=1)
    freq_label = st.sidebar.selectbox("Frequenza Dati", ["Giornaliero", "Settimanale", "Mensile"], index=1)
    freq_code = {"Giornaliero": "D", "Settimanale": "W", "Mensile": "ME"}[freq_label]
    
    if st.sidebar.button("📥 Estrai Dati e Sincronizza"):
        raw_tokens = re.findall(r"[\w\.\-\^\=]+", raw_input.upper())
        tickers_input = [ALIAS_MAP.get(t, t) for t in raw_tokens]
        
        if tickers_input:
            start_dt = datetime.now() - timedelta(days=years*365)
            end_dt = datetime.now()
            all_series = {}
            with st.spinner('Estrazione dati in corso...'):
                for t in tickers_input:
                    # Tenta prima con Yahoo
                    s = get_data_yahoo(t, start_dt, end_dt)
                    
                    # Se fallisce o restituisce vuoto, tenta Morningstar
                    if s is None or s.empty:
                        s = get_data_morningstar(t, start_dt, end_dt)
                        
                    # Se ha trovato dati validi, salva
                    if s is not None and not s.empty:
                        s.name = t
                        all_series[t] = s
                    else:
                        st.sidebar.warning(f"Dati non trovati per {t}")
            
            if all_series:
                df_raw = pd.DataFrame(all_series).ffill().dropna()
                st.session_state.master_df = resample_prices(df_raw, freq_code)
                st.session_state.data_source_type = "API"
                st.sidebar.success("Dati estratti e sincronizzati con RRG!")

elif data_mode == "Carica File (CSV/Excel)":
    uploaded = st.sidebar.file_uploader("Carica File", type=["xlsx", "xls", "csv"])
    freq_label = st.sidebar.selectbox("Resample (Opzionale)", ["Nessuno", "Settimanale", "Mensile"])
    freq_code = {"Nessuno": "D", "Settimanale": "W", "Mensile": "ME"}[freq_label]
    
    if uploaded and st.sidebar.button("📂 Analizza File e Sincronizza"):
        try:
            df_raw = parse_file(uploaded)
            st.session_state.master_df = resample_prices(df_raw, freq_code)
            st.session_state.data_source_type = "FILE"
            st.sidebar.success("File caricato e pronto per l'analisi!")
        except Exception as e:
            st.sidebar.error(f"Errore: {e}")

# ==========================================
# MAIN APP TABS
# ==========================================
st.title("Asset & RRG Intelligence")
tab_data, tab_rrg, tab_cheat = st.tabs(["📉 1. Serie Storiche & Statistiche", "🎯 2. Relative Rotation Graph (RRG)", "📋 3. Cheat Sheet Tickers"])

# --- TAB 1: SERIE STORICHE ---
with tab_data:
    if st.session_state.master_df is not None:
        df_final = st.session_state.master_df
        st.subheader("Serie Storiche Attive")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.line_chart((df_final / df_final.iloc[0]) * 100)
            
        with col2:
            metrics = []
            ann_factor = 252 if df_final.index.to_series().diff().mean().days < 4 else (52 if df_final.index.to_series().diff().mean().days < 10 else 12)
            for col in df_final.columns:
                s = df_final[col]
                ret = ((s.iloc[-1] / s.iloc[0]) - 1) * 100
                vol = s.pct_change().std() * np.sqrt(ann_factor) * 100
                dd = ((s - s.cummax()) / s.cummax()).min() * 100
                metrics.append({"Asset": col, "Rendimento %": round(ret,2), "Volatilità %": round(vol,2), "Max DD %": round(dd,2)})
            st.dataframe(pd.DataFrame(metrics).set_index("Asset"), use_container_width=True)
            
        st.markdown("---")
        st.subheader("Matrice di Correlazione")
        if len(df_final.columns) > 1:
            fig, ax = plt.subplots(figsize=(8, 3))
            # Rimosso il dark_background per allinearsi al tema chiaro
            sns.heatmap(df_final.pct_change().corr(), annot=True, cmap="RdYlBu_r", fmt=".2f", vmin=-1, vmax=1, ax=ax)
            st.pyplot(fig)
    else:
        st.info("👈 Estrai o carica dei dati dalla barra laterale per visualizzare le statistiche.")

# --- TAB 2: RRG ---
with tab_rrg:
    if st.session_state.master_df is not None and len(st.session_state.master_df.columns) >= 2:
        df_rrg = st.session_state.master_df
        all_cols = list(df_rrg.columns)
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            benchmark_col = st.selectbox("Benchmark per RRG", all_cols, index=0)
        with c2:
            sector_cols = st.multiselect("Asset da analizzare", [c for c in all_cols if c != benchmark_col], default=[c for c in all_cols if c != benchmark_col])
        with c3:
            method = st.selectbox("Metodo", ["JdK Originale", "Z-Score Statistico"])
            
        if sector_cols:
            try:
                if "JdK" in method:
                    results = compute_jdk_method(df_rrg, benchmark_col, sector_cols)
                else:
                    results = compute_zscore_method(df_rrg, benchmark_col, sector_cols)
                
                # Plotly RRG Chart (Adattata per il tema chiaro)
                fig = go.Figure()
                for name, v in results.items():
                    ratio = v["rs_ratio"].dropna()
                    mom = v["rs_momentum"].dropna()
                    if ratio.empty or mom.empty: continue
                    
                    # Scia
                    xs, ys = ratio.tail(8).values, mom.tail(8).values
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', line=dict(width=2), marker=dict(size=4), opacity=0.5, showlegend=False))
                    # Punto attuale
                    fig.add_trace(go.Scatter(x=[xs[-1]], y=[ys[-1]], mode='markers+text', name=name, text=[name], textposition="top center", marker=dict(size=14, line=dict(width=2, color="#0F2A56"))))
                
                fig.add_shape(type="line", x0=100, x1=100, y0=fig.layout.yaxis.range[0] if fig.layout.yaxis.range else 90, y1=fig.layout.yaxis.range[1] if fig.layout.yaxis.range else 110, line=dict(color="gray", dash="dot"))
                fig.add_shape(type="line", x0=fig.layout.xaxis.range[0] if fig.layout.xaxis.range else 90, x1=fig.layout.xaxis.range[1] if fig.layout.xaxis.range else 110, y0=100, y1=100, line=dict(color="gray", dash="dot"))
                
                # Sfondo bianco per Plotly
                fig.update_layout(template="plotly_white", height=600, xaxis_title="RS-Ratio (Forza)", yaxis_title="RS-Momentum (Velocità)", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabella Risultati
                st.subheader("Stato Attuale")
                tbl = []
                for name, v in results.items():
                    r, m = v["rs_ratio"].dropna(), v["rs_momentum"].dropna()
                    if r.empty: continue
                    tbl.append({"Asset": name, "RS-Ratio": round(r.iloc[-1],2), "RS-Mom": round(m.iloc[-1],2), "Quadrante": get_quadrant(r.iloc[-1], m.iloc[-1]).upper()})
                st.dataframe(pd.DataFrame(tbl).sort_values("RS-Ratio", ascending=False).set_index("Asset"), use_container_width=True)
                
            except Exception as e:
                st.error(f"Errore nel calcolo RRG: {e}")
    else:
        st.warning("Per l'RRG servono almeno 2 asset caricati (1 Benchmark + 1 Settore). Usa la barra laterale.")

# --- TAB 3: CHEAT SHEET ---
with tab_cheat:
    st.markdown("Copia questi Ticker nella barra laterale per l'estrazione dati (API).")
    c1, c2 = st.columns(2)
    c1.code("^GSPC", language="text"); c1.caption("S&P 500")
    c1.code("^NDX", language="text"); c1.caption("NASDAQ 100")
    c1.code("SWDA.MI", language="text"); c1.caption("ETF MSCI World (Milano)")
    c2.code("GC=F", language="text"); c2.caption("Oro (Gold)")
    c2.code("BTC-USD", language="text"); c2.caption("Bitcoin")
