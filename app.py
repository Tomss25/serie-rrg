import streamlit as st
import signal

# --- PATCH PER STREAMLIT CLOUD E MSTARPY ---
# Ignora l'errore del signal handler fuori dal main thread
original_signal = signal.signal
def patched_signal(signum, handler):
    try:
        return original_signal(signum, handler)
    except ValueError:
        pass
signal.signal = patched_signal
# -------------------------------------------

import yfinance as yf
import mstarpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import re
import math
from datetime import datetime, timedelta
import io

# ==========================================
# CONFIGURAZIONE & CSS (Light Blue)
# ==========================================
st.set_page_config(page_title="RRG & Macro Intelligence", layout="wide", page_icon="📊")

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
    .mc-label { font-size: 11px; color: #6B8CC4; text-transform: uppercase; font-weight: 600; }
    .mc-value { font-size: 24px; font-weight: bold; color: #1A3A72; }
    button[data-baseweb="tab"] { color: #3B82F6; font-weight: 600; }
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
# FUNZIONI ESTRAZIONE E PARSING DATI
# ==========================================
ALIAS_MAP = {"SP500": "^GSPC", "NASDAQ": "^NDX", "DOWJONES": "^DJI", "DAX": "^GDAXI", "EUROSTOXX": "^STOXX50E", "GOLD": "GC=F", "OIL": "CL=F", "BTC": "BTC-USD"}

def get_data_yahoo(ticker, start_dt, end_dt):
    try:
        df = yf.download(ticker, start=start_dt, end=end_dt, progress=False)
        if not df.empty:
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            series = df[col].squeeze()
            if isinstance(series, pd.Series) and not series.empty: return series.ffill()
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
            if not series.empty: return series
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
        if df[col].dtype == object: df[col] = df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
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
    raw_input = st.sidebar.text_area("Lista Tickers/ISIN (Primo = Benchmark)", "SP500\nSWDA.MI\nLU1287022708\nGOLD", height=100)
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
                    s = get_data_yahoo(t, start_dt, end_dt)
                    if s is None or s.empty: s = get_data_morningstar(t, start_dt, end_dt)
                    if s is not None and not s.empty:
                        s.name = t
                        all_series[t] = s
                    else: st.sidebar.warning(f"Dati non trovati per {t}")
            if all_series:
                df_raw = pd.DataFrame(all_series).ffill().dropna()
                st.session_state.master_df = resample_prices(df_raw, freq_code)
                st.session_state.data_source_type = "API"
                st.sidebar.success("Dati estratti e sincronizzati!")

elif data_mode == "Carica File (CSV/Excel)":
    uploaded = st.sidebar.file_uploader("Carica File", type=["xlsx", "xls", "csv"])
    freq_label = st.sidebar.selectbox("Resample (Opzionale)", ["Nessuno", "Settimanale", "Mensile"])
    freq_code = {"Nessuno": "D", "Settimanale": "W", "Mensile": "ME"}[freq_label]
    if uploaded and st.sidebar.button("📂 Analizza File e Sincronizza"):
        try:
            df_raw = parse_file(uploaded)
            st.session_state.master_df = resample_prices(df_raw, freq_code)
            st.session_state.data_source_type = "FILE"
            st.sidebar.success("File caricato!")
        except Exception as e: st.sidebar.error(f"Errore: {e}")

# ==========================================
# MAIN APP TABS
# ==========================================
st.title("Asset & RRG Intelligence")
tab_rrg, tab_data, tab_cheat = st.tabs(["🎯 1. Relative Rotation Graph (RRG)", "📉 2. Serie Storiche & Statistiche", "📋 3. Cheat Sheet Tickers"])

SECTOR_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

# --- TAB 1: RRG & MACRO MAP ---
with tab_rrg:
    if st.session_state.master_df is not None and len(st.session_state.master_df.columns) >= 2:
        df_rrg = st.session_state.master_df
        all_cols = list(df_rrg.columns)
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1: benchmark_col = st.selectbox("Benchmark", all_cols, index=0)
        with c2: sector_cols = st.multiselect("Asset", [c for c in all_cols if c != benchmark_col], default=[c for c in all_cols if c != benchmark_col])
        with c3: method = st.selectbox("Metodo", ["JdK Originale", "Z-Score Statistico"])
        
        st.markdown("**Impostazioni Visive RRG**")
        opt1, opt2, opt3 = st.columns(3)
        show_trails = opt1.toggle("Mostra Code storiche", value=True)
        show_labels = opt2.toggle("Mostra Etichette Nomi", value=True)
        trail_length = opt3.slider("Lunghezza Coda (barre)", min_value=2, max_value=24, value=8)

        if sector_cols:
            try:
                if "JdK" in method: results = compute_jdk_method(df_rrg, benchmark_col, sector_cols)
                else: results = compute_zscore_method(df_rrg, benchmark_col, sector_cols)
                
                # --- PLOT RRG ---
                fig = go.Figure()
                max_dev_x = 0
                max_dev_y = 0

                color_idx = 0
                for name, v in results.items():
                    ratio = v["rs_ratio"].dropna()
                    mom = v["rs_momentum"].dropna()
                    if ratio.empty or mom.empty: continue
                    
                    color = SECTOR_COLORS[color_idx % len(SECTOR_COLORS)]
                    color_idx += 1
                    
                    xs, ys = ratio.tail(trail_length).values, mom.tail(trail_length).values
                    max_dev_x = max(max_dev_x, max(abs(x - 100) for x in xs))
                    max_dev_y = max(max_dev_y, max(abs(y - 100) for y in ys))

                    if show_trails:
                        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', line=dict(width=2.5, color=color), marker=dict(size=5, color=color), opacity=0.4, showlegend=False, legendgroup=name, name=name))
                    
                    mode = 'markers+text' if show_labels else 'markers'
                    text_label = [f"<b>{name}</b>"] if show_labels else None
                    fig.add_trace(go.Scatter(x=[xs[-1]], y=[ys[-1]], mode=mode, name=name, text=text_label, textposition="top center", 
                                             textfont=dict(color="#0F2A56", size=11), marker=dict(size=14, color=color, line=dict(width=2, color="#FFFFFF")),
                                             showlegend=not show_labels, legendgroup=name))
                
                fig.add_vline(x=100, line_width=2, line_color="#1A3A72", opacity=0.8)
                fig.add_hline(y=100, line_width=2, line_color="#1A3A72", opacity=0.8)
                
                pad_x = max(2, max_dev_x * 1.1)
                pad_y = max(2, max_dev_y * 1.1)

                fig.add_shape(type="rect", x0=100, y0=100, x1=100 + pad_x, y1=100 + pad_y, fillcolor="rgba(5, 150, 105, 0.05)", line_width=0, layer="below")
                fig.add_shape(type="rect", x0=100 - pad_x, y0=100, x1=100, y1=100 + pad_y, fillcolor="rgba(59, 130, 246, 0.05)", line_width=0, layer="below")
                fig.add_shape(type="rect", x0=100 - pad_x, y0=100 - pad_y, x1=100, y1=100, fillcolor="rgba(220, 38, 38, 0.05)", line_width=0, layer="below")
                fig.add_shape(type="rect", x0=100, y0=100 - pad_y, x1=100 + pad_x, y1=100, fillcolor="rgba(217, 119, 6, 0.05)", line_width=0, layer="below")
                
                fig.update_layout(
                    template="plotly_white", height=650, 
                    xaxis=dict(title="<b>RS-Ratio (Forza Relativa) ➔</b>", zeroline=False, gridcolor="#E4EDF8", range=[100 - pad_x, 100 + pad_x]), 
                    yaxis=dict(title="<b>RS-Momentum (Velocità) ➔</b>", zeroline=False, gridcolor="#E4EDF8", range=[100 - pad_y, 100 + pad_y]),
                    margin=dict(l=50, r=50, t=30, b=50), plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
                    hoverlabel=dict(bgcolor="#0F2A56", font_size=12, font_color="#FFFFFF"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if not show_labels else None
                )
                
                fig.add_annotation(x=1.0, y=1.0, xref="paper", yref="paper", text="LEADING", showarrow=False, font=dict(size=14, color="#059669", weight="bold"), opacity=0.3)
                fig.add_annotation(x=0.0, y=1.0, xref="paper", yref="paper", text="IMPROVING", showarrow=False, font=dict(size=14, color="#3B82F6", weight="bold"), opacity=0.3)
                fig.add_annotation(x=0.0, y=0.0, xref="paper", yref="paper", text="LAGGING", showarrow=False, font=dict(size=14, color="#DC2626", weight="bold"), opacity=0.3)
                fig.add_annotation(x=1.0, y=0.0, xref="paper", yref="paper", text="WEAKENING", showarrow=False, font=dict(size=14, color="#D97706", weight="bold"), opacity=0.3)

                st.plotly_chart(fig, use_container_width=True)

                # --- PLOT MACRO MAP ---
                st.markdown("---")
                st.subheader("Macro Map (Economic Cycle Projection)")
                
                fig_macro = go.Figure()
                
                x_curve = np.linspace(0, 4, 200)
                y_curve = np.sin(x_curve * np.pi / 2 - np.pi/2)
                
                fig_macro.add_vrect(x0=0, x1=1, fillcolor="rgba(34, 197, 94, 0.1)", line_width=0, annotation_text="Expansion", annotation_position="bottom right", annotation_font_color="#059669")
                fig_macro.add_vrect(x0=1, x1=2, fillcolor="rgba(239, 68, 68, 0.1)", line_width=0, annotation_text="Contraction", annotation_position="bottom right", annotation_font_color="#DC2626")
                fig_macro.add_vrect(x0=2, x1=3, fillcolor="rgba(239, 68, 68, 0.1)", line_width=0, annotation_text="Recession", annotation_position="bottom right", annotation_font_color="#DC2626")
                fig_macro.add_vrect(x0=3, x1=4, fillcolor="rgba(34, 197, 94, 0.1)", line_width=0, annotation_text="Recovery", annotation_position="bottom right", annotation_font_color="#059669")

                fig_macro.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', line=dict(color="#1A3A72", width=2), showlegend=False))
                
                color_idx = 0
                for name, v in results.items():
                    r, m = v["rs_ratio"].dropna(), v["rs_momentum"].dropna()
                    if r.empty: continue
                    
                    color = SECTOR_COLORS[color_idx % len(SECTOR_COLORS)]
                    color_idx += 1
                    
                    rx, ry = r.iloc[-1] - 100, m.iloc[-1] - 100
                    angle = math.atan2(ry, rx)
                    if angle < 0: angle += 2 * math.pi
                    
                    if rx >= 0 and ry >= 0: macro_x = 0 + (angle / (math.pi/2))
                    elif rx >= 0 and ry < 0: macro_x = 1 + ((2*math.pi - angle) / (math.pi/2))
                    elif rx < 0 and ry < 0: macro_x = 2 + ((angle - math.pi) / (math.pi/2))
                    else: macro_x = 3 + ((math.pi - angle) / (math.pi/2))
                    
                    macro_y = np.sin(macro_x * np.pi / 2 - np.pi/2)
                    
                    mode_macro = 'markers+text' if show_labels else 'markers'
                    text_label_macro = [f"<b>{name}</b>"] if show_labels else None
                    
                    fig_macro.add_trace(go.Scatter(x=[macro_x], y=[macro_y], mode=mode_macro, name=name, text=text_label_macro, textposition="top center", textfont=dict(color="#0F2A56", size=11), marker=dict(size=14, color=color, line=dict(width=2, color="#FFFFFF")), showlegend=not show_labels, legendgroup=name))

                fig_macro.update_layout(
                    template="plotly_white", height=400, 
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), 
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), 
                    margin=dict(l=20, r=20, t=20, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if not show_labels else None
                )
                st.plotly_chart(fig_macro, use_container_width=True)

                # --- TABELLA RISULTATI (STYLE AGGIORNATO) ---
                st.subheader("Stato Attuale Asset")
                tbl = []
                for name, v in results.items():
                    r, m = v["rs_ratio"].dropna(), v["rs_momentum"].dropna()
                    if r.empty: continue
                    tbl.append({"Asset": name, "RS-Ratio": round(r.iloc[-1],2), "RS-Mom": round(m.iloc[-1],2), "Quadrante": get_quadrant(r.iloc[-1], m.iloc[-1]).upper()})
                
                df_tbl = pd.DataFrame(tbl).sort_values("RS-Ratio", ascending=False).set_index("Asset")
                
                def color_quadrants(row):
                    q = row['Quadrante']
                    if q == 'LEADING': color = 'rgba(5, 150, 105, 0.15)'
                    elif q == 'IMPROVING': color = 'rgba(59, 130, 246, 0.15)'
                    elif q == 'LAGGING': color = 'rgba(220, 38, 38, 0.15)'
                    elif q == 'WEAKENING': color = 'rgba(217, 119, 6, 0.15)'
                    else: color = ''
                    return [f'background-color: {color}' if color else ''] * len(row)
                
                st.dataframe(df_tbl.style.apply(color_quadrants, axis=1), use_container_width=True)

            except Exception as e: st.error(f"Errore: {e}")
    else: st.warning("Servono almeno 2 asset caricati dalla barra laterale.")

# --- TAB 2: SERIE STORICHE ---
with tab_data:
    if st.session_state.master_df is not None:
        df_final = st.session_state.master_df
        col1, col2 = st.columns([2, 1])
        with col1: 
            st.markdown("**Andamento Normalizzato (Base 100)**")
            st.line_chart((df_final / df_final.iloc[0]) * 100)
        with col2:
            st.markdown("**Metriche di Rischio/Rendimento**")
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
        st.subheader("Dati Storici Grezzi")
        st.dataframe(df_final.sort_index(ascending=False).round(4), use_container_width=True)
    else: st.info("👈 Estrai o carica dei dati dalla barra laterale.")

# --- TAB 3: CHEAT SHEET ---
with tab_cheat:
    st.markdown("Copia questi Ticker nella barra laterale.")
    c1, c2 = st.columns(2)
    c1.code("^GSPC", language="text"); c1.caption("S&P 500")
    c1.code("SWDA.MI", language="text"); c1.caption("ETF MSCI World (Milano)")
    c2.code("GC=F", language="text"); c2.caption("Oro (Gold)")
