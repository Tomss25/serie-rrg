import streamlit as st
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
# CONFIGURAZIONE & CSS (Modern Dashboard Style)
# ==========================================
st.set_page_config(page_title="RRG & Historical Data Engine", layout="wide", page_icon="📊")

st.markdown("""
<style>
    /* Google Fonts Import for sleek modern look */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* App Background - Soft blue/gray gradient like a modern dashboard */
    .stApp { 
        background-color: #F4F7FE; 
    }

    /* Sidebar - pure white with soft shadow */
    section[data-testid="stSidebar"] { 
        background-color: #FFFFFF !important; 
        border-right: none !important;
        box-shadow: 14px 17px 40px 4px rgba(112, 144, 176, 0.08) !important;
        padding-top: 1rem;
    }
    
    /* Input fields and Multi-select inside sidebar */
    .stTextArea textarea, .stSelectbox > div > div, .stMultiSelect > div > div {
        background-color: #F4F7FE !important; 
        color: #2B3674 !important; 
        border: none !important; 
        border-radius: 12px !important; 
        font-weight: 500;
        box-shadow: inset 0px 2px 5px rgba(112, 144, 176, 0.05);
    }
    .stTextArea textarea:focus { 
        box-shadow: 0 0 0 2px #4318FF !important; 
    }

    /* Typography */
    h1 { color: #2B3674 !important; font-weight: 800 !important; font-size: 2.2rem !important; margin-bottom: 20px;}
    h2, h3 { color: #2B3674 !important; font-weight: 700 !important; }
    p, label, span { color: #A3AED0 !important; font-weight: 500; }

    /* DataFrame styling */
    div[data-testid="stDataFrame"] { 
        border: none !important; 
        border-radius: 20px; 
        background-color: #FFFFFF; 
        box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.12) !important;
        padding: 10px;
    }

    /* Native metrics (KPIs) customized as floating cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border-radius: 20px;
        padding: 20px 24px;
        box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.12);
        margin-bottom: 20px;
        border: 1px solid rgba(226, 232, 240, 0.5); /* subtle edge */
    }
    div[data-testid="stMetricLabel"] { font-size: 13px !important; color: #A3AED0 !important; text-transform: uppercase; font-weight: 700 !important; letter-spacing: 0.5px;}
    div[data-testid="stMetricValue"] { font-size: 32px !important; font-weight: 800 !important; color: #2B3674 !important; margin-top: 5px; }

    /* Custom Streamlit Tabs looking like pills */
    div[data-baseweb="tab-list"] {
        background-color: transparent !important;
        gap: 15px;
        margin-bottom: 20px;
    }
    button[data-baseweb="tab"] {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        box-shadow: 0px 10px 20px rgba(112, 144, 176, 0.08) !important;
        border: none !important;
        color: #A3AED0 !important;
        font-weight: 700 !important;
        transition: all 0.3s ease;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #4318FF 0%, #39B8FF 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0px 15px 25px rgba(67, 24, 255, 0.3) !important;
    }

    /* Sidebar and action buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #4318FF 0%, #39B8FF 100%);
        color: white !important; 
        border: none; 
        padding: 0.8rem 1rem; 
        border-radius: 12px; 
        font-weight: 700;
        box-shadow: 0 10px 20px rgba(67, 24, 255, 0.25); 
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        box-shadow: 0 14px 24px rgba(67, 24, 255, 0.4); 
        transform: translateY(-2px);
    }
    
    /* Code blocks customization */
    .stCodeBlock {
        border-radius: 16px !important;
        box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.12) !important;
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
st.sidebar.title("App Settings  🟢")
st.sidebar.markdown("<p style='margin-top: -10px; font-size: 14px;'>Data Source configuration</p>", unsafe_allow_html=True)

data_mode = st.sidebar.radio("Data Mode", ["API Download (Yahoo/MStar)", "File Upload (CSV/Excel)"])

if data_mode == "API Download (Yahoo/MStar)":
    raw_input = st.sidebar.text_area("List of Tickers/ISIN (First = Benchmark)", "SP500\nSWDA.MI\nLU1287022708\nGOLD", height=120)
    years = st.sidebar.selectbox("Time Horizon (Years)", [1, 3, 5, 10, 20], index=1)
    freq_label = st.sidebar.selectbox("Data Frequency", ["Daily", "Weekly", "Monthly"], index=1)
    freq_code = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}[freq_label]
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    if st.sidebar.button("📥 Retrieve & Sync Data"):
        raw_tokens = re.findall(r"[\w\.\-\^\=]+", raw_input.upper())
        tickers_input = [ALIAS_MAP.get(t, t) for t in raw_tokens]
        if tickers_input:
            start_dt = datetime.now() - timedelta(days=years*365)
            end_dt = datetime.now()
            all_series = {}
            with st.spinner('Syncing global data...'):
                for t in tickers_input:
                    s = get_data_yahoo(t, start_dt, end_dt)
                    if s is None or s.empty: s = get_data_morningstar(t, start_dt, end_dt)
                    if s is not None and not s.empty:
                        s.name = t
                        all_series[t] = s
                    else: st.sidebar.warning(f"No Data for {t}")
            if all_series:
                df_raw = pd.DataFrame(all_series).ffill().dropna()
                st.session_state.master_df = resample_prices(df_raw, freq_code)
                st.session_state.data_source_type = "API"
                st.sidebar.success("✅ Dashboard synced!")

elif data_mode == "File Upload (CSV/Excel)":
    uploaded = st.sidebar.file_uploader("Upload File", type=["xlsx", "xls", "csv"])
    freq_label = st.sidebar.selectbox("Resample Data", ["None", "Weekly", "Monthly"])
    freq_code = {"None": "D", "Weekly": "W", "Monthly": "ME"}[freq_label]
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    if uploaded and st.sidebar.button("📂 Analyze File"):
        try:
            df_raw = parse_file(uploaded)
            st.session_state.master_df = resample_prices(df_raw, freq_code)
            st.session_state.data_source_type = "FILE"
            st.sidebar.success("✅ File loaded successfully!")
        except Exception as e: st.sidebar.error(f"Error: {e}")

# ==========================================
# MAIN APP TABS
# ==========================================
st.title("Asset & RRG Intelligence")
tab_rrg, tab_data, tab_cheat = st.tabs(["🎯 Relative Rotation Graph", "📉 Historical Data", "📋 Cheat Sheet"])

# Modern Dashboard Palette from the image inspiration
SECTOR_COLORS = ["#4318FF", "#01B574", "#FFCE20", "#EE5D50", "#39B8FF", "#868CFF", "#F6AD55", "#E53E3E", "#D53F8C", "#38A169"]

# --- TAB 1: RRG & MACRO MAP ---
with tab_rrg:
    if st.session_state.master_df is not None and len(st.session_state.master_df.columns) >= 2:
        df_rrg = st.session_state.master_df
        all_cols = list(df_rrg.columns)
        
        # Dashboard Config Area
        c1, c2, c3 = st.columns([1.5, 3, 1.5])
        with c1: benchmark_col = st.selectbox("Benchmark", all_cols, index=0)
        with c2: sector_cols = st.multiselect("Asset Watchlist", [c for c in all_cols if c != benchmark_col], default=[c for c in all_cols if c != benchmark_col])
        with c3: method = st.selectbox("Scoring Method", ["Standard JdK", "Z-Score Method"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        opt1, opt2, opt3 = st.columns(3)
        show_trails = opt1.toggle("Show Historical Trails", value=True)
        show_labels = opt2.toggle("Show Asset Labels", value=True)
        trail_length = opt3.slider("Trail Length (periods)", min_value=2, max_value=24, value=8)

        st.markdown("<hr style='border:1px solid #E2E8F0; margin: 30px 0;'>", unsafe_allow_html=True)

        if sector_cols:
            try:
                if "JdK" in method: results = compute_jdk_method(df_rrg, benchmark_col, sector_cols)
                else: results = compute_zscore_method(df_rrg, benchmark_col, sector_cols)
                
                # --- KPI CARDS METRICS ---
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                
                # compute best quadrant
                leads = []
                for name, v in results.items():
                    r, m = v["rs_ratio"].dropna(), v["rs_momentum"].dropna()
                    if r.empty: continue
                    if get_quadrant(r.iloc[-1], m.iloc[-1]) == "leading": leads.append(name)
                
                kpi1.metric("Assets Analyzed", len(sector_cols))
                kpi2.metric("Benchmark", benchmark_col)
                kpi3.metric("Leading Assets", len(leads))
                kpi4.metric("Strategy", "Top 5%")
                
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

                    # Subtle trails
                    if show_trails:
                        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', line=dict(width=3.5, color=color), marker=dict(size=6, color=color), opacity=0.35, showlegend=False, legendgroup=name, name=name))
                    
                    # Modern glossy markers
                    mode = 'markers+text' if show_labels else 'markers'
                    text_label = [f"<b>{name}</b>"] if show_labels else None
                    fig.add_trace(go.Scatter(x=[xs[-1]], y=[ys[-1]], mode=mode, name=name, text=text_label, textposition="top center", 
                                             textfont=dict(color="#2B3674", size=13, family="Plus Jakarta Sans"), 
                                             marker=dict(size=18, color=color, line=dict(width=3, color="#FFFFFF")),
                                             showlegend=not show_labels, legendgroup=name))
                
                fig.add_vline(x=100, line_width=2, line_color="#2B3674", opacity=0.8)
                fig.add_hline(y=100, line_width=2, line_color="#2B3674", opacity=0.8)
                
                pad_x = max(2, max_dev_x * 1.1)
                pad_y = max(2, max_dev_y * 1.1)

                # Modern glassy quadrant background
                fig.add_shape(type="rect", x0=100, y0=100, x1=100 + pad_x, y1=100 + pad_y, fillcolor="rgba(1, 181, 116, 0.05)", line_width=0, layer="below")
                fig.add_shape(type="rect", x0=100 - pad_x, y0=100, x1=100, y1=100 + pad_y, fillcolor="rgba(57, 184, 255, 0.05)", line_width=0, layer="below")
                fig.add_shape(type="rect", x0=100 - pad_x, y0=100 - pad_y, x1=100, y1=100, fillcolor="rgba(238, 93, 80, 0.05)", line_width=0, layer="below")
                fig.add_shape(type="rect", x0=100, y0=100 - pad_y, x1=100 + pad_x, y1=100, fillcolor="rgba(255, 206, 32, 0.05)", line_width=0, layer="below")
                
                fig.update_layout(
                    template="plotly_white", height=700, 
                    xaxis=dict(title="<b>RS-Ratio (Relative Strength) ➔</b>", zeroline=False, gridcolor="#F4F7FE", range=[100 - pad_x, 100 + pad_x], tickfont=dict(color="#A3AED0")), 
                    yaxis=dict(title="<b>RS-Momentum (Velocity) ➔</b>", zeroline=False, gridcolor="#F4F7FE", range=[100 - pad_y, 100 + pad_y], tickfont=dict(color="#A3AED0")),
                    margin=dict(l=50, r=50, t=30, b=50), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                    hoverlabel=dict(bgcolor="#111C44", font_size=13, font_color="#FFFFFF", font_family="Plus Jakarta Sans"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#2B3674")) if not show_labels else None
                )
                
                # Modern text annotations
                fig.add_annotation(x=1.0, y=1.0, xref="paper", yref="paper", text="LEADING", showarrow=False, font=dict(size=18, color="#01B574", family="Plus Jakarta Sans", weight="800"), opacity=0.15)
                fig.add_annotation(x=0.0, y=1.0, xref="paper", yref="paper", text="IMPROVING", showarrow=False, font=dict(size=18, color="#39B8FF", family="Plus Jakarta Sans", weight="800"), opacity=0.15)
                fig.add_annotation(x=0.0, y=0.0, xref="paper", yref="paper", text="LAGGING", showarrow=False, font=dict(size=18, color="#EE5D50", family="Plus Jakarta Sans", weight="800"), opacity=0.15)
                fig.add_annotation(x=1.0, y=0.0, xref="paper", yref="paper", text="WEAKENING", showarrow=False, font=dict(size=18, color="#FFCE20", family="Plus Jakarta Sans", weight="800"), opacity=0.15)

                st.plotly_chart(fig, use_container_width=True)

                # --- SECONDA RIGA: MACRO MAP e RISULTATI ---
                st.markdown("<hr style='border:1px solid #E2E8F0; margin: 30px 0;'>", unsafe_allow_html=True)
                
                col_macro, col_table = st.columns([1, 1])
                
                with col_macro:
                    st.subheader("Macro Mapping Cycle")
                    
                    fig_macro = go.Figure()
                    
                    x_curve = np.linspace(0, 4, 300)
                    y_curve = np.sin(x_curve * np.pi / 2 - np.pi/2)
                    
                    fig_macro.add_vrect(x0=0, x1=1, fillcolor="rgba(1, 181, 116, 0.05)", line_width=0, annotation_text="Expansion", annotation_position="bottom right", annotation_font_color="#01B574")
                    fig_macro.add_vrect(x0=1, x1=2, fillcolor="rgba(238, 93, 80, 0.05)", line_width=0, annotation_text="Contraction", annotation_position="bottom right", annotation_font_color="#EE5D50")
                    fig_macro.add_vrect(x0=2, x1=3, fillcolor="rgba(238, 93, 80, 0.05)", line_width=0, annotation_text="Recession", annotation_position="bottom right", annotation_font_color="#EE5D50")
                    fig_macro.add_vrect(x0=3, x1=4, fillcolor="rgba(1, 181, 116, 0.05)", line_width=0, annotation_text="Recovery", annotation_position="bottom right", annotation_font_color="#01B574")

                    fig_macro.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', line=dict(color="#4318FF", width=3), showlegend=False))
                    
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
                        
                        fig_macro.add_trace(go.Scatter(x=[macro_x], y=[macro_y], mode=mode_macro, name=name, text=text_label_macro, textposition="top center", 
                                                       textfont=dict(color="#2B3674", size=12, family="Plus Jakarta Sans"), 
                                                       marker=dict(size=16, color=color, line=dict(width=3, color="#FFFFFF")), 
                                                       showlegend=not show_labels, legendgroup=name))

                    fig_macro.update_layout(
                        template="plotly_white", height=450, 
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), 
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), 
                        margin=dict(l=20, r=20, t=20, b=40),
                        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if not show_labels else None
                    )
                    st.plotly_chart(fig_macro, use_container_width=True)

                with col_table:
                    # Tabella Risultati Clean
                    st.subheader("Asset Pulse")
                    tbl = []
                    for name, v in results.items():
                        r, m = v["rs_ratio"].dropna(), v["rs_momentum"].dropna()
                        if r.empty: continue
                        tbl.append({"Asset": name, "RS-Ratio": round(r.iloc[-1],2), "RS-Mom": round(m.iloc[-1],2), "Status": get_quadrant(r.iloc[-1], m.iloc[-1]).upper()})
                    
                    df_status = pd.DataFrame(tbl).sort_values("RS-Ratio", ascending=False).set_index("Asset")
                    st.dataframe(df_status, use_container_width=True, height=400)
                    
            except Exception as e: st.error(f"Error computing data: {e}")
    else: st.info("👋 Welcome! Use the sidebar settings to retrieve Yahoo/Morningstar data and populate your dashboard.")

# --- TAB 2: SERIE STORICHE ---
with tab_data:
    if st.session_state.master_df is not None:
        df_final = st.session_state.master_df
        
        st.subheader("Asset Normalized Performance")
        st.line_chart((df_final / df_final.iloc[0]) * 100, height=350, use_container_width=True)
        
        st.markdown("<hr style='border:1px solid #E2E8F0; margin: 30px 0;'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1.5, 2])
        with col1:
            st.subheader("Risk & Return Profile")
            metrics = []
            ann_factor = 252 if df_final.index.to_series().diff().mean().days < 4 else (52 if df_final.index.to_series().diff().mean().days < 10 else 12)
            for col in df_final.columns:
                s = df_final[col]
                ret = ((s.iloc[-1] / s.iloc[0]) - 1) * 100
                vol = s.pct_change().std() * np.sqrt(ann_factor) * 100
                dd = ((s - s.cummax()) / s.cummax()).min() * 100
                metrics.append({"Asset": col, "Return %": round(ret,2), "Volatility %": round(vol,2), "Max DD %": round(dd,2)})
            st.dataframe(pd.DataFrame(metrics).set_index("Asset"), use_container_width=True)
            
        with col2:
            st.subheader("Raw History Logs")
            st.dataframe(df_final.sort_index(ascending=False).round(4), use_container_width=True, height=400)
    else: st.info("👈 Please sync data from the sidebar to view historical logs.")

# --- TAB 3: CHEAT SHEET ---
with tab_cheat:
    st.markdown("### Quick Asset Identifiers")
    st.markdown("Copy these tickers into the sidebar input box.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Core Indices**")
        st.code("^GSPC", language="text"); st.caption("S&P 500")
        st.code("^NDX", language="text"); st.caption("NASDAQ-100")
        st.code("^DJI", language="text"); st.caption("Dow Jones Industrial")
        
    with col2:
        st.markdown("**European ETFs / Tickers**")
        st.code("SWDA.MI", language="text"); st.caption("ETF MSCI World (Milan)")
        st.code("^GDAXI", language="text"); st.caption("DAX")
        st.code("^STOXX50E", language="text"); st.caption("EURO STOXX 50")
        
    with col3:
        st.markdown("**Commodities & Crypto**")
        st.code("GC=F", language="text"); st.caption("Gold (Futures)")
        st.code("CL=F", language="text"); st.caption("Crude Oil (Futures)")
        st.code("BTC-USD", language="text"); st.caption("Bitcoin vs USD")
