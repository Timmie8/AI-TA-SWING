"""
Swingtrade Dashboard & Multi-Ticker Scanner (1-5 dagen horizon)
===============================================================
- Options Data (Call/Put Volume Ratio & Implied Volatility)
- Short Selling / Short Float Analysis
- Dynamic Support & Resistance Levels
- Machine Learning (XGBoost/RF) & NLP Sentiment Analysis
"""

import concurrent.futures
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Importeer NLTK VADER voor AI Sentiment Analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Importeer Scikit-learn Classifier voor AI ML Score
from sklearn.ensemble import RandomForestClassifier


@st.cache_resource
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


sia = load_vader()

# ---------------------------------------------------------------------------
# PAGINA CONFIGURATIE & STYLING
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Swingtrade Dashboard & Scanner", layout="wide")

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"

GREEN = "#16a34a"
RED = "#dc2626"
ORANGE = "#d97706"
GRAY = "#6b7280"


def colored_box(label, value, color, sub=""):
    st.markdown(
        f"""
        <div style="background-color:{color}22;border-left:6px solid {color};
                    padding:10px 14px;border-radius:6px;margin-bottom:8px;">
            <div style="font-size:13px;color:#374151;font-weight:600;">{label}</div>
            <div style="font-size:22px;font-weight:700;color:{color};">{value}</div>
            <div style="font-size:12px;color:#6b7280;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# DATA ENGINE (YFINANCE, OPTIONS & SHORT INTEREST)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_daily_data(ticker, period="1y"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval="1d", auto_adjust=False)
        return df.dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def get_ticker_info_and_options(ticker):
    """
    Haalt Short Interest, Options Chain metrics en fundamentele data op.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # Short Interest Data
        short_percent = info.get("shortPercentOfFloat", 0) or 0
        short_ratio = info.get("shortRatio", 0) or 0

        # Options Data
        calls_vol, puts_vol, avg_iv = 0, 0, 0
        try:
            expirations = t.expirations
            if expirations:
                nearest_exp = expirations[0]
                opt = t.option_chain(nearest_exp)
                calls_vol = opt.calls["volume"].sum() or 0
                puts_vol = opt.puts["volume"].sum() or 0

                # Implied Volatility berekening
                iv_calls = opt.calls["impliedVolatility"].mean() or 0
                iv_puts = opt.puts["impliedVolatility"].mean() or 0
                avg_iv = (iv_calls + iv_puts) / 2
        except Exception:
            pass

        cp_ratio = round(calls_vol / puts_vol, 2) if puts_vol > 0 else (2.0 if calls_vol > 0 else 1.0)

        return {
            "short_percent": round(short_percent * 100, 2),
            "short_ratio": round(short_ratio, 1),
            "cp_ratio": cp_ratio,
            "avg_iv": round(avg_iv * 100, 1),
        }
    except Exception:
        return {"short_percent": 0.0, "short_ratio": 0.0, "cp_ratio": 1.0, "avg_iv": 0.0}


# ---------------------------------------------------------------------------
# SUPPORT & RESISTANCE & TECHNISCHE BEREKENINGEN
# ---------------------------------------------------------------------------
def calc_support_resistance(df, window=3, lookback=60):
    """
    Berekent de belangrijkste Support en Resistance niveaus op basis van swing highs/lows.
    """
    recent = df.tail(lookback).copy()
    highs, lows = [], []
    h, l = recent["High"].values, recent["Low"].values
    for i in range(window, len(recent) - window):
        if h[i] == max(h[i - window : i + window + 1]):
            highs.append(h[i])
        if l[i] == min(l[i - window : i + window + 1]):
            lows.append(l[i])
    current_price = df["Close"].iloc[-1]
    resistances = sorted([x for x in highs if x > current_price])
    supports = sorted([x for x in lows if x < current_price], reverse=True)
    return (supports[0] if supports else None, resistances[0] if resistances else None)


@st.cache_data(ttl=600)
def compute_ml_swing_prediction(df):
    if len(df) < 80:
        return 50.0, "Onvoldoende data"
    data = df.copy()
    data["Return_1D"] = data["Close"].pct_change(1)
    data["Return_3D"] = data["Close"].pct_change(3)
    data["Return_5D"] = data["Close"].pct_change(5)
    data["Vol_Ratio"] = data["Volume"] / data["Volume"].rolling(10).mean()
    data["SMA20_Dist"] = (data["Close"] - data["Close"].rolling(20).mean()) / data["Close"].rolling(20).mean()
    data["Target"] = (data["Close"].shift(-3) > data["Close"] * 1.015).astype(int)

    feature_cols = ["Return_1D", "Return_3D", "Return_5D", "Vol_Ratio", "SMA20_Dist"]
    clean_data = data.dropna()
    if len(clean_data) < 50:
        return 50.0, "Onvoldoende schone data"

    X, y = clean_data[feature_cols], clean_data["Target"]
    model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X[:-3], y[:-3])

    prob_bullish = model.predict_proba(X.iloc[[-1]])[0][1] * 100
    label = "Stijging (3-5d)" if prob_bullish >= 60 else ("Daling/Risico" if prob_bullish <= 40 else "Neutraal")
    return round(prob_bullish, 1), f"AI Pattern: {label}"


@st.cache_data(ttl=600)
def get_ai_vader_sentiment(ticker):
    try:
        t = yf.Ticker(ticker)
        news = t.news
        if not news:
            return 5.0, "Geen nieuws"
        scores = [sia.polarity_scores(item.get("title", ""))["compound"] for item in news[:10] if item.get("title")]
        if not scores:
            return 5.0, "Geen nieuws"
        avg_compound = np.mean(scores)
        ai_score = round(float((avg_compound + 1) * 4.5 + 1), 1)
        label = "Bullish" if ai_score >= 6.0 else ("Bearish" if ai_score <= 4.0 else "Neutraal")
        return ai_score, f"{label} ({len(scores)} art.)"
    except Exception:
        return 5.0, "Neutraal"


def add_technical_indicators(df):
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


# ---------------------------------------------------------------------------
# CORE ANALYSE FUNCTIE
# ---------------------------------------------------------------------------
def compute_stock_analysis(t_code):
    df = get_daily_data(t_code)
    if df.empty:
        return None

    df = add_technical_indicators(df)
    opt_short = get_ticker_info_and_options(t_code)
    ai_score, ai_subtext = get_ai_vader_sentiment(t_code)
    ml_prob, ml_subtext = compute_ml_swing_prediction(df)
    support, resistance = calc_support_resistance(df)

    last = df.iloc[-1]
    price = last["Close"]
    rsi_val = last["RSI"]
    macd_bullish = last["MACD"] > last["MACD_signal"]
    sma_bullish = price > last["SMA20"]

    score = 0
    reasons = []

    if ml_prob >= 60.0:
        score += 1.5
        reasons.append(f"ML High Prob ({ml_prob}%)")
    elif ml_prob <= 40.0:
        score -= 1.5
        reasons.append(f"ML Low Prob ({ml_prob}%)")

    if opt_short["cp_ratio"] > 1.3:
        score += 1
        reasons.append("Options Call Dominant")
    elif opt_short["cp_ratio"] < 0.7:
        score -= 1
        reasons.append("Options Put Dominant")

    if opt_short["short_percent"] > 15.0:
        score += 0.5
        reasons.append("High Short Squeeze Potential")

    if sma_bullish:
        score += 1
        reasons.append("Boven SMA20")
    else:
        score -= 1
        reasons.append("Onder SMA20")

    if macd_bullish:
        score += 1
        reasons.append("MACD Bullish")

    score = round(score, 1)
    signal = "🟢 BULLISH" if score >= 2.5 else ("🔴 BEARISH" if score <= -2.5 else "🟠 NEUTRAAL")

    return {
        "Ticker": t_code,
        "Koers": round(price, 2),
        "Totaal Score": score,
        "Signaal": signal,
        "ML Kans": f"{ml_prob}%",
        "AI Score": f"{ai_score}/10",
        "Support": f"${support:.2f}" if support else "N/A",
        "Resistance": f"${resistance:.2f}" if resistance else "N/A",
        "Short Float": f"{opt_short['short_percent']}%",
        "Call/Put": opt_short["cp_ratio"],
        "IV": f"{opt_short['avg_iv']}%",
        "Details": " · ".join(reasons),
        "df": df,
        "support_val": support,
        "resistance_val": resistance,
        "ai_score": ai_score,
        "ai_subtext": ai_subtext,
        "ml_prob": ml_prob,
        "ml_subtext": ml_subtext,
        "opt_short": opt_short,
    }


# ---------------------------------------------------------------------------
# DASHBOARD UI
# ---------------------------------------------------------------------------
st.title("📊 Swingtrade Dashboard & Scanner (1-5 dagen)")
st.caption("Includes Options Chain, Short Float, Support & Resistance, AI ML & NLP Models.")

col_input, col_btn = st.columns([3, 1])
with col_input:
    ticker_input = (
        st.text_input("Enkel Aandeel Analyse (bv. AAPL, TSLA, NVDA)", value=st.session_state.selected_ticker)
        .upper()
        .strip()
    )
with col_btn:
    st.write("")
    st.write("")
    if st.button("Analyseer Ticker", type="primary", use_container_width=True):
        st.session_state.selected_ticker = ticker_input

ticker = st.session_state.selected_ticker

if ticker:
    with st.spinner(f"Opties, Short Interest & AI Modellen berekenen voor {ticker}..."):
        res = compute_stock_analysis(ticker)

    if res:
        df = res["df"]
        opt = res["opt_short"]
        overall_color = GREEN if res["Totaal Score"] >= 2.5 else (RED if res["Totaal Score"] <= -2.5 else ORANGE)

        st.markdown("---")
        st.subheader(f"🔍 Uitgebreide Analyse: {ticker} — Koers: ${res['Koers']}")

        colored_box(
            "Overall Swingtrade Signaal",
            f"{res['Signaal']} (Score: {res['Totaal Score']:+g})",
            overall_color,
            res["Details"],
        )

        st.markdown("---")

        # ---- RIJ 1: AI & ML MODEL BEREKENINGEN ------------------------------
        st.markdown("### 🤖 AI Pattern Recognition & Sentiment")
        c0, c1, c2, c3 = st.columns(4)
        with c0:
            ml_color = GREEN if res["ml_prob"] >= 60.0 else (RED if res["ml_prob"] <= 40.0 else ORANGE)
            colored_box("ML AI Pattern Model (3-5d)", f"{res['ml_prob']}% Stijgkans", ml_color, res["ml_subtext"])
        with c1:
            ai_color = GREEN if res["ai_score"] >= 6.0 else (RED if res["ai_score"] <= 4.0 else ORANGE)
            colored_box("NLP AI Nieuws Sentiment", f"{res['ai_score']} / 10", ai_color, res["ai_subtext"])
        with c2:
            colored_box("Support Level (Key Low)", res["Support"], GREEN, "Belangrijkste bodemzone")
        with c3:
            colored_box("Resistance Level (Key High)", res["Resistance"], RED, "Belangrijkste weerstandszone")

        # ---- RIJ 2: OPTIONS & SHORT SELLING ANALYSE -------------------------
        st.markdown("### ⚡ Options & Short Selling Metrics")
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            cp_color = GREEN if opt["cp_ratio"] > 1.2 else (RED if opt["cp_ratio"] < 0.8 else ORANGE)
            colored_box("Call / Put Volume Ratio", f"{opt['cp_ratio']}", cp_color, "> 1.0 = Bullish Sentiment")
        with o2:
            colored_box("Implied Volatility (IV)", f"{opt['avg_iv']}%", GRAY, "Verwachte 30-dagen volatiliteit")
        with o3:
            short_color = RED if opt["short_percent"] > 15.0 else ORANGE
            colored_box("Short Float %", f"{opt['short_percent']}%", short_color, "> 15% = Potential Squeeze")
        with o4:
            colored_box("Days to Cover (Short Ratio)", f"{opt['short_ratio']} Dagen", GRAY, "Tijd nodig om shorts te sluiten")

        st.markdown("---")

        # ---- GRAFIEK MET SUPPORT & RESISTANCE LINE --------------------------
        st.markdown("### Prijsgrafiek met Support & Resistance Levels")
        plot_df = df.tail(90)
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df["Open"],
                high=plot_df["High"],
                low=plot_df["Low"],
                close=plot_df["Close"],
                name="Prijs",
                increasing_line_color=GREEN,
                decreasing_line_color=RED,
            )
        )
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["SMA20"], name="SMA20", line=dict(color="#3b82f6", width=1.5))
        )
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["SMA50"], name="SMA50", line=dict(color="#a855f7", width=1.5))
        )

        if res["support_val"]:
            fig.add_hline(
                y=res["support_val"],
                line_dash="dash",
                line_color=GREEN,
                annotation_text=f"Support ${res['support_val']:.2f}",
            )
        if res["resistance_val"]:
            fig.add_hline(
                y=res["resistance_val"],
                line_dash="dash",
                line_color=RED,
                annotation_text=f"Resistance ${res['resistance_val']:.2f}",
            )

        fig.update_layout(height=450, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# MULTI-TICKER SCANNER MET ALLE STATISTIEKEN
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 Multi-Ticker Live Scanner")
default_tickers = "AAPL, TSLA, NVDA, MSFT, AMD, AMZN, GOOGL, META, PLTR"
scan_input = st.text_area("Voer tickers in:", value=default_tickers, height=70)

if st.button("🚀 Start Multi-Ticker Scan", type="secondary") and scan_input:
    tickers = [t.strip().upper() for t in scan_input.split(",") if t.strip()]
    results = []

    progress = st.progress(0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(compute_stock_analysis, t): t for t in tickers}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            data = future.result()
            if data:
                results.append(data)
            progress.progress((i + 1) / len(tickers))

    if results:
        st.session_state.scan_results = (
            pd.DataFrame(results).sort_values(by="Totaal Score", ascending=False).reset_index(drop=True)
        )

if "scan_results" in st.session_state:
    scan_df = st.session_state.scan_results
    st.markdown("### 🏆 Scan Resultaten")

    # Toon overzichtstabel met alle gezochte velden
    display_cols = ["Ticker", "Signaal", "Totaal Score", "ML Kans", "Support", "Resistance", "Short Float", "Call/Put", "Koers"]
    st.dataframe(scan_df[display_cols], use_container_width=True)

    # Doorklikknoppen
    for idx, row in scan_df.iterrows():
        c1, c2, c3, c4 = st.columns([1.5, 2, 2, 2])
        c1.markdown(f"**{row['Ticker']}** ({row['Koers']})")
        c2.markdown(f"Support: **{row['Support']}** | Res: **{row['Resistance']}**")
        c3.markdown(f"Short: **{row['Short Float']}** | Call/Put: **{row['Call/Put']}**")
        if c4.button(f"📊 Analyseer {row['Ticker']}", key=f"scan_btn_{row['Ticker']}_{idx}"):
            st.session_state.selected_ticker = row["Ticker"]
            st.rerun()
