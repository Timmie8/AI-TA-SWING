"""
Swingtrade Dashboard & Multi-Ticker Scanner (1-5 dagen horizon)
===============================================================
- Individuele diepgaande analyse
- Multi-Ticker Scanner met interactieve doorklik functionaliteit
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


# Download VADER lexicon indien nog niet aanwezig
@st.cache_resource
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


sia = load_vader()

# ---------------------------------------------------------------------------
# PAGINA CONFIGURATIE & SESSION STATE
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Swingtrade Dashboard & Scanner", layout="wide"
)

# Onthoud geselecteerde ticker
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
# DATA OPHALEN & BEREKENINGEN
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_daily_data(ticker, period="6mo"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval="1d", auto_adjust=False)
        return df.dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_intraday_data(ticker, period="5d", interval="5m"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        return df.dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def get_ai_vader_sentiment(ticker):
    try:
        t = yf.Ticker(ticker)
        news = t.news
        if not news:
            return 5.0, "Geen nieuws"

        compound_scores = []
        titles_analyzed = 0

        for item in news[:10]:
            title = item.get("title", "")
            if title:
                score_dict = sia.polarity_scores(title)
                compound_scores.append(score_dict["compound"])
                titles_analyzed += 1

        if not compound_scores:
            return 5.0, "Geen nieuws"

        avg_compound = np.mean(compound_scores)
        ai_score = round(float((avg_compound + 1) * 4.5 + 1), 1)
        label = (
            "Bullish"
            if ai_score >= 6.0
            else ("Bearish" if ai_score <= 4.0 else "Neutraal")
        )
        return ai_score, f"{label} ({titles_analyzed} art.)"
    except Exception:
        return 5.0, "Neutraal"


@st.cache_data(ttl=900)
def get_option_data(ticker):
    try:
        t = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return None
        nearest = expiries[0]
        chain = t.option_chain(nearest)
        calls, puts = chain.calls, chain.puts
        call_vol = calls["volume"].fillna(0).sum() if "volume" in calls else 0
        put_vol = puts["volume"].fillna(0).sum() if "volume" in puts else 0
        pcr_vol = put_vol / call_vol if call_vol > 0 else np.nan
        return {"expiry": nearest, "pcr_volume": pcr_vol}
    except Exception:
        return None


@st.cache_data(ttl=1800)
def get_short_data(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {"short_pct_float": info.get("shortPercentOfFloat")}
    except Exception:
        return {}


def add_moving_averages(df):
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    return df


def calc_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    avg_loss = -loss.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def calc_support_resistance(df, window=3, lookback=60):
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
    return (
        supports[0] if supports else None,
        resistances[0] if resistances else None,
    )


def intraday_money_flow(df_intraday):
    if df_intraday.empty:
        return None
    today = df_intraday.index[-1].date()
    today_df = df_intraday[df_intraday.index.date == today]
    if today_df.empty:
        today_df = df_intraday.tail(78)

    up_vol = today_df.loc[
        today_df["Close"] >= today_df["Open"], "Volume"
    ].sum()
    down_vol = today_df.loc[
        today_df["Close"] < today_df["Open"], "Volume"
    ].sum()
    total = up_vol + down_vol
    net_pct = ((up_vol - down_vol) / total * 100) if total > 0 else 0
    return {"up_vol": up_vol, "down_vol": down_vol, "net_pct": net_pct}


def analyze_3day_candles(df):
    last3 = df.tail(3)
    if len(last3) < 3:
        return "Onvoldoende data", GRAY
    colors = [
        "Groen" if c >= o else "Rood"
        for o, c in zip(last3["Open"], last3["Close"])
    ]
    closes = last3["Close"].values
    if (
        colors == ["Groen", "Groen", "Groen"]
        and closes[0] < closes[1] < closes[2]
    ):
        return "3 opeenvolgende groene candles (sterk bullish momentum)", GREEN
    if colors == ["Rood", "Rood", "Rood"] and closes[0] > closes[1] > closes[2]:
        return "3 opeenvolgende rode candles (sterk bearish momentum)", RED
    if colors[-1] == "Groen" and colors[-2] == "Rood":
        return "Herstel: laatste candle groen na rode candle(s)", GREEN
    if colors[-1] == "Rood" and colors[-2] == "Groen":
        return "Verzwakking: laatste candle rood na groene candle(s)", RED
    return f"Gemengd patroon ({', '.join(colors)})", ORANGE


def determine_trend(df):
    last = df.iloc[-1]
    if pd.isna(last.get("SMA20")):
        return "Onvoldoende data", GRAY
    if pd.isna(last.get("SMA50")):
        if last["Close"] > last["SMA20"]:
            return "Bullish (prijs > SMA20)", GREEN
        return "Bearish (prijs < SMA20)", RED
    if last["Close"] > last["SMA20"] > last["SMA50"]:
        return "Bullish (prijs > SMA20 > SMA50)", GREEN
    if last["Close"] < last["SMA20"] < last["SMA50"]:
        return "Bearish (prijs < SMA20 < SMA50)", RED
    return "Neutraal / zijwaarts", ORANGE


# ---------------------------------------------------------------------------
# CORE ANALYSE FUNCTIE
# ---------------------------------------------------------------------------
def compute_stock_analysis(t_code):
    df = get_daily_data(t_code)
    if df.empty:
        return None

    df_intraday = get_intraday_data(t_code)
    ai_score, ai_subtext = get_ai_vader_sentiment(t_code)

    df = add_moving_averages(df)
    df["RSI"] = calc_rsi(df)
    macd_line, signal_line, hist = calc_macd(df)
    df["MACD"], df["MACD_signal"] = macd_line, signal_line

    trend_label, trend_color = determine_trend(df)
    candle3_label, candle3_color = analyze_3day_candles(df)
    flow_today = intraday_money_flow(df_intraday)

    last = df.iloc[-1]
    price = last["Close"]
    rsi_val = last["RSI"]
    macd_bullish = last["MACD"] > last["MACD_signal"]

    score = 0
    reasons = []

    if ai_score >= 6.0:
        score += 1
        reasons.append("AI Bullish")
    elif ai_score <= 4.0:
        score -= 1
        reasons.append("AI Bearish")

    if trend_color == GREEN:
        score += 1
        reasons.append("Trend Bullish")
    elif trend_color == RED:
        score -= 1
        reasons.append("Trend Bearish")

    if candle3_color == GREEN:
        score += 1
        reasons.append("3-daagse Bullish")
    elif candle3_color == RED:
        score -= 1
        reasons.append("3-daagse Bearish")

    if not pd.isna(rsi_val):
        if rsi_val < 30:
            score += 1
            reasons.append("RSI Oversold")
        elif rsi_val > 70:
            score -= 1
            reasons.append("RSI Overbought")

    if macd_bullish:
        score += 1
        reasons.append("MACD Bullish")
    else:
        score -= 1
        reasons.append("MACD Bearish")

    if flow_today and flow_today["net_pct"] > 5:
        score += 1
        reasons.append("Flow Positief")
    elif flow_today and flow_today["net_pct"] < -5:
        score -= 1
        reasons.append("Flow Negatief")

    if score >= 2:
        signal = "🟢 BULLISH"
    elif score <= -2:
        signal = "🔴 BEARISH"
    else:
        signal = "🟠 NEUTRAAL"

    return {
        "Ticker": t_code,
        "Koers": round(price, 2),
        "Totaal Score": score,
        "Signaal": signal,
        "AI Score": f"{ai_score}/10",
        "RSI": round(rsi_val, 1) if not pd.isna(rsi_val) else "-",
        "Trend": trend_label.split(" (")[0],
        "Details": " · ".join(reasons),
        "df": df,
        "df_intraday": df_intraday,
        "ai_score": ai_score,
        "ai_subtext": ai_subtext,
        "trend_label": trend_label,
        "trend_color": trend_color,
        "candle3_label": candle3_label,
        "candle3_color": candle3_color,
        "flow_today": flow_today,
    }


# ---------------------------------------------------------------------------
# GEBRUIKERSINTERFACE (DASHBOARD)
# ---------------------------------------------------------------------------
st.title("📊 Swingtrade Dashboard & Scanner (1-5 dagen)")
st.caption(
    "Gratis data via yfinance · trend, momentum, NLP AI-nieuws sentiment & multi-ticker scanner."
)

col_input, col_btn = st.columns([3, 1])
with col_input:
    ticker_input = (
        st.text_input(
            "Enkel Aandeel Analyse (bv. AAPL, TSLA, NVDA)",
            value=st.session_state.selected_ticker,
        )
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
    with st.spinner(f"Data & AI Sentiment ophalen voor {ticker}..."):
        res = compute_stock_analysis(ticker)

    if not res:
        st.error(f"Geen data gevonden voor ticker '{ticker}'. Controleer de invoer.")
    else:
        df = res["df"]
        df_intraday = res["df_intraday"]
        ai_score = res["ai_score"]
        ai_subtext = res["ai_subtext"]
        trend_label = res["trend_label"]
        trend_color = res["trend_color"]
        candle3_label = res["candle3_label"]
        candle3_color = res["candle3_color"]
        flow_today = res["flow_today"]
        score = res["Totaal Score"]

        support, resistance = calc_support_resistance(df)

        last = df.iloc[-1]
        price = last["Close"]
        rsi_val = last["RSI"]

        overall_color = (
            GREEN if score >= 2 else (RED if score <= -2 else ORANGE)
        )

        # ---- DISPLAY HEADER -------------------------------------------------
        st.markdown("---")
        st.subheader(f"🔍 Uitgebreide Analyse: {ticker} — Laatste koers: ${price:.2f}")

        colored_box(
            "Overall Swingtrade Signaal",
            f"{res['Signaal']} (Score: {score:+d})",
            overall_color,
            res["Details"],
        )

        st.markdown("---")

        # ---- RIJ 1: AI Score & Trend ----------------------------------------
        st.markdown("### AI Sentiment & Basissignalen")
        c0, c1, c2, c3 = st.columns(4)
        with c0:
            ai_color = (
                GREEN if ai_score >= 6.0 else (RED if ai_score <= 4.0 else ORANGE)
            )
            colored_box(
                "NLP AI Sentiment Score", f"{ai_score} / 10", ai_color, ai_subtext
            )
        with c1:
            colored_box(
                "Trend (SMA20/SMA50)",
                trend_label.split(" (")[0],
                trend_color,
                sub=trend_label,
            )
        with c2:
            colored_box(
                "3-daagse candle status",
                candle3_label.split(" (")[0],
                candle3_color,
                sub=candle3_label,
            )
        with c3:
            if pd.isna(rsi_val):
                colored_box("RSI (14)", "n.v.t.", GRAY, "Onvoldoende historie")
            else:
                rsi_color = (
                    RED if rsi_val > 70 else (GREEN if rsi_val < 30 else ORANGE)
                )
                rsi_sub = (
                    "Overbought (>70)"
                    if rsi_val > 70
                    else ("Oversold (<30)" if rsi_val < 30 else "Neutraal")
                )
                colored_box("RSI (14)", f"{rsi_val:.1f}", rsi_color, rsi_sub)

        # ---- EXTERNE AI LINKS -----------------------------------------------
        st.markdown("### 🤖 Externe AI Scores & Platform Verwijzingen")
        ext1, ext2, ext3, ext4 = st.columns(4)
        with ext1:
            st.markdown(
                f"**[Danelfin AI Score voor {ticker}](https://danelfin.com/stock/{ticker})**"
            )
        with ext2:
            st.markdown(
                f"**[Investing.com {ticker}](https://www.investing.com/search/?q={ticker})**"
            )
        with ext3:
            st.markdown(
                f"**[MarketScreener {ticker}](https://www.marketscreener.com/search/?q={ticker})**"
            )
        with ext4:
            st.markdown(
                f"**[Finviz Quote {ticker}](https://finviz.com/quote.ashx?t={ticker})**"
            )

        st.markdown("---")

        # ---- GRAFIEK ---------------------------------------------------------
        st.markdown("### Prijsgrafiek met Moving Averages & Levels")
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
            go.Scatter(
                x=plot_df.index,
                y=plot_df["SMA20"],
                name="SMA20",
                line=dict(color="#3b82f6", width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["SMA50"],
                name="SMA50",
                line=dict(color="#a855f7", width=1.5),
            )
        )

        if support:
            fig.add_hline(
                y=support,
                line_dash="dash",
                line_color=GREEN,
                annotation_text=f"Support ${support:.2f}",
            )
        if resistance:
            fig.add_hline(
                y=resistance,
                line_dash="dash",
                line_color=RED,
                annotation_text=f"Resistance ${resistance:.2f}",
            )

        fig.update_layout(
            height=450,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# MULTI-TICKER SCANNER (ONDER HET DASHBOARD)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 Multi-Ticker Live Scanner")
st.caption(
    "Scan meerdere aandelen tegelijk. Klik op de knop naast een aandeel om direct de uitgebreide analyse te bekijken."
)

default_tickers = "AAPL, TSLA, NVDA, MSFT, AMD, AMZN, GOOGL, META, PLTR"
scan_input = st.text_area(
    "Voer tickers in (gescheiden door komma's):",
    value=default_tickers,
    height=70,
)

scan_btn = st.button("🚀 Start Multi-Ticker Scan", type="secondary")

# Sla scan resultaten op in sessie zodat ze niet verdwijnen bij interactie
if scan_btn and scan_input:
    tickers_to_scan = [
        t.strip().upper() for t in scan_input.split(",") if t.strip()
    ]

    if tickers_to_scan:
        st.info(f"Bezig met scannen van {len(tickers_to_scan)} aandelen...")
        progress_bar = st.progress(0)
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {
                executor.submit(compute_stock_analysis, t): t
                for t in tickers_to_scan
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_ticker):
                data = future.result()
                if data:
                    results.append(data)
                completed += 1
                progress_bar.progress(completed / len(tickers_to_scan))

        if results:
            st.session_state.scan_results = (
                pd.DataFrame(results)
                .sort_values(by="Totaal Score", ascending=False)
                .reset_index(drop=True)
            )

# Weergave van de resultaten indien gescand
if "scan_results" in st.session_state:
    scan_df = st.session_state.scan_results

    st.markdown("### 🏆 Scan Resultaten (Klik op de knop om te analyseren)")

    # Maak per aandeel een overzichtelijke interactieve rij
    for idx, row in scan_df.iterrows():
        col_t, col_sig, col_score, col_price, col_ai, col_rsi, col_act = (
            st.columns([1.2, 1.5, 1.2, 1.2, 1.2, 1, 2])
        )

        col_t.markdown(f"**{row['Ticker']}**")
        col_sig.markdown(row["Signaal"])
        col_score.markdown(f"**Score: {row['Totaal Score']}**")
        col_price.markdown(f"${row['Koers']}")
        col_ai.markdown(row["AI Score"])
        col_rsi.markdown(str(row["RSI"]))

        # Als op de knop geklikt wordt, update de gekozen ticker en ververs pagina
        if col_act.button(
            f"📊 Analyseer {row['Ticker']}", key=f"btn_{row['Ticker']}_{idx}"
        ):
            st.session_state.selected_ticker = row["Ticker"]
            st.rerun()

        st.markdown(
            "<hr style='margin: 4px 0px; border-top: 1px solid #eee;'>",
            unsafe_allow_html=True,
        )
