"""
Swingtrade Dashboard (1-5 dagen horizon)
=========================================
Gratis databron: yfinance

Combineert:
- Trend (SMA20/SMA50)
- Momentum (RSI, MACD)
- Support & Resistance (korte termijn swingpunten)
- Put/Call ratio (optiesentiment, dichtstbijzijnde expiratie)
- Short interest (percentage van de free float, bi-wekelijkse data)
- 1-dags money flow (intraday up-volume vs down-volume)
- 3-daagse candle status (momentum van de laatste 3 dagcandles)

LET OP: dit is een informatief hulpmiddel, geen financieel advies.
Alle data komt van Yahoo Finance via yfinance en kan vertraagd of
onvolledig zijn (vooral optiedata en short interest).
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# ---------------------------------------------------------------------------
# PAGINA CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Swingtrade Dashboard", layout="wide")

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
# DATA OPHALEN
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_daily_data(ticker, period="6mo"):
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    df = df.dropna()
    return df


@st.cache_data(ttl=300)
def get_intraday_data(ticker, period="5d", interval="5m"):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        return df.dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900)
def get_option_data(ticker):
    t = yf.Ticker(ticker)
    try:
        expiries = t.options
        if not expiries:
            return None
        nearest = expiries[0]
        chain = t.option_chain(nearest)
        calls, puts = chain.calls, chain.puts
        call_vol = calls["volume"].fillna(0).sum()
        put_vol = puts["volume"].fillna(0).sum()
        call_oi = calls["openInterest"].fillna(0).sum()
        put_oi = puts["openInterest"].fillna(0).sum()
        pcr_vol = put_vol / call_vol if call_vol > 0 else np.nan
        pcr_oi = put_oi / call_oi if call_oi > 0 else np.nan
        return {
            "expiry": nearest,
            "pcr_volume": pcr_vol,
            "pcr_oi": pcr_oi,
            "call_volume": call_vol,
            "put_volume": put_vol,
        }
    except Exception:
        return None


@st.cache_data(ttl=1800)
def get_short_data(ticker):
    try:
        info = yf.Ticker(ticker).get_info()
        return {
            "short_pct_float": info.get("shortPercentOfFloat"),
            "short_ratio": info.get("shortRatio"),
            "shares_short": info.get("sharesShort"),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# INDICATOREN
# ---------------------------------------------------------------------------
def add_moving_averages(df):
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    return df


def calc_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def calc_support_resistance(df, window=3, lookback=60):
    """Vind lokale swing-highs/lows in de laatste `lookback` dagen."""
    recent = df.tail(lookback).copy()
    highs, lows = [], []
    h, l = recent["High"].values, recent["Low"].values
    for i in range(window, len(recent) - window):
        if h[i] == max(h[i - window:i + window + 1]):
            highs.append(h[i])
        if l[i] == min(l[i - window:i + window + 1]):
            lows.append(l[i])
    current_price = df["Close"].iloc[-1]
    resistances = sorted([x for x in highs if x > current_price])
    supports = sorted([x for x in lows if x < current_price], reverse=True)
    nearest_resistance = resistances[0] if resistances else None
    nearest_support = supports[0] if supports else None
    return nearest_support, nearest_resistance


def calc_money_flow_index(df, period=14):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_flow = tp * df["Volume"]
    direction = tp.diff()
    pos_flow = raw_flow.where(direction > 0, 0.0)
    neg_flow = raw_flow.where(direction < 0, 0.0)
    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()
    mfr = pos_sum / neg_sum
    mfi = 100 - (100 / (1 + mfr))
    return mfi


def intraday_money_flow(df_intraday):
    """Vergelijkt up-volume vs down-volume van vandaag op basis van intraday bars."""
    if df_intraday.empty:
        return None
    today = df_intraday.index[-1].date()
    today_df = df_intraday[df_intraday.index.date == today]
    if today_df.empty:
        today_df = df_intraday.tail(78)  # fallback: laatste sessie bij 5m bars
    up_vol = today_df.loc[today_df["Close"] >= today_df["Open"], "Volume"].sum()
    down_vol = today_df.loc[today_df["Close"] < today_df["Open"], "Volume"].sum()
    total = up_vol + down_vol
    net_pct = ((up_vol - down_vol) / total * 100) if total > 0 else 0
    return {"up_vol": up_vol, "down_vol": down_vol, "net_pct": net_pct}


def analyze_3day_candles(df):
    last3 = df.tail(3)
    if len(last3) < 3:
        return "Onvoldoende data", GRAY
    colors = ["Groen" if c >= o else "Rood" for o, c in zip(last3["Open"], last3["Close"])]
    closes = last3["Close"].values
    if colors == ["Groen", "Groen", "Groen"] and closes[0] < closes[1] < closes[2]:
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
    if pd.isna(last["SMA50"]):
        # niet genoeg historie voor SMA50, val terug op SMA20 vs prijs
        if last["Close"] > last["SMA20"]:
            return "Bullish (prijs > SMA20)", GREEN
        return "Bearish (prijs < SMA20)", RED
    if last["Close"] > last["SMA20"] > last["SMA50"]:
        return "Bullish (prijs > SMA20 > SMA50)", GREEN
    if last["Close"] < last["SMA20"] < last["SMA50"]:
        return "Bearish (prijs < SMA20 < SMA50)", RED
    return "Neutraal / zijwaarts", ORANGE


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📊 Swingtrade Dashboard (1-5 dagen)")
st.caption(
    "Gratis data via yfinance · trend, momentum, support/resistance, "
    "put/call ratio, short interest en money flow in één overzicht."
)

col_input, col_btn = st.columns([3, 1])
with col_input:
    ticker = st.text_input("Ticker (bv. AAPL, TSLA, NVDA)", value="AAPL").upper().strip()
with col_btn:
    st.write("")
    st.write("")
    run = st.button("Analyseer", type="primary", use_container_width=True)

if run and ticker:
    with st.spinner(f"Data ophalen voor {ticker}..."):
        df = get_daily_data(ticker)
        df_intraday = get_intraday_data(ticker)
        option_data = get_option_data(ticker)
        short_data = get_short_data(ticker)

    if df.empty:
        st.error("Geen data gevonden. Controleer het ticker-symbool.")
        st.stop()

    df = add_moving_averages(df)
    df["RSI"] = calc_rsi(df)
    macd_line, signal_line, hist = calc_macd(df)
    df["MACD"] = macd_line
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = hist
    df["MFI"] = calc_money_flow_index(df)

    support, resistance = calc_support_resistance(df)
    trend_label, trend_color = determine_trend(df)
    candle3_label, candle3_color = analyze_3day_candles(df)
    flow_today = intraday_money_flow(df_intraday)

    last = df.iloc[-1]
    price = last["Close"]
    rsi_val = last["RSI"]
    mfi_val = last["MFI"]

    # ---- RIJ 1: Prijs, Trend, 3-daags candle, RSI ---------------------
    st.subheader(f"{ticker} — laatste koers: {price:.2f}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        colored_box("Trend (SMA20/SMA50)", trend_label.split(" (")[0], trend_color,
                     sub=trend_label)
    with c2:
        colored_box("3-daagse candle status", candle3_label.split(" (")[0], candle3_color,
                     sub=candle3_label)
    with c3:
        if pd.isna(rsi_val):
            colored_box("RSI (14)", "n.v.t.", GRAY, "Onvoldoende historie")
        else:
            rsi_color = RED if rsi_val > 70 else (GREEN if rsi_val < 30 else ORANGE)
            rsi_sub = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutraal")
            colored_box("RSI (14)", f"{rsi_val:.1f}", rsi_color, rsi_sub)
    with c4:
        if pd.isna(mfi_val):
            colored_box("Money Flow Index (14d)", "n.v.t.", GRAY, "Onvoldoende historie")
        else:
            mfi_color = RED if mfi_val > 80 else (GREEN if mfi_val < 20 else ORANGE)
            mfi_sub = "Overbought" if mfi_val > 80 else ("Oversold" if mfi_val < 20 else "Neutraal")
            colored_box("Money Flow Index (14d)", f"{mfi_val:.1f}", mfi_color, mfi_sub)

    # ---- RIJ 2: Support/Resistance, MACD, 1-dag money flow ------------
    c5, c6, c7 = st.columns(3)
    with c5:
        sr_text = ""
        if support:
            sr_text += f"Support: {support:.2f}  "
        if resistance:
            sr_text += f"Resistance: {resistance:.2f}"
        colored_box("Support / Resistance (kort termijn)",
                    f"{support:.2f} — {resistance:.2f}" if support and resistance else "n.v.t.",
                    GRAY, sr_text or "Geen duidelijke swingpunten gevonden")
    with c6:
        macd_bullish = last["MACD"] > last["MACD_signal"]
        macd_color = GREEN if macd_bullish else RED
        colored_box("MACD", "Bullish crossover" if macd_bullish else "Bearish crossover",
                     macd_color, f"MACD: {last['MACD']:.3f} | Signaal: {last['MACD_signal']:.3f}")
    with c7:
        if flow_today is None:
            colored_box("1-dag Money Flow (intraday)", "n.v.t.", GRAY,
                         "Geen intraday data beschikbaar (bv. bij weekend/feestdag)")
        else:
            net = flow_today["net_pct"]
            flow_color = GREEN if net > 5 else (RED if net < -5 else ORANGE)
            flow_label = "Positief" if net > 5 else ("Negatief" if net < -5 else "Neutraal")
            colored_box("1-dag Money Flow (intraday)", flow_label, flow_color,
                         f"Netto up-volume: {net:+.1f}% "
                         f"(up: {flow_today['up_vol']:,.0f} / down: {flow_today['down_vol']:,.0f})")

    # ---- RIJ 3: Put/Call ratio & Short interest ------------------------
    st.markdown("### Opties & Short interest")
    c8, c9 = st.columns(2)
    with c8:
        if option_data is None:
            colored_box("Put/Call Ratio (volume)", "n.v.t.", GRAY,
                         "Geen optieketen beschikbaar voor dit ticker")
        else:
            pcr = option_data["pcr_volume"]
            if pd.isna(pcr):
                colored_box("Put/Call Ratio (volume)", "n.v.t.", GRAY, "Geen volume vandaag")
            else:
                pcr_color = GREEN if pcr < 0.7 else (RED if pcr > 1.0 else ORANGE)
                pcr_label = "Bullish" if pcr < 0.7 else ("Bearish" if pcr > 1.0 else "Neutraal")
                colored_box("Put/Call Ratio (volume)", f"{pcr:.2f} — {pcr_label}", pcr_color,
                             f"Expiratie: {option_data['expiry']} | "
                             f"Call vol: {option_data['call_volume']:,.0f} | "
                             f"Put vol: {option_data['put_volume']:,.0f}")
    with c9:
        spf = short_data.get("short_pct_float")
        if spf is None:
            colored_box("Short % of Float", "n.v.t.", GRAY,
                         "Niet beschikbaar voor dit ticker")
        else:
            spf_pct = spf * 100
            short_color = RED if spf_pct > 15 else (ORANGE if spf_pct > 5 else GREEN)
            short_label = "Hoog" if spf_pct > 15 else ("Gemiddeld" if spf_pct > 5 else "Laag")
            colored_box("Short % of Float", f"{spf_pct:.1f}% — {short_label}", short_color,
                         "Bron: beurs-settlementdata, ±2x per maand geüpdatet (niet live)")

    st.caption(
        "Let op: Put/Call ratio is gebaseerd op de dichtstbijzijnde optie-expiratie "
        "(niet de volledige markt) en short-data wordt niet dagelijks bijgewerkt. "
        "Gebruik dit als extra context, niet als enige beslissingsfactor."
    )

    # ---- GRAFIEK -------------------------------------------------------
    st.markdown("### Prijsgrafiek met SMA's en support/resistance")
    plot_df = df.tail(90)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"], name="Prijs",
        increasing_line_color=GREEN, decreasing_line_color=RED,
    ))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA20"], name="SMA20",
                              line=dict(color="#3b82f6", width=1.5)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA50"], name="SMA50",
                              line=dict(color="#a855f7", width=1.5)))
    if support:
        fig.add_hline(y=support, line_dash="dash", line_color=GREEN,
                       annotation_text=f"Support {support:.2f}")
    if resistance:
        fig.add_hline(y=resistance, line_dash="dash", line_color=RED,
                       annotation_text=f"Resistance {resistance:.2f}")
    fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                       margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ---- SAMENVATTEND SIGNAAL ------------------------------------------
    st.markdown("### Samenvattend swingtrade-signaal")
    score = 0
    reasons = []
    if trend_color == GREEN:
        score += 1; reasons.append("Trend bullish")
    elif trend_color == RED:
        score -= 1; reasons.append("Trend bearish")
    if candle3_color == GREEN:
        score += 1; reasons.append("3-daagse candles bullish")
    elif candle3_color == RED:
        score -= 1; reasons.append("3-daagse candles bearish")
    if not pd.isna(rsi_val):
        if rsi_val < 30:
            score += 1; reasons.append("RSI oversold (mogelijk bounce)")
        elif rsi_val > 70:
            score -= 1; reasons.append("RSI overbought (mogelijk pullback)")
    if macd_bullish:
        score += 1; reasons.append("MACD bullish")
    else:
        score -= 1; reasons.append("MACD bearish")
    if flow_today and flow_today["net_pct"] > 5:
        score += 1; reasons.append("Positieve money flow vandaag")
    elif flow_today and flow_today["net_pct"] < -5:
        score -= 1; reasons.append("Negatieve money flow vandaag")
    if option_data and not pd.isna(option_data.get("pcr_volume", np.nan)):
        if option_data["pcr_volume"] < 0.7:
            score += 1; reasons.append("Put/Call ratio bullish")
        elif option_data["pcr_volume"] > 1.0:
            score -= 1; reasons.append("Put/Call ratio bearish")
    if spf is not None and spf * 100 > 15:
        score -= 1; reasons.append("Hoge short interest (risico op volatiliteit)")

    if score >= 2:
        overall_color, overall_label = GREEN, "Bullish"
    elif score <= -2:
        overall_color, overall_label = RED, "Bearish"
    else:
        overall_color, overall_label = ORANGE, "Neutraal / gemengd"

    colored_box("Overall signaal (som van indicatoren)", f"{overall_label}  (score: {score:+d})",
                 overall_color, " · ".join(reasons))

    st.info(
        "Dit dashboard is bedoeld als informatief hulpmiddel voor korte swingtrades "
        "(1-5 dagen) en is **geen financieel advies**. Combineer dit altijd met je "
        "eigen risicomanagement (stop-loss, positiegrootte) en, indien nodig, "
        "professioneel advies."
    )
else:
    st.info("Vul een ticker in en klik op 'Analyseer' om te starten.")
