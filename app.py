"""
================================================================================
اسم الأداة: محلل العقود والسوينق الاحترافي (Pro Options & Swing Analyzer)
الإصدار: v2.3.2 (إضافة مؤشر SPX 500 ودعم تتبع الأصول بدقة)
================================================================================
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from scipy.stats import norm
from datetime import datetime, timedelta
import os

# إعدادات الصفحة
st.set_page_config(
    page_title="محلل العقود والسوينق الاحترافي v2.3.2",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تنسيق بصري آمن وخفيف
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, label, button {
        font-family: 'Cairo', sans-serif !important;
    }
    .rtl-box { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True
)

LOG_FILE = "trade_history.csv"


def init_trade_log():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=[
            "Timestamp", "Ticker", "Style", "Option_Type", "Strike", "Expiry",
            "Entry_Premium", "Max_Premium", "Stop_Underlying", "Target1_Underlying",
            "Contracts", "Status", "Notes"
        ])
        df.to_csv(LOG_FILE, index=False)


def log_new_trade(ticker, style, opt_type, strike, expiry, premium, stop_u, target1_u, contracts):
    init_trade_log()
    df = pd.read_csv(LOG_FILE)
    new_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Ticker": ticker,
        "Style": style,
        "Option_Type": opt_type,
        "Strike": strike,
        "Expiry": expiry,
        "Entry_Premium": round(premium, 2),
        "Max_Premium": round(premium, 2),
        "Stop_Underlying": round(stop_u, 2),
        "Target1_Underlying": round(target1_u, 2),
        "Contracts": contracts,
        "Status": "مفتوح للمراقبة ⏳",
        "Notes": "إشارة نظامية"
    }])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)


def add_trading_days(start_date, trading_days):
    current_date = start_date
    added = 0
    while added < trading_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            added += 1
    return current_date


def round_to_strike_step(price):
    """تقريب السترايك لأقرب فئة تداول حقيقية في السوق الأمريكي"""
    if price < 50:
        return round(price * 2) / 2.0
    elif price < 250:
        return float(round(price))
    elif price < 500:
        return float(round(price / 2.5) * 2.5)
    else:
        return float(round(price / 5.0) * 5.0)


def calculate_bs_metrics(S, K, T_days, r, sigma, is_call=True):
    """حساب السعر النظري للعقد واليونانيات الكاملة (Delta, Gamma, Theta, Vega)"""
    T = max(T_days / 365.0, 1e-5)
    sigma = max(sigma, 0.10)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if is_call:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta_annual = (-(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
                        - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta_annual = (-(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
                        + r * K * np.exp(-r * T) * norm.cdf(-d2))

    gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
    vega = (S * norm.pdf(d1) * sqrt_T) / 100.0
    theta_daily = theta_annual / 365.0

    return {
        "premium": max(float(price), 0.05),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta_daily": float(theta_daily),
        "vega": float(vega),
        "iv_pct": float(sigma * 100)
    }


@st.cache_data(ttl=600)
def fetch_price_history(symbol, period, interval):
    """جلب البيانات التاريخية مع آلية احتياطية وتوافق مع رمز مؤشر SPX (^GSPC)"""
    try:
        # مؤشر SPX في ياهو فاينانس يُرمز له بـ ^GSPC
        query_symbol = "^GSPC" if symbol == "SPX" else symbol
        tk = yf.Ticker(query_symbol)
        df = tk.history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            df = yf.download(query_symbol, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is not None and not df.empty:
            cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            df = df[cols].dropna()
        return df, ""
    except Exception as e:
        return pd.DataFrame(), str(e)


@st.cache_data(ttl=600)
def fetch_live_option_hint(symbol, target_dte, is_call, target_strike):
    """جلب أقرب عقد حقيقي من سلسلة الأوبشن الحية (يدعم SPY و SPX عبر ^GSPC)"""
    try:
        query_symbol = "^GSPC" if symbol == "SPX" else symbol
        tk = yf.Ticker(query_symbol)
        expirations = tk.options
        if not expirations:
            return None
        today = datetime.now().date()
        best_exp = min(
            expirations,
            key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d").date() - today).days - target_dte)
        )
        chain = tk.option_chain(best_exp)
        df_opt = chain.calls if is_call else chain.puts
        if df_opt is None or df_opt.empty:
            return None
        idx = (df_opt['strike'] - target_strike).abs().idxmin()
        row = df_opt.loc[idx]
        bid = float(row.get('bid', 0) or 0)
        ask = float(row.get('ask', 0) or 0)
        last_p = float(row.get('lastPrice', 0) or 0)
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else last_p
        spread_pct = ((ask - bid) / ask * 100) if ask > 0 and bid > 0 else 0.0
        return {
            "expiry": best_exp,
            "strike": float(row['strike']),
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_pct": spread_pct,
            "volume": int(row.get('volume', 0) or 0),
            "oi": int(row.get('openInterest', 0) or 0),
            "iv": float(row.get('impliedVolatility', 0) or 0) * 100
        }
    except Exception:
        return None


def analyze_ticker(symbol, interval, swing_order, delta_mode, trading_mode):
    try:
        period_map = {"15m": "1mo", "1h": "3mo", "1d": "1y"}
        period = period_map.get(interval, "1y")
        df, err_msg = fetch_price_history(symbol, period, interval)
        if df is None or df.empty or len(df) < 30:
            return None, err_msg or "البيانات الواردة من المصدر فارغة أو غير مكتملة."

        df = df.copy()

        # المتوسطات المتحركة
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

        # مؤشر RSI (14)
        delta_s = df['Close'].diff()
        gain = (delta_s.where(delta_s > 0, 0)).rolling(window=14).mean()
        loss = (-delta_s.where(delta_s < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = (100 - (100 / (1 + rs))).fillna(50)

        # مؤشر MACD
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # نطاقات بولنجر (Bollinger Bands)
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Mid'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']

        # متوسط المدى الحقيقي ATR (14)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
        df['ATR'] = df['ATR'].bfill().fillna(df['Close'] * 0.015)

        # التقلب التاريخي السنوي (HV)
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        bars_per_year = {"15m": 252 * 26, "1h": 252 * 6.5, "1d": 252}.get(interval, 252)
        est_vol = float(log_returns.rolling(20).std().iloc[-1] * np.sqrt(bars_per_year))
        if np.isnan(est_vol) or est_vol <= 0.05:
            est_vol = 0.25

        # قمم وقيعان السوينق الهيكلية
        high_idx = argrelextrema(df['High'].values, np.greater_equal, order=swing_order)[0]
        low_idx = argrelextrema(df['Low'].values, np.less_equal, order=swing_order)[0]
        df['Swing_High'] = np.nan
        df['Swing_Low'] = np.nan
        df.iloc[high_idx, df.columns.get_loc('Swing_High')] = df['High'].iloc[high_idx]
        df.iloc[low_idx, df.columns.get_loc('Swing_Low')] = df['Low'].iloc[low_idx]

        last_highs = df.dropna(subset=['Swing_High'])
        last_lows = df.dropna(subset=['Swing_Low'])

        current_price = float(df['Close'].iloc[-1])
        current_atr = float(df['ATR'].iloc[-1])
        ema20_val = float(df['EMA20'].iloc[-1])
        ema50_val = float(df['EMA50'].iloc[-1])
        current_rsi = float(df['RSI'].iloc[-1])
        macd_val = float(df['MACD'].iloc[-1])
        macd_sig = float(df['MACD_Signal'].iloc[-1])
        bb_w = float(df['BB_Width'].iloc[-1])
        bb_w_avg = float(df['BB_Width'].rolling(50, min_periods=10).mean().iloc[-1])

        # تحديد الاتجاه الهيكلي والزخم
        is_bullish = current_price > ema50_val
        trend = "صاعد (Bullish) 🟢" if is_bullish else "هابط (Bearish) 🔴"
        option_type = "CALL 🟢 (شراء)" if is_bullish else "PUT 🔴 (بيع)"

        # تقييم جودة الإشارة (Confluence Score من 5)
        score = 0
        if (is_bullish and current_price > ema20_val) or (not is_bullish and current_price < ema20_val):
            score += 1
        if (is_bullish and macd_val > macd_sig) or (not is_bullish and macd_val < macd_sig):
            score += 1
        if (is_bullish and 45 <= current_rsi <= 68) or (not is_bullish and 32 <= current_rsi <= 55):
            score += 1
        is_squeeze = (not np.isnan(bb_w)) and (not np.isnan(bb_w_avg)) and (bb_w < bb_w_avg * 0.85)
        if is_squeeze:
            score += 1

        # حساب الأهداف الهيكلية بفيبوناتشي ووقف الخسارة بـ ATR
        if len(last_highs) >= 1 and len(last_lows) >= 1:
            rec_high = float(last_highs['Swing_High'].iloc[-1])
            rec_low = float(last_lows['Swing_Low'].iloc[-1])
            wave_h = max(abs(rec_high - rec_low), current_atr * 1.5)
            wave_bars = max(abs(df.index.get_loc(last_highs.index[-1]) - df.index.get_loc(last_lows.index[-1])), 5)

            if is_bullish:
                base_low = min(rec_low, current_price - current_atr)
                target_1 = max(current_price + current_atr * 1.5, base_low + wave_h * 1.272)
                target_2 = max(target_1 + current_atr, base_low + wave_h * 1.618)
                stop_loss = min(rec_low - current_atr * 0.35, current_price - current_atr * 1.2)
            else:
                base_high = max(rec_high, current_price + current_atr)
                target_1 = min(current_price - current_atr * 1.5, base_high - wave_h * 1.272)
                target_2 = min(target_1 - current_atr, base_high - wave_h * 1.618)
                stop_loss = max(rec_high + current_atr * 0.35, current_price + current_atr * 1.2)
        else:
            wave_bars = 8
            if is_bullish:
                target_1 = current_price + current_atr * 2.0
                target_2 = current_price + current_atr * 3.2
                stop_loss = current_price - current_atr * 1.0
            else:
                target_1 = current_price - current_atr * 2.0
                target_2 = current_price - current_atr * 3.2
                stop_loss = current_price + current_atr * 1.0

        # تحديد الصلاحية الزمنية (DTE) حسب أسلوب المضاربة
        if "يومي" in trading_mode or interval in ["15m", "1h"]:
            est_days_t1 = 1 if interval == "15m" else 2
            est_days_t2 = 3 if interval == "15m" else 5
            recommended_dte = 7 if interval == "15m" else 14
        else:
            est_days_t1 = max(3, int(wave_bars * 0.8))
            est_days_t2 = max(6, int(wave_bars * 1.3))
            recommended_dte = min(60, max(28, est_days_t2 + 21))

        # اختيار السترايك بناءً على Delta المستهدفة
        if "OTM" in delta_mode:
            raw_strike = current_price + (current_atr * 0.85 if is_bullish else -current_atr * 0.85)
        elif "ITM" in delta_mode:
            raw_strike = current_price - (current_atr * 0.85 if is_bullish else -current_atr * 0.85)
        else:
            raw_strike = current_price

        strike_rec = round_to_strike_step(raw_strike)

        # حساب اليونانيات والسعر النظري للعقد
        greeks = calculate_bs_metrics(
            S=current_price, K=strike_rec, T_days=recommended_dte,
            r=0.045, sigma=est_vol, is_call=is_bullish
        )

        # محاولة جلب بيانات السلسلة الحية للمقارنة
        live_opt = fetch_live_option_hint(symbol, recommended_dte, is_bullish, strike_rec)

        risk_per_share = abs(current_price - stop_loss)
        reward_t1 = abs(target_1 - current_price)
        rr_ratio = reward_t1 / risk_per_share if risk_per_share > 0 else 0
        if rr_ratio >= 2.0:
            score += 1

        last_dt = df.index[-1].to_pydatetime() if hasattr(df.index[-1], 'to_pydatetime') else datetime.now()
        if getattr(last_dt, "tzinfo", None) is not None:
            last_dt = last_dt.replace(tzinfo=None)
        date_t1 = add_trading_days(last_dt, est_days_t1).strftime("%Y-%m-%d")
        date_t2 = add_trading_days(last_dt, est_days_t2).strftime("%Y-%m-%d")
        expiry_str = (last_dt + timedelta(days=recommended_dte)).strftime("%Y-%m-%d")

        return {
            "df": df,
            "current_price": current_price,
            "current_atr": current_atr,
            "current_rsi": current_rsi,
            "is_bullish": is_bullish,
            "trend": trend,
            "option_type": option_type,
            "strike_rec": strike_rec,
            "target_1": target_1,
            "target_2": target_2,
            "stop_loss": stop_loss,
            "rr_ratio": rr_ratio,
            "est_days_t1": est_days_t1,
            "est_days_t2": est_days_t2,
            "date_t1": date_t1,
            "date_t2": date_t2,
            "recommended_dte": recommended_dte,
            "expiry_str": expiry_str,
            "greeks": greeks,
            "live_opt": live_opt,
            "score": score,
            "macd_status": "إيجابي 🟢" if macd_val > macd_sig else "سلبي 🔴",
            "bb_status": "انفجار وشيك ⚡ (Squeeze)" if is_squeeze else "نطاق طبيعي 📊"
        }, ""
    except Exception as e:
        return None, str(e)


# --- واجهة المستخدم ---
st.title("📈 محلل العقود والسوينق الاحترافي (v2.3.2)")
st.markdown("منصة تحليلية مصممة بعقلية مضارب العقود: دمج الهيكل السعري، ATR، فيبوناتشي، اليونانيات، وإدارة المخاطر لكل عقد.")

# القائمة الجانبية
st.sidebar.header("⚙️ إعدادات التداول")
trading_mode = st.sidebar.radio("أسلوب المضاربة:", ["سوينق (Swing | 21-45 DTE)", "مضاربة يومية / قصيرة (Intraday | 7-14 DTE)"])
default_tf_idx = 2 if "سوينق" in trading_mode else 1
interval = st.sidebar.selectbox("الإطار الزمني:", ["15m", "1h", "1d"], index=default_tf_idx)

# إضافة SPX بجانب SPY وبقية الأصول
ticker = st.sidebar.selectbox("الأصل المالي:", ["SPX", "SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META", "AMD",
