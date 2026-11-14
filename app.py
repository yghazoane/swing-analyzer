"""
================================================================================
اسم الأداة: محلل السوينق والموجات الاحترافي (Swing & Wave Analyzer)
الإصدار: v2.2.8
تاريخ التحديث: 2026-11-14
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

# إعدادات الصفحة الأساسية
st.set_page_config(page_title="محلل السوينق والموجات الاحترافي v2.2.8", layout="wide", initial_sidebar_state="expanded")

LOG_FILE = "trade_history.csv"

def init_trade_log():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["Timestamp", "Ticker", "Option_Type", "Strike", "Alert_Price", "Max_Price_Reached", "Status", "Notes"])
        df.to_csv(LOG_FILE, index=False)

def log_new_trade(ticker, opt_type, strike, alert_price):
    init_trade_log()
    df = pd.read_csv(LOG_FILE)
    new_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Ticker": ticker, "Option_Type": opt_type, "Strike": strike,
        "Alert_Price": alert_price, "Max_Price_Reached": alert_price,
        "Status": "مفتوح للمراقبة ⏳", "Notes": "تنبيه من الماسح"
    }])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

st.title("📈 محلل السوينق والموجات الاحترافي (v2.2.8)")
st.markdown("منصة تحليلية متكاملة لهيكل السوق، الموجات، مؤشرات الزخم، هندسة اليونانيات، وسجل تتبع الأداء.")

# --- القائمة الجانبية ---
st.sidebar.header("إعدادات التحليل والتحكم")
ticker = st.sidebar.selectbox("اختر المؤشر / السهم:", ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META"])
interval = st.sidebar.selectbox("الإطار الزمني:", ["1d", "1h"], index=0)
period = "1y" if interval == "1d" else "3mo"
swing_order = st.sidebar.slider("حساسية اكتشاف القمم والقيعان:", min_value=3, max_value=15, value=7)

st.sidebar.markdown("---")
st.sidebar.header("إعدادات العقود")
trade_style = st.sidebar.selectbox("نمط اختيار العقد:", ["متوازن (ATM)", "هجومي (OTM)", "تحوطي (ITM)"], index=0)
risk_pct = st.sidebar.number_input("نسبة المخاطرة (%):", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
portfolio_size = st.sidebar.number_input("حجم المحفظة ($):", min_value=1000, value=10000, step=1000)

def add_trading_days(start_date, trading_days):
    current_date = start_date
    added = 0
    while added < trading_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            added += 1
    return current_date

def calculate_bs_greeks(S, K, T_days, r, sigma, is_call=True):
    T = max(T_days / 365.0, 1e-5)
    sigma = max(sigma, 0.10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta_annual = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)))
    return {"delta": delta, "gamma": gamma, "theta_daily": theta_annual / 365.0, "iv_pct": sigma * 100}

@st.cache_data(ttl=300)
def analyze_ticker(symbol, period, interval, swing_order, style_choice):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        if df.empty or len(df) < 50:
            return None
        
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        delta_s = df['Close'].diff()
        gain = (delta_s.where(delta_s > 0, 0)).rolling(window=14).mean()
        loss = (-delta_s.where(delta_s < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
        
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        est_volatility = float(log_returns.rolling(20).std().iloc[-1] * np.sqrt(252 if interval == "1d" else 1638))
        if np.isnan(est_volatility) or est_volatility <= 0:
            est_volatility = 0.25
            
        df['Swing_High'] = df.iloc[argrelextrema(df['High'].values, np.greater_equal, order=swing_order)[0]]['High']
        df['Swing_Low'] = df.iloc[argrelextrema(df['Low'].values, np.less_equal, order=swing_order)[0]]['Low']
        
        current_price = float(df['Close'].iloc[-1])
        current_atr = float(df['ATR'].iloc[-1])
        ema50_val = float(df['EMA50'].iloc[-1])
        current_rsi = float(df['RSI'].iloc[-1])
        
        is_bullish = current_price > ema50_val
        trend = "صاعد (Bullish) 🟢" if is_bullish else "هابط (Bearish) 🔴"
        option_type = "CALL 🟢 (شراء)" if is_bullish else "PUT 🔴 (بيع)"
        
        target_1 = current_price * (1.02 if is_bullish else 0.98)
        target_2 = current_price * (1.04 if is_bullish else 0.96)
        stop_loss = current_price * (0.98 if is_bullish else 1.02)
        
        strike_rec = round(current_price)
        greeks = calculate_bs_greeks(S=current_price, K=strike_rec, T_days=35, r=0.045, sigma=est_volatility, is_call=is_bullish)
        
        risk_per_share = abs(current_price - stop_loss)
        rr_ratio = abs(target_1 - current_price) / risk_per_share if risk_per_share > 0 else 0
        
        return {
            "df": df, "current_price": current_price, "current_rsi": current_rsi,
            "trend": trend, "option_type": option_type, "strike_rec": strike_rec,
            "target_1": target_1, "target_2": target_2, "stop_loss": stop_loss,
            "rr_ratio": rr_ratio, "greeks": greeks, "is_bullish": is_bullish
        }
    except Exception as e:
        return None

tab1, tab2, tab3 = st.tabs(["📊 التحليل التفصيلي", "🔍 الماسح الآلي", "📈 سجل الأداء"])

with tab1:
    res = analyze_ticker(ticker, period, interval, swing_order, trade_style)
    if res is None:
        st.warning("⚠️ جاري تحميل البيانات أو تعذر جلبها لهذا الرمز حالياً.")
    else:
        df, greeks = res["df"], res["greeks"]
        st.success(f"الأصل الحالي: {ticker} | الاتجاه: {res['trend']}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("السعر الحالي", f"${res['current_price']:.2f}")
        c2.metric("الهدف الأول", f"${res['target_1']:.2f}")
        c3.metric("وقف الخسارة", f"${res['stop_loss']:.2f}")
        c4.metric("العائد للمخاطرة", f"1 : {res['rr_ratio']:.2f}")

        if st.button("📌 إضافة عقد إلى سجل تتبع الأداء"):
            log_new_trade(ticker, "CALL" if res["is_bullish"] else "PUT", res["strike_rec"], res["current_price"])
            st.success("تمت الإضافة بنجاح!")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="السعر"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI"), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="MACD Hist"), row=3, col=1)
        fig.update_layout(height=750, xaxis_rangeslider_visible=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("الماسح الآلي للفرص")
    watchlist = st.multiselect("اختر الأسهم للفحص:", ["SPY", "QQQ", "NVDA", "AAPL", "TSLA"], default=["SPY", "QQQ"])
    if st.button("🚀 بدء الفحص"):
        rows = []
        for sym in watchlist:
            d = analyze_ticker(sym, period, interval, swing_order, trade_style)
            if d:
                rows.append({"الرمز": sym, "السعر": f"${d['current_price']:.2f}", "الاتجاه": d['trend'], "العقد": d['option_type']})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab3:
    st.subheader("سجل تتبع الأداء")
    init_trade_log()
    history_df = pd.read_csv(LOG_FILE)
    if not history_df.empty:
        edited_df = st.data_editor(history_df, use_container_width=True)
        if st.button("💾 حفظ السجل"):
            edited_df.to_csv(LOG_FILE, index=False)
            st.success("تم الحفظ!")
            st.rerun()
    else:
        st.info("لا توجد صفقات مسجلة حتى الآن.")
