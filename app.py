"""
================================================================================
اسم الأداة: محلل السوينق والموجات الاحترافي مع مؤشرات MACD والبولنجر (Swing & Wave Analyzer)
الإصدار: v2.2.4
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
st.set_page_config(page_title="محلل السوينق والموجات الاحترافي v2.2.4", layout="wide", initial_sidebar_state="expanded")

# --- حزمة التصميم البصري ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, h1, h2, h3, h4, label, button {
    font-family: 'Cairo', sans-serif !important;
}
.stApp {
    background-color: #131722;
    color: #E2E8F0;
}
[data-testid="stSidebar"] {
    background-color: #1A1F2E;
    border-left: 1px solid #2A3142;
}
.rtl-box {
    direction: rtl;
    text-align: right;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1C2230, #161B26);
    border: 1px solid #2D3548;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    direction: rtl;
    text-align: right;
}
[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    justify-content: flex-end;
}
[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    direction: ltr;
    text-align: right;
}
.live-banner {
    background: linear-gradient(90deg, #1E2536 0%, #252E42 100%);
    border: 1px solid #334155;
    border-right: 5px solid #38BDF8;
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    direction: rtl;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
.banner-item {
    font-size: 0.98rem;
    font-weight: 600;
    color: #E2E8F0;
    margin: 4px 10px;
}
.badge-bull {
    background-color: rgba(16, 185, 129, 0.18);
    color: #34D399;
    border: 1px solid #10B981;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
}
.badge-bear {
    background-color: rgba(239, 68, 68, 0.18);
    color: #F87171;
    border: 1px solid #EF4444;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
}
.greeks-explain-box {
    background: linear-gradient(145deg, #192030, #141926);
    border: 1px solid #2D3748;
    border-right: 4px solid #A78BFA;
    border-radius: 10px;
    padding: 18px 22px;
    margin-top: 15px;
    margin-bottom: 20px;
    direction: rtl;
    text-align: right;
    line-height: 1.75;
}
</style>
""", unsafe_allow_html=True)

# --- نظام سجل تتبع الأداء ---
LOG_FILE = "trade_history.csv"

def init_trade_log():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=[
            "Timestamp", "Ticker", "Option_Type", "Strike", 
            "Alert_Price", "Max_Price_Reached", "Status", "Notes"
        ])
        df.to_csv(LOG_FILE, index=False)

def log_new_trade(ticker, opt_type, strike, alert_price):
    init_trade_log()
    df = pd.read_csv(LOG_FILE)
    new_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Ticker": ticker,
        "Option_Type": opt_type,
        "Strike": strike,
        "Alert_Price": alert_price,
        "Max_Price_Reached": alert_price,
        "Status": "مفتوح للمراقبة ⏳",
        "Notes": "تنبيه من الماسح أو إدخال يدوي"
    }])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

# --- ترويسة الصفحة ---
st.markdown("""
<div class="rtl-box">
    <h1 style="margin-bottom: 4px; color: #F8FAFC; font-weight: 800;">محلل السوينق والموجات الاحترافي (v2.2.4)</h1>
    <p style="color: #94A3B8; font-size: 1.05rem; margin-bottom: 18px;">منصة تحليلية متكاملة لهيكل السوق، الموجات، مؤشرات الزخم والتقلب (MACD / Bollinger)، هندسة اليونانيات، وسجل تتبع الأداء.</p>
</div>
""", unsafe_allow_html=True)

# --- القائمة الجانبية ---
st.sidebar.markdown('<div class="rtl-box"><h3>إعدادات التحليل والتحكم</h3></div>', unsafe_allow_html=True)
ticker = st.sidebar.selectbox("اختر المؤشر / السهم:", ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META"])
interval = st.sidebar.selectbox("الإطار الزمني (Timeframe):", ["1d", "1h"], index=0)
period = "1y" if interval == "1d" else "3mo"
swing_order = st.sidebar.slider("حساسية اكتشاف القمم والقيعان:", min_value=3, max_value=15, value=7)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="rtl-box"><h4>إعدادات العقود واليونانيات</h4></div>', unsafe_allow_html=True)
trade_style = st.sidebar.selectbox(
    "نمط اختيار العقد (Delta Target):",
    [
        "متوازن (ATM | Delta ~0.50)",
        "هجومي عالي الرافعة (OTM | Delta ~0.35)",
        "تحوطي عالي الأمان (ITM | Delta ~0.65)"
    ],
    index=0
)
risk_pct = st.sidebar.number_input("نسبة المخاطرة من المحفظة (%):", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
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
    
    if is_call:
        delta = norm.cdf(d1)
        theta_annual = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        delta = norm.cdf(d1) - 1.0
        theta_annual = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta_daily = theta_annual / 365.0
    return {"delta": delta, "gamma": gamma, "theta_daily": theta_daily, "iv_pct": sigma * 100}

@st.cache_data(ttl=300)
def analyze_ticker(symbol, period, interval, swing_order, style_choice):
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
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    annual_factor = 252 if interval == "1d" else (252 * 6.5)
    est_volatility = float(log_returns.rolling(20).std().iloc[-1] * np.sqrt(annual_factor))
    if np.isnan(est_volatility) or est_volatility <= 0:
        est_volatility = 0.25
        
    df['Swing_High'] = df.iloc[argrelextrema(df['High'].values, np.greater_equal, order=swing_order)[0]]['High']
    df['Swing_Low'] = df.iloc[argrelextrema(df['Low'].values, np.less_equal, order=swing_order)[0]]['Low']
    
    last_highs = df.dropna(subset=['Swing_High'])
    last_lows = df.dropna(subset=['Swing_Low'])
    
    current_price = float(df['Close'].iloc[-1])
    current_atr = float(df['ATR'].iloc[-1])
    ema50_val = float(df['EMA50'].iloc[-1])
    current_rsi = float(df['RSI'].iloc[-1])
    macd_val = float(df['MACD'].iloc[-1])
    macd_sig = float(df['MACD_Signal'].iloc[-1])
    bb_width_val = float(df['BB_Width'].iloc[-1])
    
    last_date = df.index[-1]
    if hasattr(last_date, 'to_pydatetime'):
        last_date = last_date.to_pydatetime()
    
    is_bullish = current_
