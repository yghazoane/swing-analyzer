"""
================================================================================
اسم الأداة: محلل السوينق والموجات الاحترافي مع مؤشرات MACD والبولنجر (Swing & Wave Analyzer)
الإصدار: v2.2.5
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
st.set_page_config(page_title="محلل السوينق والموجات الاحترافي v2.2.5", layout="wide", initial_sidebar_state="expanded")

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
</style>
""", unsafe_allow_html=True)

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

st.markdown("""
<div class="rtl-box">
    <h1 style="margin-bottom: 4px; color: #F8FAFC; font-weight: 800;">محلل السوينق والموجات الاحترافي (v2.2.5)</h1>
    <p style="color: #94A3B8; font-size: 1.05rem; margin-bottom: 18px;">منصة تحليلية متكاملة لهيكل السوق، الموجات، مؤشرات الزخم والتقلب، هندسة اليونانيات، وسجل تتبع الأداء.</p>
</div>
""", unsafe_allow_html=True)

# --- القائمة الجانبية ---
st.sidebar.markdown('<div class="rtl-box"><h3>إعدادات التحليل والتحكم</h3></div>', unsafe_allow_html=True)
ticker = st.sidebar.selectbox("اختر المؤشر / السهم:", ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META"])
interval = st.sidebar.
