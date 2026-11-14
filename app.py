"""
================================================================================
اسم الأداة: محلل السوينق والموجات الاحترافي مع مؤشرات MACD والبولنجر (Swing & Wave Analyzer)
الإصدار: v2.2.2
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
st.set_page_config(page_title="محلل السوينق والموجات الاحترافي v2.2", layout="wide", initial_sidebar_state="expanded")

# --- حزمة التصميم البصري: الخيار الثالث (متوازن ونظيف Balanced & Clean) + دعم RTL وخط Cairo ---
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
    border-radius:
