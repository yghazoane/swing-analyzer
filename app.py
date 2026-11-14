import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import os

# إعدادات الصفحة الأساسية
st.set_page_config(page_title="محلل السوينق والموجات الاحترافي", layout="wide", initial_sidebar_state="expanded")

# --- حزمة التصميم البصري: الخيار الثالث (متوازن ونظيف Balanced & Clean) + دعم RTL وخط Cairo ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, h1, h2, h3, h4, label, button {
    font-family: 'Cairo', sans-serif !important;
}

/* الخلفية الرئيسية المتوازنة والنظيفة */
.stApp {
    background-color: #131722;
    color: #E2E8F0;
}

/* القائمة الجانبية */
[data-testid="stSidebar"] {
    background-color: #1A1F2E;
    border-left: 1px solid #2A3142;
}

/* محاذاة النصوص العربية لليمين مع الحفاظ على اتجاه الأرقام والشارت */
.rtl-box {
    direction: rtl;
    text-align: right;
}

/* تصميم البطاقات الزجاجية النظيفة للمؤشرات */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1C2230, #161B26);
    border: 1px solid #2D3548;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
    direction: rtl;
    text-align: right;
}

[data-testid="stMetric"]:hover {
    border-color: #4A90E2;
    transform: translateY(-2px);
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

/* شريط الحالة الحي العلوي */
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

/* تحسين التبويبات */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background-color: #1A1F2E;
    padding: 6px;
    border-radius: 10px;
    border:
