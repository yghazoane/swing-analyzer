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
st.set_page_config(
    page_title="محلل السوينق والموجات الاحترافي", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS دعم RTL وخط Cairo + (Balanced & Clean) حزمة التصميم البصري ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
    direction: rtl;
    text-align: right;
}

.stButton>button {
    border-radius: 8px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# نظام سجل تتبع الأداء والصفقات الحية
# ==========================================
LOG_FILE = "trade_history.csv"

def init_trade_log():
    """تهيئة ملف السجل إذا لم يكن موجوداً"""
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=[
            "Timestamp", "Ticker", "Option_Type", "Strike", 
            "Alert_Price", "Max_Price_Reached", "Status", "Notes"
        ])
        df.to_csv(LOG_FILE, index=False)

def log_new_trade(ticker, opt_type, strike, alert_price):
    """إضافة صفقة أو تنبيه جديد للسجل"""
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

# ==========================================
# الواجهة الجانبية والتحكم الأساسي
# ==========================================
st.sidebar.title("🎛️ لوحة التحكم والتحليل")
ticker_symbol = st.sidebar.text_input("اختر الرمز (Ticker)", value="META").upper()
timeframe = st.sidebar.selectbox("الإطار الزمني (Timeframe)", ["1d", "1h", "4h"], index=0)
sensitivity = st.sidebar.slider("حساسية اكتشاف القمم والقيعان", 1, 15, 7)

st.title("📈 محلل السوينق والموجات الاحترافي مع سجل التتبع")
st.markdown("---")

# جلب بيانات السهم الحية عبر yfinance
@st.cache_data(ttl=300)
def load_data(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

data_load_state = st.text("جاري تحميل بيانات السوق الحية...")
df = load_data(ticker_symbol, period="6mo", interval=timeframe)
data_load_state.text("")

if df.empty:
    st.error("لم يتم العثور على بيانات لهذا الرمز. تأكد من صحة الرمز المدخل.")
else:
    # حساب المتوسط المتحرك والمؤشرات الفنية البسيطة للماسح
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    current_price = float(df['Close'].iloc[-1])
    current_rsi = float(df['RSI'].iloc[-1]) if not np.isnan(df['RSI'].iloc[-1]) else 50.0
    ema_50 = float(df['EMA_50'].iloc[-1])
    
    trend_status = "صاعد (Bullish) 🟢" if current_price > ema_50 else "هابط (Bearish) 🔴"
    
    # عرض ملخص التحليل الحالي
    col1, col2, col3 = st.columns(3)
    col1.metric("السعر الحالي", f"${current_price:.2f}")
    col2.metric("الاتجاه العام (EMA 50)", trend_status)
    col3.metric("مؤشر الزخم RSI", f"{current_rsi:.1f}")
    
    st.markdown("---")

# ==========================================
# قسم سجل تتبع الأداء والصفقات الحية (Trade Journal)
# ==========================================
st.subheader("📊 سجل تتبع الأداء والصفقات الحية (Trade Journal)")

init_trade_log()
history_df = pd.read_csv(LOG_FILE)

if not history_df.empty:
    if "Alert_Price" in history_df.columns and "Max_Price_Reached" in history_df.columns:
        history_df["Profit_Percentage_%"] = (
            (history_df["Max_Price_Reached"] - history_df["Alert_Price"]) / history_df["Alert_Price"] * 100
        ).round(2)

    st.write("💡 **تلميح:** يمكنك تحديث خانة (أعلى سعر وصل له Max_Price_Reached) مباشرة من الجدول أدناه لقياس الأداء بدقة (مثل تحديث عقد META إلى 20.20):")
    
    edited_df = st.data_editor(
        history_df, 
        num_rows="dynamic",
        use_container_width=True,
        key="trade_log_editor"
    )
    
    col_save, col_clear = st.columns([1, 1])
    with col_save:
        if st.button("💾 حفظ التحديثات في السجل"):
            edited_df.to_csv(LOG_FILE, index=False)
            st.success("تم تحديث وحفظ سجل الأداء بنجاح!")
            st.rerun()
            
    with col_clear:
        if st.button("🗑️ تفريغ السجل بالكامل"):
            if os.path.exists(LOG_FILE):
                os.remove(LOG_FILE)
                st.warning("تم مسح السجل بنجاح.")
                st.rerun()
else:
    st.info("لا توجد صفقات مسجلة حتى الآن. يمكنك إضافة عقود للمراقبة عبر النموذج أدناه.")

# نموذج إضافة صفقة جديدة يدوياً أو ربطها بالماسح
with st.expander("➕ إضافة عقد جديد للمراقبة يدوياً"):
    with st.form("add_trade_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            t_ticker = st.text_input("رمز السهم", "META")
        with col2:
            t_type = st.selectbox("نوع العقد", ["PUT", "CALL"])
        with col3:
            t_strike = st.number_input("السترايك (Strike)", value=635.0)
        with col4:
            t_price = st.number_input("سعر العقد عند التنبيه", value=17.06, format="%.2f")
        
        submit_btn = st.form_submit_button("إضافة إلى جدول التتبع")
        if submit_btn:
            log_new_trade(t_ticker, t_type, t_strike, t_price)
            st.success(f"تم إضافة عقد {t_ticker} بنجاح إلى جدول التتبع!")
            st.rerun()
