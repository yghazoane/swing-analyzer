import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema

# إعدادات الصفحة الأساسية
st.set_page_config(page_title="محلل السوينق والموجات", layout="wide", initial_sidebar_state="expanded")

st.title("📈 لوحة تحليل السوينق والأهداف الزمنية (SPY / QQQ)")
st.markdown("أداة احترافية لتحديد هيكل السوق، الموجات السعرية، وتقدير المدة الزمنية للوصول للأهداف.")

# --- القائمة الجانبية (المدخلات) ---
st.sidebar.header("⚙️ إعدادات التحليل")
ticker = st.sidebar.selectbox("اختر المؤشر / السهم:", ["SPY", "QQQ", "IWM", "NVDA", "AAPL"])
interval = st.sidebar.selectbox("الإطار الزمني (Timeframe):", ["1d", "1h"], index=0)
period = "1y" if interval == "1d" else "3mo"
swing_order = st.sidebar.slider("حساسية اكتشاف القمم والقيعان (Swing Sensitivity):", min_value=3, max_value=15, value=7)
risk_pct = st.sidebar.number_input("نسبة المخاطرة من المحفظة (%):", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
portfolio_size = st.sidebar.number_input("حجم المحفظة ($):", min_value=1000, value=10000, step=1000)

# --- دالة جلب البيانات وحساب المؤشرات ---
@st.cache_data(ttl=300)
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    
    # حساب المتوسطات المتحركة
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # حساب مؤشر القوة النسبية RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # حساب متوسط المدى الحقيقي ATR (لتحديد وقف الخسارة)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    return df

df = load_data(ticker, period, interval)

# --- اكتشاف القمم والقيعان المحورية (Swing Highs / Lows) ---
df['Swing_High'] = df.iloc[argrelextrema(df['High'].values, np.greater_equal, order=swing_order)[0]]['High']
df['Swing_Low'] = df.iloc[argrelextrema(df['Low'].values, np.less_equal, order=swing_order)[0]]['Low']

last_highs = df.dropna(subset=['Swing_High'])
last_lows = df.dropna(subset=['Swing_Low'])

current_price = df['Close'].iloc[-1]
current_atr = df['ATR'].iloc[-1]
ema50_val = df['EMA50'].iloc[-1]

trend = "صاعد (Bullish) 🟢" if current_price > ema50_val else "هابط (Bearish) 🔴"

# --- حساب الأهداف السعرية والزمنية (Wave & Fibonacci Projections) ---
if len(last_highs) >= 2 and len(last_lows) >= 2:
    recent_high_price = last_highs['Swing_High'].iloc[-1]
    recent_high_date = last_highs.index[-1]
    
    recent_low_price = last_lows['Swing_Low'].iloc[-1]
    recent_low_date = last_lows.index[-1]
    
    wave_height = abs(recent_high_price - recent_low_price)
    wave_bars = abs(df.index.get_loc(recent_high_date) - df.index.get_loc(recent_low_date))
    wave_bars = max(wave_bars, 5)
    
    if current_price > ema50_val:
        target_1 = recent_low_price + (wave_height * 1.272)
        target_2 = recent_low_price + (wave_height * 1.618)
        stop_loss = recent_low_price - (current_atr * 0.5)
    else:
        target_1 = recent_high_price - (wave_height * 1.272)
        target_2 = recent_high_price - (wave_height * 1.618)
        stop_loss = recent_high_price + (current_atr * 0.5)
        
    est_bars_t1 = int(wave_bars * 1.0)
    est_bars_t2 = int(wave_bars * 1.618)
    
    multiplier = 1 if interval == "1d" else (1/6.5)
    est_days_t1 = max(2, int(est_bars_t1 * multiplier))
    est_days_t2 = max(3, int(est_bars_t2 * multiplier))
    
    recommended_dte = est_days_t2 + 21
else:
    target_1, target_2, stop_loss = current_price * 1.02, current_price * 1.04, current_price * 0.98
    est_days_t1, est_days_t2, recommended_dte = 5, 10, 35

# --- عرض بطاقات الملخص العلوي ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("السعر الحالي", f"${current_price:.2f}", trend)
col2.metric("الهدف الأول (T1)", f"${target_1:.2f}", f"خلال ~{est_days_t1} أيام تداول")
col3.metric("الهدف الثاني (T2 - الموجة 3)", f"${target_2:.2f}", f"خلال ~{est_days_t2} أيام تداول")
col4.metric("انتهاء عقد الأوبشن المقترح (DTE)", f"{recommended_dte} يوم", "للحماية من تآكل Theta")

# --- رسم الشارت التفاعلي ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])

fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="السعر"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='cyan', width=1.2), name="EMA 20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='orange', width=1.5), name="EMA 50 (فلتر الاتجاه)"), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['Swing_High'], mode='markers', marker=dict(color='red', size=8, symbol='triangle-down'), name="قمة موجة (Pivot High)"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Swing_Low'], mode='markers', marker=dict(color='lime', size=8, symbol='triangle-up'), name="قاع موجة (Pivot Low)"), row=1, col=1)

fig.add_hline(y=target_1, line_dash="dash", line_color="green", annotation_text=f"الهدف 1: ${target_1:.2f}", row=1, col=1)
fig.add_hline(y=target_2, line_dash="dot", line_color="lime", annotation_text=f"الهدف 2: ${target_2:.2f}", row=1, col=1)
fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text=f"وقف الخسارة: ${stop_loss:.2f}", row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1.5), name="RSI (14)"), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark", title_text=f"التحليل الهيكلي والموجي لـ {ticker}")
st.plotly_chart(fig, use_container_width=True)

# --- قسم إدارة المخاطر وحاسبة الصفقة ---
st.subheader("🛡️ حاسبة إدارة المخاطر وحجم الصفقة (Position Sizing)")
risk_amount = portfolio_size * (risk_pct / 100)
risk_per_share = abs(current_price - stop_loss)
shares_allowed = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
rr_ratio = abs(target_1 - current_price) / risk_per_share if risk_per_share > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.info(f"**المبلغ المخاطر به:** ${risk_amount:.2f}")
c2.warning(f"**وقف الخسارة المقترح:** ${stop_loss:.2f}")
c3.success(f"**الكمية المسموحة (أسهم):** {shares_allowed} سهم")
c4.metric("نسبة العائد للمخاطرة (R:R)", f"1 : {rr_ratio:.2f}", "مقبول (> 1:2)" if rr_ratio >= 2 else "ضعيف (< 1:2)")
