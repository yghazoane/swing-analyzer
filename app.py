import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema

# إعدادات الصفحة الأساسية
st.set_page_config(page_title="محلل السوينق والموجات الاحترافي", layout="wide", initial_sidebar_state="expanded")

st.title("📈 لوحة تحليل السوينق والأهداف الزمنية (SPY / QQQ)")
st.markdown("أداة احترافية لتحديد هيكل السوق، الموجات السعرية، وتقدير المدة الزمنية للوصول للأهداف.")

# --- القائمة الجانبية (المدخلات والتحكم الحي) ---
st.sidebar.header("⚙️ إعدادات التحليل")
ticker = st.sidebar.selectbox("اختر المؤشر / السهم:", ["SPY", "QQQ", "IWM", "NVDA", "AAPL"])
interval = st.sidebar.selectbox("الإطار الزمني (Timeframe):", ["1d", "1h"], index=0)
period = "1y" if interval == "1d" else "3mo"
swing_order = st.sidebar.slider("حساسية اكتشاف القمم والقيعان (Swing Sensitivity):", min_value=3, max_value=15, value=7)
risk_pct = st.sidebar.number_input("نسبة المخاطرة من المحفظة (%):", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
portfolio_size = st.sidebar.number_input("حجم المحفظة ($):", min_value=1000, value=10000, step=1000)

# --- 1. جلب البيانات والتحقق الفوري ---
@st.cache_data(ttl=300)
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    
    # 2. المتوسطات المتحركة وفلتر الاتجاه (EMA Filters)
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # 7. عزم الزخم (RSI Momentum)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 6. متوسط المدى الحقيقي ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    return df

df = load_data(ticker, period, interval)

# --- 3. اكتشاف القمم والقيعان المحورية آلياً (Swing Pivots & Wave Structure) ---
df['Swing_High'] = df.iloc[argrelextrema(df['High'].values, np.greater_equal, order=swing_order)[0]]['High']
df['Swing_Low'] = df.iloc[argrelextrema(df['Low'].values, np.less_equal, order=swing_order)[0]]['Low']

last_highs = df.dropna(subset=['Swing_High'])
last_lows = df.dropna(subset=['Swing_Low'])

current_price = df['Close'].iloc[-1]
current_atr = df['ATR'].iloc[-1]
ema50_val = df['EMA50'].iloc[-1]
current_rsi = df['RSI'].iloc[-1]

trend = "صاعد (Bullish) 🟢" if current_price > ema50_val else "هابط (Bearish) 🔴"

# --- 4 و 5. حساب الأهداف السعرية والزمنية (Fibonacci Extensions & Time Projections) ---
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
col2.metric("الهدف الأول (T1 - 1.272)", f"${target_1:.2f}", f"خلال ~{est_days_t1} أيام تداول")
col3.metric("الهدف الثاني (T2 - 1.618)", f"${target_2:.2f}", f"خلال ~{est_days_t2} أيام تداول")
col4.metric("انتهاء عقد الأوبشن المقترح (DTE)", f"{recommended_dte} يوم", "للحماية من تآكل Theta")

# --- 8. العرض البصري التفاعلي وتجربة المستخدم (Plotly) ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])

fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="السعر"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='cyan', width=1.2), name="EMA 20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='orange', width=1.5), name="EMA 50 (فلتر الاتجاه)"), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['Swing_High'], mode='markers', marker=dict(color='red', size=8, symbol='triangle-down'), name="قمة موجة (Pivot High)"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Swing_Low'], mode='markers', marker=dict(color='lime', size=8, symbol='triangle-up'), name="قاع موجة (Pivot Low)"), row=1, col=1)

fig.add_hline(y=target_1, line_dash="dash", line_color="green", annotation_text=f"الهدف 1 (1.272): ${target_1:.2f}", row=1, col=1)
fig.add_hline(y=target_2, line_dash="dot", line_color="lime", annotation_text=f"الهدف 2 (1.618): ${target_2:.2f}", row=1, col=1)
fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text=f"وقف الخسارة (ATR): ${stop_loss:.2f}", row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1.5), name="RSI (14)"), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark", title_text=f"التحليل الهيكلي والموجي لـ {ticker}")
st.plotly_chart(fig, use_container_width=True)

# --- 6. قسم إدارة المخاطر وحاسبة الصفقة ---
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

# --- قسم التوثيق ومنهجية عمل الأداة (النقاط الـ 8 مدمجة في الواجهة) ---
with st.expander("📖 منهجية التحليل والمعايير الفنية المعتمدة في الأداة (8 ركائز أساسية)"):
    st.markdown("""
    تعتمد هذه الأداة على منظومة متكاملة من التحليل الفني، حساب هيكل السوق (Market Structure)، ودمج نسب فيبوناتشي لفحص وتحديد مستويات صفقات السوينق:

    ### 1. جلب البيانات والتحقّق الفوري (Live Data & Fetching)
    * تعتمد الأداة على جلب بيانات الأسعار التاريخية والحية (الشموع اليابانية لـ Open, High, Low, Close) عبر واجهة البيانات المالية (`yfinance`).
    * تتيح لك الأداة اختيار الإطار الزمني المناسب (سواء اليومي `1d` أو الساعي `1h`) لتحديد ما إذا كنت تريد سوينق قصير أو متوسط المدى.

    ### 2. المتوسطات المتحركة وفلتر الاتجاه (EMA Filters)
    * تحسب الأداة **المتوسط المتحرك الأسي لـ 20 يوماً (EMA 20)** و **50 يوماً (EMA 50)**.
    * **فلتر الاتجاه:** تعتمد الأداة على مقارنة السعر الحالي بـ `EMA 50` لتحديد اتجاه السوق العام (صاعد 🟢 أو هابط 🔴)، وهو المعيار الأساسي لعدم السباحة عكس تيار السوق (*Trend is your friend*).

    ### 3. اكتشاف القمم والقيعان المحورية آلياً (Swing Pivots & Wave Structure)
    * بدلاً من الرسم اليدوي، تستخدم الأداة خوارزميات رياضية متقدمة (`argrelextrema` من مكتبة `SciPy`) لرصد **القمم والقيعان المحورية (Swing Highs & Lows)** في التاريخ السعري.
    * هذا الرصد الآلي يمثل "هيكل السوق" (Market Structure) ويحدد بداية ونهاية آخر موجة سعرية رئيسية صاعدة أو هابطة.

    ### 4. حساب الأهداف السعرية ونسب امتداد فيبوناتشي (Fibonacci Extensions)
    * بناءً على قياس طول الموجة السابقة (المسافة بين آخر قاع وأحدث قمة أو العكس)، تطبق الأداة معادلات **امتداد فيبوناتشي (Fibonacci Extensions)** لتوقع مستويات الأهداف القادمة بدقة:
      * **الهدف الأول (T1):** عند امتداد نسبة **1.272** من الموجة.
      * **الهدف الثاني (T2 / الموجة الثالثة المحتملة):** عند امتداد نسبة **1.618**، والتي تمثل غالباً ذروة الموجة الدافعة القوية.

    ### 5. تقدير الوقت المتوقع والزمن (Time Projections & DTE)
    * **سرعة الموجة:** تحسب الأداة عدد الشموع (Bars) التي استغرقتها الموجة السابقة لتتحرك من البداية للنهاية.
    * **إسقاط الزمن:** تفترض الأداة تكرار النسق الزمني أو مضاعفاته للوصول للأهداف القادمة.
    * **اختيار عقد الأوبشن (DTE):** بناءً على الأيام المتوقعة للوصول للهدف الثاني، تضيف الأداة تلقائياً **هامش أمان بـ 21 يوماً** لتقترح عليك أقل تاريخ صلاحية (DTE) لعقد الخيارات يجب شراؤه لحمايتك من خطر تآكل الوقت (*Theta Decay*).

    ### 6. إدارة المخاطر وحساب متوسط المدى الحقيقي (ATR & Position Sizing)
    * **وقف الخسارة الديناميكي:** تعتمد الأداة على مؤشر **متوسط المدى الحقيقي (ATR)** لقياس تذبذب السوق، ووضعه تحت القاع الأخير بمسافة آمنة لتجنب ضرب الستوب بصورة عشوائية.
    * **حسبة المحفظة:** تحسب الأداة حجم العقود أو الأسهم المسموحة بناءً على نسبة المخاطرة المئوية المحددة من إجمالي محفظتك (مثل 1% أو 2%)، وتقيس **نسبة العائد إلى المخاطرة (R:R)** لضمان أن الصفقة تحقق معيار المحترفين (أعلى من 1:2).

    ### 7. عزم الزخم (RSI Momentum)
    * تحسب الأداة مؤشر القوة النسبية **RSI (14 فترة)** لعرضه في الشارت السفلي، مما يساعدك على رؤية ما إذا كان السعر قرب مستويات التشبع أو أن الزخم يدعم استمرار الموجة الحالية.

    ### 8. العرض البصري التفاعلي وتجربة المستخدم (Plotly & Dashboard Visualization)
    * **رسوم تفاعلية متقدمة:** تعتمد الأداة على مكتبة `Plotly` لبناء شارت شموع يابانية تفاعلي بالكامل (Dark Theme)، مع رسم مستويات الدعم والهدف ووقف الخسارة كخطوط أفقية واضحة.
    * **التحكم الحي (Sidebar Inputs):** تتيح الأداة للمستخدم تعديل المعلمات لحظياً عبر الشريط الجانبي (مثل حساسية اكتشاف القمم `Swing Sensitivity`، ونسبة المخاطرة، وحجم المحفظة)، مما يعيد حساب الأهداف وإدارة المخاطر فوراً ودون الحاجة لإعادة تحميل الصفحة.
    """)
