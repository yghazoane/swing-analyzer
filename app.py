import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from datetime import datetime, timedelta

# إعدادات الصفحة الأساسية
st.set_page_config(page_title="محلل السوينق والموجات الاحترافي", layout="wide", initial_sidebar_state="expanded")

st.title("📈 لوحة تحليل السوينق والأهداف الزمنية (SPY / QQQ)")
st.markdown("أداة احترافية لتحديد هيكل السوق، الموجات السعرية، وتقدير مناطق الاحتمالات السعرية والزمنية لعقود الخيارات.")

# --- القائمة الجانبية (المدخلات والتحكم الحي) ---
st.sidebar.header("⚙️ إعدادات التحليل")
ticker = st.sidebar.selectbox("اختر المؤشر / السهم:", ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META"])
interval = st.sidebar.selectbox("الإطار الزمني (Timeframe):", ["1d", "1h"], index=0)
period = "1y" if interval == "1d" else "3mo"
swing_order = st.sidebar.slider("حساسية اكتشاف القمم والقيعان (Swing Sensitivity):", min_value=3, max_value=15, value=7)
risk_pct = st.sidebar.number_input("نسبة المخاطرة من المحفظة (%):", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
portfolio_size = st.sidebar.number_input("حجم المحفظة ($):", min_value=1000, value=10000, step=1000)

# دالة لحساب أيام التداول القادمة (تخطي عطلة نهاية الأسبوع السبت والأحد)
def add_trading_days(start_date, trading_days):
    current_date = start_date
    added = 0
    while added < trading_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # 0-4 الإثنين إلى الجمعة
            added += 1
    return current_date

# --- دالة التحليل المتكاملة لأي سهم أو مؤشر ---
@st.cache_data(ttl=300)
def analyze_ticker(symbol, period, interval, swing_order):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    
    if df.empty or len(df) < 50:
        return None
    
    # المتوسطات المتحركة وفلتر الاتجاه
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # مؤشر القوة النسبية RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # متوسط المدى الحقيقي ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # اكتشاف القمم والقيعان المحورية آلياً
    df['Swing_High'] = df.iloc[argrelextrema(df['High'].values, np.greater_equal, order=swing_order)[0]]['High']
    df['Swing_Low'] = df.iloc[argrelextrema(df['Low'].values, np.less_equal, order=swing_order)[0]]['Low']
    
    last_highs = df.dropna(subset=['Swing_High'])
    last_lows = df.dropna(subset=['Swing_Low'])
    
    current_price = float(df['Close'].iloc[-1])
    current_atr = float(df['ATR'].iloc[-1])
    ema50_val = float(df['EMA50'].iloc[-1])
    current_rsi = float(df['RSI'].iloc[-1])
    last_date = df.index[-1]
    if hasattr(last_date, 'to_pydatetime'):
        last_date = last_date.to_pydatetime()
    
    is_bullish = current_price > ema50_val
    trend = "صاعد (Bullish) 🟢" if is_bullish else "هابط (Bearish) 🔴"
    option_type = "CALL 🟢 (شراء)" if is_bullish else "PUT 🔴 (بيع)"
    
    if len(last_highs) >= 2 and len(last_lows) >= 2:
        recent_high_price = float(last_highs['Swing_High'].iloc[-1])
        recent_high_date = last_highs.index[-1]
        
        recent_low_price = float(last_lows['Swing_Low'].iloc[-1])
        recent_low_date = last_lows.index[-1]
        
        wave_height = abs(recent_high_price - recent_low_price)
        wave_bars = abs(df.index.get_loc(recent_high_date) - df.index.get_loc(recent_low_date))
        wave_bars = max(wave_bars, 5)
        
        if is_bullish:
            target_1 = recent_low_price + (wave_height * 1.272)
            target_2 = recent_low_price + (wave_height * 1.618)
            stop_loss = recent_low_price - (current_atr * 0.5)
            strike_rec = round(current_price)  # ATM Call
        else:
            target_1 = recent_high_price - (wave_height * 1.272)
            target_2 = recent_high_price - (wave_height * 1.618)
            stop_loss = recent_high_price + (current_atr * 0.5)
            strike_rec = round(current_price)  # ATM Put
            
        est_bars_t1 = int(wave_bars * 1.0)
        est_bars_t2 = int(wave_bars * 1.618)
        
        multiplier = 1 if interval == "1d" else (1/6.5)
        est_days_t1 = max(2, int(est_bars_t1 * multiplier))
        est_days_t2 = max(3, int(est_bars_t2 * multiplier))
        recommended_dte = est_days_t2 + 21
    else:
        target_1 = current_price * 1.02 if is_bullish else current_price * 0.98
        target_2 = current_price * 1.04 if is_bullish else current_price * 0.96
        stop_loss = current_price * 0.98 if is_bullish else current_price * 1.02
        strike_rec = round(current_price)
        est_days_t1, est_days_t2, recommended_dte = 5, 10, 35

    date_t1 = add_trading_days(last_date, est_days_t1).strftime("%Y-%m-%d")
    date_t2 = add_trading_days(last_date, est_days_t2).strftime("%Y-%m-%d")
    expiry_date = (last_date + timedelta(days=recommended_dte)).strftime("%Y-%m-%d")
    
    risk_per_share = abs(current_price - stop_loss)
    rr_ratio = abs(target_1 - current_price) / risk_per_share if risk_per_share > 0 else 0
    
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
        "est_days_t1": est_days_t1,
        "est_days_t2": est_days_t2,
        "date_t1": date_t1,
        "date_t2": date_t2,
        "recommended_dte": recommended_dte,
        "expiry_date": expiry_date,
        "rr_ratio": rr_ratio
    }

# إنشاء تبويبين رئيسيين: التحليل التفصيلي + الماسح الآلي للفرص
tab1, tab2 = st.tabs(["📊 التحليل التفصيلي وبطاقة الصفقة", "🔍 الماسح الآلي للأسهم والمؤشرات (Screener)"])

with tab1:
    res = analyze_ticker(ticker, period, interval, swing_order)
    
    if res is None:
        st.error("تعذر جلب البيانات الكافية لهذا المؤشر حالياً.")
    else:
        df = res["df"]
        
        # --- عرض بطاقات الملخص العلوي ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("السعر الحالي", f"${res['current_price']:.2f}", res["trend"])
        col2.metric("منطقة الهدف الأول (1.272)", f"${res['target_1']:.2f}", f"متوقع: {res['date_t1']} (~{res['est_days_t1']} أيام)")
        col3.metric("منطقة الهدف الثاني (1.618)", f"${res['target_2']:.2f}", f"متوقع: {res['date_t2']} (~{res['est_days_t2']} أيام)")
        col4.metric("نوع العقد المقترح", res["option_type"], f"Strike: ${res['strike_rec']} | DTE: {res['recommended_dte']} يوم")

        # --- بطاقة التوصية التنفيذية التفصيلية ---
        st.subheader(f"🎯 بطاقة السيناريو التنفيذي وعقد الأوبشن المقترح ({ticker})")
        setup_df = pd.DataFrame([{
            "الأصل": ticker,
            "الاتجاه العام": res["trend"],
            "نوع العقد (Option)": res["option_type"],
            "سعر التنفيذ (Strike)": f"${res['strike_rec']}",
            "السعر الحالي": f"${res['current_price']:.2f}",
            "منطقة الهدف 1 (1.272)": f"${res['target_1']:.2f}",
            "التاريخ التقديري T1": res["date_t1"],
            "منطقة الهدف 2 (1.618)": f"${res['target_2']:.2f}",
            "التاريخ التقديري T2": res["date_t2"],
            "وقف الخسارة (ATR)": f"${res['stop_loss']:.2f}",
            "انتهاء الصلاحية المقترح": f"{res['expiry_date']} ({res['recommended_dte']} يوم)"
        }])
        st.dataframe(setup_df, use_container_width=True, hide_index=True)
        st.caption("⚠️ **تنويه احترافي:** مستويات امتداد فيبوناتشي (1.272 و 1.618) والتواريخ التقديرية هي **أهداف حسابية ومناطق احتمالات مبنية على سلوك الموجات السابقة وليست توقعات مضمونة**. الالتزام بوقف الخسارة وإدارة رأس المال هو أساس البقاء في السوق.")

        # --- الرسم البياني التفاعلي (Plotly) ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])

        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="السعر"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='cyan', width=1.2), name="EMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='orange', width=1.5), name="EMA 50 (فلتر الاتجاه)"), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['Swing_High'], mode='markers', marker=dict(color='red', size=8, symbol='triangle-down'), name="قمة موجة (Pivot High)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Swing_Low'], mode='markers', marker=dict(color='lime', size=8, symbol='triangle-up'), name="قاع موجة (Pivot Low)"), row=1, col=1)

        fig.add_hline(y=res['target_1'], line_dash="dash", line_color="green", annotation_text=f"احتمال T1 (1.272): ${res['target_1']:.2f} | {res['date_t1']}", row=1, col=1)
        fig.add_hline(y=res['target_2'], line_dash="dot", line_color="lime", annotation_text=f"احتمال T2 (1.618): ${res['target_2']:.2f} | {res['date_t2']}", row=1, col=1)
        fig.add_hline(y=res['stop_loss'], line_dash="dash", line_color="red", annotation_text=f"وقف الخسارة (ATR): ${res['stop_loss']:.2f}", row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1.5), name="RSI (14)"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

        fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark", title_text=f"التحليل الهيكلي والموجي لـ {ticker}")
        st.plotly_chart(fig, use_container_width=True)

        # --- قسم إدارة المخاطر وحاسبة الصفقة ---
        st.subheader("🛡️ حاسبة إدارة المخاطر وحجم الصفقة (Position Sizing)")
        risk_amount = portfolio_size * (risk_pct / 100)
        risk_per_share = abs(res['current_price'] - res['stop_loss'])
        shares_allowed = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        rr_ratio = res['rr_ratio']

        c1, c2, c3, c4 = st.columns(4)
        c1.info(f"**المبلغ المخاطر به:** ${risk_amount:.2f}")
        c2.warning(f"**وقف الخسارة المقترح:** ${res['stop_loss']:.2f}")
        c3.success(f"**الكمية المسموحة (أسهم):** {shares_allowed} سهم")
        c4.metric("نسبة العائد للمخاطرة (R:R)", f"1 : {rr_ratio:.2f}", "مقبول (> 1:2)" if rr_ratio >= 2 else "ضعيف (< 1:2)")

with tab2:
    st.subheader("🔍 الماسح الآلي للفرص السوينقية (Multi-Ticker Screener)")
    st.markdown("يفحص الماسح قائمة من أبرز المؤشرات والأسهم القيادية لحظياً لتحديد نوع العقد المناسب (Call / Put) والأهداف السعرية والزمنية المحتملة.")
    
    watchlist = st.multiselect(
        "اختر قائمة الأسهم والمؤشرات للفحص:",
        ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META", "AMD", "GOOGL", "NFLX"],
        default=["SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "MSFT", "META"]
    )
    
    if st.button("🚀 بدء الفحص الشامل الآن", type="primary"):
        screener_rows = []
        progress_bar = st.progress(0)
        for idx, sym in enumerate(watchlist):
            data_res = analyze_ticker(sym, period, interval, swing_order)
            if data_res:
                status = "✅ فرصة قوية (R:R >= 2)" if data_res["rr_ratio"] >= 2.0 else "⏳ مراقبة (R:R < 2)"
                screener_rows.append({
                    "الرمز": sym,
                    "السعر الحالي": f"${data_res['current_price']:.2f}",
                    "الاتجاه": data_res["trend"],
                    "نوع العقد": data_res["option_type"],
                    "Strike مقترح": f"${data_res['strike_rec']}",
                    "هدف 1 (1.272)": f"${data_res['target_1']:.2f}",
                    "تاريخ الوصول T1": data_res["date_t1"],
                    "هدف 2 (1.618)": f"${data_res['target_2']:.2f}",
                    "تاريخ الوصول T2": data_res["date_t2"],
                    "وقف الخسارة": f"${data_res['stop_loss']:.2f}",
                    "صلاحية العقد (DTE)": f"{data_res['recommended_dte']} يوم",
                    "RSI": f"{data_res['current_rsi']:.1f}",
                    "C/R (R:R)": f"1 : {data_res['rr_ratio']:.2f}",
                    "التقييم": status
                })
            progress_bar.progress((idx + 1) / len(watchlist))
            
        if screener_rows:
            screener_df = pd.DataFrame(screener_rows)
            st.dataframe(screener_df, use_container_width=True, hide_index=True)
            st.caption("⚠️ تذكير: الأهداف السعرية والتواريخ الموضحة في الجدول تمثل مناطق احتمالية رياضية بناءً على امتدادات فيبوناتشي وسرعة الموجات السابقة، وليست ضماناً لحركة السعر.")
        else:
            st.warning("لم يتم العثور على بيانات كافية للقائمة المحددة.")

# --- قسم التوثيق ومنهجية عمل الأداة ---
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

    ### 4. حساب مناطق الأهداف السعرية ونسب امتداد فيبوناتشي (Fibonacci Extensions)
    * بناءً على قياس طول الموجة السابقة، تطبق الأداة معادلات **امتداد فيبوناتشي** لتقدير مناطق جني الأرباح المحتملة:
      * **منطقة الهدف الأول (T1):** عند امتداد نسبة **1.272** من الموجة.
      * **منطقة الهدف الثاني (T2 / الموجة الثالثة المحتملة):** عند امتداد نسبة **1.618**.
      * *(ملاحظة: هذه المستويات تمثل أهدافاً حسابية محتملة وليست توقعات مضمونة)*.

    ### 5. تقدير الوقت المتوقع والتاريخ الدقيق (Time Projections & DTE)
    * تحسب الأداة عدد الشموع التي استغرقتها الموجة السابقة وتسقطها زمنياً مع استبعاد عطلات نهاية الأسبوع لإعطاء **تاريخ وصول تقديري** للهدفين، مع إضافة **21 يوماً كهامش أمان** لاختيار تاريخ انتهاء عقد الأوبشن (`DTE`).

    ### 6. إدارة المخاطر وحساب متوسط المدى الحقيقي (ATR & Position Sizing)
    * تحديد وقف خسارة ديناميكي باستخدام مؤشر `ATR` وحساب حجم الصفقة ونسبة العائد للمخاطرة (`R:R`).

    ### 7. عزم الزخم (RSI Momentum)
    * دمج مؤشر القوة النسبية `RSI (14)` لقياس تشبع السعر وقوة الزخم.

    ### 8. العرض البصري والماسح الآلي (Plotly & Multi-Ticker Screener)
    * شارت تفاعلي مع بطاقة تنفيذية تحدد نوع العقد (`CALL` أو `PUT`) وسعر التنفيذ (`Strike`)، بالإضافة إلى ماسح آلي يفحص عدة أسهم ومؤشرات دفعة واحدة.
    """)
