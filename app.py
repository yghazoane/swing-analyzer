"""
================================================================================
اسم الأداة: محلل السوينق والموجات الاحترافي مع مؤشرات MACD والبولنجر (Swing & Wave Analyzer)
الإصدار: v2.2.0
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

/* بطاقة شرح اليونانيات والمؤشرات الديناميكية */
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

/* تحسين التبويبات */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background-color: #1A1F2E;
    padding: 6px;
    border-radius: 10px;
    border: 1px solid #2A3142;
    direction: rtl;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    color: #94A3B8;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background-color: #2A344A !important;
    color: #38BDF8 !important;
}

/* تحسين الجداول */
[data-testid="stDataFrame"] {
    border: 1px solid #2D3548;
    border-radius: 10px;
    overflow: hidden;
}

/* صندوق المنهجية القابل للطي */
.streamlit-expanderHeader {
    background-color: #1A1F2E !important;
    border: 1px solid #2D3548 !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-weight: 700 !important;
    direction: rtl;
}
</style>
""", unsafe_allow_html=True)

# --- نظام سجل تتبع الأداء والصفقات الحية (Trade Journal) ---
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

# --- ترويسة الصفحة ---
st.markdown("""
<div class="rtl-box">
    <h1 style="margin-bottom: 4px; color: #F8FAFC; font-weight: 800;">📈 محلل السوينق والموجات الاحترافي (الإصدار v2.2 - مع MACD و Bollinger)</h1>
    <p style="color: #94A3B8; font-size: 1.05rem; margin-bottom: 18px;">منصة تحليلية متكاملة لهيكل السوق، الموجات، مؤشرات الزخم والتقلب (MACD / Bollinger)، هندسة اليونانيات، وسجل تتبع الأداء.</p>
</div>
""", unsafe_allow_html=True)

# --- القائمة الجانبية (المدخلات والتحكم الحي) ---
st.sidebar.markdown('<div class="rtl-box"><h3>⚙️ إعدادات التحليل والتحكم (v2.2)</h3></div>', unsafe_allow_html=True)
ticker = st.sidebar.selectbox("اختر المؤشر / السهم:", ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META"])
interval = st.sidebar.selectbox("الإطار الزمني (Timeframe):", ["1d", "1h"], index=0)
period = "1y" if interval == "1d" else "3mo"
swing_order = st.sidebar.slider("حساسية اكتشاف القمم والقيعان (Swing Sensitivity):", min_value=3, max_value=15, value=7)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="rtl-box"><h4>🎛️ إعدادات العقود واليونانيات</h4></div>', unsafe_allow_html=True)
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

# دالة لحساب أيام التداول القادمة (تخطي عطلة نهاية الأسبوع السبت والأحد)
def add_trading_days(start_date, trading_days):
    current_date = start_date
    added = 0
    while added < trading_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            added += 1
    return current_date

# --- دالة حساب اليونانيات التقديرية عبر نموذج بلاك-شولز (Black-Scholes Greeks) ---
def calculate_bs_greeks(S, K, T_days, r, sigma, is_call=True):
    T = max(T_days / 365.0, 1e-5)
    sigma = max(sigma, 0.10)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if is_call:
        delta = norm.cdf(d1)
        theta_annual = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                        - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        delta = norm.cdf(d1) - 1.0
        theta_annual = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                        + r * K * np.exp(-r * T) * norm.cdf(-d2))
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta_daily = theta_annual / 365.0
    
    return {
        "delta": delta,
        "gamma": gamma,
        "theta_daily": theta_daily,
        "iv_pct": sigma * 100
    }

# --- دالة التحليل المتكاملة لأي سهم أو مؤشر (مع MACD و Bollinger Bands) ---
@st.cache_data(ttl=300)
def analyze_ticker(symbol, period, interval, swing_order, style_choice):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    
    if df.empty or len(df) < 50:
        return None
    
    # المتوسطات المتحركة
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # مؤشر RSI
    delta_s = df['Close'].diff()
    gain = (delta_s.where(delta_s > 0, 0)).rolling(window=14).mean()
    loss = (-delta_s.where(delta_s < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # مؤشر MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # مؤشر Bollinger Bands (20, 2)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # حساب التقلب التاريخي السنوي
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
    
    is_bullish = current_price > ema50_val
    trend = "صاعد (Bullish) 🟢" if is_bullish else "هابط (Bearish) 🔴"
    option_type = "CALL 🟢 (شراء)" if is_bullish else "PUT 🔴 (بيع)"
    
    # تقييم MACD
    if macd_val > macd_sig:
        macd_status = "إيجابي / زخم صاعد 🟢"
    else:
        macd_status = "سلبي / زخم هابط 🔴"
        
    # تقييم Bollinger Bands (هل هناك انكماش Squeeze ينذر بانفجار سعري؟)
    bb_width_mean = float(df['BB_Width'].rolling(50).mean().iloc[-1])
    if not np.isnan(bb_width_mean) and bb_width_val < bb_width_mean * 0.8:
        bb_status = "انكماش حاد (Squeeze) ⚡ - توقع انفجار سعري قريب!"
    else:
        bb_status = "وضع اعتيادي (استقرار نسبي) 📊"
    
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
        target_1 = current_price * 1.02 if is_bullish else current_price * 0.98
        target_2 = current_price * 1.04 if is_bullish else current_price * 0.96
        stop_loss = current_price * 0.98 if is_bullish else current_price * 1.02
        est_days_t1, est_days_t2, recommended_dte = 5, 10, 35

    # اختيار سعر التنفيذ (Strike) بناءً على نمط الدلتا
    if "OTM" in style_choice:
        strike_rec = round(current_price + current_atr * 0.75) if is_bullish else round(current_price - current_atr * 0.75)
    elif "ITM" in style_choice:
        strike_rec = round(current_price - current_atr * 0.75) if is_bullish else round(current_price + current_atr * 0.75)
    else:
        strike_rec = round(current_price)

    # حساب اليونانيات
    greeks = calculate_bs_greeks(
        S=current_price,
        K=strike_rec,
        T_days=recommended_dte,
        r=0.045,
        sigma=est_volatility,
        is_call=is_bullish
    )

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
        "macd_status": macd_status,
        "bb_status": bb_status,
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
        "rr_ratio": rr_ratio,
        "greeks": greeks
    }

# إنشاء التبويبات الرئيسية
tab1, tab2, tab3 = st.tabs([
    "📊 التحليل التفصيلي وبطاقة الصفقة", 
    "🔍 الماسح الآلي للأسهم والمؤشرات (Screener)",
    "📈 سجل تتبع الأداء (Trade Journal)"
])

with tab1:
    res = analyze_ticker(ticker, period, interval, swing_order, trade_style)
    
    if res is None:
        st.error("تعذر جلب البيانات الكافية لهذا المؤشر حالياً.")
    else:
        df = res["df"]
        greeks = res["greeks"]
        badge_class = "badge-bull" if res["is_bullish"] else "badge-bear"
        
        # --- شريط الحالة الحي العلوي (Live Status Banner) ---
        st.markdown(f"""
        <div class="live-banner">
            <div class="banner-item">📌 الأصل المفحوص: <strong style="color:#38BDF8; font-size:1.1rem;">{ticker}</strong> ({interval})</div>
            <div class="banner-item">🧭 الاتجاه العام: <span class="{badge_class}">{res['trend']}</span></div>
            <div class="banner-item">⚡ RSI: <strong>{res['current_rsi']:.1f}</strong></div>
            <div class="banner-item">📊 MACD: <strong>{res['macd_status']}</strong></div>
            <div class="banner-item">⚖️ R:R: <strong style="color:#FBBF24;">1 : {res['rr_ratio']:.2f}</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- عرض بطاقات الملخص العلوي ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("السعر الحالي", f"${res['current_price']:.2f}", res["trend"])
        col2.metric("منطقة الهدف الأول (1.272)", f"${res['target_1']:.2f}", f"متوقع: {res['date_t1']} (~{res['est_days_t1']} أيام)")
        col3.metric("منطقة الهدف الثاني (1.618)", f"${res['target_2']:.2f}", f"متوقع: {res['date_t2']} (~{res['est_days_t2']} أيام)")
        col4.metric("نوع العقد المقترح", res["option_type"], f"Strike: ${res['strike_rec']} | DTE: {res['recommended_dte']} يوم")

        # --- بطاقة التوصية التنفيذية التفصيلية ---
        st.markdown(f'<div class="rtl-box"><h3 style="margin-top:20px;">🎯 بطاقة السيناريو التنفيذي وعقد الأوبشن المقترح ({ticker})</h3></div>', unsafe_allow_html=True)
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

        # ==============================================================================
        # هندسة اليونانيات وتقييم المخاطر (مع تحديث لون شارة الثيتا العكسي)
        # ==============================================================================
        st.markdown(f'<div class="rtl-box"><h3 style="margin-top:22px;">🧬 هندسة اليونانيات التقديرية للعقد (Greeks Risk Profile)</h3></div>', unsafe_allow_html=True)
        
        abs_delta = abs(greeks['delta'])
        theta_per_contract = abs(greeks['theta_daily']) * 100
        gamma_val = greeks['gamma']
        
        if 0.40 <= abs_delta <= 0.65:
            delta_eval = "مثالية ومتوازنة ✅"
        elif abs_delta > 0.65:
            delta_eval = "عالية الأمان (ITM) 🛡️"
        else:
            delta_eval = "هجومية (OTM) ⚡"
            
        if theta_per_contract <= 15:
            theta_eval = "تآكل بطيء وآمن 🟢"
        elif theta_per_contract <= 35:
            theta_eval = "تآكل متوسط 🟡"
        else:
            theta_eval = "تآكل مرتفع 🔴"
            
        if gamma_val >= 0.02:
            gamma_eval = "تسارع انفجاري عالي 🚀"
        elif gamma_val >= 0.008:
            gamma_eval = "تسارع جيد ومستقر 📈"
        else:
            gamma_eval = "تسارع بطيء (سهم ثقيل) 🐢"

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("الدلتا (Delta Δ)", f"{greeks['delta']:.2f}", delta_eval)
        g2.metric("الثيتا اليومية (Theta Θ)", f"-${theta_per_contract:.2f} / يوم", theta_eval, delta_color="inverse")
        g3.metric("الجاما (Gamma Γ)", f"{gamma_val:.4f}", gamma_eval)
        g4.metric("التقلب المقدر (Vol)", f"{greeks['iv_pct']:.1f}%", f"صلاحية آمنة: {res['recommended_dte']} يوم")

        # --- الشرح التحليلي الشامل (الموجات + اليونانيات + MACD + Bollinger) ---
        direction_word = "ارتفاع" if res["is_bullish"] else "انخفاض"
        st.markdown(f"""
        <div class="greeks-explain-box">
            <h4 style="color: #A78BFA; margin-top: 0; margin-bottom: 10px;">💡 التحليل الفني الشامل لـ {ticker} (اليونانيات + الزخم MACD + النطاق Bollinger)</h4>
            <ul style="margin-bottom: 0; padding-right: 20px; color: #E2E8F0;">
                <li style="margin-bottom: 8px;">
                    <strong>1. تحليل الزخم (MACD: {res['macd_status']}):</strong> 
                    مؤشر MACD يؤكد حالة الزخم الحالية؛ توافق إشارة MACD مع اتجاه الموجة يعزز من موثوقية الهدف السعري الأول (${res['target_1']:.2f}).
                </li>
                <li style="margin-bottom: 8px;">
                    <strong>2. قياس التقلب وانكماش النطاق (Bollinger Bands: {res['bb_status']}):</strong> 
                    مؤشر Bollinger Bands يراقب عرض القناة السعرية؛ إذا ظهر انكماش (Squeeze)، فهذا ينذر بانفجار حاد في حركة السعر قريباً لصالح كسر الموجة.
                </li>
                <li style="margin-bottom: 8px;">
                    <strong>3. حساسية العقد (الدلتا {greeks['delta']:.2f}):</strong> 
                    العقد يربح نحو <strong>${abs_delta * 100:.0f}</strong> تقريباً مقابل كل <strong>$1.00</strong> يتحركه السهم في اتجاه {direction_word}.
                </li>
                <li>
                    <strong>4. التآكل الزمني (الثيتا -${theta_per_contract:.2f}/يوم) & التسارع (الجاما {gamma_val:.4f}):</strong> 
                    تم اختيار صلاحية آمنة لمدة <strong>{res['recommended_dte']} يوم</strong> لتقليل تأثير الثيتا، بينما تضمن الجاما تسارع الأرباح بمجرد وصول السهم نحو الأهداف المرسومة.
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"📌 إضافة عقد {ticker} الحالي إلى سجل تتبع الأداء"):
            opt_short = "PUT" if "PUT" in res["option_type"] else "CALL"
            log_new_trade(ticker, opt_short, res["strike_rec"], float(res["current_price"]))
            st.success(f"تمت إضافة عقد {ticker} بنجاح إلى جدول سجل التتبع في التبويب الثالث!")

        st.markdown('<div class="rtl-box"><p style="color:#94A3B8; font-size:0.88rem;">⚠️ <strong>تنويه احترافي:</strong> مستويات فيبوناتشي، التواريخ التقديرية، وحسابات اليونانيات ومؤشرات الزخم هي <strong>تقديرات رياضية واحتمالات</strong> وليست ضمانات مؤكدة.</p></div>', unsafe_allow_html=True)

        # ==============================================================================
        # الرسم البياني المتقدم (4 لوحات متصلة: الشموع + Bollinger, RSI, MACD, Volume)
        # ==============================================================================
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.50, 0.18, 0.17, 0.15],
            subplot_titles=(f"التحليل الهيكلي وموجات السوينق مع Bollinger Bands لـ {ticker}", "مؤشر الزخم RSI (14)", "مؤشر الزخم والتسارع MACD", "حجم التداول (Volume)")
        )

{"""        # الشموع اليابانية والمتوسطات و Bollinger Bands
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="السعر", increasing_line_color='#10B981', decreasing_line_color='#EF4444'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(56, 189, 248, 0.5)', width=1, dash='dash'), name="Bollinger Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(56, 189, 248, 0.5)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(56, 189, 248, 0.04)', name="Bollinger Lower"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], line=dict(color='rgba(56, 189, 248, 0.8)', width=1), name="BB Middle"), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#38BDF8', width=1.4), name="EMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='#F59E0B', width=1.8), name="EMA 50"), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['Swing_High'], mode='markers', marker=dict(color='#EF4444', size=9, symbol='triangle-down'), name="قمة موجة"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Swing_Low'], mode='markers', marker=dict(color='#10B981', size=9, symbol='triangle-up'), name="قاع موجة"), row=1, col=1)

        fig.add_hline(y=res['target_1'], line_dash="dash", line_color="#10B981", annotation_text=f"هدف T1: ${res['target_1']:.2f}", row=1, col=1)
        fig.add_hline(y=res['target_2'], line_dash="dot", line_color="#34D399", annotation_text=f"هدف T2: ${res['target_2']:.2f}", row=1, col=1)
        fig.add_hline(y=res['stop_loss'], line_dash="dash", line_color="#EF4444", annotation_text=f"وقف الخسارة: ${res['stop_loss']:.2f}", row=1, col=1)

        # لوحة RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#A78BFA', width=1.6), name="RSI (14)"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#64748B", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#64748B", row=2, col=1)

        # لوحة MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#38BDF8', width=1.5), name="MACD Line"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#F59E0B', width=1.5), name="Signal Line"), row=3, col=1)
        colors_hist = ['#10B981' if val >= 0 else '#EF4444' for val in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors_hist, name="MACD Histogram"), row=3, col=1)

        # لوحة Volume
        colors_vol = ['#10B981' if c >= o else '#EF4444' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors_vol, name="Volume"), row=4, col=1)

        fig.update_layout(
            height=900,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#131722",
            plot_bgcolor="#1A1F2E",
            font=dict(family="Cairo, sans-serif", color="#E2E8F0"),
            title=dict(text=f"التحليل الفني المتقدم والمتكامل لـ {ticker}", x=0.98, xanchor="right"),
            xaxis=dict(gridcolor="#2A3142"),
            yaxis=dict(gridcolor="#2A3142"),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- قسم إدارة المخاطر وحاسبة الصفقة ---
        st.markdown('<div class="rtl-box"><h3>🛡️ حاسبة إدارة المخاطر وحجم الصفقة (Position Sizing)</h3></div>', unsafe_allow_html=True)
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
    st.markdown("""
    <div class="rtl-box">
        <h3>🔍 الماسح الآلي للفرص السوينقية واليونانيات (Multi-Ticker Screener v2.2)</h3>
        <p style="color:#94A3B8;">يفحص الماسح القائمة لحظياً، محللاً الموجات، مؤشرات الزخم MACD والبولنجر، بالإضافة إلى قيم الدلتا والثيتا للعقود.</p>
    </div>
    """, unsafe_allow_html=True)
    
    watchlist = st.multiselect(
        "اختر قائمة الأسهم والمؤشرات للفحص:",
        ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META", "AMD", "GOOGL", "NFLX"],
        default=["SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "MSFT", "META"]
    )
    
    if st.button("🚀 بدء الفحص الشامل الآن", type="primary"):
        screener_rows = []
        progress_bar = st.progress(0)
        for idx, sym in enumerate(watchlist):
            data_res = analyze_ticker(sym, period, interval, swing_order, trade_style)
            if data_res:
                g_data = data_res["greeks"]
                status = "✅ فرصة قوية (R:R >= 2)" if data_res["rr_ratio"] >= 2.0 else "⏳ مراقبة (R:R < 2)"
                screener_rows.append({
                    "الرمز": sym,
                    "السعر الحالي": f"${data_res['current_price']:.2f}",
                    "الاتجاه": data_res["trend"],
                    "نوع العقد": data_res["option_type"],
                    "Strike مقترح": f"${data_res['strike_rec']}",
                    "Delta Δ": f"{g_data['delta']:.2f}",
                    "Theta اليومية": f"-${abs(g_data['theta_daily'])*100:.1f}",
                    "MACD": data_res["macd_status"],
                    "هدف 1 (1.272)": f"${data_res['target_1']:.2f}",
                    "وقف الخسارة": f"${data_res['stop_loss']:.2f}",
                    "صلاحية (DTE)": f"{data_res['recommended_dte']} يوم",
                    "C/R (R:R)": f"1 : {data_res['rr_ratio']:.2f}",
                    "التقييم": status
                })
            progress_bar.progress((idx + 1) / len(watchlist))
            
        if screener_rows:
            screener_df = pd.DataFrame(screener_rows)
            
            def highlight_contract(row):
                if "CALL" in str(row["نوع العقد"]):
                    return ['background-color: rgba(16, 185, 129, 0.12)'] * len(row)
                elif "PUT" in str(row["نوع العقد"]):
                    return ['background-color: rgba(239, 68, 68, 0.12)'] * len(row)
                return [''] * len(row)

            styled_df = screener_df.style.apply(highlight_contract, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            st.markdown('<div class="rtl-box"><p style="color:#94A3B8; font-size:0.88rem;">⚠️ تذكير: الأهداف السعرية واليونانيات ومؤشرات الزخم تمثل تقديرات رياضية.</p></div>', unsafe_allow_html=True)
        else:
            st.warning("لم يتم العثور على بيانات كافية للقائمة المحددة.")

with tab3:
    st.markdown("""
    <div class="rtl-box">
        <h3>📊 سجل تتبع الأداء والصفقات الحية (Trade Journal)</h3>
        <p style="color:#94A3B8;">هذا الجدول مخصص لتوثيق العقود وتتبع أعلى سعر وصلت إليه لقياس كفاءة الأداة بدقة إحصائية.</p>
    </div>
    """, unsafe_allow_html=True)
    
    init_trade_log()
    history_df = pd.read_csv(LOG_FILE)

    if not history_df.empty:
        if "Alert_Price" in history_df.columns and "Max_Price_Reached" in history_df.columns:
            history_df["Profit_Percentage_%"] = (
                (history_df["Max_Price_Reached"] - history_df["Alert_Price"]) / history_df["Alert_Price"] * 100
            ).round(2)

        st.write("💡 **تلميح:** يمكنك تعديل خانة (أعلى سعر وصل له Max_Price_Reached) مباشرة من الجدول أدناه:")
        
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
        st.info("لا توجد صفقات مسجلة حتى الآن. يمكنك إضافة عقود للمراقبة من التبويب الأول أو عبر النموذج أدناه.")

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

# --- قسم التوثيق ومنهجية عمل الأداة ---
with st.expander("📖 منهجية التحليل والمعايير الفنية المعتمدة في الأداة (11 ركيزة أساسية)"):
    st.markdown("""
    <div class="rtl-box">
    تعتمد هذه الأداة في إصدارها **v2.2** على منظومة متكاملة تجمع بين التحليل الفني الهيكلي، نسب فيبوناتشي، هندسة اليونانيات (Black-Scholes)، ومؤشرات الزخم والتقلب المتقدمة:

    ### 1 الى 5. الهيكل الأساسي والموجات والزمن (EMA, Swings, Fibonacci, DTE)
    * رصد الاتجاه بـ EMA 50، اكتشاف القمم والقيعان المحورية آلياً، حساب أهداف امتداد فيبوناتشي (1.272 و 1.618)، وتقدير تواريخ الصلاحية.

    ### 6. هندسة اليونانيات وتقييم العقود (Delta, Gamma, Theta)
    * تسعير اليونانيات عبر نموذج Black-Scholes مع تعديل لون شارة الثيتا العكسي وإبراز حساسية العقد والتآكل اليومي بالدولار.

    ### 7. مؤشر الزخم MACD — جديد في v2.2
    * تتبع تقاطعات خط الإشارة والـ Histogram لتأكيد قوة اتجاه الموجة ورصد أي دايفرجنس محتمل.

    ### 8. نطاقات بولنجر Bollinger Bands — جديد في v2.2
    * قياس التذبذب ورصد حالات انكماش النطاق (Squeeze) التي تسبق الانفجارات السعرية الكبرى.

    ### 9. عزم الزخم RSI (14)
    * قياس تشبع الشراء والبيع لتأكيد توقيت الدخول.

    ### 10 & 11. الماسح الآلي وسجل تتبع الأداء (Screener & Trade Journal)
    * فحص القوائم لحظياً وتوثيق نتائج الصفقات والعقود في جدول الأداء.
    </div>
    """, unsafe_allow_html=True)
