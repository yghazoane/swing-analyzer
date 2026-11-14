"""
================================================================================
اسم الأداة: محلل العقود والسوينق الاحترافي (Pro Options & Swing Analyzer)
الإصدار: v2.3.7
الوصف: تحليل هيكلي للأسهم والمؤشرات (بما فيها SPX 500)، تحديد الأهداف بفيبوناتشي
       و ATR، تسعير العقود وحساب اليونانيات، وإدارة حجم العقود مع فلتر ذكي
       للبحث التلقائي عن العقود الرخيصة (< $3.00) حتى على مؤشر SPX.
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

# إعدادات الصفحة
st.set_page_config(
    page_title="محلل العقود والسوينق الاحترافي v2.3.7",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تنسيق بصري آمن وخفيف
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, label, button {
        font-family: 'Cairo', sans-serif !important;
    }
    .rtl-box { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True
)

LOG_FILE = "trade_history.csv"


def init_trade_log():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=[
            "Timestamp", "Ticker", "Style", "Option_Type", "Strike", "Expiry",
            "Entry_Premium", "Max_Premium", "Stop_Underlying", "Target1_Underlying",
            "Contracts", "Status", "Notes"
        ])
        df.to_csv(LOG_FILE, index=False)


def log_new_trade(ticker, style, opt_type, strike, expiry, premium, stop_u, target1_u, contracts):
    init_trade_log()
    df = pd.read_csv(LOG_FILE)
    new_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Ticker": ticker,
        "Style": style,
        "Option_Type": opt_type,
        "Strike": strike,
        "Expiry": expiry,
        "Entry_Premium": round(premium, 2),
        "Max_Premium": round(premium, 2),
        "Stop_Underlying": round(stop_u, 2),
        "Target1_Underlying": round(target1_u, 2),
        "Contracts": contracts,
        "Status": "مفتوح للمراقبة ⏳",
        "Notes": "إشارة نظامية"
    }])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)


def add_trading_days(start_date, trading_days):
    current_date = start_date
    added = 0
    while added < trading_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            added += 1
    return current_date


def round_to_strike_step(price):
    """تقريب السترايك لأقرب فئة تداول حقيقية في السوق الأمريكي"""
    if price < 50:
        return round(price * 2) / 2.0
    elif price < 250:
        return float(round(price))
    elif price < 500:
        return float(round(price / 2.5) * 2.5)
    else:
        return float(round(price / 5.0) * 5.0)


def get_strike_step(price):
    if price < 50:
        return 0.5
    elif price < 250:
        return 1.0
    elif price < 500:
        return 2.5
    else:
        return 5.0


def calculate_bs_metrics(S, K, T_days, r, sigma, is_call=True):
    """حساب السعر النظري للعقد واليونانيات الكاملة (Delta, Gamma, Theta, Vega)"""
    T = max(T_days / 365.0, 1e-5)
    sigma = max(sigma, 0.10)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if is_call:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta_annual = (-(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
                        - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta_annual = (-(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
                        + r * K * np.exp(-r * T) * norm.cdf(-d2))

    gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
    vega = (S * norm.pdf(d1) * sqrt_T) / 100.0
    theta_daily = theta_annual / 365.0

    return {
        "premium": max(float(price), 0.05),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta_daily": float(theta_daily),
        "vega": float(vega),
        "iv_pct": float(sigma * 100)
    }


def find_budget_strike(S, T_days, r, sigma, is_call, max_premium=2.85):
    """
    خوارزمية بحث تكرارية لإيجاد أول سترايك OTM يكون قسطه النظري أقل من max_premium ($3.00)
    تعمل بكفاءة حتى على المؤشرات مرتفعة السعر مثل SPX.
    """
    step = get_strike_step(S)
    candidate_k = round_to_strike_step(S)
    direction = 1 if is_call else -1

    best_k = candidate_k
    for _ in range(120):
        candidate_k += direction * step
        if candidate_k <= step:
            break
        metrics = calculate_bs_metrics(S, candidate_k, T_days, r, sigma, is_call)
        best_k = candidate_k
        if metrics["premium"] <= max_premium:
            break
    return float(best_k)


@st.cache_data(ttl=600)
def fetch_price_history(symbol, period, interval):
    """جلب البيانات التاريخية مع آلية احتياطية وتوافق مع رمز مؤشر SPX (^GSPC)"""
    try:
        query_symbol = "^GSPC" if symbol == "SPX" else symbol
        tk = yf.Ticker(query_symbol)
        df = tk.history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            df = yf.download(query_symbol, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is not None and not df.empty:
            cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            df = df[cols].dropna()
        return df, ""
    except Exception as e:
        return pd.DataFrame(), str(e)


@st.cache_data(ttl=600)
def fetch_live_option_hint(symbol, target_dte, is_call, target_strike, budget_mode=False, max_budget=3.0):
    """جلب أقرب عقد حقيقي من سلسلة الأوبشن الحية مع دعم البحث عن عقد تحت السقف السعري"""
    try:
        query_symbol = "^GSPC" if symbol == "SPX" else symbol
        tk = yf.Ticker(query_symbol)
        expirations = tk.options
        if not expirations:
            return None
        today = datetime.now().date()
        best_exp = min(
            expirations,
            key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d").date() - today).days - target_dte)
        )
        chain = tk.option_chain(best_exp)
        df_opt = chain.calls if is_call else chain.puts
        if df_opt is None or df_opt.empty:
            return None

        df_opt = df_opt.copy()
        df_opt['mid_calc'] = np.where(
            (df_opt['bid'] > 0) & (df_opt['ask'] > 0),
            (df_opt['bid'] + df_opt['ask']) / 2.0,
            df_opt['lastPrice']
        )

        if budget_mode:
            affordable = df_opt[(df_opt['mid_calc'] > 0.15) & (df_opt['mid_calc'] <= max_budget)]
            if not affordable.empty:
                # اختيار أعلى عقد سيولة وقرباً للسعر ضمن الميزانية (< $3)
                idx = (affordable['strike'] - target_strike).abs().idxmin()
                row = affordable.loc[idx]
            else:
                idx = (df_opt['strike'] - target_strike).abs().idxmin()
                row = df_opt.loc[idx]
        else:
            idx = (df_opt['strike'] - target_strike).abs().idxmin()
            row = df_opt.loc[idx]

        bid = float(row.get('bid', 0) or 0)
        ask = float(row.get('ask', 0) or 0)
        last_p = float(row.get('lastPrice', 0) or 0)
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else last_p
        spread_pct = ((ask - bid) / ask * 100) if ask > 0 and bid > 0 else 0.0
        return {
            "expiry": best_exp,
            "strike": float(row['strike']),
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_pct": spread_pct,
            "volume": int(row.get('volume', 0) or 0),
            "oi": int(row.get('openInterest', 0) or 0),
            "iv": float(row.get('impliedVolatility', 0) or 0) * 100
        }
    except Exception:
        return None


def analyze_ticker(symbol, interval, swing_order, pricing_strategy, trading_mode, max_contract_price=3.0):
    try:
        period_map = {"15m": "1mo", "1h": "3mo", "1d": "1y"}
        period = period_map.get(interval, "1y")
        df, err_msg = fetch_price_history(symbol, period, interval)
        if df is None or df.empty or len(df) < 30:
            return None, err_msg or "البيانات الواردة من المصدر فارغة أو غير مكتملة."

        df = df.copy()

        # المتوسطات المتحركة
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

        # مؤشر RSI (14)
        delta_s = df['Close'].diff()
        gain = (delta_s.where(delta_s > 0, 0)).rolling(window=14).mean()
        loss = (-delta_s.where(delta_s < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = (100 - (100 / (1 + rs))).fillna(50)

        # مؤشر MACD
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # نطاقات بولنجر (Bollinger Bands)
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Mid'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']

        # متوسط المدى الحقيقي ATR (14)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
        df['ATR'] = df['ATR'].bfill().fillna(df['Close'] * 0.015)

        # التقلب التاريخي السنوي (HV)
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        bars_per_year = {"15m": 252 * 26, "1h": 252 * 6.5, "1d": 252}.get(interval, 252)
        est_vol = float(log_returns.rolling(20).std().iloc[-1] * np.sqrt(bars_per_year))
        if np.isnan(est_vol) or est_vol <= 0.05:
            est_vol = 0.25

        # قمم وقيعان السوينق الهيكلية
        high_idx = argrelextrema(df['High'].values, np.greater_equal, order=swing_order)[0]
        low_idx = argrelextrema(df['Low'].values, np.less_equal, order=swing_order)[0]
        df['Swing_High'] = np.nan
        df['Swing_Low'] = np.nan
        df.iloc[high_idx, df.columns.get_loc('Swing_High')] = df['High'].iloc[high_idx]
        df.iloc[low_idx, df.columns.get_loc('Swing_Low')] = df['Low'].iloc[low_idx]

        last_highs = df.dropna(subset=['Swing_High'])
        last_lows = df.dropna(subset=['Swing_Low'])

        current_price = float(df['Close'].iloc[-1])
        current_atr = float(df['ATR'].iloc[-1])
        ema20_val = float(df['EMA20'].iloc[-1])
        ema50_val = float(df['EMA50'].iloc[-1])
        current_rsi = float(df['RSI'].iloc[-1])
        macd_val = float(df['MACD'].iloc[-1])
        macd_sig = float(df['MACD_Signal'].iloc[-1])
        bb_w = float(df['BB_Width'].iloc[-1])
        bb_w_avg = float(df['BB_Width'].rolling(50, min_periods=10).mean().iloc[-1])

        # تحديد الاتجاه الهيكلي والزخم
        is_bullish = current_price > ema50_val
        trend = "صاعد (Bullish) 🟢" if is_bullish else "هابط (Bearish) 🔴"
        option_type = "CALL 🟢 (شراء)" if is_bullish else "PUT 🔴 (بيع)"

        # تقييم جودة الإشارة (Confluence Score من 5)
        score = 0
        if (is_bullish and current_price > ema20_val) or (not is_bullish and current_price < ema20_val):
            score += 1
        if (is_bullish and macd_val > macd_sig) or (not is_bullish and macd_val < macd_sig):
            score += 1
        if (is_bullish and 45 <= current_rsi <= 68) or (not is_bullish and 32 <= current_rsi <= 55):
            score += 1
        is_squeeze = (not np.isnan(bb_w)) and (not np.isnan(bb_w_avg)) and (bb_w < bb_w_avg * 0.85)
        if is_squeeze:
            score += 1

        # حساب الأهداف الهيكلية بفيبوناتشي ووقف الخسارة بـ ATR
        if len(last_highs) >= 1 and len(last_lows) >= 1:
            rec_high = float(last_highs['Swing_High'].iloc[-1])
            rec_low = float(last_lows['Swing_Low'].iloc[-1])
            wave_h = max(abs(rec_high - rec_low), current_atr * 1.5)
            wave_bars = max(abs(df.index.get_loc(last_highs.index[-1]) - df.index.get_loc(last_lows.index[-1])), 5)

            if is_bullish:
                base_low = min(rec_low, current_price - current_atr)
                target_1 = max(current_price + current_atr * 1.5, base_low + wave_h * 1.272)
                target_2 = max(target_1 + current_atr, base_low + wave_h * 1.618)
                stop_loss = min(rec_low - current_atr * 0.35, current_price - current_atr * 1.2)
            else:
                base_high = max(rec_high, current_price + current_atr)
                target_1 = min(current_price - current_atr * 1.5, base_high - wave_h * 1.272)
                target_2 = min(target_1 - current_atr, base_high - wave_h * 1.618)
                stop_loss = max(rec_high + current_atr * 0.35, current_price + current_atr * 1.2)
        else:
            wave_bars = 8
            if is_bullish:
                target_1 = current_price + current_atr * 2.0
                target_2 = current_price + current_atr * 3.2
                stop_loss = current_price - current_atr * 1.0
            else:
                target_1 = current_price - current_atr * 2.0
                target_2 = current_price - current_atr * 3.2
                stop_loss = current_price + current_atr * 1.0

        # تحديد الصلاحية الزمنية (DTE) حسب أسلوب المضاربة
        if "يومي" in trading_mode or interval in ["15m", "1h"]:
            est_days_t1 = 1 if interval == "15m" else 2
            est_days_t2 = 3 if interval == "15m" else 5
            recommended_dte = 3 if interval == "15m" else 7
        else:
            est_days_t1 = max(3, int(wave_bars * 0.8))
            est_days_t2 = max(6, int(wave_bars * 1.3))
            recommended_dte = min(60, max(28, est_days_t2 + 21))

        is_budget_mode = "رخيصة" in pricing_strategy

        # هندسة السترايك الذكية
        if is_budget_mode:
            # للعقود الرخيصة نستخدم صلاحية قصيرة (1-5 أيام) ونبحث عن السترايك الذي يحقق السعر المستهدف
            recommended_dte = 2 if interval == "15m" else 5
            strike_rec = find_budget_strike(
                S=current_price,
                T_days=recommended_dte,
                r=0.045,
                sigma=est_vol,
                is_call=is_bullish,
                max_premium=max_contract_price * 0.92
            )
        elif "هجومي" in pricing_strategy:
            raw_strike = current_price + (current_atr * 0.9 if is_bullish else -current_atr * 0.9)
            strike_rec = round_to_strike_step(raw_strike)
        elif "تحوطي" in pricing_strategy:
            raw_strike = current_price - (current_atr * 0.85 if is_bullish else -current_atr * 0.85)
            strike_rec = round_to_strike_step(raw_strike)
        else:
            strike_rec = round_to_strike_step(current_price)

        # حساب اليونانيات والسعر النظري للعقد
        greeks = calculate_bs_metrics(
            S=current_price, K=strike_rec, T_days=recommended_dte,
            r=0.045, sigma=est_vol, is_call=is_bullish
        )

        # محاولة جلب بيانات السلسلة الحية للمقارنة
        live_opt = fetch_live_option_hint(
            symbol, recommended_dte, is_bullish, strike_rec,
            budget_mode=is_budget_mode, max_budget=max_contract_price
        )

        # في وضع العقود الرخيصة، إذا وجدنا عقداً حياً تحت السقف نعتمد سترايكه الفعلي
        if is_budget_mode and live_opt and 0.15 < live_opt["mid"] <= max_contract_price:
            strike_rec = live_opt["strike"]
            greeks = calculate_bs_metrics(
                S=current_price, K=strike_rec, T_days=recommended_dte,
                r=0.045, sigma=est_vol, is_call=is_bullish
            )

        risk_per_share = abs(current_price - stop_loss)
        reward_t1 = abs(target_1 - current_price)
        rr_ratio = reward_t1 / risk_per_share if risk_per_share > 0 else 0
        if rr_ratio >= 2.0:
            score += 1

        last_dt = df.index[-1].to_pydatetime() if hasattr(df.index[-1], 'to_pydatetime') else datetime.now()
        if getattr(last_dt, "tzinfo", None) is not None:
            last_dt = last_dt.replace(tzinfo=None)
        date_t1 = add_trading_days(last_dt, est_days_t1).strftime("%Y-%m-%d")
        date_t2 = add_trading_days(last_dt, est_days_t2).strftime("%Y-%m-%d")
        expiry_str = (last_dt + timedelta(days=recommended_dte)).strftime("%Y-%m-%d")

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
            "rr_ratio": rr_ratio,
            "est_days_t1": est_days_t1,
            "est_days_t2": est_days_t2,
            "date_t1": date_t1,
            "date_t2": date_t2,
            "recommended_dte": recommended_dte,
            "expiry_str": expiry_str,
            "greeks": greeks,
            "live_opt": live_opt,
            "score": score,
            "macd_status": "إيجابي 🟢" if macd_val > macd_sig else "سلبي 🔴",
            "bb_status": "انفجار وشيك ⚡ (Squeeze)" if is_squeeze else "نطاق طبيعي 📊"
        }, ""
    except Exception as e:
        return None, str(e)


# --- واجهة المستخدم ---
st.title("📈 محلل العقود والسوينق الاحترافي (v2.3.7)")
st.markdown("منصة تحليلية مصممة بعقلية مضارب العقود: محرك بحث ذكي لإيجاد العقود الرخيصة (< $3.00) حتى على مؤشر SPX.")

# القائمة الجانبية
st.sidebar.header("⚙️ إعدادات التداول")
trading_mode = st.sidebar.radio("أسلوب المضاربة:", ["مضاربة يومية / قصيرة (Intraday | 1-7 DTE)", "سوينق (Swing | 21-45 DTE)"])
default_tf_idx = 1 if "يومية" in trading_mode else 2
interval = st.sidebar.selectbox("الإطار الزمني:", ["15m", "1h", "1d"], index=default_tf_idx)
ticker = st.sidebar.selectbox("الأصل المالي:", ["SPX", "SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META", "AMD", "GOOGL", "NFLX"])
swing_order = st.sidebar.slider("حساسية القمم والقيعان:", min_value=3, max_value=15, value=7)

if st.sidebar.button("🔄 تحديث البيانات ومسح الكاش"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🎯 استراتيجية التسعير والعقود")
pricing_strategy = st.sidebar.selectbox(
    "فلتر تصفية العقود:",
    [
        "🔥 عقود رخيصة ومضاربية (تحت السقف السعري)",
        "⚖️ متوازن (ATM | Delta ~0.50)",
        "⚡ هجومي (OTM متوسط | Delta ~0.35)",
        "🛡️ تحوطي آمن (ITM | Delta ~0.65)"
    ]
)
max_contract_price = st.sidebar.number_input(
    "الحد الأقصى لسعر العقد ($):",
    min_value=0.50, max_value=50.0, value=3.00, step=0.50,
    help="عند تفعيل فلتر العقود الرخيصة، سيبحث النظام عن سترايك سعره أقل من هذا الرقم (مثلاً $3.00 = $300 للعقد)."
)
portfolio_size = st.sidebar.number_input("حجم المحفظة ($):", min_value=1000, value=10000, step=1000)
risk_pct = st.sidebar.number_input("المخاطرة القصوى للصفقة (%):", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
contract_sl_pct = st.sidebar.slider("نسبة وقف خسارة العقد (% من القسط):", min_value=20, max_value=60, value=35, step=5)

tab1, tab2, tab3 = st.tabs([
    "📊 التحليل التفصيلي وبطاقة العقد",
    "🔍 الماسح الآلي للفرص (Screener)",
    "📈 سجل الصفقات والأداء (Journal)"
])

with tab1:
    res, err_detail = analyze_ticker(ticker, interval, swing_order, pricing_strategy, trading_mode, max_contract_price)
    if res is None:
        st.warning(f"⚠️ تعذر جلب بيانات كافية للرمز ({ticker}) على الإطار ({interval}).")
        if err_detail:
            st.caption(f"تفاصيل تقنية: {err_detail}")
    else:
        df = res["df"]
        g = res["greeks"]
        live_opt = res["live_opt"]

        st.info(
            f"**الأصل:** {ticker} ({interval})  |  **الاتجاه:** {res['trend']}  |  "
            f"**تقييم التوافق:** {res['score']}/5 ⭐  |  **MACD:** {res['macd_status']}  |  **البولنجر:** {res['bb_status']}"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("السعر الحالي", f"${res['current_price']:.2f}", f"ATR: ${res['current_atr']:.2f}")
        c2.metric("الهدف الأول (1.272)", f"${res['target_1']:.2f}", f"متوقع: {res['date_t1']}")
        c3.metric("الهدف الثاني (1.618)", f"${res['target_2']:.2f}", f"متوقع: {res['date_t2']}")
        c4.metric("وقف الخسارة (للسهم)", f"${res['stop_loss']:.2f}", f"R:R = 1 : {res['rr_ratio']:.2f}")

        # اختيار السعر المعروض: إذا كان السعر الحي ضمن الميزانية نعرضه، وإلا نعرض السعر النظري المحسوب للسترايك الاقتصادي
        if "رخيصة" in pricing_strategy:
            if live_opt and 0.10 < live_opt["mid"] <= max_contract_price:
                est_premium = live_opt["mid"]
            else:
                est_premium = g["premium"]
        else:
            est_premium = live_opt["mid"] if (live_opt and live_opt["mid"] > 0.1) else g["premium"]

        contract_cost = est_premium * 100.0
        max_risk_dollars = portfolio_size * (risk_pct / 100.0)
        risk_per_contract = contract_cost * (contract_sl_pct / 100.0)
        suggested_contracts = max(1, int(max_risk_dollars // risk_per_contract)) if risk_per_contract > 0 else 1

        st.markdown("### 🎯 بطاقة العقد المقترح واليونانيات")
        g1, g2, g3, g4, g5 = st.columns(5)

        premium_badge = f"🟢 ضمن الميزانية (<${max_contract_price:.1f})" if est_premium <= max_contract_price else "⚠️ أعلى من السقف"

        g1.metric("نوع العقد والسترايك", f"{res['option_type'].split()[0]} ${res['strike_rec']}", f"DTE: (ينتهي بعد {res['recommended_dte']} أيام في: {res['expiry_str']})")
        g2.metric("سعر العقد التقديري", f"${est_premium:.2f}", f"${contract_cost:.0f}/عقد ({premium_badge})")
        g3.metric("الدلتا (Delta Δ)", f"{g['delta']:.2f}", f"الجاما: {g['gamma']:.4f}")
        g4.metric("الثيتا (Theta Θ)", f"-${abs(g['theta_daily'])*100:.2f}/يوم", f"IV: {g['iv_pct']:.1f}%")
        g5.metric("حجم الصفقة المقترح", f"{suggested_contracts} عقد", f"مخاطرة: ${max_risk_dollars:.0f}")

        if live_opt:
            spread_warn = "⚠️ سبريد واسع - استخدم Limit Order" if live_opt["spread_pct"] > 8 else "✅ سيولة جيدة"
            st.caption(
                f"📡 **بيانات السلسلة الحية (أقرب عقد متاح {live_opt['expiry']} | Strike ${live_opt['strike']}):** "
                f"Bid: ${live_opt['bid']:.2f} | Ask: ${live_opt['ask']:.2f} | السبريد: {live_opt['spread_pct']:.1f}% ({spread_warn}) | "
                f"الحجم: {live_opt['volume']} | العقود المفتوحة (OI): {live_opt['oi']}"
            )

        if st.button(f"📌 تسجيل صفقة {ticker} في سجل الأداء"):
            opt_dir = "CALL" if res["is_bullish"] else "PUT"
            log_new_trade(
                ticker, trading_mode.split()[0], opt_dir, res["strike_rec"],
                res["expiry_str"], est_premium, res["stop_loss"], res["target_1"], suggested_contracts
            )
            st.success("تم حفظ الصفقة بكامل تفاصيلها في تبويب سجل الصفقات!")

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.60, 0.20, 0.20],
            subplot_titles=(f"هيكل السعر ومستويات فيبوناتشي لـ {ticker}", "مؤشر القوة النسبية RSI (14)", "مؤشر الزخم MACD")
        )
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="السعر"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(56,189,248,0.4)', width=1, dash='dash'), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(56,189,248,0.4)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(56,189,248,0.05)', name="BB Lower"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#38BDF8', width=1.4), name="EMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='#F59E0B', width=1.8), name="EMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Swing_High'], mode='markers', marker=dict(color='#EF4444', size=8, symbol='triangle-down'), name="قمة"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Swing_Low'], mode='markers', marker=dict(color='#10B981', size=8, symbol='triangle-up'), name="قاع"), row=1, col=1)

        fig.add_hline(y=res['target_1'], line_dash="dash", line_color="#10B981", annotation_text=f"هدف 1: ${res['target_1']:.2f}", row=1, col=1)
        fig.add_hline(y=res['target_2'], line_dash="dot", line_color="#34D399", annotation_text=f"هدف 2: ${res['target_2']:.2f}", row=1, col=1)
        fig.add_hline(y=res['stop_loss'], line_dash="dash", line_color="#EF4444", annotation_text=f"وقف: ${res['stop_loss']:.2f}", row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#A78BFA', width=1.5), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#64748B", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#64748B", row=2, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#38BDF8', width=1.3), name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#F59E0B', width=1.3), name="Signal"), row=3, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=['#10B981' if v >= 0 else '#EF4444' for v in df['MACD_Hist']], name="Hist"), row=3, col=1)

        fig.update_layout(height=820, xaxis_rangeslider_visible=False, margin=dict(t=35, b=20))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🔍 الماسح الآلي للفرص (المضاربة اليومية والسوينق)")
    watchlist = st.multiselect(
        "اختر قائمة الأسهم والمؤشرات للفحص:",
        ["SPX", "SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META", "AMD", "GOOGL", "NFLX"],
        default=["SPX", "SPY", "QQQ", "NVDA", "AAPL", "TSLA", "META"]
    )
    if st.button("🚀 بدء الفحص الشامل", type="primary"):
        rows = []
        prog = st.progress(0)
        for idx, sym in enumerate(watchlist):
            d, _ = analyze_ticker(sym, interval, swing_order, pricing_strategy, trading_mode, max_contract_price)
            if d:
                g = d["greeks"]
                rows.append({
                    "الرمز": sym,
                    "التقييم": f"{d['score']}/5 ⭐",
                    "السعر الحالي": f"${d['current_price']:.2f}",
                    "الاتجاه": d['trend'],
                    "نوع العقد": d['option_type'],
                    "Strike": f"${d['strike_rec']}",
                    "قسط تقديري": f"${g['premium']:.2f}",
                    "Delta Δ": f"{g['delta']:.2f}",
                    "Theta/يوم": f"-${abs(g['theta_daily'])*100:.1f}",
                    "MACD": d['macd_status'],
                    "البولنجر": d['bb_status'],
                    "الهدف 1": f"${d['target_1']:.2f}",
                    "وقف السهم": f"${d['stop_loss']:.2f}",
                    "DTE": f"{d['recommended_dte']} يوم",
                    "R:R": f"1 : {d['rr_ratio']:.2f}"
                })
            prog.progress((idx + 1) / len(watchlist))

        if rows:
            res_df = pd.DataFrame(rows).sort_values(by="التقييم", ascending=False)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
        else:
            st.warning("لم يتم العثور على بيانات كافية.")

with tab3:
    st.subheader("📈 سجل تتبع الصفقات والأداء (Trade Journal)")
    init_trade_log()
    history_df = pd.read_csv(LOG_FILE)
    if not history_df.empty:
        if "Entry_Premium" in history_df.columns and "Max_Premium" in history_df.columns:
            history_df["ROI_%"] = ((history_df["Max_Premium"] - history_df["Entry_Premium"]) / history_df["Entry_Premium"] * 100).round(2)
        edited_df = st.data_editor(history_df, num_rows="dynamic", use_container_width=True, key="journal_editor")
        col_s, col_c = st.columns([1, 1])
        with col_s:
            if st.button("💾 حفظ التعديلات"):
                edited_df.to_csv(LOG_FILE, index=False)
                st.success("تم حفظ السجل بنجاح!")
                st.rerun()
        with col_c:
            if st.button("🗑️ مسح السجل بالكامل"):
                os.remove(LOG_FILE)
                st.warning("تم مسح السجل.")
                st.rerun()
    else:
        st.info("لا توجد صفقات مسجلة حتى الآن. أضف أي فرصة من التبويب الأول لمتابعتها هنا.")
