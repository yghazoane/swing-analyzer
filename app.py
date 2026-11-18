"""
================================================================================
اسم الأداة: محلل العقود والسوينق الاحترافي (Pro Options & Swing Analyzer)
الإصدار: v2.5.4
الوصف: منصة متكاملة للمضاربة اليومية واللحظية (0DTE Scalping) والسوينق:
       - اقتناص التقلبات اللحظية على إطارات 1m/5m (EMA9 + Bollinger Squeeze + Volume Spike).
       - تحديد السعر المستهدف للعقد والانتقاء الذكي باليونانيات (Delta, Theta, IV).
       - محرك تحليل الأسهم القيادية المتحكمة بالسوق (Market Leaders / Mega-Caps)
         وقياس مدى تأثيرها الحقيقي على اتجاه المؤشرات (SPX, SPY, QQQ).
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
    page_title="محلل العقود والسوينق الاحترافي v2.5.4",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تنسيق بصري آمن وخفيف يمنع تداخل القائمة الجانبية وقصّ الأرقام
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, label, button, [data-testid="stMetric"], [data-testid="stAlert"] {
        font-family: 'Cairo', sans-serif !important;
    }

    [data-testid="stMainBlockContainer"],
    [data-testid="stSidebarContent"],
    [data-testid="stAlert"],
    [data-testid="stExpander"] {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: plaintext;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
        width: 0px !important;
        min-width: 0px !important;
        max-width: 0px !important;
        overflow: hidden !important;
        transform: translateX(-100%) !important;
    }

    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 10px;
        padding: 10px 12px !important;
        direction: rtl !important;
        text-align: right !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.85rem !important;
        white-space: normal !important;
        overflow: visible !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.30rem !important;
        font-weight: 700 !important;
        direction: ltr !important;
        text-align: right !important;
        unicode-bidi: isolate;
    }
    [data-testid="stMetricValue"] > div {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: isolate;
    }
    [data-testid="stMetricDelta"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

LOG_FILE = "trade_history.csv"

# أوزان الأسهم القيادية المتحكمة في المؤشرات الأمريكية الرئيسية
LEADER_WEIGHTS = {
    "QQQ": {
        "NVDA": 0.145, "AAPL": 0.140, "MSFT": 0.135, "AMZN": 0.085,
        "META": 0.075, "GOOGL": 0.080, "AVGO": 0.070, "TSLA": 0.055
    },
    "SPX": {
        "NVDA": 0.072, "AAPL": 0.070, "MSFT": 0.066, "AMZN": 0.041,
        "META": 0.028, "GOOGL": 0.042, "BRK-B": 0.018, "JPM": 0.016,
        "XOM": 0.013, "UNH": 0.012
    },
    "SPY": {
        "NVDA": 0.072, "AAPL": 0.070, "MSFT": 0.066, "AMZN": 0.041,
        "META": 0.028, "GOOGL": 0.042, "BRK-B": 0.018, "JPM": 0.016,
        "XOM": 0.013, "UNH": 0.012
    },
    "DEFAULT": {
        "NVDA": 0.22, "AAPL": 0.20, "MSFT": 0.20, "AMZN": 0.14,
        "META": 0.12, "GOOGL": 0.12
    }
}


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


def get_next_valid_option_expiry(base_dt, target_dte, symbol):
    """تحديد أقرب تاريخ انتهاء تداول حقيقي (تجنب عطلة نهاية الأسبوع السبت والأحد)"""
    today_dt = datetime.now()
    ref_dt = max(base_dt, today_dt)
    target_date = (ref_dt + timedelta(days=target_dte)).date()

    daily_exp_symbols = {"SPX", "SPY", "QQQ", "IWM"}
    if symbol in daily_exp_symbols:
        while target_date.weekday() >= 5:
            target_date += timedelta(days=1)
    else:
        while target_date.weekday() != 4:
            target_date += timedelta(days=1)

    actual_dte = max(0, (target_date - today_dt.date()).days)
    return target_date.strftime("%Y-%m-%d"), actual_dte


def add_trading_days(start_date, trading_days):
    current_date = start_date
    added = 0
    while added < trading_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            added += 1
    return current_date


def round_to_strike_step(price):
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

    breakeven = (K + price) if is_call else max(K - price, 0.01)
    d2_be = (np.log(S / breakeven) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    pop_pct = float(norm.cdf(d2_be) * 100.0) if is_call else float(norm.cdf(-d2_be) * 100.0)
    pop_pct = min(max(pop_pct, 5.0), 95.0)

    return {
        "premium": max(float(price), 0.05),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta_daily": float(theta_daily),
        "vega": float(vega),
        "iv_pct": float(sigma * 100),
        "pop_pct": pop_pct,
        "breakeven": float(breakeven)
    }


def score_option_efficiency(premium, delta, gamma, theta_daily, iv_pct, target_price):
    """
    خوارزمية تقييم كفاءة العقد بالاعتماد على اليونانيات (Delta, Gamma, Theta, IV)
    والقرب من السعر المستهدف الذي حدده المستخدم.
    """
    if premium <= 0.05:
        return -999.0
    abs_delta = abs(delta)
    # نسبة تآكل الثيتا اليومي من قيمة العقد
    theta_decay_ratio = abs(theta_daily) / premium
    # مكافأة لقوة الدلتا والجاما وخصم للتآكل السريع أو التضخم المبالغ في IV
    iv_penalty = max(0.0, (iv_pct - 65.0) * 0.005)
    # مكافأة للقرب من السعر المستهدف دون تجاوزه
    price_fit = 1.0 - min(abs(target_price - premium) / max(target_price, 0.5), 1.0)
    efficiency = (abs_delta * 3.0) + (gamma * 15.0) - (theta_decay_ratio * 1.4) - iv_penalty + (price_fit * 0.8)
    return float(efficiency)


def find_smart_greek_strike(S, T_days, r, sigma, is_call, target_price=2.50):
    """
    البحث الذكي عن أفضل سترايك ضمن السعر المستهدف (target_price)
    بناءً على أعلى كفاءة تجمع بين Delta و Gamma وأقل Theta و IV متوازن.
    """
    step = get_strike_step(S)
    base_k = round_to_strike_step(S)
    direction = 1 if is_call else -1

    best_k = base_k
    best_score = -999.0
    found_under_target = False

    # مسح نطاق واسع من السترايكات حول السعر الحالي (من ITM قريب إلى OTM بعيد)
    for offset in range(-8, 125):
        cand_k = base_k + (offset * direction * step)
        if cand_k <= step:
            continue
        m = calculate_bs_metrics(S, cand_k, T_days, r, sigma, is_call)
        prem = m["premium"]
        if prem <= target_price and prem >= 0.15:
            eff = score_option_efficiency(prem, m["delta"], m["gamma"], m["theta_daily"], m["iv_pct"], target_price)
            if not found_under_target or eff > best_score:
                found_under_target = True
                best_score = eff
                best_k = cand_k
        elif not found_under_target:
            # الاحتفاظ بأقرب عقد للسعر المستهدف كخيار احتياطي
            if abs(prem - target_price) < abs(calculate_bs_metrics(S, best_k, T_days, r, sigma, is_call)["premium"] - target_price):
                best_k = cand_k

    return float(best_k)


@st.cache_data(ttl=45)
def fetch_price_history(symbol, period, interval):
    """جلب البيانات التاريخية واللحظية مع توافق رمز مؤشر SPX (^GSPC)"""
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


@st.cache_data(ttl=45)
def fetch_live_option_hint(symbol, target_dte, is_call, target_strike, budget_mode=False, max_budget=3.0):
    """جلب السلسلة الحية للأوبشن مع الانتقاء الذكي المعتمد على Delta و Theta و IV"""
    try:
        candidates = ["^SPX", "^GSPC"] if symbol == "SPX" else [symbol]
        tk = None
        expirations = ()
        for cand in candidates:
            t_cand = yf.Ticker(cand)
            exps = t_cand.options
            if exps:
                tk = t_cand
                expirations = exps
                break
        if not expirations or tk is None:
            return None

        today = datetime.now().date()
        valid_exps = [e for e in expirations if datetime.strptime(e, "%Y-%m-%d").date() >= today]
        if not valid_exps:
            valid_exps = list(expirations)
        best_exp = min(
            valid_exps,
            key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d").date() - today).days - target_dte)
        )
        chain = tk.option_chain(best_exp)
        calls_df = chain.calls if chain.calls is not None else pd.DataFrame()
        puts_df = chain.puts if chain.puts is not None else pd.DataFrame()

        total_call_oi = float(calls_df['openInterest'].fillna(0).sum()) if not calls_df.empty else 0.0
        total_put_oi = float(puts_df['openInterest'].fillna(0).sum()) if not puts_df.empty else 0.0
        total_call_vol = float(calls_df['volume'].fillna(0).sum()) if not calls_df.empty else 0.0
        total_put_vol = float(puts_df['volume'].fillna(0).sum()) if not puts_df.empty else 0.0

        pcr_oi = (total_put_oi / total_call_oi) if total_call_oi > 0 else 1.0
        pcr_vol = (total_put_vol / total_call_vol) if total_call_vol > 0 else pcr_oi

        call_wall = float(calls_df.loc[calls_df['openInterest'].fillna(0).idxmax(), 'strike']) if (not calls_df.empty and total_call_oi > 0) else None
        put_wall = float(puts_df.loc[puts_df['openInterest'].fillna(0).idxmax(), 'strike']) if (not puts_df.empty and total_put_oi > 0) else None

        df_opt = calls_df if is_call else puts_df
        if df_opt is None or df_opt.empty:
            return None

        df_opt = df_opt.copy()
        df_opt['mid_calc'] = np.where(
            (df_opt['bid'] > 0) & (df_opt['ask'] > 0),
            (df_opt['bid'] + df_opt['ask']) / 2.0,
            np.where(df_opt['ask'] > 0, df_opt['ask'] * 0.90, df_opt['lastPrice'])
        )

        if budget_mode:
            affordable = df_opt[(df_opt['mid_calc'] >= 0.15) & (df_opt['mid_calc'] <= max_budget)].copy()
            if not affordable.empty:
                # ترتيب العقود ضمن السعر المحدد بناءً على السيولة والقرب من أفضل سترايك يوناني
                affordable['dist_score'] = (affordable['strike'] - target_strike).abs()
                affordable['liq_bonus'] = np.log1p(affordable['volume'].fillna(0) + affordable['openInterest'].fillna(0))
                affordable['rank'] = affordable['dist_score'] - affordable['liq_bonus'] * 0.5
                row = affordable.sort_values('rank').iloc[0]
            else:
                idx = (df_opt['strike'] - target_strike).abs().idxmin()
                row = df_opt.loc[idx]
        else:
            idx = (df_opt['strike'] - target_strike).abs().idxmin()
            row = df_opt.loc[idx]

        bid = float(row.get('bid', 0) or 0)
        ask = float(row.get('ask', 0) or 0)
        last_p = float(row.get('lastPrice', 0) or 0)
        
        # الذكاء الاصطناعي لتسعير عقود 0DTE و OTM البعيدة:
        # إذا كان العقد 0DTE أو بعيد عن السعر (OTM) وكان الـ Bid صفراً والـ Ask منخفضاً (<= 0.10) أو غير متوفر، 
        # فإن سعر السوق الحقيقي للتنفيذ الفوري هو 0.05 (مطابق لمنصة سهم تماماً).
        strike_val = float(row.get('strike', 0))
        is_otm_contract = (is_call and strike_val > target_strike) or (not is_call and strike_val < target_strike)
        
        if bid == 0 and (ask <= 0.10 or last_p > 0.50):
            mid = 0.05 if ask <= 0.10 else min(ask if ask > 0 else 0.05, 0.05)
            has_live_quote = True
        elif bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            has_live_quote = True
        elif ask > 0:
            mid = ask
            has_live_quote = True
        elif bid > 0:
            mid = bid
            has_live_quote = True
        else:
            mid = 0.05 if is_otm_contract else last_p
            has_live_quote = False

        spread_pct = ((ask - bid) / ask * 100) if (bid > 0 and ask > 0) else 0.0
        return {
            "expiry": best_exp,
            "strike": float(row['strike']),
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "has_live_quote": has_live_quote,
            "spread_pct": spread_pct,
            "volume": int(row.get('volume', 0) or 0),
            "oi": int(row.get('openInterest', 0) or 0),
            "iv": float(row.get('impliedVolatility', 0) or 0) * 100,
            "pcr_oi": round(pcr_oi, 2),
            "pcr_vol": round(pcr_vol, 2),
            "call_wall": call_wall,
            "put_wall": put_wall
        }
    except Exception:
        return None


@st.cache_data(ttl=45)
def analyze_market_leaders(target_symbol):
    """
    تحليل الأسهم القيادية المتحكمة بالسوق (Heavyweight Market Leaders)
    وقياس مدى تأثيرها اللحظي واليومي على حركة المؤشر (SPX, SPY, QQQ).
    """
    weights_map = LEADER_WEIGHTS.get(target_symbol, LEADER_WEIGHTS["DEFAULT"])
    tickers_list = list(weights_map.keys())
    try:
        raw = yf.download(tickers_list, period="5d", interval="1d", progress=False, group_by="ticker")
        intra = yf.download(tickers_list, period="1d", interval="5m", progress=False, group_by="ticker")
    except Exception:
        return None

    leader_rows = []
    weighted_day_impact = 0.0
    weighted_1h_impact = 0.0
    total_w = sum(weights_map.values())
    bullish_count = 0
    bearish_count = 0

    for sym, w in weights_map.items():
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df_d = raw[sym].dropna()
            else:
                df_d = raw.dropna()
            if df_d.empty or len(df_d) < 2:
                continue

            prev_close = float(df_d['Close'].iloc[-2])
            curr_price = float(df_d['Close'].iloc[-1])
            day_chg_pct = ((curr_price - prev_close) / prev_close) * 100.0

            # حساب الزخم اللحظي لآخر ساعة (12 شمعة 5 دقائق)
            intraday_1h_pct = day_chg_pct
            try:
                df_i = intra[sym].dropna() if isinstance(intra.columns, pd.MultiIndex) else intra.dropna()
                if len(df_i) >= 12:
                    p_1h = float(df_i['Close'].iloc[-12])
                    p_now = float(df_i['Close'].iloc[-1])
                    intraday_1h_pct = ((p_now - p_1h) / p_1h) * 100.0
                    curr_price = p_now
                    day_chg_pct = ((curr_price - prev_close) / prev_close) * 100.0
            except Exception:
                pass

            norm_w = w / total_w
            impact_pts = day_chg_pct * norm_w
            weighted_day_impact += impact_pts
            weighted_1h_impact += intraday_1h_pct * norm_w

            if day_chg_pct > 0.15:
                bullish_count += 1
                state = "صاعد بقوة 🟢" if day_chg_pct > 1.0 else "إيجابي 🟢"
            elif day_chg_pct < -0.15:
                bearish_count += 1
                state = "هابط بقوة 🔴" if day_chg_pct < -1.0 else "سلبي 🔴"
            else:
                state = "محايد ⚪"

            leader_rows.append({
                "السهم القيادي": sym,
                "الوزن النسبي": f"{norm_w*100:.1f}%",
                "السعر": f"${curr_price:.2f}",
                "تغير اليوم %": round(day_chg_pct, 2),
                "زخم آخر ساعة %": round(intraday_1h_pct, 2),
                "التأثير على المؤشر": round(impact_pts, 3),
                "الحالة": state
            })
        except Exception:
            continue

    if not leader_rows:
        return None

    df_leaders = pd.DataFrame(leader_rows).sort_values(by="التأثير على المؤشر", key=abs, ascending=False)
    total_active = len(leader_rows)
    if weighted_day_impact > 0.25 and bullish_count >= total_active * 0.6:
        verdict = f"دعم قيادي صعودي قوي ({bullish_count}/{total_active} قياديات خضراء • زخم +{weighted_day_impact:.2f}%) 🟢"
    elif weighted_day_impact < -0.25 and bearish_count >= total_active * 0.6:
        verdict = f"ضغط بيعي قيادي قوي ({bearish_count}/{total_active} قياديات حمراء • زخم {weighted_day_impact:.2f}%) 🔴"
    elif abs(weighted_day_impact) <= 0.25:
        verdict = f"تباين قيادي / تذبذب أفقي (صافي التأثير {weighted_day_impact:+.2f}%) ⚖️"
    else:
        verdict = f"حركة يقودها سهم أو سهمان فقط (تباعد قيادي • صافي {weighted_day_impact:+.2f}%) ⚠️"

    return {
        "df": df_leaders,
        "weighted_day_impact": weighted_day_impact,
        "weighted_1h_impact": weighted_1h_impact,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "total_active": total_active,
        "verdict": verdict
    }


def analyze_ticker(symbol, interval, swing_order, pricing_strategy, trading_mode, target_contract_price=3.0):
    try:
        is_scalp_0dte = "0DTE" in trading_mode or "اللحظية" in trading_mode
        period_map = {"1m": "2d", "5m": "5d", "15m": "1mo", "1h": "3mo", "1d": "1y"}
        period = period_map.get(interval, "1y")
        df, err_msg = fetch_price_history(symbol, period, interval)
        if df is None or df.empty or len(df) < 30:
            return None, err_msg or "البيانات الواردة من المصدر فارغة أو غير مكتملة."

        df = df.copy()

        # المتوسطات المتحركة (بما فيها EMA 9 لاقتناص التقلبات اللحظية)
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

        # مؤشر RSI (14) بتنعيم وايلدر القياسي (Wilder's Smoothing)
        delta_s = df['Close'].diff()
        gain = delta_s.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss = (-delta_s.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
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
        bars_per_year = {"1m": 252 * 390, "5m": 252 * 78, "15m": 252 * 26, "1h": 252 * 6.5, "1d": 252}.get(interval, 252)
        est_vol = float(log_returns.rolling(20).std().iloc[-1] * np.sqrt(bars_per_year))
        if np.isnan(est_vol) or est_vol <= 0.05:
            est_vol = 0.25

        # قمم وقيعان السوينق الهيكلية
        eff_order = max(3, swing_order // 2) if is_scalp_0dte else swing_order
        high_idx = argrelextrema(df['High'].values, np.greater_equal, order=eff_order)[0]
        low_idx = argrelextrema(df['Low'].values, np.less_equal, order=eff_order)[0]
        df['Swing_High'] = np.nan
        df['Swing_Low'] = np.nan
        df.iloc[high_idx, df.columns.get_loc('Swing_High')] = df['High'].iloc[high_idx]
        df.iloc[low_idx, df.columns.get_loc('Swing_Low')] = df['Low'].iloc[low_idx]

        last_highs = df.dropna(subset=['Swing_High'])
        last_lows = df.dropna(subset=['Swing_Low'])

        current_price = float(df['Close'].iloc[-1])
        current_atr = float(df['ATR'].iloc[-1])
        ema9_val = float(df['EMA9'].iloc[-1])
        ema20_val = float(df['EMA20'].iloc[-1])
        ema50_val = float(df['EMA50'].iloc[-1])
        current_rsi = float(df['RSI'].iloc[-1])
        macd_val = float(df['MACD'].iloc[-1])
        macd_sig = float(df['MACD_Signal'].iloc[-1])
        bb_w = float(df['BB_Width'].iloc[-1])
        bb_w_avg = float(df['BB_Width'].rolling(50, min_periods=10).mean().iloc[-1])

        # تحديد الاتجاه: في وضع 0DTE اللحظي نعتمد على تقاطع EMA9 مع EMA20 والزخم الفوري
        if is_scalp_0dte:
            is_bullish = (current_price > ema20_val) and (ema9_val >= ema20_val or macd_val > macd_sig)
        else:
            is_bullish = current_price > ema50_val

        trend = "صاعد (Bullish) 🟢" if is_bullish else "هابط (Bearish) 🔴"
        option_type = "CALL 🟢 (شراء)" if is_bullish else "PUT 🔴 (بيع)"

        # تقييم جودة الإشارة (Confluence Score من 5)
        score = 0
        if (is_bullish and current_price > ema9_val and current_price > ema20_val) or (not is_bullish and current_price < ema9_val and current_price < ema20_val):
            score += 1
        if (is_bullish and macd_val > macd_sig) or (not is_bullish and macd_val < macd_sig):
            score += 1
        if (is_bullish and 45 <= current_rsi <= 72) or (not is_bullish and 28 <= current_rsi <= 55):
            score += 1
        is_squeeze = (not np.isnan(bb_w)) and (not np.isnan(bb_w_avg)) and (bb_w < bb_w_avg * 0.85)
        if is_squeeze:
            score += 1

        bb_up_val = float(df['BB_Upper'].iloc[-1]) if not np.isnan(df['BB_Upper'].iloc[-1]) else current_price
        bb_low_val = float(df['BB_Lower'].iloc[-1]) if not np.isnan(df['BB_Lower'].iloc[-1]) else current_price
        recent_res = float(last_highs['Swing_High'].iloc[-1]) if len(last_highs) >= 1 else bb_up_val
        recent_sup = float(last_lows['Swing_Low'].iloc[-1]) if len(last_lows) >= 1 else bb_low_val

        # التحقق من حجم التداول وانفجار السيولة اللحظي (Volume Spike Ratio)
        vol_ok = True
        vol_ratio = 1.0
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            avg_vol = float(df['Volume'].rolling(20, min_periods=5).mean().iloc[-1])
            curr_vol = float(df['Volume'].iloc[-1])
            if avg_vol > 0:
                vol_ratio = curr_vol / avg_vol
                vol_ok = vol_ratio >= (1.25 if is_scalp_0dte else 1.15)

        if is_bullish:
            broke_level = (current_price > recent_res) or (current_price > bb_up_val)
            momentum_ok = (macd_val > macd_sig) and (current_price > ema9_val)
        else:
            broke_level = (current_price < recent_sup) or (current_price < bb_low_val)
            momentum_ok = (macd_val < macd_sig) and (current_price < ema9_val)

        if broke_level and momentum_ok and vol_ok:
            breakout_status = f"انفجار سعري مؤكد ({interval} • سيولة {vol_ratio:.1f}x) 🟢"
            score = min(5, score + 1)
        elif broke_level and (not momentum_ok or not vol_ok):
            breakout_status = f"فخ صانع سوق / كسر وهمي ({interval} • سيولة {vol_ratio:.1f}x) ⚠️"
        else:
            breakout_status = f"ترقب انفجار التقلب ({interval}) ⏳"

        # حساب الأهداف ووقف الخسارة (أهداف خاطفة في 0DTE مقابل فيبوناتشي في السوينق)
        if is_scalp_0dte:
            wave_bars = 3
            est_days_t1 = 0
            est_days_t2 = 0
            if is_bullish:
                target_1 = current_price + current_atr * 1.1
                target_2 = current_price + current_atr * 2.0
                stop_loss = min(ema20_val - current_atr * 0.3, current_price - current_atr * 0.85)
            else:
                target_1 = current_price - current_atr * 1.1
                target_2 = current_price - current_atr * 2.0
                stop_loss = max(ema20_val + current_atr * 0.3, current_price + current_atr * 0.85)
        elif len(last_highs) >= 1 and len(last_lows) >= 1:
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

        # تحديد الصلاحية الزمنية (DTE)
        if is_scalp_0dte:
            est_days_t1 = 0
            est_days_t2 = 0
            recommended_dte = 0
        elif "يومي" in trading_mode or interval in ["1m", "5m", "15m", "1h"]:
            est_days_t1 = 1 if interval in ["1m", "5m", "15m"] else 2
            est_days_t2 = 2 if interval in ["1m", "5m"] else (3 if interval == "15m" else 5)
            recommended_dte = 1 if interval in ["1m", "5m"] else (3 if interval == "15m" else 7)
        else:
            est_days_t1 = max(3, int(wave_bars * 0.8))
            est_days_t2 = max(6, int(wave_bars * 1.3))
            recommended_dte = min(60, max(28, est_days_t2 + 21))

        is_target_price_mode = "السعر المستهدف" in pricing_strategy or "رخيصة" in pricing_strategy

        # هندسة السترايك الذكية المعتمدة على السعر المراد واليونانيات (Delta / Theta / IV)
        eff_t_days = 0.25 if recommended_dte == 0 else recommended_dte
        if is_target_price_mode:
            if not is_scalp_0dte and recommended_dte > 5:
                recommended_dte = 1 if interval in ["1m", "5m"] else (2 if interval == "15m" else 5)
                eff_t_days = recommended_dte
            strike_rec = find_smart_greek_strike(
                S=current_price,
                T_days=eff_t_days,
                r=0.045,
                sigma=est_vol,
                is_call=is_bullish,
                target_price=target_contract_price
            )
        elif "هجومي" in pricing_strategy:
            raw_strike = current_price + (current_atr * 0.9 if is_bullish else -current_atr * 0.9)
            strike_rec = round_to_strike_step(raw_strike)
        elif "تحوطي" in pricing_strategy:
            raw_strike = current_price - (current_atr * 0.85 if is_bullish else -current_atr * 0.85)
            strike_rec = round_to_strike_step(raw_strike)
        else:
            strike_rec = round_to_strike_step(current_price)

        greeks = calculate_bs_metrics(
            S=current_price, K=strike_rec, T_days=eff_t_days,
            r=0.045, sigma=est_vol, is_call=is_bullish
        )

        live_opt = fetch_live_option_hint(
            symbol, recommended_dte, is_bullish, strike_rec,
            budget_mode=is_target_price_mode, max_budget=target_contract_price
        )

        spread_status = "غير متوفر (تسعير نظري)"
        oi_wall_note = "بانتظار بيانات السلسلة الحية"
        pcr_note = "متعادل"
        if live_opt:
            sp = live_opt.get("spread_pct", 0.0)
            if not live_opt.get("has_live_quote", True):
                spread_status = "عروض Bid/Ask غير مكتملة - اعتماد التسعير النظري ⚠️"
            elif sp > 10.0:
                spread_status = f"سبريد خطر ({sp:.1f}%) - تجنب أمر السوق 🔴"
            elif sp > 6.0:
                spread_status = f"سبريد متوسط ({sp:.1f}%) - استخدم Limit ⚠️"
            else:
                spread_status = f"سبريد ممتاز ({sp:.1f}%) - سيولة عالية 🟢"

            cw = live_opt.get("call_wall")
            pw = live_opt.get("put_wall")
            pcr_v = live_opt.get("pcr_oi", 1.0)
            if pcr_v > 1.15:
                pcr_note = f"PCR {pcr_v} (تمركز تحوط/هبوط 🔴)"
            elif pcr_v < 0.80:
                pcr_note = f"PCR {pcr_v} (تمركز صعودي 🟢)"
            else:
                pcr_note = f"PCR {pcr_v} (متوازن ⚖️)"

            wall_parts = []
            if cw:
                dist_cw = ((cw - current_price) / current_price) * 100
                wall_parts.append(f"جدار Call: ${cw:.0f} ({dist_cw:+.1f}%)")
            if pw:
                dist_pw = ((pw - current_price) / current_price) * 100
                wall_parts.append(f"جدار Put: ${pw:.0f} ({dist_pw:+.1f}%)")
            if wall_parts:
                oi_wall_note = " | ".join(wall_parts)

        if live_opt:
            live_iv = (live_opt.get("iv", 0.0) / 100.0)
            effective_sigma = live_iv if live_iv > 0.05 else est_vol
            if is_target_price_mode and 0.15 <= live_opt["mid"] <= target_contract_price:
                strike_rec = live_opt["strike"]
            elif not is_target_price_mode:
                strike_rec = live_opt["strike"]
            greeks = calculate_bs_metrics(
                S=current_price, K=strike_rec, T_days=max(recommended_dte, 0.25),
                r=0.045, sigma=effective_sigma, is_call=is_bullish
            )

        risk_per_share = abs(current_price - stop_loss)
        reward_t1 = abs(target_1 - current_price)
        rr_ratio = reward_t1 / risk_per_share if risk_per_share > 0 else 0
        if rr_ratio >= 1.8:
            score = min(5, score + 1)

        last_dt = df.index[-1].to_pydatetime() if hasattr(df.index[-1], 'to_pydatetime') else datetime.now()
        if getattr(last_dt, "tzinfo", None) is not None:
            last_dt = last_dt.replace(tzinfo=None)
        ref_base_dt = max(last_dt, datetime.now())
        date_t1 = "خلال الجلسة (0DTE)" if est_days_t1 == 0 else add_trading_days(ref_base_dt, est_days_t1).strftime("%Y-%m-%d")
        date_t2 = "خلال الجلسة (0DTE)" if est_days_t2 == 0 else add_trading_days(ref_base_dt, est_days_t2).strftime("%Y-%m-%d")

        today_date = datetime.now().date()
        if live_opt and live_opt.get("expiry"):
            live_exp_date = datetime.strptime(live_opt["expiry"], "%Y-%m-%d").date()
            if live_exp_date >= today_date and live_exp_date.weekday() < 5:
                expiry_str = live_opt["expiry"]
                recommended_dte = max(0, (live_exp_date - today_date).days)
            else:
                expiry_str, recommended_dte = get_next_valid_option_expiry(last_dt, recommended_dte, symbol)
        else:
            expiry_str, recommended_dte = get_next_valid_option_expiry(last_dt, recommended_dte, symbol)

        is_weekend = today_date.weekday() >= 5
        market_note = "السوق مغلق حالياً (عطلة نهاية الأسبوع) • تم ضبط الانتهاء لأقرب جلسة قادمة" if is_weekend else "جلسة التداول نشطة / محدثة"

        # الأولوية المطلقة للسعر الحي من السلسلة الحية (مطابق لمنصة سهم) متى ما توفر العقد، وإلا اللجوء للتسعير النظري
        if live_opt and live_opt.get("mid", 0) > 0:
            display_premium = live_opt["mid"]
        else:
            display_premium = greeks["premium"]

        greek_eff = score_option_efficiency(
            display_premium, greeks["delta"], greeks["gamma"],
            greeks["theta_daily"], greeks["iv_pct"], target_contract_price
        )

        return {
            "display_premium": float(display_premium),
            "greek_eff": greek_eff,
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
            "bb_status": "انفجار وشيك ⚡ (Squeeze)" if is_squeeze else "نطاق طبيعي 📊",
            "breakout_status": breakout_status,
            "market_note": market_note,
            "spread_status": spread_status,
            "oi_wall_note": oi_wall_note,
            "pcr_note": pcr_note,
            "vol_ratio": vol_ratio
        }, ""
    except Exception as e:
        return None, str(e)


# --- واجهة المستخدم ---
st.title("📈 محلل العقود والسوينق الاحترافي (v2.5.4)")
st.markdown("منصة شاملة للمضاربة اللحظية (0DTE Scalping) والسوينق، مع الانتقاء الذكي باليونانيات ورصد تأثير الأسهم القيادية على المؤشرات.")

# القائمة الجانبية
st.sidebar.header("⚙️ إعدادات التداول")
trading_mode = st.sidebar.radio(
    "أسلوب المضاربة:",
    [
        "⚡ اقتناص التقلبات اللحظية (0DTE Scalping)",
        "🔥 مضاربة يومية / قصيرة (Intraday | 1-7 DTE)",
        "🌊 سوينق (Swing | 21-45 DTE)"
    ]
)

if "0DTE" in trading_mode:
    default_tf_idx = 1  # 5m
elif "يومية" in trading_mode:
    default_tf_idx = 2  # 15m
else:
    default_tf_idx = 4  # 1d

interval = st.sidebar.selectbox("الإطار الزمني:", ["1m", "5m", "15m", "1h", "1d"], index=default_tf_idx)
ticker = st.sidebar.selectbox("الأصل المالي:", ["SPX", "SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META", "AMD", "GOOGL", "NFLX"])
swing_order = st.sidebar.slider("حساسية القمم والقيعان:", min_value=3, max_value=15, value=7)

if st.sidebar.button("🔄 تحديث البيانات ومسح الكاش"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🎯 السعر المستهدف واليونانيات")
pricing_strategy = st.sidebar.selectbox(
    "آلية اختيار العقد:",
    [
        "🎯 السعر المستهدف الذكي (أفضل Delta/Theta ضمن ميزانيتك)",
        "⚖️ متوازن (ATM | Delta ~0.50)",
        "⚡ هجومي (OTM متوسط | Delta ~0.35)",
        "🛡️ تحوطي آمن (ITM | Delta ~0.65)"
    ]
)
target_contract_price = st.sidebar.number_input(
    "السعر المراد للعقد ($):",
    min_value=0.05, max_value=50.0, value=0.05, step=0.05,
    help="حدد السعر الذي ترغب به للعقد (مثلاً $0.05 أو $1.50)."
)

use_manual_price = st.sidebar.checkbox("🔧 تفعيل السعر اليدوي المباشر (مطابق لمنصة سهم تماماً)", value=True)
manual_contract_price = st.sidebar.number_input(
    "سعر العقد اليدوي ($):",
    min_value=0.01, max_value=50.0, value=0.05, step=0.05,
    help="أدخل السعر الظاهر لديك في منصة سهم (مثل 0.05) لتوحيد الحسابات تماماً."
)
portfolio_size = st.sidebar.number_input("حجم المحفظة ($):", min_value=1000, value=10000, step=1000)
risk_pct = st.sidebar.number_input("المخاطرة القصوى للصفقة (%):", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
contract_sl_pct = st.sidebar.slider("نسبة وقف خسارة العقد (% من القسط):", min_value=20, max_value=60, value=35, step=5)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 التحليل التفصيلي وبطاقة العقد",
    "👑 الأسهم القيادية وتأثيرها على المؤشر",
    "🔍 الماسح الآلي للفرص (Screener)",
    "📈 سجل الصفقات والأداء (Journal)"
])

with tab1:
    res, err_detail = analyze_ticker(ticker, interval, swing_order, pricing_strategy, trading_mode, target_contract_price)
    if res is None:
        st.warning(f"⚠️ تعذر جلب بيانات كافية للرمز ({ticker}) على الإطار ({interval}).")
        if err_detail:
            st.caption(f"تفاصيل تقنية: {err_detail}")
    else:
        df = res["df"]
        g = res["greeks"]
        live_opt = res["live_opt"]

        st.info(
            f"**الأصل:** {ticker} ({interval}) &nbsp;|&nbsp; "
            f"**الاتجاه:** {res['trend']} &nbsp;|&nbsp; "
            f"**حالة التقلب:** {res['breakout_status']} &nbsp;|&nbsp; "
            f"**التوافق:** {res['score']} من 5 ⭐ &nbsp;|&nbsp; "
            f"**الزخم (MACD):** {res['macd_status']} &nbsp;|&nbsp; "
            f"**البولنجر:** {res['bb_status']}"
        )
        if "مغلق" in res.get("market_note", ""):
            st.warning(f"🕒 **تنبيه توقيت السوق:** {res['market_note']} ({res['expiry_str']}).")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("السعر الحالي", f"${res['current_price']:.2f}", f"ATR: ${res['current_atr']:.2f}")
        c2.metric("الهدف الأول", f"${res['target_1']:.2f}", f"التوقيت: {res['date_t1']}")
        c3.metric("الهدف الثاني", f"${res['target_2']:.2f}", f"التوقيت: {res['date_t2']}")
        c4.metric("وقف الخسارة (للسهم)", f"${res['stop_loss']:.2f}", f"R:R = 1 : {res['rr_ratio']:.2f}")

        # إذا تم تفعيل السعر اليدوي، نعتمد سعر منصة سهم المباشر الذي أدخله المستخدم
        est_premium = manual_contract_price if use_manual_price else res["display_premium"]
        contract_cost = est_premium * 100.0
        max_risk_dollars = portfolio_size * (risk_pct / 100.0)
        risk_per_contract = contract_cost * (contract_sl_pct / 100.0)
        suggested_contracts = max(1, int(max_risk_dollars // risk_per_contract)) if risk_per_contract > 0 else 1

        st.markdown("### 🎯 بطاقة العقد المقترح واليونانيات")
        g1, g2, g3 = st.columns(3)
        g4, g5, g6 = st.columns(3)

        premium_badge = f"مطابق للسعر المراد (≤ ${target_contract_price:.2f}) 🟢" if est_premium <= target_contract_price else "أعلى من السعر المراد ⚠️"

        if res['recommended_dte'] == 0:
            dte_label = f"ينتهي اليوم (0DTE) • {res['expiry_str']}"
        else:
            day_word = "يوم" if res['recommended_dte'] in [1, 2] or res['recommended_dte'] > 10 else "أيام"
            dte_label = f"ينتهي بعد {res['recommended_dte']} {day_word} • {res['expiry_str']}"

        opt_dir_ar = "شراء CALL 🟢" if res['is_bullish'] else "بيع PUT 🔴"
        g1.metric(f"العقد المقترح ({opt_dir_ar})", f"{res['option_type'].split()[0]} ${res['strike_rec']}", dte_label)
        g2.metric("سعر العقد التقديري", f"${est_premium:.2f}", f"التكلفة ${contract_cost:.0f} • {premium_badge}")
        g3.metric("حجم الصفقة المقترح", f"{suggested_contracts} عقد", f"احتمالية الربح PoP: {g['pop_pct']:.0f}%")
        g4.metric("الدلتا (Delta Δ)", f"{g['delta']:.2f}", f"الجاما (Gamma): {g['gamma']:.4f}")
        g5.metric("الثيتا (Theta Θ)", f"-${abs(g['theta_daily'])*100:.2f} / يوم", f"التقلب الضمني IV: {g['iv_pct']:.1f}%")
        g6.metric("نقطة التعادل (Breakeven)", f"${g['breakeven']:.2f}", f"كفاءة اليونانيات: {res['greek_eff']:.2f}")

        st.info(
            f"🏛️ **النبض المؤسسي (OI & PCR):** {res['pcr_note']} &nbsp;|&nbsp; "
            f"**جدران السترايكات:** {res['oi_wall_note']} &nbsp;|&nbsp; "
            f"**فلتر السبريد اللحظي:** {res['spread_status']}"
        )

        # عرض نبض الأسهم القيادية مباشرة أسفل البطاقة عند تحليل المؤشرات
        leaders_quick = analyze_market_leaders(ticker if ticker in ["SPX", "SPY", "QQQ"] else "SPX")
        if leaders_quick:
            st.success(f"👑 **نبض الأسهم القيادية المتحكمة بالسوق ({ticker if ticker in ['SPX', 'SPY', 'QQQ'] else 'SPX'}):** {leaders_quick['verdict']}")

        with st.expander("🧭 خريطة إدارة الصفقة المفتوحة (جني الأرباح الجزئي وتآكل الثيتا)", expanded=False):
            move_t1 = abs(res['target_1'] - res['current_price'])
            move_t2 = abs(res['target_2'] - res['current_price'])
            est_gain_t1 = abs(g['delta']) * move_t1 + 0.5 * g['gamma'] * (move_t1 ** 2)
            est_gain_t2 = abs(g['delta']) * move_t2 + 0.5 * g['gamma'] * (move_t2 ** 2)
            opt_price_t1 = est_premium + est_gain_t1
            opt_price_t2 = est_premium + est_gain_t2
            opt_sl_price = est_premium * (1.0 - contract_sl_pct / 100.0)
            theta_2h_loss = (abs(g['theta_daily']) * 100.0) * (2.0 / 6.5)
            half_contracts = max(1, suggested_contracts // 2)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "1️⃣ جني ربح جزئي (الهدف 1)",
                f"${opt_price_t1:.2f} (+{((opt_price_t1/est_premium)-1)*100:.0f}%)",
                f"بيع {half_contracts} عقد وتأمين الباقي على الدخول"
            )
            m2.metric(
                "2️⃣ الهدف النهائي (الهدف 2)",
                f"${opt_price_t2:.2f} (+{((opt_price_t2/est_premium)-1)*100:.0f}%)",
                f"إغلاق المتبقي عند ${res['target_2']:.2f}"
            )
            m3.metric(
                "3️⃣ وقف خسارة العقد الصارم",
                f"${opt_sl_price:.2f} (-{contract_sl_pct}%)",
                f"أو كسر السهم لـ ${res['stop_loss']:.2f}"
            )
            m4.metric(
                "4️⃣ تآكل الوقت (كل ساعتين تذبذب)",
                f"-${theta_2h_loss:.2f}/عقد",
                f"نقطة تعادل السهم: ${g['breakeven']:.2f}"
            )

        if live_opt:
            spread_warn = "⚠️ عروض غير مكتملة" if not live_opt.get("has_live_quote", True) else ("⚠️ سبريد واسع - استخدم Limit Order" if live_opt["spread_pct"] > 8 else "✅ سيولة جيدة")
            st.caption(
                f"📡 **بيانات السلسلة الحية (أقرب عقد متاح {live_opt['expiry']} | Strike ${live_opt['strike']}):** "
                f"Bid: ${live_opt['bid']:.2f} | Ask: ${live_opt['ask']:.2f} | السبريد: {live_opt['spread_pct']:.1f}% ({spread_warn}) | "
                f"الحجم: {live_opt['volume']} | العقود المفتوحة (OI): {live_opt['oi']}"
            )

        if st.button(f"📌 تسجيل صفقة {ticker} في سجل الأداء"):
            opt_dir = "CALL" if res["is_bullish"] else "PUT"
            log_new_trade(
                ticker, trading_mode.split()[1], opt_dir, res["strike_rec"],
                res["expiry_str"], est_premium, res["stop_loss"], res["target_1"], suggested_contracts
            )
            st.success("تم حفظ الصفقة بكامل تفاصيلها في تبويب سجل الصفقات!")

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.60, 0.20, 0.20],
            subplot_titles=(f"هيكل السعر ومستويات التقلب لـ {ticker}", "مؤشر القوة النسبية RSI (14)", "مؤشر الزخم MACD")
        )
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="السعر"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(56,189,248,0.4)', width=1, dash='dash'), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(56,189,248,0.4)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(56,189,248,0.05)', name="BB Lower"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], line=dict(color='#10B981', width=1.2, dash='dot'), name="EMA 9 (لحظي)"), row=1, col=1)
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
    st.subheader("👑 رادار الأسهم القيادية المتحكمة بالسوق وتأثيرها على المؤشر")
    target_idx_leader = st.selectbox("اختر المؤشر لقياس تأثير قيادياته:", ["SPX", "SPY", "QQQ"], index=0)
    leaders_data = analyze_market_leaders(target_idx_leader)
    if leaders_data:
        st.info(f"**الخلاصة القيادية لمؤشر {target_idx_leader}:** {leaders_data['verdict']}")
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("صافي التأثير الموزون اليوم", f"{leaders_data['weighted_day_impact']:+.2f}%", "تأثير الأوزان الكبرى")
        l2.metric("زخم القياديات (آخر ساعة)", f"{leaders_data['weighted_1h_impact']:+.2f}%", "نبض السيولة اللحظي")
        l3.metric("القياديات الصاعدة 🟢", f"{leaders_data['bullish_count']} من {leaders_data['total_active']}", "تدعم الصعود")
        l4.metric("القياديات الهابطة 🔴", f"{leaders_data['bearish_count']} من {leaders_data['total_active']}", "تضغط للهبوط")

        df_l = leaders_data["df"]
        fig_l = go.Figure(go.Bar(
            x=df_l["التأثير على المؤشر"],
            y=df_l["السهم القيادي"],
            orientation='h',
            marker_color=['#10B981' if v >= 0 else '#EF4444' for v in df_l["التأثير على المؤشر"]],
            text=[f"{v:+.3f}%" for v in df_l["التأثير على المؤشر"]],
            textposition="auto"
        ))
        fig_l.update_layout(
            title=f"مساهمة كل سهم قيادي في حركة مؤشر {target_idx_leader} (Weighted Impact %)",
            height=380,
            yaxis=dict(autorange="reversed"),
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig_l, use_container_width=True)
        st.dataframe(df_l, use_container_width=True, hide_index=True)
    else:
        st.warning("تعذر جلب بيانات الأسهم القيادية حالياً.")

with tab3:
    st.subheader("🔍 الماسح الآلي للفرص (0DTE اللحظي / اليومي / السوينق)")
    watchlist = st.multiselect(
        "اختر قائمة الأسهم والمؤشرات للفحص:",
        ["SPX", "SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META", "AMD", "GOOGL", "NFLX"],
        default=["SPX", "SPY", "QQQ", "NVDA", "AAPL", "TSLA", "META"]
    )
    if st.button("🚀 بدء الفحص الشامل", type="primary"):
        rows = []
        prog = st.progress(0)
        for idx, sym in enumerate(watchlist):
            d, _ = analyze_ticker(sym, interval, swing_order, pricing_strategy, trading_mode, target_contract_price)
            if d:
                g = d["greeks"]
                rows.append({
                    "الرمز": sym,
                    "التقييم": f"{d['score']}/5 ⭐",
                    "السعر الحالي": f"${d['current_price']:.2f}",
                    "الاتجاه": d['trend'],
                    "نوع العقد": d['option_type'],
                    "Strike": f"${d['strike_rec']}",
                    "قسط تقديري": f"${d['display_premium']:.2f}",
                    "Delta Δ": f"{g['delta']:.2f}",
                    "كفاءة اليونانيات": f"{d['greek_eff']:.2f}",
                    "احتمالية PoP": f"{g['pop_pct']:.0f}%",
                    "Theta/يوم": f"-${abs(g['theta_daily'])*100:.1f}",
                    "البولنجر": d['bb_status'],
                    "الهدف 1": f"${d['target_1']:.2f}",
                    "وقف السهم": f"${d['stop_loss']:.2f}",
                    "DTE": "0DTE (اليوم)" if d['recommended_dte'] == 0 else f"{d['recommended_dte']} يوم",
                    "R:R": f"1 : {d['rr_ratio']:.2f}"
                })
            prog.progress((idx + 1) / len(watchlist))

        if rows:
            res_df = pd.DataFrame(rows).sort_values(by="التقييم", ascending=False)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
        else:
            st.warning("لم يتم العثور على بيانات كافية.")

with tab4:
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
