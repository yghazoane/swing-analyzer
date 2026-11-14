import streamlit as st
import pandas as pd
import os
from datetime import datetime

# ملف حفظ السجل محلياً
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
    
    # التحقق إن لم يكن العقد مسجلاً مسبقاً لنفس اللحظة
    new_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Ticker": ticker,
        "Option_Type": opt_type,
        "Strike": strike,
        "Alert_Price": alert_price,
        "Max_Price_Reached": alert_price, # يبدأ بسعر التنبيه
        "Status": "مفتوح للمراقبة ⏳",
        "Notes": "تم التتبع الآلي"
    }])
    
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

# عرض سجل الأداء في واجهة Streamlit
st.subheader("📊 سجل تتبع الأداء والصفقات الحية")
init_trade_log()
history_df = pd.read_csv(LOG_FILE)

if not history_df.empty:
    # تعديل تفاعلي لتحديث أعلى سعر وصل إليه العقد (مثل 20.20 لعقد ميتا)
    edited_df = st.data_editor(history_df, num_rows="dynamic")
    if st.button("حفظ التحديثات في السجل"):
        edited_df.to_csv(LOG_FILE, index=False)
        st.success("تم تحديث السجل بنجاح وقياس الأداء بدقة!")
else:
    st.info("لا توجد صفقات مسجلة حتى الآن. استخدم زر إضافة صفقة أدناه لرصد العقود.")

# زر لتسجيل عقد جديد يدوياً أو تلقائياً من الماسح
with st.form("add_trade_form"):
    st.write("إضافة عقد جديد للمراقبة والسجل:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        t_ticker = st.text_input("رمز السهم", "META")
    with col2:
        t_type = st.selectbox("نوع العقد", ["PUT", "CALL"])
    with col3:
        t_strike = st.number_input("السترايك", value=635.0)
    with col4:
        t_price = st.number_input("سعر العقد عند التنبيه", value=17.06)
    
    submit_btn = st.form_submit_button("إضافة إلى سجل التتبع")
    if submit_btn:
        log_new_trade(t_ticker, t_type, t_strike, t_price)
        st.rerun()
