import streamlit as st
import pandas as pd
import os
from datetime import datetime

# إعداد الصفحة (إذا لم تكن معرفة مسبقاً في ملفك)
# st.set_page_config(page_title="حاسبة ومانح صفقات الخيارات", layout="wide")

# ملف حفظ السجل محلياً بصيغة CSV
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
    """إضافة صفقة أو تنبيه جديد تلقائياً أو يدوياً للسجل"""
    init_trade_log()
    df = pd.read_csv(LOG_FILE)
    
    new_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Ticker": ticker,
        "Option_Type": opt_type,
        "Strike": strike,
        "Alert_Price": alert_price,
        "Max_Price_Reached": alert_price, # يبدأ بالسعر الأساسي عند التنبيه
        "Status": "مفتوح للمراقبة ⏳",
        "Notes": "تنبيه آلي من الماسح"
    }])
    
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

# ==========================================
# واجهة سجل تتبع الأداء
# ==========================================
st.markdown("---")
st.subheader("📊 سجل تتبع الأداء والصفقات الحية (Trade Journal)")

init_trade_log()
history_df = pd.read_csv(LOG_FILE)

if not history_df.empty:
    # حساب نسبة التغير في السعر تلقائياً بناءً على أعلى سعر وصل إليه العقد
    if "Alert_Price" in history_df.columns and "Max_Price_Reached" in history_df.columns:
        # حساب نسبة الربح المئوية: ((أعلى سعر - سعر التنبيه) / سعر التنبيه) * 100
        history_df["Profit_Percentage_%"] = (
            (history_df["Max_Price_Reached"] - history_df["Alert_Price"]) / history_df["Alert_Price"] * 100
        ).round(2)

    st.write("💡 **تلميح:** يمكنك تعديل خانة (أعلى سعر وصل له Max_Price_Reached) مباشرة من الجدول أدناه لتحديث أداء العقود (مثل تحديث عقد META إلى 20.20):")
    
    # جدول تفاعلي يتيح لك تعديل البيانات وتحديثها مباشرة
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
    st.info("لا توجد صفقات مسجلة حتى الآن. يمكنك إضافة عقود للمراقبة أدناه.")

# ==========================================
# نموذج إضافة صفقة جديدة يدوياً أو ربطها بالماسح
# ==========================================
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
            log_new_trade(t_ticker, t_type, t_strike,
