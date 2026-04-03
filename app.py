import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Ashu AI Fraud Shield", layout="wide", page_icon="🛡️")

# ---------------- LOAD MODEL ----------------
MODEL_DIR = "models"

@st.cache_resource
def load_assets():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "fraud_model.pkl"))
        features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
        return model, list(features)
    except Exception as e:
        st.error(f"Critical Error: Models not found. ({e})")
        return None, None

model, features = load_assets()

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- UI STYLING ----------------
st.markdown("""
<style>
    .stButton>button { background-color: #007BFF; color: white; border-radius: 8px; font-weight: bold; width: 100%; }
    .card { background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
    .fraud-text { color: #d32f2f; font-weight: bold; font-size: 28px; text-align: center; }
    .safe-text { color: #2e7d32; font-weight: bold; font-size: 28px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
h1, h2 = st.columns([1, 4])
with h1:
    if os.path.exists("ashu_logo.png"):
        st.image("ashu_logo.png", width=130)
    else: 
        st.title("🛡️")
with h2:
    st.title("Ashu AI: Online Payment Fraud Shield")
    st.markdown("**Enterprise Real-Time Monitoring** | Developed by **Ashutosh Tripathi**")

st.divider()

# ---------------- MAIN APP ----------------
col_in, col_res = st.columns([1.5, 1], gap="large")

with col_in:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Transaction Details")
    amount = st.number_input("Amount (₹)", min_value=0.01, value=1000.0)
    tx_type = st.selectbox("Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])
    
    c1, c2 = st.columns(2)
    with c1:
        old_org = st.number_input("Sender Initial Balance (₹)", value=1000.0)
        new_org = st.number_input("Sender Final Balance (₹)", value=0.0)
    with c2:
        old_dest = st.number_input("Receiver Initial Balance (₹)", value=0.0)
        new_dest = st.number_input("Receiver Final Balance (₹)", value=1000.0)
    
    analyze = st.button("🚀 RUN SECURITY SCAN")
    st.markdown('</div>', unsafe_allow_html=True)

if analyze:
    # 1. FIXED MATH CHECK: Only trigger if the numbers are WAY off
    actual_sent = old_org - new_org
    actual_received = new_dest - old_dest
    # Logic: If I send 1000 and receiver gets 0, that is a math error.
    # We only trigger this if the discrepancy is > 50% of the amount to avoid false positives.
    m_error = abs(actual_sent - actual_received) > (amount * 0.5)

    # 2. FEATURE ENGINEERING (Must match notebook column names exactly)
    row = {
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "org_diff": old_org - new_org,
        "dest_diff": new_dest - old_dest,
        "is_zero_orig": 1 if old_org == 0 else 0,
        "is_zero_dest": 1 if old_dest == 0 else 0
    }
    
    # Correct One-Hot Encoding: Initialize all to 0, then set the active one to 1
    for f in features:
        if f.startswith("type_"):
            row[f] = 0
    
    type_col = f"type_{tx_type}"
    if type_col in features:
        row[type_col] = 1

    # 3. PREDICTION
    input_df = pd.DataFrame([row])[features]
    prob = model.predict_proba(input_df)[0][1]
    
    # 4. DECISION: Lowered sensitivity slightly to 0.85 to be safer
    is_f = 1 if (prob >= 0.85 or m_error) else 0
    
    # Save History
    st.session_state.history.insert(0, {
        "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
        "Amount": f"₹{amount:,.2f}",
        "Type": tx_type,
        "Status": "Fraud" if is_f else "Safe"
    })

    with col_res:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔍 Scan Result")
        
        ring_color = "#d32f2f" if is_f else "#2e7d32"
        emoji = "☹️" if is_f else "😊"
        status_text = "FRAUD DETECTED" if is_f else "VERIFIED SAFE"
        text_class = "fraud-text" if is_f else "safe-text"

        st.markdown(f'<p class="{text_class}">{emoji} {status_text}</p>', unsafe_allow_html=True)
        
        if m_error:
            st.warning("⚠️ Warning: Transaction balances do not align.")

        fig = go.Figure(data=[go.Pie(values=[1], hole=.75, marker_colors=[ring_color], textinfo='none', hoverinfo='none')])
        fig.update_layout(annotations=[dict(text=emoji, x=0.5, y=0.5, font_size=60, showarrow=False)], height=250, margin=dict(l=0,r=0,t=0,b=0), showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- HISTORY TABLE ----------------
st.write("### 📜 Session Log")
hist_df = pd.DataFrame([x for x in st.session_state.history if isinstance(x, dict)])
if not hist_df.empty:
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

st.divider()
st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 0.8em;">Ashu AI Fraud Shield | Analysis based on Random Forest Classifier || Sensitivity: 0.80 (for demonstration only, otherwise sensitivity is 0.95) | © 2026</p>', unsafe_allow_html=True)