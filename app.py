import os
import pickle
import streamlit as st

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
.stTextArea textarea {
    border-radius: 10px;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
.footer {
    text-align: center;
    font-size: 14px;
    color: gray;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1 style='text-align: center;'>📧 AI Email Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Powered by Machine Learning (Naive Bayes + TF-IDF)</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------- BASE DIRECTORY ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# ---------- CHECK MODEL ----------
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error("❌ Model not found! Please run train_model.py first.")
    st.stop()

# ---------- LOAD MODEL ----------
model = pickle.load(open(MODEL_PATH, "rb"))
vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("ℹ Project Info")
    st.write("""
    **Model Used:** Multinomial Naive Bayes  
    **Vectorizer:** TF-IDF  
    **Language:** Python  
    **Framework:** Streamlit  
    """)
    st.markdown("---")
    st.write("Developed for ML Portfolio 🚀")

# ---------- INPUT AREA ----------
st.subheader("Enter Email Message")
message = st.text_area("", height=180, placeholder="Type or paste your email content here...")

st.markdown("")

# ---------- PREDICT BUTTON ----------
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_btn = st.button("🔍 Analyze Message")

# ---------- PREDICTION ----------
if predict_btn:
    if message.strip() == "":
        st.warning("⚠ Please enter a message.")
    else:
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)[0]
        probability = model.predict_proba(message_vec).max()

        st.markdown("---")
        st.subheader("Result")

        if prediction.lower() == "spam":
            st.markdown(
                f"<div class='result-box' style='background-color:#ffe6e6; color:#b30000;'>🚨 SPAM DETECTED</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box' style='background-color:#e6fff2; color:#006600;'>✅ SAFE MESSAGE</div>",
                unsafe_allow_html=True
            )

        st.markdown("### Confidence Level")
        st.progress(float(probability))
        st.write(f"Model Confidence: **{probability*100:.2f}%**")

# ---------- FOOTER ----------
st.markdown("<div class='footer'>© 2026 AI Spam Detector | Built with Streamlit</div>", unsafe_allow_html=True)
