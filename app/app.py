import streamlit as st
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

st.set_page_config(page_title="Email Spam Detection", layout="centered")

st.markdown("""
<style>
.main { background-color: #f2f7ff; }
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    font-size: 18px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.title("📧 Email Spam Detection")
st.write("Enter an email to detect spam")

email = st.text_area("Email Content", height=150)

if st.button("Detect Spam"):
    if not email.strip():
        st.warning("Enter email text")
    else:
        model = pickle.load(open(MODEL_PATH, "rb"))
        vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

        vec = vectorizer.transform([email])
        result = model.predict(vec)[0]

        if result == "spam":
            st.error("🚨 SPAM EMAIL")
        else:
            st.success("✅ NOT SPAM")
            
            

