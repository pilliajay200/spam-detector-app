import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- LOAD CSV (AUTO-DETECT) ----------
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("No CSV file found in data folder")

DATA_PATH = os.path.join(DATA_DIR, csv_files[0])
print("Using dataset:", DATA_PATH)

data = pd.read_csv(DATA_PATH)

# ---------- AUTO COLUMN DETECTION ----------
if {"label", "text"}.issubset(data.columns):
    X = data["text"]
    y = data["label"]
elif {"v1", "v2"}.issubset(data.columns):
    X = data["v2"]
    y = data["v1"]
elif {"category", "message"}.issubset(data.columns):
    X = data["message"]
    y = data["category"]
else:
    # LAST RESORT: assume first = label, second = text
    X = data.iloc[:, 1]
    y = data.iloc[:, 0]

# ---------- CRITICAL FIX (THIS SOLVES YOUR ERROR) ---
