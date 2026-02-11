import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------- BASE DIRECTORY ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- LOAD DATA ----------
DATA_PATH = os.path.join(BASE_DIR, "emails.csv")

df = pd.read_csv(DATA_PATH)

# Ensure correct column names
df.columns = ["label", "message"]

X = df["message"]
y = df["label"]

# ---------- TRAIN TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- VECTORIZATION ----------
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------- MODEL TRAINING ----------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ---------- EVALUATION ----------
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ---------- SAVE MODEL ----------
with open(os.path.join(BASE_DIR, "spam_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully.")
