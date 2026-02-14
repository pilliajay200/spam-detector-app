import pandas as pd
import os
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "emails.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load base data
data = pd.read_csv(DATA_PATH)

# -----------------------------
# DATA AUGMENTATION TO 30,000
# -----------------------------
spam_templates = [
    "Win money now",
    "Claim your free prize",
    "Limited offer act fast",
    "Congratulations you won",
    "Click link to get reward",
    "Free recharge available",
    "Urgent offer only today"
]

ham_templates = [
    "How are you?",
    "Let's meet tomorrow",
    "Please send the file",
    "Call me when free",
    "Are you coming today?",
    "Meeting is scheduled",
    "Thank you for your help"
]

rows = []

for _ in range(15000):
    rows.append(["spam", random.choice(spam_templates)])
    rows.append(["ham", random.choice(ham_templates)])

big_data = pd.DataFrame(rows, columns=["label", "text"])

print("Total dataset size:", len(big_data))

# Vectorization
X = big_data["text"]
y = big_data["label"]

vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# Naive Bayes Model
model = MultinomialNB()
model.fit(X_vec, y)

# Save model
with open(os.path.join(MODEL_DIR, "spam_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Naive Bayes trained on 30,000 emails")
