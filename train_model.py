# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ Download required NLTK data (only needed once)
nltk.download("stopwords")
nltk.download("wordnet")

# ✅ Define clean_text() BEFORE using it
def clean_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    if not isinstance(text, str):
        return ""

    # Remove URLs and special characters
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text.lower())

    # Tokenize and clean
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

    return " ".join(tokens)

# ✅ Load the labeled dataset
df = pd.read_csv("data/processed_comments.csv")

# ✅ Clean the text using the function
df["text_clean"] = df["text"].apply(clean_text)

# ✅ Vectorize using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["text_clean"])
y = df["label"]

# ✅ Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ✅ Train a logistic regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# ✅ Evaluate the model
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ✅ Save model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/classifier.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n✅ Model and vectorizer saved to 'models/' folder")
