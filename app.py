import streamlit as st
import joblib
import re
import nltk
import google.generativeai as genai
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 🧠 Set Gemini API key
genai.configure(api_key="api key")  # 🔁 Replace with your actual key

model = genai.GenerativeModel("gemini-1.5-flash")

nltk.download("stopwords")
nltk.download("wordnet")

def clean_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def predict_comment(text):
    model = joblib.load("models/classifier.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    vector = vectorizer.transform([clean_text(text)])
    return model.predict(vector)[0]

def generate_reply_with_gemini(comment, category):
    prompt = f"""
You are a helpful assistant for a social media team.

A user left this comment: "{comment}"

The system categorized this comment as: "{category}".

Generate a short, polite, empathetic reply suitable for this category.
Reply in 1-2 sentences.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# Streamlit UI
st.set_page_config(page_title="Comment Categorizer + Gemini AI", layout="centered")
st.title("🧠 Comment Categorization & Smart Reply (Gemini AI)")

comment = st.text_area("💬 Enter a user comment:")

if st.button("Categorize & Reply"):
    if not comment.strip():
        st.warning("Please enter a comment.")
    else:
        category = predict_comment(comment)
        st.success(f"🧩 Predicted Category: {category}")

        reply = generate_reply_with_gemini(comment, category)
        st.info(f"🤖 Gemini Smart Reply: {reply}")
