import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Sidebar Description
st.sidebar.title("📊 NLP Sentiment Analysis")
st.sidebar.markdown("""
### Project Overview
This app uses NLP techniques to classify text sentiment as Positive, Negative, or Neutral. 

#### Key Features:
- ✅ Data Cleaning and Preprocessing
- 📦 Logistic Regression Model
- 📈 Real-time Sentiment Prediction
- 🏷️ Sample Dataset Display

💡 **How to Use:**
1. Enter text in the input box.
2. Click **Analyze Sentiment**.
3. View the predicted sentiment.
""")

# Load dataset
file_path = "twitter_training.csv"  # Ensure this is correctly placed in your working directory
df = pd.read_csv(file_path, header=None, names=["Category", "Sentiment", "Text"])

# Data Cleaning
df = df[["Sentiment", "Text"]]
df = df.dropna()
df["Text"] = df["Text"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x.lower()))  # Clean text

# Encode sentiment labels
label_encoder = LabelEncoder()
df["Sentiment"] = label_encoder.fit_transform(df["Sentiment"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["Text"], df["Sentiment"], test_size=0.2, random_state=42)

# Build NLP Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("classifier", LogisticRegression())
])

# Train Model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Load trained model
model = joblib.load("sentiment_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Streamlit UI
st.title("📝 NLP Sentiment Analysis App")
st.write("Enter text below to predict sentiment.")

# User Input
user_text = st.text_area("Enter your text here:", "")

if st.button("Analyze Sentiment", help="Click to predict sentiment"):
    if user_text.strip():
        cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", user_text.lower())
        prediction = model.predict([cleaned_text])[0]
        sentiment = label_encoder.inverse_transform([prediction])[0]

        # Color-coded Sentiment Output
        if sentiment == "Positive":
            st.success(f"🟢 **Positive Sentiment**")
        elif sentiment == "Negative":
            st.error(f"🔴 **Negative Sentiment**")
        else:
            st.info(f"🔵 **Neutral Sentiment**")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Display dataset
if st.checkbox("📂 Show Sample Dataset"):
    st.write(df.sample(10))
