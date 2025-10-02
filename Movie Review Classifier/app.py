import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_sentiment(review):
    vec = vectorizer.transform([review])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    sentiment = "Positive" if pred == 1 else "Negative"
    return sentiment, prob

st.title("Movie Review Sentiment Classifier")
st.write("Enter movie review to see its predicted sentiment.")

review = st.text_area("Enter your review:")

if st.button("Predict"):
    if review.strip() != "":
        sentiment, prob = predict_sentiment(review)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: Positive: {prob[1]:.2f}, Negative: {prob[0]:.2f}")
    else:
        st.write("Can't be blank, enter a review.")
