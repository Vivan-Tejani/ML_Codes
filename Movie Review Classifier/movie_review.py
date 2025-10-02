import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = pd.read_csv("/Users/vivan/Codes/Datasets/IMDB Dataset.csv")

data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, Y_train, Y_test = train_test_split(
    data['review'], data['sentiment'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, Y_train)

Y_pred = model.predict(X_test_tfidf)
print("Accuracy: ", accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

def input_predict(review):
    vec = vectorizer.transform([review])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    sentiment = "Positive" if pred == 1 else "Negative"
    return sentiment, prob

# review = input()
# print(input_predict(review))