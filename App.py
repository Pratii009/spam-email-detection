from flask import Flask, render_template, request
import pickle
import string
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("spam.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocess function (same as training)
def pre_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    stemmer = SnowballStemmer("english")
    words = " ".join([stemmer.stem(i) for i in text])
    return words

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    cleaned = pre_process(message)

    # IMPORTANT: Use transform(), NOT fit_transform()!!
    x = vectorizer.transform([cleaned]).toarray()

    prediction = model.predict(x)[0]

    return render_template("result.html", message=message, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
