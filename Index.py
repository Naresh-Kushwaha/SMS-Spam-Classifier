import pickle
import string
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Download required NLTK resources
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)


download_nltk_data()


# Load model and vectorizer
@st.cache_resource
def load_models():
    with open('./vectorizer.pkl', 'rb') as file:
        tfidf = pickle.load(file)

    with open('./model.pkl', 'rb') as file:
        model = pickle.load(file)

    return tfidf, model


tfidf, model = load_models()


# Stemmer
ps = PorterStemmer()


def transform_text(text):

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Tokenization
    text = nltk.word_tokenize(text)

    # 3. Keep only alphanumeric words
    y = []

    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))

    for word in text:
        if word not in stop_words and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    # 5. Stemming
    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)


# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message:")


if st.button("Predict"):

    if input_sms.strip():

        # Transform input
        transformed_sms = transform_text(input_sms)

        # Convert text into vector
        vector_input = tfidf.transform([transformed_sms])

        # Prediction
        result = model.predict(vector_input)

        if result[0] == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")

    else:
        st.warning("Please enter a message first.")