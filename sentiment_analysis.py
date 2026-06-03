import streamlit as st
import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt

from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Download NLTK resources
# -----------------------------c
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# -----------------------------
# Page Title
# -----------------------------
st.set_page_config(page_title="Sentiment Analysis App")

st.title("Sentiment Analysis on Social Media Comments")
st.write("Analyze comments as Positive, Negative, or Neutral")

# -----------------------------
# Initialize NLP tools
# -----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):

    text = str(text).lower()

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = word_tokenize(text)

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return ' '.join(words)

# -----------------------------
# Sentiment Function
# -----------------------------
def get_sentiment(text):

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive 😊"

    elif polarity < 0:
        return "Negative 😡"

    else:
        return "Neutral 😐"

# -----------------------------
# Single Comment Analysis
# -----------------------------
st.header("Analyze Single Comment")

user_input = st.text_area("Enter your comment")

if st.button("Analyze Sentiment"):

    cleaned = clean_text(user_input)

    sentiment = get_sentiment(cleaned)

    st.subheader("Result")
    st.success(sentiment)

# -----------------------------
# CSV Upload Analysis
# -----------------------------
st.header("Upload CSV File")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Original Data")
    st.dataframe(df)

    # Clean comments
    df['cleaned_comment'] = df['comment'].apply(clean_text)

    # Predict sentiment
    df['sentiment'] = df['cleaned_comment'].apply(get_sentiment)

    st.subheader("Sentiment Results")
    st.dataframe(df)

    # Sentiment counts
    sentiment_counts = df['sentiment'].value_counts()

    st.subheader("Sentiment Counts")
    st.write(sentiment_counts)

    # -----------------------------
    # Bar Chart
    # -----------------------------
    st.subheader("Bar Chart")

    fig, ax = plt.subplots(figsize=(6, 4))

    sentiment_counts.plot(
        kind='bar',
        ax=ax
    )

    plt.title("Sentiment Distribution")

    st.pyplot(fig)

    # -----------------------------
    # Pie Chart
    # -----------------------------
    st.subheader("Pie Chart")

    fig2, ax2 = plt.subplots(figsize=(6, 6))

    sentiment_counts.plot(
        kind='pie',
        autopct='%1.1f%%',
        ax=ax2
    )

    plt.ylabel("")
    plt.title("Sentiment Percentage")

    st.pyplot(fig2)

    # -----------------------------
    # Download Results
    # -----------------------------
    csv = df.to_csv(index=False)

    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name='sentiment_results.csv',
        mime='text/csv'
    )

st.write("Project Created Using Python, NLP, TextBlob, and Streamlit")