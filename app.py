import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Downloading NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

# Function to preprocess the text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load the vectorizer and model
tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Title and description with custom HTML and CSS
st.markdown("""
    <link rel="stylesheet" href="styles.css">
    <div class="container">
        <h1 class="title">SMS Spam Detection Model 🌟</h1>
        <p class="description">This is a Machine Learning project that uses Natural Language Processing to classify SMS messages as spam or not. 🚀📱</p>
    </div>
""", unsafe_allow_html=True)

# Create a text input for the SMS
input_sms = st.text_input("Enter the SMS", key="sms", help="Type your message here...", max_chars=300)

# Add a button to predict
if st.button('Predict', key='predict', help="Click to predict if the SMS is spam or not"):
    # Preprocess the SMS input
    transformed_sms = transform_text(input_sms)

    # Vectorize the input
    vector_input = tk.transform([transformed_sms])

    # Predict the result
    result = model.predict(vector_input)[0]
    
    # Display result
    if result == 1:
        st.markdown('<div class="result" style="color: red;">Spam</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result" style="color: green;">Not Spam</div>', unsafe_allow_html=True)

# Display the watermark
st.markdown('<div class="watermark" style="color: #bbb; font-style: italic; opacity: 0.5; font-size: 14px;">Created by Gurudeep Soni 💻</div>', unsafe_allow_html=True)
